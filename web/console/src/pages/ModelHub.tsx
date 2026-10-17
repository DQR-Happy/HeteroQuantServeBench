import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Button,
  Checkbox,
  Descriptions,
  Drawer,
  Input,
  Progress,
  Select,
  Space,
  Table,
  Tag,
} from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, fmt, gib } from '../api/client';
import { PageHead, Panel, QueryState } from '../components';
import type {
  DownloadJob,
  HubCatalogue,
  HubConfig,
  Inspection,
  ModelSummary,
} from '../features/model-hub/types';
import '../features/model-hub/styles.css';

const states: Record<string, string> = {
  queued: '等待下载',
  downloading: '正在下载',
  verifying: '正在校验封存',
  completed: '已校验完成',
  cancelling: '正在取消',
  cancelled: '已取消 · 可继续',
  interrupted: '已中断 · 可继续',
  failed: '失败',
  not_loaded: '目录尚未加载',
  refreshing: '正在索引发布方目录',
  partial: '目录不完整',
};
const active = (state: string) =>
  ['queued', 'downloading', 'verifying', 'cancelling'].includes(state);
const size = (value: number | null | undefined) =>
  value == null ? '未知' : `${fmt(gib(value), 2)} GiB`;

function VersionDrawer({ modelId, onClose }: { modelId: string; onClose: () => void }) {
  const client = useQueryClient();
  const [revision, setRevision] = useState('master');
  const [inspection, setInspection] = useState<Inspection>();
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const revisions = useQuery({
    queryKey: ['model-hub-revisions', modelId],
    queryFn: () =>
      api<{ branches: string[]; tags: string[] }>('/model-hub/revisions', { model_id: modelId }),
    retry: false,
  });
  const check = async () => {
    setBusy('inspect');
    setError('');
    setInspection(undefined);
    try {
      setInspection(await api<Inspection>('/model-hub/inspect', { model_id: modelId, revision }));
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  const download = async () => {
    if (!inspection) return;
    setBusy('download');
    setError('');
    try {
      await api<DownloadJob>('/model-hub/downloads', { plan_sha256: inspection.plan_sha256 });
      await client.invalidateQueries({ queryKey: ['model-hub-downloads'] });
      onClose();
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  const options = [
    ...new Set([...(revisions.data?.branches ?? []), ...(revisions.data?.tags ?? [])]),
  ];
  return (
    <Drawer title="权重版本与下载检查" open onClose={onClose} width={680}>
      <h2 className="model-hub-id">{modelId}</h2>
      <QueryState loading={revisions.isPending} error={revisions.error} />
      {revisions.isError && <Button onClick={() => void revisions.refetch()}>重试读取版本</Button>}
      <label className="field-label" htmlFor="model-revision">
        发布方分支 / 标签
      </label>
      <Select
        id="model-revision"
        className="wide-select"
        value={revision}
        disabled={!!busy || !options.length}
        options={options.map((value) => ({ value, label: value }))}
        onChange={(value) => {
          setRevision(value);
          setInspection(undefined);
          setError('');
        }}
      />
      <p className="help-text">
        这里列出该仓库公开的全部分支与标签。master 可能变化；下载按检查时的逐文件 commit 与 SHA-256
        锁定身份。
      </p>
      <Space wrap className="section-gap">
        <Button
          loading={busy === 'inspect'}
          disabled={!!busy || !options.includes(revision)}
          onClick={() => void check()}
        >
          检查所选版本与机器资源
        </Button>
        <a href={`https://modelscope.cn/models/${modelId}`} target="_blank" rel="noreferrer">
          查看模型卡与许可证 ↗
        </a>
      </Space>
      {error && <Alert type="error" showIcon message="操作未完成" description={error} />}
      {inspection && (
        <>
          <Descriptions
            className="section-gap"
            column={1}
            size="small"
            items={[
              { key: 'type', label: '模型架构', children: inspection.plan.config.model_type },
              {
                key: 'precision',
                label: '加载精度',
                children: inspection.plan.runtime_precision ?? 'float16',
              },
              {
                key: 'input',
                label: '默认输入方式',
                children:
                  inspection.plan.input_mode === 'completion'
                    ? '原始文本续写 · 不需要聊天模板'
                    : '聊天 · 使用仓库自带模板',
              },
              {
                key: 'template',
                label: '聊天模板',
                children:
                  inspection.plan.chat_template_available === false
                    ? '仓库未提供；可通过原始文本续写推理'
                    : '随 tokenizer 配置或 .jinja 文件一起下载并校验',
              },
              {
                key: 'download',
                label: '需下载文件 / 大小',
                children: `${inspection.plan.files.length} 个 / ${size(inspection.plan.download_bytes)}`,
              },
              {
                key: 'memory',
                label: '预计 GPU 驻留与运行余量',
                children: size(inspection.admission.estimated_runtime_bytes),
              },
              {
                key: 'kv',
                label: '预计 KV（batch 1 / 2048 context）',
                children: size(inspection.admission.kv_bytes),
              },
              {
                key: 'source',
                label: '来源',
                children:
                  inspection.plan.provenance === 'mirror' ? '平台镜像，非原厂组织' : '登记原发布方',
              },
            ]}
          />
          <Alert
            showIcon
            type={inspection.admission.eligible ? 'info' : 'warning'}
            message={
              inspection.admission.eligible
                ? '配置检查与资源预估通过 · 尚未实测运行'
                : '当前版本暂不可下载部署'
            }
            description={
              inspection.admission.eligible
                ? '下载不会加载模型或替换当前驻留部署；完成后可登记到模型列表，再显式加载验证。'
                : inspection.admission.reasons.join('；')
            }
          />
          <p className="help-text">{inspection.admission.limitations.join(' ')}</p>
          <details>
            <summary>冻结文件清单与发布方哈希</summary>
            <ul className="model-file-list">
              {inspection.plan.files.map((file) => (
                <li key={file.Path}>
                  <strong>{file.Path}</strong> · {size(file.Size)}
                  <code>{file.Sha256}</code>
                </li>
              ))}
            </ul>
          </details>
          <p className="help-text mono">计划 SHA-256：{inspection.plan_sha256}</p>
          <Button
            type="primary"
            block
            disabled={!inspection.admission.eligible || !!busy}
            loading={busy === 'download'}
            onClick={() => void download()}
          >
            下载到当前机器
          </Button>
          <p className="help-text">
            下载遵循发布方访问条件与许可证；受限仓库仍需节点已有授权，本页不会代为接受协议。
          </p>
        </>
      )}
    </Drawer>
  );
}

export default function ModelHubPage() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const publisher = params.get('publisher') || 'Qwen';
  const query = params.get('q') ?? '';
  const sortBy = ['parameters', 'runtime_bytes'].includes(params.get('sort_by') ?? '')
    ? params.get('sort_by')!
    : 'default';
  const sortDirection = params.get('sort_order') === 'desc' ? 'desc' : 'asc';
  const sorting: Record<string, string> =
    sortBy === 'default' ? {} : { sort_by: sortBy, sort_order: sortDirection };
  const sortOrder = sortDirection === 'desc' ? 'descend' : 'ascend';
  const setPublisher = (value: string) =>
    setParams({ publisher: value, ...sorting }, { replace: true });
  const setQuery = (value: string) =>
    setParams({ publisher, ...sorting, ...(value ? { q: value } : {}) }, { replace: true });
  const [onlyFit, setOnlyFit] = useState(true);
  const [page, setPage] = useState(1);
  const [model, setModel] = useState('');
  const [direct, setDirect] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const config = useQuery({
    queryKey: ['model-hub-config'],
    queryFn: () => api<HubConfig>('/model-hub/config'),
    refetchInterval: 15000,
    retry: false,
  });
  const catalog = useQuery({
    queryKey: ['model-hub-catalog', publisher, query, onlyFit, page, sortBy, sortDirection],
    queryFn: () =>
      api<HubCatalogue>(
        `/model-hub/catalog?publisher=${encodeURIComponent(publisher)}&q=${encodeURIComponent(query)}&only_fit=${onlyFit}&offset=${(page - 1) * 20}&limit=20&sort_by=${sortBy}&sort_order=${sortDirection}`,
      ),
    refetchInterval: (current) => (current.state.data?.state === 'refreshing' ? 2000 : false),
    retry: false,
  });
  const downloads = useQuery({
    queryKey: ['model-hub-downloads'],
    queryFn: () => api<{ items: DownloadJob[] }>('/model-hub/downloads'),
    refetchInterval: 1500,
    retry: false,
  });
  const refresh = async () => {
    setBusy('catalog');
    setError('');
    try {
      await api(`/model-hub/catalog/refresh?publisher=${encodeURIComponent(publisher)}`, {});
      await client.invalidateQueries({ queryKey: ['model-hub-catalog'] });
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  const act = async (id: string, action: string) => {
    setBusy(id);
    setError('');
    try {
      await api(`/model-hub/downloads/${id}/${action}`, {});
      await Promise.all([
        client.invalidateQueries({ queryKey: ['model-hub-downloads'] }),
        client.invalidateQueries({ queryKey: ['deployments'] }),
      ]);
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  return (
    <>
      <PageHead
        kicker="MODELSCOPE / NODE ARTIFACTS"
        title="模型下载与接入"
        description="浏览 ModelScope 发布方权重，按当前机器筛选候选，检查具体版本后下载、校验并接入推理。"
        extra={
          <Space>
            <Link to="/guide#models">
              <Button>模型下载指南</Button>
            </Link>
            <Link to="/devices">
              <Button>查看已登记部署</Button>
            </Link>
          </Space>
        }
      />
      <QueryState
        loading={config.isPending}
        error={config.error ?? catalog.error ?? downloads.error}
      />
      {error && (
        <Alert
          className="section-gap"
          showIcon
          type="error"
          message="操作未完成"
          description={error}
        />
      )}
      {config.data && (
        <div className="model-hub-capacity">
          <strong>{config.data.capacity.gpu_name}</strong>
          <span>GPU 总容量 {size(config.data.capacity.gpu_total_bytes)}</span>
          <span>主机 / 容器可用 {size(config.data.capacity.host_available_bytes)}</span>
          <span>下载盘剩余 {size(config.data.capacity.disk_free_bytes)}</span>
          <span>保留余量 {size(config.data.disk_reserve_bytes)}</span>
          <code>{config.data.capacity.download_directory}</code>
        </div>
      )}
      <Panel
        title="发布方模型目录"
        subtitle="优先整批读取发布方公开仓库，必要时分页；按唯一仓库数校验完整性，搜索与筛选覆盖已索引模型。"
      >
        <div className="model-hub-filters">
          <div>
            <label className="field-label" htmlFor="model-publisher">
              发布方 / 平台镜像
            </label>
            <Select
              id="model-publisher"
              className="wide-select"
              value={publisher}
              options={config.data?.publishers.map((item) => ({
                value: item.id,
                label: item.name,
              }))}
              onChange={(value) => {
                setPublisher(value);
                setPage(1);
              }}
            />
          </div>
          <div>
            <label className="field-label" htmlFor="model-search">
              搜索已索引模型
            </label>
            <Input
              id="model-search"
              placeholder="模型名称、参数规模，如 Qwen3-0.6B"
              allowClear
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setPage(1);
              }}
            />
          </div>
          <Button
            loading={busy === 'catalog'}
            disabled={catalog.data?.state === 'refreshing'}
            onClick={() => void refresh()}
          >
            {catalog.data?.state === 'not_loaded' ? '加载发布方目录' : '刷新发布方目录'}
          </Button>
        </div>
        <Space wrap className="section-gap">
          <Checkbox
            checked={onlyFit}
            onChange={(event) => {
              setOnlyFit(event.target.checked);
              setPage(1);
            }}
          >
            仅显示资源候选与待检查的纯文本模型
          </Checkbox>
          <Tag>
            {catalog.data?.state === 'completed'
              ? '目录已索引'
              : (states[catalog.data?.state ?? 'not_loaded'] ?? '目录状态未知')}
          </Tag>
          <span className="help-text">
            已索引 {catalog.data?.indexed ?? 0} / {catalog.data?.total_upstream ?? '—'} 个仓库
          </span>
        </Space>
        <p className="help-text">
          列表仅根据发布方元数据预估占用，尚未验证模型运行。点击“选择版本与检查”可查看具体版本的格式、资源和部署支持情况。
        </p>
        {catalog.data?.state === 'refreshing' && (
          <Alert
            type="info"
            showIcon
            message={`正在读取 ${publisher} 目录：已扫描 ${catalog.data.scanned ?? 0} 条上游记录`}
            description={
              catalog.data.retained_cached
                ? '刷新期间保留上轮目录；模型仍可选择和检查，新目录完成后再替换。'
                : '正在读取公开仓库；若整批结果被截断，会继续分页补齐。当前没有候选不表示该发布方没有适配模型。'
            }
          />
        )}
        {(catalog.data?.error || catalog.data?.state === 'partial') && (
          <Alert
            type="warning"
            showIcon
            message="目录刷新未完整完成"
            description={
              catalog.data.error ||
              `上游报告 ${catalog.data.total_upstream ?? '未知'} 项，已取得 ${catalog.data.indexed} 个去重仓库；上游分页或可见性计数不一致。已取得模型仍可检查。`
            }
          />
        )}
        <Table<ModelSummary>
          size="small"
          rowKey="model_id"
          loading={
            catalog.isPending ||
            (catalog.data?.state === 'refreshing' && !catalog.data.items.length)
          }
          scroll={{ x: 900 }}
          dataSource={catalog.data?.items ?? []}
          onChange={(_, __, sorter, extra) => {
            if (extra.action !== 'sort') return;
            const selected = Array.isArray(sorter) ? sorter[0] : sorter;
            setParams(
              {
                publisher,
                ...(query ? { q: query } : {}),
                ...(selected.order
                  ? {
                      sort_by: String(selected.columnKey),
                      sort_order: selected.order === 'descend' ? 'desc' : 'asc',
                    }
                  : {}),
              },
              { replace: true },
            );
            setPage(1);
          }}
          locale={{
            emptyText:
              catalog.data?.state === 'not_loaded'
                ? '先加载发布方目录，也可以在下方输入完整仓库 ID 检查。'
                : catalog.data?.state === 'refreshing'
                  ? '正在读取目录，候选仍在更新中。'
                  : '暂无符合当前筛选的已索引模型；可调整搜索、取消筛选或输入完整仓库 ID 检查。',
          }}
          pagination={{
            current: page,
            pageSize: 20,
            total: catalog.data?.total ?? 0,
            showSizeChanger: false,
            onChange: setPage,
          }}
          columns={[
            {
              title: '模型权重',
              render: (_, row) => (
                <div>
                  <a href={row.url} target="_blank" rel="noreferrer">
                    {row.model_id} ↗
                  </a>
                  <div className="help-text">
                    {row.provenance === 'mirror' ? '平台镜像' : '登记原发布方'} · {row.license}
                  </div>
                </div>
              ),
            },
            {
              title: '参数规模',
              key: 'parameters',
              sorter: true,
              sortOrder: sortBy === 'parameters' ? sortOrder : null,
              render: (_, row) =>
                row.parameters == null ? '平台未提供' : `${fmt(row.parameters / 1e9, 2)} B`,
            },
            {
              title: '预计 FP16 运行占用',
              key: 'runtime_bytes',
              sorter: true,
              sortOrder: sortBy === 'runtime_bytes' ? sortOrder : null,
              render: (_, row) =>
                row.estimate.estimated_runtime_bytes == null
                  ? '待版本检查'
                  : size(row.estimate.estimated_runtime_bytes),
            },
            {
              title: '操作',
              render: (_, row) => (
                <Button size="small" onClick={() => setModel(row.model_id)}>
                  选择版本与检查
                </Button>
              ),
            },
          ]}
        />
        <div className="model-hub-direct">
          <label className="field-label" htmlFor="model-id">
            直接检查登记发布方的完整仓库 ID
          </label>
          <Space.Compact block>
            <Input
              id="model-id"
              placeholder="Qwen/Qwen3-0.6B"
              value={direct}
              onChange={(event) => setDirect(event.target.value)}
            />
            <Button disabled={!direct.trim()} onClick={() => setModel(direct.trim())}>
              读取版本
            </Button>
          </Space.Compact>
        </div>
        <p className="help-text">
          点击“参数规模”或“预计 FP16
          运行占用”列头可切换升序、降序和默认顺序；排序覆盖全部筛选结果，未知值始终置后。
          {config.data?.scope} 原生纯文本模型缺少体积摘要时保留为待检查，不当作不兼容；
          缺少架构可取消筛选或输入仓库 ID
          检查。量化格式、多模态、远程代码与不支持的架构会在检查时明确拒绝。
        </p>
      </Panel>
      <Panel
        title="当前节点下载任务"
        subtitle="字节进度来自实际传输；到 100% 后仍需校验。取消 / 中断保留部分文件，显式继续时校验并续传。"
      >
        <Table<DownloadJob>
          size="small"
          rowKey="id"
          dataSource={downloads.data?.items ?? []}
          scroll={{ x: 1050 }}
          pagination={{ pageSize: 8 }}
          locale={{ emptyText: '尚无下载任务。选择模型版本并通过检查后，点击下载。' }}
          columns={[
            {
              title: '权重与版本',
              render: (_, row) => (
                <div>
                  <strong>{row.model_id}</strong>
                  <div>
                    {row.revision} · 第 {row.attempt} 次尝试
                  </div>
                  <code>{row.id.slice(0, 21)}</code>
                </div>
              ),
            },
            {
              title: '下载与校验',
              width: 330,
              render: (_, row) => (
                <div>
                  <Tag>{states[row.state] ?? row.state}</Tag>
                  <Progress
                    percent={
                      row.total_bytes
                        ? Math.min(
                            100,
                            Math.round((row.downloaded_bytes / row.total_bytes) * 1000) / 10,
                          )
                        : 0
                    }
                    status={
                      row.state === 'failed'
                        ? 'exception'
                        : row.state === 'completed'
                          ? 'success'
                          : active(row.state)
                            ? 'active'
                            : 'normal'
                    }
                  />
                  <div className="help-text">
                    {size(row.downloaded_bytes)} / {size(row.total_bytes)} ·{' '}
                    {row.bytes_per_second == null
                      ? '—'
                      : `${fmt(row.bytes_per_second / 1024 ** 2, 1)} MiB/s`}{' '}
                    · 已校验 {row.verified_files}/{row.total_files}
                  </div>
                  {row.current_file && <code>{row.current_file}</code>}
                </div>
              ),
            },
            {
              title: '结果 / 原因',
              render: (_, row) => (
                <div className="model-hub-error">
                  {row.error ??
                    (row.state === 'completed'
                      ? '发布方 SHA-256 已核验；运行尚需验证'
                      : '保存在当前节点')}{' '}
                  {row.snapshot_path && (
                    <details>
                      <summary>制品目录与计划身份</summary>
                      <code>
                        {row.snapshot_path}
                        <br />
                        {row.plan_sha256}
                      </code>
                    </details>
                  )}
                </div>
              ),
            },
            {
              title: '操作',
              render: (_, row) => (
                <Space direction="vertical">
                  {active(row.state) && (
                    <Button
                      size="small"
                      disabled={row.state === 'cancelling' || !!busy}
                      onClick={() => void act(row.id, 'cancel')}
                    >
                      取消下载
                    </Button>
                  )}
                  {['failed', 'cancelled', 'interrupted'].includes(row.state) && (
                    <Button
                      size="small"
                      disabled={!!busy}
                      loading={busy === row.id}
                      onClick={() => void act(row.id, 'resume')}
                    >
                      继续下载
                    </Button>
                  )}
                  {row.state === 'completed' &&
                    (row.deployment_id ? (
                      <Link to={`/playground?deployment=${row.deployment_id}`}>
                        到推理页加载验证 →
                      </Link>
                    ) : (
                      <Button
                        size="small"
                        disabled={!!busy}
                        loading={busy === row.id}
                        onClick={() => void act(row.id, 'register')}
                      >
                        登记为推理部署
                      </Button>
                    ))}
                </Space>
              ),
            },
          ]}
        />
        <p className="help-text">
          下载串行排队，不自动占用
          GPU。登记后进入推理页显式加载，可继续使用固定输入评测和观测。新增模型默认框架原生，下载成功不代表自定义算子适配、质量或性能验收通过。
        </p>
      </Panel>
      {model && <VersionDrawer key={model} modelId={model} onClose={() => setModel('')} />}
    </>
  );
}
