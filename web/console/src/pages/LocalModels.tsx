import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, App, Descriptions, Input, Progress, Select, Space, Table, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import { api, fmt } from '../api/client';
import { PageHead, Panel, QueryState } from '../components';
import type { LocalInventory, LocalModel } from '../features/model-hub/types';
import '../features/model-hub/styles.css';

const kinds: Record<string, string> = {
  download: 'ModelScope 下载',
  configured: '节点配置',
  local: '本地目录',
  l4: 'L4 量化制品',
  l5: 'L5 压缩制品',
};
const bytes = (value: number) => {
  const power = value >= 1024 ** 3 ? 3 : value >= 1024 ** 2 ? 2 : value >= 1024 ? 1 : 0;
  return `${fmt(value / 1024 ** power, power ? 2 : 0)} ${['B', 'KiB', 'MiB', 'GiB'][power]}`;
};

export default function LocalModelsPage() {
  const client = useQueryClient();
  const { modal, message } = App.useApp();
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const [kind, setKind] = useState('all');
  const [root, setRoot] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const inventory = useQuery({
    queryKey: ['local-models'],
    queryFn: () => api<LocalInventory>('/model-hub/local-models'),
    refetchInterval: 10000,
    retry: false,
  });
  const invalidate = async () => {
    await Promise.all(
      ['local-models', 'model-hub-downloads', 'model-hub-config', 'deployments'].map((key) =>
        client.invalidateQueries({ queryKey: [key] }),
      ),
    );
  };
  const refresh = async (directory?: string) => {
    setBusy(directory ? 'root' : 'refresh');
    setError('');
    try {
      const data = await api<LocalInventory>(
        `/model-hub/local-models/${directory ? 'roots' : 'refresh'}`,
        directory ? { path: directory } : {},
      );
      client.setQueryData(['local-models'], data);
      if (directory) setRoot('');
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
    }
  };
  const register = async (row: LocalModel) => {
    setBusy(row.id);
    setError('');
    try {
      await api(`/model-hub/downloads/${row.download_id}/register`, {});
      void message.success('已登记部署，可到设备页加载模型');
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy('');
      await invalidate();
    }
  };
  const remove = (row: LocalModel) =>
    modal.confirm({
      title: '删除本地模型文件？',
      width: 640,
      okText: '确认删除',
      cancelText: '保留文件',
      okButtonProps: { danger: true },
      content: (
        <div className="local-model-confirm">
          <p>
            <strong>{row.model_id}</strong> · {row.revision_name}
          </p>
          <p>
            <code>{row.path}</code>
          </p>
          <p>
            磁盘占用 {bytes(row.allocated_bytes)} · 文件大小 {bytes(row.logical_bytes)} ·{' '}
            {row.file_count} 个文件
          </p>
          <p>清理此制品及其下载残留，移除关联的下载部署。历史推理、评测与观测记录继续保留。</p>
          <p>共享硬链接的实际释放空间取决于其他目录的引用。</p>
        </div>
      ),
      onOk: async () => {
        setBusy(row.id);
        setError('');
        try {
          await api(`/model-hub/local-models/${row.id}/delete`, { revision: row.revision });
          void message.success('本地文件已清理');
        } catch (failure) {
          setError((failure as Error).message);
          throw failure;
        } finally {
          setBusy('');
          await invalidate();
        }
      },
    });
  const data = inventory.data;
  const selectedDownload = params.get('download');
  const items = (data?.items ?? []).filter(
    (row) =>
      (kind === 'all' || row.kind === kind) &&
      (!selectedDownload || row.download_id === selectedDownload) &&
      `${row.model_id} ${row.path} ${row.model_type ?? ''} ${row.artifact_id ?? ''} ${row.variant ?? ''}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <>
      <PageHead
        kicker="NODE / LOCAL MODEL STORAGE"
        title="本地模型管理"
        description="集中查看当前节点的模型目录、实际磁盘占用、部署引用与清理操作。"
        extra={
          <Space>
            <Link to="/models">
              <Button>下载模型</Button>
            </Link>
            <Button
              loading={busy === 'refresh'}
              disabled={!!busy}
              disabledReason={{ reason: '正在处理本地模型操作，请等待完成后重新统计。' }}
              onClick={() => void refresh()}
            >
              刷新空间统计
            </Button>
          </Space>
        }
      />
      <QueryState loading={inventory.isPending} error={inventory.error} />
      {error && (
        <Alert
          type="error"
          showIcon
          message="操作未完成"
          description={error}
          className="section-gap"
        />
      )}
      {data && (
        <Panel
          title="节点磁盘与模型占用"
          subtitle={`最近扫描 ${new Date(data.sampled_at * 1000).toLocaleString()} · 空间扫描缓存 30 秒，可手动刷新。`}
        >
          <div className="model-hub-capacity">
            <strong>{data.items.length} 个模型目录</strong>
            <span>模型文件 {bytes(data.logical_bytes)}</span>
            <span>模型磁盘占用 {bytes(data.allocated_bytes)}</span>
            <span>{data.file_count.toLocaleString()} 个文件</span>
          </div>
          <div className="local-model-volumes">
            {data.volumes.map((volume) => (
              <div key={volume.device}>
                <div className="local-model-volume-line">
                  <code>{volume.path}</code>
                  <span>
                    已用 {bytes(volume.used_bytes)} / {bytes(volume.total_bytes)} · 剩余{' '}
                    {bytes(volume.free_bytes)}
                  </span>
                </div>
                <Progress
                  percent={
                    volume.total_bytes
                      ? Math.round((volume.used_bytes / volume.total_bytes) * 1000) / 10
                      : 0
                  }
                  size="small"
                />
              </div>
            ))}
          </div>
          <p className="help-text">
            文件大小按文件内容长度统计；磁盘占用来自文件系统分配块。总计按设备与 inode
            去重，目录行可能共享文件；磁盘已用还包括模型以外的数据。
          </p>
          <details className="section-gap">
            <summary>扫描范围与其他模型目录（{data.roots.length}）</summary>
            <div className="local-model-roots">
              {data.roots.map((path) => (
                <code key={path}>{path}</code>
              ))}
            </div>
            <label htmlFor="local-model-root" className="field-label">
              添加当前节点上的模型目录
            </label>
            <Space.Compact className="local-model-root-input">
              <Input
                id="local-model-root"
                placeholder="例如 /root/autodl-tmp/models"
                value={root}
                onChange={(event) => setRoot(event.target.value)}
              />
              <Button
                disabled={!!busy || !root.trim()}
                disabledReason={{
                  reason: busy
                    ? '正在扫描或处理文件，请等待完成。'
                    : '请在左侧填写当前节点上的模型目录绝对路径。',
                }}
                onClick={() => void refresh(root.trim())}
              >
                添加并扫描
              </Button>
            </Space.Compact>
            <p className="help-text">
              自动收录部署路径、下载目录及 L4/L5
              制品。新增目录保存在当前节点；扫描通过模型配置或制品清单定位目录，最多向下 6
              层。目录变化可点击“刷新空间统计”。
            </p>
          </details>
          {!!data.issues.length && (
            <Alert
              className="section-gap"
              type="warning"
              showIcon
              message="部分目录需要检查"
              description={data.issues.join('；')}
            />
          )}
        </Panel>
      )}
      <Panel
        title="本地模型清单"
        subtitle="按磁盘占用排序，展开目录可查看模型身份、扫描详情和部署引用。"
      >
        <div className="local-model-filters">
          <Input
            aria-label="搜索本地模型"
            placeholder="搜索模型名称、架构或目录"
            allowClear
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <Select
            aria-label="模型来源"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'all', label: '全部来源' },
              ...Object.entries(kinds).map(([value, label]) => ({ value, label })),
            ]}
          />
        </div>
        {selectedDownload && (
          <p className="help-text">
            正在查看对应下载制品。
            <Button type="link" onClick={() => setParams({}, { replace: true })}>
              显示全部本地模型
            </Button>
          </p>
        )}
        <Table<LocalModel>
          size="small"
          rowKey="id"
          dataSource={items}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1100 }}
          locale={{ emptyText: '当前扫描范围内没有匹配的模型目录，可添加目录或到下载页取得模型。' }}
          expandable={{
            expandedRowRender: (row) => (
              <Descriptions
                column={1}
                size="small"
                items={[
                  { key: 'path', label: '模型目录', children: <code>{row.path}</code> },
                  {
                    key: 'identity',
                    label: '版本 / 制品身份',
                    children: row.artifact_id ?? row.revision_name,
                  },
                  { key: 'revision', label: '目录快照', children: <code>{row.revision}</code> },
                  {
                    key: 'files',
                    label: '文件与扫描',
                    children: `${row.file_count} 个文件 · ${row.complete ? '扫描完成' : '请查看扫描问题'}`,
                  },
                  {
                    key: 'issues',
                    label: '扫描详情',
                    children: row.issues.length ? row.issues.join('；') : '已完成文件元数据统计',
                  },
                ]}
              />
            ),
          }}
          columns={[
            {
              title: '模型与目录',
              render: (_, row) => (
                <div className="local-model-name">
                  <strong>{row.model_id}</strong>
                  <span>
                    {row.variant
                      ? `${row.variant} · ${row.artifact_id?.slice(0, 15) ?? row.model_type ?? ''}`
                      : (row.model_type ?? row.artifact_id ?? row.revision_name)}
                  </span>
                  <code>{row.path}</code>
                </div>
              ),
            },
            {
              title: '来源 / 状态',
              width: 160,
              render: (_, row) => (
                <Space direction="vertical" size={4}>
                  <Tag>{kinds[row.kind] ?? row.kind}</Tag>
                  <span>
                    {!row.exists
                      ? '目录已移除'
                      : row.download_state && row.download_state !== 'completed'
                        ? `下载状态：${row.download_state}`
                        : row.complete
                          ? '文件已在本地'
                          : '扫描待检查'}
                  </span>
                </Space>
              ),
            },
            {
              title: '磁盘占用',
              dataIndex: 'allocated_bytes',
              width: 140,
              defaultSortOrder: 'descend',
              sorter: (a, b) => a.allocated_bytes - b.allocated_bytes,
              render: (value) => bytes(value),
            },
            {
              title: '文件大小',
              dataIndex: 'logical_bytes',
              width: 135,
              sorter: (a, b) => a.logical_bytes - b.logical_bytes,
              render: (value) => bytes(value),
            },
            {
              title: '部署引用',
              width: 195,
              render: (_, row) =>
                row.references.length ? (
                  <Space direction="vertical">
                    {row.references.map((ref) => (
                      <Link key={ref.id} to={`/devices#deployment-${ref.id}`}>
                        {ref.name} · {ref.state}
                      </Link>
                    ))}
                  </Space>
                ) : (
                  <span className="help-text">尚未关联部署</span>
                ),
            },
            {
              title: '操作',
              width: 160,
              render: (_, row) => (
                <Space direction="vertical" size={6}>
                  {row.download_id && row.download_state === 'completed' && !row.deployment_id && (
                    <Button
                      size="small"
                      disabled={!!busy}
                      disabledReason={{ reason: '正在处理模型操作，请等待完成后登记部署。' }}
                      onClick={() => void register(row)}
                    >
                      登记为推理部署
                    </Button>
                  )}
                  {!!row.references.length && (
                    <Link to={`/playground?deployment=${row.references[0].id}`}>
                      打开推理工作台 →
                    </Link>
                  )}
                  {row.download_id && row.download_state !== 'completed' && (
                    <Link to="/models">查看下载任务 →</Link>
                  )}
                  <Button
                    size="small"
                    danger
                    disabled={!!busy || !row.deletion.allowed}
                    disabledReason={
                      busy ? { reason: '正在处理模型文件，请等待完成。' } : row.deletion
                    }
                    onClick={() => remove(row)}
                  >
                    {row.download_state && row.download_state !== 'completed'
                      ? '清理下载文件'
                      : '删除本地制品'}
                  </Button>
                </Space>
              ),
            },
          ]}
        />
      </Panel>
    </>
  );
}
