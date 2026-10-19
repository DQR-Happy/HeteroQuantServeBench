import { useQuery } from '@tanstack/react-query';
import { Alert, Descriptions, Empty, Select, Tag } from 'antd';
import { Link, NavLink, Navigate, useParams, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import { api, fmt, gib } from '../api/client';
import type { Deployment } from '../api/types';
import { PageHead, QueryState } from '../components';
import StructureExplorer from '../features/architecture/StructureExplorer';
import ExecutionExplorer from '../features/architecture/ExecutionExplorer';
import RuntimeExplorer from '../features/architecture/RuntimeExplorer';
import StaticArchitecture from '../features/architecture/StaticArchitecture';
import ArchitectureProvenance from '../features/architecture/ArchitectureProvenance';
import type { Architecture } from '../features/architecture/types';
import '../features/architecture/styles.css';
const sections = [
  { id: 'structure', label: '模块与权重' },
  { id: 'semantic', label: '静态计算结构' },
  { id: 'execution', label: '运行算子观测' },
  { id: 'compiler', label: '编译与导出' },
  { id: 'provenance', label: '来源与凭据' },
];
function ArchitectureView({
  data,
  deployment,
  section,
  runtimeRun,
}: {
  data: Architecture;
  deployment?: Deployment;
  section: string;
  runtimeRun?: string | null;
}) {
  const [params, setParams] = useSearchParams();
  const requested = params.get('graph') ?? 'fx';
  const graphKind = ['fx', 'onnx', 'inductor_pre', 'inductor_post', 'generated_code'].includes(
    requested,
  )
    ? requested
    : 'fx';
  return (
    <>
      {data.runtime.structure_receipt?.artifact_id && <Alert type="info" showIcon message={`当前加载 L5 制品：${data.runtime.structure_receipt.artifact_id}`} description={<span>运行时模块树与参数反映实际变换；静态 config、语义图和原始权重索引来自源检查点。<Link to={`/model-structure/artifacts?deployment=${encodeURIComponent(data.deployment_id)}&artifact=${data.runtime.structure_receipt.artifact_id}`}>查看模块覆写配方与制品 →</Link></span>}/>}
      <div className="architecture-context">
        <Tag>{data.config.model_type}</Tag>
        <span>
          {data.config.num_hidden_layers ?? '?'} 层 · hidden {data.config.hidden_size ?? '?'}
        </span>
        <span>{fmt(gib(data.stored_bytes), 3)} GiB 存储</span>
        <Tag>{deployment?.state ?? '未知部署状态'}</Tag>
      </div>
      <nav className="architecture-subnav" aria-label="模型架构子页">
        {sections.map((s) => (
          <NavLink key={s.id} to={`/model-architecture/${s.id}?${params.toString()}`}>
            {s.label}
          </NavLink>
        ))}
      </nav>
      {section === 'structure' && <StructureExplorer data={data} />}
      {section === 'semantic' && <StaticArchitecture data={data} />}
      {section === 'execution' && <ExecutionExplorer data={data} deployment={deployment} />}
      {section === 'compiler' && (
        <>
          <div className="architecture-picker">
            <div>
              <label className="field-label" htmlFor="architecture-graph-type">
                计算图类型
              </label>
              <Select
                id="architecture-graph-type"
                aria-label="计算图类型"
                value={graphKind}
                style={{ width: 360, maxWidth: '100%' }}
                onChange={(kind) =>
                  setParams((previous) => {
                    const next = new URLSearchParams(previous);
                    next.set('graph', kind);
                    return next;
                  })
                }
                options={[
                  { value: 'fx', label: 'Dynamo FX · 编译输入' },
                  { value: 'onnx', label: 'ONNX · 固定 prefill 导出' },
                  { value: 'inductor_pre', label: 'Inductor IR · 融合前' },
                  { value: 'inductor_post', label: 'Inductor IR · 融合后' },
                  { value: 'generated_code', label: 'Inductor · 生成代码' },
                ]}
              />
            </div>
          </div>
          <RuntimeExplorer
            data={data}
            deployment={deployment}
            initialRun={runtimeRun}
            representation={graphKind}
          />
        </>
      )}
      {section === 'provenance' && (
        <>
          <Descriptions
            size="small"
            column={{ xs: 1, md: 2 }}
            items={[
              {
                key: 'class',
                label: '配置声明类',
                children: (data.config.architectures as string[] | null)?.join(', ') ?? '未知',
              },
              {
                key: 'runtime',
                label: '当前加载类 / 算子',
                children: `${data.runtime.model_class ?? '未加载'} / ${data.runtime.kernel_profile ?? '未知'}`,
              },
              {
                key: 'count',
                label: '存储 tensor / 元素数',
                children: `${data.stored_tensor_count} / ${fmt(data.stored_parameter_elements / 1e9, 3)} B`,
              },
              {
                key: 'dtype',
                label: '制品存储类型',
                children: Object.keys(data.dtypes).join(' / '),
              },
            ]}
          />
          <ArchitectureProvenance data={data} />
        </>
      )}
    </>
  );
}
export default function ModelArchitecture() {
  const [params, setParams] = useSearchParams();
  const { section } = useParams();
  const legacy = params.get('graph');
  const resolvedSection = sections.some((s) => s.id === section)
    ? section!
    : legacy === 'static'
      ? 'semantic'
      : legacy === 'aten'
        ? 'execution'
        : legacy || params.has('runtime_run')
          ? 'compiler'
          : 'structure';
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const id =
    params.get('deployment') ??
    deployments.data?.items.find((item) => item.provider === 'pytorch')?.id;
  const query = useQuery({
    queryKey: [
      'model-architecture',
      id,
      deployments.data?.items.find((item) => item.id === id)?.epoch,
    ],
    queryFn: () => api<Architecture>(`/deployments/${encodeURIComponent(id!)}/architecture`),
    enabled: !!id,
    staleTime: 60000,
    retry: false,
  });
  if (section !== resolvedSection)
    return <Navigate replace to={`/model-architecture/${resolvedSection}?${params.toString()}`} />;
  return (
    <>
      <PageHead
        kicker="ARTIFACT / LOGICAL ARCHITECTURE"
        title="模型架构"
        description="从当前节点已下载制品查看模块与计算依赖，区分串行关系、逻辑并行机会和实际执行证据。"
        extra={
          <Link to="/devices">
            <Button>设备与部署</Button>
          </Link>
        }
      />
      <div className="architecture-picker">
        <div>
          <label className="field-label" htmlFor="architecture-model">
            已登记的本地模型制品
          </label>
          <Select
            id="architecture-model"
            className="wide-select"
            value={id}
            options={deployments.data?.items.map((item) => ({
              value: item.id,
              label: item.name,
              disabled: item.provider !== 'pytorch',
            }))}
            onChange={(value) =>
              setParams((previous) => {
                const next = new URLSearchParams(previous);
                next.set('deployment', value);
                next.delete('runtime_run');
                return next;
              })
            }
          />
        </div>
        <Button loading={query.isFetching} onClick={() => void query.refetch()}>
          重新读取制品
        </Button>
        <Button
          disabled={!query.data}
          disabledReason={{
            reason: query.error
              ? '制品读取失败。请点击“重新读取制品”重试；若尚未下载权重，请先下载并登记模型。'
              : '尚未读取到模型制品，请等待读取完成；未登记的模型需要先下载并登记。',
            next: { to: '/models', label: '下载并登记模型' },
          }}
          onClick={() => {
            if (!query.data) return;
            const url = URL.createObjectURL(
              new Blob([JSON.stringify(query.data, null, 2)], { type: 'application/json' }),
            );
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `architecture-${query.data.deployment_id}-${query.data.snapshot_sha256.slice(0, 12)}.json`;
            anchor.click();
            setTimeout(() => URL.revokeObjectURL(url), 0);
          }}
        >
          导出结构快照
        </Button>
        <Link to="/guide#architecture">架构阅读指南 →</Link>
      </div>
      <QueryState loading={query.isPending && !!id} error={query.error ?? deployments.error} />
      {query.data ? (
        <ArchitectureView
          key={query.data.deployment_id + query.data.snapshot_sha256 + query.data.runtime.epoch}
          data={query.data}
          deployment={deployments.data?.items.find((item) => item.id === id)}
          runtimeRun={params.get('runtime_run')}
          section={resolvedSection}
        />
      ) : !query.isPending && !query.error ? (
        <Empty description="当前节点尚无可读取的本地模型制品" />
      ) : null}
    </>
  );
}
