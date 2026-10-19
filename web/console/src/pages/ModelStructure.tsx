import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import { api, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import { applyKernelProfile } from '../features/kernels/lifecycle';
import CompressionJob from '../features/structure/CompressionJob';
import StructureComparison, { type Comparison } from '../features/structure/StructureComparison';
import type { Artifact, Plan, StructureRun } from '../features/structure/types';
import '../features/structure/styles.css';

const calibrationExample =
  'A small rabbit lived near a river. Each morning it visited the forest and helped its friends.\nA teacher showed the children how seeds grow into plants with water and sunlight.';
export default function ModelStructure() {
  const { section = 'build' } = useParams();
  const [params, setParams] = useSearchParams();
  const [method, setMethod] = useState('mlp_channels');
  const [targets, setTargets] = useState<string[]>([]);
  const [maxModules, setMaxModules] = useState(4);
  const [keep, setKeep] = useState(0.5);
  const [sparsity, setSparsity] = useState(0.5);
  const [pattern, setPattern] = useState('unstructured');
  const [calibration, setCalibration] = useState(calibrationExample);
  const [calTokens, setCalTokens] = useState(64);
  const [cap, setCap] = useState<{
    id: string;
    epoch: string;
    groups: { name: string; width: number; hidden: number }[];
    linears: { name: string; in_features: number; out_features: number }[];
  }>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [lifecycle, setLifecycle] = useState<Run>();
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const [baseline, setBaseline] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<Comparison>();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const id =
    params.get('deployment') ??
    deployments.data?.items.find((d) => d.state === 'ready')?.id ??
    deployments.data?.items[0]?.id ??
    '';
  const deployment = deployments.data?.items.find((d) => d.id === id);
  useEffect(() => {
    if (!id || params.has('deployment')) return;
    // Bind the initial choice before unload/load changes which model is ready.
    setParams(
      (previous) => {
        if (previous.has('deployment')) return previous;
        const next = new URLSearchParams(previous);
        next.set('deployment', id);
        return next;
      },
      { replace: true },
    );
  }, [id, params, setParams]);
  const catalog = useQuery({
    queryKey: ['structure-catalog'],
    queryFn: () =>
      api<{
        methods: { id: string; label: string }[];
        not_implemented: { id: string; reason: string }[];
        max_deadline_ms: number;
      }>('/structure/catalog'),
  });
  const overview = useQuery({
    queryKey: ['structure-overview'],
    queryFn: () => api<{ pending: number }>('/overview'),
    refetchInterval: 2000,
  });
  const artifacts = useQuery({
    queryKey: ['structure-artifacts', id],
    queryFn: () =>
      api<{ items: Artifact[] }>(`/structure/artifacts?deployment_id=${encodeURIComponent(id)}`),
    enabled: !!id,
    refetchInterval: 5000,
  });
  const records = useQuery({
    queryKey: ['structure-runs', id],
    queryFn: () =>
      api<{ items: StructureRun[] }>(`/structure/runs?deployment_id=${encodeURIComponent(id)}`),
    enabled: !!id,
    refetchInterval: 3000,
  });
  const job = useRun(params.get('job') ?? undefined);
  const measure = useRun(params.get('run') ?? undefined);
  const selectedArtifact = params.get('artifact') ?? artifacts.data?.items[0]?.artifact_id;
  const artifact = artifacts.data?.items.find((a) => a.artifact_id === selectedArtifact);
  const actualCap = cap?.id === id && cap.epoch === deployment?.epoch ? cap : undefined;
  const deviceLink = `/devices#deployment-${encodeURIComponent(id)}`;
  const pending =
    !!overview.data?.pending ||
    (!!job.data && !terminal(job.data.state)) ||
    (!!measure.data && !terminal(measure.data.state));
  const original =
    deployment?.state === 'ready' &&
    !deployment.structure_artifact_id &&
    (deployment.quantization?.method ?? 'none') === 'none' &&
    !deployment.quantization?.kv?.bits &&
    (deployment.kernel_profile ?? 'framework-native') === 'framework-native';
  const update = (key: string, value: string) =>
    setParams((previous) => {
      const next = new URLSearchParams(previous);
      next.set(key, value);
      return next;
    });
  useEffect(() => {
    setTargets([]);
    setBaseline('');
    setCandidate('');
    setComparison(undefined);
    setCap(undefined);
  }, [id]);
  const execute = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const activeReason = busy
    ? '当前操作正在提交或加载，请等待完成。'
    : pending
      ? '当前任务还未结束，请等待清理完成或在运行记录中取消。'
      : undefined;
  const modelReason =
    activeReason ??
    (deployment?.state !== 'ready'
      ? '请先加载所选模型。'
      : deployment.provider !== 'pytorch'
        ? '此功能需要本机 PyTorch 模型；请在设备页选择本机部署。'
        : undefined);
  const buildReason =
    modelReason ??
    (!original
      ? '请在“制品与加载”恢复原始浮点模型与框架原生算子；压缩任务不叠加权重/KV 量化或已有 L5 制品。'
      : !actualCap
        ? '请先点击本页“检查可变换模块”，核对真实 MLP 与 Linear。'
        : !targets.length
          ? '请先在本页选择要变换的模块。'
          : targets.length > maxModules
            ? '选中模块数超过本页模块预算，请减少选择或提高预算。'
            : method === 'wanda' && !calibration.trim()
              ? '请先输入独立校准文本。'
              : undefined);
  const next = activeReason
    ? { to: params.get('job') ? `/runs/${params.get('job')}` : '/runs', label: '查看当前任务' }
    : deployment?.state !== 'ready'
      ? { to: deviceLink, label: '加载模型' }
      : !original
        ? {
            to: `/model-structure/artifacts?deployment=${encodeURIComponent(id)}`,
            label: '查看制品与恢复入口',
          }
        : undefined;
  const plan: Plan = {
    method,
    targets,
    max_modules: maxModules,
    sparsity: pattern === '2:4' ? 0.5 : sparsity,
    pattern: method === 'mlp_channels' || method === 'svd_low_rank' ? 'unstructured' : pattern,
    keep_ratio: keep,
    calibration_max_tokens: calTokens,
    ...(method === 'wanda' ? { calibration: calibration.split('\n').filter((s) => s.trim()) } : {}),
  };
  const load = async (selected?: Artifact) =>
    execute(async () => {
      if (!deployment) return;
      abort.current?.abort();
      abort.current = new AbortController();
      await applyKernelProfile(deployment, 'framework-native', abort.current.signal, setLifecycle, {
        precision: selected?.statistics.export_dtype ?? deployment.precision,
        quantization: { method: 'none' },
        ...(selected ? { structure_artifact_id: selected.artifact_id } : {}),
      });
      await deployments.refetch();
      setCap(undefined);
    });
  const runMeasure = (quality: boolean) =>
    execute(async () => {
      if (!deployment?.epoch) return;
      const run = await api<Run>(
        quality ? '/precision/quality-jobs' : '/requests',
        quality
          ? {
              deployment_id: id,
              expected_epoch: deployment.epoch,
              quality_plan: { benchmark_id: 'wikitext2-mini-v1' },
            }
          : {
              deployment_id: id,
              expected_epoch: deployment.epoch,
              prompt: 'Once upon a time, there was a little fox. The fox',
              max_output_tokens: 16,
              fixed_output_length: true,
              save_input: true,
              optimization: { changes: {} },
              observation_mode: 'off',
              deadline_ms: catalog.data?.max_deadline_ms ?? 120000,
            },
      );
      update('run', run.id);
      await records.refetch();
    });
  const options =
    records.data?.items
      .filter((r) => {
        const receipt = r.metrics.optimization_receipt;
        return (
          r.state === 'completed' &&
          receipt &&
          (receipt.structure_receipt?.artifact_id ||
            (receipt.precision_receipt?.method === 'none' &&
              !receipt.precision_receipt.kv?.bits &&
              receipt.runtime?.kernel === 'framework-native'))
        );
      })
      .map((r) => ({
        value: r.id,
        label: `${r.config.experiment_role === 'quality' ? 'PPL' : '生成'} · ${r.metrics.optimization_receipt?.structure_receipt?.artifact_id ? '压缩' : '原始'} · ${r.id}`,
      })) ?? [];
  return (
    <>
      <PageHead
        kicker="L5"
        title="模型结构与压缩"
        description="从原始权重生成可核对的结构制品，显式加载，再比较质量、性能和真实存储。"
      />
      <nav className="structure-nav" aria-label="L5 子页">
        {[
          ['build', '创建压缩制品'],
          ['artifacts', '制品与加载'],
          ['evaluation', '评测与对照'],
        ].map(([path, label]) => (
          <Link
            key={path}
            aria-current={section === path ? 'page' : undefined}
            to={`/model-structure/${path}?${params}`}
          >
            {label}
          </Link>
        ))}
      </nav>
      <Panel title="模型与当前结构">
        <Space wrap>
          <Select
            aria-label="L5 模型"
            showSearch
            optionFilterProp="label"
            style={{ minWidth: 280 }}
            value={id || undefined}
            onChange={(v) => setParams({ deployment: v })}
            options={deployments.data?.items.map((d) => ({
              value: d.id,
              label: `${d.name} · ${d.state}`,
            }))}
          />
          <Status state={deployment?.state ?? 'unloaded'} />
          <Link to={deviceLink}>设备与模型生命周期 →</Link>
          <Link to={`/model-architecture/structure?deployment=${encodeURIComponent(id)}`}>
            查看实际模块与权重 →
          </Link>
        </Space>
        <p>
          当前制品：{deployment?.structure_artifact_id ?? '原始模型'}。加载切换会卸载当前 GPU
          模型，请保持本页打开直到完成。
        </p>
        {lifecycle && (
          <p>
            <Link to={`/runs/${lifecycle.id}`}>
              {lifecycle.kind} · {lifecycle.id}
            </Link>{' '}
            · {lifecycle.state}
          </p>
        )}
      </Panel>
      {error && (
        <Alert type="error" showIcon message={error} closable onClose={() => setError('')} />
      )}
      <QueryState error={catalog.error ?? deployments.error ?? artifacts.error ?? records.error} />
      {section === 'build' ? (
        <>
          <Panel title="变换方法与范围">
            <Space wrap>
              <label>
                方法{' '}
                <Select
                  aria-label="L5 方法"
                  style={{ width: 330 }}
                  value={method}
                  onChange={(v) => {
                    setMethod(v);
                    setTargets([]);
                  }}
                  options={catalog.data?.methods.map((m) => ({ value: m.id, label: m.label }))}
                />
              </label>
              <Button
                disabled={!!modelReason}
                disabledReason={{
                  reason: modelReason ?? '请先加载模型。',
                  next: { to: deviceLink, label: '检查模型' },
                }}
                onClick={() =>
                  void execute(async () => {
                    const value = await api<{
                      groups: NonNullable<typeof cap>['groups'];
                      linears: NonNullable<typeof cap>['linears'];
                    }>(`/structure/capabilities?deployment_id=${encodeURIComponent(id)}`);
                    setCap({ id, epoch: deployment!.epoch!, ...value });
                  })
                }
              >
                检查可变换模块
              </Button>
            </Space>
            <p>
              显式选择目标，不会修改 Embedding、LM Head、Attention 头数或 MoE 路由。MLP
              裁剪将成组修改 Gate/Up 输出行和 Down 输入列。
            </p>
            <Select
              mode="multiple"
              aria-label="L5 目标模块"
              className="structure-targets"
              placeholder="先检查模块，再选择目标"
              value={targets}
              onChange={setTargets}
              options={(method === 'mlp_channels' ? actualCap?.groups : actualCap?.linears)?.map(
                (x) => ({ value: x.name, label: x.name }),
              )}
            />
            <Space wrap>
              <label>
                模块预算{' '}
                <InputNumber
                  aria-label="模块预算"
                  min={1}
                  max={32}
                  value={maxModules}
                  onChange={(v) => setMaxModules(v ?? 4)}
                />
              </label>
              {method === 'mlp_channels' || method === 'svd_low_rank' ? (
                <label>
                  {method === 'svd_low_rank' ? '低秩矩阵参数保留比例' : 'MLP 通道保留比例'}{' '}
                  <InputNumber
                    aria-label="保留比例"
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    value={keep}
                    onChange={(v) => setKeep(v ?? 0.5)}
                  />
                </label>
              ) : (
                <>
                  <label>
                    剪枝形式{' '}
                    <Select
                      aria-label="剪枝形式"
                      style={{ width: 180 }}
                      value={pattern}
                      onChange={setPattern}
                      options={[
                        { value: 'unstructured', label: '按输出行非结构化' },
                        { value: '2:4', label: '2:4 模式 / 稠密执行' },
                      ]}
                    />
                  </label>
                  {pattern === 'unstructured' && (
                    <label>
                      移除比例{' '}
                      <InputNumber
                        aria-label="移除比例"
                        min={0}
                        max={0.9}
                        step={0.1}
                        value={sparsity}
                        onChange={(v) => setSparsity(v ?? 0.5)}
                      />
                    </label>
                  )}
                </>
              )}
            </Space>
            {method === 'wanda' && (
              <>
                <p>
                  独立校准文本，每行一条。Wanda 使用权重绝对值 ×
                  输入激活范数评分；本实现冻结一次统计，不复现官方逐层重校准协议。不要填写质量评测原文。
                </p>
                <Input.TextArea
                  aria-label="结构校准文本"
                  rows={5}
                  value={calibration}
                  onChange={(e) => setCalibration(e.target.value)}
                />
                <label>
                  每条校准 Token 上限{' '}
                  <InputNumber
                    aria-label="结构校准 Token 上限"
                    min={8}
                    max={256}
                    value={calTokens}
                    onChange={(v) => setCalTokens(v ?? 64)}
                  />
                </label>
              </>
            )}
            <p>
              置零仍执行稠密
              Linear；只有成组缩维和双矩阵低秩执行才改变实际参数形状。所有方法都可能损害质量。
            </p>
            <Button
              type="primary"
              disabled={!!buildReason}
              disabledReason={{ reason: buildReason ?? '请核对原模型与变换范围。', next }}
              loading={busy}
              onClick={() =>
                void execute(async () => {
                  const run = await api<Run>('/structure/jobs', {
                    deployment_id: id,
                    expected_epoch: deployment?.epoch,
                    plan,
                  });
                  update('job', run.id);
                })
              }
            >
              创建独立压缩制品
            </Button>
          </Panel>
          {job.data && (
            <>
              <CompressionJob run={job.data as StructureRun} />
              <Button
                disabled={terminal(job.data.state) || busy}
                disabledReason={{
                  reason: busy ? '当前操作正在提交，请稍候。' : '所选压缩任务已结束，无需取消。',
                }}
                onClick={() =>
                  void execute(async () => {
                    await api(`/requests/${job.data!.id}/cancel`, {});
                  })
                }
              >
                取消压缩任务
              </Button>
            </>
          )}
        </>
      ) : section === 'artifacts' ? (
        <>
          <Panel title="独立保存的结构制品">
            <Table<Artifact>
              rowKey="artifact_id"
              pagination={{ pageSize: 8 }}
              scroll={{ x: 900 }}
              dataSource={artifacts.data?.items}
              columns={[
                {
                  title: '制品',
                  dataIndex: 'artifact_id',
                  render: (v: string) => (
                    <Link
                      to={`/model-structure/artifacts?deployment=${encodeURIComponent(id)}&artifact=${v}`}
                    >
                      {v}
                    </Link>
                  ),
                },
                { title: '方法', render: (_, r) => r.recipe.method },
                {
                  title: '原参数 / 压缩参数',
                  render: (_, r) =>
                    `${fmt(r.statistics.parameters_before, 0)} / ${fmt(r.statistics.parameters_after, 0)}`,
                },
                {
                  title: '参数减少',
                  render: (_, r) => `${fmt(r.statistics.parameter_reduction * 100, 2)}%`,
                },
                { title: '磁盘 MiB', render: (_, r) => fmt(r.disk_bytes / 1024 ** 2, 2) },
              ]}
            />
            <Space wrap>
              <Select
                aria-label="L5 制品"
                showSearch
                optionFilterProp="label"
                style={{ width: 350 }}
                value={selectedArtifact}
                onChange={(v) => update('artifact', v)}
                options={artifacts.data?.items.map((a) => ({
                  value: a.artifact_id,
                  label: `${a.recipe.method} · ${a.artifact_id}`,
                }))}
              />
              <Button
                type="primary"
                disabled={!!activeReason || !artifact}
                disabledReason={{
                  reason: activeReason ?? '请先在创建页完成压缩任务并选择制品。',
                  next: { to: `/model-structure/build?${params}`, label: '创建压缩制品' },
                }}
                loading={busy}
                onClick={() => void load(artifact)}
              >
                卸载当前模型并加载此制品
              </Button>
              <Button
                disabled={!!activeReason || !deployment}
                disabledReason={{
                  reason: activeReason ?? '请先选择模型。',
                  next: { to: deviceLink, label: '选择模型' },
                }}
                onClick={() => void load()}
              >
                恢复原始浮点模型
              </Button>
            </Space>
            {artifact && (
              <details>
                <summary>变换配方与实际形状</summary>
                <JsonView value={artifact} />
              </details>
            )}
            <p>
              制品是 HQSB 显式结构配方 + safetensors，加载时恢复模块形状并严格匹配全部 state
              keys。不会在推理中重建完整低秩权重。不是可直接交给任意 Transformers.from_pretrained
              的通用检查点。
            </p>
          </Panel>
        </>
      ) : (
        <>
          <Panel title="对当前驻留方案执行真实评测">
            <p>
              先对原始浮点模型运行 PPL
              和固定生成；加载制品后分别再次运行。质量和生成性能分开记录。选择双方运行 ID
              查看相对原始模型的下降或变化。
            </p>
            <Space wrap>
              <Button
                type="primary"
                disabled={!!modelReason}
                disabledReason={{
                  reason: modelReason ?? '请先加载模型。',
                  next: { to: deviceLink, label: '加载模型' },
                }}
                onClick={() => void runMeasure(true)}
              >
                运行 WikiText-2 Mini / PPL
              </Button>
              <Button
                disabled={!!modelReason}
                disabledReason={{
                  reason: modelReason ?? '请先加载模型。',
                  next: { to: deviceLink, label: '加载模型' },
                }}
                onClick={() => void runMeasure(false)}
              >
                运行固定生成 / 16 Token
              </Button>
              {measure.data && (
                <Link to={`/runs/${measure.data.id}`}>
                  {measure.data.id} · {measure.data.state}
                </Link>
              )}
            </Space>
          </Panel>
          <Panel title="原始基线与结构候选">
            <Space wrap>
              <Select
                aria-label="L5 基线"
                showSearch
                optionFilterProp="label"
                style={{ width: 350 }}
                placeholder="选择原始基线"
                value={baseline || undefined}
                options={options}
                onChange={(v) => {
                  setBaseline(v);
                  setComparison(undefined);
                }}
              />
              <Select
                aria-label="L5 候选"
                showSearch
                optionFilterProp="label"
                style={{ width: 350 }}
                placeholder="选择结构候选"
                value={candidate || undefined}
                options={options}
                onChange={(v) => {
                  setCandidate(v);
                  setComparison(undefined);
                }}
              />
              <Button
                disabled={busy || !baseline || !candidate || baseline === candidate}
                disabledReason={{
                  reason: busy
                    ? '比较请求正在提交，请稍候。'
                    : '请在本页选择原始基线与结构候选的两条不同记录；需要相同类型和协议。',
                  next: { to: `/model-structure/evaluation?${params}`, label: '运行评测' },
                }}
                onClick={() =>
                  void execute(async () =>
                    setComparison(
                      await api<Comparison>('/structure/comparisons', {
                        baseline_run_id: baseline,
                        candidate_run_id: candidate,
                      }),
                    ),
                  )
                }
              >
                比较质量或性能差异
              </Button>
            </Space>
            {comparison && <StructureComparison data={comparison} />}
          </Panel>
        </>
      )}
      <details>
        <summary>本层其他方向与未接入能力</summary>
        {catalog.data?.not_implemented.map((x) => (
          <p key={x.id}>
            <strong>{x.id}</strong>：{x.reason}
          </p>
        ))}
        <p>此处未接入稀疏 GPU kernel、恢复训练，也不承诺任意压缩比例保持质量或必然加速。</p>
      </details>
    </>
  );
}
