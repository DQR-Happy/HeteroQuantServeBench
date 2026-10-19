import Button from '../components/ActionButton';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Checkbox, Descriptions, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, ApiError, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { applyKernelProfile, unloadDeployment } from '../features/kernels/lifecycle';
import QualityBaseline from '../features/experiments/QualityBaseline';

type Receipt = {
  method: string;
  compute_dtype: string;
  artifact_id?: string;
  quantized_module_count: number;
  quantized_storage_bytes: number;
  remaining_float_parameter_bytes: number;
  execution: string;
};
type Method = {
  id: string;
  name: string;
  source: string;
  requires_calibration: boolean;
  calibration_available?: boolean;
  availability: string;
  reasons?: string[];
};
type Capabilities = {
  methods: Method[];
  compute_dtypes: string[];
  effective?: Receipt;
  sm?: number[];
};
type Artifact = {
  artifact_id: string;
  method: string;
  compute_dtype: string;
  disk_bytes: number;
  source_model_hash: string;
  source_model?: string;
};
type QualityBenchmark = {
  id: string;
  name: string;
  sample_count: number;
  max_tokens: number;
  dataset_sha256: string;
  articles: string[];
  source: { url: string; variant: string };
};
type QualityComparison = {
  candidates: {
    run_id: string;
    quality: {
      metric?: string;
      status: string;
      baseline_perplexity?: number;
      candidate_perplexity?: number;
      mean_nll_increase?: number;
      ppl_ratio?: number;
    };
  }[];
};
const qualityStatuses: Record<string, string> = {
  measured_no_acceptance_limits: '仅测量，未设阈值',
  passed: '满足预登记阈值',
  failed: '超出预登记阈值',
};
const calibrationExample = [
  'Once there was a small rabbit who lived near a river. Every morning it visited its friends.',
  'A boy found an old key in the garden. He asked his mother what the key could open.',
  'The blue bird flew over the mountains and returned home before sunset.',
  'A kind teacher explained how plants grow when they have water and sunlight.',
].join('\n');

export default function Precision() {
  const client = useQueryClient();
  const [params] = useSearchParams();
  const [selected, setSelected] = useState(params.get('deployment') ?? '');
  const [dtype, setDtype] = useState('float16');
  const [allowTf32, setAllowTf32] = useState(false);
  const [method, setMethod] = useState('none');
  const [bits, setBits] = useState(4);
  const [group, setGroup] = useState(128);
  const [calibration, setCalibration] = useState(calibrationExample);
  const [calibrationTokens, setCalibrationTokens] = useState(128);
  const [alpha, setAlpha] = useState(0.5);
  const [doubleQuant, setDoubleQuant] = useState(true);
  const [threshold, setThreshold] = useState(6);
  const [excluded, setExcluded] = useState('lm_head');
  const [kvBits, setKvBits] = useState<number | null>(null);
  const [kvResidual, setKvResidual] = useState(128);
  const [artifact, setArtifact] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [lifecycleRun, setLifecycleRun] = useState<Run>();
  const [jobId, setJobId] = useState('');
  const [prompt, setPrompt] = useState('Once upon a time a little fox found a ball. The fox');
  const [outputTokens, setOutputTokens] = useState(16);
  const [qualityText, setQualityText] = useState(
    'A red cat sat beside the window.\nThe young girl smiled at her sister.',
  );
  const [qualityLimits, setQualityLimits] = useState('');
  const [qualityPreset, setQualityPreset] = useState('wikitext2-mini-v1');
  const [pplLimits, setPplLimits] = useState('');
  const [qualityJson, setQualityJson] = useState('');
  const [qualityGeneration, setQualityGeneration] = useState(4);
  const [baseline, setBaseline] = useState('');
  const [candidates, setCandidates] = useState('');
  const [comparison, setComparison] = useState<QualityComparison>();
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const deployment =
    deployments.data?.items.find((d) => d.id === selected) ??
    deployments.data?.items.find((d) => d.provider === 'pytorch' && d.state === 'ready') ??
    deployments.data?.items.find((d) => d.provider === 'pytorch');
  useEffect(() => {
    if (!selected && deployment) setSelected(deployment.id);
  }, [deployment, selected]);
  const catalog = useQuery({
    queryKey: ['precision-catalog'],
    queryFn: () => api<Capabilities>('/precision/catalog'),
  });
  const job = useQuery({
    queryKey: ['run', jobId],
    queryFn: () => api<Run>(`/runs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data && terminal(q.state.data.state) ? false : 1000),
  });
  // A successful POST only accepts the job; the worker remains busy until its
  // terminal record is observed. Missing poll data cannot establish completion.
  const running = !!jobId && (!job.data || !terminal(job.data.state));
  const transitioning = deployments.data?.items.some(
    (d) => d.provider === 'pytorch' && !['ready', 'failed', 'unloaded'].includes(d.state),
  );
  const blocked = busy || running || transitioning || deployments.isError;
  const blockedReason = deployments.isError
    ? '部署状态读取失败，请先检查当前后端连接并刷新部署列表。'
    : running
      ? '当前量化或质量评测任务尚未完成，请先等待完成或查看运行进度。'
      : transitioning
        ? '模型正在加载、卸载或量化，请等待部署状态恢复后再操作。'
        : busy
          ? '当前精度方案操作正在提交或切换模型，请等待完成。'
          : '';

  const capability = useQuery({
    queryKey: ['precision-capability', deployment?.id, deployment?.epoch],
    queryFn: ({ signal }) =>
      api<Capabilities>(`/deployments/${deployment!.id}/precision-capabilities`, undefined, signal),
    enabled: deployment?.state === 'ready' && !blocked,
    // Re-enabling after any job must refresh even if that job finished inside
    // the global stale window. Avoid a second imperative fetch racing this one.
    staleTime: 0,
    retry: false,
    refetchInterval: (q) => (q.state.error ? 3000 : false),
  });
  const capabilityBusy =
    capability.error instanceof ApiError && capability.error.code === 'DEVICE_BUSY';
  const artifacts = useQuery({
    queryKey: ['precision-artifacts'],
    queryFn: () => api<{ items: Artifact[] }>('/precision/artifacts'),
  });
  const qualityBenchmarks = useQuery({
    queryKey: ['precision-quality-benchmarks'],
    queryFn: () => api<{ items: QualityBenchmark[] }>('/precision/quality-benchmarks'),
  });
  const benchmark = qualityBenchmarks.data?.items.find((b) => b.id === qualityPreset);
  const methods = capability.data?.methods ?? catalog.data?.methods ?? [];
  const chosen = methods.find((m) => m.id === method);
  const precisionError =
    method !== 'none' && dtype === 'float32'
      ? '量化执行请选择 FP16 或 BF16；FP32 用作浮点基线。'
      : ['rtn', 'gptq', 'awq'].includes(method) && dtype !== 'float16'
        ? '当前融合反量化算子要求 FP16。'
        : '';
  const calibrated = chosen?.requires_calibration;
  const effective = deployment?.state === 'ready' ? capability.data?.effective : undefined;
  const recipe = (): Record<string, unknown> => ({
    method,
    allow_tf32: allowTf32,
    ...(['rtn', 'gptq', 'awq'].includes(method)
      ? { bits: method === 'awq' ? 4 : bits, group_size: group }
      : {}),
    ...(calibrated && !artifact
      ? {
          calibration: calibration
            .split('\n')
            .map((s) => s.trim())
            .filter(Boolean),
          calibration_tokens: calibrationTokens,
        }
      : {}),
    ...(method === 'smoothquant' ? { smooth_alpha: alpha } : {}),
    ...(method === 'bnb_int8' ? { outlier_threshold: threshold } : {}),
    ...(['bnb_nf4', 'bnb_fp4'].includes(method) ? { double_quant: doubleQuant } : {}),
    exclude_modules: excluded
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
    ...(kvBits ? { kv_bits: kvBits, kv_residual_length: kvResidual, kv_group_size: 32 } : {}),
    ...(artifact ? { artifact_id: artifact } : {}),
  });
  const lifecycle = async (action: 'load' | 'unload') => {
    if (!deployment || blocked || controller.current) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError('');
    try {
      if (action === 'load')
        await applyKernelProfile(deployment, 'framework-native', abort.signal, setLifecycleRun, {
          precision: dtype,
          quantization: recipe(),
        });
      else await unloadDeployment(deployment, abort.signal, setLifecycleRun);
    } catch (failure) {
      if (!abort.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!abort.signal.aborted) {
        controller.current = null;
        // Publish the new deployment epoch before re-enabling its capability
        // query. An imperative refetch here bypasses the query's busy guard.
        await client.invalidateQueries({ queryKey: ['deployments'] });
        void client.invalidateQueries({ queryKey: ['precision-artifacts'] });
        if (!abort.signal.aborted) setBusy(false);
      }
    }
  };
  const submit = async (quality: boolean) => {
    if (!deployment || deployment.state !== 'ready' || blocked) return;
    setBusy(true);
    setError('');
    try {
      const result = await api<Run>(quality ? '/precision/quality-jobs' : '/requests', {
        deployment_id: deployment.id,
        expected_epoch: deployment.epoch,
        ...(quality && qualityPreset !== 'custom'
          ? {
              quality_plan: { benchmark_id: qualityPreset },
              ...(pplLimits.trim() ? { ppl_limits: JSON.parse(pplLimits) } : {}),
            }
          : quality
            ? {
                quality_plan: {
                  samples: qualityJson.trim()
                    ? JSON.parse(qualityJson)
                    : qualityText
                        .split('\n')
                        .filter((s) => s.trim())
                        .map((text, i) => ({ id: `sample-${i}`, text })),
                  max_tokens: 64,
                  generation_tokens: qualityGeneration,
                },
                ...(qualityLimits.trim() ? { quality_limits: JSON.parse(qualityLimits) } : {}),
              }
            : { prompt, max_output_tokens: outputTokens, observation_mode: 'basic' }),
      });
      client.setQueryData(['run', result.id], result);
      setJobId(result.id);
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHead
        kicker="L4"
        title="表示精度与量化"
        description="选择计算精度、校准算法与实际低比特执行路径，核对存储、调用凭据和独立质量评测。"
        extra={<Link to="/quantization">历史 RTN 制品与质量证据</Link>}
      />
      <details className="section-gap">
        <summary>使用顺序与指标说明</summary>
        <ol>
          <li>先用 Floating point / FP16 加载同一模型，运行独立质量评测，保存基线运行 ID。</li>
          <li>选择量化方案；需要校准的方法填代表性文本，与独立质量文本分开，再应用并重载。</li>
          <li>
            检查实际方法、模块存储与调用凭据。测试当前模型得到 Token/s、TTFT
            和内存；冷启动和稳定运行分别比较。
          </li>
          <li>
            用相同质量文本评测候选，填写基线与候选运行 ID
            生成对照。若要判定达标，双方运行前登记相同阈值。
          </li>
          <li>
            制品按源权重、配方和版本保存，可复用加载。量化模块字节、其余浮点参数、分配器峰值是不同口径；各位宽方案的速度与质量可通过相同负载的评测比较。
          </li>
        </ol>
      </details>
      <Panel
        title="模型与量化方案"
        subtitle="应用时卸载当前模型，再加载所选方案。校准在独立环境执行；保持本页打开直到切换完成。"
      >
        <QueryState
          loading={deployments.isPending || catalog.isPending}
          error={deployments.error ?? catalog.error}
        />
        <Space wrap>
          <Checkbox
            checked={allowTf32}
            disabled={!!blocked}
            onChange={(e) => setAllowTf32(e.target.checked)}
          >
            允许 TF32（FP32 GEMM）
          </Checkbox>
          <Select
            aria-label="L4 实验模型"
            style={{ minWidth: 270 }}
            value={deployment?.id}
            disabled={!!blocked}
            options={deployments.data?.items
              .filter((d) => d.provider === 'pytorch')
              .map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
            onChange={(value) => {
              setSelected(value);
              setArtifact(undefined);
            }}
          />
          <Select
            aria-label="计算精度"
            value={dtype}
            disabled={!!blocked}
            onChange={setDtype}
            options={(catalog.data?.compute_dtypes ?? []).map((value) => ({ value, label: value }))}
          />
          <Select
            aria-label="量化方法"
            value={method}
            disabled={!!blocked}
            style={{ minWidth: 260 }}
            onChange={(value) => {
              setMethod(value);
              setArtifact(undefined);
            }}
            options={methods.map((m) => ({
              value: m.id,
              label: m.name,
              disabled: m.availability === 'unsupported',
              title: m.reasons?.length ? m.reasons.join('；') : m.name,
            }))}
          />
        </Space>
        {chosen && (
          <p>
            <a href={chosen.source} target="_blank" rel="noreferrer">
              方法原始资料
            </a>{' '}
            ·{' '}
            {chosen.availability === 'unsupported'
              ? chosen.reasons?.join('；')
              : '最终能力以实际加载与执行结果为准'}
          </p>
        )}
        <Space wrap>
          {['rtn', 'gptq'].includes(method) && (
            <label>
              权重位宽{' '}
              <Select
                aria-label="权重位宽"
                value={bits}
                onChange={setBits}
                options={[4, 8].map((value) => ({ value, label: `W${value}` }))}
              />
            </label>
          )}
          {['rtn', 'gptq', 'awq'].includes(method) && (
            <label>
              分组大小{' '}
              <Select
                aria-label="分组大小"
                value={group}
                onChange={setGroup}
                options={[32, 64, 128].map((value) => ({ value, label: String(value) }))}
              />
            </label>
          )}
          {method === 'smoothquant' && (
            <label>
              平滑 α{' '}
              <InputNumber
                min={0}
                max={1}
                step={0.1}
                value={alpha}
                onChange={(v) => setAlpha(v ?? 0.5)}
              />
            </label>
          )}
          {method === 'bnb_int8' && (
            <label>
              离群值阈值{' '}
              <InputNumber
                min={0}
                max={64}
                value={threshold}
                onChange={(v) => setThreshold(v ?? 6)}
              />
            </label>
          )}
          {['bnb_nf4', 'bnb_fp4'].includes(method) && (
            <Checkbox checked={doubleQuant} onChange={(e) => setDoubleQuant(e.target.checked)}>
              二次量化 scale
            </Checkbox>
          )}
          <label>
            保留浮点模块{' '}
            <Input
              aria-label="保留浮点模块"
              value={excluded}
              onChange={(e) => setExcluded(e.target.value)}
              placeholder="逗号分隔模块名"
            />
          </label>
          <label>
            KV 存储{' '}
            <Select
              aria-label="KV 量化"
              value={kvBits}
              onChange={setKvBits}
              options={[
                { value: null, label: '模型默认' },
                { value: 4, label: 'INT4 Quanto' },
                { value: 2, label: 'INT2 Quanto' },
              ]}
              style={{ width: 150 }}
            />
          </label>
          {kvBits && (
            <label>
              浮点 KV 残留长度{' '}
              <InputNumber
                min={1}
                max={4096}
                value={kvResidual}
                onChange={(v) => setKvResidual(v ?? 128)}
              />
            </label>
          )}
        </Space>
        {calibrated && (
          <>
            <p>
              校准文本，每行一条。下面是功能验证示例；正式评测应换成代表性校准集，并与质量评测集分开。
            </p>
            <Input.TextArea
              aria-label="校准文本"
              rows={5}
              value={calibration}
              disabled={!!artifact || !!blocked}
              onChange={(e) => setCalibration(e.target.value)}
            />
            <label>
              每条校准 Token 上限{' '}
              <InputNumber
                min={16}
                max={512}
                value={calibrationTokens}
                onChange={(v) => setCalibrationTokens(v ?? 128)}
              />
            </label>
          </>
        )}
        {calibrated && chosen?.calibration_available === false && !artifact && (
          <Alert
            type="warning"
            showIcon
            message="此机器未安装独立校准环境；可复用已保存制品，或先在服务端安装 L4 校准依赖。"
          />
        )}
        <p>
          <Select
            aria-label="复用量化制品"
            allowClear
            placeholder="可选：复用已保存制品，跳过重新校准"
            style={{ width: 'min(100%, 650px)' }}
            value={artifact}
            onChange={setArtifact}
            options={artifacts.data?.items
              .filter(
                (a) =>
                  a.method === method &&
                  a.compute_dtype === `torch.${dtype}` &&
                  (!a.source_model || a.source_model === deployment?.model),
              )
              .map((a) => ({
                value: a.artifact_id,
                label: `${a.source_model ?? a.source_model_hash.slice(0, 8)} · ${a.method} · ${a.artifact_id.slice(0, 12)}`,
              }))}
          />
        </p>
        <Space wrap>
          <Button
            type="primary"
            loading={busy}
            disabledReason={{
              reason: blocked
                ? blockedReason
                : precisionError ||
                  (!catalog.data
                    ? '量化目录尚未读取成功，请先刷新页面。'
                    : chosen?.availability === 'unsupported'
                      ? '当前方法在此环境不可用，请选择本页标记可用的方法。'
                      : '此校准方法尚不可用，请选择已有量化制品或其他可用方法。'),
              next: { to: '/runs', label: '查看模型切换与评测任务' },
            }}
            disabled={
              !deployment ||
              !!blocked ||
              !!precisionError ||
              !catalog.data ||
              chosen?.availability === 'unsupported' ||
              (calibrated && chosen?.calibration_available === false && !artifact)
            }
            onClick={() => void lifecycle('load')}
          >
            应用方案并重载模型
          </Button>
          <Button
            disabledReason={{
              reason: blocked ? blockedReason : '所选模型尚未加载，只有就绪模型可以卸载。',
              next: { to: '/devices', label: '查看所选模型状态' },
            }}
            disabled={!deployment || deployment.state !== 'ready' || !!blocked}
            onClick={() => void lifecycle('unload')}
          >
            卸载所选模型
          </Button>
          <Link to="/models">下载与接入模型</Link>
        </Space>
        {lifecycleRun && (
          <p role="status">
            {lifecycleRun.kind === 'load' ? '加载／校准' : '卸载'}：
            <Status state={lifecycleRun.state} />{' '}
            {busy ? lifecycleRun.id : <Link to={`/runs/${lifecycleRun.id}`}>查看任务记录</Link>}
          </p>
        )}
        {error && <Alert className="section-gap" type="error" showIcon message={error} />}
        {precisionError && (
          <Alert className="section-gap" type="warning" showIcon message={precisionError} />
        )}
        <p className="help-text">
          RTN/GPTQ/AWQ 的紧凑执行要求 FP16；FP32 是独立浮点基线。FP8 以目标硬件能力检查为准。量化 KV
          当前仅接入单请求完整 Attention，不能与 L8 合批组合。
        </p>
      </Panel>
      <Panel
        title="实际生效与存储"
        subtitle="模块绑定和调用凭据不等同于 GPU 指令跟踪；存储字节与分配器峰值分别记录。"
      >
        <QueryState error={capabilityBusy ? null : capability.error} />
        {(busy || running || transitioning || capabilityBusy) && (
          <p role="status">
            任务执行中，精度凭据暂停刷新；设备空闲后自动更新。已有凭据为上次读取的快照。
          </p>
        )}
        {effective ? (
          <>
            <Descriptions
              size="small"
              column={{ xs: 1, md: 3 }}
              items={[
                { key: 'dtype', label: '当前计算精度', children: effective.compute_dtype },
                { key: 'method', label: '当前方法', children: effective.method },
                { key: 'execution', label: '执行路径', children: effective.execution },
                { key: 'modules', label: '量化模块数', children: effective.quantized_module_count },
                {
                  key: 'bytes',
                  label: '量化模块实际存储',
                  children: `${fmt(effective.quantized_storage_bytes / 1024 ** 2)} MiB`,
                },
                {
                  key: 'float',
                  label: '其余浮点参数',
                  children: `${fmt(effective.remaining_float_parameter_bytes / 1024 ** 2)} MiB`,
                },
              ]}
            />
            <details>
              <summary>完整生效凭据与模块布局</summary>
              <JsonView value={effective} />
            </details>
          </>
        ) : (
          <p>加载模型后读取实际能力与存储。{deployment?.last_error}</p>
        )}
        <Button
          size="small"
          disabledReason={{
            reason: capability.isFetching
              ? '正在读取当前模型的实际精度凭据，请等待完成。'
              : blocked
                ? blockedReason
                : '需要先加载模型，才能读取实际生效凭据。',
            next: { to: '/devices', label: '加载或检查模型' },
          }}
          disabled={deployment?.state !== 'ready' || !!blocked || capability.isFetching}
          onClick={() => void capability.refetch()}
        >
          刷新实际凭据
        </Button>
      </Panel>
      <Panel
        title="当前已加载模型：推理与独立质量评测"
        subtitle="下列任务使用实际驻留模型；修改上方草稿需要先应用。内置评测保存逐 Token NLL，自定义评测另存 logits；质量任务不计入性能。"
      >
        <Input.TextArea
          aria-label="L4 推理输入"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={2}
        />
        <Space wrap>
          <label>
            输出 Token{' '}
            <InputNumber
              min={1}
              max={256}
              value={outputTokens}
              onChange={(v) => setOutputTokens(v ?? 16)}
            />
          </label>
          <Button
            disabledReason={{
              reason: blocked
                ? blockedReason
                : '所选模型尚未就绪。请先点击本页“应用方案并重载模型”。',
              next: { to: '/devices', label: '查看模型加载状态' },
            }}
            disabled={deployment?.state !== 'ready' || !!blocked}
            onClick={() => void submit(false)}
          >
            测试当前已加载模型
          </Button>
        </Space>
        <p>
          质量评测数据集{' '}
          <Select
            aria-label="质量评测数据集"
            value={qualityPreset}
            disabled={!!blocked}
            style={{ minWidth: 300 }}
            onChange={setQualityPreset}
            options={[
              ...(qualityBenchmarks.data?.items ?? []).map((b) => ({ value: b.id, label: b.name })),
              { value: 'custom', label: '自定义短文本 · logits / 标准答案' },
            ]}
          />
        </p>
        {qualityPreset !== 'custom' && (
          <>
            <QueryState loading={qualityBenchmarks.isPending} error={qualityBenchmarks.error} />
            <p>
              WikiText-2 英文测试集固定子集，{benchmark?.sample_count ?? 8} 段，每段最多{' '}
              {benchmark?.max_tokens ?? 64} Token。使用当前模型与 KV 逐 Token 计算 NLL 和 Token 加权
              PPL， 不生成回答、不保存全词表 logits。PPL
              越低越好，只有同一模型和分词器的同协议结果可直接比较。
            </p>
            <p className="help-text">
              用于快速检查量化对英文语料预测的影响；中文、数学、代码等能力可使用各自的任务数据集评测。
              上下文在每段重置，首 Token 不计分；没有阈值只报告测量值。
            </p>
            {benchmark && (
              <details>
                <summary>固定样本来源与版本</summary>
                <p>
                  <a href={benchmark.source.url} target="_blank" rel="noreferrer">
                    上游数据
                  </a>{' '}
                  · {benchmark.source.variant}
                </p>
                <p>
                  子集 SHA256：<code>{benchmark.dataset_sha256}</code>
                </p>
                <ol>
                  {benchmark.articles.map((article) => (
                    <li key={article}>{article}</li>
                  ))}
                </ol>
              </details>
            )}
            <Input
              aria-label="预登记 PPL 阈值"
              placeholder={'可选：{"max_mean_nll_increase":0.1}，基线和候选须提前填写相同阈值'}
              value={pplLimits}
              onChange={(e) => setPplLimits(e.target.value)}
            />
          </>
        )}
        <div hidden={qualityPreset !== 'custom'}>
          <p>独立质量文本，每行一个样本，最多 4 条；与校准文本分开。没有阈值时只报告测量结果。</p>
          <Input.TextArea
            aria-label="独立质量文本"
            rows={3}
            value={qualityText}
            onChange={(e) => setQualityText(e.target.value)}
          />
          <details>
            <summary>带标准答案的质量样本（可选）</summary>
            <p>
              填写 JSON 数组后使用这些样本替代上方文本。expected 是原始 prompt
              续写的标准答案，按去除首尾空白后的完全匹配计算准确率。
            </p>
            <Input.TextArea
              aria-label="带标签质量样本 JSON"
              rows={3}
              value={qualityJson}
              onChange={(e) => setQualityJson(e.target.value)}
              placeholder={'[{"id":"arithmetic","text":"1 + 1 =","expected":"2"}]'}
            />
          </details>
          <label>
            质量评测生成 Token 上限{' '}
            <InputNumber
              min={1}
              max={32}
              value={qualityGeneration}
              onChange={(v) => setQualityGeneration(v ?? 4)}
            />
          </label>
          <Input
            aria-label="预登记质量阈值"
            placeholder="可选：预登记 max_logit_abs_error / max_kl / max_nll_increase / max_accuracy_drop 的 JSON"
            value={qualityLimits}
            onChange={(e) => setQualityLimits(e.target.value)}
          />
        </div>
        <p>
          <Button
            disabledReason={{
              reason: blocked
                ? blockedReason
                : deployment?.state !== 'ready'
                  ? '所选模型尚未就绪，请先点击本页“应用方案并重载模型”。'
                  : '质量 benchmark 尚未读取成功，请先选择可用 benchmark 或重试读取。',
              next: {
                to: deployment?.state !== 'ready' ? '/devices' : '/runs',
                label: deployment?.state !== 'ready' ? '查看模型加载状态' : '查看当前评测任务',
              },
            }}
            disabled={
              deployment?.state !== 'ready' ||
              !!blocked ||
              (qualityPreset !== 'custom' && !benchmark)
            }
            onClick={() => void submit(true)}
          >
            运行独立质量评测
          </Button>
        </p>
        {job.data && (
          <>
            <p>
              <Status state={job.data.state} /> ·{' '}
              <Link to={`/runs/${job.data.id}`}>{job.data.id}</Link> · Token/s{' '}
              {fmt(job.data.metrics.output_tokens_per_s)} · PPL{' '}
              {fmt(job.data.metrics.quality_evaluation?.perplexity, 3)}
            </p>
            {job.data.metrics.quality_evaluation?.score_mode === 'corpus_nll' && (
              <p>
                WikiText-2 Mini · 平均 NLL {fmt(job.data.metrics.quality_evaluation.mean_nll, 4)} ·{' '}
                {job.data.metrics.quality_evaluation.scored_tokens ?? '—'} 个计分 Token ·{' '}
                {job.data.metrics.quality_evaluation.document_count ?? '—'}{' '}
                段。质量任务不计入推理吞吐。
              </p>
            )}
            {job.data.metrics.quality_evaluation && <QualityBaseline run={job.data} />}
            {job.data.error && <Alert type="error" message={job.data.error} />}
            <details>
              <summary>任务输出、内存与观测</summary>
              <JsonView value={job.data} />
            </details>
          </>
        )}
      </Panel>
      <Panel
        title="基线与候选质量对照"
        subtitle="填写上面独立质量任务的运行 ID；后端核验模型来源、数据、协议和封存哈希，并检查完全相同的校准／评测文本。"
      >
        <Space wrap>
          <Input
            aria-label="质量基线运行 ID"
            placeholder="基线 run_…"
            value={baseline}
            onChange={(e) => setBaseline(e.target.value)}
            style={{ width: 340, maxWidth: '100%' }}
          />
          <Input
            aria-label="质量候选运行 ID"
            placeholder="候选 run_…，多个逗号分隔"
            value={candidates}
            onChange={(e) => setCandidates(e.target.value)}
            style={{ width: 400, maxWidth: '100%' }}
          />
          <Button
            aria-label="生成质量对照"
            disabledReason={{
              reason: blocked
                ? blockedReason
                : '请先填写基线 run ID 和至少一个候选 run ID；两者都需要完成独立质量评测。',
              next: { to: '/runs', label: '选择质量评测记录' },
            }}
            disabled={!baseline || !candidates || !!blocked}
            loading={busy}
            onClick={async () => {
              setError('');
              setBusy(true);
              try {
                setComparison(
                  await api<QualityComparison>('/precision/comparisons', {
                    baseline_run_id: baseline,
                    candidate_run_ids: candidates
                      .split(',')
                      .map((s) => s.trim())
                      .filter(Boolean),
                  }),
                );
              } catch (e) {
                setError((e as Error).message);
              } finally {
                setBusy(false);
              }
            }}
          >
            生成质量对照
          </Button>
        </Space>
        {comparison?.candidates.some((c) => c.quality.metric === 'token_perplexity') && (
          <Table
            rowKey="run_id"
            dataSource={comparison.candidates}
            pagination={false}
            scroll={{ x: 760 }}
            columns={[
              {
                title: '候选运行',
                dataIndex: 'run_id',
                render: (id: string) => <Link to={`/runs/${id}`}>{id}</Link>,
              },
              { title: '基线 PPL ↓', render: (_, c) => fmt(c.quality.baseline_perplexity, 3) },
              { title: '候选 PPL ↓', render: (_, c) => fmt(c.quality.candidate_perplexity, 3) },
              { title: '平均 NLL 增量 ↓', render: (_, c) => fmt(c.quality.mean_nll_increase, 4) },
              { title: 'PPL 倍率 ↓', render: (_, c) => fmt(c.quality.ppl_ratio, 3) },
              {
                title: '判定',
                render: (_, c) => qualityStatuses[c.quality.status] ?? c.quality.status,
              },
            ]}
          />
        )}
        {comparison != null && <JsonView value={comparison} />}
      </Panel>
      <Panel title="已保存的独立量化制品">
        <QueryState error={artifacts.error} />
        <Table<Artifact>
          rowKey="artifact_id"
          dataSource={artifacts.data?.items}
          scroll={{ x: 900 }}
          columns={[
            {
              title: '制品',
              dataIndex: 'artifact_id',
              render: (id: string) => <code>{id.slice(0, 20)}</code>,
            },
            { title: '方法', dataIndex: 'method' },
            {
              title: '源模型',
              dataIndex: 'source_model',
              render: (s?: string) => s ?? '见源权重身份',
            },
            { title: '计算精度', dataIndex: 'compute_dtype' },
            {
              title: '磁盘 MiB',
              dataIndex: 'disk_bytes',
              render: (n: number) => fmt(n / 1024 ** 2),
            },
            {
              title: '源权重身份',
              dataIndex: 'source_model_hash',
              render: (s: string) => s.slice(0, 16),
            },
          ]}
        />
      </Panel>
    </>
  );
}
