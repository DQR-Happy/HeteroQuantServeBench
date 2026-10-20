import Button from '../components/ActionButton';
import { App, Space, Tabs, Tag, Alert } from 'antd';
import { DownloadOutlined, StopOutlined } from '@ant-design/icons';
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import RunEvidence from '../features/experiments/RunEvidence';
import QualityRun from '../features/experiments/QualityRun';
import { isQualityRun } from '../features/experiments/model';
import { api, base, date, fmt, gib, terminal } from '../api/client';
import {
  Chart,
  JsonView,
  PageHead,
  Panel,
  QueryState,
  Stat,
  Status,
  runKindLabel,
} from '../components';
import { useRun } from '../hooks';
import ObservationPanel from '../features/observability/ObservationPanel';
import QuantizationRun from '../features/research/QuantizationRun';
import LifecycleResult from '../features/kernels/LifecycleResult';
import CompressionJob from '../features/structure/CompressionJob';
import DecodingEvidence from '../features/decoding/DecodingEvidence';
import type { DecodeRun } from '../features/decoding/types';

export { default as Runs } from './RunDirectory';

export function RunDetail() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const query = useRun(id);
  const r = query.data;
  const { message } = App.useApp();
  const itl = r?.metrics.token_itl_ms ?? [];
  const lifecycle = r?.kind === 'load' || r?.kind === 'unload';
  const quality = !!r && isQualityRun(r);
  const decoding = (r as DecodeRun | undefined)?.metrics.decoding_receipt;
  if (r?.kind === 'application')
    return <Navigate replace to={`/application/results?run=${r.id}`} />;
  if (r?.kind === 'system')
    return (
      <Navigate
        replace
        to={`/system/results?deployment=${encodeURIComponent(r.config.deployment.id)}&run=${r.id}`}
      />
    );
  if (r?.kind === 'distributed')
    return (
      <Navigate
        replace
        to={`/parallel/run?deployment=${encodeURIComponent(r.config.deployment.id)}&run=${r.id}`}
      />
    );
  if (r?.kind === 'draft_unload')
    return (
      <>
        <PageHead kicker="L7" title="草稿模型卸载" description={r.id} />
        <Panel title="真实生命周期结果">
          <Status state={r.state} />
          <JsonView value={r} />
          <Link to={`/decoding/draft?deployment=${encodeURIComponent(r.config.deployment.id)}`}>
            查看草稿资源 →
          </Link>
        </Panel>
      </>
    );
  if (r?.kind === 'scheduling') return <Navigate replace to={`/scheduling?experiment=${r.id}`} />;
  if (r?.kind === 'compress')
    return (
      <>
        <PageHead kicker="L5" title="结构压缩任务详情" description={r.id} />
        <CompressionJob run={r} />
        <Button
          disabled={terminal(r.state)}
          disabledReason={{
            reason: '任务已结束；请到模型结构页加载制品或创建新的任务。',
            next: {
              to: `/model-structure/artifacts?deployment=${encodeURIComponent(r.config.deployment.id)}`,
              label: '查看结构制品',
            },
          }}
          onClick={() =>
            void api(`/requests/${r.id}/cancel`, {}).catch((e) => message.error(e.message))
          }
        >
          取消压缩任务
        </Button>
      </>
    );
  return (
    <>
      <PageHead
        kicker="RUN INSPECTOR"
        title={r ? `${quality ? '独立质量评测' : runKindLabel(r.kind)}详情` : '运行详情'}
        description={id ?? ''}
        extra={
          <Space>
            {r && <Status state={r.state} />}
            {r?.config.runtime_plan && (
              <Link
                to={`/model-architecture?deployment=${encodeURIComponent(r.config.deployment.id)}&runtime_run=${r.id}`}
              >
                <Button>计算图与 L3 分析</Button>
              </Link>
            )}
            {r?.kind === 'generate' && !quality && (
              <Link to={`/observations?run=${r.id}`}>
                <Button>机制观测</Button>
              </Link>
            )}
            {r?.kind === 'generate' && !quality && (
              <Link to={`/memory-flow?run=${encodeURIComponent(r.id)}`}>
                <Button>内存结构与数据流</Button>
              </Link>
            )}
            <Button icon={<DownloadOutlined />} href={`${base}/runs/${id}/export`}>
              JSON
            </Button>
            <Button href={`${base}/runs/${id}/export?format=markdown`}>报告</Button>
          </Space>
        }
      />
      <QueryState loading={query.isPending} error={query.error} />
      {r && (
        <>
          {(!lifecycle || r.error || !terminal(r.state)) && (
            <Alert
              className="section-gap"
              type={r.error ? 'error' : 'info'}
              showIcon
              message={
                r.error ??
                (quality
                  ? '独立质量任务：默认展示评分、逐样本结果与协议；实际配置和封存文件见「优化与质量凭据」。此任务不参与推理性能比较。'
                  : '本页展示实际运行记录，性能对照与质量评分可进入对应评测查看。')
              }
              action={
                !terminal(r.state) && (
                  <Button
                    danger
                    icon={<StopOutlined />}
                    onClick={() => {
                      void api(`/requests/${id}/cancel`, {}).catch((e) => message.error(e.message));
                    }}
                  >
                    取消
                  </Button>
                )
              }
            />
          )}
          {r.kind === 'quantize' && <QuantizationRun run={r} />}
          {!lifecycle && !quality && (
            <div className="stats-grid">
              <Stat label="服务端首内容" value={r.metrics.console_first_content_ms} unit="ms" />
              <Stat label="Runtime 首 token" value={r.metrics.runtime_first_token_ms} unit="ms" />
              <Stat label="请求总时间" value={r.metrics.console_e2e_ms} unit="ms" />
              <Stat
                label="生成速度"
                value={r.metrics.output_tokens_per_s}
                unit="Token/s"
                hint="输出 Token 数 ÷ 生成循环时间，含首 Token 等待"
              />
              <Stat
                label="Decode 速度"
                value={r.metrics.decode_tail_tokens_per_s}
                unit="Token/s"
                hint="首 Token 之后；少于 2 个 Token 时无定义"
              />
              <Stat
                label="峰值分配内存"
                value={gib(r.metrics.memory?.peak_allocated_bytes)}
                unit="GiB"
                hint="worker allocator · 非整机内存"
              />
            </div>
          )}
          <Panel>
            <Tabs
              activeKey={
                (lifecycle
                  ? ['config', 'metrics', 'identity', 'output']
                  : [
                      'observation',
                      'config',
                      'metrics',
                      'identity',
                      ...(quality ? ['quality'] : ['output']),
                      'experiment',
                      ...(decoding ? ['decoding'] : []),
                    ]
                ).includes(params.get('tab') ?? '')
                  ? params.get('tab')!
                  : quality
                    ? 'quality'
                    : 'output'
              }
              onChange={(tab) =>
                setParams(
                  (prev) => {
                    const next = new URLSearchParams(prev);
                    next.set('tab', tab);
                    return next;
                  },
                  { replace: true },
                )
              }
              items={[
                ...(decoding
                  ? [
                      {
                        key: 'decoding',
                        label: 'L7 解码与验证',
                        children: <DecodingEvidence receipt={decoding} />,
                      },
                    ]
                  : []),
                ...(quality
                  ? [{ key: 'quality', label: '质量评分与样本', children: <QualityRun run={r} /> }]
                  : []),
                ...(!lifecycle
                  ? [
                      {
                        key: 'experiment',
                        label: '优化与质量凭据',
                        children: <RunEvidence run={r} />,
                      },
                    ]
                  : []),
                ...(!quality
                  ? [
                      {
                        key: 'output',
                        label: lifecycle ? '生命周期结果与观测' : '输出与计时',
                        children: lifecycle ? (
                          <LifecycleResult run={r} />
                        ) : (
                          <>
                            <div className="grid-two">
                              <div className="markdown detail-output">
                                <ReactMarkdown skipHtml>
                                  {r.output || '此任务没有文本输出。'}
                                </ReactMarkdown>
                              </div>
                              <div>
                                <h3>计时边界</h3>
                                <div className="timeline">
                                  <div>
                                    <i />
                                    请求创建 <span>{date(r.created_at)}</span>
                                  </div>
                                  <div>
                                    <i />
                                    排队 <span>{fmt(r.metrics.queue_ms)} ms</span>
                                  </div>
                                  <div>
                                    <i />
                                    Runtime 首 token{' '}
                                    <span>{fmt(r.metrics.runtime_first_token_ms)} ms</span>
                                  </div>
                                  <div>
                                    <i />
                                    服务端首内容{' '}
                                    <span>{fmt(r.metrics.console_first_content_ms)} ms</span>
                                  </div>
                                  <div>
                                    <i />
                                    完成与清理 <span>{r.cleanup}</span>
                                  </div>
                                </div>
                                <Tag>{r.metrics.measurement_profile ?? 'deployment lifecycle'}</Tag>
                              </div>
                            </div>
                            {itl.length > 0 && (
                              <Chart
                                label="真实生成 token 间隔"
                                option={{
                                  tooltip: { trigger: 'axis' },
                                  grid: { left: 50, right: 20, bottom: 35, top: 30 },
                                  xAxis: {
                                    type: 'category',
                                    name: 'step',
                                    data: itl.map((_, i) => i + 2),
                                  },
                                  yAxis: { type: 'value', name: 'ms' },
                                  series: [
                                    {
                                      type: 'line',
                                      data: itl,
                                      lineStyle: { color: '#238f80' },
                                      showSymbol: itl.length < 30,
                                    },
                                  ],
                                }}
                              />
                            )}
                          </>
                        ),
                      },
                    ]
                  : []),
                ...(!lifecycle
                  ? [
                      {
                        key: 'observation',
                        label: '全链路观测',
                        children:
                          quality && !r.metrics.observation ? (
                            <Alert
                              type="info"
                              message="本质量任务未采集性能与机制详录"
                              description="评分数据在「质量评分与样本」，实际配置在「优化与质量凭据」。TTFT、生成吞吐、GPU profiler 和 OS 页级观测需要另行运行对应推理或诊断任务。"
                            />
                          ) : (
                            <ObservationPanel runId={r.id} observation={r.metrics.observation} />
                          ),
                      },
                    ]
                  : []),
                { key: 'config', label: '实际配置', children: <JsonView value={r.config} /> },
                {
                  key: 'metrics',
                  label: '指标与原始结果',
                  children: <JsonView value={{ metrics: r.metrics, result: r.result }} />,
                },
                {
                  key: 'identity',
                  label: '身份与边界',
                  children: (
                    <>
                      <div className="key-values">
                        <span>部署</span>
                        <strong>{r.config.deployment.name}</strong>
                        <span>任务类型</span>
                        <strong>{quality ? '独立质量评测' : runKindLabel(r.kind)}</strong>
                        {!lifecycle && (
                          <>
                            <span>Epoch</span>
                            <code>{r.config.epoch ?? '—'}</code>
                            <span>输入摘要</span>
                            <code>{r.config.input_sha256 ?? '—'}</code>
                          </>
                        )}
                        <span>精度 / 算子</span>
                        <strong>
                          {r.config.deployment.precision} / {r.config.deployment.kernel}
                        </strong>
                        <span>{quality ? '质量计分' : '质量裁决'}</span>
                        <strong>
                          {r.metrics.quality_evaluation
                            ? `${r.metrics.quality_evaluation.status === 'complete' ? '计分完成' : r.metrics.quality_evaluation.status}；需与基线对照才能判定质量达标`
                            : '未执行质量基准'}
                        </strong>
                        {!lifecycle && (
                          <>
                            <span>{quality ? '评分数据' : '输入保留'}</span>
                            <strong>
                              {quality
                                ? '评分协议、固定数据版本与输入 Token 哈希随记录保存'
                                : r.config.messages
                                  ? '已按选择保存'
                                  : '仅摘要；精确复现需再次提供原输入'}
                            </strong>
                          </>
                        )}
                      </div>
                    </>
                  ),
                },
              ]}
            />
          </Panel>
        </>
      )}
    </>
  );
}
