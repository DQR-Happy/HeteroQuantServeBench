import { App, Button, Space, Tabs, Tag, Alert } from 'antd';
import { DownloadOutlined, StopOutlined } from '@ant-design/icons';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import RunEvidence from '../features/experiments/RunEvidence';
import { api, base, date, fmt, gib, terminal } from '../api/client';
import { Chart, JsonView, PageHead, Panel, QueryState, Stat, Status } from '../components';
import { useRun } from '../hooks';
import ObservationPanel from '../features/observability/ObservationPanel';
import QuantizationRun from '../features/research/QuantizationRun';

export { default as Runs } from './RunDirectory';

export function RunDetail() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const query = useRun(id);
  const r = query.data;
  const { message } = App.useApp();
  const itl = r?.metrics.token_itl_ms ?? [];
  return (
    <>
      <PageHead
        kicker="RUN INSPECTOR"
        title="运行详情"
        description={id ?? ''}
        extra={
          <Space>
            {r && <Status state={r.state} />}
            {r?.kind === 'generate' && (
              <Link to={`/observations?run=${r.id}`}>
                <Button>机制观测</Button>
              </Link>
            )}
            {r?.kind === 'generate' && (
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
          <Alert
            className="section-gap"
            type={r.error ? 'error' : 'info'}
            showIcon
            message={
              r.error ??
              (r.config.experiment_role === 'quality'
                ? '限定质量任务：请在「优化与质量凭据」查看评分协议与原件；此任务不参与推理性能比较。'
                : '这是实际运行记录；单次交互不能形成正式加速结论，质量以独立评测为准。')
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
          {r.kind === 'quantize' && <QuantizationRun run={r} />}
          <div className="stats-grid">
            <Stat label="服务端首内容" value={r.metrics.console_first_content_ms} unit="ms" />
            <Stat label="Runtime 首 token" value={r.metrics.runtime_first_token_ms} unit="ms" />
            <Stat label="请求总时间" value={r.metrics.console_e2e_ms} unit="ms" />
            <Stat
              label="峰值分配内存"
              value={gib(r.metrics.memory?.peak_allocated_bytes)}
              unit="GiB"
              hint="worker allocator · 非整机内存"
            />
          </div>
          <Panel>
            <Tabs
              activeKey={
                ['observation', 'config', 'metrics', 'identity', 'output', 'experiment'].includes(
                  params.get('tab') ?? '',
                )
                  ? params.get('tab')!
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
                { key: 'experiment', label: '优化与质量凭据', children: <RunEvidence run={r} /> },
                {
                  key: 'output',
                  label: '输出与计时',
                  children: (
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
                              服务端首内容 <span>{fmt(r.metrics.console_first_content_ms)} ms</span>
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
                {
                  key: 'observation',
                  label: '全链路观测',
                  children: <ObservationPanel runId={r.id} observation={r.metrics.observation} />,
                },
                { key: 'config', label: '实际配置', children: <JsonView value={r.config} /> },
                {
                  key: 'metrics',
                  label: '指标与原始结果',
                  children: <JsonView value={r.metrics} />,
                },
                {
                  key: 'identity',
                  label: '身份与边界',
                  children: (
                    <>
                      <div className="key-values">
                        <span>部署</span>
                        <strong>{r.config.deployment.name}</strong>
                        <span>Epoch</span>
                        <code>{r.config.epoch ?? '—'}</code>
                        <span>输入摘要</span>
                        <code>{r.config.input_sha256 ?? '—'}</code>
                        <span>精度 / 算子</span>
                        <strong>
                          {r.config.deployment.precision} / {r.config.deployment.kernel}
                        </strong>
                        <span>质量裁决</span>
                        <strong>
                          {r.metrics.quality_evaluation
                            ? `限定评测 ${r.metrics.quality_evaluation.status}；非通用基准`
                            : '未执行质量基准'}
                        </strong>
                        <span>输入保留</span>
                        <strong>
                          {r.config.messages ? '已按选择保存' : '仅摘要；精确复现需再次提供原输入'}
                        </strong>
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
