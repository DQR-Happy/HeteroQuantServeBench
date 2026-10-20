import { Alert, Select, Table } from 'antd';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { JsonView, Panel } from '../../components';
import { base, fmt } from '../../api/client';
import type { Receipt } from './types';
const labels: Record<string, string> = {
  workload_ms: '负载窗口 ms',
  startup_ms: '初始预热/加载 ms',
  completed: '完成请求',
  failed: '失败请求',
  backend_invocations: '实际推理调用',
  served_output_tokens: '交付输出 Token',
  computed_output_tokens: '实际计算输出 Token',
  gateway_served_token_s: '服务交付 Token/s（含缓存）',
  request_s: '完成请求/s',
  p50_e2e_ms: 'P50 服务 E2E ms',
  p95_e2e_ms: 'P95 服务 E2E ms',
  p99_e2e_ms: 'P99 服务 E2E ms',
  peak_observed_allocator_bytes: '后端已观测分配器峰值 bytes',
  submitted_input_tokens: '实际提交模型的输入 Token',
  logical_input_tokens_before: '逻辑输入 Token / 处理前',
  logical_input_tokens_after: '逻辑输入 Token / 处理后',
  prefix_reused_tokens: '后端实际复用 KV 前缀 Token',
  deadline_met: '满足请求时限',
};
export default function ApplicationEvidence({ receipt: r }: { receipt: Receipt }) {
  const [index, setIndex] = useState(0);
  const v = r.variants[index];
  const a = r.variants[0],
    b = r.variants[1];
  return (
    <>
      <Panel title="服务与应用对照结果">
        <p>
          执行：{r.status} · 清理：{r.cleanup} · 整体实验 {fmt(r.job_total_ms)}{' '}
          ms（包含副本加载和回收）
        </p>
        <Alert
          showIcon
          type={r.status === 'applied' && r.comparison?.all_tokens_exact ? 'success' : 'warning'}
          message={
            r.status === 'applied' && r.comparison?.all_tokens_exact
              ? '固定负载逐 Token 输出一致'
              : '输出发生变化或实验未完整完成'
          }
          description="裁剪提示词、减少输出或更换模型可能改变质量。本页展示真实输出差异；任务准确率仍需独立评测。含缓存的服务交付速度不是 GPU 推理速度。"
        />
        <Table
          size="small"
          rowKey="key"
          pagination={false}
          scroll={{ x: 720 }}
          dataSource={Object.keys(a?.metrics ?? {}).map((key) => ({
            key,
            a: a?.metrics[key],
            b: b?.metrics[key],
          }))}
          columns={[
            { title: '指标', render: (_, x) => labels[x.key] ?? x.key },
            { title: '基线 A', dataIndex: 'a', render: (n) => fmt(n, 2) },
            { title: '候选 B', dataIndex: 'b', render: (n) => fmt(n, 2) },
            {
              title: '候选 − 基线',
              render: (_, x) => (x.a != null && x.b != null ? fmt(x.b - x.a, 2) : '—'),
            },
          ]}
        />
        <p>{r.timing_scope}</p>
        <a href={`${base}/runs/${r.run_id}/application`} target="_blank" rel="noreferrer">
          查看封存核验后的完整 JSON →
        </a>
      </Panel>
      <Panel title="逐请求路由、缓存与真实输出">
        <Select
          aria-label="L10 查看方案"
          style={{ width: 260 }}
          value={v ? index : undefined}
          onChange={setIndex}
          options={r.variants.map((x, i) => ({
            value: i,
            label: x.name === 'baseline' ? '基线 A' : '候选 B',
          }))}
        />
        {v && (
          <>
            <p>
              结果缓存命中 {v.cache.hits ?? 0} 次；CPU 缓存 {fmt((v.cache.bytes ?? 0) / 1024, 2)}{' '}
              KiB；淘汰 {v.cache.evictions ?? 0} 次。方案总时间 {fmt(v.variant_total_ms)}{' '}
              ms（包含回收）。
            </p>
            <Table
              size="small"
              pagination={false}
              rowKey="id"
              scroll={{ x: 1120 }}
              dataSource={v.requests}
              columns={[
                { title: '请求 / 租户', render: (_, x) => `${x.id} / ${x.tenant}` },
                { title: '目标', dataIndex: 'target' },
                { title: '缓存', dataIndex: 'cache' },
                { title: '调用模型', render: (_, x) => (x.backend_invoked ? '是' : '否') },
                {
                  title: '输入 Token 前 → 后',
                  render: (_, x) =>
                    x.input
                      ? `${x.input.input_tokens_before} → ${x.input.input_tokens_after}`
                      : '—',
                },
                { title: '输出 Token', dataIndex: 'output_tokens' },
                { title: '排队 ms', dataIndex: 'queue_ms', render: (n) => fmt(n) },
                { title: '服务 E2E ms', dataIndex: 'gateway_e2e_ms', render: (n) => fmt(n) },
                {
                  title: '后端生成 Token/s',
                  render: (_, x) =>
                    x.backend_invoked ? fmt(x.backend_metrics?.output_tokens_per_s, 2) : '未调用',
                },
                { title: '结果', render: (_, x) => x.error ?? x.status },
              ]}
            />
            {v.requests.map((x) => (
              <details key={x.id}>
                <summary>
                  {x.id} · {x.target ?? '未路由'} · {x.cache} · {x.backend_run_id ?? '未提交后端'}
                </summary>
                <pre style={{ whiteSpace: 'pre-wrap' }}>{x.output ?? x.error}</pre>
                {x.backend_run_id &&
                (r.plan.targets.find((t) => t.id === x.target)?.machine_id ?? 'local') ===
                  'local' ? (
                  <Link to={`/runs/${x.backend_run_id}`}>查看实际后端推理记录 →</Link>
                ) : (
                  <Link to="/machines">切换到对应节点后查看子记录 →</Link>
                )}
                <JsonView value={x} />
              </details>
            ))}
          </>
        )}
      </Panel>
      <Panel title="副本与控制事件">
        <Table
          rowKey={(_, i) => String(i)}
          size="small"
          pagination={{ pageSize: 10 }}
          dataSource={r.events.filter((e) => e.kind !== 'request.finished')}
          scroll={{ x: 650 }}
          columns={[
            { title: '时间 ms', dataIndex: 'offset_ms', render: (n) => fmt(n) },
            { title: '事件', dataIndex: 'kind' },
            { title: '目标', dataIndex: 'target' },
            { title: '原因', dataIndex: 'reason' },
            { title: '子运行', dataIndex: 'child_run_id' },
          ]}
        />
        <details>
          <summary>输入身份与输出对齐、冻结计划和完整事件</summary>
          <JsonView value={r} />
        </details>
      </Panel>
    </>
  );
}
