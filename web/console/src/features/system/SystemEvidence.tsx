import { Alert, Select, Table } from 'antd';
import { useState } from 'react';
import { Chart, JsonView, Panel } from '../../components';
import { base, fmt } from '../../api/client';
import type { SystemReceipt } from './types';

const names: Record<string, string> = {
  token_s: '生成 Token/s',
  first_token_ms: '平均首 Token ms',
  peak_allocated_bytes: '分配器峰值 bytes',
  rss_bytes: '进程 RSS bytes',
  energy_j: 'GPU 板卡能量 J（计数器）',
};
function cpuLabel(ids: number[]) {
  const ranges: string[] = [];
  for (let i = 0; i < ids.length; i++) {
    const start = ids[i];
    while (i + 1 < ids.length && ids[i + 1] === ids[i] + 1) i++;
    ranges.push(start === ids[i] ? `${start}` : `${start}–${ids[i]}`);
  }
  return `${ranges.join(', ')}（${ids.length} 个逻辑 CPU）`;
}
export default function SystemEvidence({ receipt: r }: { receipt: SystemReceipt }) {
  const [trial, setTrial] = useState(0);
  const t = r.trials[trial];
  const complete =
    r.status === 'applied' && r.cleanup === 'succeeded' && r.comparison.complete_pair !== false;
  return (
    <>
      <Panel title="配置对照结果">
        <p>
          执行：{r.status} · 进程清理 / 功耗恢复：{r.cleanup}
        </p>
        <a href={`${base}/runs/${r.run_id}/system`} target="_blank" rel="noreferrer">
          查看经过封存核验的 JSON →
        </a>
        <Alert
          type={complete && r.comparison.greedy_tokens_exact ? 'success' : 'warning'}
          showIcon
          message={
            complete && r.comparison.greedy_tokens_exact
              ? '固定输入的逐 Token 输出一致'
              : '输出未全部一致或实验不完整，请检查每轮输出'
          }
          description="这里只验证固定输入的贪心输出。失败、取消或身份核验失败的任务仅保留已有测量，不能形成完整对照结论。模型任务准确率仍需独立质量评测；性能中位数不代表通用加速结论。"
        />
        <Table
          size="small"
          pagination={false}
          rowKey="key"
          scroll={{ x: 650 }}
          dataSource={Object.entries(r.comparison.metrics).map(([key, v]) => ({ key, ...v }))}
          columns={[
            { title: '指标', render: (_, x) => names[x.key] ?? x.key },
            { title: '基线', dataIndex: 'baseline', render: (n) => fmt(n, 3) },
            { title: '候选', dataIndex: 'candidate', render: (n) => fmt(n, 3) },
            { title: '候选 − 基线', dataIndex: 'delta', render: (n) => fmt(n, 3) },
            { title: '变化 %', dataIndex: 'delta_percent', render: (n) => fmt(n, 2) },
          ]}
        />
        <p>
          生成 Token/s = 实际输出 Token 数 / 生成循环时间；包括首 Token。能量来自 NVML
          板卡计数器；不支持时显示 —，采样积分估算另存原始凭据。
        </p>
      </Panel>
      <Panel title="每轮配置回读与输出">
        <Table
          size="small"
          rowKey="trial"
          pagination={false}
          scroll={{ x: 820 }}
          dataSource={r.trials}
          columns={[
            { title: '轮次', render: (_, x) => `${x.trial + 1} · ${x.label}` },
            {
              title: '实际 CPU',
              width: 240,
              render: (_, x) => (
                <span title={x.effective.cpu_ids.join(', ')}>{cpuLabel(x.effective.cpu_ids)}</span>
              ),
            },
            {
              title: '算子内 / 间线程',
              render: (_, x) => `${x.effective.intra_threads} / ${x.effective.interop_threads}`,
            },
            { title: '功耗上限 W', render: (_, x) => fmt(x.effective.power_limit_w) },
            { title: 'Token/s', render: (_, x) => fmt(x.metrics.token_s, 2) },
            {
              title: '缺页 minor / major',
              render: (_, x) => `${x.metrics.minor_faults_delta} / ${x.metrics.major_faults_delta}`,
            },
          ]}
        />
        <Select
          aria-label="L1 观测轮次"
          style={{ width: 240 }}
          value={t ? trial : undefined}
          options={r.trials.map((x, i) => ({ value: i, label: `轮次 ${i + 1} · ${x.label}` }))}
          onChange={setTrial}
        />
        {t && (
          <>
            <Chart
              label="L1 GPU 频率、功耗与温度时间线"
              option={{
                tooltip: { trigger: 'axis' },
                legend: { data: ['SM MHz', '功耗 W', '温度 °C'] },
                xAxis: { type: 'value', name: '秒' },
                yAxis: [
                  { type: 'value', name: 'SM MHz' },
                  { type: 'value', name: 'W / °C' },
                ],
                series: [
                  ['sm_clock_mhz', 'SM MHz', 0],
                  ['power_w', '功耗 W', 1],
                  ['temperature_c', '温度 °C', 1],
                ].map(([key, name, axis]) => ({
                  name,
                  type: 'line' as const,
                  yAxisIndex: Number(axis),
                  connectNulls: false,
                  data: t.samples.map((s) => [
                    s.time_s - t.samples[0].time_s,
                    s[key as keyof typeof s],
                  ]),
                })),
              }}
            />
            <p>
              250 ms 周期的设备级采样，短任务可能仅有边界样本；不等同于每个 kernel 的时钟或功耗。
            </p>
            {t.requests.map((x) => (
              <div key={x.index}>
                <strong>
                  输入 {x.index + 1} · {x.token_ids.length} Token · {fmt(x.token_s, 2)} Token/s
                </strong>
                <pre style={{ whiteSpace: 'pre-wrap' }}>{x.output}</pre>
              </div>
            ))}
            <details>
              <summary>本轮线程亲和性、NUMA 页、页错误、采样及完整凭据</summary>
              <JsonView value={t} />
            </details>
          </>
        )}
        <details>
          <summary>冻结计划、硬件身份与完整恢复凭据</summary>
          <JsonView value={r} />
        </details>
      </Panel>
    </>
  );
}
