import { Empty, Table } from 'antd';
import { Chart } from '../../components';
import { fmt } from '../../api/client';
import type { PhaseSpan, TokenSpan } from './types';

export default function PhaseTimeline({
  phases,
  tokens,
}: {
  phases: PhaseSpan[];
  tokens: TokenSpan[];
}) {
  return (
    <>
      <h3>阶段瀑布 · 工作进程主机时间</h3>
      <p className="small muted">
        原点为工作进程接收请求。阶段可能嵌套或重叠，不将所有耗时相加；异步 GPU 执行以 profiler
        证据为准。
      </p>
      {phases.length ? (
        <Chart
          label="真实请求阶段时间线"
          height={Math.max(220, Math.min(600, phases.length * 32 + 80))}
          option={{
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { left: 12, right: 30, top: 20, bottom: 30, containLabel: true },
            xAxis: { type: 'value', name: 'ms' },
            yAxis: {
              type: 'category',
              inverse: true,
              data: phases.map((phase) => phase.name),
              axisLabel: { width: 160, overflow: 'truncate' },
            },
            series: [
              {
                name: '起始偏移',
                type: 'bar',
                stack: 'phase',
                silent: true,
                itemStyle: { color: 'transparent' },
                emphasis: { disabled: true },
                data: phases.map((phase) => phase.start_ms),
              },
              {
                name: '阶段持续时间',
                type: 'bar',
                stack: 'phase',
                itemStyle: { color: '#258c7e', borderRadius: [0, 3, 3, 0] },
                data: phases.map((phase) => phase.duration_ms),
                barMaxWidth: 18,
              },
            ],
          }}
        />
      ) : (
        <Empty description="当前观测没有阶段事件" />
      )}
      {tokens.length > 0 && (
        <>
          <h3>逐 token 成本</h3>
          <p className="small muted">
            模型调用、token 选择与反分词的主机侧时间；与浏览器收到 SSE 内容的间隔不同。未标出的 IPC
            / 导出等成本不自动归入模型计算。
          </p>
          <Chart
            label="逐token模型调用、选择与反分词成本"
            option={{
              tooltip: { trigger: 'axis' },
              legend: { top: 0 },
              grid: { left: 50, right: 20, bottom: 30, top: 45 },
              xAxis: { type: 'category', data: tokens.map((token) => token.index), name: 'token' },
              yAxis: { type: 'value', name: 'ms' },
              series: [
                {
                  name: '模型调用（host）',
                  type: 'bar',
                  stack: 'token',
                  data: tokens.map((token) => token.model_host_ms),
                  itemStyle: { color: '#238f80' },
                },
                {
                  name: 'token 选择',
                  type: 'bar',
                  stack: 'token',
                  data: tokens.map((token) => token.selection_ms),
                  itemStyle: { color: '#e4b563' },
                },
                {
                  name: '反分词',
                  type: 'bar',
                  stack: 'token',
                  data: tokens.map((token) => token.detokenize_ms),
                  itemStyle: { color: '#6b92c2' },
                },
                {
                  name: '整个 step',
                  type: 'line',
                  data: tokens.map((token) => token.duration_ms),
                  showSymbol: tokens.length < 30,
                  lineStyle: { color: '#836594' },
                },
              ],
            }}
          />
          <details>
            <summary>逐 token 原始时间表</summary>
            <Table<TokenSpan>
              size="small"
              rowKey="index"
              dataSource={tokens}
              pagination={{ pageSize: 12 }}
              scroll={{ x: 660 }}
              columns={[
                { title: 'Token', dataIndex: 'index' },
                { title: '阶段', dataIndex: 'phase' },
                {
                  title: '开始 / ms',
                  dataIndex: 'start_ms',
                  render: (value: number) => fmt(value, 3),
                },
                {
                  title: '总计 / ms',
                  dataIndex: 'duration_ms',
                  render: (value: number) => fmt(value, 3),
                },
                {
                  title: '模型 host / ms',
                  dataIndex: 'model_host_ms',
                  render: (value: number) => fmt(value, 3),
                },
                {
                  title: '选择 / ms',
                  dataIndex: 'selection_ms',
                  render: (value: number) => fmt(value, 3),
                },
                {
                  title: '反分词 / ms',
                  dataIndex: 'detokenize_ms',
                  render: (value: number) => fmt(value, 3),
                },
              ]}
            />
          </details>
        </>
      )}
    </>
  );
}
