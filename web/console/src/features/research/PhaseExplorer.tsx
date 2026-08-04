import { useState } from 'react';
import { Alert, Empty, Select, Table, Tabs, Tag } from 'antd';
import { fmt } from '../../api/client';
import { JsonView, Panel, Stat } from '../../components';
import ArtifactLinks from './ArtifactLinks';
import type { HistoricalPhase, Research } from './types';

export default function PhaseExplorer({ profiling }: { profiling: Research['profiling'] }) {
  const [selected, setSelected] = useState('');
  const [kernelId, setKernelId] = useState('');
  const phase = profiling.phases.find((row) => row.id === selected) ?? profiling.phases[0];
  const kernel = profiling.kernels.find((row) => row.id === kernelId) ?? profiling.kernels[0];
  return (
    <>
      <Panel
        title="阶段 → 框架算子 → GPU kernel"
        subtitle={`${profiling.source} · 历史采集，不能归因到当前交互请求`}
      >
        {profiling.limitations.length > 0 && (
          <Alert
            className="section-gap"
            type="info"
            showIcon
            message="观测边界"
            description={profiling.limitations.join('；')}
          />
        )}
        {phase ? (
          <>
            <label className="field-label" htmlFor="research-phase">
              选择样本与阶段
            </label>
            <Select
              id="research-phase"
              className="wide-select"
              value={phase.id}
              onChange={setSelected}
              options={profiling.phases.map((row) => ({
                label: `${row.sample} · ${row.phase}`,
                value: row.id,
              }))}
            />
            <div className="stats-grid section-gap">
              <Stat
                label="阶段 span"
                value={phase.span_ms}
                unit="ms"
                hint="时间线范围；请查看原始采集口径"
              />
              <Stat
                label="累计 kernel 工作时间"
                value={phase.kernel_work_ms}
                unit="ms"
                hint="重叠执行时不可视为关键路径耗时"
              />
              <Stat
                label="Kernel 次数"
                value={phase.kernel_count}
                hint="已归属到该阶段的采集事件"
              />
              <Stat
                label="GPU 空闲比例"
                value={phase.idle_ratio == null ? null : phase.idle_ratio * 100}
                unit="%"
                hint="该次采集窗口中的观测结果"
              />
            </div>
            <Table<HistoricalPhase['hotspots'][number]>
              size="small"
              rowKey={(row) => row.name}
              dataSource={phase.hotspots}
              pagination={{ pageSize: 8 }}
              scroll={{ x: 880 }}
              expandable={{
                expandedRowRender: (row) => (
                  <div className="grid-two">
                    <div>
                      <h4>框架关联</h4>
                      {row.ops.map((op) => (
                        <Tag key={op}>{op}</Tag>
                      ))}
                      <p className="small">{row.dims.join(' · ')}</p>
                    </div>
                    <JsonView
                      value={{ streams: row.streams, grids: row.grids, blocks: row.blocks }}
                    />
                  </div>
                ),
              }}
              columns={[
                {
                  title: '实际 kernel',
                  dataIndex: 'name',
                  width: 360,
                  render: (value: string) => <code className="kernel-name">{value}</code>,
                },
                { title: '次数', dataIndex: 'count', sorter: (a, b) => a.count - b.count },
                {
                  title: '累计 / ms',
                  dataIndex: 'total_ms',
                  render: (value: number) => fmt(value, 3),
                  sorter: (a, b) => a.total_ms - b.total_ms,
                  defaultSortOrder: 'descend',
                },
                {
                  title: '平均 / μs',
                  dataIndex: 'mean_us',
                  render: (value: number) => fmt(value, 2),
                },
                {
                  title: '工作时间占比',
                  dataIndex: 'time_share',
                  render: (value: number | null) => (value == null ? '—' : `${fmt(value * 100)}%`),
                },
              ]}
            />
          </>
        ) : (
          <Empty description="没有可读取的阶段归因证据" />
        )}
      </Panel>
      <Panel
        title="热点硬件计数器"
        subtitle="NCU 捕获与前面的系统时间线属于不同采集；保留 mode、shape 和原始单位"
      >
        <Alert
          type="warning"
          showIcon
          className="section-gap"
          message="硬件计数器可能来自重放；其执行时间不能替代未插桩基准。"
        />
        {kernel ? (
          <>
            <label className="field-label" htmlFor="research-kernel">
              选择内核采集
            </label>
            <Select
              id="research-kernel"
              className="wide-select"
              value={kernel.id}
              onChange={setKernelId}
              options={profiling.kernels.map((row) => ({
                label: `${row.sample} · ${row.phase} · ${row.role} · ${row.mode}`,
                value: row.id,
              }))}
            />
            <Tabs
              className="section-gap"
              items={[
                {
                  key: 'metrics',
                  label: '计数器',
                  children: kernel.observations.map((observation, index) => (
                    <div key={index} className="counter-capture">
                      <code className="kernel-name">{observation.name}</code>
                      <Table<{ key: string; value: number | null }>
                        rowKey="key"
                        size="small"
                        pagination={{ pageSize: 12, hideOnSinglePage: true }}
                        dataSource={Object.entries(observation.metrics).map(([key, value]) => ({
                          key,
                          value,
                        }))}
                        columns={[
                          {
                            title: '指标（原始名称包含单位）',
                            dataIndex: 'key',
                            render: (value: string) => <code className="kernel-name">{value}</code>,
                          },
                          {
                            title: '观测值',
                            dataIndex: 'value',
                            width: 160,
                            render: (value: number | null) => fmt(value, 6),
                          },
                        ]}
                      />
                      <details>
                        <summary>Grid / block / stall 证据</summary>
                        <JsonView
                          value={{
                            grid: observation.grid,
                            block: observation.block,
                            stalls: observation.stalls,
                          }}
                        />
                      </details>
                    </div>
                  )),
                },
                {
                  key: 'roofline',
                  label: 'Roofline 与输入',
                  children: (
                    <JsonView
                      value={{
                        kernel_regex: kernel.kernel_regex,
                        shape: kernel.shape,
                        roofline: kernel.roofline,
                      }}
                    />
                  ),
                },
                {
                  key: 'source',
                  label: '原始证据',
                  children: <ArtifactLinks artifacts={kernel.artifacts} />,
                },
              ]}
            />
          </>
        ) : (
          <Empty description="没有可读取的硬件计数器证据" />
        )}
      </Panel>
      <Panel title="采集扰动与原件">
        <details>
          <summary>查看已有扰动测量</summary>
          <JsonView value={profiling.perturbation} />
        </details>
        <ArtifactLinks artifacts={profiling.artifacts} />
      </Panel>
    </>
  );
}
