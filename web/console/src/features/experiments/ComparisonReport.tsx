import { useState } from 'react';
import { Alert, Collapse, Input, Table, Tabs, Tag } from 'antd';
import { fmt } from '../../api/client';
import { JsonView } from '../../components';
import type { Report } from './types';

export default function ComparisonReport({ report }: { report: Report }) {
  const [mechanismSearch, setMechanismSearch] = useState('');
  const positions = report.quality.flatMap((item) => item.positions ?? []);
  const mechanisms = report.mechanisms.flatMap((item, pair) =>
    item.differences.map((delta) => ({ ...delta, pair, interpretation: item.interpretation })),
  );
  return (
    <>
      <Alert
        type={report.decision === 'hold' ? 'warning' : 'info'}
        showIcon
        message={`采用判定：${report.decision}`}
        description={`证据完整度 ${report.completeness} · 可比性 ${report.comparable ? '满足注册控制条件' : '条件未满足'}。判定来自后端；缺失或未评测不会显示为通过。`}
      />
      {[...report.reasons, ...report.adoption_reasons].length > 0 && (
        <ul className="experiment-reasons">
          {[...report.reasons, ...report.adoption_reasons].map((reason) => (
            <li key={reason}>
              <code>{reason}</code>
            </li>
          ))}
        </ul>
      )}
      <Tabs
        items={[
          {
            key: 'quality',
            label: '质量对照',
            children: (
              <>
                <Tag>数值合同 {report.quality_status}</Tag>
                <span className="help-text">
                  只覆盖注册数据与阈值；数值一致不能证明任务准确率足够。
                </span>
                {report.quality.map((item, i) => (
                  <div key={i} className="help-text">
                    PPL 基线 {fmt(item.baseline_perplexity, 6)} / 候选{' '}
                    {fmt(item.candidate_perplexity, 6)} · exact-match 基线{' '}
                    {fmt(item.baseline_task_accuracy, 6)} / 候选{' '}
                    {fmt(item.candidate_task_accuracy, 6)} {item.reason}
                  </div>
                ))}
                <Table
                  size="small"
                  rowKey={(_, index) => String(index)}
                  dataSource={positions}
                  scroll={{ x: 650 }}
                  pagination={{ pageSize: 10 }}
                  columns={[
                    { title: '样本', dataIndex: 'sample_id' },
                    { title: '位置', dataIndex: 'position' },
                    {
                      title: 'Logit 最大误差',
                      dataIndex: 'logit_max_abs_error',
                      render: (v: number) => fmt(v, 8),
                    },
                    { title: 'KL', dataIndex: 'kl_fp16_quant', render: (v: number) => fmt(v, 8) },
                    {
                      title: 'NLL 增量',
                      dataIndex: 'nll_increase',
                      render: (v: number) => fmt(v, 8),
                    },
                    {
                      title: '数值门',
                      dataIndex: 'passed',
                      render: (v: boolean) => (v ? '满足' : '未满足'),
                    },
                  ]}
                />
              </>
            ),
          },
          {
            key: 'performance',
            label: '性能与观测成本',
            children: (
              <>
                <Tag>{report.performance.status}</Tag>
                <div className="key-values">
                  <span>有效配对</span>
                  <strong>{report.performance.pairs}</strong>
                  <span>独立进程</span>
                  <strong>
                    {report.performance.independent_processes ? '是' : '否 / 未形成证据'}
                  </strong>
                  <span>生成 E2E 中位变化</span>
                  <strong>
                    {report.performance.median_relative_e2e == null
                      ? '—'
                      : `${fmt(report.performance.median_relative_e2e * 100, 4)}%`}
                  </strong>
                  <span>95% 区间</span>
                  <strong>
                    {report.performance.interval_95
                      ? report.performance.interval_95
                          .map((v) => `${fmt(v * 100, 4)}%`)
                          .join(' ～ ')
                      : '—'}
                  </strong>
                </div>
                <p className="help-text">
                  负值表示生成循环更快；不含加载、质量评测和归档。API 共用驻留
                  worker，不能满足独立进程性能门；正式实验使用远端 CLI。
                </p>
                <Collapse
                  items={[
                    {
                      key: 'cost',
                      label: '诊断采集成本（描述性）',
                      children: <JsonView value={report.observer_cost} />,
                    },
                  ]}
                />
              </>
            ),
          },
          {
            key: 'mechanism',
            label: '机制变化',
            children: (
              <>
                <Tag>机制预期 {report.mechanism_gate}</Tag>
                <Input.Search
                  aria-label="筛选机制对照指标"
                  placeholder="指标、阶段或口径"
                  allowClear
                  value={mechanismSearch}
                  onChange={(event) => setMechanismSearch(event.target.value)}
                />
                <p className="help-text">
                  allocator、进程与整机内存口径不能相加；CPU 页不是 GPU
                  页表。上下文分叉的比较仅作描述，来源序号对应两侧原始观测记录。
                </p>
                <Table
                  size="small"
                  rowKey={(_, index) => String(index)}
                  dataSource={mechanisms.filter((item) =>
                    `${item.metric_id} ${item.label ?? ''} ${item.scope}`
                      .toLowerCase()
                      .includes(mechanismSearch.trim().toLowerCase()),
                  )}
                  scroll={{ x: 1100 }}
                  pagination={{ pageSize: 10 }}
                  columns={[
                    { title: '指标', dataIndex: 'metric_id' },
                    { title: '阶段', dataIndex: 'label' },
                    { title: '候选 − 基线', dataIndex: 'delta', render: (v: number) => fmt(v, 6) },
                    { title: '单位', dataIndex: 'unit' },
                    { title: '口径', dataIndex: 'scope' },
                    { title: '解释', dataIndex: 'interpretation' },
                    {
                      title: '原始 seq（基线 / 候选）',
                      render: (_, row) =>
                        `${row.baseline_source_record_seqs.join(',')} / ${row.candidate_source_record_seqs.join(',')}`,
                    },
                  ]}
                />
              </>
            ),
          },
          { key: 'raw', label: '完整报告与来源', children: <JsonView value={report} /> },
        ]}
      />
      {(report.missing_pairs.length > 0 || report.failed_pairs.length > 0) && (
        <Collapse
          items={[
            {
              key: 'incomplete',
              label: `缺失 ${report.missing_pairs.length} 对 · 非完成 ${report.failed_pairs.length} 对`,
              children: (
                <JsonView
                  value={{ missing: report.missing_pairs, non_completed: report.failed_pairs }}
                />
              ),
            },
          ]}
        />
      )}
    </>
  );
}
