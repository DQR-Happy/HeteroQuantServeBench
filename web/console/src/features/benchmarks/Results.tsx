import { useState } from 'react';
import { Alert, Descriptions, Select, Space, Table, Tabs, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { base, fmt, gib } from '../../api/client';
import type { Benchmark, ResultRow } from './types';

const percent = (value: number | null) => (value == null ? '—' : `${fmt(value * 100, 1)}%`);
const reasons: Record<string, string> = {
  TIMING_UNAVAILABLE: '计时数据缺失',
  TOKEN_COUNTS_UNAVAILABLE: 'Token 数量证据缺失',
  INCOMPLETE_MEASUREMENTS: '测量未完整',
  OUTPUT_LENGTH_CHANGED: '实际输出 Token 数不同',
  INPUT_TOKENS_DIFFER: '输入 Token 不一致',
  MODEL_IDENTITY_DIFFERS: '模型身份不一致',
  SOURCE_DIGEST_DIFFERS: '源码身份不一致',
  EXECUTION_NOT_CONFIRMED: '实际执行未确认',
  BASELINE_INCOMPLETE: '基线未完整',
  TOKENIZER_IDENTITY_DIFFERS: '分词器身份不一致',
  BASELINE_EXECUTION_NOT_CONFIRMED: '基线执行未确认',
};
export default function Results({ benchmark }: { benchmark: Benchmark }) {
  const [selected, setSelected] = useState('');
  const activeCase =
    benchmark.plan.cases.find((item) => item.id === selected) ?? benchmark.plan.cases[0];
  const [repeat, setRepeat] = useState(0);
  const schemeName = (id: string) =>
    benchmark.plan.schemes.find((item) => item.id === id)?.name ?? id;
  const rows = benchmark.report.rows.filter((row) => row.case_id === activeCase.id);
  return (
    <>
      <Descriptions
        size="small"
        column={{ xs: 1, md: 3 }}
        items={[
          { key: 'suite', label: '固定输入版本', children: benchmark.plan.suite_version },
          {
            key: 'tokens',
            label: '输出预算',
            children: `${benchmark.plan.output_tokens} tokens · greedy`,
          },
          {
            key: 'repeats',
            label: '测量重复',
            children: `${benchmark.plan.repeats} 次 / 用例 / 方案`,
          },
        ]}
      />
      <div className="help-text mono">计划 SHA-256：{benchmark.plan_sha256}</div>
      <Tabs
        activeKey={activeCase.id}
        onChange={setSelected}
        items={benchmark.plan.cases.map((item) => ({ key: item.id, label: item.name }))}
      />
      <Table<ResultRow>
        size="small"
        rowKey="scheme_id"
        pagination={false}
        scroll={{ x: 1250 }}
        dataSource={rows}
        columns={[
          {
            title: '推理方案',
            dataIndex: 'scheme_id',
            render: (id: string) => (
              <>
                {schemeName(id)} {id === benchmark.plan.baseline_id && <Tag>基线</Tag>}
              </>
            ),
          },
          { title: '测量数', render: (_, row) => `${row.completed} / ${row.expected}` },
          { title: '任务正确率', dataIndex: 'task_accuracy', render: percent },
          { title: 'Token 一致率', dataIndex: 'token_agreement', render: percent },
          {
            title: '首 Token / ms',
            render: (_, row) => fmt(row.metrics.runtime_first_token_ms, 2),
          },
          { title: '生成 E2E / ms', render: (_, row) => fmt(row.metrics.runtime_e2e_ms, 2) },
          {
            title: 'Decode / token·s⁻¹',
            render: (_, row) => fmt(row.metrics.decode_tail_tokens_per_s, 2),
          },
          {
            title: '输入 / 输出 Token',
            render: (_, row) =>
              `${fmt(row.metrics.input_tokens, 0)} / ${fmt(row.metrics.output_tokens, 0)}`,
          },
          {
            title: '峰值分配 / GiB',
            render: (_, row) => fmt(gib(row.metrics.peak_allocated_bytes), 3),
          },
          {
            title: 'E2E 相对基线',
            render: (_, row) =>
              row.relative_e2e == null
                ? '—'
                : `${row.relative_e2e >= 0 ? '+' : ''}${percent(row.relative_e2e)}`,
          },
          {
            title: '比较条件',
            render: (_, row) =>
              row.comparable
                ? '同输入、同输出数量'
                : row.reasons.map((reason) => reasons[reason] ?? reason).join('；') ||
                  '尚无完整证据',
          },
        ]}
      />
      <p className="help-text">
        各指标为本用例测量样本的中位数；预热、加载、恢复和机制诊断均不计入。E2E 相对值为候选 / 基线
        − 1，负值表示生成循环更短；输出数量变化时不计算该比例。
      </p>
      <Space wrap className="section-gap">
        <h3>同一输入的输出对照</h3>
        <Select
          aria-label="查看输出重复轮次"
          value={repeat}
          onChange={setRepeat}
          options={Array.from({ length: benchmark.plan.repeats }, (_, value) => ({
            value,
            label: `第 ${value + 1} 次测量`,
          }))}
        />
      </Space>
      <div className="benchmark-outputs">
        {benchmark.plan.schemes.map((scheme) => {
          const run = benchmark.records.find(
            (row) =>
              row.scheme_id === scheme.id &&
              row.case_id === activeCase.id &&
              row.role === 'measurement' &&
              row.repeat === repeat,
          );
          return (
            <section key={scheme.id} className="benchmark-output">
              <h3>{scheme.name}</h3>
              <Tag>{run?.state ?? '等待执行'}</Tag>
              <Tag>
                {run?.score == null
                  ? '未作客观任务评分'
                  : run.score
                    ? '参考答案满足'
                    : '参考答案未满足'}
              </Tag>
              {run && (
                <div className="help-text mono">
                  实际方案：{run.receipt.effective?.kernel_profile ?? '尚无凭据'} ·{' '}
                  {run.receipt.status === 'applied' ? '生效已确认' : '生效未确认'}
                </div>
              )}
              <pre>{run?.output || (run?.error ?? '尚无输出')}</pre>
              {run?.finish_reason === 'length' && (
                <div className="help-text">已达到输出预算，回答可能未完成。</div>
              )}
              {run && (
                <Space wrap>
                  <Link to={`/runs/${run.run_id}?tab=experiment`}>生效凭据与原件</Link>
                  <Link to={`/runs/${run.run_id}`}>完整运行</Link>
                </Space>
              )}
            </section>
          );
        })}
      </div>
      <Alert
        showIcon
        type="info"
        message="质量口径与性能口径分别判断"
        description="算术、结构化提取和检索按固定参考答案评分；解释生成仅展示内容和基线一致率，不自动判优。Token 一致率衡量输出变化，不代表答案正确率。快速对照使用驻留 worker，不能替代独立进程、多轮交替的正式性能与质量验收。"
      />
      {benchmark.records.some((row) => row.role === 'diagnostic') && (
        <details className="section-gap">
          <summary>独立机制诊断（不计入性能统计）</summary>
          <ul>
            {benchmark.records
              .filter((row) => row.role === 'diagnostic')
              .map((row) => (
                <li key={row.run_id}>
                  {schemeName(row.scheme_id)} · {row.case_id} ·{' '}
                  <Link to={`/observations?run=${row.run_id}`}>内存 / KV / OS 页观测</Link>
                </li>
              ))}
          </ul>
        </details>
      )}
      <Space wrap className="section-gap">
        <a href={`${base}/benchmarks/${benchmark.id}`} target="_blank" rel="noreferrer">
          原始对照记录 JSON
        </a>
        <Link to="/optimization">高级预注册实验</Link>
        <Link to="/playground">交互推理案例</Link>
      </Space>
    </>
  );
}
