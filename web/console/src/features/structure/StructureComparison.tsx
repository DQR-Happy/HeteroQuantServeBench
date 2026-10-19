import { Alert, Table } from 'antd';
import { fmt } from '../../api/client';
import { JsonView } from '../../components';
export type Comparison = {
  comparable: boolean;
  reasons: string[];
  quality: Record<string, unknown> | null;
  performance: Record<string, unknown> | null;
  structure: Record<string, unknown>;
};
export default function StructureComparison({ data }: { data: Comparison }) {
  const q = data.quality,
    p = data.performance;
  const rows = q
    ? [
        {
          name: 'PPL',
          unit: '越低越好',
          a: q.baseline_perplexity as number,
          b: q.candidate_perplexity as number,
        },
        {
          name: '平均 NLL',
          unit: '越低越好',
          a: q.baseline_mean_nll as number,
          b: q.candidate_mean_nll as number,
        },
      ]
    : p
      ? [
          {
            name: '生成速度',
            unit: 'Token/s',
            a: (p.output_tokens_per_s as number[])?.[0],
            b: (p.output_tokens_per_s as number[])?.[1],
          },
          {
            name: '生成循环首 Token',
            unit: 'ms',
            a: (p.runtime_first_token_ms as number[])?.[0],
            b: (p.runtime_first_token_ms as number[])?.[1],
          },
          {
            name: '生成循环总时间',
            unit: 'ms',
            a: (p.runtime_e2e_ms as number[])?.[0],
            b: (p.runtime_e2e_ms as number[])?.[1],
          },
          {
            name: '驻留参数',
            unit: 'MiB',
            a: (p.parameter_bytes as number[])?.[0] / 1024 ** 2,
            b: (p.parameter_bytes as number[])?.[1] / 1024 ** 2,
          },
        ]
      : [];
  return (
    <>
      <Alert
        type={data.comparable ? 'info' : 'warning'}
        showIcon
        message={
          data.comparable
            ? '来源、精度与评测协议核对通过'
            : `不能直接对照：${data.reasons.join('；')}`
        }
      />
      {data.comparable && (
        <Table
          size="small"
          pagination={false}
          rowKey="name"
          dataSource={rows}
          columns={[
            { title: '指标', dataIndex: 'name' },
            { title: '单位', dataIndex: 'unit' },
            { title: '原始基线', dataIndex: 'a', render: (v: number) => fmt(v, 6) },
            { title: '结构候选', dataIndex: 'b', render: (v: number) => fmt(v, 6) },
            { title: '候选 − 基线', render: (_, r) => fmt(r.b - r.a, 6) },
          ]}
        />
      )}
      {q && (
        <p>
          PPL 相对增加：{fmt(q.ppl_relative_increase_percent as number, 3)}%；平均 NLL 增量：
          {fmt(q.mean_nll_increase as number, 6)}。此结果覆盖内置英文子集，不是综合任务准确率。
        </p>
      )}
      <details>
        <summary>完整可比性、结构及测量响应</summary>
        <JsonView value={data} />
      </details>
    </>
  );
}
