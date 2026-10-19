import { Alert, Table } from 'antd';
import { fmt } from '../../api/client';
import { JsonView } from '../../components';
export type Comparison = {
  comparable: boolean;
  reasons: string[];
  generation_check: {
    exact_tokens: boolean;
    max_abs_selected_token_logprob_delta: number | null;
    scope: string;
  } | null;
  performance: ({ status: string } & Record<string, unknown>) | null;
  limitations: string[];
};
const fields = [
  ['worker_request_total_ms', 'Worker 请求总时间（含前缀与清理）', 'ms'],
  ['runtime_first_token_ms', '生成循环首 Token', 'ms'],
  ['runtime_e2e_ms', '生成循环总时间', 'ms'],
  ['output_tokens_per_s', '生成速度', 'Token/s'],
  ['decode_tail_tokens_per_s', 'Decode 速度', 'Token/s'],
];
export default function MemoryComparison({ data }: { data: Comparison }) {
  const check = data.generation_check;
  return (
    <>
      <Alert
        showIcon
        type={!data.comparable ? 'warning' : check?.exact_tokens ? 'info' : 'error'}
        message={
          !data.comparable
            ? `不能直接对照：${data.reasons.join('；')}`
            : check?.exact_tokens
              ? '同输入生成 Token 完全一致'
              : '同输入生成 Token 出现差异'
        }
        description={
          check
            ? `已选 Token logprob 最大绝对差：${fmt(check.max_abs_selected_token_logprob_delta, 6)}。这是当前输入的数值检查，不能代替质量 benchmark。`
            : undefined
        }
      />
      {data.comparable && (
        <Table
          size="small"
          rowKey="key"
          pagination={false}
          dataSource={fields.map(([key, label, unit]) => ({
            key,
            label,
            unit,
            values: data.performance?.[key] as (number | null)[] | undefined,
          }))}
          columns={[
            { title: '指标', dataIndex: 'label' },
            { title: '单位', dataIndex: 'unit' },
            { title: '基线', render: (_, row) => fmt(row.values?.[0], 3) },
            { title: '候选', render: (_, row) => fmt(row.values?.[1], 3) },
            {
              title: '候选 − 基线',
              render: (_, row) =>
                row.values?.[0] != null && row.values?.[1] != null
                  ? fmt(row.values[1] - row.values[0], 3)
                  : '—',
            },
          ]}
        />
      )}
      <p className="muted">
        这是描述性对照。冷启动、前缀未命中和命中样本应分开看；反复测试同一配置后再判断性能收益。
      </p>
      <details>
        <summary>比较条件、存储差异与原始响应</summary>
        <JsonView value={data} />
      </details>
    </>
  );
}
