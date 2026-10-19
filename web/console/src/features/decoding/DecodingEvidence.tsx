import { Alert, Table } from 'antd';
import { JsonView, Panel } from '../../components';
import { fmt } from '../../api/client';
import type { DecodeReceipt, DecodeComparison } from './types';
export default function DecodingEvidence({ receipt }: { receipt: DecodeReceipt }) {
  const s = receipt.statistics;
  const rows = [
    ['目标模型 Forward', s.target_forward_calls, '次'],
    ['目标验证批次', s.verification_calls, '次'],
    ['草稿模型 Forward', s.draft_forward_calls, '次'],
    ['提出 / 接受 Token', `${s.proposed_tokens ?? 0} / ${s.accepted_tokens ?? 0}`, '个'],
    [
      '接受率',
      typeof s.acceptance_rate === 'number'
        ? `${fmt(s.acceptance_rate * 100, 2)}%`
        : '没有候选，未定义',
      '',
    ],
    [
      '目标 / 草稿 KV 回滚',
      `${s.target_rollback_tokens ?? 0} / ${s.draft_rollback_tokens ?? 0}`,
      'Token',
    ],
  ].map(([name, value, unit]) => ({ name, value, unit }));
  return (
    <Panel title="L7 实际解码与验证凭据">
      <p>
        执行状态：{receipt.status}。约束校验：
        {receipt.constraint_satisfied == null
          ? '未配置有限候选'
          : receipt.constraint_satisfied
            ? '命中合法候选终点'
            : '未完成合法候选'}
        。
      </p>
      <Table
        rowKey="name"
        size="small"
        pagination={false}
        dataSource={rows}
        columns={[
          { title: '机制指标', dataIndex: 'name' },
          { title: '实际记录', dataIndex: 'value' },
          { title: '单位', dataIndex: 'unit' },
        ]}
      />
      <p>
        同一批验证可同时确认多个 Token，批内到达间隔可能为零；这不等于每个 Token 的 GPU
        计算耗时为零。Token/s 包含生成循环，冷启动草稿加载另计入请求总时间。
      </p>
      <details>
        <summary>完整配方、草稿身份、逐轮验证与清理</summary>
        <JsonView value={receipt} />
      </details>
    </Panel>
  );
}
export function ComparisonView({ data }: { data: DecodeComparison }) {
  const p = data.performance;
  const rows = [
    ['生成速度', 'Token/s', 'output_tokens_per_s'],
    ['生成循环首 Token', 'ms', 'runtime_first_token_ms'],
    ['生成循环总时间', 'ms', 'runtime_e2e_ms'],
    ['Worker 请求总时间（含冷加载/清理）', 'ms', 'worker_request_total_ms'],
  ].map(([name, unit, key]) => ({
    name,
    unit,
    a: (p[key] as number[])?.[0],
    b: (p[key] as number[])?.[1],
  }));
  return (
    <>
      <Alert
        type={data.comparable ? 'info' : 'warning'}
        showIcon
        message={
          data.comparable ? '来源与负载核对通过' : `不能直接比较：${data.reasons.join('；')}`
        }
      />
      <p>
        逐 Token 相同：{data.quality.exact_tokens ? '是' : '否'}。结论：{data.quality.status}
        。采样不以一次输出相同来证明分布或质量等价，需另做多随机种子、具体任务评测。
      </p>
      <Table
        rowKey="name"
        pagination={false}
        size="small"
        scroll={{ x: 650 }}
        dataSource={rows}
        columns={[
          { title: '指标', dataIndex: 'name' },
          { title: '单位', dataIndex: 'unit' },
          { title: '基线', dataIndex: 'a', render: (v) => fmt(v, 3) },
          { title: '候选', dataIndex: 'b', render: (v) => fmt(v, 3) },
        ]}
      />
      <p>
        这里只展示单对观测，不作正式加速结论。Teacher-forcing PPL
        不执行解码策略，不能作为本页采样或推测的质量分数。
      </p>
      <details>
        <summary>可比性与原始测量</summary>
        <JsonView value={data} />
      </details>
    </>
  );
}
