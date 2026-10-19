import { Table } from 'antd';
import { fmt } from '../../api/client';
import { JsonView, Panel } from '../../components';
import type { ParallelReceipt } from './types';
export default function ParallelEvidence({ receipt: r }: { receipt: ParallelReceipt }) {
  return (
    <>
      <Panel title="L9 实际执行与资源凭据">
        <p>
          执行：{r.status} · 清理：{r.cleanup} · 输出 {r.output_tokens} Token ·{' '}
          {fmt(r.output_tokens_per_s, 2)} Token/s
        </p>
        <p>{r.timing_scope}</p>
        <Table
          size="small"
          rowKey="rank"
          scroll={{ x: 780 }}
          pagination={false}
          dataSource={r.ranks}
          columns={[
            { title: 'Rank / PID', render: (_, x) => `${x.rank} / ${x.pid}` },
            { title: '实际设备', render: (_, x) => x.devices.map((d) => d.name).join(', ') },
            { title: '进程组', dataIndex: 'group_backend', render: (v) => v ?? '无' },
            {
              title: '本地参数 GiB',
              render: (_, x) => fmt(x.parameters.local_parameter_bytes / 2 ** 30, 3),
            },
            { title: '加载 ms', dataIndex: 'load_ms', render: (v) => fmt(v) },
            { title: '请求数', render: (_, x) => x.requests.length },
          ]}
        />
        <details>
          <summary>分片形状、实际 TP 计划、层放置、通信事件与分配器</summary>
          <JsonView value={r.ranks} />
        </details>
      </Panel>
      <Panel title="每请求输出与 KV">
        <Table
          size="small"
          rowKey="index"
          pagination={false}
          scroll={{ x: 850 }}
          dataSource={r.requests}
          columns={[
            { title: '请求', dataIndex: 'index' },
            {
              title: '输入 / 输出 Token',
              render: (_, x) => `${x.input_tokens} / ${x.token_ids.length}`,
            },
            { title: '首 Token ms', dataIndex: 'first_token_ms', render: (v) => fmt(v) },
            { title: '生成 Token/s', dataIndex: 'token_s', render: (v) => fmt(v, 2) },
            { title: 'KV MiB', render: (_, x) => fmt(x.kv.bytes / 2 ** 20, 2) },
            {
              title: 'KV 传输 MiB',
              render: (_, x) => (x.handoff ? fmt(x.handoff.kv_payload_bytes / 2 ** 20, 2) : '—'),
            },
            { title: '输出', dataIndex: 'output' },
          ]}
        />
        <details>
          <summary>时间口径、边界与完整凭据</summary>
          <JsonView value={r} />
        </details>
      </Panel>
    </>
  );
}
