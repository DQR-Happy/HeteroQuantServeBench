import { Checkbox, InputNumber, Select, Space } from 'antd';
import { Panel } from '../../components';
import type { Policy } from './types';
export default function PolicyEditor({
  title,
  value,
  onChange,
}: {
  title: string;
  value: Policy;
  onChange: (p: Policy) => void;
}) {
  const change = (patch: Partial<Policy>) => onChange({ ...value, ...patch });
  return (
    <Panel title={title}>
      <Space wrap align="start">
        <label>
          路由策略{' '}
          <Select
            aria-label={`${title} 路由`}
            value={value.routing}
            style={{ width: 230 }}
            options={[
              { value: 'round_robin', label: 'Round Robin / 轮询' },
              { value: 'least_pending', label: '空闲优先 / 最少在途' },
              { value: 'cache_affinity', label: '精确结果缓存感知' },
              { value: 'length_model', label: '输入字符长度 → 模型路由' },
            ]}
            onChange={(routing) => change({ routing })}
          />
        </label>
        <Checkbox
          checked={value.result_cache}
          onChange={(e) => change({ result_cache: e.target.checked })}
        >
          精确结果缓存
        </Checkbox>
        <Checkbox
          checked={value.prefix_reuse}
          onChange={(e) => change({ prefix_reuse: e.target.checked })}
        >
          L6 KV 前缀复用
        </Checkbox>
        <label>
          缓存 TTL 秒{' '}
          <InputNumber
            min={1}
            max={3600}
            value={value.cache_ttl_s}
            onChange={(n) => change({ cache_ttl_s: n ?? 60 })}
          />
        </label>
        <label>
          缓存条目上限{' '}
          <InputNumber
            min={1}
            max={256}
            value={value.cache_max_entries}
            onChange={(n) => change({ cache_max_entries: n ?? 32 })}
          />
        </label>
        <label>
          输入处理{' '}
          <Select
            aria-label={`${title} 输入处理`}
            style={{ width: 240 }}
            value={value.input_transform}
            options={[
              { value: 'none', label: '原始输入' },
              { value: 'head_tail_tokens', label: '按 Token 保留首尾（有损）' },
              { value: 'recent_messages', label: '保留系统消息及最近对话（有损）' },
            ]}
            onChange={(input_transform) => change({ input_transform })}
          />
        </label>
        <label>
          首尾 Token 上限{' '}
          <InputNumber
            min={16}
            max={4096}
            value={value.input_token_cap}
            onChange={(n) => change({ input_token_cap: n ?? 256 })}
          />
        </label>
        <label>
          最近非系统消息数{' '}
          <InputNumber
            min={1}
            max={16}
            value={value.recent_messages}
            onChange={(n) => change({ recent_messages: n ?? 4 })}
          />
        </label>
        <label>
          输出预算上限（空值继承）
          <InputNumber
            aria-label={`${title} 输出预算`}
            min={1}
            max={512}
            value={value.output_cap}
            onChange={(output_cap) => change({ output_cap })}
          />
        </label>
        <label>
          短输入字符阈值{' '}
          <InputNumber
            min={1}
            max={8192}
            value={value.short_input_chars}
            onChange={(n) => change({ short_input_chars: n ?? 128 })}
          />
        </label>
      </Space>
      <p>
        精确结果缓存与 KV 前缀缓存分开计量；KV 前缀要求受支持的原生全 Attention
        模型。字符长度路由需要分别标记 small / large 的两个不同节点副本；其阈值不是 Token 数。
      </p>
      <Space wrap>
        <Checkbox
          checked={value.autoscale}
          onChange={(e) => change({ autoscale: e.target.checked })}
        >
          按需加载与空闲回收托管副本
        </Checkbox>
        <label>
          最少副本{' '}
          <InputNumber
            min={0}
            max={4}
            value={value.min_replicas}
            onChange={(n) => change({ min_replicas: n ?? 0 })}
          />
        </label>
        <label>
          最多副本{' '}
          <InputNumber
            min={1}
            max={4}
            value={value.max_replicas}
            onChange={(n) => change({ max_replicas: n ?? 1 })}
          />
        </label>
        <label>
          空闲回收 ms{' '}
          <InputNumber
            min={100}
            max={10000}
            value={value.idle_unload_ms}
            onChange={(n) => change({ idle_unload_ms: n ?? 1000 })}
          />
        </label>
      </Space>
      <p>
        托管副本由本实验实际加载和卸载，必须从未加载状态开始。已有副本不会被自动卸载；本功能管理已登记机器的模型进程，不创建云机器或容器。
      </p>
    </Panel>
  );
}
