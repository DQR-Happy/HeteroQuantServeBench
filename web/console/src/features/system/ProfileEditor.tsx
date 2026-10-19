import { InputNumber, Select, Space } from 'antd';
import { Panel } from '../../components';
import type { Hardware, SystemProfile } from './types';

export default function ProfileEditor({
  label,
  value,
  hardware,
  device,
  onChange,
}: {
  label: string;
  value: SystemProfile;
  hardware?: Hardware;
  device: number;
  onChange: (p: SystemProfile) => void;
}) {
  const change = (p: Partial<SystemProfile>) => onChange({ ...value, ...p });
  const gpu = hardware?.devices[device];
  return (
    <Panel title={label}>
      <Space wrap align="start">
        <label>
          CPU 亲和性
          <Select
            aria-label={`${label} CPU 亲和性`}
            mode="multiple"
            allowClear
            placeholder="继承进程允许的全部 CPU"
            style={{ width: 300, maxWidth: '100%' }}
            value={value.cpu_ids ?? []}
            options={hardware?.allowed_cpus.map((i) => ({ value: i, label: `CPU ${i}` }))}
            onChange={(cpu_ids) => change({ cpu_ids: cpu_ids.length ? cpu_ids : null })}
          />
        </label>
        <label>
          算子内线程
          <InputNumber
            aria-label={`${label} 算子内线程`}
            min={1}
            max={64}
            value={value.intra_threads}
            onChange={(n) => change({ intra_threads: n ?? 2 })}
          />
        </label>
        <label>
          算子间线程
          <InputNumber
            aria-label={`${label} 算子间线程`}
            min={1}
            max={16}
            value={value.interop_threads}
            onChange={(n) => change({ interop_threads: n ?? 1 })}
          />
        </label>
        <label>
          NUMA 后续分配
          <Select
            aria-label={`${label} NUMA`}
            style={{ width: 180 }}
            value={value.numa_node ?? -1}
            options={[
              { value: -1, label: '继承原策略' },
              ...(hardware?.numa_policy.available
                ? hardware.numa_nodes.map((n) => ({ value: n.id, label: `绑定 node ${n.id}` }))
                : []),
            ]}
            onChange={(numa_node) => change({ numa_node: numa_node < 0 ? null : numa_node })}
          />
        </label>
        <label>
          GPU 功耗上限 W（留空继承）
          <InputNumber
            aria-label={`${label} GPU 功耗上限`}
            min={gpu?.power_constraints_w?.[0] ?? 1}
            max={gpu?.power_constraints_w?.[1] ?? 1000}
            value={value.power_limit_w}
            onChange={(power_limit_w) => change({ power_limit_w })}
          />
        </label>
      </Space>
      <p>
        线程配置在新进程导入 PyTorch 后、模型加载前设置。亲和性约束 CPU 执行位置，不增加容器配额。
      </p>
      {!hardware?.numa_policy.available && (
        <p>
          NUMA 绑定不可用：{hardware?.numa_policy.reason ?? '等待硬件检查'}。可在执行节点检查容器的
          mempolicy 权限。
        </p>
      )}
      <p>
        功耗范围：{gpu?.power_constraints_w?.join('–') ?? '不可确认'}{' '}
        W；写权限在应用时由驱动核验，失败会保留错误与恢复凭据。
      </p>
    </Panel>
  );
}
