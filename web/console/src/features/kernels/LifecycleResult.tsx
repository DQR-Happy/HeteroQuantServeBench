import { Alert, Descriptions, Empty, Space, Table } from 'antd';
import { Link } from 'react-router-dom';
import type { Run } from '../../api/types';
import { fmt, gib } from '../../api/client';
import { Stat } from '../../components';
import type { Observation } from '../observability/types';

/** Lifecycle observations belong to result, independently of generation metrics. */
export default function LifecycleResult({ run }: { run: Run }) {
  const result = run.result;
  const observation = result?.observation;
  const loading = run.kind === 'load';
  return (
    <>
      <Alert
        className="section-gap"
        showIcon
        type="info"
        message={loading ? '模型加载记录' : '模型卸载记录'}
        description={
          loading
            ? '加载与预热用于准备模型；本任务没有提交评测输入，不产生输出质量、首 Token 或吞吐指标。完成加载后，仍需在评测页点击「开始固定输入对照」。'
            : '此任务记录模型资源释放，不产生文本输出或推理质量评分。'
        }
        action={
          <Space wrap>
            <Link to={`/benchmarks?deployment=${encodeURIComponent(run.config.deployment.id)}`}>
              前往固定输入评测
            </Link>
            <Link to={`/playground?deployment=${encodeURIComponent(run.config.deployment.id)}`}>
              前往交互推理
            </Link>
            <Link to="/precision">前往精度与量化</Link>
          </Space>
        }
      />
      <div className="stats-grid">
        <Stat label="任务总时间" value={run.metrics.console_e2e_ms} unit="ms" />
        {loading && (
          <>
            <Stat label="权重加载" value={result?.load_seconds} unit="s" hint="loader 内部计时" />
            <Stat
              label={result?.model_storage_bytes != null ? '已观测模型张量存储' : '参数存储'}
              value={fmt(gib(result?.model_storage_bytes ?? result?.parameter_bytes), 3)}
              unit="GiB"
              hint={
                result?.model_storage_bytes != null
                  ? '参数、buffer 与已知量化状态去重；不含 workspace / allocator 余量'
                  : '旧记录：仅参数字节'
              }
            />
            <Stat
              label="加载峰值分配"
              value={fmt(gib(result?.memory?.peak_allocated_bytes), 3)}
              unit="GiB"
              hint="worker allocator · 含加载与预热"
            />
          </>
        )}
      </div>
      {loading && (
        <Descriptions
          size="small"
          column={{ xs: 1, md: 2 }}
          items={[
            { key: 'model', label: '模型', children: run.config.deployment.model },
            { key: 'class', label: '实际模型类', children: result?.model_class ?? '尚无结果' },
            { key: 'device', label: '执行设备', children: result?.device_name ?? '尚无结果' },
            {
              key: 'placement',
              label: '参数设备',
              children: result?.parameter_devices?.join(', ') ?? '尚无结果',
            },
            {
              key: 'weights',
              label: '权重校验',
              children: result?.weight_verification ?? '尚无结果',
            },
            {
              key: 'kernel',
              label: '已加载算子方案',
              children: result?.kernel_plan?.profile_id ?? '尚无结果',
            },
            {
              key: 'torch',
              label: 'PyTorch / CUDA',
              children: `${result?.torch_version ?? '—'} / ${result?.cuda_version ?? '—'}`,
            },
            {
              key: 'inventory',
              label: '参数与 buffer 清单',
              children: result?.parameter_inventory
                ? `${result.parameter_inventory.entry_count ?? '—'} 项 · ${result.parameter_inventory.availability ?? '未知'}`
                : '未采集',
            },
          ]}
        />
      )}
      {observation ? (
        <>
          <h3>加载阶段 · worker 主机时间</h3>
          <Table<Observation['phases'][number]>
            size="small"
            rowKey={(row) => `${row.name}:${row.start_ms}`}
            pagination={false}
            dataSource={observation.phases}
            columns={[
              { title: '阶段', dataIndex: 'name' },
              { title: '起始 / ms', render: (_, row) => fmt(row.start_ms, 2) },
              { title: '持续 / ms', render: (_, row) => fmt(row.duration_ms, 2) },
              { title: '来源', dataIndex: 'source' },
            ]}
            scroll={{ x: 600 }}
          />
          <p className="help-text">
            权重加载是 loader 内部计时；model_load 主机阶段包含分词器和其他准备工作，设备 Kernel
            时间单独统计。
          </p>
          <h3>加载与预热内存快照</h3>
          <Table<Observation['memory_snapshots'][number]>
            size="small"
            rowKey={(row) => `${row.label}:${row.start_ms}`}
            pagination={false}
            dataSource={observation.memory_snapshots}
            columns={[
              { title: '采样点', dataIndex: 'label' },
              {
                title: '主机可用 / GiB',
                render: (_, row) => fmt(gib(row.host_available_bytes), 3),
              },
              { title: 'Allocated / GiB', render: (_, row) => fmt(gib(row.allocated_bytes), 3) },
              { title: 'Reserved / GiB', render: (_, row) => fmt(gib(row.reserved_bytes), 3) },
              { title: 'Peak / GiB', render: (_, row) => fmt(gib(row.peak_allocated_bytes), 3) },
            ]}
            scroll={{ x: 850 }}
          />
          <p className="help-text">
            快照是采样时刻的 worker allocator 与主机内存，不能相加；本记录未采集生成阶段 KV、OS 页或
            GPU Cache 命中率。参数清单与其边界可在「指标与原始结果」查看。
          </p>
        </>
      ) : (
        <Empty description="该生命周期任务没有阶段观测；完整后端结果可在「指标与原始结果」查看。" />
      )}
    </>
  );
}
