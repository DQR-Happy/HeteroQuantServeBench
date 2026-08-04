import { useState } from 'react';
import { Alert, Empty, Select, Table, Tabs, Tag } from 'antd';
import { fmt, gib } from '../../api/client';
import { JsonView, Panel, Stat, Status } from '../../components';
import ArtifactLinks from './ArtifactLinks';
import type { QuantizationMethod, Research } from './types';

export default function QuantizationEvidence({
  quantization,
}: {
  quantization: Research['quantization'];
}) {
  const [selected, setSelected] = useState('');
  const method = quantization.methods.find((row) => row.id === selected) ?? quantization.methods[0];
  return (
    <Panel
      title="量化：文件体积、执行路径与质量"
      subtitle={`${quantization.source} · 历史实验，未改变当前部署`}
      extra={<Status state={quantization.verdict} />}
    >
      {quantization.limitations.length > 0 && (
        <Alert
          showIcon
          type="warning"
          className="section-gap"
          message="实验结论与限制"
          description={quantization.limitations.join('；')}
        />
      )}
      {!method ? (
        <Empty description="没有可读取的量化实验" />
      ) : (
        <>
          <label className="field-label" htmlFor="research-quant">
            选择历史量化方案
          </label>
          <Select
            id="research-quant"
            className="wide-select"
            value={method.id}
            onChange={setSelected}
            options={quantization.methods.map((row) => ({
              label: `${row.id} · ${row.bits ?? '—'} bit${row.group_size ? ` · group ${row.group_size}` : ''}`,
              value: row.id,
            }))}
          />
          <div className="execution-proof section-gap">
            <Tag color="blue">{method.execution_label}</Tag>
            <Tag color={method.native_low_bit_kernel ? 'green' : undefined}>
              {method.native_low_bit_kernel == null
                ? '原生低比特执行未知'
                : method.native_low_bit_kernel
                  ? '原生低比特执行'
                  : '未使用原生低比特 kernel'}
            </Tag>
            <Tag
              color={
                method.quality.passed === true
                  ? 'green'
                  : method.quality.passed === false
                    ? 'red'
                    : undefined
              }
            >
              {method.quality.passed === true
                ? '质量门通过'
                : method.quality.passed === false
                  ? '质量门未通过'
                  : '质量结果见原始证据'}
            </Tag>
            <p>{method.execution_description}</p>
          </div>
          <div className="stats-grid">
            <Stat
              label="被量化源权重"
              value={gib(method.storage_sizes.fp16_source_bytes)}
              unit="GiB"
              hint="选中权重的 FP16 字节数"
            />
            <Stat
              label="量化制品大小"
              value={gib(method.storage_sizes.quantized_bytes)}
              unit="GiB"
              hint="packed 权重 + scale + manifest"
            />
            <Stat
              label="保留 FP16 部分"
              value={gib(method.storage_sizes.retained_fp16_bytes)}
              unit="GiB"
              hint="量化范围外的权重"
            />
            <Stat
              label="整模型等效存储"
              value={gib(method.storage_sizes.whole_model_equivalent_bytes)}
              unit="GiB"
              hint="存储估算；不等于实际驻留内存"
            />
          </div>
          <Tabs
            items={[
              {
                key: 'quality',
                label: '质量与阈值',
                children: (
                  <>
                    <Table<{ name: string; passed: boolean }>
                      rowKey="name"
                      size="small"
                      pagination={false}
                      dataSource={Object.entries(method.quality.checks ?? {}).map(
                        ([name, passed]) => ({ name, passed }),
                      )}
                      columns={[
                        { title: '验收项', dataIndex: 'name' },
                        {
                          title: '结果',
                          dataIndex: 'passed',
                          render: (passed: boolean) => <Status state={passed ? 'PASS' : 'FAIL'} />,
                        },
                      ]}
                    />
                    <div className="grid-two section-gap">
                      <div>
                        <h3>实测质量</h3>
                        <JsonView value={method.quality.observed ?? method.quality} />
                      </div>
                      <div>
                        <h3>预定阈值</h3>
                        <JsonView value={method.quality.thresholds ?? null} />
                      </div>
                    </div>
                  </>
                ),
              },
              {
                key: 'memory',
                label: '实际驻留',
                children: (
                  <>
                    <Alert
                      type="info"
                      showIcon
                      className="section-gap"
                      message="检查文件缩小是否真正转化为 allocator 内存下降。以下是历史运行实测。"
                    />
                    <Table<QuantizationMethod['runtime_memory'][number]>
                      rowKey="run_id"
                      size="small"
                      scroll={{ x: 620 }}
                      dataSource={method.runtime_memory}
                      columns={[
                        { title: '运行', dataIndex: 'run_id' },
                        {
                          title: 'Allocated / GiB',
                          render: (_, row) => fmt(gib(row.allocated_bytes), 3),
                        },
                        {
                          title: 'Reserved / GiB',
                          render: (_, row) => fmt(gib(row.reserved_bytes), 3),
                        },
                        {
                          title: 'Peak / GiB',
                          render: (_, row) => fmt(gib(row.peak_allocated_bytes), 3),
                        },
                      ]}
                    />
                  </>
                ),
              },
              {
                key: 'performance',
                label: '性能与制品明细',
                children: (
                  <JsonView
                    value={{ storage_sizes: method.storage_sizes, performance: method.performance }}
                  />
                ),
              },
              {
                key: 'artifacts',
                label: '原始证据',
                children: <ArtifactLinks artifacts={method.artifacts} />,
              },
            ]}
          />
        </>
      )}
    </Panel>
  );
}
