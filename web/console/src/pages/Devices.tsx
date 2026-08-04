import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Alert, App, Button, Descriptions, Space, Table, Tag } from 'antd';
import { api, fmt, gib } from '../api/client';
import type { Deployment, Telemetry } from '../api/types';
import { Chart, JsonView, PageHead, Panel, QueryState, Stat, Status } from '../components';
import MemoryInspector from '../features/memory/MemoryInspector';
import TensorLedger from '../features/memory/TensorLedger';

export default function Devices() {
  const client = useQueryClient();
  const { message, modal } = App.useApp();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const telemetry = useQuery({
    queryKey: ['telemetry'],
    queryFn: () => api<Telemetry>('/telemetry'),
    refetchInterval: 2000,
  });
  const samples = telemetry.data?.samples ?? [];
  const last = samples.at(-1);
  const action = async (d: Deployment, kind: string) => {
    try {
      await api(`/deployments/${d.id}/${kind}`, {});
      void client.invalidateQueries({ queryKey: ['deployments'] });
      void message.success(
        kind === 'load' ? '加载任务已提交' : '停止接收新请求，等待队列排空后卸载',
      );
    } catch (e) {
      void message.error((e as Error).message);
    }
  };
  return (
    <>
      <PageHead
        kicker="COMPUTE & DEPLOYMENTS"
        title="设备与部署"
        description="管理模型生命周期，观察实际执行节点。平台名称不代表已完成硬件适配认证。"
        extra={
          <Link to="/memory-flow">
            <Button type="primary">打开内存结构与数据流</Button>
          </Link>
        }
      />
      <div className="stats-grid">
        <Stat
          label="主机可用内存"
          value={gib(last?.host_available_bytes)}
          unit="GiB"
          hint="/proc/meminfo · MemAvailable"
        />
        <Stat
          label="GPU busy"
          value={last?.gpu_util_pct}
          unit="%"
          hint="tegrastats · 不等同 SM occupancy"
        />
        <Stat label="GPU 温度" value={last?.gpu_temp_c} unit="°C" hint="控制节点传感器" />
        <Stat
          label="SoC 输入功率"
          value={last?.power_w}
          unit="W"
          hint="VDD_IN · 不能等同 GPU 独立功耗"
        />
      </div>
      <Panel
        title="已配置部署"
        subtitle="当前本地 GPU 同时只允许一个模型驻留；外部引擎通过明确配置接入"
      >
        <QueryState loading={deployments.isPending} error={deployments.error} />
        {deployments.data?.items.map((d) => (
          <div className="deployment-card" key={d.id}>
            <div className="panel-head">
              <div>
                <h2>{d.name}</h2>
                <span className="muted mono">{d.id}</span>
              </div>
              <Space>
                <Status state={d.state} />
                <Button
                  disabled={!['unloaded', 'failed'].includes(d.state)}
                  onClick={() => void action(d, 'load')}
                >
                  加载部署
                </Button>
                <Button
                  disabled={d.state !== 'ready'}
                  onClick={() =>
                    modal.confirm({
                      title: '排空并卸载此部署？',
                      content: '现有请求执行结束后释放模型资源，新的推理请求将被拒绝。',
                      okText: '排空并卸载',
                      cancelText: '返回',
                      onOk: () => action(d, 'unload'),
                    })
                  }
                >
                  卸载
                </Button>
              </Space>
            </div>
            <Descriptions
              size="small"
              column={{ xs: 1, md: 3 }}
              items={[
                { key: 'model', label: '模型', children: d.model },
                {
                  key: 'platform',
                  label: '平台 / Provider',
                  children: `${d.platform} / ${d.provider}`,
                },
                {
                  key: 'precision',
                  label: '精度 / 算子',
                  children: `${d.precision} / ${d.kernel}`,
                },
                {
                  key: 'epoch',
                  label: '部署版本',
                  children: <span className="mono small">{d.epoch ?? '未加载'}</span>,
                },
                {
                  key: 'limits',
                  label: '上下文 / 输出上限',
                  children: `${d.context_limit} / ${d.max_output_tokens}`,
                },
                { key: 'load', label: '加载耗时', children: `${fmt(d.detail.load_seconds)} s` },
              ]}
            />
            {d.last_error && <Alert type="error" message={d.last_error} />}
            <details>
              <summary>加载结果与制品身份</summary>
              <JsonView value={d.detail} />
            </details>
          </div>
        ))}
      </Panel>
      <MemoryInspector />
      <TensorLedger />
      <div className="grid-two">
        <Panel
          title="共享内存趋势"
          subtitle={`${telemetry.data?.host ?? '—'} · ${telemetry.data?.architecture ?? '—'}`}
        >
          <QueryState error={telemetry.error} />
          <Chart
            label="主机可用内存趋势"
            option={{
              tooltip: { trigger: 'axis' },
              grid: { left: 48, right: 16, top: 16, bottom: 28 },
              xAxis: {
                type: 'category',
                data: samples.map((x) => new Date(x.time * 1000).toLocaleTimeString()),
              },
              yAxis: { type: 'value', name: 'GiB' },
              series: [
                {
                  name: '可用内存',
                  type: 'line',
                  showSymbol: false,
                  data: samples.map((x) => gib(x.host_available_bytes)),
                  lineStyle: { color: '#258c7e' },
                },
              ],
            }}
          />
        </Panel>
        <Panel title="异构扩展边界" subtitle="没有真实端点与证据的平台，不会变成在线设备">
          <Table
            size="small"
            pagination={false}
            rowKey="platform"
            dataSource={[
              { platform: 'Jetson / CUDA', state: '本地 PyTorch FP16 交互路径' },
              { platform: 'ROCm / W7900D', state: '外部引擎接口可接入，硬件未认证' },
              { platform: 'Ascend / MUSA / MTT S4000', state: '需实际兼容服务与对应证据' },
              { platform: 'RK3588 / RKNN', state: '专用 Provider 尚未实现' },
            ]}
            columns={[
              { title: '平台', dataIndex: 'platform' },
              { title: '当前边界', dataIndex: 'state' },
            ]}
          />
          <div className="help-text">
            外部推理服务不共享本页控制节点遥测；内存与 GPU 调优尚未开放写操作。
          </div>
          <Tag>能力按部署生效</Tag>
          <Tag>无远程 shell 入口</Tag>
        </Panel>
      </div>
    </>
  );
}
