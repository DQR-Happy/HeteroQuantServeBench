import Button from '../components/ActionButton';
import { useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { Alert, App, Descriptions, Space, Table, Tag } from 'antd';
import { api, fmt, gib } from '../api/client';
import type { Deployment, Telemetry } from '../api/types';
import { Chart, JsonView, PageHead, Panel, QueryState, Stat, Status } from '../components';
import MemoryInspector from '../features/memory/MemoryInspector';
import TensorLedger from '../features/memory/TensorLedger';

export default function Devices() {
  const { hash } = useLocation();
  const focused = useRef('');
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
  useEffect(() => {
    if (!hash || focused.current === hash) return;
    const target = document.getElementById(hash.slice(1));
    if (target) {
      target.scrollIntoView({ block: 'center' });
      focused.current = hash;
    }
  }, [hash, deployments.data]);
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
        description="管理模型生命周期，观察实际执行节点。各平台的适配状态见节点能力与运行记录。"
        extra={
          <Space wrap>
            <Link to="/models">
              <Button>下载模型</Button>
            </Link>
            <Link to="/models/local">
              <Button>管理本地模型</Button>
            </Link>
            <Link to="/memory-flow">
              <Button type="primary">打开内存结构与数据流</Button>
            </Link>
          </Space>
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
          hint={`${last?.source ?? '等待采样'} · 不等同 SM occupancy`}
        />
        <Stat label="GPU 温度" value={last?.gpu_temp_c} unit="°C" hint="控制节点传感器" />
        <Stat
          label={
            last?.power_scope === 'GPU board'
              ? 'GPU 板卡功率'
              : last?.power_scope === 'SoC VDD_IN'
                ? 'SoC 输入功率'
                : '功率（范围未确认）'
          }
          value={last?.power_w}
          unit="W"
          hint={
            last?.power_scope === 'GPU board'
              ? 'nvidia-smi · 物理 GPU 0，非整机功耗'
              : 'VDD_IN 仅适用于 Jetson；缺失传感器不记作零'
          }
        />
      </div>
      <Panel
        title="已配置部署"
        subtitle="当前本地 GPU 同时只允许一个模型驻留；外部引擎通过明确配置接入"
      >
        <QueryState loading={deployments.isPending} error={deployments.error} />
        {deployments.data?.items.map((d) => (
          <div className="deployment-card" id={`deployment-${d.id}`} key={d.id}>
            <div className="panel-head">
              <div>
                <h2>{d.name}</h2>
                <span className="muted mono">{d.id}</span>
              </div>
              <Space>
                <Status state={d.state} />
                {d.provider === 'pytorch' && (
                  <Link to={`/model-architecture?deployment=${encodeURIComponent(d.id)}`}>
                    <Button>查看模型架构</Button>
                  </Link>
                )}
                <Button
                  disabledReason={{
                    reason:
                      d.state === 'ready'
                        ? '此部署已经加载。需要更换精度或量化方案时，请在表示精度页应用并重载。'
                        : '此部署正在执行加载、排空或量化操作，请等待完成后再加载。',
                    next: { to: '/precision?deployment=' + d.id, label: '查看或调整模型方案' },
                  }}
                  disabled={!['unloaded', 'failed'].includes(d.state)}
                  onClick={() => void action(d, 'load')}
                >
                  加载部署
                </Button>
                <Button
                  disabledReason={{
                    reason:
                      '只有就绪的驻留模型可以卸载。当前模型未加载或仍在切换中，请检查部署状态。',
                    next: { to: '/runs', label: '查看加载与卸载任务' },
                  }}
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
                  children: `${d.precision} / ${d.kernel_profile ?? d.kernel}`,
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
            {d.provider === 'pytorch' && (
              <div className="help-text">
                <Link to="/playground">在推理工作台选择具体算子实现与版本、加载或重载模型 →</Link>
              </div>
            )}
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
          title="主机内存趋势"
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
              {
                platform: 'CUDA / Jetson / NVIDIA 独显',
                state: '本地 PyTorch FP16 交互路径；按目标单独验证',
              },
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
