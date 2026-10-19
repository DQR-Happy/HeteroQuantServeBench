import type { Deployment } from '../../api/types';
import type { ActionBlock } from '../../components/ActionButton';

export function captureBlock(
  deployment: Deployment | undefined,
  busy: boolean,
  runId?: string,
  nativeRequired = false,
): ActionBlock | undefined {
  if (busy)
    return {
      reason: '当前诊断或评测任务尚未结束。请先等待完成，或到运行详情查看进度并取消。',
      next: { to: runId ? `/runs/${runId}` : '/runs', label: '查看当前任务' },
    };
  if (!deployment)
    return {
      reason: '尚未取得所选模型的部署状态。请在设备与部署页确认模型已登记，再刷新。',
      next: { to: '/devices', label: '查看设备与部署' },
    };
  if (deployment.provider !== 'pytorch')
    return {
      reason: '此采集只支持本机 PyTorch 模型，当前部署来自其他推理服务。请选择本地模型部署。',
      next: { to: '/devices', label: '选择本地部署' },
    };
  if (nativeRequired && (deployment.kernel_profile ?? 'framework-native') !== 'framework-native')
    return {
      reason:
        '当前加载的是自定义算子方案。编译或导出诊断需要 framework-native，请在交互推理页的算子实现中选择该方案并应用重载。',
      next: {
        to: `/playground?deployment=${encodeURIComponent(deployment.id)}`,
        label: '切换为 framework-native 算子',
      },
    };
  if (nativeRequired && deployment.structure_artifact_id)
    return {
      reason:
        '当前加载 L5 结构压缩制品；与编译/导出诊断的组合尚未验收。请先在模型结构页恢复原始浮点模型。',
      next: {
        to: `/model-structure/artifacts?deployment=${encodeURIComponent(deployment.id)}`,
        label: '恢复原始浮点模型',
      },
    };
  if (
    nativeRequired &&
    ((deployment.quantization?.method ?? 'none') !== 'none' ||
      (deployment.quantization as { kv?: { bits?: number } })?.kv?.bits)
  )
    return {
      reason:
        '编译或导出诊断需要原生浮点权重和默认 KV。请在表示精度页选择 Floating point、默认 KV，然后点击“应用方案并重载模型”。',
      next: {
        to: `/precision?deployment=${encodeURIComponent(deployment.id)}`,
        label: '配置浮点权重与默认 KV',
      },
    };
  if (deployment.state !== 'ready' || !deployment.epoch)
    return {
      reason: `当前模型${deployment.state === 'unloaded' ? '尚未加载' : deployment.state === 'failed' ? '加载失败' : `状态为 ${deployment.state}`}，没有可执行的就绪实例。请在设备与部署页加载该模型；加载或卸载中的模型需先等待完成。`,
      next: { to: `/devices#deployment-${deployment.id}`, label: '加载或检查此模型' },
    };
  return undefined;
}
