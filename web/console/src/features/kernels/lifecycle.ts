import { api, terminal } from '../../api/client';
import type { Deployment, Run } from '../../api/types';

function pause(signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(new DOMException('Observation stopped', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, 1000);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}

// One explicit user action owns this sequence. Leaving the page stops follow-up
// submissions; accepted jobs remain in the server store and are never resubmitted.
export async function applyKernelProfile(
  deployment: Deployment,
  profileId: string,
  signal: AbortSignal,
  onProgress: (run: Run) => void,
): Promise<Deployment> {
  const checkAbort = () => signal.throwIfAborted();
  const wait = async (run: Run) => {
    const deadline = Date.now() + 360000;
    while (true) {
      checkAbort();
      onProgress(run);
      if (terminal(run.state)) {
        if (run.state !== 'completed' || run.cleanup !== 'succeeded')
          throw new Error(
            run.error ?? `模型${run.kind === 'load' ? '加载' : '卸载'}未成功：${run.state}`,
          );
        return run;
      }
      if (Date.now() > deadline) throw new Error('等待模型任务超时，请在运行记录中检查原任务。');
      await pause(signal);
      run = await api<Run>(`/runs/${run.id}`, undefined, signal);
    }
  };
  let items: Deployment[] = [];
  const current = async () => {
    const data = await api<{ items: Deployment[] }>('/deployments', undefined, signal);
    items = data.items;
    const row = data.items.find((item) => item.id === deployment.id);
    if (!row) throw new Error('部署已不存在，请刷新部署列表。');
    return row;
  };
  checkAbort();
  const initial = await current();
  if (initial.epoch !== deployment.epoch || initial.state !== deployment.state)
    throw new Error('部署状态已变化，请核对当前方案后重新操作。');
  if (deployment.provider === 'pytorch') {
    const owners = items.filter(
      (item) => item.id !== deployment.id && item.provider === 'pytorch' &&
        !['unloaded', 'failed'].includes(item.state),
    );
    if (owners.some((item) => item.state !== 'ready'))
      throw new Error('其他模型正在加载或卸载，请等待生命周期任务完成。');
    for (const owner of owners) {
      checkAbort();
      await wait(await api<Run>(`/deployments/${owner.id}/unload`, {}, signal));
    }
  }
  if (initial.state === 'ready') {
    checkAbort();
    await wait(await api<Run>(`/deployments/${deployment.id}/unload`, {}, signal));
  } else if (!['unloaded', 'failed'].includes(initial.state)) {
    throw new Error('部署正在执行生命周期任务，请等待完成。');
  }
  checkAbort();
  const loaded = await wait(
    await api<Run>(
      `/deployments/${deployment.id}/load`,
      deployment.provider === 'pytorch' ? { kernel_profile: profileId } : {},
      signal,
    ),
  );
  const actual = await current();
  if (
    actual.state !== 'ready' ||
    !actual.epoch ||
    actual.epoch === deployment.epoch ||
    actual.load_run_id !== loaded.id ||
    (deployment.provider === 'pytorch' &&
      (actual.kernel_profile ?? 'framework-native') !== profileId)
  )
    throw new Error('当前部署与本次加载结果不一致，请重新核对，尚不能确认应用成功。');
  return actual;
}
