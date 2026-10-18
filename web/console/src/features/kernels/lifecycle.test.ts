import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../api/client';
import type { Deployment, Run } from '../../api/types';
import { applyKernelProfile, unloadDeployment } from './lifecycle';

vi.mock('../../api/client', async (original) => ({
  ...(await original<typeof import('../../api/client')>()),
  api: vi.fn(),
}));
const request = vi.mocked(api);
const initial = {
  id: 'model',
  provider: 'pytorch',
  state: 'ready',
  epoch: 'old',
  kernel_profile: 'framework-native',
} as Deployment;
const done = (kind: string, state = 'completed') =>
  ({ id: `${kind}-job`, kind, state, cleanup: 'succeeded' }) as Run;
const loaded = {
  ...initial,
  epoch: 'new',
  kernel_profile: 'qwen3-norm-vector-v2',
  load_run_id: 'load-job',
};
beforeEach(() => request.mockReset());
describe('kernel lifecycle owns unload, load and confirmation in order', () => {
  it('applies L4 load options and verifies the effective method and compute dtype', async () => {
    const start = { ...initial, state: 'unloaded', epoch: null };
    const actual = {
      ...loaded,
      kernel_profile: 'framework-native',
      precision: 'bfloat16',
      quantization: { method: 'bnb_nf4' },
    };
    request
      .mockResolvedValueOnce({ items: [start] })
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({ items: [actual] });
    expect(
      await applyKernelProfile(start, 'framework-native', new AbortController().signal, vi.fn(), {
        precision: 'bfloat16',
        quantization: { method: 'bnb_nf4' },
      }),
    ).toEqual(actual);
    expect(request.mock.calls[1][1]).toEqual({
      kernel_profile: 'framework-native',
      precision: 'bfloat16',
      quantization: { method: 'bnb_nf4' },
    });
  });
  it('does not accept a precision reload that reports the wrong method', async () => {
    const start = { ...initial, state: 'unloaded', epoch: null };
    request
      .mockResolvedValueOnce({ items: [start] })
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({
        items: [
          {
            ...loaded,
            kernel_profile: 'framework-native',
            precision: 'float16',
            quantization: { method: 'none' },
          },
        ],
      });
    await expect(
      applyKernelProfile(start, 'framework-native', new AbortController().signal, vi.fn(), {
        precision: 'float16',
        quantization: { method: 'bnb_nf4' },
      }),
    ).rejects.toThrow('尚不能确认应用成功');
  });
  it('drains the other local model before loading the selected model', async () => {
    const start = { ...initial, state: 'unloaded', epoch: null };
    request
      .mockResolvedValueOnce({ items: [start, { ...initial, id: 'other-model' }] })
      .mockResolvedValueOnce(done('unload'))
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({ items: [loaded] });
    await applyKernelProfile(start, loaded.kernel_profile, new AbortController().signal, vi.fn());
    expect(request.mock.calls.map(([path]) => path)).toEqual([
      '/deployments',
      '/deployments/other-model/unload',
      '/deployments/model/load',
      '/deployments',
    ]);
  });
  it('never loads a new model if the previous owner did not release the device', async () => {
    const start = { ...initial, state: 'unloaded', epoch: null };
    request
      .mockResolvedValueOnce({ items: [start, { ...initial, id: 'other-model' }] })
      .mockResolvedValueOnce({ ...done('unload', 'failed'), error: 'device retained' });
    await expect(
      applyKernelProfile(start, 'framework-native', new AbortController().signal, vi.fn()),
    ).rejects.toThrow('device retained');
    expect(request).toHaveBeenCalledTimes(2);
  });
  it('reloads with the exact selected version and verifies the new epoch', async () => {
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce(done('unload'))
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({ items: [loaded] });
    const progress = vi.fn();
    expect(
      await applyKernelProfile(
        initial,
        loaded.kernel_profile,
        new AbortController().signal,
        progress,
      ),
    ).toEqual(loaded);
    expect(request.mock.calls.map(([path, body]) => [path, body])).toEqual([
      ['/deployments', undefined],
      ['/deployments/model/unload', {}],
      ['/deployments/model/load', { kernel_profile: loaded.kernel_profile }],
      ['/deployments', undefined],
    ]);
    expect(progress.mock.calls.map(([run]) => run.kind)).toEqual(['unload', 'load']);
  });
  it('loads an unloaded deployment without another unload', async () => {
    const start = { ...initial, state: 'unloaded', epoch: null };
    request
      .mockResolvedValueOnce({ items: [start] })
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({ items: [loaded] });
    await applyKernelProfile(start, loaded.kernel_profile, new AbortController().signal, vi.fn());
    expect(request.mock.calls.some(([path]) => path.endsWith('/unload'))).toBe(false);
  });
  it('does not submit a load after an unload failure', async () => {
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce({ ...done('unload', 'failed'), error: 'cleanup failed' });
    await expect(
      applyKernelProfile(initial, loaded.kernel_profile, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('cleanup failed');
    expect(request).toHaveBeenCalledTimes(2);
  });
  it('stops follow-up submissions when the page leaves after an accepted unload', async () => {
    const abort = new AbortController();
    request.mockResolvedValueOnce({ items: [initial] }).mockResolvedValueOnce(done('unload'));
    await expect(
      applyKernelProfile(initial, loaded.kernel_profile, abort.signal, () => abort.abort()),
    ).rejects.toThrow();
    expect(request).toHaveBeenCalledTimes(2);
  });
  it('rejects a stale selection before unloading', async () => {
    request.mockResolvedValueOnce({ items: [{ ...initial, epoch: 'another-window' }] });
    await expect(
      applyKernelProfile(initial, loaded.kernel_profile, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('部署状态已变化');
    expect(request).toHaveBeenCalledTimes(1);
  });
  it('never reports success for the wrong effective implementation', async () => {
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce(done('unload'))
      .mockResolvedValueOnce(done('load'))
      .mockResolvedValueOnce({ items: [{ ...loaded, kernel_profile: 'framework-native' }] });
    await expect(
      applyKernelProfile(initial, loaded.kernel_profile, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('尚不能确认应用成功');
  });
});

describe('standalone unload confirms model resource release', () => {
  it('waits for a successful unload and checks the actual deployment', async () => {
    const unloaded = { ...initial, state: 'unloaded', epoch: null };
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce(done('unload'))
      .mockResolvedValueOnce({ items: [unloaded] });
    expect(await unloadDeployment(initial, new AbortController().signal, vi.fn())).toEqual(
      unloaded,
    );
    expect(request.mock.calls.map(([path]) => path)).toEqual([
      '/deployments',
      '/deployments/model/unload',
      '/deployments',
    ]);
  });
  it('rejects stale epochs without submitting an unload', async () => {
    request.mockResolvedValueOnce({ items: [{ ...initial, epoch: 'another-load' }] });
    await expect(unloadDeployment(initial, new AbortController().signal, vi.fn())).rejects.toThrow(
      '部署状态已变化',
    );
    expect(request).toHaveBeenCalledTimes(1);
  });
  it('does not report success when cleanup or observed release is unconfirmed', async () => {
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce({ ...done('unload'), cleanup: 'unknown' });
    await expect(unloadDeployment(initial, new AbortController().signal, vi.fn())).rejects.toThrow(
      '卸载未成功',
    );
    request.mockReset();
    request
      .mockResolvedValueOnce({ items: [initial] })
      .mockResolvedValueOnce(done('unload'))
      .mockResolvedValueOnce({ items: [initial] });
    await expect(unloadDeployment(initial, new AbortController().signal, vi.fn())).rejects.toThrow(
      '尚不能确认模型已卸载',
    );
  });
});
