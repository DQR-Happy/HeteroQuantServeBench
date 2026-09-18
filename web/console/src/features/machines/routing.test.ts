import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

function browser(machine = 'remote') {
  const setItem = vi.fn();
  const assign = vi.fn();
  const dispatchEvent = vi.fn();
  vi.stubGlobal('window', {
    sessionStorage: { getItem: () => machine, setItem },
    location: { assign },
    dispatchEvent,
  });
  return { setItem, assign, dispatchEvent };
}

describe('machine routing ownership', () => {
  it('submits model downloads to the selected execution node', async () => {
    browser('compute');
    vi.stubGlobal('crypto', { randomUUID: () => 'download-request' });
    const fetch = vi.fn().mockResolvedValue(new Response('{"id":"download-owned"}'));
    vi.stubGlobal('fetch', fetch);
    const { api } = await import('../../api/client');
    const plan = { plan_sha256: 'a'.repeat(64) };
    await api('/model-hub/downloads', plan);
    expect(fetch.mock.calls[0][0]).toBe('/api/console/v1/nodes/compute/model-hub/downloads');
    expect(fetch.mock.calls[0][1].method).toBe('POST');
    expect(fetch.mock.calls[0][1].body).toBe(JSON.stringify(plan));
  });
  it('keeps local machine management accessible after a remote auth failure', async () => {
    const ui = browser();
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('{"detail":"remote auth required"}', { status: 401 })),
    );
    const { api } = await import('../../api/client');
    await expect(api('/runs')).rejects.toThrow('remote auth required');
    expect(ui.dispatchEvent).not.toHaveBeenCalled();
  });
  it('pins data and download URLs while keeping management local', async () => {
    browser();
    const fetch = vi.fn().mockImplementation(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const { api, base } = await import('../../api/client');
    expect(base).toBe('/api/console/v1/nodes/remote');
    await api('/runs/example');
    expect(fetch.mock.calls[0][0]).toBe('/api/console/v1/nodes/remote/runs/example');
    await api('/machines');
    expect(fetch.mock.calls[1][0]).toBe('/api/console/v1/machines');
    await api('/session');
    expect(fetch.mock.calls[2][0]).toBe('/api/console/v1/session');
  });

  it('does not change selection or reload when activation fails', async () => {
    const ui = browser('local');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('{"error":{"message":"offline"}}', { status: 502 })),
    );
    const { activateMachine } = await import('./types');
    await expect(activateMachine('remote')).rejects.toThrow('offline');
    expect(ui.setItem).not.toHaveBeenCalled();
    expect(ui.assign).not.toHaveBeenCalled();
  });

  it('verifies the backend before persisting selection and replacing the document', async () => {
    const ui = browser('local');
    const fetch = vi.fn().mockImplementation(async () => {
      expect(ui.setItem).not.toHaveBeenCalled();
      return new Response('{"selected_id":"remote"}', { status: 200 });
    });
    vi.stubGlobal('fetch', fetch);
    const { activateMachine } = await import('./types');
    await activateMachine('remote');
    expect(fetch.mock.calls[0][0]).toBe('/api/console/v1/machines/remote/activate');
    expect(ui.setItem).toHaveBeenCalledWith('hqsb.machine', 'remote');
    expect(ui.assign).toHaveBeenCalledWith('/devices');
  });
});
