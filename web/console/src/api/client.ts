const controlBase = '/api/console/v1';
// Pin all API/stream/download URLs to this tab's machine. Other tabs cannot
// retarget an old view or an in-flight submission by changing a shared cookie.
const machine =
  typeof window === 'undefined'
    ? 'local'
    : (window.sessionStorage.getItem('hqsb.machine') ?? 'local');
export const selectedMachine = machine;
export const base =
  machine === 'local' ? controlBase : `${controlBase}/nodes/${encodeURIComponent(machine)}`;
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const management =
    path === '/machines' || path.startsWith('/machines/') || path.startsWith('/session');
  const response = await fetch((management ? controlBase : base) + path, {
    method: body === undefined ? 'GET' : 'POST',
    credentials: 'same-origin',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'X-HQSB-Client': 'console',
      'X-HQSB-Machine': machine,
      ...(body === undefined ? {} : { 'Idempotency-Key': crypto.randomUUID() }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.error?.message ?? data.detail ?? `HTTP ${response.status}`;
    if (response.status === 401 && path !== '/session/login' && (machine === 'local' || management))
      window.dispatchEvent(new Event('hqsb:unauth'));
    throw new ApiError(
      typeof detail === 'string' ? detail : JSON.stringify(detail),
      response.status,
    );
  }
  return response.json() as Promise<T>;
}
export const terminal = (state: string) =>
  ['completed', 'failed', 'cancelled', 'timed_out', 'interrupted'].includes(state);
export const fmt = (value: number | null | undefined, digits = 1) =>
  value == null || !Number.isFinite(value)
    ? '—'
    : value.toLocaleString('zh-CN', { maximumFractionDigits: digits });
export const date = (value: number) =>
  new Date(value * 1000).toLocaleString('zh-CN', { hour12: false });
export const gib = (value: number | null | undefined) =>
  value == null ? undefined : value / 1024 ** 3;
