export const base = '/api/console/v1';
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(base + path, {
    method: body === undefined ? 'GET' : 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'X-HQSB-Client': 'console',
      ...(body === undefined ? {} : { 'Idempotency-Key': crypto.randomUUID() }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.error?.message ?? data.detail ?? `HTTP ${response.status}`;
    if (response.status === 401 && path !== '/session/login')
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
