export type Metric = {
  metric_id: string;
  label?: string;
  value: number | null;
  unit: string;
  scope: string;
  evidence_kind: string;
  availability: string;
  source_record_seqs?: number[];
};
export type Tensor = {
  name: string;
  kind: string;
  shape: number[];
  stride: number[];
  dtype: string;
  device: string;
  contiguous: boolean;
  logical_bytes: number;
  storage_bytes?: number;
  storage_key?: string;
  storage_address?: string;
  storage_offset_bytes?: number;
};
export type Inventory = {
  entries: Tensor[];
  availability: string;
  logical_total_bytes: number;
  unique_storage_bytes: number;
  truncated?: boolean;
};
export type KV = {
  label: string;
  step?: number;
  logical_bytes: number;
  unique_storage_bytes: number;
  availability: string;
  entries: Tensor[];
};
export type Allocator = {
  label: string;
  segments: number;
  blocks: number;
  active_bytes: number;
  awaiting_free_bytes: number;
  inactive_bytes: number;
  history_prefix_may_be_overwritten: boolean;
};
export type Lifetime = {
  device: number;
  address: string;
  bytes: number;
  generation: number;
  alloc_seq: number | null;
  free_requested_seq: number | null;
  free_completed_seq: number | null;
  start_status: string;
  end_status: string;
};
export type Attribution = {
  tensor_name: string;
  storage_bytes: number;
  allocation_address: string | null;
  confidence: string;
  generation: number | null;
  anchor: string;
};
export type Evidence = {
  availability?: string;
  capture_status?: string;
  publication?: string;
  metrics: Metric[];
  kv?: KV[];
  allocator?: Allocator[];
  allocation_lifetimes?: {
    items: Lifetime[];
    total_items?: number;
    projection_limit?: number;
    anchor?: string;
    prefix_may_be_overwritten?: boolean;
  } | null;
  allocation_attribution?: Attribution[];
  tensor_inventories?: { label: string; inventory: Inventory }[];
  page_samples?: {
    label: string;
    sampled_pages: number;
    present_pages: number;
    swapped_pages: number;
    scope: string;
    selection: unknown;
  }[];
  source_problems?: unknown[];
  limitations?: string[];
  context?: { source?: Record<string, unknown>; [key: string]: unknown };
  os_source_availability?: Record<string, string>;
  provenance?: unknown;
};
export const scopeNames: Record<string, string> = {
  worker_allocator: 'Torch 分配器',
  worker_cpu_process: 'Worker CPU 进程',
  worker_process: 'Worker 进程',
  system: '操作系统整机',
  system_not_attributable_to_model: '整机 · 不可归因模型',
};
export function scopeName(scope: string) {
  return scopeNames[scope] ?? scope;
}
export function byteValue(value: number | null | undefined) {
  if (value == null) return '—';
  const unit = Math.abs(value) >= 1024 ** 3 ? 'GiB' : Math.abs(value) >= 1024 ** 2 ? 'MiB' : 'B';
  const divisor = unit === 'GiB' ? 1024 ** 3 : unit === 'MiB' ? 1024 ** 2 : 1;
  return `${(value / divisor).toLocaleString('zh-CN', { maximumFractionDigits: unit === 'B' ? 0 : 3 })} ${unit}`;
}
