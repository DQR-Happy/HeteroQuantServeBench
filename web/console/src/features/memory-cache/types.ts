import type { Run } from '../../api/types';
export type MemoryPlan = {
  mode: string;
  prefix_reuse: boolean;
  max_cache_tokens?: number;
  prefix_budget_bytes?: number;
  prefix_max_entries?: number;
  prefix_max_tokens?: number;
};
export type MemoryReceipt = {
  plan: MemoryPlan;
  status: string;
  cache_class: string | null;
  use_cache: boolean;
  processed_input_positions: number;
  forward_calls: number;
  static_capacity_tokens: number | null;
  prefix: {
    lookup: string;
    reused_tokens: number;
    restore_copy_bytes: number;
    snapshot_copy_bytes: number;
    retained_bytes: number;
    retained_entries: number;
    evictions: number;
  };
  transfers: {
    d2h_bytes: number;
    h2d_bytes: number;
    offload_calls: number;
    prefetch_calls: number;
  };
  storage_before_release: {
    bytes_by_device: Record<string, number>;
    tensors: {
      layer: number;
      role: string;
      shape: number[];
      dtype: string;
      device: string;
      unique_storage_bytes: number;
    }[];
  };
  process_faults: { minor: number; major: number; scope: string };
  completion: string;
  allocator_after_prefix_publish?: {
    allocated_bytes: number;
    reserved_bytes: number;
    peak_allocated_bytes: number;
  };
};
export type MemoryRun = Run & {
  config: Run['config'] & { memory_plan?: MemoryPlan };
  metrics: Run['metrics'] & { memory_receipt?: MemoryReceipt; worker_request_total_ms?: number };
};
export type Capability = {
  modes: { id: string; label: string; supported: boolean; reason: string | null }[];
  prefix_supported: boolean;
  prefix_reason: string | null;
  prefix_retained_bytes: number;
};
