import type { Run } from '../../api/types';

export type TensorEntry = {
  name: string;
  kind: string;
  shape: number[];
  dtype: string;
  device: string;
  logical_bytes: number;
  storage_key: string | null;
  storage_offset: number | null;
  storage_offset_bytes?: number | null;
  storage_bytes: number | null;
  layer?: number | string | null;
};
export type Inventory = {
  scope: string;
  sampled_at?: number;
  availability: string;
  entries: TensorEntry[];
  logical_total_bytes: number | null;
  unique_storage_bytes: number | null;
  entry_count: number;
  truncated: boolean;
  limitations: string[];
};
export type DeploymentInventory = {
  sampled_at: number | null;
  parameters: Inventory | null;
  latest_kv: { run_id: string; sampled_at: number; inventory: Inventory } | null;
};
export type ObservedRun = Run & {
  config: Run['config'] & { observation_mode?: string; load_run_id?: string | null };
  metrics: Run['metrics'] & { kv_inventory?: Inventory };
  result?: Run['result'] & { parameter_inventory?: Inventory };
};
export type CopyEvent = {
  name: string;
  direction: string;
  start_ms: number | null;
  duration_ms: number | null;
  bytes: number | null;
  correlation: string | number | null;
  stream: string | number | null;
  device: string | number | null;
};
export type FlowEdge = {
  direction: string;
  source: string;
  target: string;
  label: string;
  count: number;
  bytes: number | null;
  known_bytes: number | null;
  bytes_known_events: number;
  duration_ms_sum: number | null;
};
export type Dataflow = {
  schema_version: number;
  run_id: string;
  status: string;
  source: string;
  clock_domain: string;
  coverage: Record<string, unknown>;
  nodes: { id: string; label: string }[];
  edges: FlowEdge[];
  totals: {
    copy_events: number;
    kernel_events: number;
    bytes: number | null;
    known_bytes: number | null;
    bytes_known_events: number;
    duration_ms_sum: number | null;
  };
  copy_events: CopyEvent[];
  copy_events_truncated: number;
  limitations: string[];
};
