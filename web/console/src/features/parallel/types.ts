import type { Run } from '../../api/types';
export type ParallelPlan = {
  strategy: string;
  device_type: 'cuda' | 'cpu';
  devices: number[];
  precision: string;
  prompts: string[];
  max_output_tokens: number;
  fixed_output_length: boolean;
  trace_collectives: boolean;
  timeout_s: number;
};
export type RequestResult = {
  index: number;
  token_ids: number[];
  output: string;
  input_tokens: number;
  first_token_ms: number;
  generation_ms: number;
  token_s: number;
  kv: { bytes: number };
  handoff?: { kv_payload_bytes: number; handoff_ms: number; transport: string };
};
export type RankResult = {
  rank: number;
  pid: number;
  group_backend: string | null;
  devices: { id: number; name: string }[];
  load_ms: number;
  parameters: { local_parameter_bytes: number; tensors: unknown[] };
  tp_plan: unknown;
  layer_placement: unknown;
  requests: RequestResult[];
  allocator?: unknown;
  collectives: unknown;
  prefill?: unknown;
};
export type ParallelReceipt = {
  run_id: string;
  status: string;
  plan: ParallelPlan;
  ranks: RankResult[];
  requests: RequestResult[];
  output_tokens: number;
  output_tokens_per_s: number | null;
  workload_ms: number | null;
  job_total_ms: number;
  cleanup: string;
  timing_scope: string;
  limitations: string[];
};
export type ParallelRun = Run & {
  metrics: Run['metrics'] & { parallel_receipt?: ParallelReceipt };
};
