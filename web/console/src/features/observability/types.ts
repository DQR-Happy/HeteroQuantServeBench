export type ObservationMode = 'off' | 'basic' | 'operators';
export type PhaseSpan = { name: string; start_ms: number; duration_ms: number; source: string };
export type TokenSpan = {
  index: number;
  phase: string;
  start_ms: number;
  duration_ms: number;
  model_host_ms: number;
  selection_ms: number;
  detokenize_ms: number;
  source: string;
};
export type Operator = {
  name: string;
  calls: number;
  cpu_ms: number;
  self_cpu_ms: number;
  cuda_ms: number | null;
  self_cuda_ms: number | null;
  cpu_memory_bytes: number;
  device_memory_bytes: number | null;
  input_shapes: unknown;
  source: string;
};
export type Observation = {
  schema_version: string;
  mode: ObservationMode;
  clock_domain: string;
  time_origin: string;
  phases: PhaseSpan[];
  tokens: TokenSpan[];
  memory_snapshots: {
    label: string;
    time: number;
    start_ms: number;
    host_total_bytes: number | null;
    host_available_bytes: number | null;
    allocated_bytes: number | null;
    reserved_bytes: number | null;
    peak_allocated_bytes: number | null;
    scope: string;
    source: string;
  }[];
  execution: Record<string, unknown>;
  limitations: string[];
  profile?: {
    status: string;
    source: string;
    capture_id?: string;
    activities_requested?: string[];
    activities?: string[];
    operators?: Operator[];
    operator_groups_total?: number;
    operator_groups_truncated?: boolean;
    trace_available?: boolean;
    trace_bytes?: number;
    coverage?: {
      prefill: boolean;
      decode_steps: number;
      max_decode_steps: number;
      total_output_tokens: number;
      scope: string;
      module_scope?: string;
      module_names?: string[];
    };
    limitations?: string[];
  };
};
