import type { components } from './generated';
export type InferenceRequest = components['schemas']['InferenceRequest'];
export type Deployment = {
  id: string;
  name: string;
  provider: string;
  platform: string;
  model: string;
  attention: string;
  precision: string;
  kernel: string;
  state: string;
  epoch: string | null;
  context_limit: number;
  max_output_tokens: number;
  last_error: string | null;
  detail: {
    device_name?: string;
    load_seconds?: number;
    weight_verification?: string;
    metadata_hashes?: Record<string, string>;
  };
};
export type Metrics = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  console_first_content_ms?: number | null;
  runtime_first_token_ms?: number | null;
  console_e2e_ms?: number | null;
  runtime_e2e_ms?: number | null;
  queue_ms?: number;
  output_tokens_per_s?: number | null;
  decode_tail_tokens_per_s?: number | null;
  token_itl_ms?: number[];
  generated_token_ids?: number[];
  measurement_profile?: string;
  memory?: {
    allocated_bytes?: number;
    reserved_bytes?: number;
    peak_allocated_bytes?: number;
    scope?: string;
  };
};
export type Run = {
  id: string;
  kind: string;
  state: string;
  created_at: number;
  updated_at: number;
  seq: number;
  config: {
    deployment: Deployment;
    epoch: string | null;
    input_sha256?: string;
    messages?: Message[];
    max_output_tokens?: number;
    deadline_ms?: number;
  };
  output: string;
  metrics: Metrics;
  error: string | null;
  cleanup: string;
  quality: string;
  finish_reason?: string;
};
export type Message = components['schemas']['Message'];
export type RunPage = { items: Run[]; total: number; next_offset: number | null };
export type EvidenceFile = { id: string; name: string; relative_path: string; bytes: number };
export type Evidence = {
  id: string;
  stage: string;
  experiment: string;
  status: string;
  updated_at: number;
  source: string;
  files: EvidenceFile[];
};
export type EvidenceDetail = {
  id: string;
  name: string;
  sha256: string;
  bytes: number;
  content: unknown;
  format: string;
  source: string;
};
export type Sample = {
  time: number;
  host_total_bytes: number | null;
  host_available_bytes: number | null;
  load_1m: number;
  gpu_util_pct: number | null;
  gpu_temp_c: number | null;
  power_w: number | null;
  source: string;
  gpu_availability: string;
};
export type Telemetry = { host: string; architecture: string; samples: Sample[]; scope: string };
export type Overview = {
  deployments: Deployment[];
  runs: RunPage;
  evidence_count: number;
  verdict_counts: Record<string, number>;
  pending: number;
  telemetry: Telemetry;
  observed_at: number;
};
export type StreamEvent = {
  event_version: string;
  request_id: string;
  seq: number;
  kind: string;
  data: { text?: string; metrics?: Metrics; [key: string]: unknown };
};
export type Session = {
  api_version: string;
  role: string;
  version: string;
  mode: string;
  features: string[];
  limits: { max_pending: number; max_deadline_ms: number };
};
