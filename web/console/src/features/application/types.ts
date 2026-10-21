import type { components } from '../../api/generated';
import type { Run } from '../../api/types';
export type Plan = components['schemas']['ApplicationJobRequest']['plan'];
export type Policy = components['schemas']['ApplicationPolicy'];
export type Target = components['schemas']['ApplicationTarget'];
export type Input = components['schemas']['ApplicationInput'];
export type RequestResult = {
  id: string;
  tenant: string;
  status: string;
  target?: string;
  cache: string;
  backend_invoked: boolean;
  backend_run_id?: string;
  output?: string;
  output_tokens?: number;
  input_tokens?: number;
  token_ids?: number[];
  gateway_e2e_ms: number;
  queue_ms?: number;
  deadline_met: boolean;
  error?: string;
  backend_first_token_ms?: number | null;
  input?: {
    input_tokens_before: number;
    input_tokens_after: number;
    lossy: boolean;
    effective_output_tokens: number;
  };
  backend_metrics?: { memory?: unknown; output_tokens_per_s?: number; memory_mechanism?: unknown };
};
export type Variant = {
  name: string;
  requests: RequestResult[];
  policy: Policy;
  metrics: Record<string, number | null>;
  variant_total_ms?: number;
  cache: { hits: number; misses: number; bytes: number; evictions: number };
};
export type Receipt = {
  run_id: string;
  status: string;
  cleanup: string;
  plan: Plan;
  variants: Variant[];
  events: {
    kind: string;
    offset_ms: number;
    target?: string;
    reason?: string;
    child_run_id?: string;
  }[];
  comparison?: {
    all_tokens_exact: boolean;
    metric_deltas: Record<string, number>;
    pairs: {
      id: string;
      complete: boolean;
      token_ids_exact: boolean;
      same_effective_input: boolean;
      same_model: boolean;
      output_token_delta: number;
      e2e_delta_ms: number;
    }[];
  } | null;
  limitations: string[];
  timing_scope: string;
  job_total_ms: number;
  error?: string | null;
};
export type ApplicationRun = Run & { metrics: Run['metrics'] & { application_receipt?: Receipt } };
