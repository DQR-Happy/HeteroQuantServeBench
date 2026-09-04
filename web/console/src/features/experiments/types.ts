import type { components } from '../../api/generated';
import type { KernelSummary } from '../kernels/types';
export type Arm = 'baseline' | 'candidate';
export type Role = 'performance' | 'diagnostic' | 'quality';
export type ExperimentJobRequest = components['schemas']['JobRequest'];
export type Receipt = {
  runtime?: { kernel: string; dtype: string; executor: string };
  kernel_receipt?: KernelSummary | null;
  status: string;
  requested: Record<string, unknown>;
  resolved: Record<string, unknown>;
  effective: Record<string, unknown>;
  plan_sha256: string;
  measurement_role: string;
  proof_scope: string;
  experiment_links: { campaign_id?: string; arm?: Arm; role?: Role };
  proof: Record<string, unknown>;
};
export type Plan = {
  experiment_id: string;
  registered_at: number;
  spec_sha256: string;
  spec: {
    performance_pairs: number;
    input_tokens: number;
    output_tokens: number;
    declared_changes: string[];
    baseline: { changes: Record<string, unknown> };
    candidate: { changes: Record<string, unknown> };
    [key: string]: unknown;
  };
};
export type Slot = { arm: Arm; role: Role; pair: number };
export type Job = Slot & { run_id: string; state: string; error: string | null; cleanup: string };
export type Report = {
  completeness: string;
  decision: string;
  comparable: boolean;
  reasons: string[];
  adoption_reasons: string[];
  quality_status: string;
  performance: {
    status: string;
    pairs: number;
    independent_processes?: boolean;
    median_relative_e2e?: number;
    interval_95?: [number, number];
  };
  quality: {
    status: string;
    reason?: string;
    baseline_perplexity?: number;
    candidate_perplexity?: number;
    baseline_task_accuracy?: number | null;
    candidate_task_accuracy?: number | null;
    positions?: {
      sample_id: string;
      position: number;
      logit_max_abs_error: number;
      kl_fp16_quant: number;
      nll_increase: number;
      passed: boolean;
      source_logits: unknown;
    }[];
  }[];
  mechanism_gate: string;
  mechanisms: {
    generated_context_equal: boolean;
    interpretation: string;
    differences: {
      metric_id: string;
      label: string | null;
      delta: number;
      unit: string;
      scope: string;
      baseline_source_record_seqs: number[];
      candidate_source_record_seqs: number[];
    }[];
  }[];
  missing_pairs: { role: Role; pair: number }[];
  failed_pairs: { role: Role; pair: number; states: string[] }[];
  observer_cost: unknown;
};
export type Bundle = {
  manifest_sha256: string;
  manifest: {
    status: string;
    publication: string;
    artifacts: { path: string; bytes: number; sha256: string }[];
  };
  receipt: Receipt | null;
  quality: {
    status: string;
    perplexity?: number;
    task_accuracy?: number | null;
    scored_tokens?: number;
    samples: unknown[];
    protocol: unknown;
  } | null;
};
export type Catalogue = {
  layers: Record<string, string>;
  items: {
    id: string;
    layers: number[];
    scope?: string;
    implementation_status: string;
    runtime_capability: string;
  }[];
};
