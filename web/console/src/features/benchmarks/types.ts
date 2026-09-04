export type FixedCase = {
  id: string;
  name: string;
  category: string;
  prompt: string;
  scoring: string;
  expected: unknown;
  input_sha256: string;
  messages: { role: string; content: string }[];
};
export type Scheme = {
  id: string;
  name: string;
  kernel_profile: string;
  cleanup_policy: string;
  availability: { loadable: boolean; reason: string | null };
};
export type BenchmarkCatalogue = {
  version: string;
  suite_sha256: string;
  cases: FixedCase[];
  schemes: Scheme[];
  active_id: string | null;
  defaults: {
    baseline_id: string;
    candidate_ids: string[];
    case_ids: string[];
    repeats: number;
    output_tokens: number;
    diagnostics: boolean;
  };
};
export type Measurement = {
  receipt: { status: string | null; effective: { kernel_profile?: string } | null };
  run_id: string;
  scheme_id: string;
  case_id: string;
  role: string;
  repeat: number;
  state: string;
  cleanup: string;
  output: string;
  error: string | null;
  finish_reason: string | null;
  score: boolean | null;
};
export type ResultRow = {
  scheme_id: string;
  case_id: string;
  completed: number;
  expected: number;
  metrics: Record<string, number | null>;
  task_accuracy: number | null;
  scored_samples: number;
  token_agreement: number | null;
  text_equal_rate: number | null;
  comparable: boolean;
  reasons: string[];
  relative_e2e: number | null;
  runs: string[];
};
export type Benchmark = {
  id: string;
  state: string;
  phase: string;
  created_at: number;
  updated_at: number;
  plan_sha256: string;
  current_run_id: string | null;
  error: string | null;
  restore_error: string | null;
  plan: {
    baseline_id: string;
    schemes: Scheme[];
    cases: FixedCase[];
    output_tokens: number;
    repeats: number;
    diagnostics: boolean;
    suite_version: string;
    warmups_per_case: number;
  };
  records: Measurement[];
  report: { rows: ResultRow[]; performance_scope: string; timing_scope: string };
};
export type BenchmarkList = {
  items: { id: string; state: string; phase: string; created_at: number; error: string | null }[];
  total: number;
};
export const finished = (state: string) =>
  ['completed', 'failed', 'cancelled', 'interrupted'].includes(state);
