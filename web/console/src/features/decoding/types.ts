import type { Run } from '../../api/types';
export type DecodePlan = {
  method: 'autoregressive' | 'prompt_lookup' | 'assistant';
  sampling: 'greedy' | 'multinomial';
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  min_new_tokens: number;
  seed: number;
  draft_tokens: number;
  draft_deployment_id?: string;
  constraint_backend: 'mask' | 'gather';
};
export const defaults: DecodePlan = {
  method: 'autoregressive',
  sampling: 'greedy',
  temperature: 1,
  top_k: 0,
  top_p: 1,
  repetition_penalty: 1,
  min_new_tokens: 0,
  seed: 42,
  draft_tokens: 4,
  constraint_backend: 'gather',
};
export type DecodeReceipt = {
  status: string;
  plan: Record<string, unknown>;
  statistics: Record<string, unknown>;
  constraint_satisfied?: boolean | null;
  resident_draft?: Record<string, unknown>;
  cleanup?: Record<string, unknown>;
  [key: string]: unknown;
};
export type DecodeRun = Run & { metrics: Run['metrics'] & { decoding_receipt?: DecodeReceipt } };
export type DecodeComparison = {
  comparable: boolean;
  reasons: string[];
  quality: {
    status: string;
    exact_tokens: boolean;
    constraint_satisfied: (boolean | null)[];
    ppl_not_applicable: string;
  };
  decoding: DecodeReceipt[];
  performance: Record<string, unknown>;
};
