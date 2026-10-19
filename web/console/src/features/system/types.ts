import type { components } from '../../api/generated';
import type { Run } from '../../api/types';
export type SystemPlan = components['schemas']['SystemPlan'];
export type SystemProfile = components['schemas']['SystemProfile'];
export type Hardware = {
  allowed_cpus: number[];
  cpu_quota: string | null;
  numa_policy: { available: boolean; reason: string };
  numa_nodes: { id: number; cpulist: string; distance: string }[];
  devices: {
    id: number;
    uuid: string;
    name: string;
    total_bytes: number;
    power_limit_w?: number;
    power_constraints_w: number[] | null;
    power_reason: string;
  }[];
  quarantined: boolean;
  recovery_reason: string;
  limitations: string[];
};
export type Trial = {
  trial: number;
  label: string;
  requested: SystemProfile;
  effective: {
    cpu_ids: number[];
    intra_threads: number;
    interop_threads: number;
    power_limit_w?: number;
  };
  metrics: Record<string, number | null>;
  requests: { index: number; token_ids: number[]; output: string; token_s: number }[];
  samples: {
    time_s: number;
    sm_clock_mhz: number | null;
    power_w: number | null;
    temperature_c: number | null;
  }[];
};
export type SystemReceipt = {
  run_id: string;
  status: string;
  cleanup: string;
  plan: SystemPlan;
  trials: Trial[];
  comparison: {
    complete_pair?: boolean;
    greedy_tokens_exact: boolean;
    metrics: Record<
      string,
      {
        baseline: number | null;
        candidate: number | null;
        delta: number | null;
        delta_percent: number | null;
      }
    >;
  };
};
export type SystemRun = Run & {
  progress?: Run['progress'] & { trial?: number; label?: string; phase?: string };
  metrics: Run['metrics'] & { system_receipt?: SystemReceipt };
};
