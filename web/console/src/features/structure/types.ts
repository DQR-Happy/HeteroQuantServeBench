import type { Run } from '../../api/types';
export type Plan = {
  method: string;
  targets: string[];
  max_modules: number;
  sparsity: number;
  pattern: string;
  keep_ratio: number;
  calibration?: string[];
  calibration_max_tokens: number;
};
export type Artifact = {
  artifact_id: string;
  creation_run_id: string;
  recipe: Plan;
  statistics: {
    parameters_before: number;
    parameters_after: number;
    parameter_reduction: number;
    same_dtype_parameter_bytes_before: number;
    same_dtype_parameter_bytes_after: number;
    export_dtype: string;
    sparse_kernel: boolean;
  };
  transforms: Record<string, unknown>[];
  disk_bytes: number;
  creation_seconds: number;
  publication: string;
};
export type StructureRun = Run & {
  progress?: unknown;
  metrics: Run['metrics'] & {
    structure_artifact?: Artifact;
    optimization_receipt?: Run['metrics']['optimization_receipt'] & {
      structure_receipt?: Record<string, unknown>;
    };
  };
};
