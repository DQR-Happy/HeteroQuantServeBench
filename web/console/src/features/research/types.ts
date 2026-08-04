export type ResearchArtifact = {
  id: string;
  name: string;
  relative_path: string;
  bytes: number;
  sha256: string;
  format: string;
};
export type HistoricalPhase = {
  id: string;
  sample: string;
  phase: string;
  span_ms: number | null;
  kernel_work_ms: number | null;
  kernel_count: number | null;
  idle_ratio: number | null;
  hotspots: {
    name: string;
    count: number;
    total_ms: number;
    mean_us: number;
    time_share: number;
    ops: string[];
    dims: string[];
    streams: number[];
    grids: unknown[];
    blocks: unknown[];
  }[];
};
export type HistoricalKernel = {
  id: string;
  mode: string;
  sample: string;
  phase: string;
  role: string;
  kernel_regex: string;
  shape: unknown;
  observations: {
    name: string;
    grid: unknown;
    block: unknown;
    metrics: Record<string, number | null>;
    stalls: unknown;
  }[];
  roofline: unknown;
  artifacts: ResearchArtifact[];
};
export type QuantizationMethod = {
  id: string;
  bits: number | null;
  group_size: number | null;
  execution_label: string;
  execution_description: string;
  native_low_bit_kernel: boolean | null;
  storage_sizes: {
    fp16_source_bytes: number | null;
    quantized_bytes: number | null;
    retained_fp16_bytes: number | null;
    whole_model_equivalent_bytes: number | null;
    compression_ratio: number | null;
    qvalues_bytes: number | null;
    scales_bytes: number | null;
    manifest_bytes: number | null;
  };
  quality: {
    passed?: boolean;
    checks?: Record<string, boolean>;
    observed?: unknown;
    thresholds?: unknown;
    ppl_ratio?: number;
  };
  runtime_memory: {
    run_id: string;
    allocated_bytes: number | null;
    reserved_bytes: number | null;
    peak_allocated_bytes: number | null;
  }[];
  performance: unknown;
  artifacts: ResearchArtifact[];
};
export type Research = {
  schema_version: string;
  historical: boolean;
  profiling: {
    status: string;
    source: string;
    verdict: string;
    limitations: string[];
    phases: HistoricalPhase[];
    kernels: HistoricalKernel[];
    perturbation: unknown;
    artifacts: ResearchArtifact[];
  };
  quantization: {
    status: string;
    source: string;
    verdict: string;
    limitations: string[];
    methods: QuantizationMethod[];
  };
  capabilities: { id: string; status: string; detail: string }[];
  artifacts: ResearchArtifact[];
};
