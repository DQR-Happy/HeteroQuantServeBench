export interface ArchitectureNode {
  id: string;
  label: string;
  column: number;
  row: number;
  module: string | null;
  kind: string;
  note: string;
}
export interface ArchitectureGraph {
  nodes: ArchitectureNode[];
  edges: { source: string; target: string; kind: string }[];
}
export interface Architecture {
  snapshot_sha256: string;
  observed_at: number;
  deployment_id: string;
  model: string;
  name: string;
  status: 'verified_static' | 'metadata_only';
  config: Record<string, string | number | boolean | string[] | null>;
  config_sha256: string;
  implementation: {
    verified: boolean;
    version?: string;
    file?: string;
    sha256?: string;
    reason: string | null;
  };
  runtime: {
    state: string;
    epoch: string | null;
    load_run_id: string | null;
    kernel_profile: string | null;
    model_class: string | null;
    weight_verification: string | null;
    loaded_config_matches: boolean | null;
  };
  layers: {
    index: number;
    path: string;
    tensor_count: number;
    stored_bytes: number;
    attention_type: string;
  }[];
  tensors: { name: string; shape: number[]; dtype: string; stored_bytes: number; file: string }[];
  stored_tensor_count: number;
  stored_parameter_elements: number;
  stored_bytes: number;
  dtypes: Record<string, number>;
  graphs: Record<string, ArchitectureGraph> | null;
  shards: { file: string; header_sha256: string; file_bytes: number }[];
  issues: string[];
  limitations: string[];
  evidence_scope: string;
}
