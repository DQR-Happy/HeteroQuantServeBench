export interface Capacity {
  gpu_total_bytes: number | null;
  gpu_name: string;
  host_available_bytes: number | null;
  disk_free_bytes: number;
  download_directory: string;
}
export interface HubConfig {
  publishers: { id: string; name: string; provenance: string }[];
  capacity: Capacity;
  disk_reserve_bytes: number;
  runtime: string;
  scope: string;
}
export interface ModelSummary {
  model_id: string;
  name: string;
  provenance: string;
  architectures: string[];
  parameters: number | null;
  weight_bytes: number | null;
  license: string;
  url: string;
  resource_candidate: boolean;
  candidate_status?: 'estimated_fit' | 'needs_inspection' | 'excluded';
  estimate: { estimated_runtime_bytes: number | null };
}
export interface HubCatalogue {
  publisher: string;
  state: string;
  items: ModelSummary[];
  indexed: number;
  total: number;
  total_upstream: number | null;
  scanned?: number;
  duplicates?: number;
  retained_cached?: boolean;
  error: string | null;
}
export interface Inspection {
  plan_sha256: string;
  plan: {
    model_id: string;
    revision: string;
    input_mode?: 'chat' | 'completion';
    chat_template_available?: boolean;
    runtime_precision?: 'float16' | 'bfloat16';
    download_bytes: number;
    weight_bytes: number;
    runtime_reasons: string[];
    config: { model_type: string };
    files: { Path: string; Size: number; Sha256: string; Revision: string }[];
    license_url: string;
    provenance: string;
  };
  admission: {
    eligible: boolean;
    reasons: string[];
    estimated_runtime_bytes: number | null;
    kv_bytes: number;
    limitations: string[];
  };
}
export interface DownloadJob {
  id: string;
  model_id: string;
  revision: string;
  state: string;
  downloaded_bytes: number;
  total_bytes: number;
  verified_files: number;
  total_files: number;
  current_file: string | null;
  bytes_per_second: number | null;
  error: string | null;
  deployment_id: string | null;
  snapshot_path?: string;
  plan_sha256: string;
  attempt: number;
}
