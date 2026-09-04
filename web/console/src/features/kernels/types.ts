export type KernelPlan = {
  profile_id: string;
  operators: Record<string, string>;
  profile_sha256?: string;
  native_artifact?: { manifest_sha256: string; library_sha256: string };
};
export type KernelProfile = KernelPlan & {
  family: string;
  label: string;
  requires_native_artifact: boolean;
  availability?: { loadable: boolean; reason: string | null };
};
export type KernelCatalogue = {
  profiles: KernelProfile[];
  native_artifact?: {
    status: string;
    reason: string | null;
    manifest_sha256?: string;
    library_sha256?: string;
  };
};
export type KernelSite = {
  site_id: string;
  op: string;
  variant: string;
  calls: { warmup: number; prefill: number; decode: number };
};
export type KernelSummary = {
  profile_id: string;
  status: string;
  profile_sha256: string;
  kernel_plan_sha256: string;
  binding_manifest_sha256: string;
  sites: KernelSite[];
  not_exercised: string[];
  fallbacks: unknown[];
  proof_scope: string;
  gpu_observed: string;
  workspace_bytes: number;
  weight_packing_bytes: number;
};
