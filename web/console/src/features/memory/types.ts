export type MemoryValues = {
  allocated_bytes?: number | null;
  reserved_bytes?: number | null;
  peak_allocated_bytes?: number | null;
  scope?: string;
};

export type MemoryObservation = {
  sampled_at?: number | null;
  run_id?: string;
  memory?: MemoryValues | null;
};

export type Resources = {
  schema_version: number;
  sampled_at: number;
  host: {
    total_bytes: number | null;
    available_bytes: number | null;
    free_bytes: number | null;
    cached_bytes: number | null;
    swap_used_bytes: number | null;
    memory_scope: string;
  };
  processes: {
    pid: number;
    role: string;
    rss_bytes: number | null;
    pss_bytes?: number | null;
    availability: string;
  }[];
  deployments: {
    id: string;
    state: string;
    load_observation?: MemoryObservation | null;
    latest_observation?: MemoryObservation | null;
    lifecycle?: {
      label: string;
      sampled_at: number;
      total_bytes?: number | null;
      available_bytes?: number | null;
    }[];
  }[];
  limitations: string[];
};
