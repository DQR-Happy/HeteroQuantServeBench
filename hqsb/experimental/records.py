"""S14 vocabularies, state machines and table schemas.

This module is the single place where the *words* of S14 live, so that the
thirteen experiment modules, the driver and the reports cannot drift into three
spellings of the same status.

Sources (all verbatim from the protocol tree, none invented):

* ``docs/stage_experiments/README.md`` §3 — the seven experiment statuses;
* ``docs/stage_experiments/details/S14/README.md`` §6 — the nine maturity
  levels; §7.7 — the nine ``AdoptionDecision`` values; §21 — the twelve
  failure/terminal statuses; §18.3 — the experiment unit; §24 — the profiling
  and resource-ledger observables;
* ``details/S14/E14-01_*.md`` §3.2 — the capability states and §4 — the reason
  codes;
* ``details/S14/E14-05_*.md`` §4 — the four frontier branches.

Nothing here measures anything: the tables describe *columns*, and every table
ships with ``measured: false`` semantics — the driver refuses to write a
conclusion into them without ``--execute``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError


# ── experiment statuses (手册 §3 + S14 README §21) ─────────────────────────

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_PASS = "PASS"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_NOT_RUN = "NOT_RUN"
STATUS_BLOCKED_PREREQUISITE = "BLOCKED_PREREQUISITE"
STATUS_BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
STATUS_INVALID_PROTOCOL = "INVALID_PROTOCOL"
STATUS_INVALID_IDENTITY = "INVALID_IDENTITY"
STATUS_FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
STATUS_FAIL_QUALITY = "FAIL_QUALITY"
STATUS_FAIL_PERFORMANCE_HYPOTHESIS = "FAIL_PERFORMANCE_HYPOTHESIS"
STATUS_FAIL_RECOVERY = "FAIL_RECOVERY"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

#: The seven statuses the manual (``README.md`` §3) allows in a checklist.
MANUAL_STATUSES: Tuple[str, ...] = (
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

#: The twelve terminal/blocking statuses of ``details/S14/README.md`` §21.
S14_FAILURE_STATUSES: Tuple[str, ...] = (
    STATUS_NOT_RUN,
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_BLOCKED_CAPABILITY,
    STATUS_INVALID_PROTOCOL,
    STATUS_INVALID_IDENTITY,
    STATUS_FAIL_CORRECTNESS,
    STATUS_FAIL_QUALITY,
    STATUS_FAIL_PERFORMANCE_HYPOTHESIS,
    STATUS_FAIL_RECOVERY,
    STATUS_PASS_NEGATIVE,
    STATUS_PASS,
    STATUS_N_A_BY_ADR,
)

ALL_STATUSES: Tuple[str, ...] = MANUAL_STATUSES + (STATUS_NOT_RUN,) + (
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_BLOCKED_CAPABILITY,
    STATUS_INVALID_PROTOCOL,
    STATUS_INVALID_IDENTITY,
    STATUS_FAIL_CORRECTNESS,
    STATUS_FAIL_QUALITY,
    STATUS_FAIL_PERFORMANCE_HYPOTHESIS,
    STATUS_FAIL_RECOVERY,
)

#: Only these may ever be written as a *conclusion* by the scaffolding.
CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_FAIL_PERFORMANCE_HYPOTHESIS,
)

#: Statuses that are always allowed and never claim a measured result.
NON_CONCLUSION_STATUSES: Tuple[str, ...] = tuple(
    status for status in ALL_STATUSES if status not in CONCLUSION_STATUSES
)

#: ``details/S14/README.md`` §21: these categories may **not** be replaced by a
#: negative result (mirrors the manual §3 rule for safety/correctness gates).
NEGATIVE_NOT_ALLOWED_FOR: Tuple[str, ...] = (
    "artifact-compatibility",
    "data-isolation",
    "quality-gate",
    "correctness",
    "safety",
    "semantic-parity",
)

# ── maturity levels (S14 README §6) ────────────────────────────────────────

MATURITY_DESIGN_ONLY = "DESIGN_ONLY"
MATURITY_SOURCE_INTEGRATED = "SOURCE_INTEGRATED"
MATURITY_SEMANTIC_VERIFIED = "SEMANTIC_VERIFIED"
MATURITY_QUALITY_VERIFIED = "QUALITY_VERIFIED"
MATURITY_KERNEL_PROFILED = "KERNEL_PROFILED"
MATURITY_RUNTIME_PROFILED = "RUNTIME_PROFILED"
MATURITY_SERVICE_PROFILED = "SERVICE_PROFILED"
MATURITY_ADOPTED_EXPERIMENTAL = "ADOPTED_EXPERIMENTAL"
MATURITY_ADOPTED_CORE = "ADOPTED_CORE"

#: Ordered from the weakest evidence to the strongest.
MATURITY_LEVELS: Tuple[str, ...] = (
    MATURITY_DESIGN_ONLY,
    MATURITY_SOURCE_INTEGRATED,
    MATURITY_SEMANTIC_VERIFIED,
    MATURITY_QUALITY_VERIFIED,
    MATURITY_KERNEL_PROFILED,
    MATURITY_RUNTIME_PROFILED,
    MATURITY_SERVICE_PROFILED,
    MATURITY_ADOPTED_EXPERIMENTAL,
    MATURITY_ADOPTED_CORE,
)

#: What each level is allowed to claim verbatim (S14 README §6 table).
MATURITY_CLAIM: Mapping[str, str] = {
    MATURITY_DESIGN_ONLY: "已完成实验设计",
    MATURITY_SOURCE_INTEGRATED: "已实现实验性路径",
    MATURITY_SEMANTIC_VERIFIED: "在给定矩阵内语义一致",
    MATURITY_QUALITY_VERIFIED: "质量变化在预注册边界内",
    MATURITY_KERNEL_PROFILED: "瓶颈位于……，实际路径为……",
    MATURITY_RUNTIME_PROFILED: "Runtime 层观察到……",
    MATURITY_SERVICE_PROFILED: "服务口径下观察到……",
    MATURITY_ADOPTED_EXPERIMENTAL: "进入 experimental 能力",
    MATURITY_ADOPTED_CORE: "成为受支持能力",
}


def maturity_rank(level: str) -> int:
    if level not in MATURITY_LEVELS:
        raise ConfigError(f"unknown S14 maturity level {level!r}; known: {', '.join(MATURITY_LEVELS)}")
    return MATURITY_LEVELS.index(level)


def maturity_is_lower_than(level: str, other: str) -> bool:
    """True when ``level`` claims less than ``other`` (the report's downgrade rule)."""
    return maturity_rank(level) < maturity_rank(other)


# ── adoption decisions (S14 README §7.7) ───────────────────────────────────

ADOPT_CORE = "ADOPT_CORE"
ADOPT_EXPERIMENTAL = "ADOPT_EXPERIMENTAL"
RESEARCH_ONLY = "RESEARCH_ONLY"
REJECT_NO_BENEFIT = "REJECT_NO_BENEFIT"
REJECT_QUALITY = "REJECT_QUALITY"
REJECT_COMPLEXITY = "REJECT_COMPLEXITY"
REJECT_PORTABILITY = "REJECT_PORTABILITY"
BLOCKED_EVIDENCE = "BLOCKED_EVIDENCE"
N_A_BY_ADR = "N/A_BY_ADR"

ADOPTION_DECISIONS: Tuple[str, ...] = (
    ADOPT_CORE,
    ADOPT_EXPERIMENTAL,
    RESEARCH_ONLY,
    REJECT_NO_BENEFIT,
    REJECT_QUALITY,
    REJECT_COMPLEXITY,
    REJECT_PORTABILITY,
    BLOCKED_EVIDENCE,
    N_A_BY_ADR,
)

#: Decisions that do **not** allow any capability claim.
NON_ADOPTION_DECISIONS: Tuple[str, ...] = (
    RESEARCH_ONLY,
    REJECT_NO_BENEFIT,
    REJECT_QUALITY,
    REJECT_COMPLEXITY,
    REJECT_PORTABILITY,
    BLOCKED_EVIDENCE,
    N_A_BY_ADR,
)

# ── capability states (E14-01 §3.2) ────────────────────────────────────────

CAP_AVAILABLE = "AVAILABLE"
CAP_DISABLED_BY_CONFIG = "DISABLED_BY_CONFIG"
CAP_UNAVAILABLE_DEPENDENCY = "UNAVAILABLE_DEPENDENCY"
CAP_UNAVAILABLE_CAPABILITY = "UNAVAILABLE_CAPABILITY"
CAP_ABI_MISMATCH = "ABI_MISMATCH"
CAP_DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
CAP_BUG = "BUG"

CAPABILITY_STATES: Tuple[str, ...] = (
    CAP_AVAILABLE,
    CAP_DISABLED_BY_CONFIG,
    CAP_UNAVAILABLE_DEPENDENCY,
    CAP_UNAVAILABLE_CAPABILITY,
    CAP_ABI_MISMATCH,
    CAP_DEVICE_UNAVAILABLE,
    CAP_BUG,
)

#: States that mean "not this machine": the corresponding experiment is
#: ``BLOCKED_CAPABILITY``/``BLOCKED_PREREQUISITE``, never ``FAIL`` and never
#: ``PASS`` (``details/S14/README.md`` §21: 缺硬件/OOM 未执行不能写成负结论).
BLOCKING_CAPABILITY_STATES: Tuple[str, ...] = (
    CAP_UNAVAILABLE_DEPENDENCY,
    CAP_ABI_MISMATCH,
    CAP_DEVICE_UNAVAILABLE,
)

#: Reason codes recorded on every fallback/auto-selection (E14-01 step 4).
REASON_CODES: Tuple[str, ...] = (
    "DISABLED_BY_CONFIG",
    "DEPENDENCY_ABSENT",
    "VERSION_UNSUPPORTED",
    "ABI_MISMATCH",
    "DEVICE_UNAVAILABLE",
    "SHAPE_UNSUPPORTED",
    "DTYPE_UNSUPPORTED",
    "PATTERN_UNSUPPORTED",
    "CAPACITY_EXCEEDED",
    "QUALITY_GATE_FAILED",
    "IMPLEMENTATION_ERROR",
    "NOT_IMPLEMENTED",
    "UNKNOWN_FEATURE_FLAG",
    "CONFLICTING_FEATURE_FLAGS",
)

# ── dependency / install profiles (E14-01) ─────────────────────────────────

#: ``details/S14/README.md`` §20: core install plus named extras.
INSTALL_PROFILES: Tuple[str, ...] = (
    "core",
    "train",
    "rl",
    "frontier",
    "multimodal",
    "agent",
    "edge",
    "all",
)

#: Extra name → the feature flags it admits (frozen mapping, audited by
#: ``hqsb.experimental.specs``; the drift check runs in ``--spec-check``).
EXTRA_FEATURE_MAPPING: Mapping[str, Tuple[str, ...]] = {
    "train": ("train.distributed", "train.checkpoint"),
    "rl": ("rl.rollout", "rl.reward"),
    "frontier": ("frontier.speculative", "frontier.moe", "frontier.long_context", "frontier.sparse"),
    "multimodal": ("multimodal.vlm", "multimodal.diffusion", "multimodal.audio"),
    "agent": ("agent.workflow", "agent.tools", "agent.memory"),
    "edge": ("edge.arm", "edge.mobile_runtime"),
}

#: ``E14-01`` §10 verdict per extra.
DEPENDENCY_VERDICTS: Tuple[str, ...] = ("CORE_SAFE", "EXPERIMENTAL_ONLY", "BLOCKED")

# ── frontier branches (E14-05 §4) ──────────────────────────────────────────

FRONTIER_BRANCH_NAMES: Mapping[str, str] = {
    "E14-F1": "Speculative/MTP",
    "E14-F2": "MoE",
    "E14-F3": "Long context",
    "E14-F4": "Sparsity",
}

#: The layer each branch must have evidence for (E14-05 §4 / §17).
FRONTIER_REQUIRED_LAYERS: Mapping[str, Tuple[str, ...]] = {
    "E14-F1": ("algorithm", "kernel", "runtime", "service"),
    "E14-F2": ("routing", "collective", "kernel", "runtime", "service"),
    "E14-F3": ("kv", "kernel", "runtime", "service"),
    "E14-F4": ("representation", "kernel", "model", "service"),
}

#: ``E14-05`` §4 risk column, kept as data so the ADR cannot forget it.
FRONTIER_PRIMARY_RISKS: Mapping[str, str] = {
    "E14-F1": "acceptance 低、额外算力/KV/调度",
    "E14-F2": "router skew、All-to-All、总显存",
    "E14-F3": "长文质量、TTFT、混合负载饿死",
    "E14-F4": "无真实 sparse kernel、索引/转换开销",
}

# ── profiling layers (S14 README §17) ─────────────────────────────────────

PROFILE_LAYERS: Tuple[str, ...] = (
    "algorithm_data",
    "operator_kernel",
    "runtime",
    "distributed",
    "service",
    "production",
)

#: Observables that must be present once a layer is claimed (S14 README §17).
PROFILE_LAYER_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "algorithm_data": ("sample_ids", "versions", "statistics", "quality"),
    "operator_kernel": ("shape", "dtype", "actual_kernel", "time", "bytes_or_flops", "counters"),
    "runtime": ("request_or_iteration", "queue", "memory", "fallback", "state"),
    "distributed": ("bytes", "peer", "overlap", "imbalance", "straggler"),
    "service": ("ttft", "tpot", "p99", "reject", "cost", "energy"),
    "production": ("digest", "readiness", "failure_recovery", "audit"),
}

#: Measurement strata (S14 README §18.3): what one independent repetition is.
EXPERIMENT_UNITS: Mapping[str, str] = {
    "training": "独立 run/seed/time block（step 不是独立重复）",
    "conversion": "独立 source checkpoint × conversion invocation",
    "rollout": "独立 prompt/group/trajectory",
    "runtime_service": "独立 workload episode/time block",
    "distributed": "独立 rank group/topology/run",
    "edge_energy": "独立 thermal/power steady episode",
}

# ── state machines ────────────────────────────────────────────────────────

#: ``E14-03`` §6 step 3: one conversion node advances through these states.
CONVERSION_NODE_STATES: Tuple[str, ...] = (
    "PINNED",
    "INPUTS_VERIFIED",
    "EXECUTED",
    "VALIDATED",
    "PUBLISHED",
    "REJECTED",
)

CONVERSION_NODE_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "PINNED": ("INPUTS_VERIFIED", "REJECTED"),
    "INPUTS_VERIFIED": ("EXECUTED", "REJECTED"),
    "EXECUTED": ("VALIDATED", "REJECTED"),
    "VALIDATED": ("PUBLISHED", "REJECTED"),
    "PUBLISHED": (),
    "REJECTED": (),
}

#: ``E14-04`` step 9: the synchronous post-training state machine.
POSTTRAINING_SYNC_STATES: Tuple[str, ...] = (
    "GENERATE",
    "SCORE",
    "VALIDATE",
    "TRAIN",
    "PUBLISH",
    "ACTIVATE",
    "COMPLETED",
    "ABORTED",
)

POSTTRAINING_SYNC_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "GENERATE": ("SCORE", "ABORTED"),
    "SCORE": ("VALIDATE", "ABORTED"),
    "VALIDATE": ("TRAIN", "ABORTED"),
    "TRAIN": ("PUBLISH", "ABORTED"),
    "PUBLISH": ("ACTIVATE", "ABORTED"),
    "ACTIVATE": ("COMPLETED", "ABORTED"),
    "COMPLETED": (),
    "ABORTED": (),
}

#: ``E14-07`` §4.1: the agent workflow state machine.
WORKFLOW_STATES: Tuple[str, ...] = (
    "CREATED",
    "MODEL_DECISION",
    "TOOL_DISPATCHED",
    "TOOL_RUNNING",
    "TOOL_RESULT_VALIDATED",
    "MEMORY_UPDATED",
    "NEXT_DECISION",
    "RETRY_WAIT",
    "CANCELLED",
    "TIMED_OUT",
    "FAILED",
    "COMPENSATING",
    "COMPLETED",
)

WORKFLOW_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "CREATED": ("MODEL_DECISION", "CANCELLED", "FAILED"),
    "MODEL_DECISION": ("TOOL_DISPATCHED", "COMPLETED", "CANCELLED", "FAILED"),
    "TOOL_DISPATCHED": ("TOOL_RUNNING", "RETRY_WAIT", "TIMED_OUT", "CANCELLED", "FAILED"),
    "TOOL_RUNNING": ("TOOL_RESULT_VALIDATED", "RETRY_WAIT", "TIMED_OUT", "CANCELLED", "FAILED"),
    "TOOL_RESULT_VALIDATED": ("MEMORY_UPDATED", "FAILED", "CANCELLED"),
    "MEMORY_UPDATED": ("NEXT_DECISION", "FAILED", "CANCELLED"),
    "NEXT_DECISION": ("MODEL_DECISION", "COMPLETED", "FAILED", "CANCELLED"),
    "RETRY_WAIT": ("TOOL_DISPATCHED", "FAILED", "TIMED_OUT", "CANCELLED"),
    "CANCELLED": (),
    "TIMED_OUT": ("COMPENSATING", "FAILED"),
    "FAILED": ("COMPENSATING",),
    "COMPENSATING": ("COMPLETED", "FAILED"),
    "COMPLETED": (),
}

#: ``E14-F1`` §6 step 17 — one speculation cycle's terminal states.
SPECULATION_CYCLE_STATES: Tuple[str, ...] = (
    "PROPOSED",
    "VERIFIED",
    "COMMITTED",
    "ROLLED_BACK",
    "CANCELLED",
    "FAILED",
)

SPECULATION_CYCLE_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "PROPOSED": ("VERIFIED", "CANCELLED", "FAILED"),
    "VERIFIED": ("COMMITTED", "ROLLED_BACK", "CANCELLED", "FAILED"),
    "COMMITTED": (),
    "ROLLED_BACK": (),
    "CANCELLED": (),
    "FAILED": (),
}

#: ``E14-08`` §4.2 — the three timing regimes that must never be mixed.
EDGE_MODES: Tuple[str, ...] = ("cold", "warm", "sustained")

#: ``E14-08`` §7 step 40 — the two evidence levels the experiment separates.
EDGE_EVIDENCE_LEVELS: Tuple[str, ...] = ("MAP_ONLY", "DEVICE_MEASURED")

#: ``E14-08`` §2 — the two legitimate completion routes.
EDGE_ROUTES: Tuple[str, ...] = ("ROUTE_A_DEVICE_ADAPTER", "ROUTE_B_TECHNOLOGY_MAP")

#: ``E14-04`` §1 — the three post-training algorithms; exactly one is primary.
POSTTRAINING_ALGORITHMS: Tuple[str, ...] = ("sft", "dpo", "grpo")

#: ``E14-F4`` §2 — route A/B must be chosen; the two cannot be merged.
SPARSITY_ROUTES: Tuple[str, ...] = ("ROUTE_A_STRUCTURED_WEIGHT", "ROUTE_B_SPARSE_ATTENTION")

#: ``E14-06`` §1 — one modality only.
MULTIMODAL_MODALITIES: Tuple[str, ...] = ("vlm", "diffusion", "audio")

#: ``E14-04`` step 10 — async queue/lease semantics that must be declared.
CONSUMPTION_SEMANTICS: Tuple[str, ...] = ("exactly-once", "at-least-once", "at-most-once", "unknown")

STATE_MACHINES: Mapping[str, Tuple[Tuple[str, ...], Mapping[str, Tuple[str, ...]]]] = {
    "conversion_node": (CONVERSION_NODE_STATES, CONVERSION_NODE_TRANSITIONS),
    "posttraining_sync": (POSTTRAINING_SYNC_STATES, POSTTRAINING_SYNC_TRANSITIONS),
    "agent_workflow": (WORKFLOW_STATES, WORKFLOW_TRANSITIONS),
    "speculation_cycle": (SPECULATION_CYCLE_STATES, SPECULATION_CYCLE_TRANSITIONS),
}


def validate_state_machines() -> List[str]:
    """Return every structural problem in the declared state machines.

    A machine is well formed when every transition target is a declared state,
    every non-terminal state has at least one outgoing transition, and every
    terminal state has none.  The audit fails loudly instead of ignoring a typo,
    because a mis-declared terminal state silently accepts an impossible run.
    """
    problems: List[str] = []
    for name, (states, transitions) in STATE_MACHINES.items():
        if len(set(states)) != len(states):
            problems.append(f"{name}: duplicate states")
        for state in states:
            if state not in transitions:
                problems.append(f"{name}: state {state} has no transition entry")
        for state, targets in transitions.items():
            if state not in states:
                problems.append(f"{name}: transition from unknown state {state}")
            for target in targets:
                if target not in states:
                    problems.append(f"{name}: unknown transition target {target!r} from {state}")
    return problems


def is_valid_transition(machine: str, old_state: str, new_state: str) -> bool:
    if machine not in STATE_MACHINES:
        raise ConfigError(f"unknown state machine {machine!r}")
    _states, transitions = STATE_MACHINES[machine]
    return new_state in transitions.get(old_state, ())


# ── table schemas (S14 README §22/§23 raw + normalized outputs) ────────────

#: Column contracts for the raw/normalized tables S14 experiments must emit.
#: ``stage`` records which experiment owns the table, so an orphan table in a
#: run directory is detectable.
TABLE_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "dependency_trees": {
        "stage": "E14-01",
        "columns": ("install_profile", "package", "version", "requested_by", "license", "origin"),
    },
    "side_effect_audit": {
        "stage": "E14-01",
        "columns": ("run_id", "category", "name", "observed", "detail"),
    },
    "capability_matrix": {
        "stage": "E14-01",
        "columns": ("capability", "state", "reason_code", "requested_impl", "actual_impl", "detail"),
    },
    "feature_flag_matrix": {
        "stage": "E14-01",
        "columns": ("flag", "value", "valid", "registry_diff", "config_digest", "capability_state"),
    },
    "distributed_steps": {
        "stage": "E14-02",
        "columns": (
            "training_run_id",
            "step",
            "rank",
            "world_size",
            "sample_ids",
            "effective_token_count",
            "loss_sum",
            "loss_denominator",
            "grad_norm",
            "lr",
            "scaler",
            "timing_ms",
            "collective_bytes",
            "memory_bytes",
            "parameter_digest_after",
            "checkpoint_id",
            "status",
        ),
    },
    "collective_ledger": {
        "stage": "E14-02",
        "columns": (
            "training_run_id",
            "step",
            "rank",
            "collective_type",
            "shape",
            "dtype",
            "bytes",
            "calls",
            "stream",
            "start_ns",
            "end_ns",
            "overlapped",
        ),
    },
    "memory_timeline": {
        "stage": "E14-02",
        "columns": ("training_run_id", "step", "rank", "phase", "allocated_bytes", "reserved_bytes",
                    "peak_bytes", "host_bytes", "pinned_bytes", "category"),
    },
    "resume_comparison": {
        "stage": "E14-02",
        "columns": ("training_run_id", "step_after_resume", "quantity", "control", "resumed", "abs_diff",
                    "rel_diff", "tolerance", "within"),
    },
    "tensor_mapping": {
        "stage": "E14-03",
        "columns": ("conversion_graph_id", "source_name", "target_name", "axis", "slice", "permute",
                    "cast", "shape_before", "shape_after", "source_digest"),
    },
    "tensor_diffs": {
        "stage": "E14-03",
        "columns": ("conversion_graph_id", "layer", "tensor", "shape", "dtype", "max_abs", "max_rel",
                    "cosine", "ulp_max", "finite", "within"),
    },
    "token_parity": {
        "stage": "E14-03",
        "columns": ("conversion_graph_id", "prompt_id", "step", "reference_token", "candidate_token",
                    "match", "top1_margin", "first_divergence"),
    },
    "trajectory_records": {
        "stage": "E14-04",
        "columns": ("trajectory_id", "sample_id", "policy_snapshot_id", "reference_snapshot_id",
                    "reward_definition_id", "environment_version", "prompt_token_hash",
                    "response_token_hash", "sampling_config_hash", "termination_reason",
                    "produced_at_step", "consumed_at_step", "staleness_steps", "status"),
    },
    "reward_records": {
        "stage": "E14-04",
        "columns": ("trajectory_id", "reward_definition_id", "attempt", "components", "total",
                    "deterministic", "timeout", "error"),
    },
    "consumed_batch_ledger": {
        "stage": "E14-04",
        "columns": ("training_step", "policy_snapshot_ids", "trajectory_ids", "staleness_max",
                    "mixed_version", "duplicate_consumed"),
    },
    "queue_timeline": {
        "stage": "E14-04",
        "columns": ("stage", "queue_name", "t_ns", "depth", "oldest_age_ms", "capacity", "rejected"),
    },
    "staleness_results": {
        "stage": "E14-04",
        "columns": ("config_id", "staleness_steps", "utilization", "wall_time_s", "kl", "ratio_mean",
                    "quality_metric", "quality_value", "drop_rate"),
    },
    "speculation_cycles": {
        "stage": "E14-F1",
        "columns": ("request_id", "cycle_index", "target_artifact_id", "proposer_artifact_id",
                    "proposed_token_ids", "accepted_count", "reject_position", "committed_token_ids",
                    "draft_time_ms", "verify_time_ms", "rollback_time_ms", "target_calls",
                    "kv_written_bytes", "kv_discarded_bytes", "actual_backend", "status"),
    },
    "acceptance_by_position": {
        "stage": "E14-F1",
        "columns": ("config_id", "position", "proposed", "accepted", "acceptance", "domain", "entropy_bin"),
    },
    "routing_events": {
        "stage": "E14-F2",
        "columns": ("request_id", "layer", "router_version", "top_k", "tokens_per_expert", "expert_to_rank",
                    "dropped_tokens", "rerouted_tokens", "dispatch_bytes_per_peer", "dispatch_ms",
                    "expert_compute_ms_per_rank", "combine_ms", "slowest_rank", "actual_kernels", "status"),
    },
    "expert_load_statistics": {
        "stage": "E14-F2",
        "columns": ("layer", "domain", "window", "expert", "token_count", "cv", "gini", "max_mean",
                    "padding_efficiency"),
    },
    "alltoall_ledger": {
        "stage": "E14-F2",
        "columns": ("layer", "rank", "peer", "send_count", "recv_count", "bytes", "dtype", "padding",
                    "start_ns", "end_ns", "stream"),
    },
    "expert_gemm_ledger": {
        "stage": "E14-F2",
        "columns": ("layer", "expert", "rank", "M", "N", "K", "kernel", "padding_tokens", "time_ms",
                    "occupancy", "throughput_tflops"),
    },
    "input_token_ledger": {
        "stage": "E14-F3",
        "columns": ("episode_id", "document_id", "token_ids_digest", "submitted_tokens", "needle_positions",
                    "source", "license"),
    },
    "kv_allocator_timeline": {
        "stage": "E14-F3",
        "columns": ("episode_id", "t_ns", "logical_bytes", "physical_bytes", "metadata_bytes", "blocks",
                    "free_blocks", "fragmentation", "oom_margin_bytes"),
    },
    "context_sweep": {
        "stage": "E14-F3",
        "columns": ("episode_id", "model_artifact_id", "candidate_config_id", "submitted_tokens",
                    "accepted_tokens", "processed_tokens", "truncated_tokens", "output_tokens",
                    "kv_logical_bytes", "kv_physical_bytes", "kv_metadata_bytes", "prefill_ms", "ttft_ms",
                    "tpot_ms", "actual_backend", "status"),
    },
    "quality_by_length_position": {
        "stage": "E14-F3",
        "columns": ("episode_id", "length_bucket", "needle_depth", "task", "metric", "reference", "candidate",
                    "delta"),
    },
    "pattern_compliance": {
        "stage": "E14-F4",
        "columns": ("sparse_artifact_id", "tensor", "pattern", "logical_sparsity", "compliant_groups",
                    "total_groups", "pattern_compliance", "exceptions"),
    },
    "kernel_results": {
        "stage": "E14-F4",
        "columns": ("operator_id", "sparse_artifact_id", "shape", "dtype", "variant", "actual_kernel",
                    "fallback", "latency_ms", "tflops", "bandwidth_gbps", "occupancy", "metadata_bytes",
                    "status"),
    },
    "actual_dispatch": {
        "stage": "E14-F4",
        "columns": ("operator_id", "requested", "actual_kernel", "fallback", "reason_code", "evidence"),
    },
    "phase_timeline": {
        "stage": "E14-06",
        "columns": ("request_id", "modality", "phase", "input_shape", "output_shape", "actual_backend",
                    "actual_kernels", "start_ns", "end_ns", "device_memory_peak_bytes",
                    "quality_artifact_id", "status"),
    },
    "operator_shape_profile": {
        "stage": "E14-06",
        "columns": ("request_id", "phase", "op", "shape", "dtype", "calls", "time_ms", "backend", "stack"),
    },
    "workflow_state_events": {
        "stage": "E14-07",
        "columns": ("workflow_id", "task_id", "tenant_id", "old_state", "new_state", "attempt",
                    "idempotency_key_hash", "model_artifact_id", "tool_contract_id", "trace_id",
                    "timestamp_ns", "cost_accumulated", "status"),
    },
    "trace_spans": {
        "stage": "E14-07",
        "columns": ("trace_id", "span_id", "parent_span_id", "links", "kind", "name", "start_ns", "end_ns",
                    "waiting", "attributes"),
    },
    "critical_path": {
        "stage": "E14-07",
        "columns": ("workflow_id", "span_id", "kind", "start_ns", "end_ns", "on_critical_path", "slack_ns"),
    },
    "operator_partition": {
        "stage": "E14-08",
        "columns": ("model_artifact_id", "runtime_id", "node", "requested_backend", "actual_backend",
                    "partition", "boundary_copy", "fallback_ops", "evidence_level"),
    },
    "cold_warm_latency": {
        "stage": "E14-08",
        "columns": ("device_id", "runtime_id", "model_artifact_id", "mode", "repeat", "latency_ms",
                    "memory_peak_bytes", "temperature_c", "clock_hz", "evidence_level", "status"),
    },
    "sustained_thermal_timeline": {
        "stage": "E14-08",
        "columns": ("device_id", "t_ns", "latency_ms", "temperature_c", "clock_hz", "average_power_w",
                    "steady_window"),
    },
    "power_energy": {
        "stage": "E14-08",
        "columns": ("device_id", "window_start_ns", "window_end_ns", "average_power_w", "energy_j",
                    "source", "calibrated"),
    },
    "adoption_decision": {
        "stage": "*",
        "columns": ("decision_id", "experiment_id", "decision", "allowed_claims", "forbidden_claims",
                    "quality_status", "performance_status", "maturity", "limitations"),
    },
    "evidence_manifest": {
        "stage": "*",
        "columns": ("run_id", "stage", "experiment_id", "claim_ids", "git_commit", "git_dirty",
                    "model_artifact_id", "config_uri", "config_sha256", "environment_uri",
                    "environment_sha256", "commands", "raw_artifacts", "normalized_artifacts", "reports",
                    "correctness_status", "claim_level", "limitations"),
    },
}

#: ``S14 README`` §22 — the mandatory run-directory layout.
RUN_DIRECTORY_LAYOUT: Tuple[str, ...] = (
    "run_manifest.yaml",
    "preregistration.yaml",
    "environment_fingerprint.json",
    "dependency_lock",
    "source_identity.json",
    "model_and_checkpoint",
    "data_identity.json",
    "configs",
    "raw",
    "profiles",
    "traces",
    "quality",
    "failures",
    "normalized",
    "plots",
    "decisions",
    "evidence_manifest.yaml",
    "report.md",
)

#: ``S14 README`` §23 — the run manifest minimum fields.
RUN_MANIFEST_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "run_id",
    "status",
    "timestamps_utc",
    "git",
    "environment_fingerprint_uri",
    "release_bundle_id",
    "model_artifact_id",
    "checkpoint_artifact_id",
    "policy_snapshot_id",
    "data_artifact_id",
    "workload_spec_id",
    "backend_capability_id",
    "frontier_study_contract_id",
    "seed_bundle_id",
    "raw_artifacts",
    "profile_artifacts",
    "quality_artifacts",
    "decision_artifact",
    "limitations",
)

#: ``S14 实验清单`` §公共必做实验 — id, level, title (frozen copy of the table).
EXPERIMENT_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("E14-01", "P0", "Core/Experimental 依赖、Feature Flag 与 CI 隔离"),
    ("E14-02", "P0", "分布式训练语义、状态、通信与 Checkpoint Smoke"),
    ("E14-03", "P0", "Checkpoint→转换/Merge→Quant→Runtime 的训推一致性"),
    ("E14-04", "P0", "SFT/DPO/GRPO Rollout、同步/异步与 Policy Staleness"),
    ("E14-05", "P0", "前沿方向 ADR、可证伪假设与跨层预注册"),
    ("E14-F1", "条件 P0", "Speculative/MTP Acceptance、额外计算与服务调度"),
    ("E14-F2", "条件 P0", "MoE Router、Expert Parallel、All-to-All 与负载均衡"),
    ("E14-F3", "条件 P0", "长上下文 KV、Chunked Prefill、容量—质量—延迟"),
    ("E14-F4", "条件 P0", "结构化稀疏/Sparse Attention 的真实硬件加速"),
    ("E14-06", "P1", "VLM、Diffusion/DiT 或 Audio 的跨模型形态 Profiling"),
    ("E14-07", "P2", "Agent Tool/Memory/异步 Workflow 的等待、失败与 Trace"),
    ("E14-08", "P2", "Android/ARM/端侧 Runtime Adapter 与采用决策"),
)

#: ``S14 README`` §25 — the mandated execution order.
EXECUTION_ORDER: Tuple[str, ...] = (
    "E14-01",
    "E14-02",
    "E14-03",
    "E14-04",
    "E14-05",
    "frontier_branch",
    "optional_transfer",
    "adoption_and_handoff",
)

#: Prerequisite edges used by ``experiment.check_prerequisites``: a blocked
#: upstream keeps the downstream experiment ``BLOCKED_PREREQUISITE``
#: (``S14 README`` §25: E14-03 不能在 E14-02 无可信 checkpoint 时宣称闭环).
EXPERIMENT_DEPENDENCIES: Mapping[str, Tuple[str, ...]] = {
    "E14-01": (),
    "E14-02": ("E14-01",),
    "E14-03": ("E14-02",),
    "E14-04": ("E14-02", "E14-03"),
    "E14-05": ("E14-01", "E14-02", "E14-03", "E14-04"),
    "E14-F1": ("E14-05",),
    "E14-F2": ("E14-05",),
    "E14-F3": ("E14-05",),
    "E14-F4": ("E14-05",),
    "E14-06": ("E14-01",),
    "E14-07": ("E14-01",),
    "E14-08": ("E14-01",),
}


def table_columns(table: str) -> Tuple[str, ...]:
    if table not in TABLE_SCHEMAS:
        raise ConfigError(f"unknown S14 table {table!r}; known: {', '.join(sorted(TABLE_SCHEMAS))}")
    return tuple(TABLE_SCHEMAS[table]["columns"])


def table_owner(table: str) -> str:
    return str(TABLE_SCHEMAS[table]["stage"])


def tables_for(experiment_id: str) -> Tuple[str, ...]:
    return tuple(sorted(name for name, spec in TABLE_SCHEMAS.items() if spec["stage"] in (experiment_id, "*")))


def audit_tables(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    """Check a set of emitted rows against the declared columns.

    Unknown columns are refused rather than ignored: a typo in a raw table is
    how a "measured" field silently becomes null in the normalized output.
    """
    problems: List[str] = []
    for table, entries in rows.items():
        if table not in TABLE_SCHEMAS:
            problems.append(f"unknown table {table!r}")
            continue
        expected = set(table_columns(table))
        for index, entry in enumerate(entries):
            unknown = sorted(set(entry) - expected)
            if unknown:
                problems.append(f"{table}[{index}] has unknown columns: {', '.join(unknown)}")
    return problems


def experiment_record_template(experiment_id: str) -> Dict[str, Any]:
    """The per-experiment record of the manual §4, prefilled for ``NOT_RUN``.

    The record exists so the driver has somewhere to write the run identity;
    it is *not* a result and carries no measurement.
    """
    return {
        "stage": "S14",
        "experiment_id": experiment_id,
        "status": STATUS_NOT_STARTED,
        "question": "",
        "hypothesis": "",
        "run_id": "",
        "git_commit": "",
        "git_dirty": False,
        "model_manifest_sha256": "",
        "config_sha256": "",
        "operator_or_binary_sha256": "",
        "environment_uri": "",
        "hardware_and_power_mode": "",
        "requested_implementation": "",
        "actual_implementation": "",
        "controls": {},
        "independent_variables": {},
        "correctness_metrics": {},
        "performance_samples_uri": "",
        "profile_artifacts_uri": "",
        "started_at": "",
        "ended_at": "",
        "decision": "",
        "limitations": [],
    }
