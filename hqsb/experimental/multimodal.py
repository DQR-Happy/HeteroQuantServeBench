"""E14-06 — VLM / Diffusion-DiT / Audio cross-morphology profiling.

Optional P1: ``N/A_BY_ADR`` when not selected, and an unexecuted E14-06 must not
be described as "multimodal support" (``E14-06`` §11).

The value of this experiment is proving that HQSB's method *transfers* to a
different computation graph — not that each modality has a wrapper (§1).  Three
consequences shape the interfaces:

* **the metric must match the graph.** TTFT/TPOT are LLM units; diffusion needs
  step latency and images/s, audio needs real-time factor and WER, VLM must
  separate vision encoding from language generation (§3.1).  Reusing token/s is a
  reported FAIL, so :func:`modality_metric_name` refuses it;
* **preprocessing is part of the measured system** (§3.2): image resize/
  normalise/tokenise and audio resample/feature extraction affect shape, quality
  and end-to-end time.  They may be excluded only for a *model-core* study, and
  then the E2E number is reported alongside;
* **sub-models belong to the artifact identity** (§3.3): a VLM's vision encoder
  and projector, a diffusion model's VAE and scheduler, an audio model's
  feature extractor are all part of what must be hashed and gated.

Interfaces provided:

* :class:`ModalityChoice` (step 1), :class:`ModelComponentManifest` (step 3),
  :class:`InputArtifactManifest` (step 4);
* :class:`WorkloadContractExtension` — the C2 extension with strict rejection of
  unknown fields (step 7);
* :class:`ModalityTraceMapping` — modality-specific C6/C7 names (step 8);
* :func:`preprocess_manifest` / :func:`verify_preprocess` (steps 13–14);
* :class:`PhaseStateMachine` — preprocess/encode/project/denoise/decode/
  postprocess with legal transitions (step 6);
* :func:`phase_timeline` — framework-level stages including CPU and H2D (step 18);
* :func:`operator_shape_profile` (step 19), :func:`memory_lifecycle` (step 21),
  :func:`roofline_amdhahl` (step 22);
* :func:`determinism_report` (step 16), :func:`streaming_semantics` (step 30);
* :func:`bad_input_rejection`, :func:`missing_component_check` (steps 33–34);
* :func:`cancel_release`, :func:`cold_start` (steps 35, 31);
* :class:`MultimodalAdoptionDecision` (step 40).

Nothing here loads a model or decodes media; the arithmetic is over supplied
shapes, timings and hashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import canonical_digest, is_digest
from hqsb.experimental.telemetry import MODALITY_METRICS, modality_metric_name

EXPERIMENT_ID = "E14-06"
TITLE = "VLM、Diffusion/DiT 或 Audio 的跨模型形态 Profiling"
LEVEL = "P1"

CLAIM_BOUNDARY = (
    "只证明一个模型/任务/平台上的方法迁移，不声称通用多模态平台；未执行时必须标 N/A_BY_ADR，"
    "并在文档中同步缩小能力声明（E14-06 §11）。"
)

#: Phases of the modality state machine (§6 step 6 / §3.3).
PHASES: Tuple[str, ...] = (
    "preprocess",
    "encode",
    "project",
    "denoise",
    "decode",
    "postprocess",
)

#: Which phases each modality uses; a missing phase is ``N/A``, not zero time.
MODALITY_PHASE_MAP: Mapping[str, Tuple[str, ...]] = {
    "vlm": ("preprocess", "encode", "project", "decode", "postprocess"),
    "diffusion": ("preprocess", "encode", "denoise", "decode", "postprocess"),
    "audio": ("preprocess", "encode", "decode", "postprocess"),
}

#: Sub-model roles that must appear in the component manifest (§3.3).
COMPONENT_ROLES: Tuple[str, ...] = (
    "tokenizer",
    "preprocessor",
    "vision_encoder",
    "audio_encoder",
    "text_encoder",
    "projector",
    "denoiser",
    "vae",
    "scheduler",
    "language_model",
    "decoder",
    "postprocessor",
)

#: Timings that must be separated because they have different bottlenecks (step 18).
TIMING_STAGES: Tuple[str, ...] = (
    "cpu_preprocess_ms",
    "h2d_copy_ms",
    "device_compute_ms",
    "d2h_copy_ms",
    "cpu_postprocess_ms",
    "sync_wait_ms",
)

#: Resources reported for a modality workload (step 32).
RESOURCE_KEYS: Tuple[str, ...] = (
    "device_memory_peak_bytes",
    "host_memory_peak_bytes",
    "energy_j",
    "device_count",
)


@dataclass(frozen=True)
class ModalityChoice:
    """Step 1: one modality, chosen by an ADR with the alternatives named."""

    modality: str
    candidate_workload: str
    rationale: str
    job_relevance: str = ""
    rejected_alternatives: Tuple[str, ...] = ()
    licence: str = ""
    hardware_requirements: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.modality not in rec.MULTIMODAL_MODALITIES:
            findings.append(
                f"ModalityChoice: modality {self.modality!r} must be one of "
                f"{', '.join(rec.MULTIMODAL_MODALITIES)}"
            )
        for name in ("candidate_workload", "rationale"):
            if not getattr(self, name):
                findings.append(f"ModalityChoice: {name} is required")
        if len(self.rejected_alternatives) < 2:
            findings.append(
                "ModalityChoice: the ADR must state why the other two modalities were not chosen "
                "（三类各启动一次 demo = FAIL）"
            )
        if not self.licence:
            findings.append("ModalityChoice: the model licence must be recorded for a public claim")
        return findings

    def phases(self) -> Tuple[str, ...]:
        if self.modality not in MODALITY_PHASE_MAP:
            raise ConfigError(f"unknown modality {self.modality!r}")
        return MODALITY_PHASE_MAP[self.modality]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "candidate_workload": self.candidate_workload,
            "rationale": self.rationale,
            "job_relevance": self.job_relevance,
            "rejected_alternatives": list(self.rejected_alternatives),
            "licence": self.licence,
            "hardware_requirements": self.hardware_requirements,
            "phases": list(self.phases()),
        }


def check_capability_claim(modality: str, *, claimed: Sequence[str]) -> List[str]:
    """Step 2: the claim must be narrowed to one model/task/platform.

    A claim that mentions a modality without its model and task is the wrapper
    claim ``E14-06`` §11 forbids.
    """
    problems: List[str] = []
    if modality not in rec.MULTIMODAL_MODALITIES:
        return [f"unknown modality {modality!r}"]
    joined = " ".join(claimed).lower()
    if modality not in joined:
        problems.append(f"the capability claim does not name its modality ({modality})")
    for token in ("platform", "device", "model"):
        if token not in joined:
            problems.append(
                f"the capability claim does not constrain its {token}: a wrapper is not a capability"
            )
    for overreach in ("all modalities", "general multimodal", "any model", "通用多模态"):
        if overreach in joined:
            problems.append(f"the capability claim contains an overreach: {overreach!r}")
    return problems


@dataclass
class ModelComponentManifest:
    """Step 3: every sub-model, preprocessor and scheduler is part of the identity."""

    modality_artifact_id: str
    components: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    top_level_licence: str = ""

    REQUIRED_KEYS: Tuple[str, ...] = ("artifact_id", "revision", "digest", "licence")

    def required_roles(self) -> Tuple[str, ...]:
        if self.modality_artifact_id.startswith("vlm"):
            return ("preprocessor", "vision_encoder", "projector", "tokenizer", "language_model")
        if self.modality_artifact_id.startswith("diffusion"):
            return ("text_encoder", "denoiser", "vae", "scheduler")
        if self.modality_artifact_id.startswith("audio"):
            return ("preprocessor", "audio_encoder", "decoder")
        return ("tokenizer",)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.modality_artifact_id:
            findings.append("ModelComponentManifest: modality_artifact_id is required")
        for role, entry in sorted(self.components.items()):
            if role not in COMPONENT_ROLES:
                findings.append(f"ModelComponentManifest: unknown component role {role!r}")
            missing = [key for key in self.REQUIRED_KEYS if not entry.get(key)]
            if missing:
                findings.append(f"ModelComponentManifest: {role} is missing {', '.join(missing)}")
            digest = entry.get("digest", "")
            if digest and not is_digest(digest):
                findings.append(f"ModelComponentManifest: {role} digest must be sha256:<hex>")
        missing_roles = [role for role in self.required_roles() if role not in self.components]
        if missing_roles:
            findings.append(
                "ModelComponentManifest: sub-models not hashed: "
                + ", ".join(missing_roles)
                + "（顶层 hash 未覆盖子制品 = FAIL）"
            )
        return findings

    def digest(self) -> str:
        return canonical_digest({role: dict(entry) for role, entry in sorted(self.components.items())})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality_artifact_id": self.modality_artifact_id,
            "components": {role: dict(entry) for role, entry in sorted(self.components.items())},
            "top_level_licence": self.top_level_licence,
            "aggregate_digest": self.digest(),
        }


#: Parameters that must be recorded per media kind (§3.2).  Module level, not a
#: dataclass attribute: a mutable class attribute default is a shared-state bug.
MEDIA_PARAMETERS: Mapping[str, Tuple[str, ...]] = {
    "image": ("resolution", "color_space", "resize_mode", "normalise"),
    "audio": ("sample_rate", "channels", "feature", "window_ms"),
    "video": ("fps", "frame_count", "resolution"),
}


@dataclass
class InputArtifactManifest:
    """Step 4: the raw media plus the exact preprocessing version that consumed it."""

    input_id: str
    media_kind: str
    digest: str
    preprocess_version: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    prompt: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("input_id", "media_kind", "preprocess_version"):
            if not getattr(self, name):
                findings.append(f"InputArtifactManifest: {name} is required")
        if self.digest and not is_digest(self.digest):
            findings.append("InputArtifactManifest: digest must be sha256:<hex>")
        expected = MEDIA_PARAMETERS.get(self.media_kind)
        if expected is None:
            findings.append(f"InputArtifactManifest: unknown media_kind {self.media_kind!r}")
        else:
            missing = [key for key in expected if key not in self.parameters]
            if missing:
                findings.append(
                    f"InputArtifactManifest: {self.media_kind} parameters not recorded: {', '.join(missing)}"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "media_kind": self.media_kind,
            "digest": self.digest,
            "preprocess_version": self.preprocess_version,
            "parameters": {key: self.parameters[key] for key in sorted(self.parameters)},
            "prompt": self.prompt,
        }


def verify_preprocess(
    *, before: Mapping[str, Any], after: Mapping[str, Any], expected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Steps 13–14: raw decode and preprocess semantics, checked as tensors.

    ``E14-06`` §10: resize/normalise 误差被归为模型质量 is avoided only by saving
    the tensor shape/dtype/range/hash of the preprocessed input and comparing it
    against a trusted implementation.
    """
    problems: List[str] = []
    for key in ("shape", "dtype", "range_min", "range_max", "digest"):
        if key not in after:
            problems.append(f"preprocessed tensor does not record {key!r}")
    for key, value in expected.items():
        if key not in after:
            problems.append(f"expected value {key!r} has no counterpart")
        elif after[key] != value:
            problems.append(f"{key} differs from the trusted implementation: {after[key]!r} != {value!r}")
    if before.get("decode_errors"):
        problems.append(f"raw decode reported errors: {before['decode_errors']}")
    return {
        "recorded": dict(sorted(after.items())),
        "problems": problems,
        "ok": not problems,
    }


#: The C2 extension for a modality workload (step 7).
WORKLOAD_EXTENSION_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "vlm": ("modality", "image_resolution", "image_count", "max_new_tokens"),
    "diffusion": ("modality", "resolution", "steps", "guidance_scale", "seed"),
    "audio": ("modality", "sample_rate", "duration_s", "chunk_ms", "streaming"),
}

#: Fields that must never express a non-LLM workload (step 8 / §10).
FORBIDDEN_WORKLOAD_FIELDS: Tuple[str, ...] = ("tokens_per_second", "tpot", "ttft", "itl")


@dataclass
class WorkloadContractExtension:
    """Step 7: a versioned C2 extension whose unknown fields are rejected."""

    modality: str
    version: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.modality not in rec.MULTIMODAL_MODALITIES:
            findings.append(f"WorkloadContractExtension: unknown modality {self.modality!r}")
            return findings
        if not self.version:
            findings.append("WorkloadContractExtension: an explicit version is required")
        required = WORKLOAD_EXTENSION_FIELDS[self.modality]
        missing = [name for name in required if name not in self.fields]
        if missing:
            findings.append(f"WorkloadContractExtension: required fields missing: {', '.join(missing)}")
        unknown = sorted(set(self.fields) - set(required))
        if unknown:
            findings.append(
                f"WorkloadContractExtension: unknown fields must be rejected, not ignored: {', '.join(unknown)}"
            )
        for name in FORBIDDEN_WORKLOAD_FIELDS:
            if name in self.fields:
                findings.append(
                    f"WorkloadContractExtension: {name!r} is an LLM-only unit and may not express "
                    f"{self.modality}"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "version": self.version,
            "fields": {key: self.fields[key] for key in sorted(self.fields)},
        }


@dataclass
class ModalityTraceMapping:
    """Step 8: the C6/C7 extension for this modality, without breaking LLM fields."""

    modality: str
    c6_metric: str = ""
    c7_phase_fields: Tuple[str, ...] = ()
    extra_metric_fields: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.modality not in MODALITY_METRICS:
            findings.append(f"ModalityTraceMapping: unknown modality {self.modality!r}")
            return findings
        if not self.c6_metric:
            findings.append("ModalityTraceMapping: a task-native C6 metric is required")
        else:
            try:
                modality_metric_name(self.modality, self.c6_metric)
            except ConfigError as exc:
                findings.append(str(exc))
        if not self.c7_phase_fields:
            findings.append("ModalityTraceMapping: C7 phase fields are required")
        for name in self.extra_metric_fields:
            if name in FORBIDDEN_WORKLOAD_FIELDS:
                findings.append(f"ModalityTraceMapping: {name!r} reuses an LLM unit for {self.modality}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "c6_metric": self.c6_metric,
            "c7_phase_fields": list(self.c7_phase_fields),
            "extra_metric_fields": list(self.extra_metric_fields),
        }


@dataclass
class PhaseStateMachine:
    """Step 6: which phases ran, in order, with the modality's legal set."""

    modality: str
    events: List[Mapping[str, Any]] = field(default_factory=list)

    def record(self, phase: str, *, start_ns: int = 0, end_ns: int = 0, detail: Optional[Mapping[str, Any]] = None) -> None:
        self.events.append(
            {"phase": phase, "start_ns": start_ns, "end_ns": end_ns, "detail": dict(detail or {})}
        )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.modality not in MODALITY_PHASE_MAP:
            findings.append(f"PhaseStateMachine: unknown modality {self.modality!r}")
            return findings
        legal = MODALITY_PHASE_MAP[self.modality]
        seen: List[str] = []
        for index, event in enumerate(self.events):
            phase = str(event.get("phase", ""))
            if phase not in PHASES:
                findings.append(f"event {index}: unknown phase {phase!r}")
                continue
            if phase not in legal:
                findings.append(f"event {index}: phase {phase!r} is not part of the {self.modality} graph")
            if event.get("end_ns") and event.get("start_ns") and event["end_ns"] < event["start_ns"]:
                findings.append(f"event {index}: end_ns precedes start_ns")
            seen.append(phase)
        expected = [phase for phase in legal if phase in seen]
        if seen != expected:
            findings.append(f"phases ran out of order: {seen} (expected the {self.modality} order {expected})")
        missing = [phase for phase in legal if phase not in seen]
        if missing:
            findings.append(f"phases not executed: {', '.join(missing)}")
        if not self.events:
            findings.append("no phase events recorded")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "events": [dict(event) for event in self.events],
            "problems": self.problems(),
        }


def phase_timeline(events: Sequence[Mapping[str, Any]], *, require_e2e: bool = True) -> Dict[str, Any]:
    """Step 18: framework-level stages, including CPU and copies.

    ``E14-06`` §10: 只采 GPU 不采 CPU/IO loses the bottleneck on a modality graph,
    whose preprocessing is often the dominant cost.
    """
    problems: List[str] = []
    totals: Dict[str, float] = {phase: 0.0 for phase in PHASES}
    stage_totals: Dict[str, float] = {stage: 0.0 for stage in TIMING_STAGES}
    device_only = 0.0
    for event in events:
        phase = str(event.get("phase", ""))
        if phase not in PHASES:
            problems.append(f"timeline event has unknown phase {phase!r}")
            continue
        for stage in TIMING_STAGES:
            value = float(event.get(stage, 0.0) or 0.0)
            stage_totals[stage] += value
            if stage == "device_compute_ms":
                device_only += value
        totals[phase] += sum(float(event.get(stage, 0.0) or 0.0) for stage in TIMING_STAGES)
    e2e = sum(totals.values())
    preprocess_ratio = (
        (stage_totals["cpu_preprocess_ms"] + stage_totals["h2d_copy_ms"]) / e2e if e2e else 0.0
    )
    if require_e2e and not events:
        problems.append("no timeline events: an E2E number cannot be reported")
    return {
        "phase_totals_ms": totals,
        "stage_totals_ms": stage_totals,
        "end_to_end_ms": e2e,
        "device_only_ms": device_only,
        "preprocess_ratio": preprocess_ratio,
        "problems": problems,
        "note": "预处理与 CPU 搬运属于被测系统；排除它们却称 E2E 是 FAIL（§10）",
    }


def operator_shape_profile(rows: Sequence[Mapping[str, Any]], *, top_n: int = 10) -> Dict[str, Any]:
    """Step 19: op × shape × dtype × call count, so the real hotspot is named."""
    problems: List[str] = []
    totals: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        for key in ("op", "shape", "dtype", "calls", "time_ms"):
            if key not in row:
                problems.append(f"operator row is missing {key!r}")
                break
        else:
            key = (str(row["op"]), str(row["shape"]))
            bucket = totals.setdefault(
                key, {"op": row["op"], "shape": row["shape"], "dtype": row["dtype"], "calls": 0, "time_ms": 0.0}
            )
            bucket["calls"] += int(row["calls"])
            bucket["time_ms"] += float(row["time_ms"])
    ranked = sorted(totals.values(), key=lambda item: item["time_ms"], reverse=True)[:top_n]
    grand_total = sum(item["time_ms"] for item in totals.values())
    for item in ranked:
        item["share"] = item["time_ms"] / grand_total if grand_total else 0.0
    return {"ops": len(totals), "top": ranked, "problems": problems, "note": "从 operator 名猜硬件原因是不可靠的（step 20）"}


def memory_lifecycle(phases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 21: weights/activation/cache/workspace per phase, plus the peak.

    ``E14-06`` §10: 只有全程 peak 无法优化, so the phase at which the peak occurs is
    returned explicitly.
    """
    problems: List[str] = []
    if not phases:
        return {"error": "no phases supplied"}
    required = ("phase", "weights_bytes", "activation_bytes", "cache_bytes", "workspace_bytes", "reserved_bytes")
    peak = 0
    peak_phase = ""
    rows: List[Dict[str, Any]] = []
    for entry in phases:
        missing = [key for key in required if key not in entry]
        if missing:
            problems.append(f"phase {entry.get('phase', '<unnamed>')} is missing {', '.join(missing)}")
            continue
        total = sum(int(entry[key]) for key in required[1:])
        rows.append({**{key: entry[key] for key in required}, "total_bytes": total})
        if total > peak:
            peak = total
            peak_phase = str(entry["phase"])
    return {
        "phases": rows,
        "peak_bytes": peak,
        "peak_phase": peak_phase,
        "fragmentation_bytes": max((int(entry.get("reserved_bytes", 0)) for entry in phases), default=0) - peak,
        "problems": problems,
    }


def roofline_amdhahl(
    *, hotspot_time_ms: float, total_time_ms: float, flops: float, bytes_moved: float,
    peak_tflops: float, peak_gbps: float,
) -> Dict[str, Any]:
    """Step 22: which roof the hotspot is under, and the maximum visible gain.

    ``E14-06`` §10: 优化非热点 is the error this avoids — the Amdahl ceiling is
    computed *before* implementing a candidate.
    """
    problems: List[str] = []
    if hotspot_time_ms <= 0 or total_time_ms <= 0:
        problems.append("hotspot and total time must be positive")
    if hotspot_time_ms > total_time_ms:
        problems.append("hotspot time exceeds the total time")
    achieved_tflops = flops / (hotspot_time_ms / 1000.0) / 1e12 if hotspot_time_ms else 0.0
    achieved_gbps = bytes_moved / (hotspot_time_ms / 1000.0) / 1e9 if hotspot_time_ms else 0.0
    compute_ratio = achieved_tflops / peak_tflops if peak_tflops else 0.0
    bandwidth_ratio = achieved_gbps / peak_gbps if peak_gbps else 0.0
    return {
        "hotspot_share": hotspot_time_ms / total_time_ms if total_time_ms else 0.0,
        "achieved_tflops": achieved_tflops,
        "achieved_gbps": achieved_gbps,
        "compute_utilisation": compute_ratio,
        "bandwidth_utilisation": bandwidth_ratio,
        "bound": "compute" if compute_ratio >= bandwidth_ratio else "memory",
        "amdahl_ceiling_speedup": total_time_ms / (total_time_ms - hotspot_time_ms) if total_time_ms > hotspot_time_ms else float("inf"),
        "problems": problems,
    }


def determinism_report(
    *, modality: str, seed: Optional[int], repeated_outputs: Sequence[str], tolerance: str
) -> Dict[str, Any]:
    """Step 16: algorithmic randomness, non-deterministic kernels and sampling.

    ``E14-06`` §10 implies a diffusion model cannot be required to be bitwise
    identical; what is required is that the *allowed* variation is declared.
    """
    problems: List[str] = []
    if modality not in rec.MULTIMODAL_MODALITIES:
        problems.append(f"unknown modality {modality!r}")
    if seed is None and modality in ("diffusion", "audio"):
        problems.append(f"{modality} needs a recorded seed to be reproducible")
    identical = len(set(repeated_outputs)) == 1 if repeated_outputs else False
    if not repeated_outputs:
        problems.append("no repeated outputs supplied: the variation bound is unverified")
    return {
        "modality": modality,
        "seed": seed,
        "repeats": len(repeated_outputs),
        "bitwise_identical": identical,
        "declared_tolerance": tolerance,
        "problems": problems,
        "note": "随机性必须有界且被声明；要求无定义的 bitwise 相同同样不合格",
    }


def streaming_semantics(
    *, modality: str, applicable: bool, chunks: Sequence[Mapping[str, Any]], backpressure: bool
) -> Dict[str, Any]:
    """Step 30: chunk latency, partial/final results and state cache, or an explicit N/A."""
    problems: List[str] = []
    if not applicable:
        return {
            "applicable": False,
            "status": rec.N_A_BY_ADR,
            "problems": [],
            "note": "不适用时显式声明 N/A，而不是省略（step 30）",
        }
    if modality not in ("audio", "vlm"):
        problems.append(f"streaming declared for {modality!r}, which has no streaming path in this design")
    if not chunks:
        problems.append("streaming declared but no chunks recorded")
    for index, chunk in enumerate(chunks):
        for key in ("chunk_index", "first_result_ms", "final_ms", "partial"):
            if key not in chunk:
                problems.append(f"chunk {index} is missing {key!r}")
    if not backpressure:
        problems.append("streaming without backpressure recording can hide unbounded buffering")
    finals = [chunk for chunk in chunks if chunk.get("partial") is False]
    if len(finals) > 1:
        problems.append(f"{len(finals)} chunks claim to be final")
    return {"applicable": True, "chunks": len(chunks), "problems": problems, "ok": not problems}


def cold_start(
    *, sub_model_load_ms: Mapping[str, float], compile_ms: float, warmup_ms: float, readiness_declared: bool
) -> Dict[str, Any]:
    """Step 31: multi sub-model download/load/compile/warmup and readiness.

    ``E14-06`` §10: 预热完毕的模型冒充即时可用 is a FAIL, so readiness is only
    declared once every component has been loaded and warmed.
    """
    problems: List[str] = []
    total = sum(float(value) for value in sub_model_load_ms.values()) + compile_ms + warmup_ms
    if not sub_model_load_ms:
        problems.append("no sub-model load times recorded")
    if compile_ms < 0 or warmup_ms < 0:
        problems.append("compile/warmup times must be >= 0")
    if readiness_declared and total <= 0:
        problems.append("readiness declared while the cold path costs nothing: the measurement is missing")
    return {
        "sub_models": len(sub_model_load_ms),
        "load_ms": sum(float(value) for value in sub_model_load_ms.values()),
        "compile_ms": compile_ms,
        "warmup_ms": warmup_ms,
        "cold_total_ms": total,
        "readiness_declared": readiness_declared,
        "problems": problems,
    }


def bad_input_rejection(
    cases: Sequence[Mapping[str, Any]], *, max_bytes: int, max_duration_s: float
) -> List[str]:
    """Step 33: corrupt/oversized/empty input must be refused before a large allocation."""
    problems: List[str] = []
    known = ("corrupt_media", "oversized_resolution", "oversized_duration", "empty_input", "illegal_parameter")
    for case in cases:
        kind = str(case.get("kind", ""))
        if kind not in known:
            problems.append(f"unknown bad-input case {kind!r}")
            continue
        if not case.get("rejected"):
            problems.append(f"{kind}: was not rejected before the expensive allocation")
        if case.get("crashed"):
            problems.append(f"{kind}: crashed the parser/decoder instead of returning a structured error")
        if case.get("allocated_bytes", 0) > max_bytes:
            problems.append(f"{kind}: allocated {case['allocated_bytes']} bytes beyond the cap")
        if case.get("duration_s", 0.0) > max_duration_s:
            problems.append(f"{kind}: exceeded the duration cap before rejection")
    if not cases:
        problems.append("no bad-input cases recorded")
    return problems


def missing_component_check(
    manifest: ModelComponentManifest, *, removed_role: str, top_level_digest_unchanged: bool
) -> Dict[str, Any]:
    """Step 34: removing/replacing a sub-model must be caught by the artifact gate.

    ``E14-06`` §10: 顶层 hash 未覆盖子制品 is the failure; the check therefore
    reports whether the top-level digest still matches after the mutation.
    """
    problems: List[str] = []
    if removed_role not in manifest.components:
        problems.append(f"the fixture removed {removed_role!r}, which is not in the manifest")
    if top_level_digest_unchanged:
        problems.append(
            "the top-level digest did not change when a sub-model was removed: the identity does not "
            "cover the sub-artifacts"
        )
    return {
        "removed_role": removed_role,
        "top_level_digest_unchanged": top_level_digest_unchanged,
        "problems": problems,
        "fail_closed": not problems,
    }


# ── cancel, ablation, reconciliation, adoption (steps 35–40) ──────────────


def cancel_release(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 35: cancelling during encode/denoise/stream must free activation and cache."""
    problems: List[str] = []
    known = ("preprocess", "encode", "denoise", "decode", "stream")
    for event in events:
        stage = str(event.get("stage", ""))
        if stage not in known:
            problems.append(f"cancel stage {stage!r} is not documented")
        if event.get("work_remaining"):
            problems.append(f"cancel at {stage}: background work continued")
        if event.get("activation_leaked"):
            problems.append(f"cancel at {stage}: activation memory leaked")
        if event.get("worker_leaked"):
            problems.append(f"cancel at {stage}: a worker/cache entry leaked")
    if not events:
        problems.append("no cancellation events recorded")
    return problems


def candidate_ablation(
    *, factors: Mapping[str, Any], budget: int, explanation: str
) -> Dict[str, Any]:
    """Step 36: only preregistered parameters, and the explanation must name a phase."""
    problems: List[str] = []
    if len(factors) > budget:
        problems.append(f"{len(factors)} factors exceed the preregistered budget {budget}")
    if not explanation:
        problems.append("the ablation must explain which phase the gain came from")
    elif not any(phase in explanation.lower() for phase in PHASES):
        problems.append("the ablation explanation names no modality phase")
    return {
        "factors": {key: factors[key] for key in sorted(factors)},
        "budget": budget,
        "problems": problems,
        "ok": not problems,
    }


def reconcile_prediction(
    *,
    predicted_e2e_ms: float,
    measured_e2e_ms: float,
    hotspot_share: float,
    kernel_speedup: float,
    preprocessing_ratio: float,
) -> Dict[str, Any]:
    """Step 37: rebuild the E2E change from hotspot coverage and the measured stages."""
    if predicted_e2e_ms <= 0:
        return {"error": "predicted_e2e_ms must be positive"}
    residual = measured_e2e_ms - predicted_e2e_ms
    return {
        "predicted_e2e_ms": predicted_e2e_ms,
        "measured_e2e_ms": measured_e2e_ms,
        "residual_ms": residual,
        "residual_ratio": residual / predicted_e2e_ms,
        "mediators": {"hotspot_share": hotspot_share, "kernel_speedup": kernel_speedup,
                      "preprocessing_ratio": preprocessing_ratio},
        "explained": abs(residual) <= 0.15 * predicted_e2e_ms,
        "note": "micro 优化直接外推全任务是不合格的；残差必须由阶段与 kernel 解释（step 37）",
    }


#: Metric fields the driver reads to decide whether a modality report is complete.
REQUIRED_REPORT_FIELDS: Tuple[str, ...] = (
    "modality",
    "model_component_manifest",
    "input_artifact_manifest",
    "workload_contract",
    "quality_results",
    "phase_timeline",
    "operator_shape_profile",
    "memory_timeline",
    "adoption_decision",
)


def check_report_completeness(report: Mapping[str, Any]) -> List[str]:
    """Steps 40 / §11: a wrapper or a screenshot cannot substitute for these files."""
    findings: List[str] = []
    for name in REQUIRED_REPORT_FIELDS:
        if not report.get(name):
            findings.append(f"modality report is missing {name!r}")
    if report.get("modality") not in rec.MULTIMODAL_MODALITIES:
        findings.append("modality report does not name exactly one modality")
    for overreach in ("通用多模态", "general multimodal", "supports all"):
        if overreach in str(report).lower():
            findings.append(f"report contains an overreach: {overreach!r}")
    return findings


def multimodal_adoption(
    *,
    decision_id: str,
    modality: str,
    quality: Mapping[str, Any],
    profile: Mapping[str, Any],
    actual_path: Mapping[str, Any],
    dependency_isolation_ok: bool,
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: "it runs" is not a reason to keep a modality in the main line (§11)."""
    problems: List[str] = []
    if modality not in rec.MULTIMODAL_MODALITIES:
        problems.append(f"unknown modality {modality!r}")
    if quality.get("problems"):
        problems.append("task-native quality gate failed")
    if not profile.get("phase_totals_ms"):
        problems.append("no cross-layer phase profile")
    if not actual_path.get("actual_backend"):
        problems.append("actual backend is unknown")
    if not dependency_isolation_ok:
        problems.append("the modality extra is not isolated from the core install")
    if problems:
        decision = rec.REJECT_QUALITY if quality.get("problems") else rec.BLOCKED_EVIDENCE
        allowed: Tuple[str, ...] = ()
    else:
        decision = rec.ADOPT_EXPERIMENTAL
        allowed = (f"{modality} 的 profile 方法迁移在给定模型/平台上成立",)
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "存在 wrapper 即宣称支持多模态",
            "用 token/s 表达图像/音频工作量",
            "单张图/短音频外推",
            "排除预处理却称 E2E",
        ),
        quality_status=rec.STATUS_PASS if not quality.get("problems") else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_RUNTIME_PROFILED if not problems else rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("结论限定于一个模型/任务/平台，不替代 LLM 主线故事",),
        reopened_if=("新增一个可直接测量的 modality 且核心依赖仍被隔离时可重开",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-06 interfaces (labelled smoke, not an experiment)."""
    choice = ModalityChoice(
        modality="audio", candidate_workload="streaming ASR",
        rationale="音频有独立的实时因子与 chunk 语义", job_relevance="端侧/流式",
        rejected_alternatives=("vlm", "diffusion"), licence="Apache-2.0",
    )
    mapping = ModalityTraceMapping(
        modality="audio", c6_metric="wer", c7_phase_fields=("modality_phase", "chunk_index"),
    )
    rejected_metric = False
    try:
        modality_metric_name("audio", "tokens_per_second")
    except ConfigError:
        rejected_metric = True
    runtime = PhaseStateMachine(modality="audio")
    runtime.record("preprocess", start_ns=0, end_ns=10)
    runtime.record("encode", start_ns=10, end_ns=20)
    timeline = phase_timeline(
        [
            {"phase": "preprocess", "cpu_preprocess_ms": 4.0, "device_compute_ms": 0.0},
            {"phase": "encode", "device_compute_ms": 6.0},
            {"phase": "decode", "device_compute_ms": 5.0},
            {"phase": "postprocess", "cpu_postprocess_ms": 1.0},
        ]
    )
    manifest = ModelComponentManifest(
        modality_artifact_id="audio-model",
        components={
            role: {"artifact_id": role, "revision": "r1", "digest": "sha256:" + "0" * 64, "licence": "MIT"}
            for role in ("preprocessor", "audio_encoder", "decoder")
        },
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "phases": list(choice.phases()),
        "choice_problems": choice.problems(),
        "mapping_problems": mapping.problems(),
        "llm_metric_rejected": rejected_metric,
        "runtime_phase_problems": len(runtime.problems()),
        "e2e_ms": timeline["end_to_end_ms"],
        "preprocess_ratio": round(timeline["preprocess_ratio"], 4),
        "manifest_problems": manifest.problems(),
        "required_report_fields": len(REQUIRED_REPORT_FIELDS),
    }


# ── protocol step table (40 steps of details/S14/E14-06) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "用 ADR 选择一种模型形态", ("multimodal:ModalityChoice", "records:MULTIMODAL_MODALITIES")),
    (2, "冻结能力声明", ("multimodal:check_capability_claim", "multimodal:CLAIM_BOUNDARY")),
    (3, "冻结 ModelArtifact", ("multimodal:ModelComponentManifest", "multimodal:COMPONENT_ROLES")),
    (4, "冻结输入制品", ("multimodal:InputArtifactManifest", "multimodal:MEDIA_PARAMETERS")),
    (5, "冻结任务质量 oracle", ("telemetry:MODALITY_METRICS", "multimodal:ModalityTraceMapping.c6_metric")),
    (6, "冻结阶段状态机", ("multimodal:PhaseStateMachine", "multimodal:PHASES")),
    (7, "扩展 WorkloadSpec", ("multimodal:WorkloadContractExtension", "multimodal:WORKLOAD_EXTENSION_FIELDS")),
    (8, "扩展结果与 Trace 映射", ("multimodal:ModalityTraceMapping", "telemetry:C6_EXTENSION_FIELDS")),
    (9, "冻结 baseline 与单一候选变化", ("frontier:BaselinePair", "frontier:BaselinePair.intended_difference")),
    (10, "冻结 workload 矩阵", ("frontier:WorkloadStrata", "multimodal:WORKLOAD_EXTENSION_FIELDS")),
    (11, "运行 E14-01 依赖隔离验证", ("dependencies:probe_extra", "dependencies:analyse_import_trace")),
    (12, "运行 capability/actual-path probe", ("contracts:check_actual_path_recorded", "records:STATUS_INVALID_IDENTITY")),
    (13, "验证原始输入 decode", ("multimodal:InputArtifactManifest", "multimodal:verify_preprocess")),
    (14, "验证预处理语义", ("multimodal:verify_preprocess", "multimodal:InputArtifactManifest.parameters")),
    (15, "建立高精度/reference 输出", ("parity:REFERENCE_PATHS", "parity:evaluate_gates")),
    (16, "验证确定性/随机性", ("multimodal:determinism_report", "identity:SeedBundle")),
    (17, "运行 correctness/quality baseline", ("multimodal:ModalityTraceMapping.c6_metric",
                                                "contracts:check_quality_before_performance")),
    (18, "采集框架级阶段 profile", ("multimodal:phase_timeline", "multimodal:TIMING_STAGES")),
    (19, "采集 operator/shape profile", ("multimodal:operator_shape_profile", "records:PROFILE_LAYER_FIELDS")),
    (20, "采集 kernel/system profile", ("multimodal:operator_shape_profile", "records:PROFILE_LAYERS")),
    (21, "建立显存生命周期", ("multimodal:memory_lifecycle", "contracts:check_resource_ledger")),
    (22, "建立 baseline Roofline/Amdahl", ("multimodal:roofline_amdhahl", "frontier:PredictionModel")),
    (23, "实现/启用候选路径", ("dependencies:FeatureFlagRegistry.resolve", "frontier:BaselinePair")),
    (24, "验证 candidate actual path", ("multimodal:roofline_amdhahl", "contracts:check_actual_path_recorded")),
    (25, "运行中间 tensor correctness", ("parity:evaluate_gates", "multimodal:verify_preprocess")),
    (26, "运行 candidate 质量门", ("contracts:check_quality_before_performance", "multimodal:ModalityTraceMapping")),
    (27, "运行单样本阶段 benchmark", ("multimodal:phase_timeline", "frontier:TimingBoundaries")),
    (28, "运行 batch/shape 扫描", ("frontier:WorkloadStrata", "multimodal:operator_shape_profile")),
    (29, "运行并发/服务 workload", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan")),
    (30, "验证流式语义（若适用）", ("multimodal:streaming_semantics", "records:N_A_BY_ADR")),
    (31, "测冷启动和模型切换", ("multimodal:cold_start", "multimodal:ModelComponentManifest")),
    (32, "测内存/能耗/成本", ("multimodal:RESOURCE_KEYS", "contracts:check_resource_ledger")),
    (33, "注入坏输入", ("multimodal:bad_input_rejection", "campaign:isolation_clause")),
    (34, "注入缺子模型/错 preprocessor", ("multimodal:missing_component_check", "records:STATUS_INVALID_IDENTITY")),
    (35, "验证取消/超时/资源恢复", ("multimodal:cancel_release", "records:STATUS_FAIL_RECOVERY")),
    (36, "执行候选消融", ("multimodal:candidate_ablation", "frontier:AblationMatrix")),
    (37, "对账预测与实测", ("multimodal:reconcile_prediction", "multimodal:roofline_amdhahl")),
    (38, "在 holdout 输入确认", ("frontier:WorkloadStrata.holdout_id", "frontier:StatisticsPlan")),
    (39, "跨 run/time block 重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 MultimodalAdoptionDecision", ("multimodal:multimodal_adoption", "multimodal:check_report_completeness",
                                              "contracts:AdoptionDecision.validate")),
)
