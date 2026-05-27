"""Calibration data identity, leakage audit, factor sampling and selection.

E05-03 requires more than "run some text through the model": the calibration
protocol must be *auditable* — four isolated splits, per-sample identity
(dataset/revision/id/token hash), a leakage audit, factor sampling with
seeds, statistics with stability metrics, and a pre-registered rule that
decides the minimum sufficient budget **without looking at the final
evaluation** (details README §8, E05-03 §4/§5/§8).

This module owns all of that. It is pure Python: tokenization and activation
capture need the model stack and are provided through hooks
(:func:`collect_activation_stats`), which are lazy and optional.

Forbidden and therefore *unrepresentable* here:

* a sample that belongs to two splits (the manifest refuses it);
* a calibration sample selected by looking at final-evaluation results
  (selection only consumes policy-validation metrics);
* an activation statistic that silently includes padding tokens (the
  collector requires an explicit mask and records the effective count).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant import stats as qstats

#: Split identifiers (E05-03 §4). There is no "default" split: an unset split
#: is a hard error, because "which split is this?" must always be answerable.
SPLIT_CALIBRATION = "calibration"
SPLIT_POLICY_VALIDATION = "policy-validation"
SPLIT_FINAL_EVALUATION = "final-evaluation"
SPLIT_STRESS = "stress-ood"

SPLITS = (
    SPLIT_CALIBRATION,
    SPLIT_POLICY_VALIDATION,
    SPLIT_FINAL_EVALUATION,
    SPLIT_STRESS,
)

#: Splits that must never feed quantization parameters.
NON_CALIBRATION_SPLITS = SPLITS[1:]

_NORMALIZE_RE = re.compile(r"\s+")


def normalized_text_hash(text: str) -> str:
    """Hash of whitespace/case-normalized text (duplicate detection)."""
    normalized = _NORMALIZE_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def token_ids_hash(token_ids: Sequence[int]) -> str:
    """Stable hash of a token id sequence (the reproducible identity)."""
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


@dataclass
class SampleRecord:
    """One calibration/evaluation sample with its full provenance."""

    dataset: str
    revision: str
    split: str
    sample_id: str
    text_hash: str
    token_ids: Sequence[int]
    license: str = ""
    language: str = ""
    domain: str = ""
    length_bucket: str = ""
    task_type: str = ""
    template: str = ""
    preprocess_version: str = ""
    tokenizer_revision: str = ""
    truncated: bool = False
    parent_id: str = ""
    sampling_seed: Optional[int] = None
    selection_probability: Optional[float] = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ConfigError(
                f"unknown split {self.split!r}; supported: {list(SPLITS)}"
            )
        if not self.sample_id:
            raise ConfigError("sample_id must not be empty")

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def token_hash(self) -> str:
        return token_ids_hash(self.token_ids)

    def as_dict(self, *, include_tokens: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset": self.dataset,
            "revision": self.revision,
            "split": self.split,
            "sample_id": self.sample_id,
            "text_hash": self.text_hash,
            "token_hash": self.token_hash,
            "num_tokens": self.num_tokens,
            "license": self.license,
            "language": self.language,
            "domain": self.domain,
            "length_bucket": self.length_bucket,
            "task_type": self.task_type,
            "template": self.template,
            "preprocess_version": self.preprocess_version,
            "tokenizer_revision": self.tokenizer_revision,
            "truncated": self.truncated,
            "parent_id": self.parent_id,
            "sampling_seed": self.sampling_seed,
            "selection_probability": self.selection_probability,
        }
        if include_tokens:
            payload["token_ids"] = [int(token) for token in self.token_ids]
        return payload


@dataclass
class SplitManifest:
    """All samples of one split plus the dataset identity."""

    split: str
    samples: List[SampleRecord] = field(default_factory=list)
    dataset_revision: str = ""
    tokenizer_revision: str = ""
    preprocess_version: str = ""

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ConfigError(f"unknown split {self.split!r}")

    def add(self, sample: SampleRecord) -> None:
        if sample.split != self.split:
            raise ConfigError(
                f"sample {sample.sample_id!r} belongs to split {sample.split!r}, "
                f"not {self.split!r}"
            )
        self.samples.append(sample)

    @property
    def sample_ids(self) -> List[str]:
        return [sample.sample_id for sample in self.samples]

    @property
    def total_tokens(self) -> int:
        return sum(sample.num_tokens for sample in self.samples)

    @property
    def manifest_hash(self) -> str:
        payload = {
            "split": self.split,
            "dataset_revision": self.dataset_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "preprocess_version": self.preprocess_version,
            "samples": sorted(
                (
                    sample.sample_id,
                    sample.text_hash,
                    sample.token_hash,
                    sample.dataset,
                    sample.revision,
                )
                for sample in self.samples
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(sample.as_dict(), sort_keys=True, ensure_ascii=False)
            for sample in self.samples
        )

    def summary(self) -> Dict[str, Any]:
        buckets: Dict[str, int] = {}
        domains: Dict[str, int] = {}
        for sample in self.samples:
            buckets[sample.length_bucket or "unknown"] = (
                buckets.get(sample.length_bucket or "unknown", 0) + 1
            )
            domains[sample.domain or "unknown"] = (
                domains.get(sample.domain or "unknown", 0) + 1
            )
        return {
            "split": self.split,
            "samples": len(self.samples),
            "total_tokens": self.total_tokens,
            "manifest_hash": self.manifest_hash,
            "length_buckets": dict(sorted(buckets.items())),
            "domains": dict(sorted(domains.items())),
            "truncated_samples": sum(1 for sample in self.samples if sample.truncated),
        }


# ── leakage audit (E05-03 §5) ─────────────────────────────────────────────


@dataclass
class LeakageFinding:
    kind: str
    split_a: str
    split_b: str
    detail: str
    count: int


def _ngrams(tokens: Sequence[int], n: int) -> set:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _minhash_signature(ngram_set: Iterable[tuple], num_hashes: int = 32) -> Tuple[int, ...]:
    """Deterministic MinHash signature (pure Python, fixed hash seeds)."""
    signature = []
    for seed in range(num_hashes):
        lowest = None
        for ngram in ngram_set:
            digest = hashlib.sha256(
                (str(seed) + ":" + ",".join(str(item) for item in ngram)).encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:8], "big")
            if lowest is None or value < lowest:
                lowest = value
        signature.append(lowest if lowest is not None else 0)
    return tuple(signature)


def minhash_similarity(left: Tuple[int, ...], right: Tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return float("nan")
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def audit_leakage(
    manifests: Sequence[SplitManifest],
    *,
    ngram: int = 13,
    near_duplicate_threshold: float = 0.85,
    benchmark_answer_strings: Optional[Sequence[str]] = None,
    text_lookup: Optional[Callable[[SampleRecord], str]] = None,
) -> Dict[str, Any]:
    """Audit cross-split leakage between all split pairs (E05-03 §5).

    Performs: id intersection, normalized text-hash intersection, token
    n-gram/MinHash near-duplicate detection, prompt-template overlap, parent
    document cross-split check and (optionally) benchmark answer/label string
    contamination. Returns counts plus *identifiers and hashes only* — never
    raw private text.
    """
    by_split: Dict[str, SplitManifest] = {manifest.split: manifest for manifest in manifests}
    findings: List[LeakageFinding] = []
    for index, left_split in enumerate(SPLITS):
        if left_split not in by_split:
            continue
        for right_split in SPLITS[index + 1 :]:
            if right_split not in by_split:
                continue
            left = by_split[left_split]
            right = by_split[right_split]
            findings.extend(
                _audit_pair(left, right, ngram=ngram, threshold=near_duplicate_threshold)
            )
    contamination: List[Dict[str, Any]] = []
    if benchmark_answer_strings:
        needles = [string.strip().lower() for string in benchmark_answer_strings if string]
        if text_lookup is None:
            # Substring contamination cannot be decided from hashes. Recording
            # "not checked, here is why" is the only honest option; claiming a
            # clean audit would be fabricated evidence.
            contamination.append(
                {
                    "checked": False,
                    "reason": "no text_lookup provided; hashes cannot detect "
                    "substring contamination",
                    "needles": len(needles),
                }
            )
        else:
            hits: List[str] = []
            for manifest in manifests:
                for sample in manifest.samples:
                    text = text_lookup(sample).lower()
                    if any(needle in text for needle in needles):
                        hits.append(f"{manifest.split}:{sample.sample_id}")
            contamination.append(
                {"checked": True, "hits": hits, "hit_count": len(hits)}
            )
    report = {
        "pairs_checked": sum(
            1
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
            if left in by_split and right in by_split
        ),
        "findings": [
            {
                "kind": finding.kind,
                "splits": [finding.split_a, finding.split_b],
                "count": finding.count,
                "detail": finding.detail,
            }
            for finding in findings
        ],
        "leaked": any(finding.count > 0 for finding in findings),
        "ngram": ngram,
        "near_duplicate_threshold": near_duplicate_threshold,
        "benchmark_contamination": contamination,
    }
    return report


def _audit_pair(
    left: SplitManifest,
    right: SplitManifest,
    *,
    ngram: int,
    threshold: float,
) -> List[LeakageFinding]:
    findings: List[LeakageFinding] = []

    left_ids = set(left.sample_ids)
    right_ids = set(right.sample_ids)
    shared_ids = left_ids & right_ids
    if shared_ids:
        findings.append(
            LeakageFinding(
                kind="id_intersection",
                split_a=left.split,
                split_b=right.split,
                count=len(shared_ids),
                detail=json.dumps(sorted(shared_ids)[:10]),
            )
        )

    left_text = {sample.text_hash for sample in left.samples}
    right_text = {sample.text_hash for sample in right.samples}
    shared_text = left_text & right_text
    if shared_text:
        findings.append(
            LeakageFinding(
                kind="normalized_text_hash",
                split_a=left.split,
                split_b=right.split,
                count=len(shared_text),
                detail=json.dumps(sorted(shared_text)[:10]),
            )
        )

    left_parent = {sample.parent_id for sample in left.samples if sample.parent_id}
    right_parent = {sample.parent_id for sample in right.samples if sample.parent_id}
    if left_parent & right_parent:
        findings.append(
            LeakageFinding(
                kind="parent_document",
                split_a=left.split,
                split_b=right.split,
                count=len(left_parent & right_parent),
                detail=json.dumps(sorted(left_parent & right_parent)[:10]),
            )
        )

    templates = {sample.template for sample in left.samples if sample.template} & {
        sample.template for sample in right.samples if sample.template
    }
    if templates:
        findings.append(
            LeakageFinding(
                kind="prompt_template_overlap",
                split_a=left.split,
                split_b=right.split,
                count=len(templates),
                detail=json.dumps(sorted(templates)[:10]),
            )
        )

    # Near-duplicate detection is O(n*m) over signatures; for large pools the
    # caller should pre-bucket by length. The implementation records how many
    # pairs were actually compared so the audit cannot be over-claimed.
    right_signatures = [
        (sample.sample_id, _minhash_signature(_ngrams(sample.token_ids, ngram)))
        for sample in right.samples
    ]
    near_dup = 0
    examples: List[str] = []
    for sample in left.samples:
        signature = _minhash_signature(_ngrams(sample.token_ids, ngram))
        for other_id, other_signature in right_signatures:
            if minhash_similarity(signature, other_signature) >= threshold:
                near_dup += 1
                if len(examples) < 10:
                    examples.append(f"{sample.sample_id}~{other_id}")
                break
    if near_dup:
        findings.append(
            LeakageFinding(
                kind="near_duplicate",
                split_a=left.split,
                split_b=right.split,
                count=near_dup,
                detail=json.dumps(examples),
            )
        )
    return findings


# ── data spec and factor sampling (E05-03 §6/§10 step 1/5) ────────────────


@dataclass
class DataSpec:
    """Pre-registered calibration data plan and quality thresholds."""

    name: str
    sources: Sequence[Mapping[str, Any]]
    length_buckets: Mapping[str, Tuple[int, int]]
    sample_counts: Sequence[int] = (8, 16, 32, 64, 128, 256)
    token_budgets: Sequence[int] = (0,)
    subset_seeds: Sequence[int] = (0, 1, 2)
    ngram: int = 13
    near_duplicate_threshold: float = 0.85
    primary_quality_metric: str = "perplexity"
    primary_direction: str = qstats.LOWER_IS_BETTER
    quality_margin: float = 0.0
    stabilization_epsilon: float = 0.0
    seed_variance_threshold: float = 0.0
    offline_cost_ceiling_s: float = 0.0
    notes: str = ""

    def to_json(self) -> str:
        payload = {
            "name": self.name,
            "sources": [dict(source) for source in self.sources],
            "length_buckets": {
                name: list(bounds) for name, bounds in sorted(self.length_buckets.items())
            },
            "sample_counts": list(self.sample_counts),
            "token_budgets": list(self.token_budgets),
            "subset_seeds": list(self.subset_seeds),
            "ngram": self.ngram,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "primary_quality_metric": self.primary_quality_metric,
            "primary_direction": self.primary_direction,
            "quality_margin": self.quality_margin,
            "stabilization_epsilon": self.stabilization_epsilon,
            "seed_variance_threshold": self.seed_variance_threshold,
            "offline_cost_ceiling_s": self.offline_cost_ceiling_s,
            "notes": self.notes,
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def data_spec_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def bucket_of(num_tokens: int, buckets: Mapping[str, Tuple[int, int]]) -> str:
    """Return the length-bucket name for a sample (``"out-of-range"`` if none)."""
    for name, (low, high) in buckets.items():
        if low <= num_tokens < high:
            return name
    return "out-of-range"


def sample_pool(
    pool: Sequence[SampleRecord],
    spec: DataSpec,
    *,
    source: Optional[str] = None,
    domain: Optional[str] = None,
    length_bucket: Optional[str] = None,
) -> List[SampleRecord]:
    """Filter a candidate pool by source/domain/length (deterministic order)."""
    selected = []
    for sample in pool:
        if sample.split != SPLIT_CALIBRATION:
            raise ConfigError(
                f"sample {sample.sample_id!r} is in split {sample.split!r}; only "
                f"{SPLIT_CALIBRATION!r} samples may enter the calibration pool"
            )
        if source is not None and sample.dataset != source:
            continue
        if domain is not None and sample.domain != domain:
            continue
        if length_bucket is not None and sample.length_bucket != length_bucket:
            continue
        selected.append(sample)
    return selected


def draw_subset(
    pool: Sequence[SampleRecord],
    *,
    sample_count: int,
    seed: int,
    token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Draw one immutable subset by deterministic seeded sampling.

    When ``token_budget`` is set the draw stops as soon as the cumulative
    *valid* token count reaches the budget (the actual achieved budget is
    reported; the draw never truncates a sample to hit a number). The returned
    record contains the ids and token hashes so a re-run can prove it drew the
    same subset — replacing a "bad" subset afterwards is therefore detectable.
    """
    if sample_count <= 0:
        raise ConfigError(f"sample_count must be positive, got {sample_count}")
    ordered = sorted(pool, key=lambda sample: sample.sample_id)
    import random

    rng = random.Random(seed)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    chosen: List[SampleRecord] = []
    valid_tokens = 0
    for index in indices:
        if len(chosen) >= sample_count:
            break
        sample = ordered[index]
        if token_budget is not None and valid_tokens + sample.num_tokens > token_budget:
            continue
        chosen.append(sample)
        valid_tokens += sample.num_tokens
    return {
        "subset_seed": seed,
        "requested_sample_count": sample_count,
        "sample_count": len(chosen),
        "valid_tokens": valid_tokens,
        "token_budget": token_budget,
        "sample_ids": [sample.sample_id for sample in chosen],
        "token_hashes": [sample.token_hash for sample in chosen],
        "length_buckets": _count_field(chosen, "length_bucket"),
        "domains": _count_field(chosen, "domain"),
        "languages": _count_field(chosen, "language"),
        "subset_hash": _subset_hash(chosen),
    }


def _count_field(samples: Sequence[SampleRecord], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        key = getattr(sample, field_name) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _subset_hash(samples: Sequence[SampleRecord]) -> str:
    payload = sorted((sample.sample_id, sample.token_hash) for sample in samples)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ── activation statistics (E05-03 §7.1) ───────────────────────────────────


@dataclass
class ModuleStats:
    """Statistics for one module/tensor at one hook point."""

    module: str
    layer_index: Optional[int]
    direction: str  # "input" | "output" | "weight" | "kv" | ...
    shape: Tuple[int, ...]
    valid_count: int
    padded_count: int
    summary: qstats.DistributionSummary
    per_channel_absmax: List[float] = field(default_factory=list)
    nan_count: int = 0
    dtype: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "module": self.module,
            "layer_index": self.layer_index,
            "direction": self.direction,
            "shape": list(self.shape),
            "valid_count": self.valid_count,
            "padded_count": self.padded_count,
            "nan_count": self.nan_count,
            "dtype": self.dtype,
            "per_channel_absmax_count": len(self.per_channel_absmax),
        }
        payload.update(self.summary.as_dict())
        return payload


def collect_activation_stats(
    module_name: str,
    tensor,
    *,
    mask=None,
    layer_index: Optional[int] = None,
    direction: str = "input",
    outlier_sigma: float = 6.0,
) -> ModuleStats:
    """Summarize one activation tensor, excluding padding via ``mask``.

    ``tensor`` may be a torch tensor (lazy import) or any nested sequence of
    floats. Padding *must* be excluded with an explicit mask: E05-03 §13
    forbids padding tokens entering the statistics, and a silently ignored
    mask is not detectable afterwards.
    """
    import math as _math

    values: List[float] = []
    per_channel: List[float] = []
    shape: Tuple[int, ...]
    dtype_name = ""
    padded = 0
    nan_count = 0

    if hasattr(tensor, "detach"):  # torch tensor
        detached = tensor.detach().float().cpu()
        shape = tuple(int(dim) for dim in detached.shape)
        dtype_name = str(tensor.dtype)
        if mask is None:
            rows = detached.reshape(-1, shape[-1]) if detached.dim() > 1 else detached
            for row in rows:
                row_values = [float(value) for value in row.tolist()]
                values.extend(row_values)
                per_channel.append(max((abs(value) for value in row_values), default=0.0))
        else:
            mask_list = mask.detach().to("cpu").reshape(-1).tolist()
            flat = detached.reshape(-1).tolist()
            total = len(flat)
            for index, value in enumerate(flat):
                keep = bool(mask_list[index]) if index < len(mask_list) else False
                if keep:
                    values.append(float(value))
                else:
                    padded += 1
            if total != len(values) + padded:  # pragma: no cover - defensive
                raise ConfigError("mask accounting mismatch")
        nan_count = sum(1 for value in values if _math.isnan(value))
    else:
        flat_nested = _flatten_any(tensor)
        shape = _infer_shape(tensor)
        values = [float(value) for value in flat_nested]
        nan_count = sum(1 for value in values if _math.isnan(value))

    summary = qstats.summarize_distribution(values, outlier_sigma=outlier_sigma)
    return ModuleStats(
        module=module_name,
        layer_index=layer_index,
        direction=direction,
        shape=shape,
        valid_count=len(values),
        padded_count=padded,
        summary=summary,
        per_channel_absmax=per_channel,
        nan_count=nan_count,
        dtype=dtype_name,
    )


def _flatten_any(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            out.extend(_flatten_any(item))
        return out
    return [value]


def _infer_shape(value: Any) -> Tuple[int, ...]:
    shape: List[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else None
    return tuple(shape)


def merge_module_stats(
    left: ModuleStats, right: ModuleStats
) -> ModuleStats:
    """Merge two statistics records for the same module (multi-batch capture).

    Only *compatible* records can merge: a different module/direction/shape
    would silently mix distributions, so it is refused.
    """
    for field_name in ("module", "direction", "shape"):
        if getattr(left, field_name) != getattr(right, field_name):
            raise ConfigError(
                f"cannot merge module stats with different {field_name}: "
                f"{getattr(left, field_name)!r} vs {getattr(right, field_name)!r}"
            )
    count = left.summary.count + right.summary.count
    if count == 0:
        merged_summary = left.summary
    else:
        mean = (
            left.summary.mean * left.summary.count
            + right.summary.mean * right.summary.count
        ) / count
        merged_summary = qstats.summarize_distribution(
            [left.summary.mean] * left.summary.count + [right.summary.mean] * right.summary.count
        )
        merged_summary.mean = mean
    return ModuleStats(
        module=left.module,
        layer_index=left.layer_index if left.layer_index is not None else right.layer_index,
        direction=left.direction,
        shape=left.shape,
        valid_count=left.valid_count + right.valid_count,
        padded_count=left.padded_count + right.padded_count,
        summary=merged_summary,
        per_channel_absmax=[
            max(a, b) for a, b in zip(left.per_channel_absmax, right.per_channel_absmax)
        ]
        if len(left.per_channel_absmax) == len(right.per_channel_absmax)
        else [],
        nan_count=left.nan_count + right.nan_count,
        dtype=left.dtype or right.dtype,
    )


# ── minimum-sufficient-budget selection (E05-03 §8.3/§10 step 12) ─────────


@dataclass
class CandidateResult:
    """One candidate (source x budget x length x seed) evaluated on policy-validation."""

    candidate_id: str
    source: str
    sample_count: int
    valid_tokens: int
    length_coverage: str
    subset_seed: int
    #: Primary quality value on policy-validation (never final-evaluation).
    quality: float
    quality_ci_low: float
    quality_ci_high: float
    slice_quality: Dict[str, float] = field(default_factory=dict)
    statistics_stability: float = float("nan")
    offline_cost_s: float = 0.0
    artifact_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "sample_count": self.sample_count,
            "valid_tokens": self.valid_tokens,
            "length_coverage": self.length_coverage,
            "subset_seed": self.subset_seed,
            "quality": self.quality,
            "quality_ci_low": self.quality_ci_low,
            "quality_ci_high": self.quality_ci_high,
            "slice_quality": dict(self.slice_quality),
            "statistics_stability": self.statistics_stability,
            "offline_cost_s": self.offline_cost_s,
            "artifact_hash": self.artifact_hash,
        }


@dataclass
class SelectionDecision:
    """The pre-registered selection outcome (rule + trace + winner)."""

    rule: Dict[str, Any]
    selected: Optional[str]
    reason: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    stable_candidates: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": dict(self.rule),
            "selected": self.selected,
            "reason": self.reason,
            "trace": list(self.trace),
            "stable_candidates": list(self.stable_candidates),
        }


def select_minimum_sufficient(
    candidates: Sequence[CandidateResult],
    spec: DataSpec,
    *,
    slice_names: Sequence[str] = (),
) -> SelectionDecision:
    """Apply the pre-registered saturation rule (E05-03 §8.3).

    The rule is *not* "the curve looks flat". A candidate is **stable** when

    1. doubling the calibration budget improves the primary quality by less
       than ``spec.stabilization_epsilon`` (measured against the next larger
       budget of the same source/length/seed family), and
    2. the two confidence intervals overlap on the primary metric, and
    3. the between-seed variance of the primary metric is below
       ``spec.seed_variance_threshold``, and
    4. every pre-registered slice passes the quality margin, and
    5. the statistics-stability metric is at or above the pre-registered bar
       (recorded per candidate; ``NaN`` means "not measured" → not stable), and
    6. the offline cost stays below ``spec.offline_cost_ceiling_s``.

    Among the stable candidates the **cheapest** (fewest samples, then fewest
    tokens) is selected. Selection consumes policy-validation candidates only;
    the function has no access to final-evaluation numbers, which makes "tune
    on test" structurally impossible here.
    """
    if not candidates:
        return SelectionDecision(
            rule=spec.to_json() and json.loads(spec.to_json()),
            selected=None,
            reason="no candidates evaluated",
            trace=[],
        )

    by_key: Dict[Tuple[str, str, int], List[CandidateResult]] = {}
    for candidate in candidates:
        key = (candidate.source, candidate.length_coverage, candidate.subset_seed)
        by_key.setdefault(key, []).append(candidate)
    for family in by_key.values():
        family.sort(key=lambda item: (item.sample_count, item.valid_tokens))

    stability: Dict[str, bool] = {}
    trace: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        checks: Dict[str, bool] = {}
        family = by_key[(candidate.source, candidate.length_coverage, candidate.subset_seed)]
        larger = [
            other
            for other in family
            if (other.sample_count, other.valid_tokens)
            > (candidate.sample_count, candidate.valid_tokens)
        ]
        if larger:
            next_budget = min(
                larger, key=lambda item: (item.sample_count, item.valid_tokens)
            )
            improvement = _oriented_improvement(
                next_budget.quality, candidate.quality, spec.primary_direction
            )
            checks["budget_saturation"] = abs(improvement) < spec.stabilization_epsilon
            checks["ci_overlap"] = (
                candidate.quality_ci_low <= next_budget.quality_ci_high
                and next_budget.quality_ci_low <= candidate.quality_ci_high
            )
        else:
            checks["budget_saturation"] = False
            checks["ci_overlap"] = False

        same_budget = [
            other
            for other in candidates
            if other.source == candidate.source
            and other.sample_count == candidate.sample_count
            and other.length_coverage == candidate.length_coverage
        ]
        if len(same_budget) >= 2:
            values = [other.quality for other in same_budget]
            spread = max(values) - min(values)
            checks["seed_variance"] = spread < spec.seed_variance_threshold
        else:
            checks["seed_variance"] = False

        slice_ok = True
        for name in slice_names:
            value = candidate.slice_quality.get(name)
            if value is None:
                slice_ok = False
                break
            improvement = _oriented_improvement(value, 0.0, spec.primary_direction)
            if improvement < -spec.quality_margin:
                slice_ok = False
                break
        checks["slices"] = slice_ok
        checks["statistics_stability"] = (
            not _is_nan(candidate.statistics_stability)
            and candidate.statistics_stability >= 0.0
        )
        checks["offline_cost"] = (
            spec.offline_cost_ceiling_s <= 0.0
            or candidate.offline_cost_s <= spec.offline_cost_ceiling_s
        )
        stable = all(checks.values())
        stability[candidate.candidate_id] = stable
        trace.append(
            {
                "candidate_id": candidate.candidate_id,
                "checks": checks,
                "stable": stable,
                "budget": {
                    "sample_count": candidate.sample_count,
                    "valid_tokens": candidate.valid_tokens,
                },
                "quality": candidate.quality,
                "quality_ci": [candidate.quality_ci_low, candidate.quality_ci_high],
                "reason": "; ".join(
                    name for name, ok in checks.items() if not ok
                )
                or "all pre-registered conditions satisfied",
            }
        )

    stable_ids = [cid for cid, stable in stability.items() if stable]
    if not stable_ids:
        return SelectionDecision(
            rule=json.loads(spec.to_json()),
            selected=None,
            reason=(
                "no candidate satisfied the pre-registered saturation rule; the "
                "minimum sufficient budget is not established by these candidates"
            ),
            trace=trace,
            stable_candidates=[],
        )
    stable_candidates = [c for c in candidates if c.candidate_id in stable_ids]
    winner = min(
        stable_candidates,
        key=lambda item: (item.sample_count, item.valid_tokens, item.offline_cost_s),
    )
    return SelectionDecision(
        rule=json.loads(spec.to_json()),
        selected=winner.candidate_id,
        reason=(
            f"cheapest candidate satisfying all pre-registered saturation "
            f"conditions ({winner.sample_count} samples, "
            f"{winner.valid_tokens} valid tokens, {winner.subset_seed=})"
        ),
        trace=trace,
        stable_candidates=sorted(stable_ids),
    )


def _oriented_improvement(
    candidate: float, baseline: float, direction: str
) -> float:
    delta = candidate - baseline
    return delta if direction == qstats.HIGHER_IS_BETTER else -delta


def _is_nan(value: float) -> bool:
    return value != value


# ── offline cost accounting (E05-03 §10 step 16) ──────────────────────────


@dataclass
class OfflineCost:
    """One offline phase measurement (never mixed into online latency)."""

    phase: str
    wall_time_s: float
    host_peak_bytes: int = 0
    device_peak_bytes: int = 0
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "wall_time_s": self.wall_time_s,
            "host_peak_bytes": self.host_peak_bytes,
            "device_peak_bytes": self.device_peak_bytes,
            "notes": self.notes,
        }


OFFLINE_PHASES = (
    "data_loading",
    "stat_collection",
    "search",
    "quantization",
    "pack",
    "save",
    "load",
    "repack",
    "compile",
)


def total_offline_cost(costs: Sequence[OfflineCost]) -> Dict[str, Any]:
    """Sum offline costs by phase, keeping device peak as a maximum."""
    by_phase: Dict[str, float] = {}
    device_peak = 0
    host_peak = 0
    for cost in costs:
        if cost.phase not in OFFLINE_PHASES:
            raise ConfigError(
                f"unknown offline phase {cost.phase!r}; supported: "
                f"{list(OFFLINE_PHASES)}"
            )
        by_phase[cost.phase] = by_phase.get(cost.phase, 0.0) + cost.wall_time_s
        device_peak = max(device_peak, cost.device_peak_bytes)
        host_peak = max(host_peak, cost.host_peak_bytes)
    return {
        "by_phase_s": dict(sorted(by_phase.items())),
        "total_s": sum(by_phase.values()),
        "host_peak_bytes": host_peak,
        "device_peak_bytes": device_peak,
    }


def amortized_cost_per_request(
    offline_cost_s: float, served_requests: int
) -> float:
    """Amortize an offline cost; the unamortized value stays in the record."""
    if served_requests <= 0:
        raise ConfigError(
            f"served_requests must be positive, got {served_requests}"
        )
    return offline_cost_s / served_requests


__all__ = [
    "CandidateResult",
    "DataSpec",
    "LeakageFinding",
    "ModuleStats",
    "NON_CALIBRATION_SPLITS",
    "OFFLINE_PHASES",
    "OfflineCost",
    "SPLITS",
    "SPLIT_CALIBRATION",
    "SPLIT_FINAL_EVALUATION",
    "SPLIT_POLICY_VALIDATION",
    "SPLIT_STRESS",
    "SampleRecord",
    "SelectionDecision",
    "SplitManifest",
    "amortized_cost_per_request",
    "audit_leakage",
    "bucket_of",
    "collect_activation_stats",
    "draw_subset",
    "merge_module_stats",
    "minhash_similarity",
    "normalized_text_hash",
    "sample_pool",
    "select_minimum_sufficient",
    "token_ids_hash",
    "total_offline_cost",
]
