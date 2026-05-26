"""Strict YAML loaders for the S05 quantization configs.

Every shipped config declares a ``kind`` so a loader can refuse a document
that was written for another purpose. Unknown keys are rejected (a typo would
otherwise silently keep a default), and every loader returns a validated
object — no config is "close enough".
"""

from __future__ import annotations

import os
from typing import Any, Dict

from hqsb.core.errors import ConfigError
from hqsb.quant.activation import ActivationQuantSpec
from hqsb.quant.calibration import DataSpec
from hqsb.quant.kv import KvQuantSpec
from hqsb.quant.policy import MixedPrecisionPolicy
from hqsb.quant.spec import QuantScheme

KIND_SCHEME = "hqsb.quant.scheme"
KIND_SCHEME_MATRIX = "hqsb.quant.scheme_matrix"
KIND_CALIBRATION = "hqsb.quant.calibration_spec"
KIND_ACTIVATION = "hqsb.quant.activation_spec"
KIND_KV = "hqsb.quant.kv_spec"
KIND_POLICY = "hqsb.quant.mixed_precision_policy"
KIND_KERNEL = "hqsb.quant.kernel_spec"
KIND_DECISION = "hqsb.quant.decision_spec"

KINDS = (
    KIND_SCHEME,
    KIND_SCHEME_MATRIX,
    KIND_CALIBRATION,
    KIND_ACTIVATION,
    KIND_KV,
    KIND_POLICY,
    KIND_KERNEL,
    KIND_DECISION,
)


def load_yaml_document(path: str) -> Dict[str, Any]:
    import yaml

    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: YAML root must be a mapping")
    kind = payload.get("kind")
    if kind not in KINDS:
        raise ConfigError(
            f"{path}: kind {kind!r} is not one of {list(KINDS)}"
        )
    return payload


def load_quant_scheme(path: str) -> QuantScheme:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_SCHEME:
        raise ConfigError(f"{path}: expected kind {KIND_SCHEME!r}, got {payload['kind']!r}")
    unknown = set(payload) - {"kind", "description", "scheme"}
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    return QuantScheme.from_mapping(payload["scheme"])


def load_scheme_matrix(path: str) -> Dict[str, Any]:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_SCHEME_MATRIX:
        raise ConfigError(
            f"{path}: expected kind {KIND_SCHEME_MATRIX!r}, got {payload['kind']!r}"
        )
    allowed = {"kind", "description", "bits", "granularities", "symmetric", "group_sizes", "shape_relative_group_sizes"}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    return payload


def load_calibration_spec(path: str) -> DataSpec:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_CALIBRATION:
        raise ConfigError(
            f"{path}: expected kind {KIND_CALIBRATION!r}, got {payload['kind']!r}"
        )
    allowed = {
        "kind",
        "description",
        "name",
        "sources",
        "length_buckets",
        "sample_counts",
        "token_budgets",
        "subset_seeds",
        "ngram",
        "near_duplicate_threshold",
        "primary_quality_metric",
        "primary_direction",
        "quality_margin",
        "stabilization_epsilon",
        "seed_variance_threshold",
        "offline_cost_ceiling_s",
        "notes",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    lengths = {
        name: tuple(bounds) for name, bounds in (payload.get("length_buckets") or {}).items()
    }
    return DataSpec(
        name=str(payload.get("name", os.path.basename(path))),
        sources=payload.get("sources") or (),
        length_buckets=lengths,
        sample_counts=tuple(payload.get("sample_counts") or (8, 16, 32, 64, 128, 256)),
        token_budgets=tuple(payload.get("token_budgets") or (0,)),
        subset_seeds=tuple(payload.get("subset_seeds") or (0, 1, 2)),
        ngram=int(payload.get("ngram", 13)),
        near_duplicate_threshold=float(payload.get("near_duplicate_threshold", 0.85)),
        primary_quality_metric=str(payload.get("primary_quality_metric", "perplexity")),
        primary_direction=str(payload.get("primary_direction", "lower_is_better")),
        quality_margin=float(payload.get("quality_margin", 0.0)),
        stabilization_epsilon=float(payload.get("stabilization_epsilon", 0.0)),
        seed_variance_threshold=float(payload.get("seed_variance_threshold", 0.0)),
        offline_cost_ceiling_s=float(payload.get("offline_cost_ceiling_s", 0.0)),
        notes=str(payload.get("notes", "")),
    )


def load_activation_spec(path: str) -> ActivationQuantSpec:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_ACTIVATION:
        raise ConfigError(
            f"{path}: expected kind {KIND_ACTIVATION!r}, got {payload['kind']!r}"
        )
    unknown = set(payload) - {"kind", "description", "spec"}
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    return ActivationQuantSpec(**payload["spec"])


def load_kv_spec(path: str) -> KvQuantSpec:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_KV:
        raise ConfigError(f"{path}: expected kind {KIND_KV!r}, got {payload['kind']!r}")
    unknown = set(payload) - {"kind", "description", "spec"}
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    return KvQuantSpec(**payload["spec"])


def load_policy(path: str) -> MixedPrecisionPolicy:
    """Load a mixed-precision policy (validated on load, not on use)."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return MixedPrecisionPolicy.from_yaml(text)


def load_kernel_spec(path: str) -> Dict[str, Any]:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_KERNEL:
        raise ConfigError(
            f"{path}: expected kind {KIND_KERNEL!r}, got {payload['kind']!r}"
        )
    required = {"kind", "kernel_id", "provider", "layouts", "bits", "group_sizes"}
    missing = required - set(payload)
    if missing:
        raise ConfigError(f"{path}: kernel spec is missing {sorted(missing)}")
    return payload


def load_decision_spec(path: str) -> Dict[str, Any]:
    payload = load_yaml_document(path)
    if payload["kind"] != KIND_DECISION:
        raise ConfigError(
            f"{path}: expected kind {KIND_DECISION!r}, got {payload['kind']!r}"
        )
    required = {"kind", "hardware", "workloads", "objectives", "scenarios", "quality_gate"}
    missing = required - set(payload)
    if missing:
        raise ConfigError(f"{path}: decision spec is missing {sorted(missing)}")
    return payload


def load_any(path: str) -> Dict[str, Any]:
    """Load by declared kind and return a normalized summary."""
    payload = load_yaml_document(path)
    kind = payload["kind"]
    summary: Dict[str, Any] = {"path": path, "kind": kind}
    if kind == KIND_SCHEME:
        summary["object"] = load_quant_scheme(path)
    elif kind == KIND_SCHEME_MATRIX:
        summary["object"] = load_scheme_matrix(path)
    elif kind == KIND_CALIBRATION:
        summary["object"] = load_calibration_spec(path)
    elif kind == KIND_ACTIVATION:
        summary["object"] = load_activation_spec(path)
    elif kind == KIND_KV:
        summary["object"] = load_kv_spec(path)
    elif kind == KIND_POLICY:
        summary["object"] = load_policy(path)
    elif kind == KIND_KERNEL:
        summary["object"] = load_kernel_spec(path)
    elif kind == KIND_DECISION:
        summary["object"] = load_decision_spec(path)
    return summary


__all__ = [
    "KINDS",
    "KIND_ACTIVATION",
    "KIND_CALIBRATION",
    "KIND_DECISION",
    "KIND_KERNEL",
    "KIND_KV",
    "KIND_POLICY",
    "KIND_SCHEME",
    "KIND_SCHEME_MATRIX",
    "load_activation_spec",
    "load_any",
    "load_calibration_spec",
    "load_decision_spec",
    "load_kernel_spec",
    "load_kv_spec",
    "load_policy",
    "load_quant_scheme",
    "load_scheme_matrix",
    "load_yaml_document",
]
