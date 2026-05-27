"""Intervention units for sensitivity analysis (E05-05 §4/§11 step 1).

A sensitivity experiment must intervene on *real* module boundaries, not on a
hand-written list of layer indices. This module derives units from the model's
module tree and intersects them with the backend capability constraints
(fused QKV, shared/tied weights, per-group granularity), so a mixed-precision
policy can never select a configuration the kernel cannot execute
(E05-05 §4, §14 "不考虑fused/shared/kernel约束").

Units are logical groups:

``unit_id``      stable identifier (e.g. ``layer.3.mlp.gate_proj``);
``module_paths`` the physical modules it covers;
``constraint``   ``independent``, ``fused_group:<name>`` or ``tied:<name>``;
``legal_configs`` bits/group/method combinations the backend supports;
``shared_with``  unit ids that must move together when a caller models a
                 fused/tied constraint as several units; HQSB's default
                 builder instead collapses such a group into one unit, so this
                 stays empty there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Roles a unit can play in the model (E05-05 §4).
ROLE_ATTENTION_QKV = "attention_qkv"
ROLE_ATTENTION_OUT = "attention_out"
ROLE_MLP_GATE_UP = "mlp_gate_up"
ROLE_MLP_DOWN = "mlp_down"
ROLE_EMBEDDING = "embedding"
ROLE_LM_HEAD = "lm_head"
ROLE_NORM = "norm"
ROLE_OTHER = "other"

ROLES = (
    ROLE_ATTENTION_QKV,
    ROLE_ATTENTION_OUT,
    ROLE_MLP_GATE_UP,
    ROLE_MLP_DOWN,
    ROLE_EMBEDDING,
    ROLE_LM_HEAD,
    ROLE_NORM,
    ROLE_OTHER,
)

CONSTRAINT_INDEPENDENT = "independent"
CONSTRAINT_FUSED = "fused_group"
CONSTRAINT_TIED = "tied"


@dataclass
class UnitConfig:
    """One legal configuration for a unit (from the backend capability)."""

    bits: int
    group_size: Optional[int]
    method: str
    scheme_hash: str = ""
    packed_layout: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "method": self.method,
            "scheme_hash": self.scheme_hash,
            "packed_layout": self.packed_layout,
        }


@dataclass
class InterventionUnit:
    """One intervention unit of the sensitivity/mixed-precision search."""

    unit_id: str
    role: str
    module_paths: Tuple[str, ...]
    parameter_count: int = 0
    constraint: str = CONSTRAINT_INDEPENDENT
    shared_with: Tuple[str, ...] = ()
    legal_configs: Tuple[UnitConfig, ...] = ()
    #: Backend constraint that forced a fused/tied grouping, for the report.
    constraint_reason: str = ""
    layer_index: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "role": self.role,
            "module_paths": list(self.module_paths),
            "parameter_count": self.parameter_count,
            "constraint": self.constraint,
            "shared_with": list(self.shared_with),
            "legal_configs": [config.as_dict() for config in self.legal_configs],
            "constraint_reason": self.constraint_reason,
            "layer_index": self.layer_index,
        }


def _role_for(name: str) -> str:
    lowered = name.lower()
    if "q_proj" in lowered or "k_proj" in lowered or "v_proj" in lowered:
        return ROLE_ATTENTION_QKV
    if "o_proj" in lowered:
        return ROLE_ATTENTION_OUT
    if "gate_proj" in lowered or "up_proj" in lowered:
        return ROLE_MLP_GATE_UP
    if "down_proj" in lowered:
        return ROLE_MLP_DOWN
    if "embed" in lowered:
        return ROLE_EMBEDDING
    if "lm_head" in lowered:
        return ROLE_LM_HEAD
    if "norm" in lowered or "ln_" in lowered:
        return ROLE_NORM
    return ROLE_OTHER


def _layer_index_for(name: str) -> Optional[int]:
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def build_units(
    weight_rows: Sequence[Mapping[str, Any]],
    *,
    legal_configs: Sequence[UnitConfig],
    fused_groups: Optional[Mapping[str, Sequence[str]]] = None,
    tied_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[InterventionUnit]:
    """Derive intervention units from coverage rows plus capability constraints.

    Args:
        weight_rows: rows produced by :mod:`hqsb.quant.coverage` (each with
            ``name``, ``num_parameters``, ``quantized``).
        legal_configs: configurations the backend can execute.
        fused_groups: named groups of module names that must share one
            configuration (e.g. a fused QKV kernel).
        tied_groups: names that share storage (tied embeddings/lm_head).
    """
    if not legal_configs:
        raise ConfigError(
            "at least one legal configuration is required; an empty capability "
            "set would make every policy non-executable"
        )
    fused_groups = {name: tuple(paths) for name, paths in (fused_groups or {}).items()}
    tied_groups = {name: tuple(paths) for name, paths in (tied_groups or {}).items()}

    path_to_fused = {}
    for group_name, paths in fused_groups.items():
        for path in paths:
            path_to_fused[path] = group_name
    path_to_tied = {}
    for group_name, paths in tied_groups.items():
        for path in paths:
            path_to_tied[path] = group_name

    units: Dict[str, InterventionUnit] = {}
    for row in weight_rows:
        name = str(row.get("name", ""))
        if not name.endswith(".weight"):
            continue
        module_path = name[: -len(".weight")]
        parameter_count = int(row.get("num_parameters", 0) or 0)

        fused_group = path_to_fused.get(module_path)
        tied_group = path_to_tied.get(module_path)
        if fused_group:
            unit_id = f"fused:{fused_group}"
            constraint = CONSTRAINT_FUSED
            members = fused_groups[fused_group]
            reason = f"backend executes fused group {fused_group!r} as one unit"
        elif tied_group:
            unit_id = f"tied:{tied_group}"
            constraint = CONSTRAINT_TIED
            members = tied_groups[tied_group]
            reason = f"weights are tied ({tied_group!r}); they must share a config"
        else:
            unit_id = module_path
            constraint = CONSTRAINT_INDEPENDENT
            members = (module_path,)
            reason = ""

        unit = units.get(unit_id)
        if unit is None:
            units[unit_id] = InterventionUnit(
                unit_id=unit_id,
                role=_role_for(module_path),
                module_paths=tuple(sorted(members)),
                parameter_count=parameter_count,
                constraint=constraint,
                # A fused/tied group collapses into a *single* unit, so there
                # are no peer units to coordinate with: the constraint is
                # enforced structurally (one unit -> one configuration). The
                # field stays for callers that model distinct units sharing
                # one physical weight.
                shared_with=(),
                legal_configs=tuple(legal_configs),
                constraint_reason=reason,
                layer_index=_layer_index_for(module_path),
            )
        else:
            units[unit_id] = InterventionUnit(
                unit_id=unit.unit_id,
                role=unit.role,
                module_paths=unit.module_paths,
                parameter_count=unit.parameter_count + parameter_count,
                constraint=unit.constraint,
                shared_with=unit.shared_with,
                legal_configs=unit.legal_configs,
                constraint_reason=unit.constraint_reason,
                layer_index=unit.layer_index,
            )
    return [units[key] for key in sorted(units)]


def units_from_coverage(
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    legal_configs: Sequence[UnitConfig],
    fused_groups: Optional[Mapping[str, Sequence[str]]] = None,
    tied_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[InterventionUnit]:
    """Convenience wrapper that also drops explicitly excluded tensors.

    Only units that the coverage policy selected are returned; excluded
    tensors are not intervention candidates (their exclusion is recorded in
    the coverage table instead).
    """
    selected = [row for row in coverage_rows if row.get("quantized")]
    return build_units(
        selected,
        legal_configs=legal_configs,
        fused_groups=fused_groups,
        tied_groups=tied_groups,
    )


def units_to_json(units: Sequence[InterventionUnit]) -> str:
    return json.dumps(
        [unit.as_dict() for unit in units], sort_keys=True, indent=2, ensure_ascii=False
    )


def units_by_role(units: Sequence[InterventionUnit]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for unit in units:
        grouped.setdefault(unit.role, []).append(unit.unit_id)
    return {role: sorted(ids) for role, ids in sorted(grouped.items())}


__all__ = [
    "CONSTRAINT_FUSED",
    "CONSTRAINT_INDEPENDENT",
    "CONSTRAINT_TIED",
    "InterventionUnit",
    "ROLES",
    "ROLE_ATTENTION_OUT",
    "ROLE_ATTENTION_QKV",
    "ROLE_EMBEDDING",
    "ROLE_LM_HEAD",
    "ROLE_MLP_DOWN",
    "ROLE_MLP_GATE_UP",
    "ROLE_NORM",
    "ROLE_OTHER",
    "UnitConfig",
    "build_units",
    "units_by_role",
    "units_from_coverage",
    "units_to_json",
]
