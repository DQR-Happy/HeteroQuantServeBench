"""Model/backend adapter boundary, anti-hard-coding and reuse audit (E06-11).

S06's mechanism must depend on stable Operator/Backend/Quant/Result/Trace
contracts — not on a Qwen class name, a module path, a hard-coded head count, a
single CUDA function or the current graph node naming (E06-11 §1).

This module makes that checkable instead of aspirational:

* :class:`ModelAdapter` / :class:`BackendAdapter` protocols — the *only* place
  model/backend specifics may live;
* :func:`hardcode_scan` — a static scanner with rule ids, so "core has no model
  names" becomes a test rather than a review opinion;
* :class:`ChangeBudget` — every core change must be classified; a
  model/backend-specific leak inside core is a failure signal, not progress;
* :class:`DummyBackendAdapter` — a spy backend that exercises the backend
  contract **without** producing hardware performance claims;
* :class:`ReuseMatrix` — what was reused, extended, or refused, per target.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from hqsb.core.errors import ConfigError

# ── model census ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModuleCensusEntry:
    """One module discovered at runtime (never hand-written)."""

    path: str
    module_type: str
    parameter_count: int = 0
    dtype: str = ""
    device: str = ""
    shape: Tuple[int, ...] = ()
    stride: Tuple[int, ...] = ()
    tied_to: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "module_type": self.module_type,
            "parameter_count": self.parameter_count,
            "dtype": self.dtype,
            "device": self.device,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "tied_to": self.tied_to,
        }


def model_census(model: Any, *, max_modules: int = 0) -> Tuple[ModuleCensusEntry, ...]:
    """Enumerate modules/params at runtime; importing torch is the caller's choice."""
    entries: List[ModuleCensusEntry] = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise ConfigError(
            "model_census requires a torch-like nn.Module (named_modules missing); "
            "import torch in the runner, not in this module",
            details={"field": "model"},
        )
    tied_map: Dict[int, str] = {}
    for name, module in named_modules():
        params = list(getattr(module, "parameters", lambda recurse=False: [])())
        if not params:
            continue
        weight = params[0]
        shape = tuple(int(dim) for dim in getattr(weight, "shape", ()))
        stride = tuple(int(dim) for dim in getattr(weight, "stride", lambda: ())())
        key = id(weight)
        tied_to = tied_map.get(key, "")
        if not tied_to:
            tied_map[key] = name
        entries.append(
            ModuleCensusEntry(
                path=name,
                module_type=type(module).__name__,
                parameter_count=sum(int(param.numel()) for param in params),
                dtype=str(getattr(weight, "dtype", "")),
                device=str(getattr(weight, "device", "")),
                shape=shape,
                stride=stride,
                tied_to=tied_to,
            )
        )
        if max_modules and len(entries) >= max_modules:
            break
    return tuple(entries)


# ── adapter protocols ─────────────────────────────────────────────────────


@runtime_checkable
class ModelAdapter(Protocol):
    """Everything model-specific lives here (E06-11 §2)."""

    name: str

    def module_census(self, model: Any) -> Tuple[ModuleCensusEntry, ...]:
        ...

    def map_config(self, config: Any) -> Mapping[str, Any]:
        ...

    def state_dict_policy(self) -> Mapping[str, Any]:
        ...

    def attention_semantics(self) -> Mapping[str, Any]:
        ...


@runtime_checkable
class BackendAdapter(Protocol):
    """Everything backend-specific lives here (E06-11 §2)."""

    name: str

    def capability(self) -> Mapping[str, Any]:
        ...

    def route(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def error_mapping(self) -> Mapping[str, str]:
        ...

    def observed_implementation(self) -> str:
        ...


# ── core contract snapshot ────────────────────────────────────────────────


@dataclass(frozen=True)
class CoreContractSnapshot:
    """The public surface a second target may rely on (E06-11 step 2)."""

    public_api: Tuple[str, ...]
    registries: Tuple[str, ...]
    patterns: Tuple[str, ...]
    forbidden_hardcode_rules: Tuple[str, ...]
    version: str = "1.0.0"

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "public_api": list(self.public_api),
                "registries": list(self.registries),
                "patterns": list(self.patterns),
                "rules": list(self.forbidden_hardcode_rules),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "public_api": list(self.public_api),
            "registries": list(self.registries),
            "patterns": list(self.patterns),
            "forbidden_hardcode_rules": list(self.forbidden_hardcode_rules),
            "digest": self.digest,
        }


def core_contract_snapshot() -> CoreContractSnapshot:
    """Snapshot the integration core surface (imported lazily to avoid cycles)."""
    from hqsb.integration import patterns

    return CoreContractSnapshot(
        public_api=(
            "hqsb.integration.specs.OpSchema",
            "hqsb.integration.dispatch.RegistrationMatrix",
            "hqsb.integration.meta.MetadataContract",
            "hqsb.integration.graph.Graph",
            "hqsb.integration.patterns.PatternSpec",
            "hqsb.integration.lowering.LoweringRegistry",
            "hqsb.integration.taxonomy.FallbackRegistry",
            "hqsb.integration.telemetry.TraceCollector",
        ),
        registries=(
            "hqsb.integration.dispatch.RegistrationMatrix",
            "hqsb.integration.patterns.PatternRegistry",
            "hqsb.integration.lowering.LoweringRegistry",
        ),
        patterns=tuple(spec.pattern_id for spec in patterns.frozen_patterns()),
        forbidden_hardcode_rules=tuple(rule.id for rule in HARDCODE_RULES),
    )


# ── hard-code scanner ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardcodeRule:
    """One static rule with a regex and how to classify a hit."""

    id: str
    description: str
    pattern: str
    change_class: str
    adapter_allowed: bool = False
    note: str = ""

    def compiled(self) -> "re.Pattern[str]":
        return re.compile(self.pattern)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "pattern": self.pattern,
            "change_class": self.change_class,
            "adapter_allowed": self.adapter_allowed,
            "note": self.note,
        }


class ChangeClass:
    GENERIC_BUG_FIX = "generic_bug_fix"
    CONTRACT_EXTENSION = "contract_extension"
    MODEL_SPECIFIC_LEAK = "model_specific_leak"
    BACKEND_SPECIFIC_LEAK = "backend_specific_leak"

    ALL = (GENERIC_BUG_FIX, CONTRACT_EXTENSION, MODEL_SPECIFIC_LEAK, BACKEND_SPECIFIC_LEAK)

    #: Classes that must never appear inside core code.
    FORBIDDEN_IN_CORE = (MODEL_SPECIFIC_LEAK, BACKEND_SPECIFIC_LEAK)


HARDCODE_RULES: Tuple[HardcodeRule, ...] = (
    HardcodeRule(
        "HC-MODEL-CLASS",
        "model implementation class names in core",
        r"\b(Qwen|Llama|Mistral|GPTNeoX|Qwen2|Qwen3)\w*(Model|ForCausalLM|Attention|MLP|Block|DecoderLayer|RMSNorm)\b",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-MODEL-NAME",
        "model family string literal in core",
        r"[\"'](?:qwen|llama|mistral|gpt)[\w\-\./]*[\"']",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-MODULE-PATH",
        "module path string literal in core",
        # ``layers`` alone is a legitimate vocabulary word (cache layers);
        # the rule requires an actual path continuation or a submodule name.
        r"[\"'](?:model\.)?(?:layers\.\d|layers\.[a-z]|self_attn|mlp\.|input_layernorm"
        r"|post_attention_layernorm|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r"[\w\.]*[\"']",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-FIXED-LAYERS",
        "fixed layer/head/group count in core",
        r"\b(num_hidden_layers|num_attention_heads|num_key_value_heads|n_layers|num_layers)\s*=\s*\d+",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-ISINSTANCE-THIRDPARTY",
        "isinstance against a concrete third-party module type",
        r"isinstance\(\s*\w+\s*,\s*(?:Qwen|Llama|Mistral)\w*",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-BACKEND-BRANCH",
        "backend name if/elif chain in core selection",
        r"(?:if|elif)\s+(?:backend|provider|impl)\s*==\s*[\"'](?:cuda|triton|cutlass|ascend|inductor)[\"']",
        ChangeClass.BACKEND_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-NODE-NAME",
        "graph node name matching in core",
        r"name\s*==\s*[\"'](?:add|mul|rms_norm|add_\d+|mul_\d+|view_\d+)[\"']",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-SHAPE-CONSTANT",
        "shape/hidden constants in core",
        r"\b(?:hidden_size|intermediate_size|head_dim)\s*=\s*\d{3,}",
        ChangeClass.MODEL_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-ENV-FLAG",
        "global environment flag used as behaviour switch",
        r"os\.environ(?:\.get)?\(\s*[\"']HQSB_[A-Z_]+[\"']",
        ChangeClass.BACKEND_SPECIFIC_LEAK,
    ),
    HardcodeRule(
        "HC-ERROR-TEXT",
        "error text instead of a reason code",
        r"raise\s+\w*Error\(\s*f?[\"'](?:unsupported|not supported|failed)\b",
        ChangeClass.CONTRACT_EXTENSION,
    ),
)

#: Directories where model/backend specifics are expected and versioned.
ADAPTER_PATH_MARKERS = ("adapters", "backends/", "hqsb/backends/", "tests/")

#: A line carrying this marker is skipped by :func:`hardcode_scan`.  The skip is
#: *counted and reported* (``allowed_lines``), so an exception stays visible
#: instead of silently weakening the rule — the same discipline the project
#: applies to every other relaxation.
ALLOW_MARKER = "hqsb-hardcode-allow"


@dataclass(frozen=True)
class HardcodeFinding:
    rule_id: str
    file: str
    line: int
    snippet: str
    classification: str
    in_adapter: bool

    @property
    def is_leak(self) -> bool:
        return (
            self.classification in ChangeClass.FORBIDDEN_IN_CORE and not self.in_adapter
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "file": self.file,
            "line": self.line,
            "snippet": self.snippet,
            "classification": self.classification,
            "in_adapter": self.in_adapter,
            "is_leak": self.is_leak,
        }


def _iter_python_files(paths: Sequence[str]) -> Sequence[str]:
    collected: List[str] = []
    for path in paths:
        if os.path.isfile(path):
            if path.endswith(".py"):
                collected.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name not in ("__pycache__", ".venv")]
            for filename in filenames:
                if filename.endswith(".py"):
                    collected.append(os.path.join(dirpath, filename))
    return sorted(collected)


def hardcode_scan(paths: Sequence[str], *, adapter_markers: Sequence[str] = ADAPTER_PATH_MARKERS) -> Tuple[HardcodeFinding, ...]:
    """Scan files for frozen hard-coding rules, classifying each hit.

    A hit inside an adapter directory is recorded but not a leak: the adapter
    exists to absorb model differences (E06-11 §12).  Every finding keeps the
    rule id, file and line so it can be re-checked.
    """
    findings: List[HardcodeFinding] = []
    compiled = [(rule, rule.compiled()) for rule in HARDCODE_RULES]
    for file_path in _iter_python_files(paths):
        normalized = file_path.replace(os.sep, "/")
        in_adapter = any(marker in normalized for marker in adapter_markers)
        try:
            with open(file_path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError:  # pragma: no cover - unreadable file
            continue
        for line_number, line in enumerate(lines, start=1):
            if ALLOW_MARKER in line:
                # Explicit, counted exception (e.g. a denylist that must spell
                # the tokens it forbids).
                findings.append(
                    HardcodeFinding(
                        rule_id="ALLOWED",
                        file=file_path,
                        line=line_number,
                        snippet=line.strip()[:120],
                        classification="allowed_line",
                        in_adapter=True,
                    )
                )
                continue
            for rule, pattern in compiled:
                if pattern.search(line):
                    findings.append(
                        HardcodeFinding(
                            rule_id=rule.id,
                            file=file_path,
                            line=line_number,
                            snippet=line.strip()[:120],
                            classification=rule.change_class,
                            in_adapter=in_adapter,
                        )
                    )
    return tuple(findings)


def hardcode_report(findings: Sequence[HardcodeFinding]) -> Dict[str, Any]:
    """Summary: leaks vs adapter-allowed hits, grouped by rule."""
    leaks = [item for item in findings if item.is_leak]
    allowed = [item for item in findings if item.rule_id == "ALLOWED"]
    by_rule: Dict[str, int] = {}
    for item in findings:
        by_rule[item.rule_id] = by_rule.get(item.rule_id, 0) + 1
    return {
        "ok": not leaks,
        "findings": len(findings),
        "leaks": len(leaks),
        "by_rule": by_rule,
        "leak_details": [item.as_dict() for item in leaks],
        "adapter_allowed": sum(1 for item in findings if item.in_adapter),
        "allowed_lines": [item.as_dict() for item in allowed],
    }


# ── change budget ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChangeBudget:
    """One changed file with its size delta and classification."""

    path: str
    loc_added: int
    loc_removed: int
    change_class: str
    reason: str = ""
    target: str = ""

    def __post_init__(self) -> None:
        if self.change_class not in ChangeClass.ALL:
            raise ConfigError(
                f"{self.path}: unknown change class {self.change_class!r}",
                details={"field": "change_class", "allowed": list(ChangeClass.ALL)},
            )

    @property
    def net_loc(self) -> int:
        return self.loc_added - self.loc_removed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "loc_added": self.loc_added,
            "loc_removed": self.loc_removed,
            "net_loc": self.net_loc,
            "change_class": self.change_class,
            "reason": self.reason,
            "target": self.target,
        }


def summarize_changes(budgets: Sequence[ChangeBudget]) -> Dict[str, Any]:
    """Totals per class; a leak inside core is a failure signal (E06-11 §8)."""
    totals: Dict[str, int] = {name: 0 for name in ChangeClass.ALL}
    for item in budgets:
        totals[item.change_class] += item.net_loc
    leaks = [
        item.as_dict()
        for item in budgets
        if item.change_class in ChangeClass.FORBIDDEN_IN_CORE
        and item.path.startswith(("hqsb/integration/", "hqsb/core/"))
    ]
    return {
        "net_loc_by_class": totals,
        "files": len(budgets),
        "core_leaks": leaks,
        "ok": not leaks,
    }


# ── dummy backend adapter ─────────────────────────────────────────────────


@dataclass
class SpyEvent:
    kind: str
    name: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "detail": dict(self.detail)}


@dataclass
class DummyBackendAdapter:
    """Contract-conformance backend that records every call it receives.

    It exists to verify registration, capability, forced/auto dispatch,
    fake/meta, compile/lowering, failure/reason, artifact compatibility, C6/C7
    and lifecycle (E06-11 §4) — **not** to produce performance numbers.
    """

    name: str = "dummy"
    spy: List[SpyEvent] = field(default_factory=list)
    strict: bool = False

    def _record(self, kind: str, name: str, **detail: Any) -> SpyEvent:
        event = SpyEvent(kind=kind, name=name, detail=detail)
        self.spy.append(event)
        return event

    def capability(self) -> Dict[str, Any]:
        self._record("capability", "probe")
        return {
            "backend": self.name,
            "dtypes": ["float16", "bfloat16", "float32"],
            "layouts": ["contiguous"],
            "compile_safe": True,
            "hardware_performance_claim": False,
        }

    def route(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._record("route", "select", request=dict(request))
        dtype = str(request.get("dtype", "float16"))
        supported = dtype in ("float16", "bfloat16", "float32")
        return {
            "requested": f"{self.name}.custom",
            "actual": f"{self.name}.custom" if supported else f"{self.name}.reference",
            "reason": "CAPABILITY_OK" if supported else "DTYPE",
            "fallback_used": not supported,
        }

    def execute(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        self._record("execute", "run", keys=sorted(payload))
        return {"status": "ok", "observed_implementation": f"{self.name}.custom"}

    def error_mapping(self) -> Dict[str, str]:
        self._record("error_mapping", "read")
        return {
            "unsupported_dtype": "INPUT_DTYPE_UNSUPPORTED",
            "missing_artifact": "QUANT_ARTIFACT_MISMATCH",
            "compile_failure": "COMPILE_FAILED",
        }

    def observed_implementation(self) -> str:
        return f"{self.name}.custom"

    def performance_claim_allowed(self) -> Dict[str, Any]:
        """A dummy backend can never support a hardware performance claim."""
        return {
            "allowed": False,
            "reason": "DUMMY_BACKEND_IS_A_SPY: framework overhead only, no hardware claim",
        }

    def lifecycle(self) -> Tuple[str, ...]:
        return ("REGISTERED", "MODEL_ATTACHED", "ACTIVE", "DISABLED", "CLOSED")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spy_events": [event.as_dict() for event in self.spy],
            "performance_claim": self.performance_claim_allowed(),
        }


# ── second-target registration (config only) ──────────────────────────────


@dataclass(frozen=True)
class AdapterRegistration:
    """A target is registered through config/registry — never a global default."""

    target: str
    kind: str  # "model" | "backend"
    adapter_path: str
    config_path: str = ""
    version: str = "1.0.0"
    changes_global_default: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("model", "backend"):
            raise ConfigError(
                f"{self.target}: adapter kind must be 'model' or 'backend'",
                details={"field": "kind", "actual": self.kind},
            )
        if self.changes_global_default:
            raise ConfigError(
                f"{self.target}: a new target must not change the global default "
                "(that would silently move Qwen off its validated path)",
                details={"field": "changes_global_default"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "adapter_path": self.adapter_path,
            "config_path": self.config_path,
            "version": self.version,
            "changes_global_default": self.changes_global_default,
        }


# ── reuse matrix ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReuseRow:
    """One reuse claim with its evidence pointer."""

    item: str
    status: str  # "REUSED" | "EXTENDED" | "REFUSED"
    evidence: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("REUSED", "EXTENDED", "REFUSED"):
            raise ConfigError(
                f"{self.item}: status must be REUSED/EXTENDED/REFUSED",
                details={"field": "status", "actual": self.status},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "status": self.status,
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass
class ReuseMatrix:
    rows: List[ReuseRow] = field(default_factory=list)

    def add(self, row: ReuseRow) -> None:
        self.rows.append(row)

    @property
    def status_counts(self) -> Dict[str, int]:
        counts = {"REUSED": 0, "EXTENDED": 0, "REFUSED": 0}
        for row in self.rows:
            counts[row.status] += 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "counts": self.status_counts,
        }


def reuse_matrix_template() -> ReuseMatrix:
    """The items E06-11 requires an explicit statement about."""
    matrix = ReuseMatrix()
    for item in (
        "operator_schema",
        "pattern_spec",
        "graph_pass",
        "lowering_registry",
        "capability_interface",
        "error_reason_codes",
        "c6_c7_projection",
        "cache_identity",
        "fake_metadata_contract",
    ):
        matrix.add(ReuseRow(item=item, status="REUSED", note="pending measurement"))
    return matrix


def identity_collision_check(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, Any]:
    """Refuse cross-target reuse when identities collide (E06-11 step 18)."""
    collisions = [
        key
        for key in sorted(set(left) & set(right))
        if left[key] == right[key] and key.endswith(("_id", "_hash", "_key", "_identity"))
    ]
    return {
        "ok": not collisions,
        "collisions": collisions,
        "keys_compared": sorted(set(left) | set(right)),
    }


def code_change_from_git(root: str, base: str, head: str = "HEAD") -> Tuple[ChangeBudget, ...]:
    """Best-effort diff stats via git; a missing git is reported, not faked."""
    import subprocess

    proc = subprocess.run(
        ["git", "diff", "--numstat", base, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(
            f"git diff failed: {proc.stderr.strip()}",
            details={"field": "git", "base": base, "head": head},
        )
    budgets: List[ChangeBudget] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, path = parts
        budgets.append(
            ChangeBudget(
                path=path,
                loc_added=int(added) if added.isdigit() else 0,
                loc_removed=int(removed) if removed.isdigit() else 0,
                change_class=ChangeClass.CONTRACT_EXTENSION,
                reason="unclassified: fill in from the change review",
            )
        )
    return tuple(budgets)


def classify_change(path: str, description: str) -> str:
    """Deterministic first-pass classification.

    The reviewer may override it, but must then record a reason
    (:class:`ChangeBudget.reason`): E06-11 §8 requires every core modification to
    be justified as a generic fix or a contract extension, never as progress.
    """
    in_adapter = "/adapters/" in path or "/backends/" in path or path.startswith("tests/")
    if re.search(r"\b(bug|fix|regression|deadlock|race|leak)\b", description, re.IGNORECASE):
        return ChangeClass.GENERIC_BUG_FIX
    if not in_adapter and re.search(
        r"\b(qwen|llama|module path|class name|layers\.)\b", description, re.IGNORECASE
    ):
        return ChangeClass.MODEL_SPECIFIC_LEAK
    if not in_adapter and re.search(
        r"\b(cuda|triton|inductor|cutlass|ascend)\b", description, re.IGNORECASE
    ):
        return ChangeClass.BACKEND_SPECIFIC_LEAK
    return ChangeClass.CONTRACT_EXTENSION


__all__ = [
    "ADAPTER_PATH_MARKERS",
    "AdapterRegistration",
    "BackendAdapter",
    "ChangeBudget",
    "ChangeClass",
    "CoreContractSnapshot",
    "DummyBackendAdapter",
    "HARDCODE_RULES",
    "HardcodeFinding",
    "HardcodeRule",
    "ModelAdapter",
    "ModuleCensusEntry",
    "ReuseMatrix",
    "ReuseRow",
    "SpyEvent",
    "classify_change",
    "code_change_from_git",
    "core_contract_snapshot",
    "hardcode_report",
    "hardcode_scan",
    "identity_collision_check",
    "model_census",
    "reuse_matrix_template",
    "summarize_changes",
]
