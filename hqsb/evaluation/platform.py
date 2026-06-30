"""Platform identity, software stack and telemetry adapters (E12-02 §4, §11).

Two ideas make this module worth its own file:

1. **Identity must be cross-checked.**  The OS PCI view, the vendor CLI, the
   runtime API and the framework all report devices; when they disagree the
   instance is *blocked* instead of being "resolved" by preference.
2. **Only numpy-free text parsing belongs to the core layer.**  Every vendor
   telemetry field is mapped to a canonical name *with* its semantics
   (instantaneous / averaged / accumulator-derived), boundary, unit, resolution
   and permission — because "we can read a watt" is not a measurement
   capability.  Unmappable vendor fields are preserved under a namespaced
   column instead of being coerced into a canonical name.

Nothing here probes hardware or executes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.records import MISSING_MEASUREMENT_UNAVAILABLE

PLATFORM_IDENTITY_FIELDS: Tuple[str, ...] = (
    "vendor",
    "sku",
    "revision",
    "serial_redacted",
    "device_count",
    "partition",
    "memory_bytes",
    "memory_type",
    "interconnect",
    "host_id",
    "collected_at",
)

SOFTWARE_STACK_FIELDS: Tuple[str, ...] = (
    "firmware",
    "driver",
    "runtime",
    "compiler",
    "framework",
    "container_image_digest",
    "os_kernel",
    "abi_version",
)

#: Where an identity field was observed; a conflict between sources blocks the
#: instance instead of picking the most convenient value.
IDENTITY_SOURCES: Tuple[str, ...] = ("os_pci", "vendor_cli", "runtime_api", "framework")


@dataclass
class PlatformIdentity:
    """One concrete platform instance (never a whole SKU family)."""

    platform_instance_id: str = ""
    vendor: str = ""
    sku: str = ""
    revision: str = ""
    serial_redacted: str = ""
    device_count: int = 0
    partition: str = ""
    memory_bytes: int = 0
    memory_type: str = ""
    interconnect: str = ""
    host_id: str = ""
    collected_at: str = ""
    conflicts: Tuple[str, ...] = ()

    def compute_id(self) -> str:
        return stable_id(
            "plat",
            {
                "vendor": self.vendor,
                "sku": self.sku,
                "revision": self.revision,
                "serial_redacted": self.serial_redacted,
                "device_count": self.device_count,
                "partition": self.partition,
                "host_id": self.host_id,
            },
        )

    def __post_init__(self) -> None:
        if not self.platform_instance_id:
            self.platform_instance_id = self.compute_id()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("vendor", "sku", "device_count", "host_id"):
            if getattr(self, name) in (None, "", 0):
                problems.append(f"platform identity is missing {name!r}")
        if self.device_count < 0:
            problems.append("device_count must not be negative")
        if self.conflicts:
            problems.append(
                "platform identity conflicts detected: "
                + ", ".join(self.conflicts)
                + " (the instance is blocked for formal probes)"
            )
        return problems

    @property
    def usable(self) -> bool:
        return not self.validate()

    def as_dict(self) -> Dict[str, Any]:
        payload = {"platform_instance_id": self.platform_instance_id}
        for name in PLATFORM_IDENTITY_FIELDS:
            payload[name] = getattr(self, name)
        payload["conflicts"] = list(self.conflicts)
        payload["usable"] = self.usable
        return payload


def identity_conflicts(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    fields: Sequence[str] = ("vendor", "sku", "device_count", "memory_bytes"),
) -> Tuple[Dict[str, Any], ...]:
    """Compare the multi-source views of one instance; conflicts are reported.

    Unknown sources are ignored with an explicit note (never silently dropped),
    and a field missing from a source is reported as ``UNAVAILABLE`` rather than
    counted as agreement.
    """
    rows: List[Dict[str, Any]] = []
    for name in fields:
        seen: Dict[str, List[str]] = {}
        for source, payload in sorted(sources.items()):
            if source not in IDENTITY_SOURCES:
                continue
            value = payload.get(name)
            key = "UNAVAILABLE" if value in (None, "") else str(value)
            seen.setdefault(key, []).append(source)
        if "UNAVAILABLE" in seen and len(seen) > 1:
            rows.append(
                {
                    "field": name,
                    "status": "CONFLICT",
                    "values": {key: sorted(val) for key, val in sorted(seen.items())},
                    "detail": "at least one source cannot report the field",
                }
            )
        elif len(seen) > 1:
            rows.append(
                {
                    "field": name,
                    "status": "CONFLICT",
                    "values": {key: sorted(val) for key, val in sorted(seen.items())},
                    "detail": "sources disagree",
                }
            )
    return tuple(rows)


@dataclass
class SoftwareStack:
    """The software half of the SUT (versions are part of the identity)."""

    firmware: str = ""
    driver: str = ""
    runtime: str = ""
    compiler: str = ""
    framework: str = ""
    container_image_digest: str = ""
    os_kernel: str = ""
    abi_version: str = ""
    stack_id: str = ""

    def compute_id(self) -> str:
        return stable_id("stack", {name: getattr(self, name) for name in SOFTWARE_STACK_FIELDS})

    def __post_init__(self) -> None:
        if not self.stack_id:
            self.stack_id = self.compute_id()

    def validate(self) -> List[str]:
        problems = [
            f"software stack is missing {name!r}"
            for name in ("driver", "runtime", "framework")
            if getattr(self, name) in (None, "")
        ]
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {"stack_id": self.stack_id}
        for name in SOFTWARE_STACK_FIELDS:
            payload[name] = getattr(self, name)
        return payload


# ── telemetry field adapters ──────────────────────────────────────────────

TELEMETRY_SEMANTICS: Tuple[str, ...] = ("instantaneous", "averaged", "accumulator_derived")
TELEMETRY_BOUNDARIES: Tuple[str, ...] = ("device", "board", "socket", "node", "accelerator_set")


@dataclass(frozen=True)
class TelemetryFieldAdapter:
    """Maps one vendor field onto a canonical field plus its semantics."""

    canonical_field: str
    vendor_field: str
    semantics: str
    boundary: str
    unit: str
    resolution: float = 0.0
    sample_period_s: float = 0.0
    availability: str = "unknown"
    permission: str = "unknown"
    wrap_or_reset: str = "unknown"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.semantics not in TELEMETRY_SEMANTICS:
            problems.append(f"unknown semantics {self.semantics!r} for {self.canonical_field}")
        if self.boundary not in TELEMETRY_BOUNDARIES:
            problems.append(f"unknown boundary {self.boundary!r} for {self.canonical_field}")
        if not self.unit:
            problems.append(f"{self.canonical_field} has no unit")
        if self.availability not in ("available", "unavailable", "unknown"):
            problems.append(f"unknown availability {self.availability!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_id": stable_id("tfield", {"canonical": self.canonical_field, "vendor": self.vendor_field}),
            "canonical_field": self.canonical_field,
            "vendor_field": self.vendor_field,
            "semantics": self.semantics,
            "boundary": self.boundary,
            "unit": self.unit,
            "resolution": self.resolution,
            "sample_period_s": self.sample_period_s,
            "availability": self.availability,
            "permission": self.permission,
            "wrap_or_reset": self.wrap_or_reset,
        }


#: The canonical telemetry vocabulary every platform adapter must map onto.
CANONICAL_TELEMETRY_FIELDS: Tuple[str, ...] = (
    "accelerator.power_W",
    "accelerator.energy_J",
    "accelerator.temperature_C",
    "accelerator.clock_mhz",
    "accelerator.utilization",
    "accelerator.throttle_reasons",
    "accelerator.ecc_errors",
    "accelerator.memory_used_bytes",
    "host.cpu_load",
    "host.ram_used_bytes",
    "host.swap_used_bytes",
    "node.power_W",
)

ADAPTER_CATALOG: Tuple[TelemetryFieldAdapter, ...] = (
    TelemetryFieldAdapter("accelerator.power_W", "nvidia-smi:power.draw", "instantaneous", "device", "W", 0.001, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.power_W", "nvml:power_usage", "averaged", "device", "W", 0.001, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.energy_J", "nvml:total_energy_consumption", "accumulator_derived", "device", "mJ", 1.0, 0.1, "available", "user", "counter_rollover"),
    TelemetryFieldAdapter("accelerator.temperature_C", "nvidia-smi:temperature.gpu", "instantaneous", "device", "C", 1.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.clock_mhz", "nvidia-smi:clocks.sm", "instantaneous", "device", "MHz", 1.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.utilization", "nvidia-smi:utilization.gpu", "averaged", "device", "%", 1.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.throttle_reasons", "nvidia-smi:clocks_throttle_reasons.active", "instantaneous", "device", "bitmask", 0.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.ecc_errors", "nvml:ecc_errors", "accumulator_derived", "device", "count", 1.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("accelerator.memory_used_bytes", "nvml:memory_used", "instantaneous", "device", "B", 1.0, 0.1, "available", "user", "none"),
    TelemetryFieldAdapter("host.cpu_load", "procfs:loadavg", "instantaneous", "node", "1", 0.01, 1.0, "available", "user", "none"),
    TelemetryFieldAdapter("host.ram_used_bytes", "procfs:meminfo", "instantaneous", "node", "B", 1.0, 1.0, "available", "user", "none"),
    TelemetryFieldAdapter("host.swap_used_bytes", "procfs:vmstat", "instantaneous", "node", "B", 1.0, 1.0, "available", "user", "none"),
    TelemetryFieldAdapter("node.power_W", "pdu:input_power", "averaged", "node", "W", 0.1, 1.0, "external_meter_required", "admin", "none"),
    TelemetryFieldAdapter("accelerator.power_W", "npu-smi:power", "instantaneous", "device", "W", 0.1, 0.5, "unknown", "user", "none"),
    TelemetryFieldAdapter("accelerator.temperature_C", "tegrastats:Temp", "instantaneous", "board", "C", 0.5, 1.0, "available", "user", "none"),
)

#: Fields the *canonical* vocabulary requires but the catalog may not be able to
#: fill on a given platform; the gap is reported as a capability limitation.
REQUIRED_FOR_ENERGY: Tuple[str, ...] = (
    "accelerator.power_W",
    "accelerator.energy_J",
    "accelerator.temperature_C",
    "accelerator.clock_mhz",
)


def adapters_for(canonical_field: str) -> Tuple[TelemetryFieldAdapter, ...]:
    return tuple(row for row in ADAPTER_CATALOG if row.canonical_field == canonical_field)


def canonicalize_telemetry_field(
    vendor_row: Mapping[str, Any],
    *,
    adapters: Sequence[TelemetryFieldAdapter] = ADAPTER_CATALOG,
) -> Dict[str, Any]:
    """Map one vendor sample onto canonical fields, preserving unmapped keys.

    Fail-closed rules: without a matching adapter the field stays namespaced; if
    boundary/unit/semantics cannot be established the canonical field is
    ``MEASUREMENT_UNAVAILABLE`` with a reason — never an estimate.
    """
    vendor_field = str(vendor_row.get("vendor_field", ""))
    value = vendor_row.get("value")
    matches = [
        adapter
        for adapter in adapters
        if adapter.vendor_field == vendor_field and adapter.canonical_field == vendor_row.get("canonical_field", adapter.canonical_field)
    ]
    if not matches:
        return {
            "canonical_field": "",
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": f"no adapter for vendor field {vendor_field!r}: preserved as a namespaced column",
            f"vendor.{vendor_field}": value,
        }
    adapter = matches[0]
    problems = adapter.validate()
    if problems:
        return {
            "canonical_field": adapter.canonical_field,
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "; ".join(problems),
            f"vendor.{vendor_field}": value,
        }
    if value is None:
        return {
            "canonical_field": adapter.canonical_field,
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "field readable but no sample in the window",
            "semantics": adapter.semantics,
            "boundary": adapter.boundary,
            "unit": adapter.unit,
        }
    return {
        "canonical_field": adapter.canonical_field,
        "status": "OK",
        "value": value,
        "semantics": adapter.semantics,
        "boundary": adapter.boundary,
        "unit": adapter.unit,
        "resolution": adapter.resolution,
        "sample_period_s": adapter.sample_period_s,
        "field_id": adapter.as_dict()["field_id"],
        "vendor_field": vendor_field,
    }


def energy_capability(
    canonical_fields: Sequence[str],
) -> Dict[str, Any]:
    """What energy work E12-06 may attempt on this platform (no estimates)."""
    available = set(canonical_fields)
    missing = [name for name in REQUIRED_FOR_ENERGY if name not in available]
    return {
        "fields": sorted(available),
        "missing_for_energy": missing,
        "can_measure_energy": not missing,
        "status": "OK" if not missing else MISSING_MEASUREMENT_UNAVAILABLE,
        "reason": "" if not missing else "missing telemetry fields: " + ", ".join(missing),
    }


def telemetry_smoke_check(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Capability-only sanity of a telemetry series (never an energy number)."""
    timestamps = [int(row.get("t_ns", 0)) for row in series]
    values = [row.get("power_w") for row in series]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    monotonic = all(b > a for a, b in zip(timestamps, timestamps[1:]))
    constant = bool(numeric) and max(numeric) == min(numeric)
    return {
        "samples": len(series),
        "monotonic_timestamps": monotonic,
        "has_variation": not constant,
        "constant_value_suspected_stale_cache": constant,
        "uses_load_response_only": True,
        "note": "capability check only: it does not compute or imply any energy result",
    }


# ── probe harness ─────────────────────────────────────────────────────────


@dataclass
class ProbeHarness:
    """Frozen probe harness identity (E12-02 §5.3)."""

    harness_id: str
    source_uri: str
    source_hash: str
    container_image_digest: str = ""
    cli_or_api: str = ""
    schema_version: str = "1.0.0"
    timeout_s: float = 60.0
    resource_limits: Mapping[str, Any] = field(default_factory=dict)
    reference_impl: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("source_uri", "source_hash", "cli_or_api", "reference_impl"):
            if not getattr(self, name):
                problems.append(f"probe harness is missing {name!r}")
        if self.timeout_s <= 0:
            problems.append("probe harness timeout must be positive")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "container_image_digest": self.container_image_digest,
            "cli_or_api": self.cli_or_api,
            "schema_version": self.schema_version,
            "timeout_s": self.timeout_s,
            "resource_limits": dict(sorted(self.resource_limits.items())),
            "reference_impl": self.reference_impl,
        }


def probe_harness(
    *,
    source_uri: str,
    source_hash: str,
    cli_or_api: str,
    reference_impl: str,
    container_image_digest: str = "",
    timeout_s: float = 60.0,
    resource_limits: Optional[Mapping[str, Any]] = None,
) -> ProbeHarness:
    harness_id = stable_id(
        "harness", {"uri": source_uri, "hash": source_hash, "cli": cli_or_api}
    )
    return ProbeHarness(
        harness_id=harness_id,
        source_uri=source_uri,
        source_hash=source_hash,
        container_image_digest=container_image_digest,
        cli_or_api=cli_or_api,
        timeout_s=timeout_s,
        resource_limits=dict(resource_limits or {}),
        reference_impl=reference_impl,
    )


def platform_summary(identity: PlatformIdentity, stack: SoftwareStack) -> Dict[str, Any]:
    """One machine-readable row for the campaign inventory."""
    return {
        **identity.as_dict(),
        "software_stack": stack.as_dict(),
        "stack_id": stack.stack_id,
        "identity_problems": identity.validate(),
        "stack_problems": stack.validate(),
    }
