"""Versioned, bounded observation plans. No device imports or initialization."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

SCHEMA_VERSION = "1.0.0"
MODES = frozenset({"off", "basic", "operators", "memory", "pages"})
RUN_ID = re.compile(r"run_[a-f0-9]{32}\Z")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ObservationPlan:
    mode: str = "basic"
    os_interval_ms: int = 250
    max_records: int = 100_000
    max_bytes: int = 128 * 1024 * 1024
    max_pages: int = 4096
    max_decode_steps: int = 8
    sources: tuple | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("OBSERVATION_MODE_UNSUPPORTED")
        if self.sources is not None:
            if not isinstance(self.sources, (list, tuple)) or len(set(self.sources)) != len(self.sources) or set(self.sources) - {"allocator", "pages", "operators"}:
                raise ValueError("OBSERVATION_SOURCES_INVALID")
            if self.mode == "off" and self.sources:
                raise ValueError("OFF_PLAN_CANNOT_COLLECT")
            object.__setattr__(self, "sources", tuple(sorted(self.sources)))
        for field, low, high in (
            ("os_interval_ms", 250, 10_000),
            ("max_records", 1, 100_000),
            ("max_bytes", 1024, 512 * 1024 * 1024),
            ("max_pages", 1, 4096),
            ("max_decode_steps", 0, 32),
        ):
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"OBSERVATION_PLAN_INVALID:{field}")

    def as_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def collects(self, source):
        defaults = {"memory": ("allocator",), "pages": ("pages",), "operators": ("operators",)}
        return source in (self.sources if self.sources is not None else defaults.get(self.mode, ()))


def resolve_observation(raw, mode):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("OBSERVATION_PLAN_INVALID")
    fields = {key: value for key, value in raw.items() if key != "schema_version"}
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("OBSERVATION_VERSION_UNSUPPORTED")
    if fields.get("mode", mode) != mode:
        raise ValueError("OBSERVATION_MODE_CONFLICT")
    try:
        return ObservationPlan(**{**fields, "mode": mode})
    except TypeError as exc:
        raise ValueError("OBSERVATION_PLAN_FIELDS_INVALID") from exc


def metric(name, value, unit, *, source, scope, kind="measured", reason=None):
    return {
        "metric_id": name,
        "value": value,
        "unit": unit,
        "evidence_kind": kind,
        "scope": scope,
        "source": source,
        "availability": "observed" if value is not None else "not_collected",
        "reason_code": reason if value is None else None,
    }
