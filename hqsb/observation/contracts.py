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

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("OBSERVATION_MODE_UNSUPPORTED")
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
