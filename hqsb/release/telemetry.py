"""S15 telemetry: action logs, time-to-event, evidence lookup, detector metrics.

Every S15 experiment produces the same observational skeleton (manual §22 and
§13.4): an action log with UTC ordering, a timeline, a time-to-event summary with
explicit sample-size limits, evidence-lookup trials ("how long does it take a
reader to reach the raw file"), and findings with severity/owner/disposition.

Two S15-specific rules shape this module:

* **small samples stay small** — ``E15-11`` runs 2–3 participants and
  ``E15-09`` one or two reviewers.  :func:`time_to_event_summary` therefore
  refuses to emit a percentile for tiny samples (a "P95 of 3" is a fabricated
  precision, not a measurement) and reports the raw values instead;
* **detectors need negative controls** — every scanner in S15 (claims, links,
  commands, figures, secrets) must prove detection power on an *injected*
  defect set before its "zero findings" means anything (``E15-01`` step 35/36,
  ``E15-04`` step 39/40, ``E15-06`` step 38–42).  :class:`DetectorMetrics`
  computes per-severity recall and refuses to call a run clean when a P0
  injection was missed.

Nothing here executes an experiment; it records what an execution reports.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import canonical_digest, sort_actions

#: Sample sizes below this are reported as raw values, not percentiles.
MIN_PERCENTILE_N = 5

#: The S15 "severe" injection classes that must never be missed.
SEVERE_INJECTION_CLASSES: Tuple[str, ...] = ("P0", "P1")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ConfigError("cannot take a percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


@dataclass
class ActionLog:
    """An append-only ``command_or_action_log.jsonl`` (manual §22).

    The log records *what actually happened*: actor, action, inputs, outputs,
    exit code and duration.  It refuses to drop an entry that already has a
    start timestamp — a log with holes is how an undocumented intervention
    disappears (``E15-02`` step 38).
    """

    path: str
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        *,
        actor: str,
        action: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        exit_code: Optional[int] = None,
        duration_s: Optional[float] = None,
        notes: str = "",
        started_at_utc: str = "",
    ) -> Dict[str, Any]:
        if not actor or not action:
            raise ConfigError("ActionLog: actor and action are required")
        if duration_s is not None and duration_s < 0:
            raise ConfigError("ActionLog: duration may not be negative")
        entry = {
            "utc": started_at_utc or utc_now(),
            "actor": actor,
            "action": action,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "exit_code": exit_code,
            "duration_s": duration_s,
            "notes": notes,
        }
        entry["entry_sha256"] = canonical_digest(entry)
        self.entries.append(entry)
        return entry

    def flush(self) -> str:
        """Write the log, UTC-monotonic, as JSON lines."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        ordered = sort_actions(self.entries)
        with open(self.path, "w", encoding="utf-8") as handle:
            for entry in ordered:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return self.path

    def monotonic_problems(self) -> List[str]:
        """Every entry carries a timestamp and the order is non-decreasing."""
        problems: List[str] = []
        stamps = [str(entry.get("utc", "")) for entry in self.entries]
        for index, stamp in enumerate(stamps):
            if not stamp:
                problems.append(f"ActionLog entry {index} has no UTC timestamp")
        if [stamp for stamp in stamps if stamp] != sorted(stamp for stamp in stamps if stamp):
            problems.append("ActionLog entries are not UTC-monotonic when written in insertion order")
        return problems


@dataclass
class TimeToEventSummary:
    """A time-to-event summary that does not invent precision (§13.4)."""

    event: str
    samples_s: Tuple[float, ...] = ()
    censored: int = 0
    notes: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.event:
            findings.append("TimeToEventSummary: event is required")
        if any(value < 0 for value in self.samples_s):
            findings.append("TimeToEventSummary: negative durations")
        if self.censored < 0:
            findings.append("TimeToEventSummary: censored count may not be negative")
        return findings

    def summary(self) -> Dict[str, Any]:
        values = sorted(float(value) for value in self.samples_s)
        payload: Dict[str, Any] = {
            "event": self.event,
            "n": len(values),
            "censored": self.censored,
            "raw_s": values,
            "notes": self.notes,
        }
        if len(values) >= MIN_PERCENTILE_N:
            payload["median_s"] = _percentile(values, 0.5)
            payload["p95_s"] = _percentile(values, 0.95)
            payload["precision_claim"] = "percentiles reported"
        else:
            payload["precision_claim"] = (
                f"raw values only: n={len(values)} < {MIN_PERCENTILE_N} (small-sample 不外推 — §13.4)"
            )
        return payload

    def as_dict(self) -> Dict[str, Any]:
        return self.summary()


@dataclass
class EvidenceLookupTrial:
    """One evidence drill-down trial (``E15-06`` §9.2, ``E15-07`` step 35)."""

    trial_id: str
    start_ref: str
    target_kind: str
    duration_s: Optional[float] = None
    clicks: int = 0
    success: bool = False
    correct_candidate: bool = False
    over_budget: bool = False
    notes: str = ""

    #: Target kinds a drill-down may look for (claim → raw → command chain).
    TARGET_KINDS: Tuple[str, ...] = (
        "claim_record",
        "figure_point",
        "normalized_row",
        "raw_sample",
        "manifest",
        "command",
        "environment",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.trial_id or not self.start_ref:
            findings.append("EvidenceLookupTrial: trial_id and start_ref are required")
        if self.target_kind not in self.TARGET_KINDS:
            findings.append(f"EvidenceLookupTrial: unknown target_kind {self.target_kind!r}")
        if self.duration_s is not None and self.duration_s < 0:
            findings.append("EvidenceLookupTrial: negative duration")
        if self.success and not self.correct_candidate:
            findings.append(
                "EvidenceLookupTrial: opening a same-named older report is not a successful lookup "
                "(E15-08 §9.2)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "start_ref": self.start_ref,
            "target_kind": self.target_kind,
            "duration_s": self.duration_s,
            "clicks": self.clicks,
            "success": self.success,
            "correct_candidate": self.correct_candidate,
            "over_budget": self.over_budget,
            "notes": self.notes,
        }


@dataclass
class DetectorMetrics:
    """Precision/recall of a scanner against an injected defect set.

    ``E15-01`` step 36 and ``E15-04`` step 40 require per-category reporting and
    a hard rule: **a missed severe injection invalidates a "clean" result**, no
    matter how many benign items were scanned.
    """

    detector: str
    injected: Tuple[Mapping[str, Any], ...] = ()
    detected: Tuple[str, ...] = ()
    false_positives: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.detector:
            findings.append("DetectorMetrics: detector is required")
        if not self.injected:
            findings.append(
                "DetectorMetrics: no injections means no detection power; a scanner returning zero "
                "findings on an empty control set is INVALID (manual §23 closing note)"
            )
        for item in self.injected:
            if "injection_id" not in item or "severity" not in item:
                findings.append("DetectorMetrics: every injected defect needs injection_id and severity")
                break
        return findings

    def metrics(self) -> Dict[str, Any]:
        injected_ids = [str(item["injection_id"]) for item in self.injected]
        detected = set(self.detected)
        true_positives = [name for name in injected_ids if name in detected]
        false_negatives = [name for name in injected_ids if name not in detected]
        recall = len(true_positives) / len(injected_ids) if injected_ids else 0.0
        fp = list(self.false_positives)
        precision_denominator = len(true_positives) + len(fp)
        precision = len(true_positives) / precision_denominator if precision_denominator else 0.0
        by_severity: Dict[str, Dict[str, int]] = {}
        for item in self.injected:
            severity = str(item["severity"])
            bucket = by_severity.setdefault(severity, {"injected": 0, "missed": 0})
            bucket["injected"] += 1
            if str(item["injection_id"]) not in detected:
                bucket["missed"] += 1
        severe_missed = [
            code
            for severity in SEVERE_INJECTION_CLASSES
            for code in [name for name in false_negatives if self._severity_of(name) == severity]
        ]
        return {
            "detector": self.detector,
            "injected": len(injected_ids),
            "detected": len(true_positives),
            "missed": false_negatives,
            "false_positives": fp,
            "recall": recall,
            "precision": precision,
            "by_severity": {key: by_severity[key] for key in sorted(by_severity)},
            "severe_missed": severe_missed,
            "clean_result_valid": not severe_missed,
        }

    def _severity_of(self, injection_id: str) -> str:
        for item in self.injected:
            if str(item["injection_id"]) == injection_id:
                return str(item["severity"])
        return ""


@dataclass
class FindingTracker:
    """Findings with severity, owner, affected claims and disposition (§22)."""

    findings: List[rec.Finding] = field(default_factory=list)

    def add(self, finding: rec.Finding) -> rec.Finding:
        problems = finding.validate()
        if problems:
            raise ConfigError("invalid finding: " + "; ".join(problems))
        self.findings.append(finding)
        return finding

    def by_severity(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.severity, []).append(finding.finding_id)
        return {level: sorted(ids) for level, ids in sorted(grouped.items())}

    def blocking(self) -> List[str]:
        """Findings that block release (§10: P0 unconditional, P1 blocks its claim)."""
        return sorted(
            finding.finding_id
            for finding in self.findings
            if finding.severity in ("P0", "P1") and finding.disposition not in ("FIXED_IN_NEW_CANDIDATE",)
        )

    def assert_no_open_p0(self) -> List[str]:
        """Open P0/P1 findings, refused by a release GO."""
        return self.blocking()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "findings": [finding.as_dict() for finding in self.findings],
            "by_severity": self.by_severity(),
            "blocking": self.blocking(),
        }


@dataclass
class SegmentTiming:
    """Segment budget vs. actual timing (``E15-07`` step 4/40)."""

    segment: str
    budget_s: float
    actual_s: Optional[float] = None

    def overrun_s(self) -> Optional[float]:
        if self.actual_s is None:
            return None
        return max(0.0, self.actual_s - self.budget_s)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.segment:
            findings.append("SegmentTiming: segment is required")
        if self.budget_s <= 0:
            findings.append("SegmentTiming: a zero/negative budget is not a budget")
        if self.actual_s is not None and self.actual_s < 0:
            findings.append("SegmentTiming: negative actual duration")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment,
            "budget_s": self.budget_s,
            "actual_s": self.actual_s,
            "overrun_s": self.overrun_s(),
        }


def summarise_segments(segments: Iterable[SegmentTiming]) -> Dict[str, Any]:
    items = list(segments)
    problems: List[str] = []
    for item in items:
        problems.extend(item.problems())
    return {
        "segments": [item.as_dict() for item in items],
        "total_budget_s": sum(item.budget_s for item in items),
        "total_actual_s": sum(item.actual_s for item in items if item.actual_s is not None),
        "overran": [item.segment for item in items if (item.overrun_s() or 0) > 0],
        "problems": problems,
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the telemetry layer (labelled smoke, not an experiment)."""
    problems: List[str] = []
    log = ActionLog(path="/tmp/hqsb-s15-action-log.jsonl")
    log.append(
        actor="driver",
        action="smoke",
        inputs=["--smoke"],
        exit_code=0,
        duration_s=0.1,
        started_at_utc="2999-01-01T00:00:00Z",
    )
    log.append(actor="driver", action="smoke", inputs=["--smoke"], exit_code=0, duration_s=0.2)
    if not log.monotonic_problems():
        problems.append("the reverse-ordered action log was not detected")
    tiny = TimeToEventSummary(event="first_verified_report", samples_s=(12.0, 30.0)).summary()
    if "p95_s" in tiny:
        problems.append("a 2-sample run reported a P95 (fabricated precision)")
    metrics = DetectorMetrics(
        detector="claim-scanner",
        injected=(
            {"injection_id": "inj-orphan", "severity": "P0"},
            {"injection_id": "inj-unit", "severity": "P2"},
        ),
        detected=("inj-unit",),
    ).metrics()
    if metrics["clean_result_valid"]:
        problems.append("a missed P0 injection still produced a valid clean result")
    tracker = FindingTracker()
    tracker.add(rec.Finding(finding_id="F-1", severity="P2", owner="docs", disposition="OPEN"))
    if tracker.blocking():
        problems.append("a P2 finding must not block release")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "min_percentile_n": MIN_PERCENTILE_N,
        "tiny_sample_claim": tiny["precision_claim"],
        "detector_recall": metrics["recall"],
        "severe_missed": metrics["severe_missed"],
        "problems": problems,
    }
