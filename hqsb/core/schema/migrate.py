"""Migration from legacy benchmark documents to the current schema (E01-07).

S00 left two real legacy document families in the tree:

* **legacy golden** — produced by ``generate_golden.py``.  The document
  carries ``input_token_ids``/``generated_tokens``/``first_token`` and a
  partial ``model`` block (id/source/dtype/config_hash) but no run/result
  envelope, no workload name and **no timings** (it records *what* was
  produced, not *how long*).
* **legacy result** — produced by ``run_model_core.py``.  The document
  carries ``repetitions`` with per-pass ``prefill_forward_ms``,
  ``first_token_selection_ms``, generated token IDs and **summary-only ITL
  statistics** (no per-token ``raw_itl_ms``), a ``workload`` block and a
  ``deterministic``/``generated_token_sha256`` pair.  It has no ``run_id``
  and no contract-typed model.

E01-07 contract rules implemented here
--------------------------------------

* A migrated :class:`BenchmarkResult` (C6) never fabricates per-sample data
  from aggregate statistics.  Legacy ITL summary stats are preserved under
  ``summary["itl_summary"]`` and the sample ``itl_ms`` stays empty; a
  ``legacy_deterministic`` flag is *not* reinterpreted as a reference
  correctness gate.
* Source fields that have no faithful target are explicitly retained (in
  ``summary``/``artifact_links``) **and** declared as a loss row in
  ``summary["migration"]["losses"]`` with a machine-readable ``loss_class``
  (``none`` | ``representation`` | ``insufficient`` | ``unexpressible`` |
  ``semantic_change``) and ``acceptance`` (``regression_ready`` |
  ``historical_only`` | ``reject``).
* Source version identification is explicit: an approved unversioned legacy
  shape is recognised only by an unambiguous structural rule; an unknown
  shape is rejected with a reason; a declared *future* version is rejected
  (``UnsupportedSchemaVersionError``).
* Source/target hashes and both provenances (original experiment vs migration
  execution) are recorded in ``summary["migration"]``; the legacy identity is
  never overwritten.

Usage (CLI)::

    python scripts/migrate_legacy.py input.json output.json
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Mapping, Optional

from hqsb.core.contracts.result import (
    BenchmarkResult,
    EnvironmentInfo,
)
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import (
    SchemaError,
    SchemaMigrationRequiredError,
    SchemaVersionError,
    UnsupportedSchemaVersionError,
)
from hqsb.core.ids import new_run_id
from hqsb.core.schema.versioning import SchemaVersion

#: Version of this migration implementation (E01-07 report provenance).
MIGRATOR_VERSION = "1.1.0"
#: The only legacy family version this migrator understands.
LEGACY_FAMILY_VERSION = "1.0.0"
#: Target C6 schema version.
C6_SCHEMA_VERSION = "1.0.0"


# ── Source version identification ──────────────────────────────────────


def _parse_declared_version(document: Mapping[str, Any]) -> Optional[str]:
    """Return a validated declared ``schema_version`` string, else ``None``.

    Raises:
        SchemaVersionError: If ``schema_version`` is present but not a
            ``major.minor.patch`` string.
    """
    raw = document.get("schema_version")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SchemaVersionError(
            f"'schema_version' must be a string, got {type(raw).__name__}"
        )
    # Validate, but keep the original string for the report.
    SchemaVersion.parse(raw)
    return raw


def detect_family(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Detect the legacy document family with an explicit reason.

    The structural rule is deliberately conservative and unambiguous so a
    document is never classified on a guess:

    * ``legacy_golden``: has both ``input_token_ids`` and ``first_token``
      (the two markers that only ``generate_golden.py`` emits).
    * ``legacy_result``: has ``repetitions`` (only ``run_model_core.py``
      emits a top-level ``repetitions`` list).

    Raises:
        SchemaError: For a document matching neither family (with a reason).
    """
    has_golden_markers = (
        "input_token_ids" in document and "first_token" in document
    )
    has_result_marker = "repetitions" in document

    if has_golden_markers and has_result_marker:
        raise SchemaError(
            "document is ambiguous: it matches both legacy golden "
            "(input_token_ids/first_token) and legacy result (repetitions); "
            "refusing to guess the source schema",
            details={"error_code": "ambiguous_legacy_family"},
        )
    if has_golden_markers:
        return {"family": "legacy_golden", "reason": "input_token_ids+first_token"}
    if has_result_marker:
        return {"family": "legacy_result", "reason": "top-level repetitions"}
    raise SchemaError(
        "unrecognized legacy document shape: expected a legacy golden "
        "(input_token_ids + first_token) or a legacy model-core result "
        "(repetitions); neither marker is present",
        details={"error_code": "unrecognized_legacy_shape"},
    )


def source_version_info(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the source version before migration.

    Returns a dict with ``declared_version`` (string or ``None``), the
    effective family version and the resolution policy used.

    Raises:
        UnsupportedSchemaVersionError: If the document declares a version
            newer than :data:`LEGACY_FAMILY_VERSION`.
        SchemaMigrationRequiredError: If the document declares an *older*
            version for which no migration path exists.
    """
    declared = _parse_declared_version(document)

    if declared is None:
        # Approved unversioned legacy format: recognised only by shape; the
        # shape rule above already guarantees an unambiguous family.
        return {
            "declared_version": None,
            "policy": "approved_unversioned_legacy_shape",
            "source_version": LEGACY_FAMILY_VERSION,
        }

    declared_v = SchemaVersion.parse(declared)
    known = SchemaVersion.parse(LEGACY_FAMILY_VERSION)
    if declared_v > known:
        raise UnsupportedSchemaVersionError(
            f"legacy document declares schema_version {declared!r} which is "
            f"newer than the supported legacy family version "
            f"{LEGACY_FAMILY_VERSION}; refusing to guess a future legacy schema",
            details={"error_code": "unsupported_future_schema_version",
                     "field_path": "schema_version"},
        )
    if declared_v < known:
        raise SchemaMigrationRequiredError(
            f"legacy document declares schema_version {declared!r}; only "
            f"family version {LEGACY_FAMILY_VERSION} is supported and no "
            f"older-version migration path is registered",
            details={"error_code": "unsupported_legacy_older_version",
                     "field_path": "schema_version"},
        )
    return {
        "declared_version": declared,
        "policy": "explicit_version_matches_family",
        "source_version": LEGACY_FAMILY_VERSION,
    }


# ── Loss / migration report helpers ────────────────────────────────────


def _loss_row(
    source_path: str,
    target_path: str,
    loss_class: str,
    acceptance: str,
    note: str,
) -> Dict[str, str]:
    return {
        "source": source_path,
        "target": target_path,
        "loss_class": loss_class,
        "acceptance": acceptance,
        "note": note,
    }


def _canonical_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SOFTWARE_CUDA_KEYS = ("cuda", "torch_cuda")


def _environment(document: Mapping[str, Any]) -> EnvironmentInfo:
    hardware = document.get("hardware", {}) or {}
    software = document.get("software", {}) or {}
    framework: Dict[str, str] = {}
    for key, value in software.items():
        if key not in ("python", "torch") + _SOFTWARE_CUDA_KEYS and isinstance(
            value, str
        ):
            framework[key] = value
    return EnvironmentInfo(
        platform=hardware.get("platform", ""),
        device=hardware.get("device", ""),
        compute_capability=hardware.get("compute_capability"),
        python_version=software.get("python", ""),
        torch_version=software.get("torch", ""),
        # legacy golden names it software.cuda; the model-core producer names
        # it software.torch_cuda (torch.version.cuda). Same meaning.
        cuda_version=next(
            (software[k] for k in _SOFTWARE_CUDA_KEYS if software.get(k)),
            "",
        ),
        framework_versions=framework,
    )


def _environment_losses(document: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Field-level losses for hardware/software → C6 environment mapping."""
    rows: List[Dict[str, str]] = []
    hardware = document.get("hardware", {}) or {}
    software = document.get("software", {}) or {}
    for key in sorted(hardware):
        if key in ("platform", "device", "compute_capability"):
            continue
        rows.append(
            _loss_row(
                f"hardware.{key}", "<environment-unmapped>", "representation",
                "historical_only",
                f"hardware detail {key!r} has no C6 EnvironmentInfo field; "
                "preserved only in migration metadata",
            )
        )
    for key in sorted(software):
        if key in ("python", "torch") + _SOFTWARE_CUDA_KEYS:
            continue
        rows.append(
            _loss_row(
                f"software.{key}", "environment.framework_versions",
                "representation", "regression_ready",
                f"framework version {key!r} is kept verbatim in "
                "environment.framework_versions",
            )
        )
    return rows


def _model_losses(document: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    model = document.get("model", {}) or {}
    if model.get("config_hash"):
        rows.append(
            _loss_row(
                "model.config_hash", "artifact_links.model_config_sha256",
                "representation", "historical_only",
                "config_hash is a SHA256 of the model config.json only; it is "
                "NOT a C1 ModelArtifact identity hash (which covers the full "
                "weight/tokenizer manifest). Stored under "
                "artifact_links.model_config_sha256 and model_artifact_hash "
                "is left null; token-level regression requires re-verified "
                "artifact identity (E00-02)",
            )
        )
    return rows


#: Production-code evidence that every default-fill below is based on.
#: E01-07 §6.3 forbids filling a gap with a *current* default; here each fill
#: cites the exact legacy producer line (module + behaviour), so the value is
#: evidence-backed, not a present-day assumption.
_WORKLOAD_DEFAULT_EVIDENCE = {
    "sampling": (
        "legacy producers generate with torch argmax only "
        "(benchmarks/scripts/generate_golden.py + hqsb/benchmark/model_core.py "
        "lines argmax(dim=-1)); no temperature/top-p/sampler was used, so "
        "sampling='greedy' is a faithful reading, not a default guess"
    ),
    "seed": (
        "no random sampler exists in the legacy production path "
        "(argmax decode is deterministic); seed=0 is therefore semantically "
        "inert and records 'not applicable', not an assumed sampling seed"
    ),
    "warmup_result": (
        "run_model_core.py runs exactly one warmup pass "
        "(benchmark_model_core(..., min(output_tokens, warmup_output_tokens))) "
        "before the timed repetitions, so warmup=1 is the legacy producer's "
        "behaviour"
    ),
    "repetitions_result": (
        "timed passes equal the length of the legacy 'repetitions' list"
    ),
}


def defaults_filled(family: str, document: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return every workload default the migration fills, with its evidence.

    E01-07 §6.3: a missing legacy field may only be filled when the legacy
    producer's own behaviour proves the value; otherwise it stays a gap.  All
    rows below cite production code (argmax-only decode, single warmup pass),
    never a present-day default.
    """
    if family == "legacy_golden":
        return [
            {
                "field": "workload.name",
                "value": "legacy",
                "evidence": "legacy golden has no workload case name and is "
                "NOT auto-renamed to one of the six official workload names "
                "(E01-07 §4.2)",
            },
            {
                "field": "workload.sampling",
                "value": "greedy",
                "evidence": _WORKLOAD_DEFAULT_EVIDENCE["sampling"],
            },
            {
                "field": "workload.seed",
                "value": "0",
                "evidence": _WORKLOAD_DEFAULT_EVIDENCE["seed"],
            },
            {
                "field": "workload.warmup",
                "value": "0",
                "evidence": "golden records a single reference generation and "
                "no timed/warmup passes; warmup=0 means 'not applicable'",
            },
            {
                "field": "workload.repetitions",
                "value": "1",
                "evidence": "one golden reference generation per document",
            },
        ]
    # legacy_result
    reps = document.get("repetitions", [])
    return [
        {
            "field": "workload.name",
            "value": "legacy",
            "evidence": "legacy result workload block carries no case name and "
            "is NOT auto-mapped to the six official workload names (E01-07 "
            "§4.2)",
        },
        {
            "field": "workload.batch_size",
            "value": str(document.get("workload", {}).get("batch_size", 1)),
            "evidence": "batch_size copied verbatim from the workload block",
        },
        {
            "field": "workload.sampling",
            "value": "greedy",
            "evidence": _WORKLOAD_DEFAULT_EVIDENCE["sampling"],
        },
        {
            "field": "workload.seed",
            "value": "0",
            "evidence": _WORKLOAD_DEFAULT_EVIDENCE["seed"],
        },
        {
            "field": "workload.warmup",
            "value": "1",
            "evidence": _WORKLOAD_DEFAULT_EVIDENCE["warmup_result"],
        },
        {
            "field": "workload.repetitions",
            "value": str(max(len(reps), 1)),
            "evidence": _WORKLOAD_DEFAULT_EVIDENCE["repetitions_result"],
        },
    ]


def _workload_golden(document: Mapping[str, Any]) -> WorkloadSpec:
    # Legacy golden records greedy/argmax decode with no sampling seed and no
    # measured timings; name is generic (never mapped to the six official
    # workload names, per E01-07 §4.2).
    return WorkloadSpec(
        name="legacy",
        input_tokens=int(document["input_tokens"]),
        output_tokens=int(document["output_tokens"]),
        seed=0,
        sampling="greedy",
        warmup=0,
        repetitions=1,
        token_ids=document.get("input_token_ids"),
        stop_condition="output_tokens",
    )


def _workload_result(document: Mapping[str, Any]) -> WorkloadSpec:
    workload_block = document.get("workload", {}) or {}
    reps = document.get("repetitions", [])
    return WorkloadSpec(
        name="legacy",
        input_tokens=int(workload_block.get("input_tokens", 0)),
        output_tokens=int(workload_block.get("output_tokens", 0)),
        batch_size=int(workload_block.get("batch_size", 1)),
        seed=0,
        sampling="greedy",
        warmup=1,
        repetitions=max(len(reps), 1),
        stop_condition="output_tokens",
    )


# ── Migrations ─────────────────────────────────────────────────────────


def migrate_legacy_golden(document: Mapping[str, Any]) -> BenchmarkResult:
    """Migrate a legacy golden document to a C6 :class:`BenchmarkResult`.

    Semantic rules (E01-07):

    * ``input_token_ids`` → ``workload.token_ids`` (requested input tokens);
    * ``generated_tokens`` → ``raw_samples[0].generated_token_ids``;
    * ``first_token`` payload → ``summary.first_token`` (full top-K evidence);
    * golden has **no timings**; the raw sample carries explicit zero timing
      only as a placeholder and the loss row marks ``insufficient`` so no
      consumer can treat it as a measured performance sample.
    * the partial ``model.config_hash`` is *not* promoted to
      ``model_artifact_hash`` (would fake a C1 identity); see loss row.
    """
    if "input_token_ids" not in document or "first_token" not in document:
        raise SchemaError(
            "document does not look like a legacy golden file",
            details={"error_code": "not_legacy_golden"},
        )
    family_info = source_version_info(document)
    model_block = document.get("model", {}) or {}
    generated = list(document.get("generated_tokens", []))

    losses = _environment_losses(document) + _model_losses(document)
    losses.append(
        _loss_row(
            "(no timing recorded)",
            "raw_samples[0].prefill/itl timing fields",
            "insufficient", "historical_only",
            "golden records WHAT was produced, not HOW LONG; the prefill/"
            "first-token/itl sample fields are explicit zero placeholders "
            "with loss_class=insufficient and must not be read as measured "
            "performance",
        )
    )
    losses.append(
        _loss_row(
            "generated_tokens", "raw_samples[0].generated_token_ids", "none",
            "historical_only",
            "golden token sequence preserved verbatim",
        )
    )
    losses.append(
        _loss_row(
            "first_token", "summary.first_token", "none", "historical_only",
            "top-K token ids/logits and L2 norm preserved verbatim under "
            "summary.first_token",
        )
    )
    for key in ("model.id", "model.source", "model.dtype"):
        if model_block.get(key.split(".")[1]):
            losses.append(
                _loss_row(key, "artifact_links." + key.split(".")[1], "none",
                          "regression_ready", "string identity copied verbatim")
            )

    raw_sample: Dict[str, Any] = {
        "input_tokens": int(document["input_tokens"]),
        "output_tokens": int(document["output_tokens"]),
        "generated_token_ids": generated,
        # Explicit placeholder, not a measurement.
        "prefill_forward_ms": 0.0,
        "first_token_selection_ms": 0.0,
        "itl_ms": [],
        "peak_cuda_allocated_mb": 0.0,
        "peak_cuda_reserved_mb": 0.0,
    }

    artifact_links = {
        "legacy_kind": "golden",
        "model_id": str(model_block.get("id", "")),
        "source": str(model_block.get("source", "")),
        "dtype": str(model_block.get("dtype", "")),
    }
    if model_block.get("config_hash"):
        artifact_links["model_config_sha256"] = str(model_block["config_hash"])

    result = BenchmarkResult(
        run_id=new_run_id(),
        timestamp=document.get("timestamp", time.time()),
        environment=_environment(document),
        model_artifact_hash=None,
        config_hash=None,
        workload=_workload_golden(document),
        raw_samples=[raw_sample],
        summary={
            "first_token": document.get("first_token"),
            "input_token_ids": document.get("input_token_ids"),
        },
        correctness=None,
        resource=None,
        artifact_links=artifact_links,
    )
    _attach_migration_meta(result, document, "legacy_golden", family_info,
                           losses)
    return result


def migrate_legacy_result(document: Mapping[str, Any]) -> BenchmarkResult:
    """Migrate a legacy model-core result document to C6 :class:`BenchmarkResult`.

    Semantic rules (E01-07):

    * each ``repetitions`` entry becomes one raw sample with **every** source
      field preserved verbatim under the same key (including
      ``model_core_*``, ``tegrastats_*``, ``energy_*``, memory and process
      fields).  Unknown/extra fields are never silently deleted just to make
      the output "fit" the new envelope;
    * ITL is **summary-only** in the legacy format (no per-token raw list);
      the per-repetition summary dict is preserved under
      ``sample["itl"]``/``summary["itl_summary"]`` and ``itl_ms`` stays empty
      — never fabricated from the mean;
    * ``deterministic`` is a token-repetition flag, not a reference
      correctness gate; it is recorded in ``summary`` and **not** promoted to
      ``CorrectnessReport``.
    * ``generated_token_sha256`` is preserved under ``summary``; timing/
      power/energy fields that have a C6-native meaning are also mirrored to
      their C6-native location without dropping the legacy copy.
    """
    if "repetitions" not in document:
        raise SchemaError(
            "document does not look like a legacy result file",
            details={"error_code": "not_legacy_result"},
        )
    family_info = source_version_info(document)
    model_block = document.get("model", {}) or {}

    losses = _environment_losses(document)
    losses.append(
        _loss_row(
            "repetitions[*].itl", "summary.itl_summary / sample.itl_ms=[]",
            "insufficient", "historical_only",
            "legacy result stores ITL summary statistics only; per-token "
            "raw_itl_ms is not recoverable. itl_ms is left empty and the "
            "summary dict is preserved verbatim; no mean-replication raw "
            "samples are fabricated",
        )
    )
    losses.append(
        _loss_row(
            "deterministic", "summary.deterministic", "representation",
            "historical_only",
            "legacy deterministic flag means repeated runs produced the same "
            "generated-token hash; it is NOT a reference-correctness verdict, "
            "so CorrectnessReport is left null",
        )
    )
    for key in ("id", "backend", "dtype"):
        if model_block.get(key):
            losses.append(
                _loss_row(
                    f"model.{key}", f"artifact_links.{key}", "none",
                    "regression_ready", "string identity copied verbatim",
                )
            )
    for key in ("source", "local_path", "attention_backend"):
        if model_block.get(key):
            losses.append(
                _loss_row(
                    f"model.{key}", f"artifact_links.{key}", "representation",
                    "historical_only",
                    "loader/provenance detail preserved verbatim under "
                    "artifact_links but has no C1/C6-native field",
                )
            )
    if model_block.get("load_time_s") is not None:
        losses.append(
            _loss_row(
                "model.load_time_s", "summary.model_load_time_s",
                "representation", "historical_only",
                "model warmup/load wall-time preserved under summary; it is "
                "not part of the timed prefill/decode samples",
            )
        )

    workload = _workload_result(document)

    raw_samples: List[Dict[str, Any]] = []
    itl_summaries: List[Dict[str, Any]] = []
    for rep in document.get("repetitions", []):
        itl_data = rep.get("itl", {}) or {}
        itl_summaries.append(dict(itl_data))
        # Preserve the ENTIRE legacy repetition verbatim so no measured
        # metric is silently dropped; then mirror native C6 semantics.
        sample: Dict[str, Any] = {k: v for k, v in rep.items()}
        sample["input_tokens"] = int(rep.get("input_tokens", workload.input_tokens))
        sample["output_tokens"] = int(rep.get("output_tokens", workload.output_tokens))
        sample["generated_token_ids"] = list(rep.get("generated_token_ids", []))
        sample["itl_ms"] = []  # explicit: not recoverable (see loss row)
        raw_samples.append(sample)

    artifact_links: Dict[str, str] = {
        "legacy_kind": "result",
        "model_id": str(model_block.get("id", "")),
        "backend": str(model_block.get("backend", "")),
        "dtype": str(model_block.get("dtype", "")),
    }
    for key in ("source", "local_path", "attention_backend"):
        if model_block.get(key):
            artifact_links[key] = str(model_block[key])

    summary: Dict[str, Any] = {
        "deterministic": document.get("deterministic"),
        "generated_token_sha256": document.get("generated_token_sha256"),
        "itl_summary": itl_summaries,
    }
    if model_block.get("load_time_s") is not None:
        summary["model_load_time_s"] = model_block["load_time_s"]

    result = BenchmarkResult(
        run_id=new_run_id(),
        timestamp=document.get("timestamp", time.time()),
        environment=_environment(document),
        model_artifact_hash=None,
        config_hash=None,
        workload=workload,
        raw_samples=raw_samples,
        summary=summary,
        correctness=None,
        resource=None,
        artifact_links=artifact_links,
    )
    _attach_migration_meta(result, document, "legacy_result", family_info,
                           losses)
    return result


def _attach_migration_meta(
    result: BenchmarkResult,
    document: Mapping[str, Any],
    family: str,
    family_info: Dict[str, Any],
    losses: List[Dict[str, str]],
) -> None:
    """Attach machine-readable migration provenance to the C6 summary."""
    source_hash = _canonical_hash(dict(document))
    migration = {
        "migrator": "hqsb.core.schema.migrate",
        "migrator_version": MIGRATOR_VERSION,
        "source_family": family,
        "source_version": family_info["source_version"],
        "declared_version": family_info.get("declared_version"),
        "source_version_policy": family_info["policy"],
        "target_schema": "C6/BenchmarkResult",
        "target_version": C6_SCHEMA_VERSION,
        "source_document_hash": source_hash,
        "execution": {
            # NOTE: the migration-execution run_id is carried by the top-level
            # result.run_id (and never duplicated here) so that the *semantic*
            # payload of the summary is stable across re-runs; re-running only
            # changes the execution identity, not the meaning (E01-07 step 9).
            "note": "migration-execution provenance lives in result.run_id; "
            "the original experiment provenance (legacy timestamp / model "
            "block) is preserved separately and is never overwritten",
        },
        "field_mapping_losses": losses,
        "loss_summary": {
            cls: sum(1 for row in losses if row["loss_class"] == cls)
            for cls in ("none", "representation", "insufficient",
                        "unexpressible", "semantic_change")
        },
        "defaults_filled": defaults_filled(family, document),
    }
    result.summary["migration"] = migration
    result.summary["migration_losses"] = losses


def is_current_c6_document(document: Mapping[str, Any]) -> bool:
    """Heuristic for an *already current* C6 envelope.

    A current :class:`BenchmarkResult` carries a ``run_id`` plus a
    ``raw_samples`` list (and never the legacy golden/result markers).  This
    lets ``migrate_any`` accept a current document as a no-op instead of
    wrapping it again (E01-07 step 9: no nested migration metadata, no new
    experiment identity).
    """
    markers = {"run_id", "raw_samples"}
    if not markers.issubset(document):
        return False
    if "input_token_ids" in document or "repetitions" in document:
        # The document claims current markers but still carries legacy golden/
        # result markers — ambiguous; do not guess.
        return False
    return True


def migrate_any(document: Mapping[str, Any]) -> BenchmarkResult:
    """Auto-detect the document, gate the version, migrate.

    * an already-current C6 envelope is accepted as a no-op (parsed once and
      returned unchanged; no nested migration metadata is added);
    * a legacy golden/result document is migrated after the source-version
      gate.

    Raises:
        SchemaError: If the document matches neither legacy shape nor current
            C6.
        UnsupportedSchemaVersionError / SchemaMigrationRequiredError:
            For a declared future/older legacy version (or a future C6).
    """
    if is_current_c6_document(document):
        # model_post_init performs the C6 version gate; a future C6 payload is
        # refused here.
        return BenchmarkResult.model_validate(dict(document))
    detected = detect_family(document)
    source_version_info(document)  # version gate before any conversion
    if detected["family"] == "legacy_golden":
        return migrate_legacy_golden(document)
    return migrate_legacy_result(document)


# ── Planning / dry-run support (E01-07 steps 4–6) ──────────────────────


def plan_migration(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a migration plan without writing any output.

    The plan freezes the field mapping, loss rows and the source→target
    version pair exactly as ``migrate_any`` would apply them, so a dry-run
    and the real run cannot diverge (E01-07 step 4).
    """
    detected = detect_family(document)
    family_info = source_version_info(document)
    family = detected["family"]

    losses = _environment_losses(document) + _model_losses(document)
    if family == "legacy_golden":
        for row in (
            _loss_row("generated_tokens", "raw_samples[0].generated_token_ids",
                      "none", "historical_only",
                      "token sequence preserved verbatim"),
            _loss_row("first_token", "summary.first_token", "none",
                      "historical_only",
                      "top-K evidence preserved verbatim"),
            _loss_row(
                "(no timing recorded)",
                "raw_samples[0].prefill/itl timing fields",
                "insufficient", "historical_only",
                "golden records WHAT was produced, not HOW LONG; sample "
                "timing fields are explicit zero placeholders",
            ),
        ):
            losses.append(row)
    else:
        losses.append(
            _loss_row(
                "repetitions[*].itl", "summary.itl_summary / itl_ms=[]",
                "insufficient", "historical_only",
                "summary-only ITL; no fabricated per-token samples",
            )
        )
        losses.append(
            _loss_row(
                "deterministic", "summary.deterministic", "representation",
                "historical_only",
                "not promoted to a correctness gate",
            )
        )

    return {
        "migrator": "hqsb.core.schema.migrate",
        "migrator_version": MIGRATOR_VERSION,
        "source_family": family,
        "source_detection_reason": detected["reason"],
        "source_version": family_info["source_version"],
        "declared_version": family_info.get("declared_version"),
        "source_version_policy": family_info["policy"],
        "target_schema": "C6/BenchmarkResult",
        "target_version": C6_SCHEMA_VERSION,
        "source_document_hash": _canonical_hash(dict(document)),
        "field_mapping_losses": losses,
        "defaults_filled": defaults_filled(family, document),
        "writes_output": False,
        "note": "dry-run plan; no source or target file is modified",
    }


__all__ = [
    "C6_SCHEMA_VERSION",
    "LEGACY_FAMILY_VERSION",
    "MIGRATOR_VERSION",
    "defaults_filled",
    "detect_family",
    "is_current_c6_document",
    "migrate_any",
    "migrate_legacy_golden",
    "migrate_legacy_result",
    "plan_migration",
    "source_version_info",
]
