#!/usr/bin/env python3
"""Recompute the S05 decision audit from saved evidence, on the remote target.

Standard library aggregation; --plot uses an already-installed matplotlib. This
audit never initializes a model or device, changes upstream verdicts, or turns
an incomplete research candidate into a release.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path("docs/stage_experiments/S05")
METHODS = ("fp16", "rtn_w8", "rtn_w4")
GATES = ("correctness", "artifact", "quality", "execution", "measurement")
DELIVERY_SIDECARS = {"frontend_api_validation.json", "frontend_api_validation.log", "frontend_api_unit_tests.log"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def interval(values):
    values = list(values)
    if not values:
        return {"mean": None, "median": None, "ci95": None, "n_processes": 0}
    mean = statistics.mean(values)
    ci = None
    method = "unavailable: fewer than two independent processes"
    if len(values) == 3:
        half = 4.302652729911275 * statistics.stdev(values) / math.sqrt(3)
        ci, method = [mean - half, mean + half], "Student-t, df=2, independent process means"
    elif len(values) > 1:
        rng = random.Random(51010)
        samples = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(4000)]
        ci, method = [quantile(samples, .025), quantile(samples, .975)], "process bootstrap percentile, B=4000"
    return {"mean": mean, "median": statistics.median(values), "ci95": ci,
            "n_processes": len(values), "process_values": values, "ci_method": method}


def ratio_interval(baseline, candidate):
    if not baseline or not candidate or any(value <= 0 for value in candidate):
        return {"ratio": None, "ci95": None}
    rng = random.Random(51010)
    draws = [statistics.mean(rng.choices(baseline, k=len(baseline))) /
             statistics.mean(rng.choices(candidate, k=len(candidate))) for _ in range(4000)]
    return {"ratio": statistics.mean(baseline) / statistics.mean(candidate),
            "ci95": [quantile(draws, .025), quantile(draws, .975)],
            "ci_method": "independent process bootstrap; not paired run-index samples; B=4000"}


class Sources:
    def __init__(self, root):
        self.root, self.records = root, {}

    def record(self, relative):
        relative = Path(relative)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source escaped repository")
        key = str(relative)
        self.records[key] = {"path": key, "exists": path.is_file(),
                             "sha256": sha(path) if path.is_file() else None,
                             "bytes": path.stat().st_size if path.is_file() else None}
        return self.records[key]

    def read(self, relative):
        record = self.record(relative)
        return json.loads((self.root / relative).read_text()) if record["exists"] else {}

    def rows(self, relative):
        record = self.record(relative)
        return [json.loads(line) for line in (self.root / relative).read_text().splitlines() if line.strip()] if record["exists"] else []


def stage_status(value):
    return value.get("overall", value.get("status", "NOT_EXECUTED"))


def quality_metrics(document):
    teacher = document.get("teacher_forcing", [])
    weighted = [(row.get("nll"), row.get("valid_next_tokens")) for row in teacher]
    valid = [(nll, count) for nll, count in weighted if number(nll) and number(count) and count > 0]
    nll = sum(value * count for value, count in valid) / sum(count for _, count in valid) if valid else None
    comparisons = [row.get("comparison", {}).get("summary", {}) for row in teacher]
    cosine = [row["mean_cosine"] for row in comparisons if number(row.get("mean_cosine"))]
    agreement = [row["top1_agreement"] for row in comparisons if number(row.get("top1_agreement"))]
    return {"aggregate_nll": nll, "aggregate_ppl": math.exp(nll) if nll is not None else None,
            "minimum_logit_cosine": min(cosine) if cosine else None,
            "minimum_top1_agreement": min(agreement) if agreement else None,
            "task_accuracy": document.get("summary", {}).get("task_accuracy"),
            "finite": bool(teacher) and all(row.get("finite") is True for row in teacher),
            "quality_ci95": None,
            "quality_ci_reason": "No preregistered paired non-inferiority CI / representative critical-slice protocol in E05-02"}


def collect_runs(sources, spec):
    process_rows, request_rows, run_records, issues = [], [], [], []
    expected_workloads = {row["name"] for row in spec.get("workloads", [])}
    for method in METHODS:
        for index in range(3):
            relative = STAGE / f"E05-02/raw/runs/{method}_run_{index}.json"
            run = sources.read(relative)
            if not run:
                issues.append({"method": method, "run_index": index, "reason": "missing process raw"})
                continue
            source = sources.records[str(relative)]
            env = run.get("environment", {})
            hardware = {key: env.get(key) for key in ("hostname", "device", "compute_capability", "platform", "torch", "cuda_runtime")}
            fingerprint = digest(hardware)
            names = {row.get("name") for row in run.get("performance", [])}
            identity_valid = (run.get("method") == method and run.get("run_index") == index
                              and run.get("spec_hash") == spec.get("spec_hash") and names == expected_workloads)
            record = {"method": method, "run_id": run.get("run_id"), "run_index": index,
                      "source_path": str(relative), "source_sha256": source["sha256"],
                      "spec_hash": run.get("spec_hash"), "hardware_fingerprint": fingerprint,
                      "hardware": hardware, "identity_valid": identity_valid,
                      "execution_truth": run.get("execution_truth"),
                      "cold_load_s": run.get("load_memory", {}).get("load_time_s"),
                      "first_request": run.get("first_request"),
                      "load_memory": run.get("load_memory"),
                      "artifact_load": run.get("artifact_load"),
                      "steady_memory": run.get("steady_memory_before_workloads")}
            run_records.append(record)
            if not identity_valid:
                issues.append({"method": method, "run_index": index, "reason": "run/spec/workload identity mismatch; excluded"})
                continue
            telemetry_file = run.get("telemetry_file")
            if telemetry_file:
                sources.record(telemetry_file)
            for wi, workload in enumerate(run["performance"]):
                samples = workload.get("samples", [])
                base = {key: record[key] for key in ("method", "run_id", "run_index", "source_path", "source_sha256", "spec_hash", "hardware_fingerprint")}
                base.update({"workload": workload["name"], "input_tokens": workload.get("input_tokens"),
                             "output_tokens": workload.get("output_tokens"), "batch_size": workload.get("batch_size"),
                             "source_pointer": f"/performance/{wi}"})
                values = {}
                for si, sample in enumerate(samples):
                    metrics = {"ttft_ms": sample.get("model_core_ttft_ms"),
                               "prefill_ms": sample.get("prefill_forward_ms"),
                               "e2e_ms": sample.get("model_core_e2e_ms"),
                               "decode_total_ms": sample.get("decode_total_ms"),
                               "prefill_tokens_per_s": sample.get("prefill_tokens_per_s"),
                               "decode_tokens_per_s": sample.get("decode_tokens_per_s"),
                               "output_tokens_per_s": sample.get("model_core_output_tokens_per_s"),
                               "host_rss_bytes": sample.get("process_rss_bytes"),
                               "host_swap_bytes": sample.get("process_swap_bytes"),
                               "kv_bytes": sample.get("kv_cache_total_bytes")}
                    itl = sample.get("raw_itl_ms", [])
                    metrics["tpot_ms"] = statistics.mean(itl) if itl else None
                    metrics["itl_p95_ms"] = quantile(itl, .95) if itl else None
                    for name in ("allocated", "reserved"):
                        value = sample.get(f"peak_cuda_{name}_mb")
                        metrics[f"device_peak_{name}_bytes"] = value * 1024**2 if number(value) else None
                    request_rows.append({**base, "repetition": si, "metrics": metrics,
                                         "source_pointer": f"/performance/{wi}/samples/{si}"})
                    for key, value in metrics.items():
                        if number(value):
                            values.setdefault(key, []).append(value)
                means = {key: statistics.mean(items) if len(items) == len(samples) else None for key, items in values.items()}
                telemetry = workload.get("telemetry", {})
                energy = telemetry.get("j_per_request") if telemetry.get("energy_usable") is True else None
                means["energy_j_per_request"] = energy if number(energy) else None
                means["energy_j_per_output_token"] = energy / base["output_tokens"] if number(energy) and base["output_tokens"] else None
                means["energy_j_per_input_token"] = energy / base["input_tokens"] if number(energy) and base["input_tokens"] else None
                process_rows.append({**base, "metrics": means, "requests_in_process": len(samples),
                                     "warmup_count": len(workload.get("warmup", [])), "anomalies": workload.get("anomalies", []),
                                     "telemetry": telemetry,
                                     "energy_scope": "SoC VDD_IN request window; not GPU-only; no idle subtraction; phase energy unavailable"})
    return process_rows, request_rows, run_records, issues


def summarize(process_rows):
    grouped = {}
    for row in process_rows:
        key = (row["method"], row["workload"], row["hardware_fingerprint"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (method, workload, hardware), rows in sorted(grouped.items()):
        names = sorted({key for row in rows for key in row["metrics"]})
        for metric in names:
            values = [row["metrics"].get(metric) for row in rows]
            available = [value for value in values if number(value)]
            summary = interval(available)
            output.append({"method": method, "workload": workload, "hardware_fingerprint": hardware,
                           "metric": metric, **summary, "missing_processes": len(values) - len(available),
                           "process_mean_scope": "requests averaged within process; process is independent unit",
                           "source_refs": [{key: row[key] for key in ("run_id", "source_path", "source_sha256", "source_pointer", "spec_hash")} for row in rows]})
    return output


def gate(status, reason, evidence):
    return {"status": status, "reason": reason, "evidence": evidence}


def build_registry(sources, spec, upstream, runs, process_rows):
    artifact = sources.read(STAGE / "E05-02/raw/artifact_summary.json")
    quality = {method: sources.read(STAGE / f"E05-02/raw/quality/{method}/quality.json") for method in METHODS}
    qm = {method: quality_metrics(value) for method, value in quality.items()}
    result = []
    methods = [("fp16", "E05-02", 16), ("rtn_w8", "E05-02", 8), ("rtn_w4", "E05-02", 4),
               ("gptq_w4", "E05-04", 4), ("awq_w4", "E05-04", 4), ("smoothquant_w8a8", "E05-04", 8),
               ("mixed_precision", "E05-05", None), ("hqsb_fused_w8", "E05-06", 8), ("hqsb_fused_w4", "E05-06", 4),
               ("activation_w8a8_static", "E05-07", 8), ("activation_w8a8_dynamic", "E05-07", 8),
               ("kv_int8", "E05-08", 8), ("kv_int4", "E05-08", 4)]
    for method, experiment, bits in methods:
        recorded = [row for row in runs if row["method"] == method]
        rows = [row for row in process_rows if row["method"] == method]
        art = artifact.get("methods", {}).get(method, {})
        profiler = sources.read(STAGE / f"E05-02/raw/profiler/{method}/summary.json") if method in METHODS else {}
        identity = {"method": method, "bits": bits, "model": spec.get("model"),
                    "resolved_config": spec.get("methods", {}).get(method),
                    "canonical_artifact_hash": art.get("artifact_id"), "packed_variant_hash": None,
                    "mixed_policy_hash": None, "calibration_hash": None,
                    "model_module_policy": spec.get("module_policy") if method in METHODS else None,
                    "runtime": "torch FP16 eager" if method in METHODS else None,
                    "hardware_fingerprints": sorted({row["hardware_fingerprint"] for row in recorded}),
                    "source_experiment": experiment,
                    "source_verdict_sha256": sources.records[str(STAGE / experiment / "raw/verdict.json")]["sha256"],
                    "performance_result_hashes": [row["source_sha256"] for row in recorded],
                    "quality_result_hash": sources.records.get(str(STAGE / f"E05-02/raw/quality/{method}/quality.json"), {}).get("sha256")}
        candidate = {"candidate_id": f"{method}:{digest(identity)[:16]}", "identity": identity, "method": method,
                     "source_experiment": experiment, "source_verdict": stage_status(upstream[experiment]),
                     "complete_identity": method in METHODS and len(recorded) == 3,
                     "quality": qm.get(method), "artifact": art,
                     "offline_quant_s": art.get("offline", {}).get("wall_time_s"),
                     "profile": {key: profiler.get(key) for key in ("available", "native_low_bit_kernel", "expected_kernel", "observed_execution", "fallback_reason", "trace_sha256")},
                     "actual_low_bit_call_coverage": 0 if method in METHODS else None,
                     "actual_low_bit_time_coverage": 0 if method in METHODS else None,
                     "missing_fields": ["preregistered_quality_CI", "formal_cross_experiment_baseline", "candidate_bound_compatibility", "stable_power_clock_validation"]}
        evidence = [str(STAGE / experiment / "raw/verdict.json")]
        gates = {key: gate("INCOMPLETE", "No complete candidate-bound model validation for this gate", evidence) for key in GATES}
        if method in METHODS:
            baseline = method == "fp16"
            gates["correctness"] = gate("PASS" if qm[method]["finite"] else "FAIL", "Recorded FP16-path finite oracle; does not validate a packed low-bit model", evidence)
            gates["artifact"] = gate("INCOMPLETE", "E05-02 new-process loading recorded; formal candidate-bound E05-09 release compatibility and M4 acceptance absent", evidence + [str(STAGE / "E05-09/raw/verdict.json")])
            if baseline:
                gates["quality"] = gate("BASELINE", "Self-reference only; not an independently accepted deployment quality claim", evidence)
            else:
                inherited = upstream["E05-02"].get("quality", {}).get(method, {})
                thresholds = spec.get("quality_gates", {}).get(method, {})
                base_ppl = qm["fp16"]["aggregate_ppl"]
                q = qm[method]
                ratio = q["aggregate_ppl"] / base_ppl if q["aggregate_ppl"] and base_ppl else None
                checks = {"ppl_ratio": ratio is not None and ratio <= thresholds.get("max_ppl_ratio", 0),
                          "logit_cosine": q["minimum_logit_cosine"] is not None and q["minimum_logit_cosine"] >= thresholds.get("min_mean_logit_cosine", 1),
                          "top1_agreement": q["minimum_top1_agreement"] is not None and q["minimum_top1_agreement"] >= thresholds.get("min_top1_agreement", 1)}
                candidate["quality"].update({"ppl_ratio": ratio, "thresholds": thresholds, "recomputed_checks": checks,
                                             "upstream_gate": inherited})
                gates["quality"] = gate("FAIL" if inherited.get("passed") is False or not all(checks.values()) else "INCONCLUSIVE", "Preserve preregistered point-gate failures; missing paired quality CI cannot become PASS", evidence)
            native = bool(recorded) and all(row.get("execution_truth", {}).get("native_low_bit_kernel") is True for row in recorded)
            gates["execution"] = gate("PASS" if baseline else "PASS" if native else "FAIL",
                                      "Real FP16 reference" if baseline else "Whole-weight FP16 dequant control; no packed low-bit model execution", evidence)
            measurement_valid = (len(recorded) == 3 and all(row["identity_valid"] for row in recorded)
                                 and len({row["run_id"] for row in recorded}) == 3
                                 and len(rows) == 18 and all(row["requests_in_process"] >= 2 and row["warmup_count"] >= 1 for row in rows))
            gates["measurement"] = gate("INCOMPLETE", "Three-process diagnostic statistics valid; fixed clock/power and cross-experiment drift acceptance not recorded" if measurement_valid else "Process identity/repetition/workload coverage incomplete", evidence)
            candidate["diagnostic_measurement_valid"] = measurement_valid
            selected = art.get("coverage", {}).get("selected_source_bytes")
            source_bytes = artifact.get("source_load", {}).get("parameter_bytes")
            disk = art.get("disk", {}).get("total")
            candidate["storage"] = {"source_parameter_bytes": source_bytes, "quantized_artifact_bytes": disk,
                                    "retained_fp16_parameter_bytes": source_bytes - selected if number(source_bytes) and number(selected) else source_bytes if baseline else None,
                                    "whole_model_equivalent_bytes": disk + source_bytes - selected if all(number(value) for value in (disk, source_bytes, selected)) else source_bytes if baseline else None,
                                    "scope": "logical storage estimate; not runtime memory; full checkpoint disk/host peak missing"}
        candidate["gates"] = gates
        candidate["eligible_low_bit_deployment"] = bits != 16 and all(value["status"] == "PASS" for value in gates.values())
        candidate["recommendation_class"] = "reference_only" if method == "fp16" else "rejected" if any(value["status"] == "FAIL" for value in gates.values()) else "incomplete"
        candidate["exclusions"] = [{"gate": key, **value} for key, value in gates.items() if value["status"] != "PASS"]
        result.append(candidate)
    return result


def fmt(value):
    return "缺失" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def render_diagnostic(output, summary, workloads, ids):
    """Saved-data figure only; points are never labelled a deployment frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10})
    colors = {"fp16": "#205493", "rtn_w8": "#c26518", "rtn_w4": "#147d70"}
    labels = {"fp16": "FP16 reference", "rtn_w8": "RTN W8 -> FP16", "rtn_w4": "RTN W4 -> FP16"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    trace = []
    for mi, method in enumerate(METHODS):
        for wi, workload in enumerate(workloads):
            y = wi + (mi - 1) * .20
            for axis, metric, factor in ((axes[0], "tpot_ms", 1), (axes[1], "device_peak_allocated_bytes", 1024**2)):
                rows = [row for row in summary if row["method"] == method and row["workload"] == workload and row["metric"] == metric]
                if len(rows) != 1 or rows[0]["mean"] is None:
                    continue
                row = rows[0]
                x = row["mean"] / factor
                ci = row["ci95"]
                error = [[(row["mean"] - ci[0]) / factor], [(ci[1] - row["mean"]) / factor]] if ci else None
                axis.errorbar(x, y, xerr=error, fmt="o", color=colors[method], capsize=3, markersize=5,
                              linewidth=1.4, label=labels[method] if wi == 0 and metric == "tpot_ms" else None)
                trace.append({"figure": "13_tpot_memory_diagnostic.svg", "candidate_id": ids[method], "method": method,
                              "workload": workload, "metric": metric, "plotted_value": x, "ci95_original_units": ci,
                              "source_refs": row["source_refs"], "summary_source": "metrics/summary.jsonl"})
    axes[0].set_yticks(range(len(workloads)), workloads)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("TPOT (ms) — mean and process 95% CI")
    axes[1].set_xlabel("Peak allocated device memory (MiB)")
    axes[0].set_title("Decode: three independent process means", fontsize=11, pad=12)
    axes[1].set_title("Equal memory: full weights resident in FP16", fontsize=11, pad=12)
    for axis in axes:
        axis.grid(axis="x", color="#dfe5ed", linewidth=.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=3, loc="upper center", bbox_to_anchor=(.56, .90), frameon=False)
    fig.suptitle("S05 storage-only diagnostics — no low-bit deployment recommendation", fontsize=14, fontweight="bold", y=.98)
    fig.text(.08, .035, "FP16 is a measured reference. RTN weights are fully dequantized before execution.\nCI: Student-t, df=2; requests/tokens are not independent runs. Raw-to-point links: 13_plot_points.csv.", fontsize=9, color="#465466")
    fig.subplots_adjust(left=.15, right=.975, bottom=.18, top=.76, wspace=.23)
    fig.savefig(output / "figures/13_tpot_memory_diagnostic.svg")
    fig.savefig(output / "figures/13_tpot_memory_diagnostic.png", dpi=160)
    plt.close(fig)
    table(output / "figures/13_plot_points", trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true", help="Use existing matplotlib to render diagnostic SVG/PNG; no package installation")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / STAGE / "E05-10/raw"
    output.mkdir(parents=True, exist_ok=True)
    sources = Sources(root)
    spec = sources.read(STAGE / "E05-02/raw/spec.json")
    upstream = {f"E05-{index:02d}": sources.read(STAGE / f"E05-{index:02d}/raw/verdict.json") for index in range(1, 10)}
    for index in range(1, 10):
        sources.record(STAGE / f"E05-{index:02d}/raw/EVIDENCE_MANIFEST.json")
    # Later diagnostic rounds do not rewrite the original weight-only verdict.
    # Bind them separately without promoting an incomplete mixed candidate.
    screening = sources.read(STAGE / "E05-05/model_diagnostics/summary.json")
    for name in ("verdict.json", "EVIDENCE_MANIFEST.json", "模型诊断报告.md"):
        sources.record(STAGE / "E05-05/model_diagnostics" / name)
    sources.record(STAGE / "E05-06/raw/model/probe.json")
    source_code = Path(__file__).resolve()
    sources.record(source_code.relative_to(root))
    details = Path("docs/stage_experiments/details/S05/E05-10_pareto_deployment_decision.md")
    sources.record(details)
    process_rows, request_rows, runs, issues = collect_runs(sources, spec)
    summary = summarize(process_rows)
    candidates = build_registry(sources, spec, upstream, runs, process_rows)
    ids = {row["method"]: row["candidate_id"] for row in candidates}
    for row in process_rows + request_rows + runs + summary:
        row["candidate_id"] = ids[row["method"]]
    decision_spec = {"schema": "hqsb.e05_10.decision/v1", "experiment_id": "E05-10",
                     "scope": "Retrospective evidence-completeness and diagnostic decision audit; no new inference",
                     "quality_rule": "Preserve frozen E05-02 gates; paired quality CIs missing => no formal admission",
                     "ci": "Student-t df=2 on three independent process means; token/request samples are not independent runs",
                     "ratio_ci": "unpaired process bootstrap B=4000, seed=51010",
                     "hardware": "Exact E05-02 hardware fingerprint only; no cross-device claims",
                     "workloads": spec.get("workloads", []), "business_slo": None,
                     "business_slo_reason": "Not supplied; do not invent a latency/energy SLO to select a winner",
                     "objectives": ["device_peak_allocated_bytes", "ttft_ms", "tpot_ms", "energy_j_per_output_token"],
                     "dominance": "All gates PASS, same hardware/workload, all dimensions <= and at least one <; robust only if all CI bounds separate",
                     "missing_values": "null/INCOMPLETE; never zero-fill energy or substitute artifact bytes for allocated memory",
                     "format_deviation": "Dependency-free JSON/JSONL/CSV replace parquet/yaml; no misleading renamed binary files",
                     "release_policy": "This diagnostic audit cannot authorize an S06 release"}
    write(output / "spec.json", decision_spec)
    write(output / "environment.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": platform.node(),
                                          "python": platform.python_version(), "execution": "Jetson via scripts/remote_run.sh; CPU saved-data aggregation",
                                          "collector_sha256": sha(source_code), "model_or_device_initialized": False})
    write(output / "prerequisites.json", {key: {"overall": stage_status(value), "scientific": value.get("scientific_execution_verdict"),
                                                "source": sources.records[str(STAGE / key / "raw/verdict.json")]} for key, value in upstream.items()})
    write(output / "model_screening_reference.json", {
        "source": str(STAGE / "E05-05/model_diagnostics/summary.json"),
        "source_sha256": sources.records[str(STAGE / "E05-05/model_diagnostics/summary.json")]["sha256"],
        "screening_status": screening["status"], "conditions": len(screening["conditions"]),
        "formal_experiment_overall": screening["formal_experiment_overall"],
        "deployable_policy_selected": screening["deployable_policy_selected"],
        "use": "diagnostic provenance only; no quality/execution/measurement gate promotion"})
    table(output / "candidates/registry", candidates)
    table(output / "metrics/request_raw", request_rows)
    table(output / "metrics/unified_raw", process_rows)
    table(output / "metrics/processes", runs)
    table(output / "metrics/summary", summary)
    for key in GATES:
        table(output / f"gates/{key}", [{"candidate_id": row["candidate_id"], "method": row["method"], **row["gates"][key]} for row in candidates])
    waterfall = [{"candidate_id": row["candidate_id"], "method": row["method"],
                  **{key: row["gates"][key]["status"] for key in GATES}, "recommendation": row["recommendation_class"],
                  "eligible_low_bit_deployment": row["eligible_low_bit_deployment"], "exclusions": row["exclusions"]} for row in candidates]
    table(output / "figures/01_gate_waterfall", waterfall)
    lookup = {(row["method"], row["workload"], row["hardware_fingerprint"], row["metric"]): row for row in summary}
    points, ratios = [], []
    for candidate in candidates:
        method = candidate["method"]
        for workload in spec.get("workloads", []):
            fingerprints = candidate["identity"]["hardware_fingerprints"] or [None]
            for fingerprint in fingerprints:
                point = {"candidate_id": candidate["candidate_id"], "method": method, "workload": workload["name"],
                         "hardware_fingerprint": fingerprint, "eligible": candidate["eligible_low_bit_deployment"],
                         "scope": "FP16 reference / fake-dequant diagnostic; not a deployment frontier" if method in METHODS else "incomplete candidate",
                         "summary_source": "metrics/summary.jsonl", "source_refs": []}
                for metric in decision_spec["objectives"] + ["e2e_ms", "prefill_tokens_per_s", "decode_tokens_per_s", "energy_j_per_request"]:
                    row = lookup.get((method, workload["name"], fingerprint, metric), {})
                    point[metric], point[metric + "_ci95"] = row.get("mean"), row.get("ci95")
                    point["source_refs"] += row.get("source_refs", [])
                points.append(point)
                if method in METHODS:
                    for metric in ("ttft_ms", "prefill_ms", "tpot_ms", "e2e_ms"):
                        measured = lookup.get((method, workload["name"], fingerprint, metric), {})
                        baseline = lookup.get(("fp16", workload["name"], fingerprint, metric), {})
                        ratios.append({"candidate_id": candidate["candidate_id"], "method": method, "workload": workload["name"], "metric": metric,
                                       "hardware_fingerprint": fingerprint,
                                       **({"ratio": 1.0, "ci95": [1.0, 1.0], "ci_method": "identical reference"} if method == "fp16" else ratio_interval(baseline.get("process_values"), measured.get("process_values"))),
                                       "scope": "FP16/fake-dequant observed ratio; no low-bit speedup claim",
                                       "source_refs": baseline.get("source_refs", []) + measured.get("source_refs", [])})
    table(output / "figures/02_memory_phase_by_workload", points)
    table(output / "figures/03_quality_storage", [{"candidate_id": row["candidate_id"], "quality": row["quality"], "storage": row.get("storage"), "scope": "quality failures retained; storage is not memory"} for row in candidates])
    table(output / "figures/04_energy_latency", points)
    table(output / "figures/05_offline_online", [{"candidate_id": row["candidate_id"], "offline_quant_s": row["offline_quant_s"], "online_scope": row["recommendation_class"], "amortization_requests": None} for row in candidates])
    table(output / "figures/06_prefill_decode_ratios", ratios)
    table(output / "figures/07_parameter_execution_coverage", [{"candidate_id": row["candidate_id"], "parameter_coverage": row["artifact"].get("coverage", {}).get("parameter_coverage"), "call_coverage": row["actual_low_bit_call_coverage"], "time_coverage": row["actual_low_bit_time_coverage"]} for row in candidates])
    calibration_path = STAGE / "E05-03/raw/selection/family_summary.jsonl"
    calibration = sources.rows(calibration_path)
    table(output / "figures/08_calibration_variance", [{**row, "source_path": str(calibration_path),
                                                      "source_sha256": sources.records[str(calibration_path)]["sha256"],
                                                      "source_line": index + 1, "status": "REJECTED_CALIBRATION_DIAGNOSTIC",
                                                      "reason": "No accepted calibration handoff; these observed seed statistics do not admit a deployment candidate"}
                                                     for index, row in enumerate(calibration)] or [{"status": "INCOMPLETE", "variance": None, "reason": "No calibration raw families"}])
    write(output / "ablations/calibration_factor_effects.json", {"source": str(STAGE / "E05-03/raw/quality/factor_effects.json"),
                                                               "observed": sources.read(STAGE / "E05-03/raw/quality/factor_effects.json"),
                                                               "scope": "Historical rejected-calibration diagnostics only"})
    table(output / "figures/09_capability_fallback", [{"candidate_id": row["candidate_id"], "profile": row["profile"], "execution_gate": row["gates"]["execution"], "source_verdict": row["source_verdict"]} for row in candidates])
    table(output / "figures/10_micro_model", [{"status": "INCOMPLETE", "micro_predicted_speedup": None, "model_speedup": None, "source": str(STAGE / "E05-06/raw/verdict.json"), "reason": "E05-06 projection diagnostics cannot be substituted for three-process six-workload model execution"}])
    table(output / "figures/11_kv_context", [{"status": "INCOMPLETE", "context": None, "tpot_ms": None, "source": str(STAGE / "E05-08/raw/verdict.json"), "reason": "No accepted candidate-bound long-generation model measurements"}])
    table(output / "figures/12_uncertainty", summary)
    frontiers = [{"hardware_fingerprint": fingerprint, "workload": workload["name"], "eligible_candidates": [], "frontier": [],
                  "baseline_candidate": ids["fp16"], "baseline_is_reference_only": True,
                  "dominance": "NOT_EVALUATED_EMPTY_ELIGIBLE_SET", "bootstrap_selection_probability": None,
                  "reason": "No low-bit candidate passed all five gates; no artificial champion"}
                 for fingerprint in sorted({row["hardware_fingerprint"] for row in runs}) for workload in spec.get("workloads", [])]
    table(output / "pareto/by_hardware_workload", frontiers)
    table(output / "pareto/dominance", frontiers)
    table(output / "pareto/bootstrap", frontiers)
    write(output / "ablations/availability.json", {"bit": "RTN W8/W4 same FP16-dequant runtime; quality/storage comparison only", "algorithm": "BLOCKED: industrial artifacts absent", "policy": "INCOMPLETE: no accepted mixed model candidate", "kernel": "INCOMPLETE: E05-06 diagnostics do not share full E05-02 model protocol", "activation_kv": "INCOMPLETE: no accepted end-to-end model evidence", "runtime": "INCOMPLETE"})
    scenarios = [{"scenario": scenario, "recommendation": [], "status": "NO_QUALIFIED_LOW_BIT_CANDIDATE", "fallback": "retain existing FP16 reference; no automatic deployment mutation", "slo": None}
                 for scenario in ("memory_budget", "interactive_latency", "batch_prefill", "long_context", "portability")]
    write(output / "recommendations/deployment_matrix.json", {"hardware_scope": decision_spec["hardware"], "scenarios": scenarios, "qualified_low_bit_recommendations": []})
    table(output / "recommendations/rejected", waterfall)
    table(output / "gates/completeness", [{"candidate_id": row["candidate_id"], "complete_identity": row["complete_identity"], "missing_fields": row["missing_fields"]} for row in candidates])
    table(output / "candidates/evidence_links", list(sources.records.values()))
    reasons = [f"{key}={stage_status(upstream[key])}" for key in ("E05-01", "E05-02", "E05-03", "E05-04", "E05-05", "E05-06", "E05-09") if stage_status(upstream[key]) != "PASS"]
    reasons += ["No accepted M4 / common cross-experiment baseline", "No complete candidate-bound five-gate low-bit deployment", "Quality non-inferiority CI and critical-slice evidence incomplete", "No accepted two-industrial-method comparison"]
    criteria = [("候选身份与证据完整", False), ("五门分开", True), ("质量非劣 CI 与关键 slice", False), ("fake-dequant 禁入低比特前沿", True),
                ("跨实验 FP16 基线一致性", False), ("六 workload 与 prefill/decode 分开", len(process_rows) == 54), ("全部核心指标与能力完整", False),
                ("按硬件/workload 独立处理", True), ("有效候选的点估计与不确定性支配", False), ("核心因果消融完整", False),
                ("理论资源/算子/模型闭环", False), ("推荐带场景/SLO/边界", False), ("失败与不完整候选保留", True),
                ("raw→summary→CSV 可追溯", bool(summary) and not issues), ("S06 可接受 release bundle", False), ("无越界 activation/KV/跨硬件 claim", True)]
    verdict = {"experiment_id": "E05-10", "overall": "BLOCKED", "scientific_execution_verdict": "COMPLETED_NEGATIVE_DECISION_AUDIT",
               "expected_effect_met": False, "single_item_standard_met": False, "formal_pass_allowed": False,
               "blocking_reasons": reasons, "source_issues": issues, "qualified_low_bit_candidates": 0, "release_allowed": False,
               "detail_pass_criteria": [{"id": index + 1, "name": name, "passed": passed} for index, (name, passed) in enumerate(criteria)],
               "claim_boundary": "Complete saved-data audit, not completed industrial/mixed/KV model experiments; numerical reference does not release S05"}
    write(output / "verdict.json", verdict)
    write(output / "release/manifest.json", {"schema": "hqsb.s05.release-decision/v1", "release_allowed": False, "s06_admission": "DENIED",
                                              "reason": reasons, "artifacts": [], "policies": [], "deployment_mutations": [],
                                              "evidence_only_bundle": True, "candidate_registry": "candidates/registry.jsonl"})
    write(output / "summary.json", {"experiment_id": "E05-10", "overall": "BLOCKED", "candidate_count": len(candidates),
                                    "process_count": len(runs), "process_workloads": len(process_rows), "request_count": len(request_rows),
                                    "metric_summary_rows": len(summary), "qualified_low_bit_candidates": 0, "release_allowed": False,
                                    "diagnostic_fp16_baseline": ids["fp16"], "tables": "figures/*.csv", "statistical_unit": "independent process"})
    lines = ["# E05-10：质量约束部署决策实验报告", "", "## 裁决", "", "**BLOCKED。完成历史原始记录的负结论决策审计；没有合规低比特部署推荐，S05 尚未达到进入 S06 的条件。**",
             "", "本次只处理已保存数据，不重新运行模型，不修改 E05-01～09 原始结论。FP16 是真实运行的诊断基线；RTN-W8/W4 全权重还原 FP16，不能作为原生低比特性能或运行内存收益。",
             "", "## 输入、统计口径与可复现性", "", f"共 {len(runs)} 个原始进程、{len(process_rows)} 个进程×workload、{len(request_rows)} 次稳态请求。逐请求原值保存在 raw/metrics/request_raw.jsonl；先对同一进程请求取均值，再用 3 个进程均值计算 Student-t 95% CI（df=2）。不把 token 或同进程重复当独立样本。", "",
             "TTFT、prefill、TPOT、decode TPS 与端到端分别保留。能耗是历史 tegrastats 的 SoC VDD_IN 请求窗口积分，未扣 idle，不是 GPU 独立能耗；phase energy 缺失。没有能量数据的候选保留 null。95% CI 是事后描述性区间，不能替代预注册质量非劣检验。n=3 区间受小样本限制。",
             "", "重算命令：`./scripts/remote_run.sh python3 scripts/audit/run_e05_10_deployment_decision.py`。输入 SHA-256 与文件路径见 raw/candidates/evidence_links.jsonl；每个统计点带 run_id/spec_hash/source_pointer。Parquet/YAML 用真实 CSV/JSONL/JSON 替代，未伪造扩展名。",
             "", "## 五门失败瀑布", "", "| 候选 | Correctness | Artifact | Quality | Execution | Measurement | 分类 |", "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join([row["method"]] + [row["gates"][key]["status"] for key in GATES] + [row["recommendation_class"]]) + " |" for row in candidates]
    lines += ["", "FP16 的 BASELINE 是自参考身份，不是已验收部署质量。RTN Quality FAIL 继承并从原始 teacher-forcing 记录复核冻结阈值；即使部分点指标通过，缺少关键 slice 和质量 CI 也不能升级 PASS。工业、mixed、W8A8、KV 与 fused 候选的局部诊断不能替代完整模型证据。", "", "## 六 workload 真实诊断结果", "", "下列数据为进程均值；完整 95% CI 见 raw/figures/02_memory_phase_by_workload.csv 与 12_uncertainty.csv。所有行均不属于已准入低比特部署 Pareto。", "", "| 方法 | workload | TTFT ms | TPOT ms | peak allocated MiB | SoC J/request |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {row['method']} | {row['workload']} | {fmt(row['ttft_ms'])} | {fmt(row['tpot_ms'])} | {fmt(row['device_peak_allocated_bytes'] / 1024**2 if row['device_peak_allocated_bytes'] is not None else None)} | {fmt(row['energy_j_per_request'])} |" for row in points if row["method"] in METHODS]
    lines += ["", "## 资源与因果链", "", "压缩 artifact + 未量化参数的字节只代表逻辑存储估算，不能替代 device allocated/reserved、load peak 或 phase peak。完整 FP16 dequant 解释了为何磁盘压缩并不自动变成运行内存下降。", "", "W8/W4 共同 FP16 执行路径只能讨论 bit 对质量与存储的影响；算法、mixed policy 与低比特 kernel 的模型级因果消融仍缺证据。E05-06 的 projection/micro 结果不能和 E05-02 六 workload 拼成同一模型加速结论。未设定真实 SLO、请求量或生命周期，故不计算任意加权总分或乐观摊销。", "", "## 必采集信息与图表交付", "", "raw/figures/01～12 提供失败瀑布、按 workload 的 memory/phase、质量/存储、能量/latency、离线成本、分 phase 比率、覆盖率、calibration、capability/fallback、micro/model、KV/context 与 CI 的 CSV/JSONL。缺少有效输入的表明确标 INCOMPLETE 和原因；没有捏造坐标点。输入来源逐项 hash；所有正式前沿为空，bootstrap 入选概率为 null，不是 0% 胜率。", "", "## 与预计效果和单项标准比较", "", "预计的部署决策产出已形成保守拒绝结果，但完整质量—内存—速度—能耗且可验收的 S05 部署证据链未形成，因此预计效果和单项通过标准均不满足。", "", "| 细则 | 标准 | 满足 |", "|---|---|---|"]
    lines += [f"| {index + 1} | {name} | {'是' if passed else '否'} |" for index, (name, passed) in enumerate(criteria)]
    lines += ["", "## S06 交接与前端访问", "", "raw/release/manifest.json 固定 `release_allowed=false`、`s06_admission=DENIED`，只允许消费诊断证据，不发布 QuantArtifact 或 mixed policy。保持当前 FP16 reference，不自动更改部署。", "", "前端证据中心、实验地图、量化与质量页按 raw/verdict.json 自动索引。新版索引支持嵌套 JSON/JSONL/CSV 原件；旧运行服务可通过同实验的《原始证据汇编》读取原样文本和来源哈希，下载格式是 Markdown 容器。实际服务与新版代码的验证分开记录在 frontend_api_validation.json；不把代码同步当作已热更新。`/research` 与 `/historical/performance` 保留原先 E05-02 专用语义。", "", "阻塞项：", ""]
    lines += [f"- {reason}" for reason in reasons]
    lines += ["", "交付验证作为独立附属记录：`frontend_api_validation.json` 及同名验证日志、`frontend_api_unit_tests.log` 不进入本实验科学数据 manifest；验证记录反向保存科学 manifest 的哈希，避免循环自引用。验证脚本仍逐项核对所有被 manifest 列出的文件，不跳过其他哈希失败。"]
    lines += ["", "## 后续模型诊断的适用范围", "",
              "E05-05 已追加 58 配置、232 次真实模型前向的 block 级单块 W4 / 留一块恢复 FP16 筛查，来源与哈希见 raw/model_screening_reference.json 及 evidence_links.jsonl。仅使用四个固定 policy 样本；它是 fake-dequant 失败归因，没有冻结 mixed policy，不能提升候选五门。E05-06 的单 projection 真实 fused prefill/decode 已观察到 kernel，decode 对同量化参考的相对误差仍超出该轮保守诊断阈值；同样不能与旧六 workload 拼成端到端部署结果。"]
    if args.plot:
        render_diagnostic(output, summary, [row["name"] for row in spec["workloads"]], ids)
        lines += ["", "## 独立诊断图", "", "![六 workload TPOT 与实测 peak memory；仅诊断，非部署前沿](raw/figures/13_tpot_memory_diagnostic.png)",
                  "", "矢量原件：raw/figures/13_tpot_memory_diagnostic.svg；每个图点的候选身份、95% CI 与原始进程文件指针见 13_plot_points.csv。"]
    report = output.parent / "E05-10_实验报告.md"
    report.write_text("\n".join(lines) + "\n")
    # Delivery validation hashes this manifest and writes a separate attestation;
    # including it here would create a circular, necessarily stale hash chain.
    entries = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha(path)}
               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "EVIDENCE_MANIFEST.json" and str(path.relative_to(output)) not in DELIVERY_SIDECARS]
    write(output / "EVIDENCE_MANIFEST.json", {"experiment_id": "E05-10", "files": entries, "file_count": len(entries),
                                             "root_hash": digest(entries), "report": {"path": "../" + report.name, "sha256": sha(report)},
                                             "separate_attestation": "frontend_api_validation.json hashes this manifest and is intentionally outside its file set",
                                             "excluded_delivery_sidecars": sorted(DELIVERY_SIDECARS)})
    print(json.dumps({"overall": "BLOCKED", "processes": len(runs), "candidates": len(candidates), "release_allowed": False, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
