"""Compare like-scoped diagnostic evidence while retaining engine changes."""


def compare(a, b):
    mismatches = []
    for key in ("tool", "source_workload_sha256", "capture_window", "replay_mode", "cache_control", "clock_control"):
        if a.get(key) != b.get(key):
            mismatches.append(key)
    for key in ("kernel_filter", "launch_skip", "launch_count", "sections"):
        if a["tool"] == "ncu" and a["plan"].get(key) != b["plan"].get(key):
            mismatches.append("selection:" + key)
    if a["tool"] == "eretrace":
        for key in ("ere_kernel", "launch_skip", "launch_count", "ere_verify"):
            if a["plan"].get(key) != b["plan"].get(key):
                mismatches.append("selection:" + key)
        if a.get("panel", {}).get("counts", {}).get("launches") != b.get("panel", {}).get("counts", {}).get("launches"):
            mismatches.append("captured_launch_count")
    for key in ("model_manifest_sha256", "hardware_inventory"):
        if a["source_engine"].get(key) != b["source_engine"].get(key):
            mismatches.append(key)
    versions = [r["source_engine"].get("versions", {}) for r in (a, b)]
    engines = [r["source_engine"].get("engine") for r in (a, b)]
    version_keys = set(versions[0]) | set(versions[1])
    if engines[0] != engines[1]:
        version_keys -= {"vllm", "sglang"}
    for key in version_keys:
        if versions[0].get(key) != versions[1].get(key):
            mismatches.append("version:" + key)
    for report in (a, b):
        if report["status"] != "completed" or report["cleanup"] != "succeeded" or not report.get("replay_workloads_match_source"):
            mismatches.append(report["run_id"] + ":incomplete_capture")
        fields = ("versions", "model_manifest_sha256", "requested", "hardware_inventory", "environment_freeze_sha256")
        attempts = report.get("attempts", [])
        environment_matches = report.get("replay_environment_matches_source", bool(attempts) and all(
            a.get("engine") and all(a["engine"].get(k) == report["source_engine"].get(k) for k in fields) for a in attempts))
        if not environment_matches:
            mismatches.append(report["run_id"] + ":environment_changed_from_source")
    options = [r["source_engine"]["requested"] for r in (a, b)]
    changes = {k: {"baseline": options[0].get(k), "candidate": options[1].get(k)}
               for k in set(options[0]) | set(options[1]) if options[0].get(k) != options[1].get(k)}
    if engines[0] != engines[1]:
        changes["engine"] = {"baseline": engines[0], "candidate": engines[1]}
    result = {"baseline_run_id": a["run_id"], "candidate_run_id": b["run_id"], "comparable": not mismatches,
              "mismatches": mismatches, "configuration_changes": changes, "measurement_role": "diagnostic",
              "interpretation": "mechanism_evidence_with_profiler_perturbation",
              "outputs_match_source": [a.get("replay_outputs_match_source"), b.get("replay_outputs_match_source")]}
    if mismatches:
        return result
    if a["tool"] == "eretrace":
        groups = []
        for report in (a, b):
            counters, occurrences = {}, {}
            for launch in report["panel"]["launches"]:
                signature = (launch["kernel"], tuple(launch["grid"]), tuple(launch["block"]))
                occurrence = occurrences.get(signature, 0)
                occurrences[signature] = occurrence + 1
                for metric in ("records", "observed_records", "elapsed_cycles_mean", "elapsed_cycles_p50", "elapsed_cycles_p95", "elapsed_cycles_max"):
                    counters[(*signature, occurrence, metric)] = launch.get(metric)
            groups.append(counters)
        rows = []
        for key in sorted(groups[0].keys() | groups[1].keys()):
            left, right = [g.get(key) for g in groups]
            rows.append({"name": key[0], "grid": key[1], "block": key[2], "occurrence": key[3], "metric": key[4],
                         "unit": "records" if "records" in key[4] else "cycles", "baseline": left, "candidate": right,
                         "delta": right - left if left is not None and right is not None else None})
        result.update(metric_changes=rows, scope="selected_kernel_signature_and_occurrence_SM_local_cycles",
                      collector_changes={"ere_codec": {"baseline": a["plan"].get("ere_codec"), "candidate": b["plan"].get("ere_codec")}},
                      sm_distributions=[r["panel"]["sm_distribution"] for r in (a, b)])
    elif a["tool"] == "nsys":
        groups = [{(k.get("device"), k["name"]): k for k in r["panel"]["top_kernels"]} for r in (a, b)]
        rows = []
        for key in groups[0].keys() | groups[1].keys():
            left, right = [g.get(key, {}) for g in groups]
            rows.append({"device": key[0], "name": key[1], "baseline_launches": left.get("launches"),
                         "candidate_launches": right.get("launches"), "baseline_summed_ms": left["summed_duration_ns"] / 1e6 if left else None,
                         "candidate_summed_ms": right["summed_duration_ns"] / 1e6 if right else None,
                         "baseline_in_top_kernels": bool(left), "candidate_in_top_kernels": bool(right)})
        result.update(kernel_changes=sorted(rows, key=lambda r: abs((r["candidate_summed_ms"] or 0) - (r["baseline_summed_ms"] or 0)), reverse=True),
                      scope="union_of_each_report_top_50_kernels_summed_durations_can_overlap")
    else:
        if any(o.get("tensor_parallel_size", 1) != 1 or o.get("pipeline_parallel_size", 1) != 1 for o in options):
            result.update(comparable=False, mismatches=["counter_matching_requires_single_rank"])
            return result
        groups = []
        for report in (a, b):
            counters, occurrences = {}, {}
            for kernel in report["panel"]["kernels"]:
                signature = (kernel["name"], kernel.get("grid"), kernel.get("block"))
                occurrence = occurrences.get(signature, 0)
                occurrences[signature] = occurrence + 1
                for metric in kernel["metrics"]:
                    counters[(*signature, occurrence, metric["name"], metric.get("unit"))] = metric.get("value")
            groups.append(counters)
        rows = []
        for key in groups[0].keys() | groups[1].keys():
            left, right = [g.get(key) for g in groups]
            rows.append({"name": key[0], "grid": key[1], "block": key[2], "occurrence": key[3], "metric": key[4],
                         "unit": key[5], "baseline": left, "candidate": right,
                         "delta": right - left if left is not None and right is not None else None})
        result.update(metric_changes=rows, scope="single_rank_kernel_signature_and_occurrence")
    return result
