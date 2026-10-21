"""Own exclusive diagnostic replays, native reports and immutable lineage."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time

from hqsb.console.service import ServiceError
from hqsb.core.diagnostics import DiagnosticPlan, profiler_arguments
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json, sha256
from hqsb.observation.nsight_tools import discover


COUNTER_PERMISSION = "GPU_COUNTER_PERMISSION_REQUIRED: 执行节点驱动限制了 GPU 硬件计数器访问。请由宿主机管理员开放 profiling 权限或授予容器相应 capability，再重试；原始错误见 profiler.log。"


def directory_bytes(root):
    """JIT compilers atomically rename/remove temporary files during capture."""
    total = 0
    for path in root.rglob("*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def log_tail(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        return stream.read().decode(errors="replace")


class DiagnosticJobs:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.root = (settings.diagnostics_dir or settings.data_dir / "diagnostics").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diagnostic")
        self.stop = threading.Event()

    def inventory(self, tool):
        if tool == "eretrace":
            from hqsb.observation.eretrace_tools import discover as ere_discover
            return ere_discover(self.settings.eretrace_runtime)
        return discover(tool, self.settings.diagnostic_tools.get(tool))

    def catalog(self):
        items = [discover(tool, self.settings.diagnostic_tools.get(tool)) for tool in ("nsys", "ncu")]
        items.append(self.inventory("eretrace"))
        recent = self.service.store.list(30, kind="diagnostic")["items"]
        for item in items:
            last = next((r for r in recent if r["config"]["diagnostic_plan"]["tool"] == item["tool"] and r["state"] in {"completed", "failed"}), None)
            if last:
                item["last_collection_run_id"] = last["id"]
                item["collection_permission"] = "permission_required" if "GPU_COUNTER_PERMISSION_REQUIRED" in (last.get("error") or "") else "collected_successfully" if last["state"] == "completed" else "last_collection_failed"
        return {"items": items,
                "source_kind": "native_benchmark", "capture_window": "steady_state_after_explicit_warmup",
                "providers": ["vllm", "sglang"], "measurement_role": "diagnostic",
                "prerequisite": "卸载驻留 GPU 模型后，从原生引擎公共压测记录发起独立诊断。"}

    def sources(self):
        items = self.service.store.list(100)["items"]
        native = self.service.store.list(100, kind="native_benchmark")["items"]
        candidates = {r["id"]: r for r in items + native if r["state"] == "completed" and
                      (r["kind"] == "native_benchmark" or (r["kind"] == "generate" and r["config"]["deployment"]["provider"] == "pytorch"))}
        return {"items": sorted(candidates.values(), key=lambda r: r["created_at"], reverse=True)}

    def start(self, raw, key):
        plan = DiagnosticPlan.model_validate(raw)
        svc = self.service
        digest = fingerprint(["diagnostic", plan.model_dump()])
        with svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute("SELECT digest,run_id FROM idempotency WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])
            svc._reserve()
            if self.stop.is_set():
                raise ServiceError("SHUTTING_DOWN", "Service closing", 503)
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请在运行记录中等待或取消当前任务。")
            if any(d["provider"] in {"pytorch", "vllm", "sglang"} and d["state"] not in {"unloaded", "failed"} for d in svc.states.values()):
                raise ServiceError("DIAGNOSTIC_UNLOAD_REQUIRED", "请在原生引擎配置或设备与部署页卸载驻留模型，再启动独立诊断。")
            try:
                source = svc.store.get(plan.source_run_id)
                cfg = svc.configs[source["config"]["deployment"]["id"]]
                pytorch = cfg.provider == "pytorch" and plan.tool == "eretrace" and plan.ere_mode == "memory"
                if pytorch:
                    from hqsb.observation.dmat_source import pytorch_reference
                    try:
                        reference = pytorch_reference(source, cfg)
                    except (ValueError, KeyError) as exc:
                        raise ServiceError("DMAT_SOURCE_REQUIRED", str(exc), 422) from exc
                    source = {**source, "metrics": {**source["metrics"], "native_benchmark": reference}}
                receipt = source["metrics"]["native_benchmark"]
                workload, engine = receipt["plan"], receipt["engine"]
            except KeyError as exc:
                raise ServiceError("DIAGNOSTIC_SOURCE_REQUIRED", "请选择部署仍在当前节点登记的原生公共压测记录。", 422) from exc
            if not pytorch and (source["kind"] != "native_benchmark" or source["state"] != "completed" or cfg.provider not in {"vllm", "sglang"}):
                raise ServiceError("DIAGNOSTIC_SOURCE_REQUIRED", "请选择已完成的原生公共压测记录。", 422)
            if sha256(Path(cfg.manifest).expanduser()) != engine["model_manifest_sha256"]:
                raise ServiceError("DIAGNOSTIC_ARTIFACT_CHANGED", "源运行制品清单与当前部署发生变化，请重新建立评测记录。", 422)
            if fingerprint({k: v for k, v in workload.items() if k != "expected_epoch"}) != receipt["workload_sha256"]:
                raise ServiceError("DIAGNOSTIC_WORKLOAD_CHANGED", "源记录工作负载哈希校验失败，请重新评测。", 422)
            if plan.tool == "eretrace" and not pytorch:
                from hqsb.observation.eretrace_tools import validate_scope
                try:
                    validate_scope(engine["requested"], workload)
                except ValueError as exc:
                    raise ServiceError("ERE_SOURCE_SCOPE", str(exc), 422) from exc
            inventory = self.inventory(plan.tool)
            if not inventory["installed"]:
                raise ServiceError("DIAGNOSTIC_TOOL_REQUIRED", inventory.get("reason", "工具版本检查失败"), 422)
            if plan.tool == "eretrace" and plan.ere_mode == "memory" and "dmat-dual-clock" not in inventory.get("capabilities", []):
                raise ServiceError("DMAT_RUNTIME_REQUIRED", "请用当前安装脚本构建含 dmat-dual-clock 能力的 ERETrace runtime，并更新服务配置。", 422)
            budget = plan.artifact_limit_mb * 1024**2
            used = directory_bytes(self.root)
            if shutil.disk_usage(self.root).free < budget + 256 * 1024**2 or used + budget > self.settings.artifact_budget_bytes:
                raise ServiceError("DIAGNOSTIC_DISK_BUDGET", "请在机器与存储页为诊断报告释放空间或调整观测配额。")
            config = {**cfg.model_dump(), "engine_log_root": str(self.root),
                      "engine_cache_root": str(self.settings.native_engine_cache_dir or self.settings.data_dir / "native-engines/cache")}
            if pytorch:
                config["_pytorch_dmat"] = True
            if plan.tool == "eretrace":
                config["_eretrace"] = {"runtime": inventory["executable"], "plan": plan.model_dump()}
            run, _ = svc.store.create("diagnostic", {"deployment": cfg.public(), "diagnostic_plan": plan.model_dump(),
                                     "source_run_id": source["id"], "measurement_role": "diagnostic"}, key, digest)
            svc.pending += 1
            svc.benchmark_owner = "diagnostic:" + run["id"]
            self.executor.submit(self.execute, run["id"], config, plan, source, inventory)
            return run

    def execute(self, identity, config, plan, source, inventory):
        svc = self.service
        directory = self.root / identity
        process = None
        report = {"schema_version": "nsight-v1", "run_id": identity, "source_run_id": source["id"],
                  "tool": plan.tool, "tool_inventory": inventory, "plan": plan.model_dump(),
                  "measurement_role": "diagnostic", "capture_window": "requests_after_load_and_explicit_warmup",
                  "replay_mode": "application" if plan.tool == "ncu" else "single_pass",
                  "cache_control": "none", "clock_control": "none", "request_kernel_attribution": "engine_instance_and_capture_window"}
        if plan.tool == "eretrace":
            report.update(schema_version="eretrace-v1", request_kernel_attribution="selected_kernel_launch_CTA_warp_in_engine_window",
                          sampling_scope="kernel_name_substring_then_launch_skip_and_attempt_count",
                          selection_endpoint="explicit_worker_stop_after_measured_requests",
                          cold_probe_compilation="included_in_diagnostic_interval",
                          clock_domain="clock64_per_SM", same_execution_validation=plan.ere_verify)
            if plan.ere_mode == "memory":
                report.update(schema_version="hqsb-dmat-2", clock_domain=plan.memory_clock,
                              request_kernel_attribution="selected_kernel_launch_thread_instruction_site",
                              same_execution_validation="raw_memory_event_format")
        final, error, cleanup = "failed", None, "pending"
        try:
            directory.mkdir(mode=0o700)
            source_root = Path(__file__).resolve().parents[2]
            report["diagnostic_source_sha256"] = {name: sha256(source_root / name) for name in (
                "hqsb/core/diagnostics.py", "hqsb/console/diagnostic_jobs.py", "hqsb/observation/diagnostic_target.py",
                "hqsb/observation/nsight_reports.py", "hqsb/observation/nsight_tools.py", "hqsb/observation/diagnostic_comparison.py")}
            if plan.tool == "eretrace":
                for name in ("hqsb/observation/eretrace_tools.py", "hqsb/observation/eretrace_reports.py", "hqsb/observation/eretrace_ptx.py"):
                    report["diagnostic_source_sha256"][name] = sha256(source_root / name)
                if plan.ere_mode == "memory":
                    for name in ("hqsb/observation/dmat_probe.py", "hqsb/observation/dmat_reports.py", "hqsb/observation/dmat_source.py", "hqsb/observation/dmat_target.py", "hqsb/observation/dmat_instrument.py", "hqsb/observation/dmat_grid.py", "hqsb/observation/dmat_context.py", "hqsb/backends/model_kernels/triton_attention.py"):
                        report["diagnostic_source_sha256"][name] = sha256(source_root / name)
            reference = source["metrics"]["native_benchmark"]
            spec = {"run_id": identity, "directory": str(directory), "config": config, "plan": plan.model_dump(),
                    "options": reference["engine"]["requested"], "workload": reference["plan"]}
            atomic_json(directory / "input.json", spec)
            command = [sys.executable, "-m", "hqsb.observation.diagnostic_target", str(directory / "input.json")]
            child_environment = None
            if config.get("_pytorch_dmat"):
                from hqsb.observation.eretrace_tools import launch_environment
                command[2] = "hqsb.observation.dmat_target"
                cache = directory / "triton"
                child_environment = {**os.environ, **launch_environment(inventory["executable"], directory, plan.model_dump(), cache)}
                child_environment.update(LD_PRELOAD=str(Path(inventory["executable"]) / "neutrino/build/preload-pytorch.so"), TRITON_CACHE_DIR=str(cache))
            argv = profiler_arguments(plan, inventory["executable"], directory / "capture", command)
            report.update(argv=argv, source_workload_sha256=reference["workload_sha256"],
                          source_engine=reference["engine"])
            svc.store.update(identity, {"state": "running"}, "started")
            if self.cancelled(identity):
                final = "cancelled"
            else:
                with (directory / "profiler.log").open("wb") as log:
                    process = subprocess.Popen([sys.executable, "-m", "hqsb.backends.native_supervisor", *argv],
                                               stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                               **({"env": child_environment} if child_environment else {}))
                    started = time.monotonic()
                    progress = None
                    while process.poll() is None:
                        usage = directory_bytes(directory)
                        permission_error = "ERR_NVGPUCTRPERM" in log_tail(directory / "profiler.log")
                        if permission_error or self.cancelled(identity) or time.monotonic() - started > plan.timeout_s or usage > plan.artifact_limit_mb * 1024**2:
                            final = "cancelled" if self.cancelled(identity) else "timed_out" if usage <= plan.artifact_limit_mb * 1024**2 else "failed"
                            error = "DIAGNOSTIC_ARTIFACT_LIMIT" if usage > plan.artifact_limit_mb * 1024**2 else None
                            if permission_error:
                                final, error = "failed", COUNTER_PERMISSION
                            (directory / "cancel").touch()
                            process.stdin.close()
                            process.wait(timeout=25)
                            break
                        if (directory / "progress.json").exists():
                            current = json.loads((directory / "progress.json").read_text())
                            if current != progress:
                                svc.store.update(identity, {"progress": current}, "diagnostic.progress", current)
                                progress = current
                        time.sleep(.3)
                    else:
                        final = "completed" if process.returncode == 0 else "failed"
                    report["returncode"] = process.returncode
                if final == "completed":
                    svc.store.update(identity, {"progress": {"stage": "exporting"}}, "diagnostic.progress", {"stage": "exporting"})
                    self.export(plan.tool, inventory["executable"], directory,
                                limit_bytes=plan.artifact_limit_mb * 1024**2,
                                is_cancelled=lambda: self.cancelled(identity))
                    panel = json.loads((directory / "panel.json").read_text())
                    report["panel"] = panel
                    if panel["availability"] != "observed":
                        if plan.tool == "eretrace":
                            if panel["availability"] == "partial":
                                if plan.ere_mode == "memory":
                                    raise RuntimeError("DMAT_CAPTURE_PARTIAL: 已保存实际访存图；请查看丢失事件和探针失败数量，调整每线程容量或缩小启动范围后重新采集。")
                                raise RuntimeError("ERE_CAPTURE_PARTIAL: 部分启动的 PTX 插桩失败或来自其他提交线程；已保存可用记录，请查看采集覆盖与探针日志。")
                            if plan.ere_mode == "memory":
                                raise RuntimeError("DMAT_NO_MEMORY_EVENTS: 请检查 Kernel 子串和探针日志，选择有受支持 PTX 全局读写指令的启动。覆盖记录已保存。")
                            raise RuntimeError("ERE_NO_WARP_RECORDS: 请检查 Kernel 子串、匹配启动数量和 PTX 插桩日志；本次覆盖记录已保留。")
                        raise RuntimeError("DIAGNOSTIC_NO_GPU_DATA: 请核对 Kernel 过滤、工具权限和 GPU worker 覆盖。")
                if final == "failed" and error is None:
                    tail = log_tail(directory / "profiler.log")
                    reason = "GPU_COUNTER_PERMISSION_REQUIRED" if "ERR_NVGPUCTRPERM" in tail else "NATIVE_PROFILER_FAILED"
                    error = reason + ": " + tail[-1600:]
                    if reason == "GPU_COUNTER_PERMISSION_REQUIRED":
                        error = COUNTER_PERMISSION
        except Exception as exc:
            final, error = "cancelled" if self.cancelled(identity) else "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                try:
                    process.wait(timeout=25)
                    cleanup = "succeeded"
                except subprocess.TimeoutExpired:
                    cleanup = "unconfirmed"
            else:
                cleanup = "not_started"
            report.update(status=final, cleanup=cleanup, error=error)
            if cleanup == "unconfirmed":
                report.update(status="failed", error="DIAGNOSTIC_CLEANUP_UNCONFIRMED")
            try:
                self.seal(directory, report, source)
            except Exception as exc:
                report.update(status="failed", error=f"DIAGNOSTIC_EVIDENCE_WRITE_FAILED:{type(exc).__name__}: {exc}")
            try:
                svc.store.update(identity, {"state": report["status"], "cleanup": cleanup, "error": report["error"],
                                 "progress": {"stage": report["status"]},
                                 "metrics": {"diagnostic": report}, "quality": "not_evaluated"}, "diagnostic.finished")
            finally:
                with svc.lock:
                    svc.pending -= 1
                    if svc.benchmark_owner == "diagnostic:" + identity and cleanup != "unconfirmed":
                        svc.benchmark_owner = None

    @staticmethod
    def seal(directory, report, source):
        reference = source["metrics"]["native_benchmark"]
        attempts = [json.loads(p.read_text()) for p in sorted(directory.glob("attempts/*/result.json"))]
        for attempt in sorted(directory.glob("attempts/*")):
            if not (attempt / "result.json").exists():
                engine = next(attempt.glob("engine/*/receipt.json"), None)
                if engine:
                    attempts.append({"attempt_id": attempt.name, "status": "interrupted", "error": "ATTEMPT_STOPPED_BY_SUPERVISOR",
                                     "engine": json.loads(engine.read_text())})
        report["attempts"] = [{"attempt_id": a["attempt_id"], "status": a["status"], "error": a.get("error"),
                               "engine": a.get("engine"), "window_start_unix_s": a.get("window_start_unix_s"),
                               "window_end_unix_s": a.get("window_end_unix_s")} for a in attempts]
        expected = [r.get("text") for r in reference["requests"]]
        outputs = [a.get("result", {}).get("metrics", {}).get("native_benchmark", {}).get("requests", []) for a in attempts]
        report["replay_outputs_complete"] = bool(outputs) and all(len(rows) == len(expected) and all(r.get("status") == "completed" for r in rows) for rows in outputs)
        report["replay_outputs_match_source"] = all([r.get("text") for r in rows] == expected for rows in outputs) if report["replay_outputs_complete"] else None
        if "source_output_token_ids" in reference:
            report["replay_token_ids_match_source"] = bool(attempts) and all(a.get("generated_token_ids") == reference["source_output_token_ids"] for a in attempts)
        report["replay_workload_hashes"] = [a.get("result", {}).get("metrics", {}).get("native_benchmark", {}).get("workload_sha256") for a in attempts]
        report["replay_workloads_match_source"] = bool(attempts) and all(h == reference["workload_sha256"] for h in report["replay_workload_hashes"])
        fields = ("versions", "model_manifest_sha256", "requested", "hardware_inventory", "environment_freeze_sha256", "kernel_plan_sha256")
        report["replay_environment_matches_source"] = bool(attempts) and all(
            a.get("engine") and all(a["engine"].get(k) == reference["engine"].get(k) for k in fields) for a in attempts)
        if directory.exists():
            atomic_json(directory / "receipt.json", report)
            artifacts = [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"]
            atomic_json(directory / "manifest.json", {"run_id": report["run_id"], "status": report["status"], "artifacts": artifacts})

    def cancelled(self, identity):
        return self.stop.is_set() or self.service.store.get(identity)["state"] == "cancel_requested"

    @staticmethod
    def export(tool, executable, directory, *, limit_bytes=None, is_cancelled=lambda: False):
        from hqsb.observation.nsight_reports import ncu_metrics, nsys_timeline
        if tool == "eretrace":
            from hqsb.observation.eretrace_reports import summarize
            if is_cancelled():
                raise RuntimeError("DIAGNOSTIC_EXPORT_STOPPED")
            plan = json.loads((directory / "input.json").read_text()).get("plan", {}) if (directory / "input.json").exists() else {}
            if plan.get("ere_mode") == "memory":
                from hqsb.observation.dmat_reports import summarize_memory
                panel = summarize_memory(directory, page_bytes=plan["memory_page_bytes"], time_bins=plan["memory_time_bins"], clock=plan.get("memory_clock"))
            else:
                panel = summarize(directory)
            atomic_json(directory / "panel.json", panel)
            if limit_bytes and directory_bytes(directory) > limit_bytes:
                raise RuntimeError("DIAGNOSTIC_ARTIFACT_LIMIT")
            return

        def run(argv, stdout, stderr):
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            deadline = time.monotonic() + 120
            try:
                while process.poll() is None:
                    if is_cancelled() or time.monotonic() >= deadline:
                        raise RuntimeError("DIAGNOSTIC_EXPORT_STOPPED")
                    if limit_bytes and directory_bytes(directory) > limit_bytes:
                        raise RuntimeError("DIAGNOSTIC_ARTIFACT_LIMIT")
                    time.sleep(.2)
                if process.returncode:
                    raise RuntimeError(f"DIAGNOSTIC_EXPORT_FAILED:{process.returncode}")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
        report = directory / ("capture.nsys-rep" if tool == "nsys" else "capture.ncu-rep")
        if not report.is_file():
            raise RuntimeError("NATIVE_CAPTURE_ARTIFACT_MISSING")
        with (directory / "export.log").open("w") as log:
            if tool == "nsys":
                target = directory / "capture.sqlite"
                run([executable, "export", "--type=sqlite", "--output", str(target), str(report)], log, subprocess.STDOUT)
                panel = nsys_timeline(target)
            else:
                csv_path = directory / "metrics.csv"
                with csv_path.open("w") as output:
                    run([executable, "--import", str(report), "--csv", "--page", "raw"], output, log)
                panel = ncu_metrics(csv_path.read_text())
        atomic_json(directory / "panel.json", panel)

    def artifact(self, identity, relative):
        run = self.service.store.get(identity)
        if run["kind"] != "diagnostic":
            raise ServiceError("NOT_FOUND", "请选择诊断运行", 404)
        root = (self.root / identity).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path == root / "input.json":
            raise ServiceError("NOT_FOUND", "诊断制品路径不存在", 404)
        if path != root / "manifest.json":
            try:
                manifest = json.loads((root / "manifest.json").read_text())
                entry = next(a for a in manifest["artifacts"] if a["path"] == str(path.relative_to(root)))
                if entry["bytes"] != path.stat().st_size or entry["sha256"] != sha256(path):
                    raise ServiceError("DIAGNOSTIC_ARTIFACT_CHANGED", "诊断制品与封存清单不一致，请检查执行节点文件。")
            except (FileNotFoundError, StopIteration, ValueError, KeyError) as exc:
                raise ServiceError("NOT_FOUND", "请等待诊断制品封存完成。", 404) from exc
        return path

    def memory_view(self, identity, launch_index, *, clock, page_bytes, time_bins, start=None, end=None):
        from hqsb.observation.dmat_reports import read_frame, aggregate
        if page_bytes not in (4096, 16384, 65536):
            raise ServiceError("DMAT_ADDRESS_GRAIN", "请选择 4、16 或 64 KiB 地址粒度。", 422)
        panel = json.loads(self.artifact(identity, "panel.json").read_text())
        launches = panel.get("memory_launches", [])
        if launch_index < 0 or launch_index >= len(launches):
            raise ServiceError("DMAT_LAUNCH_REQUIRED", "请选择当前诊断记录中的访存启动。", 422)
        launch = launches[launch_index]
        if (start is None) != (end is None):
            raise ServiceError("DMAT_TIME_WINDOW", "时间窗口需要同时设置起点和终点。", 422)
        try:
            plot = aggregate(read_frame(self.artifact(identity, launch["trace"])), launch["probe"]["sites"],
                             page_bytes=page_bytes, time_bins=time_bins, clock=clock,
                             tensors=launch.get("context", {}).get("tensors", []),
                             window=(start, end) if start is not None else None)
        except ValueError as exc:
            raise ServiceError("DMAT_VIEW", str(exc), 422) from exc
        return {**launch, "plot": plot, "derived_view": {
            "raw_artifact": launch["trace"], "raw_integrity": "verified_against_sealed_manifest",
            "analysis_source_sha256": {name: sha256(Path(__file__).resolve().parents[1] / "observation" / name)
                                       for name in ("dmat_grid.py", "dmat_reports.py")}}}

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True)
