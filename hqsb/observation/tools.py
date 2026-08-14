"""Whole-run Nsight capture, separated from request timing and framework capture."""

from pathlib import Path
import shutil
import subprocess
import time
import uuid

from hqsb.observation.storage import atomic_json, sha256


def capture(
    tool,
    command,
    directory,
    *,
    timeout_s=900,
    phase="prefill",
    launch_count=3,
    kernel_filter=None,
):
    if tool not in {"nsys", "ncu"}:
        raise ValueError("TOOL_UNSUPPORTED")
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("COMMAND_ARGV_INVALID")
    if (
        phase not in {"prefill", "decode"}
        or type(launch_count) is not int
        or not 1 <= launch_count <= 32
    ):
        raise ValueError("TOOL_SELECTION_INVALID")
    if kernel_filter is not None and (
        not isinstance(kernel_filter, str) or len(kernel_filter) > 256
    ):
        raise ValueError("TOOL_KERNEL_FILTER_INVALID")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    executable = shutil.which(tool)
    if tool == "ncu" and executable is None:
        candidate = Path("/opt/nvidia/nsight-compute/2024.3.1/ncu")
        executable = str(candidate) if candidate.is_file() else None
    report = {
        "schema_version": "1.0.0",
        "capture_id": "tool_" + uuid.uuid4().hex,
        "tool": tool,
        "command": command,
        "availability": "tool_unavailable",
        "measurement_role": "diagnostic_not_performance",
    }
    if executable:
        try:
            version = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, timeout=10
            )
            report["tool_version"] = (version.stdout + version.stderr)[:2000]
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["tool_version"] = f"unavailable:{type(exc).__name__}"
        prefix = str(directory / "capture")
        if tool == "nsys":
            argv = [
                executable,
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--output",
                prefix,
                *command,
            ]
        else:
            # In-model application replay; cache policy explicitly preserves app priming.
            argv = [
                executable,
                "--replay-mode",
                "application",
                "--cache-control",
                "none",
                "--clock-control",
                "none",
                "--launch-count",
                str(launch_count),
                "--csv",
                "--nvtx",
                "--nvtx-include",
                f"hqsb.{phase}/",
                "--section",
                "SpeedOfLight",
                "--section",
                "MemoryWorkloadAnalysis",
                "--export",
                prefix,
            ]
            if kernel_filter:
                argv.extend(
                    [
                        "--kernel-name-base",
                        "demangled",
                        "--kernel-name",
                        "regex:" + kernel_filter,
                    ]
                )
            argv.extend(command)
        report.update(
            argv=argv,
            replay="application" if tool == "ncu" else "single_pass",
            selection={
                "phase": phase,
                "launch_count": launch_count,
                "kernel_filter": kernel_filter,
            }
            if tool == "ncu"
            else "whole_process",
            cache_control="none" if tool == "ncu" else "application_default",
        )
        start = time.monotonic()
        with (
            (directory / "stdout.txt").open("w") as stdout,
            (directory / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                report["returncode"] = process.wait(timeout=timeout_s)
                report["availability"] = (
                    "observed" if process.returncode == 0 else "error"
                )
            except subprocess.TimeoutExpired:
                import os
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                report.update(
                    availability="partial",
                    reason="TOOL_TIMEOUT",
                    returncode=process.returncode,
                )
        report["seconds"] = time.monotonic() - start
        extension = ".ncu-rep" if tool == "ncu" else ".nsys-rep"
        if report["availability"] == "observed" and not list(
            directory.glob("*" + extension)
        ):
            report.update(availability="partial", reason="NO_NATIVE_CAPTURE_ARTIFACT")
        if tool == "ncu" and report["availability"] == "observed":
            with (
                (directory / "metrics.csv").open("w") as stdout,
                (directory / "import-stderr.txt").open("w") as stderr,
            ):
                try:
                    exported = subprocess.run(
                        [
                            executable,
                            "--import",
                            prefix + extension,
                            "--csv",
                            "--page",
                            "raw",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        timeout=60,
                    )
                    report["metrics_export_returncode"] = exported.returncode
                except subprocess.TimeoutExpired:
                    report["metrics_export_returncode"] = None
            if report["metrics_export_returncode"] != 0:
                report.update(availability="partial", reason="METRICS_EXPORT_FAILED")

    files = []
    for path in directory.iterdir():
        if path.is_file():
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    report["artifacts"] = files
    report["limitations"] = [
        "Separate capture; never merge counters into another run by temporal proximity.",
        "Selected kernel counter ratios are not a whole-model cache hit rate.",
        "Replay and profiler perturb execution; not a performance benchmark.",
    ]
    atomic_json(directory / "tool-manifest.json", report)
    return report
