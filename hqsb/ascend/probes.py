"""E09-01 probe suite: device, stack, compile, load, async run, framework, profiler.

A probe answers one question with **evidence or an explicit absence of
evidence**.  Three rules shape the module:

* A missing tool is ``UNAVAILABLE`` + the reason, never ``pass`` and never a
  fabricated value.  This host has no ``npu-smi``/``msprof``/CANN, so every
  device probe returns ``UNAVAILABLE`` here — which is exactly the honest
  answer and the reason S09 is ``BLOCKED`` rather than ``PASS``.
* ``import`` success is not a probe result.  The framework probe (step 16)
  requires a tensor, a builtin op and a synchronization; step 17 separates
  "bare ACL runs" from "the framework can call it".
* Every probe records its own cleanup, because a leaked context or allocation
  invalidates the health baseline the next probe reads (E09-10 §6).

Probes take a :class:`CommandExecutor`.  Production uses
:class:`SubprocessExecutor`; tests use :class:`FixtureExecutor` so the parsing,
status mapping and cleanup accounting are verifiable without hardware.  A
fixture result is stamped ``simulated=True`` and can never satisfy a
prerequisite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PASS = "pass"
FAIL = "fail"
UNAVAILABLE = "UNAVAILABLE"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"

PROBE_STATUSES: Tuple[str, ...] = (PASS, FAIL, UNAVAILABLE, BLOCKED, NOT_RUN)

#: Probe layers, mirroring the E09-10 §2 fault-layer table so a probe failure
#: and a fault injection speak the same vocabulary.
LAYERS: Tuple[str, ...] = (
    "environment",
    "capability",
    "artifact",
    "compile",
    "host_tiling",
    "registration",
    "runtime_launch",
    "async_device",
    "memory",
    "profiler",
)

#: Probes that must pass before the matching chain may be called verified
#: (E09-01 step 21: device→compile→load→run→framework→profile).
MATCHING_CHAIN: Tuple[str, ...] = (
    "device_query",
    "device_memory",
    "compile",
    "load_artifact",
    "async_kernel_smoke",
    "framework_device",
    "custom_op_framework_call",
    "profiler_smoke",
)

#: Proves-of-absence for the negative matrix (E09-01 steps 22-24).
NEGATIVE_PROBES: Tuple[str, ...] = (
    "negative_version_mismatch",
    "negative_target_mismatch",
    "negative_unsupported_capability",
)

TOOL_NOT_FOUND = "tool_not_found"
FIXTURE_MISSING = "fixture_missing"


@dataclass(frozen=True)
class CommandResult:
    """One external command's outcome."""

    argv: Tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    found: bool = True
    simulated: bool = False
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout_chars": len(self.stdout),
            "stderr_chars": len(self.stderr),
            "found": self.found,
            "simulated": self.simulated,
            "error": self.error,
        }


class CommandExecutor:
    """Interface every probe uses to reach the outside world."""

    simulated = False

    def run(self, argv: Sequence[str], *, timeout_s: float = 30.0) -> CommandResult:
        raise NotImplementedError

    def which(self, program: str) -> Optional[str]:
        raise NotImplementedError


class SubprocessExecutor(CommandExecutor):
    """Runs real commands.  A missing binary is ``found=False``, not an exception."""

    def run(self, argv: Sequence[str], *, timeout_s: float = 30.0) -> CommandResult:
        if not argv:
            return CommandResult(argv=(), returncode=-1, found=False, error="empty argv")
        resolved = self.which(argv[0])
        if resolved is None:
            return CommandResult(
                argv=tuple(argv),
                returncode=127,
                found=False,
                error=f"{TOOL_NOT_FOUND}:{argv[0]}",
            )
        try:
            completed = subprocess.run(  # noqa: S603 - argv is caller-supplied, never a shell string
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                argv=tuple(argv), returncode=-1, found=True, error=f"timeout:{timeout_s}s"
            )
        except OSError as exc:
            return CommandResult(
                argv=tuple(argv), returncode=-1, found=True, error=f"{type(exc).__name__}: {exc}"
            )
        return CommandResult(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            found=True,
        )

    def which(self, program: str) -> Optional[str]:
        return shutil.which(program)


class FixtureExecutor(CommandExecutor):
    """Canned results keyed by the program name — the CPU test double."""

    simulated = True

    def __init__(self, results: Optional[Mapping[str, CommandResult]] = None) -> None:
        self.results: Dict[str, CommandResult] = dict(results or {})
        self.calls: List[Tuple[str, ...]] = []

    def set_result(self, program: str, result: CommandResult) -> None:
        self.results[program] = result

    def run(self, argv: Sequence[str], *, timeout_s: float = 30.0) -> CommandResult:
        self.calls.append(tuple(argv))
        if not argv:
            return CommandResult(argv=(), returncode=-1, found=False, error="empty argv")
        result = self.results.get(argv[0])
        if result is None:
            return CommandResult(
                argv=tuple(argv),
                returncode=127,
                found=False,
                simulated=True,
                error=f"{FIXTURE_MISSING}:{argv[0]}",
            )
        return CommandResult(
            argv=tuple(argv),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            found=result.found,
            simulated=True,
            error=result.error,
        )

    def which(self, program: str) -> Optional[str]:
        result = self.results.get(program)
        return f"/fixture/{program}" if result is not None and result.found else None


@dataclass(frozen=True)
class ProbeResult:
    """One probe's verdict, evidence and cleanup accounting."""

    probe_id: str
    layer: str
    purpose: str
    status: str
    reason: str = ""
    commands: Tuple[CommandResult, ...] = ()
    parsed: Mapping[str, Any] = field(default_factory=dict)
    cleanup: Mapping[str, Any] = field(default_factory=dict)
    capability_updates: Tuple[Mapping[str, Any], ...] = ()
    evidence_uri: str = ""
    simulated: bool = False

    def __post_init__(self) -> None:
        if self.status not in PROBE_STATUSES:
            raise ValueError(
                f"unknown probe status {self.status!r} for {self.probe_id!r}; allowed {list(PROBE_STATUSES)}"
            )

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "layer": self.layer,
            "purpose": self.purpose,
            "status": self.status,
            "reason": self.reason,
            "commands": [item.as_dict() for item in self.commands],
            "parsed": dict(self.parsed),
            "cleanup": dict(self.cleanup),
            "capability_updates": [dict(item) for item in self.capability_updates],
            "evidence_uri": self.evidence_uri,
            "simulated": self.simulated,
        }


def _unavailable(probe_id: str, layer: str, purpose: str, reason: str, commands: Sequence[CommandResult] = ()) -> ProbeResult:
    return ProbeResult(
        probe_id=probe_id,
        layer=layer,
        purpose=purpose,
        status=UNAVAILABLE,
        reason=reason,
        commands=tuple(commands),
        simulated=any(command.simulated for command in commands),
    )


def _first_missing(executor: CommandExecutor, programs: Sequence[str]) -> Optional[str]:
    for program in programs:
        if executor.which(program) is None:
            return program
    return None


# ── read-only stack probes (E09-01 steps 4-8) ────────────────────────────────


@dataclass(frozen=True)
class ProbeSpec:
    """Declarative read-only probe: argv, layer, purpose and a parser."""

    probe_id: str
    layer: str
    purpose: str
    argv: Tuple[str, ...]
    requires: Tuple[str, ...]
    parser: Callable[[str], Dict[str, Any]] = field(default=lambda _text: {})
    required: bool = True
    capability_key: str = ""


def _parse_npu_smi(text: str) -> Dict[str, Any]:
    """Keep the raw table; extract only what is unambiguous from text form."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"line_count": len(lines), "raw_head": lines[:8]}


def _parse_version_lines(text: str) -> Dict[str, Any]:
    """Parse ``key=value`` and ``key : value`` lines into a flat mapping.

    ``npu-smi`` prints aligned ``key : value`` tables while CANN's ``*.info``
    files use ``key=value``; both are accepted, otherwise the probe silently
    reports ``version_count: 0`` on a correctly installed board.  A ``:`` line
    whose key contains digits is skipped so timestamp/id noise cannot pollute
    the mapping, and lines without a value (a bare ``Usage:`` heading) are
    dropped rather than recorded as empty facts.
    """
    versions: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
        elif ":" in stripped:
            key, _, value = stripped.partition(":")
            if any(character.isdigit() for character in key):
                continue
        else:
            continue
        key, value = key.strip(), value.strip()
        if key and value:
            versions[key] = value
    return {"versions": versions, "version_count": len(versions)}


def _parse_ldd(text: str) -> Dict[str, Any]:
    """Detect one library name resolving from more than one root (step 8)."""
    resolved: Dict[str, List[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, _, target = line.partition("=>")
        name = name.strip()
        target = target.strip().split(" ")[0]
        if name and target:
            resolved.setdefault(name, []).append(target)
    conflicts = {name: paths for name, paths in resolved.items() if len(set(paths)) > 1}
    roots: Dict[str, int] = {}
    for paths in resolved.values():
        for path in paths:
            parts = path.split(os.sep)
            for index, part in enumerate(parts):
                if part.lower().startswith("cann") or part.lower() == "ascend":
                    root = os.sep.join(parts[: index + 2])
                    roots[root] = roots.get(root, 0) + 1
                    break
    return {
        "resolved_count": len(resolved),
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "install_roots_seen": roots,
        "multi_root_pollution": len(roots) > 1 or bool(conflicts),
    }


def _parse_msprof_help(text: str) -> Dict[str, Any]:
    """Read msprof's own help text.

    ``msprof --version`` is not a supported option on CANN 9.x (it exits 255),
    while ``--help`` exits 0, so the probe uses ``--help``.  The metric groups
    the tool advertises are recorded instead of a version string it never
    prints; a missing metric group stays an absence of evidence.
    """
    metric_sets: List[str] = []
    for line in text.splitlines():
        if "--aic-metrics" in line and "include" in line:
            _, _, payload = line.partition("include")
            metric_sets = [
                item.strip().rstrip(".") for item in payload.split(",") if item.strip()
            ]
            break
    return {
        "help_line_count": len(text.splitlines()),
        "mentions_aic_metrics": "--aic-metrics" in text,
        "metric_sets": metric_sets,
    }


#: Read-only stack probes.  ``npu-smi`` is the vendor tool; every argv is a
#: list, never a shell string, so a path with spaces cannot become an injection.
STACK_PROBES: Tuple[ProbeSpec, ...] = (
    ProbeSpec(
        probe_id="device_query",
        layer="environment",
        purpose="enumerate devices, chip SKU/SoC version, logical↔physical ids (step 2)",
        argv=("npu-smi", "info"),
        requires=("npu-smi",),
        parser=_parse_npu_smi,
        capability_key="device.enumeration",
    ),
    ProbeSpec(
        probe_id="device_health",
        layer="environment",
        purpose="read-only health/temperature/frequency/memory/occupancy baseline (step 4)",
        argv=("npu-smi", "info", "-t", "health", "-i", "0"),
        requires=("npu-smi",),
        parser=_parse_npu_smi,
        required=False,
    ),
    ProbeSpec(
        probe_id="firmware_driver",
        layer="environment",
        purpose="full firmware/driver component versions, install mode, loaded modules (step 5)",
        argv=("npu-smi", "info", "-t", "product", "-i", "0"),
        requires=("npu-smi",),
        parser=_parse_version_lines,
        capability_key="stack.firmware_driver_versions",
    ),
    ProbeSpec(
        probe_id="cann_components",
        layer="environment",
        purpose="toolkit/runtime/compiler/ops/kernel package versions separately (step 6)",
        argv=("bash", "-lc", "true"),  # replaced by cann_component_commands()
        requires=("bash",),
        parser=_parse_version_lines,
        capability_key="stack.cann_component_versions",
    ),
    ProbeSpec(
        probe_id="env_audit",
        layer="environment",
        purpose="Ascend path/visible-device/log/compile-target env vars (step 8)",
        argv=("env",),
        requires=("env",),
        parser=lambda text: {
            "ascend_vars": sorted(
                line.partition("=")[0]
                for line in text.splitlines()
                if line.split("=", 1)[0].upper().startswith(("ASCEND", "LD_LIBRARY_PATH", "PATH", "DEVICE", "NPU"))
            )
        },
        required=False,
    ),
    ProbeSpec(
        probe_id="profiler_version",
        layer="profiler",
        purpose="msprof presence and declared metric set (step 18 prerequisite)",
        argv=("msprof", "--help"),
        requires=("msprof",),
        parser=_parse_msprof_help,
        capability_key="profiler.msprof_available",
    ),
)


def cann_component_commands(cann_root: str) -> Tuple[Tuple[str, ...], ...]:
    """Candidate argv list for the CANN component probe, in priority order.

    Paths come from configuration, never from a hard-coded default: writing this
    host's absolute path into a tracked file is forbidden (AGENTS.md §6).

    CANN 9.x ships no ``version.cfg``; its component versions live in
    ``<root>/{opp,compiler}/version.info``.  The caller tries each candidate in
    turn and treats "this release does not ship that file" as an absence of
    evidence rather than a capability failure.
    """
    if not cann_root:
        raise ValueError("cann_root must be provided by the locked environment config")
    return (
        ("cat", os.path.join(cann_root, "version.cfg")),
        ("cat", os.path.join(cann_root, "version.info")),
        ("cat", os.path.join(cann_root, "compiler", "version.info")),
        ("cat", os.path.join(cann_root, "opp", "version.info")),
        ("cat", os.path.join(cann_root, "aarch64-linux", "ascend_toolkit_install.info")),
        ("cat", os.path.join(cann_root, "latest", "ascend_toolkit_install.info")),
        ("ls", os.path.join(cann_root, "opp", "built-in", "op_impl")),
    )


def run_probe_spec(spec: ProbeSpec, executor: CommandExecutor, *, argv: Optional[Sequence[str]] = None) -> ProbeResult:
    """Execute one declarative probe."""
    command_argv = tuple(argv) if argv is not None else spec.argv
    missing = _first_missing(executor, spec.requires)
    if missing is not None:
        return _unavailable(
            spec.probe_id,
            spec.layer,
            spec.purpose,
            f"{TOOL_NOT_FOUND}:{missing} — the probe cannot run on this host; it is not a pass",
        )
    result = executor.run(command_argv)
    if not result.found:
        return _unavailable(spec.probe_id, spec.layer, spec.purpose, result.error or TOOL_NOT_FOUND, (result,))
    if result.returncode != 0:
        return ProbeResult(
            probe_id=spec.probe_id,
            layer=spec.layer,
            purpose=spec.purpose,
            status=FAIL,
            reason=f"exit={result.returncode}: {result.stderr.strip()[:400]}",
            commands=(result,),
            simulated=result.simulated,
        )
    parsed = spec.parser(result.stdout)
    return ProbeResult(
        probe_id=spec.probe_id,
        layer=spec.layer,
        purpose=spec.purpose,
        status=PASS,
        reason="command succeeded",
        commands=(result,),
        parsed=parsed,
        capability_updates=(
            ({"key": spec.capability_key, "probe_id": spec.probe_id, "probe_status": PASS},)
            if spec.capability_key
            else ()
        ),
        simulated=result.simulated,
    )


def probe_dynamic_library_resolution(
    executor: CommandExecutor, targets: Sequence[str]
) -> ProbeResult:
    """Step 8: resolve key binaries/extensions and detect multi-root pollution."""
    spec_id = "dynamic_library_resolution"
    purpose = "resolve dynamic libraries for key binaries and detect multi-CANN-root pollution (step 8)"
    if not targets:
        return _unavailable(spec_id, "environment", purpose, "no target binaries configured")
    ldd = _first_missing(executor, ("ldd",))
    if ldd is not None:
        return _unavailable(spec_id, "environment", purpose, f"{TOOL_NOT_FOUND}:{ldd}")
    results: List[CommandResult] = []
    merged = ""
    for target in targets:
        result = executor.run(("ldd", target))
        results.append(result)
        merged += result.stdout + "\n"
    parsed = _parse_ldd(merged)
    parsed["targets"] = list(targets)
    status = FAIL if parsed["conflict_count"] else PASS
    reason = (
        f"{parsed['conflict_count']} library name(s) resolve from more than one location"
        if status == FAIL
        else "no conflicting resolution"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="environment",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=tuple(results),
        parsed=parsed,
        capability_updates=(
            {"key": "stack.no_library_pollution", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=any(item.simulated for item in results),
    )


# ── runtime / compile / load / async chain (E09-01 steps 12-17) ───────────────


@dataclass(frozen=True)
class ApiCall:
    """One runtime API call with its return code (step 12 requires each)."""

    api: str
    returncode: int = 0
    ok: bool = True
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"api": self.api, "returncode": self.returncode, "ok": self.ok, "detail": self.detail}


def probe_device_memory(executor: CommandExecutor, harness: Optional[Sequence[str]] = None) -> ProbeResult:
    """Step 12: context/stream, small allocation, H↔D copy, free.

    A device *query* success must not substitute for this (E09-01 step 12): the
    probe needs the runtime API return codes and its own cleanup record.
    """
    spec_id = "device_memory"
    purpose = "create context/stream, allocate, copy H↔D and free; record each API return code (step 12)"
    argv = tuple(harness) if harness else ("hqsb-ascend-memory-probe",)
    missing = _first_missing(executor, (argv[0],))
    if missing is not None:
        return _unavailable(spec_id, "memory", purpose, f"{TOOL_NOT_FOUND}:{missing}")
    result = executor.run(argv)
    calls = _parse_api_calls(result.stdout)
    cleanup = {
        "released": all(call.api != "free" or call.ok for call in calls) and bool(calls),
        "api_call_count": len(calls),
        "failed_apis": [call.api for call in calls if not call.ok],
    }
    if not result.found:
        return _unavailable(spec_id, "memory", purpose, result.error or TOOL_NOT_FOUND, (result,))
    status = PASS if (result.returncode == 0 and all(call.ok for call in calls) and calls) else FAIL
    reason = (
        "all runtime API calls returned success and cleanup completed"
        if status == PASS
        else f"exit={result.returncode}, failed_apis={cleanup['failed_apis']}"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="memory",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed={"api_calls": [call.as_dict() for call in calls]},
        cleanup=cleanup,
        capability_updates=(
            {"key": "runtime.device_memory_roundtrip", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def _parse_api_calls(text: str) -> List[ApiCall]:
    """Parse ``api=<name> rc=<code>`` lines emitted by the probe harnesses."""
    calls: List[ApiCall] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("api="):
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        name = fields.get("api", "")
        if not name:
            continue
        code = int(fields.get("rc", "0") or 0)
        calls.append(ApiCall(api=name, returncode=code, ok=code == 0, detail=fields.get("detail", "")))
    return calls


def probe_compile(
    executor: CommandExecutor,
    *,
    source_uri: str,
    soc_version: str,
    compiler: str = "ccec",
    build_dir: str = "",
    disable_cache: bool = True,
) -> ProbeResult:
    """Step 13: compile a minimal Ascend C source for the *actual* SoC.

    ``disable_cache`` defaults to True because E09-02 step 17 requires a
    cache-cleared rebuild to prove the artifact is not a stale binary.
    """
    spec_id = "compile"
    purpose = "compile a minimal Ascend C source for the real SoC target; record command/target/version/log (step 13)"
    if not source_uri:
        return _unavailable(spec_id, "compile", purpose, "no source URI configured")
    if not soc_version:
        return _unavailable(spec_id, "compile", purpose, "SoC version unknown; compiling for a guessed target is forbidden")
    missing = _first_missing(executor, (compiler,))
    if missing is not None:
        return _unavailable(
            spec_id,
            "compile",
            purpose,
            f"{TOOL_NOT_FOUND}:{compiler} — the CANN compiler is absent, so no Ascend artifact can be produced here",
        )
    argv = (compiler, f"--soc={soc_version}", "-O2", source_uri)
    if build_dir:
        argv += ("-o", os.path.join(build_dir, "probe_kernel.o"))
    if disable_cache:
        argv += ("--no-cache",)
    result = executor.run(argv, timeout_s=600.0)
    status = PASS if result.returncode == 0 else FAIL
    return ProbeResult(
        probe_id=spec_id,
        layer="compile",
        purpose=purpose,
        status=status,
        reason=(
            "compiler produced an artifact"
            if status == PASS
            else f"exit={result.returncode}: {result.stderr.strip()[:600]}"
        ),
        commands=(result,),
        parsed={
            "soc_version": soc_version,
            "compiler": compiler,
            "cache_disabled": disable_cache,
            "source_uri": source_uri,
        },
        capability_updates=(
            {"key": "toolchain.ascend_c_compile", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def probe_load_artifact(executor: CommandExecutor, *, artifact_uri: str, expected_soc: str) -> ProbeResult:
    """Step 14: load the just-built artifact and check name/symbol/ABI/SoC."""
    spec_id = "load_artifact"
    purpose = "load the compiled artifact; verify registered name, symbol, ABI and SoC target (step 14)"
    if not artifact_uri:
        return _unavailable(spec_id, "artifact", purpose, "no artifact URI; the compile probe must run first")
    loader = "hqsb-ascend-load-probe"
    missing = _first_missing(executor, (loader,))
    if missing is not None:
        return _unavailable(spec_id, "artifact", purpose, f"{TOOL_NOT_FOUND}:{loader}")
    result = executor.run((loader, "--artifact", artifact_uri, "--soc", expected_soc))
    parsed = {"artifact_uri": artifact_uri, "expected_soc": expected_soc}
    parsed.update(_parse_version_lines(result.stdout))
    mismatch = parsed.get("soc_mismatch") == "true"
    status = FAIL if (result.returncode != 0 or mismatch) else PASS
    reason = (
        "target mismatch: the artifact was built for another SoC"
        if mismatch
        else ("artifact loaded and symbols resolved" if status == PASS else f"exit={result.returncode}")
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="artifact",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed=parsed,
        cleanup={"context_released": parsed.get("context_released", "unknown") == "true"},
        capability_updates=(
            {"key": "toolchain.artifact_load", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def probe_async_kernel_smoke(executor: CommandExecutor, *, stream: str = "non_default") -> ProbeResult:
    """Step 15: deterministic Add on an explicit stream, read after a real sync.

    The probe keeps ``host_enqueue`` and ``sync_complete`` as separate statuses:
    an async launch that succeeded is *not* evidence the kernel computed.
    """
    spec_id = "async_kernel_smoke"
    purpose = "launch a deterministic Add on an explicit stream and read the result after a sync boundary (step 15)"
    harness = "hqsb-ascend-async-probe"
    missing = _first_missing(executor, (harness,))
    if missing is not None:
        return _unavailable(spec_id, "async_device", purpose, f"{TOOL_NOT_FOUND}:{harness}")
    result = executor.run((harness, "--stream", stream))
    fields_parsed = _parse_version_lines(result.stdout)
    enqueue = str(fields_parsed.get("versions", {}).get("host_enqueue_status", ""))
    complete = str(fields_parsed.get("versions", {}).get("sync_complete_status", ""))
    matched = str(fields_parsed.get("versions", {}).get("result_matches_oracle", "")) == "true"
    status = PASS if (enqueue == "ok" and complete == "ok" and matched) else FAIL
    reason = (
        "enqueue and synchronized completion both reported and the result matched the oracle"
        if status == PASS
        else f"host_enqueue={enqueue or 'unreported'} sync_complete={complete or 'unreported'} matches={matched}"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="async_device",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed={
            "host_enqueue_status": enqueue,
            "sync_complete_status": complete,
            "result_matches_oracle": matched,
            "stream": stream,
        },
        capability_updates=(
            {"key": "runtime.async_launch_and_sync", "probe_id": spec_id, "probe_status": status},
            {"key": "stream.non_default", "probe_id": spec_id, "probe_status": status if stream != "default" else PASS},
        ),
        simulated=result.simulated,
    )


def probe_framework_device(executor: CommandExecutor, *, integration_path: str) -> ProbeResult:
    """Step 16: create a device tensor, run a builtin op, synchronize.

    ``import`` success is explicitly rejected as a pass condition.
    """
    spec_id = "framework_device"
    purpose = "create an NPU tensor, run a known builtin op and synchronize; import success is not a pass (step 16)"
    if integration_path in ("", NOT_RUN, "NOT_SELECTED", UNAVAILABLE):
        return _unavailable(
            spec_id,
            "registration",
            purpose,
            f"no integration path selected (got {integration_path!r}); E09-05 §1 requires exactly one main path",
        )
    harness = "hqsb-ascend-framework-probe"
    missing = _first_missing(executor, (harness,))
    if missing is not None:
        return _unavailable(spec_id, "registration", purpose, f"{TOOL_NOT_FOUND}:{harness}")
    result = executor.run((harness, "--path", integration_path))
    parsed = _parse_version_lines(result.stdout).get("versions", {})
    required_fields = ("requested_device", "actual_device", "dtype", "format", "op_executed", "synchronized")
    missing_fields = [name for name in required_fields if not parsed.get(name)]
    status = FAIL if (result.returncode != 0 or missing_fields) else PASS
    reason = (
        "device tensor, builtin op and synchronization all reported"
        if status == PASS
        else f"exit={result.returncode}, missing_fields={missing_fields}"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="registration",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed=dict(parsed),
        capability_updates=(
            {"key": f"framework.{integration_path}", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def probe_custom_op_framework_call(executor: CommandExecutor, *, op_name: str, integration_path: str) -> ProbeResult:
    """Step 17: call a minimal *custom* op through the chosen binding.

    Separating this from :func:`probe_framework_device` is the point: "bare ACL
    runs" and "the framework can call our op" fail for different reasons, and
    E09-05 needs to know which.
    """
    spec_id = "custom_op_framework_call"
    purpose = "call a minimal custom Ascend C op through the planned binding/dispatcher (step 17)"
    if not op_name:
        return _unavailable(spec_id, "registration", purpose, "no custom op name configured")
    harness = "hqsb-ascend-customop-probe"
    missing = _first_missing(executor, (harness,))
    if missing is not None:
        return _unavailable(spec_id, "registration", purpose, f"{TOOL_NOT_FOUND}:{harness}")
    result = executor.run((harness, "--op", op_name, "--path", integration_path))
    parsed = _parse_version_lines(result.stdout).get("versions", {})
    hit = parsed.get("actual_kernel_hit") == "true"
    status = PASS if (result.returncode == 0 and hit) else FAIL
    reason = (
        "the custom kernel was actually hit (not a framework decomposition)"
        if status == PASS
        else f"exit={result.returncode}, actual_kernel_hit={parsed.get('actual_kernel_hit', 'unreported')}"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="registration",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed=dict(parsed),
        capability_updates=(
            {"key": f"custom_op.{op_name}", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def probe_profiler(executor: CommandExecutor, *, output_dir: str, metric_set: str = "") -> ProbeResult:
    """Step 18: profile a short workload and confirm parseable output.

    A missing metric is recorded as unavailable, never filled with zero
    (E09-01 §10, details README §8.4).
    """
    spec_id = "profiler_smoke"
    purpose = "profile a short workload; confirm msprof starts, writes parseable files and exposes the expected levels (step 18)"
    missing = _first_missing(executor, ("msprof",))
    if missing is not None:
        return _unavailable(
            spec_id,
            "profiler",
            purpose,
            f"{TOOL_NOT_FOUND}:msprof — profiling capability is absent, which blocks E09-04/E09-08 "
            "separately from the compute chain (E09-01 §9)",
        )
    argv = ["msprof", "--output", output_dir, "--aic-metrics", metric_set or "PipeUtilization"]
    result = executor.run(tuple(argv), timeout_s=600.0)
    produced: List[str] = []
    if os.path.isdir(output_dir):
        for root, _dirs, files in os.walk(output_dir):
            produced.extend(os.path.join(root, name) for name in sorted(files))
    parseable = bool(produced)
    status = PASS if (result.returncode == 0 and parseable) else FAIL
    reason = (
        f"profiler wrote {len(produced)} parseable file(s)"
        if status == PASS
        else f"exit={result.returncode}, files={len(produced)} — no metric may be filled with zero"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="profiler",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed={
            "output_dir": output_dir,
            "metric_set": metric_set or "PipeUtilization",
            "file_count": len(produced),
            "files": produced[:20],
            "unavailable_metrics": [] if parseable else [metric_set or "PipeUtilization"],
        },
        capability_updates=(
            {"key": "profiler.msprof_smoke", "probe_id": spec_id, "probe_status": status},
        ),
        simulated=result.simulated,
    )


def _state_from_answer(answer: str) -> str:
    """Map a probe answer to a capability state *suggestion*.

    A suggestion only: :func:`hqsb.ascend.capability.merge_probe_updates` still
    needs an official citation before it will emit ``SUPPORTED_VERIFIED``
    (E09-01 §4.4 requires official evidence AND a local probe).
    """
    if answer == "yes":
        return "SUPPORTED_UNVERIFIED"
    if answer == "no":
        return "UNSUPPORTED"
    return "UNKNOWN"


def _answer_status(answer: str) -> str:
    return PASS if answer in ("yes", "no") else UNAVAILABLE


def probe_dtype_layout_format(
    executor: CommandExecutor, matrix: Sequence[Mapping[str, Any]]
) -> ProbeResult:
    """Step 19: storage / cast / actual-compute support, recorded separately.

    ``matrix`` rows are ``{"key", "storage", "cast", "compute_kernel"}`` where
    each value is ``yes``/``no``/``unknown``.  Conflating them is exactly the
    E09-06 §3 error this probe exists to prevent.
    """
    spec_id = "dtype_layout_format"
    purpose = "probe each dtype/format for storage, cast and actual compute-kernel support separately (step 19)"
    if not matrix:
        return _unavailable(spec_id, "capability", purpose, "no dtype/format matrix configured")
    rows: List[Dict[str, Any]] = []
    updates: List[Dict[str, Any]] = []
    for row in matrix:
        key = str(row["key"])
        storage = str(row.get("storage", "unknown"))
        cast = str(row.get("cast", "unknown"))
        compute = str(row.get("compute_kernel", "unknown"))
        rows.append({"key": key, "storage": storage, "cast": cast, "compute_kernel": compute})
        updates.append(
            {
                "key": f"dtype.{key}.compute",
                "state_suggestion": _state_from_answer(compute),
                "probe_id": spec_id,
                "probe_status": _answer_status(compute),
            }
        )
        updates.append(
            {
                "key": f"dtype.{key}.storage",
                "state_suggestion": _state_from_answer(storage),
                "probe_id": spec_id,
                "probe_status": _answer_status(storage),
            }
        )
    # The probe itself is a *declaration audit*: it can run without a device, but
    # an all-``unknown`` matrix means nothing was actually probed.
    all_unknown = all(row["compute_kernel"] == "unknown" for row in rows)
    status = UNAVAILABLE if all_unknown else PASS
    reason = (
        "every compute-kernel column is unknown; a probe is required before any of these may be claimed"
        if all_unknown
        else f"{len(rows)} dtype/format row(s) classified"
    )
    return ProbeResult(
        probe_id=spec_id,
        layer="capability",
        purpose=purpose,
        status=status,
        reason=reason,
        parsed={"rows": rows},
        capability_updates=tuple(updates),
    )


def probe_stream_workspace_error(executor: CommandExecutor) -> ProbeResult:
    """Step 20: non-default stream, custom workspace, event timing, async error."""
    spec_id = "stream_workspace_error"
    purpose = "confirm non-default stream, custom workspace, event timing, async error query and context lifecycle (step 20)"
    harness = "hqsb-ascend-stream-probe"
    missing = _first_missing(executor, (harness,))
    if missing is not None:
        return _unavailable(spec_id, "runtime_launch", purpose, f"{TOOL_NOT_FOUND}:{harness}")
    result = executor.run((harness,))
    parsed = _parse_version_lines(result.stdout).get("versions", {})
    required = ("non_default_stream", "custom_workspace", "event_timing", "async_error_query", "context_lifecycle")
    absent = [name for name in required if parsed.get(name) not in ("yes", "no")]
    status = FAIL if (result.returncode != 0 or absent) else PASS
    reason = (
        "all five runtime capability questions answered"
        if status == PASS
        else f"exit={result.returncode}, unanswered={absent}"
    )
    updates = [
        {"key": f"runtime.{name}", "probe_id": spec_id, "probe_status": PASS, "answer": parsed.get(name)}
        for name in required
    ]
    return ProbeResult(
        probe_id=spec_id,
        layer="runtime_launch",
        purpose=purpose,
        status=status,
        reason=reason,
        commands=(result,),
        parsed=dict(parsed),
        capability_updates=tuple(updates),
        simulated=result.simulated,
    )


# ── negative injections (E09-01 steps 22-24) ──────────────────────────────────


@dataclass(frozen=True)
class NegativeInjection:
    """A safely-injected mismatch and what the preflight must do about it.

    Safety is declared structurally, not in prose: :attr:`isolation_method` must
    be one of :data:`SAFE_ISOLATION_METHODS` and :attr:`modifies_shared_stack`
    must stay ``False``.  A substring search over the description would accept an
    unsafe method described politely and reject a safe one described bluntly.
    """

    probe_id: str
    kind: str
    isolation_method: str
    expected_error_class: str
    fixture_uri: str
    forbidden_action: str
    modifies_shared_stack: bool = False

    def __post_init__(self) -> None:
        if self.isolation_method not in SAFE_ISOLATION_METHODS:
            raise ValueError(
                f"{self.probe_id}: isolation method {self.isolation_method!r} is not one of "
                f"{list(SAFE_ISOLATION_METHODS)}; E09-01 §6 requires a container, an isolated "
                "virtualenv, a read-only fixture or mock metadata"
            )
        if self.modifies_shared_stack:
            raise ValueError(
                f"{self.probe_id}: this injection would modify the shared driver/firmware/CANN "
                "install, which E09-01 §2 and E09-10 §4 forbid on a shared host"
            )
        if not self.expected_error_class:
            raise ValueError(
                f"{self.probe_id}: an injection without an expected error class cannot be judged"
            )
        if not self.fixture_uri:
            raise ValueError(
                f"{self.probe_id}: the injected artifact must be a named read-only fixture"
            )


#: Isolation methods E09-01 §6 permits for a "wrong version" test.
SAFE_ISOLATION_METHODS: Tuple[str, ...] = (
    "container",
    "isolated_virtualenv",
    "read_only_fixture",
    "mock_metadata",
)


NEGATIVE_INJECTIONS: Tuple[NegativeInjection, ...] = (
    NegativeInjection(
        probe_id="negative_version_mismatch",
        kind="framework_version_mismatch",
        isolation_method="isolated_virtualenv",
        expected_error_class="ENV_MISMATCH",
        fixture_uri="configs/ascend/compatibility_spec.yaml#negative_fixtures.framework_version",
        forbidden_action="must not modify the host driver, firmware or shared CANN install",
    ),
    NegativeInjection(
        probe_id="negative_target_mismatch",
        kind="wrong_soc_artifact",
        isolation_method="read_only_fixture",
        expected_error_class="ARTIFACT_INCOMPATIBLE",
        fixture_uri="configs/ascend/compatibility_spec.yaml#negative_fixtures.wrong_soc_artifact",
        forbidden_action="must not write to the shared artifact cache",
    ),
    NegativeInjection(
        probe_id="negative_unsupported_capability",
        kind="unsupported_dtype_layout_op",
        isolation_method="mock_metadata",
        expected_error_class="UNSUPPORTED",
        fixture_uri="configs/ascend/compatibility_spec.yaml#negative_fixtures.unsupported_capability",
        forbidden_action="must not silently cast to FP16 and report the requested dtype as successful",
    ),
)


def run_negative_injection(
    injection: NegativeInjection,
    preflight: Callable[[NegativeInjection], Mapping[str, Any]],
) -> ProbeResult:
    """Run one negative injection through the *preflight*, not the device.

    The whole point (E09-01 §9) is that a mismatch must fail **before** model
    load.  A crash at the first kernel means E09-01 fails even though the error
    was eventually caught.
    """
    outcome = dict(preflight(injection))
    detected = str(outcome.get("error_class", "")) == injection.expected_error_class
    before_load = bool(outcome.get("detected_before_model_load", False))
    fields_reported = bool(outcome.get("field_pointers"))
    status = PASS if (detected and before_load and fields_reported) else FAIL
    reason = (
        f"preflight returned {injection.expected_error_class} before model load with field-level detail"
        if status == PASS
        else (
            f"error_class={outcome.get('error_class')!r} (expected {injection.expected_error_class!r}), "
            f"before_load={before_load}, field_pointers={fields_reported}"
        )
    )
    return ProbeResult(
        probe_id=injection.probe_id,
        layer="environment" if injection.kind.startswith("framework") else "artifact",
        purpose=f"safely inject {injection.kind} and require a preflight failure ({injection.isolation_method})",
        status=status,
        reason=reason,
        parsed={
            "kind": injection.kind,
            "isolation_method": injection.isolation_method,
            "expected_error_class": injection.expected_error_class,
            "forbidden_action": injection.forbidden_action,
            "observed": outcome,
        },
        cleanup={"context_released": bool(outcome.get("cleanup_complete", False))},
    )


# ── plan / summary ───────────────────────────────────────────────────────────


@dataclass
class ProbeSummary:
    """All probe results, shaped for :func:`hqsb.ascend.compatibility.compatibility_verdict`."""

    results: Dict[str, ProbeResult] = field(default_factory=dict)
    required_probes: Tuple[str, ...] = MATCHING_CHAIN
    official_coverage_ok: bool = False

    def add(self, result: ProbeResult) -> None:
        self.results[result.probe_id] = result

    @property
    def simulated(self) -> bool:
        """True if any result came from a fixture executor.

        A simulated summary can never satisfy a prerequisite: the interface
        tests exercise probe *logic*, they do not produce environment evidence.
        """
        return any(result.simulated for result in self.results.values())

    def statuses(self) -> Dict[str, str]:
        return {probe_id: self.results[probe_id].status for probe_id in sorted(self.results)}

    def chain_status(self) -> Dict[str, Any]:
        """The device→compile→load→run→framework→profile chain of step 21."""
        rows = [
            {"probe": probe_id, "status": self.results[probe_id].status if probe_id in self.results else NOT_RUN}
            for probe_id in self.required_probes
        ]
        first_failure = next((row["probe"] for row in rows if row["status"] not in (PASS,)), "")
        return {
            "chain": rows,
            "complete": all(row["status"] == PASS for row in rows),
            "first_non_pass": first_failure,
            # E09-01 §9: a compute chain that passes with a dead profiler is a
            # distinct verdict, not a lower bar.
            "blocked_by_profiler_capability": (
                all(row["status"] == PASS for row in rows if row["probe"] != "profiler_smoke")
                and self.results.get("profiler_smoke") is not None
                and self.results["profiler_smoke"].status != PASS
            ),
        }

    def capability_updates(self) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for probe_id in sorted(self.results):
            updates.extend(dict(item) for item in self.results[probe_id].capability_updates)
        return updates

    def as_dict(self) -> Dict[str, Any]:
        return {
            "simulated": self.simulated,
            "official_coverage_ok": self.official_coverage_ok,
            "required_probes": list(self.required_probes),
            "statuses": self.statuses(),
            "chain": self.chain_status(),
            "probes": {probe_id: self.results[probe_id].status for probe_id in self.results},
            "results": {probe_id: self.results[probe_id].as_dict() for probe_id in sorted(self.results)},
            "capability_updates": self.capability_updates(),
        }

    def verdict_inputs(self, *, hash_recomputable: bool) -> Dict[str, Any]:
        """Exactly the shape :func:`compatibility.compatibility_verdict` expects."""
        return {
            "hash_recomputable": hash_recomputable,
            "official_coverage_ok": self.official_coverage_ok,
            "required_probes": list(self.required_probes),
            "probes": {probe_id: self.results[probe_id].status for probe_id in self.results},
            "simulated": self.simulated,
        }


def run_stack_probes(
    executor: CommandExecutor, *, cann_root: str = "", library_targets: Sequence[str] = ()
) -> ProbeSummary:
    """Run every read-only stack probe, tolerating absent tools."""
    summary = ProbeSummary(results={})
    for spec in STACK_PROBES:
        if spec.probe_id == "cann_components":
            if not cann_root:
                summary.add(
                    _unavailable(
                        spec.probe_id,
                        spec.layer,
                        spec.purpose,
                        "no CANN root configured; guessing an install path would audit the wrong stack",
                    )
                )
                continue
            # Try each candidate location.  Every CANN release ships a different
            # subset of version files, so "this release has no version.cfg" is an
            # absence of evidence, not a capability failure — returning FAIL for
            # it would misreport a healthy board.
            resolved: Optional[ProbeResult] = None
            for argv in cann_component_commands(cann_root):
                candidate = run_probe_spec(spec, executor, argv=argv)
                if candidate.status == PASS:
                    resolved = candidate
                    break
            if resolved is None:
                resolved = _unavailable(
                    spec.probe_id,
                    spec.layer,
                    spec.purpose,
                    "none of the configured CANN version files could be read; "
                    "an absent file is not a capability failure",
                )
            summary.add(resolved)
            continue
        summary.add(run_probe_spec(spec, executor))
    if library_targets:
        summary.add(probe_dynamic_library_resolution(executor, library_targets))
    return summary


__all__ = [
    "ApiCall",
    "BLOCKED",
    "CommandExecutor",
    "CommandResult",
    "FAIL",
    "FixtureExecutor",
    "LAYERS",
    "MATCHING_CHAIN",
    "NEGATIVE_PROBES",
    "NOT_RUN",
    "NegativeInjection",
    "NEGATIVE_INJECTIONS",
    "PASS",
    "ProbeResult",
    "ProbeSpec",
    "ProbeSummary",
    "SAFE_ISOLATION_METHODS",
    "STACK_PROBES",
    "SubprocessExecutor",
    "UNAVAILABLE",
    "cann_component_commands",
    "probe_async_kernel_smoke",
    "probe_compile",
    "probe_custom_op_framework_call",
    "probe_device_memory",
    "probe_dtype_layout_format",
    "probe_dynamic_library_resolution",
    "probe_framework_device",
    "probe_load_artifact",
    "probe_profiler",
    "probe_stream_workspace_error",
    "run_negative_injection",
    "run_probe_spec",
    "run_stack_probes",
]
