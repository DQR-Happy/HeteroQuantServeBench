"""E02-08 orchestration: device state, telemetry, steady windows, mode blocks.

This module holds everything that needs a real device: it queries and mutates
the Jetson power/clock state, runs the two telemetry samplers (``tegrastats``
plus a sysfs frequency/thermal/cooling sampler), and executes the pre-registered
continuous steady-state request windows.

The *analysis* of what is collected lives in
:mod:`hqsb.benchmark.power_thermal` (pure logic, unit-tested).  Keeping the two
apart is what lets the parser, the energy integral and the thermal classifier be
tested without a GPU.

All system mutation is funnelled through :func:`set_power_mode`,
:func:`lock_clocks`, :func:`store_clock_state` and :func:`restore_clock_state`
so the runner always has a single place to record "what was changed" and a
single place to undo it.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.power_thermal import (
    POWER_THERMAL_PROTOCOL,
    parse_jetson_clocks_show,
    parse_nvpmodel_query,
)
from hqsb.benchmark.resource_monitor import TegrastatsMonitor

logger = logging.getLogger(__name__)

_GPU_DEVFREQ_DIR = "/sys/devices/platform/17000000.gpu/devfreq_dev"
_GPU_SYSFS_DIR = "/sys/devices/platform/17000000.gpu"
_EMC_RATE_PATH = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
_EMC_CAP_PATH = "/sys/kernel/nvpmodel_clk_cap/emc"
_FAN_PWM_PATH = "/sys/class/hwmon/hwmon0/pwm1"
_THERMAL_ZONE_GLOB = "/sys/class/thermal/thermal_zone*"
_COOLING_GLOB = "/sys/class/thermal/cooling_device*"
_CPU_FREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq"


# ── small sysfs helpers ────────────────────────────────────────────────────


def read_text(path: str) -> Optional[str]:
    """Read a sysfs file, returning None instead of raising on any failure.

    The read is done in binary and decoded afterwards: several Tegra thermal
    sysfs nodes (``cv0-thermal``..``cv2-thermal``) intermittently hand the text
    layer a ``None`` chunk, which raises ``TypeError`` deep inside ``codecs``
    rather than an ``OSError``.  A missing sensor must degrade the evidence, not
    kill the run.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (OSError, ValueError, TypeError):
        return None
    if not raw:
        return None
    try:
        return raw.decode("utf-8").strip()
    except (UnicodeDecodeError, TypeError, AttributeError):
        return None


def read_int(path: str) -> Optional[int]:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def read_float(path: str) -> Optional[float]:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _run_command(cmd: Sequence[str]) -> Dict[str, Any]:
    """Run a command and capture rc/stdout/stderr without ever raising."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=120
        )
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": " ".join(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


# ── device state ───────────────────────────────────────────────────────────


def query_nvpmodel() -> Dict[str, Any]:
    """Run ``nvpmodel -q --verbose`` and parse the *actual* state."""
    run = _run_command(["nvpmodel", "-q", "--verbose"])
    parsed = parse_nvpmodel_query(run["stdout"])
    parsed["command_result"] = {
        "command": run["command"],
        "returncode": run["returncode"],
        "stderr": run["stderr"],
    }
    return parsed


def query_jetson_clocks_show() -> Dict[str, Any]:
    """Run ``jetson_clocks --show`` and parse the requested/observed clocks."""
    run = _run_command(["jetson_clocks", "--show"])
    parsed = parse_jetson_clocks_show(run["stdout"] or run["stderr"])
    parsed["command_result"] = {
        "command": run["command"],
        "returncode": run["returncode"],
        "stderr": run["stderr"],
    }
    return parsed


def read_sysfs_state() -> Dict[str, Any]:
    """Read the live sysfs power/clock/thermal state (no command needed)."""
    cpu_freqs: Dict[str, Any] = {}
    for directory in sorted(glob.glob(_CPU_FREQ_GLOB)):
        cpu = directory.split("/")[-2]
        cpu_freqs[cpu] = {
            "governor": read_text(f"{directory}/scaling_governor"),
            "min_freq_hz": read_int(f"{directory}/scaling_min_freq"),
            "max_freq_hz": read_int(f"{directory}/scaling_max_freq"),
            "cur_freq_hz": read_int(f"{directory}/scaling_cur_freq"),
        }

    temperatures: Dict[str, Optional[float]] = {}
    for zone in sorted(glob.glob(_THERMAL_ZONE_GLOB)):
        name = read_text(f"{zone}/type") or os.path.basename(zone)
        millideg = read_int(f"{zone}/temp")
        temperatures[name] = millideg / 1000.0 if millideg is not None else None
        # Trip points are the device's own thermal policy; keep them as the
        # justification for the pre-registered thresholds.
        trips: Dict[str, Any] = {}
        for trip in sorted(glob.glob(f"{zone}/trip_point_*_temp")):
            key = os.path.basename(trip)[len("trip_point_") : -len("_temp")]
            value = read_int(trip)
            kind = read_text(
                f"{zone}/trip_point_{key}_type"
            )
            trips[key] = {
                "temp_c": (value / 1000.0) if value is not None else None,
                "type": kind,
            }
        temperatures[f"{name}:trips"] = trips  # type: ignore[assignment]

    cooling: Dict[str, Optional[int]] = {}
    cooling_max: Dict[str, Optional[int]] = {}
    for device in sorted(glob.glob(_COOLING_GLOB)):
        name = read_text(f"{device}/type") or os.path.basename(device)
        cooling[name] = read_int(f"{device}/cur_state")
        cooling_max[name] = read_int(f"{device}/max_state")

    return {
        "nvpmodel_clk_cap": {"emc_hz": read_int(_EMC_CAP_PATH)},
        "gpu_devfreq": {
            "governor": read_text(f"{_GPU_DEVFREQ_DIR}/governor"),
            "min_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/min_freq"),
            "max_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/max_freq"),
            "cur_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/cur_freq"),
            "target_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/target_freq"),
            "available_frequencies_hz": read_text(
                f"{_GPU_DEVFREQ_DIR}/available_frequencies"
            ),
        },
        "cpu_cpufreq": cpu_freqs,
        "emc_rate_hz": read_int(_EMC_RATE_PATH),
        "fan": {
            "pwm": read_int(_FAN_PWM_PATH),
            "hwmon_pwm1": read_int(_FAN_PWM_PATH),
        },
        "thermal_zones_c": temperatures,
        "cooling_cur_state": cooling,
        "cooling_max_state": cooling_max,
        "gpu_power_control": read_text(f"{_GPU_SYSFS_DIR}/power/control"),
        "gpu_devfreq_trans_stat": read_text(f"{_GPU_DEVFREQ_DIR}/trans_stat"),
    }


def read_device_state() -> Dict[str, Any]:
    """Full device snapshot: nvpmodel + jetson_clocks + sysfs."""
    return {
        "nvpmodel": query_nvpmodel(),
        "jetson_clocks": query_jetson_clocks_show(),
        "sysfs": read_sysfs_state(),
        "collected_at_ns": time.monotonic_ns(),
    }


def configured_state_view(state: Dict[str, Any]) -> Dict[str, Any]:
    """Project a snapshot onto the *controlled* variables used for restore checks."""
    nvpmodel = state.get("nvpmodel") or {}
    clocks = state.get("jetson_clocks") or {}
    sysfs = state.get("sysfs") or {}
    cpu = {
        name: {
            "governor": entry.get("governor"),
            "min_freq_hz": entry.get("min_freq_hz"),
            "max_freq_hz": entry.get("max_freq_hz"),
        }
        for name, entry in (sysfs.get("cpu_cpufreq") or {}).items()
    }
    return {
        "nvpmodel_mode_id": nvpmodel.get("mode_id"),
        "nvpmodel_mode_name": nvpmodel.get("mode_name"),
        "jetson_clocks_gpu": clocks.get("gpu"),
        "jetson_clocks_cpu": {
            name: {
                "governor": entry.get("governor"),
                "min_freq_hz": entry.get("min_freq_hz"),
                "max_freq_hz": entry.get("max_freq_hz"),
            }
            for name, entry in (clocks.get("cpufreq") or {}).items()
        },
        "jetson_clocks_emc": clocks.get("emc"),
        "sysfs_gpu": {
            "governor": (sysfs.get("gpu_devfreq") or {}).get("governor"),
            "min_freq_hz": (sysfs.get("gpu_devfreq") or {}).get("min_freq_hz"),
            "max_freq_hz": (sysfs.get("gpu_devfreq") or {}).get("max_freq_hz"),
        },
        "sysfs_cpu": cpu,
        "emc_cap_hz": (sysfs.get("nvpmodel_clk_cap") or {}).get("emc_hz"),
    }


# ── system mutation (single funnel) ────────────────────────────────────────


_JC_CONF_RE = re.compile(r"(/\S*jetsonclocks_conf\.txt)")


def store_clock_state() -> Dict[str, Any]:
    """``jetson_clocks --store``: snapshot the clock state for later restore.

    ``jetson_clocks --store`` refuses to overwrite an existing snapshot and
    instead asks ``Can I overwrite it? Y/N`` on stdin; in a non-interactive run
    that is an immediate ``returncode=1`` and **no** snapshot is taken.  The
    stale file is removed and the store retried once, so the snapshot that
    ``--restore`` later uses always describes this session's starting state.
    The retry is recorded, never hidden.
    """
    result = _run_command(["jetson_clocks", "--store"])
    result["retried_after_removing_stale_snapshot"] = False
    if result.get("returncode") == 0:
        return result

    combined = f"{result.get('stdout', '')}{result.get('stderr', '')}"
    match = _JC_CONF_RE.search(combined)
    if "already exists" not in combined or not match:
        return result

    stale = match.group(1)
    try:
        os.remove(stale)
    except OSError as exc:
        result["stale_snapshot_removal_error"] = f"{type(exc).__name__}: {exc}"
        return result

    retry = _run_command(["jetson_clocks", "--store"])
    retry["retried_after_removing_stale_snapshot"] = True
    retry["removed_snapshot"] = stale
    retry["first_attempt"] = {
        "returncode": result.get("returncode"),
        "stdout": result.get("stdout"),
    }
    return retry


def restore_clock_state() -> Dict[str, Any]:
    """``jetson_clocks --restore``: put the stored clock state back."""
    return _run_command(["jetson_clocks", "--restore"])


def lock_clocks() -> Dict[str, Any]:
    """``jetson_clocks``: pin every clock to its maximum (fixed policy)."""
    run = _run_command(["jetson_clocks"])
    run["observed_after"] = query_jetson_clocks_show()
    return run


def set_power_mode(mode_id: int) -> Dict[str, Any]:
    """``nvpmodel -m <id>`` and immediately re-query the observed state.

    A zero return code is *not* accepted as proof: the caller compares the
    observed mode/limits returned here against the requested ones.
    """
    run = _run_command(["nvpmodel", "-m", str(mode_id)])
    run["requested_mode_id"] = mode_id
    run["observed_after"] = query_nvpmodel()
    run["observed_mode_matches_request"] = (
        run["observed_after"].get("mode_id") == mode_id
    )
    return run


# ── sysfs telemetry sampler ────────────────────────────────────────────────


class SysfsSampler:
    """Background sampler for GPU/EMC/CPU frequency, thermals and cooling state.

    ``tegrastats`` on this board reports GPU *utilisation* but not GPU *frequency*,
    and it has no cooling-device state at all.  Those two series are exactly what
    the protocol needs to distinguish "frequency dropped because it is hot" from
    "frequency dropped because the power mode caps it", so they are sampled
    separately, each record stamped with the same ``time.monotonic_ns()`` clock
    that brackets the request windows.
    """

    def __init__(self, interval_ms: int = 250) -> None:
        self.interval_ms = interval_ms
        self.records: List[Dict[str, Any]] = []
        self.read_errors = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thermal_zones: List[tuple] = []
        self._cooling_devices: List[tuple] = []

    def _discover(self) -> None:
        self._thermal_zones = []
        for zone in sorted(glob.glob(_THERMAL_ZONE_GLOB)):
            name = read_text(f"{zone}/type") or os.path.basename(zone)
            self._thermal_zones.append((name, f"{zone}/temp"))
        self._cooling_devices = []
        for device in sorted(glob.glob(_COOLING_GLOB)):
            name = read_text(f"{device}/type") or os.path.basename(device)
            self._cooling_devices.append((name, f"{device}/cur_state"))

    def _sample(self) -> Dict[str, Any]:
        temperatures: Dict[str, Optional[float]] = {}
        for name, path in self._thermal_zones:
            millideg = read_int(path)
            temperatures[name] = (
                millideg / 1000.0 if millideg is not None else None
            )
        cooling: Dict[str, Optional[int]] = {}
        for name, path in self._cooling_devices:
            cooling[name] = read_int(path)

        cpu_cur = {}
        for directory in sorted(glob.glob(_CPU_FREQ_GLOB)):
            cpu = directory.split("/")[-2]
            cpu_cur[cpu] = {
                "cur_freq_hz": read_int(f"{directory}/scaling_cur_freq"),
                "governor": read_text(f"{directory}/scaling_governor"),
            }
        return {
            "time_ns": time.monotonic_ns(),
            "gpu_cur_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/cur_freq"),
            "gpu_target_freq_hz": read_int(f"{_GPU_DEVFREQ_DIR}/target_freq"),
            "gpu_load_pct": read_int(f"{_GPU_SYSFS_DIR}/load"),
            "emc_rate_hz": read_int(_EMC_RATE_PATH),
            "cpu": cpu_cur,
            "temperatures_c": temperatures,
            "cooling_cur_state": cooling,
            "fan_pwm": read_int(_FAN_PWM_PATH),
        }

    def _loop(self) -> None:
        interval_s = self.interval_ms / 1000.0
        next_at = time.monotonic()
        while not self._stop.is_set():
            try:
                record = self._sample()
            except Exception:  # noqa: BLE001 — a sampler must never kill the run
                with self._lock:
                    self.read_errors += 1
                record = None
            if record is not None:
                with self._lock:
                    self.records.append(record)
            next_at += interval_s
            sleep_for = next_at - time.monotonic()
            if sleep_for < 0:
                next_at = time.monotonic()
                sleep_for = 0
            self._stop.wait(sleep_for)

    def start(self) -> None:
        self._discover()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="e02-08-sysfs-sampler"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class TelemetrySuite:
    """``tegrastats`` + sysfs sampler, started and stopped together.

    The two sources are kept separate on disk (raw tegrastats text vs. structured
    JSON) because they have different failure modes: a tegrastats parse failure
    invalidates the energy integral, while a sysfs read failure only degrades the
    frequency/throttle evidence.
    """

    def __init__(self, interval_ms: int = 250) -> None:
        self.interval_ms = interval_ms
        self.tegrastats = TegrastatsMonitor(interval_ms=interval_ms)
        self.sysfs = SysfsSampler(interval_ms=interval_ms)
        self.tegrastats_available = False
        self.tegrastats_error: Optional[str] = None

    def start(self) -> None:
        try:
            self.tegrastats.start()
            self.tegrastats_available = True
        except Exception as exc:  # noqa: BLE001 — non-Jetson / permissions
            self.tegrastats_available = False
            self.tegrastats_error = f"{type(exc).__name__}: {exc}"
            logger.warning("tegrastats unavailable: %s", exc)
        self.sysfs.start()

    def stop(self) -> None:
        if self.tegrastats_available:
            self.tegrastats.stop()
        self.sysfs.stop()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tegrastats_available": self.tegrastats_available,
            "tegrastats_error": self.tegrastats_error,
            "interval_ms": self.interval_ms,
            "tegrastats_records": len(self.tegrastats.records),
            "sysfs_records": len(self.sysfs.records),
            "sysfs_read_errors": self.sysfs.read_errors,
        }


# ── window execution ───────────────────────────────────────────────────────


def run_idle_window(duration_s: float, device: str = "cuda") -> Dict[str, Any]:
    """Fixed-duration idle window with the model still resident.

    An idle reference has to state whether the weights were loaded, otherwise a
    "no model" idle could be subtracted from an "model resident" active window
    (protocol step 4).  The model stays loaded for the whole session, so this
    window is the ``model_loaded`` reference.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    begin_ns = time.monotonic_ns()
    target_ns = begin_ns + int(duration_s * 1e9)
    while time.monotonic_ns() < target_ns:
        remaining = (target_ns - time.monotonic_ns()) / 1e9
        time.sleep(min(0.5, max(0.0, remaining)))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_ns = time.monotonic_ns()
    return {
        "kind": "idle",
        "model_loaded": True,
        "requested_duration_s": duration_s,
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "actual_duration_s": (end_ns - begin_ns) / 1e9,
        "num_requests": 0,
    }


def run_steady_window(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    num_requests: int,
    *,
    device: str = "cuda",
    workload_name: str = "",
) -> Dict[str, Any]:
    """Execute ``num_requests`` back-to-back identical requests as one window.

    Every request goes through :func:`benchmark_model_core`, which performs its
    own prefill and therefore starts from an empty KV cache — the "KV reset"
    requirement is met by construction and the token hash of each request is
    recorded so a request cannot silently misfire.

    A single request shorter than the telemetry interval would be unmeasurable
    (protocol section 5); chaining N of them builds a steady measurement window
    whose energy is integrated once and then normalised by the *actual* number
    of completed requests and tokens.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    window_begin_ns = time.monotonic_ns()
    requests: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for index in range(num_requests):
        request_begin_ns = time.monotonic_ns()
        try:
            result = benchmark_model_core(model, inputs, output_tokens)
        except Exception as exc:  # noqa: BLE001 — OOM etc. must be recorded
            failures.append(
                {
                    "request_index": index,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.warning(
                "%s request %d/%d failed: %s", workload_name, index, num_requests, exc
            )
            continue
        request_end_ns = time.monotonic_ns()
        requests.append(
            {
                "request_index": index,
                "begin_ns": request_begin_ns,
                "end_ns": request_end_ns,
                "wall_ms": (request_end_ns - request_begin_ns) / 1e6,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "prefill_forward_ms": result["prefill_forward_ms"],
                "first_token_selection_ms": result["first_token_selection_ms"],
                "model_core_ttft_ms": result["model_core_ttft_ms"],
                "decode_total_ms": result["decode_total_ms"],
                "model_core_e2e_ms": result["model_core_e2e_ms"],
                "prefill_tokens_per_s": result["prefill_tokens_per_s"],
                "decode_tokens_per_s": result["decode_tokens_per_s"],
                "model_core_output_tokens_per_s": result[
                    "model_core_output_tokens_per_s"
                ],
                "raw_itl_ms": result["raw_itl_ms"],
                "sequence_sha256": hash_token_sequence(
                    result["generated_token_ids"]
                ),
                "kv_cache_total_bytes": result["kv_cache"]["total_bytes"],
                "peak_cuda_allocated_mb": result["peak_cuda_allocated_mb"],
                "peak_cuda_reserved_mb": result["peak_cuda_reserved_mb"],
                "process_rss_bytes": result["process_rss_bytes"],
                "process_swap_bytes": result["process_swap_bytes"],
            }
        )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    window_end_ns = time.monotonic_ns()

    return {
        "kind": "steady",
        "workload_name": workload_name,
        "model_loaded": True,
        "requested_requests": num_requests,
        "valid_requests": len(requests),
        "failures": failures,
        "begin_ns": window_begin_ns,
        "end_ns": window_end_ns,
        "window_s": (window_end_ns - window_begin_ns) / 1e9,
        "input_tokens_total": sum(r["input_tokens"] for r in requests),
        "output_tokens_total": sum(r["output_tokens"] for r in requests),
        "processed_tokens_total": sum(
            r["input_tokens"] + r["output_tokens"] for r in requests
        ),
        "sequence_sha256_set": sorted({r["sequence_sha256"] for r in requests}),
        "requests": requests,
    }


# ── cooldown ───────────────────────────────────────────────────────────────


def wait_for_temperature(
    target_c: float,
    *,
    zone: str = "tj-thermal",
    timeout_s: float = 900.0,
    poll_s: float = 2.0,
) -> Dict[str, Any]:
    """Block until one thermal zone is at or below ``target_c``.

    Returns the full polled series so a timeout is visible as evidence rather
    than silently turning into "it was fine".
    """
    series: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    reached = False
    while True:
        path = None
        for candidate in sorted(glob.glob(_THERMAL_ZONE_GLOB)):
            if (read_text(f"{candidate}/type") or "") == zone:
                path = candidate
                break
        millideg = read_int(f"{path}/temp") if path else None
        current = millideg / 1000.0 if millideg is not None else None
        series.append(
            {
                "time_ns": time.monotonic_ns(),
                "temp_c": current,
            }
        )
        if current is not None and current <= target_c:
            reached = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_s)

    return {
        "zone": zone,
        "target_c": target_c,
        "reached": reached,
        "timeout_s": timeout_s,
        "num_polls": len(series),
        "start_c": series[0]["temp_c"] if series else None,
        "end_c": series[-1]["temp_c"] if series else None,
        "series": series,
    }


# ── telemetry post-processing ──────────────────────────────────────────────


def slice_sysfs_records(
    records: Sequence[Dict[str, Any]],
    begin_ns: int,
    end_ns: int,
) -> List[Dict[str, Any]]:
    return [
        r
        for r in records
        if isinstance(r.get("time_ns"), int) and begin_ns <= r["time_ns"] <= end_ns
    ]


def _mean_of(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def attach_request_telemetry(
    requests: Sequence[Dict[str, Any]],
    sysfs_records: Sequence[Dict[str, Any]],
    *,
    hot_zone: str = "tj-thermal",
    temp_zone: str = "tj-thermal",
) -> List[Dict[str, Any]]:
    """Attach per-request observed temperature and GPU frequency.

    This is what makes the thermal verdict possible at the *request* level: the
    protocol requires temperature, observed frequency and latency to be aligned
    per request, not averaged over a whole run.
    """
    enriched: List[Dict[str, Any]] = []
    for request in requests:
        window = slice_sysfs_records(
            sysfs_records, request["begin_ns"], request["end_ns"]
        )
        temps = [
            (r.get("temperatures_c") or {}).get(temp_zone) for r in window
        ]
        freqs = [r.get("gpu_cur_freq_hz") for r in window]
        loads = [r.get("gpu_load_pct") for r in window]
        item = dict(request)
        item["num_sysfs_samples"] = len(window)
        item["temp_c"] = _mean_of(temps)
        item["temp_peak_c"] = max(
            [t for t in temps if t is not None], default=None
        )
        item["hot_zone"] = hot_zone
        item["gpu_freq_hz"] = _mean_of(freqs)
        item["gpu_freq_min_hz"] = min(
            [f for f in freqs if f is not None], default=None
        )
        item["gpu_load_pct"] = _mean_of(loads)
        enriched.append(item)
    return enriched


__all__ = [
    "POWER_THERMAL_PROTOCOL",
    "SysfsSampler",
    "TelemetrySuite",
    "attach_request_telemetry",
    "configured_state_view",
    "lock_clocks",
    "query_jetson_clocks_show",
    "query_nvpmodel",
    "read_device_state",
    "read_sysfs_state",
    "restore_clock_state",
    "run_idle_window",
    "run_steady_window",
    "set_power_mode",
    "slice_sysfs_records",
    "store_clock_state",
    "wait_for_temperature",
]
