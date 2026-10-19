"""L1 independent rollback guardian. API death must not kill this process.

The guardian owns the device lease, durable power journal and worker process.
On cancellation/parent exit it first kills the worker, then restores and reads
back the original cap. SIGKILL/host failure leaves a durable quarantine journal.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from hqsb.core.system_execution import SystemPlan
from hqsb.hardware.system import Nvidia
from hqsb.observation.storage import atomic_json
from hqsb.backends.process_supervision import (
    mapped_visibility,
    terminate,
    parent_identity,
)


def main(path):
    source = Path(path)
    job = json.loads(source.read_text())
    plan = SystemPlan.model_validate(job["plan"])
    parent, original_parent = job["supervisor_pid"], job["supervisor_start_ticks"]
    directory = source.parent
    journal = Path(job["journal"])
    base = Path(
        job["config"].get("device_lock_path")
        or str(Path.home() / ".cache/hqsb/device0.lock")
    )
    base.parent.mkdir(parents=True, exist_ok=True)
    lock = (
        base if plan.device == 0 else base.with_name(base.name + f".gpu{plan.device}")
    ).open("a")
    process, nvml, original, changed = None, None, None, False
    result = {
        "status": "failed",
        "cleanup": "pending",
        "trials": [],
        "power_before_w": None,
    }

    def aborted():
        return (directory / "cancel").exists() or parent_identity(
            parent
        ) != original_parent

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if journal.exists():
            raise RuntimeError("L1_UNRESOLVED_POWER_JOURNAL")
        uses_power = any(
            p.power_limit_w is not None for p in (plan.baseline, plan.candidate)
        )
        try:
            nvml = Nvidia()
            original = nvml.power(job["uuid"])
        except Exception:
            if uses_power:
                raise
        result["power_before_w"] = original
        started = time.monotonic()
        for i in range(plan.rounds * 2):
            # Alternating AB/BA blocks reduce systematic time/order drift.
            label = ("baseline", "candidate", "candidate", "baseline")[i % 4]
            profile = getattr(plan, label)
            if aborted():
                result["status"] = "cancelled"
                break
            if uses_power:
                if nvml.processes(job["uuid"]):
                    raise RuntimeError("L1_EXTERNAL_GPU_PROCESS_POWER_CHANGE_REJECTED")
                # Write before attempting mutation, including errors whose
                # driver side effects cannot be inferred from their return.
                atomic_json(
                    journal,
                    {
                        "run_id": job["run_id"],
                        "uuid": job["uuid"],
                        "original_power_w": original,
                        "state": "restore_required",
                        "guard_pid": os.getpid(),
                    },
                )
                changed = True
                requested = (
                    profile.power_limit_w
                    if profile.power_limit_w is not None
                    else original
                )
                try:
                    effective = nvml.set_power(job["uuid"], requested)
                except Exception as exc:
                    raise RuntimeError(
                        f"L1_POWER_SET_FAILED(uuid={job['uuid']}, requested_w={requested}): {exc}"
                    ) from exc
            else:
                effective = original
            atomic_json(
                directory / "progress.json",
                {"trial": i, "label": label, "phase": "loading_or_measuring"},
            )
            trial_input = directory / "trial-input.json"
            atomic_json(
                trial_input,
                {
                    **job,
                    "profile": profile.model_dump(),
                    "label": label,
                    "effective_power_w": effective,
                },
            )
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": mapped_visibility([plan.device]),
            }
            with (directory / f"trial-{i}.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "hqsb.backends.system_worker",
                        str(trial_input),
                        str(i),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
                while process.poll() is None:
                    if aborted() or time.monotonic() - started > plan.timeout_s:
                        result["status"] = "cancelled" if aborted() else "timed_out"
                        terminate(process)
                        break
                    time.sleep(0.15)
            if result["status"] in {"cancelled", "timed_out"}:
                break
            if process.returncode:
                with (directory / f"trial-{i}.log").open("rb") as f:
                    f.seek(max(0, (directory / f"trial-{i}.log").stat().st_size - 2200))
                    raise RuntimeError(f.read().decode(errors="replace"))
            doc = json.loads((directory / f"trial-{i}.json").read_text())
            if (
                doc["run_id"] != job["run_id"]
                or doc["trial"] != i
                or doc["status"] != "completed"
            ):
                raise RuntimeError("L1_TRIAL_IDENTITY_MISMATCH")
            result["trials"].append(doc)
        else:
            result["status"] = "completed"
    except Exception as exc:
        result["error"] = str(exc)[:2600]
    finally:
        try:
            terminate(process)
            if changed:
                result["power_restored_w"] = nvml.set_power(job["uuid"], original)
                journal.unlink()
            result["cleanup"] = "succeeded"
        except Exception as exc:
            result.update(
                status="failed",
                cleanup="failed",
                error="L1_RESTORE_OR_PROCESS_CLEANUP_FAILED: " + str(exc),
            )
        finally:
            if nvml:
                try:
                    nvml.close()
                except Exception as exc:
                    result["nvml_shutdown_error"] = str(exc)
            atomic_json(directory / "guard-result.json", result)
            lock.close()


if __name__ == "__main__":
    main(sys.argv[1])
