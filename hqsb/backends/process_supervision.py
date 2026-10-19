"""Private Linux process-tree lifetime and CUDA visibility, shared by L1/L9."""

import os
from pathlib import Path
import signal
import subprocess


def parent_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
    except OSError:
        return None


def mapped_visibility(devices):
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = inherited.split(",") if inherited is not None else None
    return ",".join(visible[i] if visible is not None else str(i) for i in devices)


def terminate(process):
    """Reap a private session and known descendants, including orphan ranks."""
    if process is None:
        return
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError("L9_RANK_CLEANUP_NOT_CONFIRMED")
