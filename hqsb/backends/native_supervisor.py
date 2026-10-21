"""Stop an upstream process group when its owning Actor's pipe closes.

The pipe survives normal HTTP requests. EOF also handles an Actor killed before
its finally block. The upstream group owns all engine/rank subprocesses.
"""
import os
import select
import signal
import subprocess
import sys
import time


def main():
    import ctypes
    # Adopt orphaned rank/compiler children so shutdown can reap them as well.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Native supervisor requires Linux child-subreaper support")
    child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while child.poll() is None:
            if select.select([sys.stdin.buffer], [], [], 0.25)[0]:
                if not os.read(sys.stdin.fileno(), 1):
                    break
        return child.poll() or 0
    finally:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            pass
        # A server can exit before its rank/compiler children. Reap the entire
        # group before the inherited device lease is released by this supervisor.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.05)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        while True:
            try:
                os.waitpid(-1, 0)
            except ChildProcessError:
                break


if __name__ == "__main__":
    raise SystemExit(main())
