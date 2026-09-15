"""Private UST sessions and verified aggregate address-space ceilings.

The consumer daemon reparents itself. Register it by its private IPC path,
including forked run-as helpers, rather than assuming PPID traversal is enough.
"""

import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import uuid
import json


def bounded_command(argv, mib, cpu):
    return ["taskset", "-c", str(cpu), "prlimit", f"--as={mib*1024**2}",
            "--stack=2097152", "--", *argv]


def cli(args, env, directory, label, cpu):
    with (directory / (label + ".log")).open("w") as log:
        return subprocess.run(bounded_command(["lttng", *args], 384, cpu), env=env,
                              stdout=log, stderr=subprocess.STDOUT, timeout=30, check=True)


def descendants(roots):
    found = set(roots)
    frontier = list(found)
    while frontier:
        pid = frontier.pop()
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def stop(process):
    """Only terminate the session and consumer registered by this launch."""
    for pid in getattr(process, "hqsb_consumer_roots", []):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def start(directory, env, cpu, profile):
    # An optional isolated 2.16 prefix does not replace the distribution tools.
    isolated = env.get("HQSB_LTTNG_PREFIX")
    prefix = Path(isolated) if isolated else Path("/usr")
    home = Path("/tmp") / ("hqsf-" + uuid.uuid4().hex[:12])
    home.mkdir()
    env.update(PATH=str(prefix / "bin") + ":" + env["PATH"], LTTNG_HOME=str(home),
               LTTNG_RUNDIR=str(home), LTTNG_UST_CTL_PATH=str(home),
               LTTNG_UST_APP_PATH=str(home), LTTNG_UST_REGISTER_TIMEOUT="10000")
    env.pop("LTTNG_DEBUG_NOCLONE", None)
    if isolated:
        env["LD_LIBRARY_PATH"] = str(prefix / "lib")
    (directory / "lttng-rundir.txt").write_text(str(home))
    consumers = list(prefix.glob("lib/*-linux-gnu/lttng/libexec/lttng-consumerd"))
    # The isolated build installs directly in lib/lttng/libexec.
    direct = prefix / "lib/lttng/libexec/lttng-consumerd"
    consumers = sorted(set(consumers + ([direct] if direct.is_file() else [])))
    if len(consumers) != 1:
        raise RuntimeError(f"expected one consumer binary under {prefix}: {consumers}")
    wrapper = directory / "consumer-wrapper.sh"
    wrapper.write_text("#!/bin/sh\nset -e\nulimit -v 262144\nulimit -s 2048\nexec "
                       + str(consumers[0]) + ' "$@"\n')
    wrapper.chmod(0o700)
    with (directory / "sessiond.log").open("w") as log:
        process = subprocess.Popen(bounded_command(
            [str(prefix / "bin/lttng-sessiond"), "--no-kernel", "--consumerd64-path=" + str(wrapper)],
            256, cpu), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(250):
            if list(home.rglob("client-lttng-sessiond")):
                break
            if process.poll() is not None:
                raise RuntimeError("private sessiond exited; inspect sessiond.log")
            time.sleep(.02)
        else:
            raise TimeoutError("private sessiond IPC was not ready")
        env["LTTNG_NO_SESSIOND"] = "1"
        cli(["create", "hqsb", *(["--trace-format=ctf-1.8"] if isolated else []),
             "--output=" + str(directory / "ctf")], env, directory, "create", cpu)
        size = ((8192, 1048576, 4194304) if isolated else (8192, 32768, 131072))[profile]
        allocation = "per-cpu" if profile == 0 else "per-channel"
        ownership = (["--buffer-ownership=process", "--buffer-allocation=" + allocation,
                      "--buffer-preallocation=on-demand"] if isolated else ["--buffers-pid"])
        cli(["enable-channel", "--userspace", *ownership, "--subbuf-size=" + str(size),
             "--num-subbuf=4", "--discard", "capture"], env, directory, "channel", cpu)
        cli(["enable-event", "--userspace", "--channel=capture", "hqsb_capture:event"],
            env, directory, "enable", cpu)
        cli(["start"], env, directory, "start", cpu)
        roots = set()
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                args = (path / "cmdline").read_bytes().split(b"\0")
                comm = (path / "comm").read_text().strip()
            except OSError:
                continue
            if (args and Path(args[0].decode(errors="replace")).name == "lttng-consumerd"
                    and any(str(home).encode() in arg for arg in args) and "runas" not in comm):
                roots.add(int(path.name))
        process.hqsb_consumer_roots = sorted(roots)
        if len(roots) != 1:
            raise RuntimeError(f"expected one owned consumer, found {roots}")
        ids = descendants({process.pid} | roots)
        limits = {}
        helpers = 0
        for pid in ids:
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            helper = "runas" in comm
            ceiling = (64 if helper else 256) * 1024**2
            if helper:
                status = Path(f"/proc/{pid}/status").read_text()
                size = int(next(s for s in status.splitlines() if s.startswith("VmSize:")).split()[1]) * 1024
                if size > ceiling:
                    raise RuntimeError(f"run-as helper {pid} exceeds declared ceiling")
                resource.prlimit(pid, resource.RLIMIT_AS, (ceiling, ceiling))
                helpers += 1
            if resource.prlimit(pid, resource.RLIMIT_AS) != (ceiling, ceiling):
                raise RuntimeError(f"unexpected actual AS limit for {pid}")
            limits[str(pid)] = dict(AS_bytes=ceiling, comm=comm)
        if helpers > 2 or len(ids) - helpers != 2:
            raise RuntimeError(f"unexpected session process set: {limits}")
        if 384 * 1024**2 + sum(v["AS_bytes"] for v in limits.values()) > 1024**3:
            raise RuntimeError("aggregate AS ceiling exceeds 1GiB")
        process.hqsb_detached_pids = sorted(ids - descendants({process.pid}))
        (directory / "verified-daemon-limits.json").write_text(json.dumps(limits, indent=2))
    except BaseException:
        stop(process)
        raise
    return process
