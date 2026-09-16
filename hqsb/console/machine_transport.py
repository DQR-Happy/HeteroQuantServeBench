"""Owned SSH tunnels and bounded HTTP streaming to other HQSB consoles."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from hqsb.console.machine_registry import MachineConfig, MachineError, MachineRegistry

PREFIX = "/api/console/v1"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def ssh_arguments(config: MachineConfig, port: int) -> list[str]:
    # No shell or remote command. SSH config aliases, keys and agent belong to
    # the control server, not to the browser or the editing workstation.
    # Resolve aliases, then use an isolated config: ClearAllForwardings would
    # clear our own -L too, while inheriting config can open unrelated forwards.
    resolve = ["ssh", "-G"]
    if config.ssh_port:
        resolve += ["-p", str(config.ssh_port)]
    if config.ssh_user:
        resolve += ["-l", config.ssh_user]
    resolve += ["--", config.ssh_host]
    try:
        output = subprocess.run(
            resolve, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MachineError(
            "SSH_CONFIG_INVALID", "Cannot resolve server-side SSH configuration", 502
        ) from exc
    options: dict[str, list[str]] = {}
    for line in output.stdout.splitlines():
        key, _, value = line.partition(" ")
        options.setdefault(key, []).append(value)
    args = [
        "ssh",
        "-F",
        "/dev/null",
        "-N",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
    ]
    args += ["-L", f"127.0.0.1:{port}:127.0.0.1:{config.remote_port}"]
    args += ["-p", options["port"][0], "-l", options["user"][0]]
    keys = (
        [config.identity_file]
        if config.identity_file
        else options.get("identityfile", [])
    )
    for key in keys:
        args += ["-i", str(Path(key).expanduser())]
    if config.identity_file:
        args += ["-o", "IdentitiesOnly=yes"]
    # Preserve explicit aliases/proxies without executing browser-provided code.
    for key in ("proxyjump", "hostkeyalias", "identityagent", "userknownhostsfile"):
        if key in options and options[key][0] not in {"none", "SSH_AUTH_SOCK"}:
            args += ["-o", f"{key}={options[key][0]}"]
    return [*args, "--", options["hostname"][0]]


class MachineConnections:
    def __init__(self, registry: MachineRegistry):
        self.registry = registry
        # A removed/re-added ID must never reuse a tunnel to its former host.
        self.tunnels: dict[MachineConfig, tuple[subprocess.Popen, int]] = {}
        self.lock = threading.Lock()

    def origin(self, config: MachineConfig) -> str:
        if config.transport == "http":
            return config.base_url.rstrip("/")
        with self.lock:
            existing = self.tunnels.get(config)
            if existing and existing[0].poll() is None:
                return f"http://127.0.0.1:{existing[1]}"
            if existing:
                existing[0].wait()
                self.tunnels.pop(config, None)
            if len(self.tunnels) >= 64:
                raise MachineError(
                    "SSH_TUNNEL_LIMIT",
                    "Tunnel history limit reached; restart the control service after active streams finish",
                    409,
                )
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            try:
                process = subprocess.Popen(
                    ssh_arguments(config, port),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise MachineError(
                    "SSH_UNAVAILABLE", "Cannot start the SSH client", 502
                ) from exc
            self.tunnels[config] = process, port
            until = time.monotonic() + 10
            while time.monotonic() < until:
                if process.poll() is not None:
                    process.communicate()
                    self.tunnels.pop(config, None)
                    raise MachineError(
                        "SSH_CONNECT_FAILED",
                        "SSH failed: check host key, credentials, port and server-side SSH config",
                        502,
                    )
                with socket.socket() as probe:
                    probe.settimeout(0.1)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        return f"http://127.0.0.1:{port}"
                time.sleep(0.05)
            process.terminate()
            process.communicate(timeout=3)
            self.tunnels.pop(config, None)
            raise MachineError(
                "SSH_CONNECT_TIMEOUT", "SSH tunnel did not become ready", 504
            )

    def headers(self, config: MachineConfig) -> dict[str, str]:
        headers = {"X-HQSB-Client": "console", "X-HQSB-Relay": self.registry.node_id}
        if config.token_env:
            token = os.environ.get(config.token_env)
            if not token:
                raise MachineError(
                    "MACHINE_TOKEN_MISSING",
                    "Configured token environment variable is unset",
                    502,
                )
            headers["Authorization"] = "Bearer " + token
        return headers

    def probe(self, config: MachineConfig) -> dict:
        origin = self.origin(config)
        try:
            with httpx.Client(
                base_url=origin,
                headers=self.headers(config),
                timeout=10,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                health = client.get("/healthz")
                health.raise_for_status()
                data = health.json()
                session = client.get(PREFIX + "/session")
                session.raise_for_status()
                capabilities = session.json()
                if (
                    data.get("status") != "ok"
                    or data.get("api_version") != "1"
                    or capabilities.get("api_version") != "1"
                    or "inference" not in capabilities.get("features", [])
                ):
                    raise ValueError("Not a compatible HQSB service")
                if data.get("node_id") == self.registry.node_id:
                    raise MachineError(
                        "MACHINE_SELF_REFERENCE",
                        "This address points back to the control service; choose the local node",
                        409,
                    )
                return {
                    "status": "ready",
                    "machine_id": config.id,
                    "version": data.get("version"),
                    "node_id": data.get("node_id"),
                    "auth_required": capabilities.get("auth_required"),
                    "checked_at": time.time(),
                }
        except (httpx.HTTPError, ValueError) as exc:
            raise MachineError(
                "MACHINE_UNREACHABLE",
                "HQSB service probe failed; check service URL, remote port, authentication and API version",
                502,
            ) from exc

    async def forward(self, identity: str, request, path: str):
        config = self.registry.get(identity)
        origin = await asyncio.to_thread(self.origin, config)
        headers = self.headers(config)
        for key in ("content-type", "idempotency-key", "last-event-id", "range"):
            if key in request.headers:
                headers[key] = request.headers[key]
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(35, connect=8),
            trust_env=False,
            follow_redirects=False,
        )
        url = origin + path
        if request.url.query:
            url += "?" + request.url.query
        try:
            upstream = await client.send(
                client.build_request(
                    request.method, url, headers=headers, content=await request.body()
                ),
                stream=True,
            )
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse(
                {
                    "error": {
                        "code": "MACHINE_UNREACHABLE",
                        "message": "Selected machine is unavailable; no request was retried or sent to another machine",
                    }
                },
                502,
            )
        except BaseException:
            await client.aclose()
            raise

        async def close():
            await upstream.aclose()
            await client.aclose()

        # Downloads/traces and SSE retain exact bytes and response metadata.
        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in HOP_HEADERS | {"set-cookie"}
        }
        response_headers["X-HQSB-Machine"] = identity
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(close),
        )

    def close(self):
        with self.lock:
            for process, _port in self.tunnels.values():
                if process.poll() is None:
                    process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
            self.tunnels.clear()
