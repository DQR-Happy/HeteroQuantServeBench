"""Node ownership and real HTTP relay contracts; no GPU evidence is implied."""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.machine_registry import MachineConfig, MachineError, MachineRegistry
from hqsb.console.machine_transport import MachineConnections, ssh_arguments


@pytest.fixture
def upstream():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append((self.path, dict(self.headers)))
            if self.path == "/healthz":
                value = {"status": "ok", "api_version": "1", "node_id": "remote-node"}
            elif self.path == PREFIX + "/session":
                value = {
                    "api_version": "1",
                    "features": ["inference"],
                    "auth_required": False,
                }
            elif self.path == PREFIX + "/runs/original/export":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(
                    "Content-Disposition", 'attachment; filename="original.bin"'
                )
                self.end_headers()
                self.wfile.write(b"\x00\xfforiginal-evidence")
                return
            elif self.path.startswith(PREFIX + "/requests/original/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'id: 2\ndata: {"owner":"remote"}\n\n')
                return
            else:
                value = {"owner": "remote", "path": self.path}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_POST(self):
            calls.append(
                (
                    self.path,
                    dict(self.headers),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id":"remote-request","owner":"remote"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()
    server.server_close()
    thread.join()


def config(url, **changes):
    return MachineConfig(
        id="remote", name="Remote", transport="http", base_url=url, **changes
    )


@pytest.fixture
def app(tmp_path, upstream):
    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        machines=[config(upstream[0])],
        deployments=[DeploymentConfig(id="reference", name="Local")],
    )
    return create_app(settings, monitor=False)


def test_persistent_profiles_never_store_token_values(tmp_path, monkeypatch):
    monkeypatch.setenv("NODE_TOKEN", "private-secret")
    registry = MachineRegistry(tmp_path, "Local", [])
    registry.add(config("https://example.org", token_env="NODE_TOKEN"))
    restored = MachineRegistry(tmp_path, "Local", [])
    assert restored.node_id == registry.node_id
    assert restored.get("remote").token_env == "NODE_TOKEN"
    assert "private-secret" not in registry.path.read_text()
    assert registry.path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(MachineError, match="new machine ID"):
        restored.add(config("https://example.org"))
    restored.remove("remote")
    assert MachineRegistry(tmp_path, "Local", []).public("local")["items"] == [
        {"id": "local", "name": "Local", "transport": "local"}
    ]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:secret@host",
        "https://host/api",
        "https://host?token=secret",
        "http://host:99999",
        "http://host\n",
        "http://ho st",
    ],
)
def test_rejects_non_origins_and_credentials(url):
    with pytest.raises(ValidationError):
        config(url)


def test_removed_ids_cannot_retarget_old_tabs_even_after_restart(tmp_path):
    registry = MachineRegistry(tmp_path, "Local", [config("http://first.example")])
    registry.remove("remote")
    restored = MachineRegistry(tmp_path, "Local", [])
    with pytest.raises(MachineError, match="cannot be reused"):
        restored.add(config("http://different.example"))
    with pytest.raises(MachineError, match="not found"):
        restored.get("remote")


def test_relay_routes_posts_streams_and_downloads_without_changing_local_owner(
    app, upstream
):
    with TestClient(app) as client:
        headers = {"X-HQSB-Client": "console", "Idempotency-Key": "original-key"}
        assert (
            client.post(
                PREFIX + "/machines/remote/probe", json={}, headers=headers
            ).json()["status"]
            == "ready"
        )
        assert (
            client.post(
                PREFIX + "/machines/remote/activate", json={}, headers=headers
            ).status_code
            == 200
        )
        # Activation does not globally redirect other tabs' API requests.
        assert (
            client.get(PREFIX + "/deployments").json()["items"][0]["id"] == "reference"
        )
        remote = PREFIX + "/nodes/remote"
        assert client.get(remote + "/deployments?cursor=3").json() == {
            "owner": "remote",
            "path": PREFIX + "/deployments?cursor=3",
        }
        posted = client.post(
            remote + "/requests", json={"input": "original"}, headers=headers
        )
        assert posted.status_code == 202 and posted.json()["id"] == "remote-request"
        assert app.state.service.store.list()["total"] == 0
        download = client.get(remote + "/runs/original/export")
        assert download.content == b"\x00\xfforiginal-evidence"
        assert (
            download.headers["content-disposition"]
            == 'attachment; filename="original.bin"'
        )
        stream = client.get(remote + "/requests/original/events?after=1")
        assert stream.content == b'id: 2\ndata: {"owner":"remote"}\n\n'
        assert stream.headers["content-type"] == "text/event-stream"
        post_calls = [row for row in upstream[1] if len(row) == 3]
        assert (
            len(post_calls) == 1
            and {key.lower(): value for key, value in post_calls[0][1].items()}[
                "idempotency-key"
            ]
            == "original-key"
        )
        assert json.loads(post_calls[0][2]) == {"input": "original"}


def test_management_remains_local_and_origin_guard_still_applies(app):
    with TestClient(app) as client:
        bad = client.post(
            PREFIX + "/nodes/remote/requests",
            json={},
            headers={"X-HQSB-Client": "console", "Origin": "https://other.invalid"},
        )
        assert bad.status_code == 403
        assert (
            client.post(PREFIX + "/nodes/remote/requests", json={}).status_code == 403
        )
        assert client.get(PREFIX + "/nodes/absent/runs").status_code == 404
        loop = client.get(
            PREFIX + "/nodes/remote/runs", headers={"X-HQSB-Relay": "previous"}
        )
        assert loop.status_code == 409
        assert client.get(PREFIX + "/machines").json()["items"][0]["id"] == "local"


def test_no_fallback_or_retry_on_connection_failure(tmp_path):
    registry = MachineRegistry(tmp_path, "Local", [config("http://127.0.0.1:1")])
    transport = MachineConnections(registry)
    with pytest.raises(MachineError, match="probe failed"):
        transport.probe(registry.get("remote"))
    transport.close()


def test_failed_remote_submission_never_enters_local_queue(app):
    app.state.machines.add(
        MachineConfig(
            id="offline",
            name="Offline",
            transport="http",
            base_url="http://127.0.0.1:1",
        )
    )
    with TestClient(app) as client:
        response = client.post(
            PREFIX + "/nodes/offline/requests",
            json={},
            headers={"X-HQSB-Client": "console"},
        )
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "MACHINE_UNREACHABLE"
        assert app.state.service.store.list()["total"] == 0


def test_relay_uses_only_target_credentials(app, upstream, monkeypatch):
    monkeypatch.setenv("TARGET_TOKEN", "target-secret")
    app.state.machines.add(
        MachineConfig(
            id="protected",
            name="Protected",
            transport="http",
            base_url=upstream[0],
            token_env="TARGET_TOKEN",
        )
    )
    with TestClient(app) as client:
        response = client.get(
            PREFIX + "/nodes/protected/deployments",
            headers={
                "Authorization": "Bearer control-secret",
                "Cookie": "hqsb_session=private-control-session",
            },
        )
        assert response.status_code == 200
    headers = {key.lower(): value for key, value in upstream[1][-1][1].items()}
    assert headers["authorization"] == "Bearer target-secret"
    assert "cookie" not in headers
    assert "control-secret" not in str(headers)


def test_ssh_alias_resolution_does_not_inherit_extra_forwards(monkeypatch):
    def resolved(argv, **kwargs):
        assert argv == ["ssh", "-G", "--", "node-alias"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="hostname node.example\nport 2222\nuser operator\nidentityfile ~/.ssh/node_key\nremoteforward 7890 localhost:7890\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", resolved)
    args = ssh_arguments(
        MachineConfig(id="ssh", name="SSH", transport="ssh", ssh_host="node-alias"),
        12345,
    )
    assert args[:3] == ["ssh", "-F", "/dev/null"]
    assert "127.0.0.1:12345:127.0.0.1:8768" in args
    assert "StrictHostKeyChecking=yes" in args
    assert "7890" not in " ".join(args)
    assert args[-1] == "node.example"
