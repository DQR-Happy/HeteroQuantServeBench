import json

from hqsb.observation import tools


def executable(tmp_path, body):
    path = tmp_path / "native-tool-test-double"
    path.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo test-double; exit 0; fi\n'
        + body
    )
    path.chmod(0o700)
    return str(path)


def test_missing_tool_is_retained_without_invented_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    report = tools.capture("nsys", ["ignored"], tmp_path / "missing")
    assert report["availability"] == "tool_unavailable"
    assert report["artifacts"] == []
    assert (
        json.loads((tmp_path / "missing/tool-manifest.json").read_text())[
            "availability"
        ]
        == "tool_unavailable"
    )


def test_zero_exit_without_native_artifact_is_partial(tmp_path, monkeypatch):
    tool = executable(tmp_path, "echo no-native-artifact\n")
    monkeypatch.setattr(tools.shutil, "which", lambda _: tool)
    report = tools.capture("nsys", ["ignored"], tmp_path / "empty")
    assert (
        report["availability"] == "partial"
        and report["reason"] == "NO_NATIVE_CAPTURE_ARTIFACT"
    )
    assert report["tool_version"].strip() == "test-double"
    assert report["artifacts"][0]["sha256"]


def test_timeout_terminates_tool_group_and_retains_failure(tmp_path, monkeypatch):
    tool = executable(tmp_path, "sleep 30\n")
    monkeypatch.setattr(tools.shutil, "which", lambda _: tool)
    report = tools.capture("nsys", ["ignored"], tmp_path / "timeout", timeout_s=0.05)
    assert report["availability"] == "partial" and report["reason"] == "TOOL_TIMEOUT"
    assert report["returncode"] != 0
    assert report["seconds"] < 10
