"""All stage writers must share persistence semantics and stay inside a run."""

import importlib
import json
from pathlib import Path

import pytest

from hqsb.core.errors import ConfigError

pytestmark = pytest.mark.integration
STAGES = (
    "quant",
    "integration",
    "runtime",
    "serving",
    "distributed",
    "compiler",
    "evaluation",
    "infra",
)


@pytest.mark.parametrize("area", STAGES)
def test_stage_command_and_json_roundtrip(area, tmp_path):
    module = importlib.import_module(f"hqsb.{area}.experiment")
    experiment = next(iter(module.EXPERIMENTS))
    run = module.RunDirectory(str(tmp_path), experiment, "roundtrip")
    run.create()
    record = run.record_command(
        1,
        ["/usr/bin/python3", "../driver.py", "--help"],
        stdout="output",
        stderr="diagnostic",
        returncode=2,
    )
    path = Path(run.path)
    assert (path / record["stdout"]).read_text() == "output"
    commands = list((path / "commands").glob("*.json"))
    assert len(commands) == 1
    assert json.loads(commands[0].read_text())["command"] == record["command"]
    assert json.loads(
        Path(run.write_json("raw/nested/data.json", {"ok": True})).read_text()
    ) == {"ok": True}


@pytest.mark.parametrize("area", STAGES)
def test_stage_rejects_escaping_run_or_output(area, tmp_path):
    module = importlib.import_module(f"hqsb.{area}.experiment")
    experiment = next(iter(module.EXPERIMENTS))
    for run_id in ("../../outside", "/absolute", "..", "", "bad\\name"):
        with pytest.raises(ConfigError):
            module.RunDirectory(str(tmp_path), experiment, run_id)
    run = module.RunDirectory(str(tmp_path), experiment, "valid")
    run.create()
    for output in ("../../outside.json", "/tmp/outside.json"):
        with pytest.raises(ConfigError):
            run.write_json(output, {})
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (Path(run.path) / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigError):
        run.write_text("escape/data.txt", "must not escape")
