"""Acquisition invariants with a byte transport double; not GPU/model evidence."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.model_hub import ModelHub, save_json
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.models.model_admission import GIB, admission
from hqsb.models.modelscope_source import file_url, validate_repo

CONTENTS = {"config.json": b'{"model_type":"qwen3"}', "tokenizer_config.json": b'{}',
            "tokenizer.json": b'{}', "model.safetensors": b"verified-publisher-weights" * 4}


def plan():
    return {"model_id": "Qwen/Test", "revision": "master", "files": [
        {"Path": name, "Size": len(content), "Sha256": hashlib.sha256(content).hexdigest(), "Revision": "a" * 40}
        for name, content in CONTENTS.items()], "config": {"model_type": "qwen3"},
        "download_bytes": sum(map(len, CONTENTS.values())), "weight_bytes": len(CONTENTS["model.safetensors"]), "runtime_reasons": []}


class Source:
    offsets = []
    corrupt = False
    slow = False

    def close(self):
        pass

    def page(self, owner, page, page_size=50):
        return [{"model_id": "Qwen/Test", "parameters": 100, "weight_bytes": 200,
                 "architectures": ["Qwen3ForCausalLM"]}], 1

    def revisions(self, repo):
        validate_repo(repo)
        return {"branches": ["master"], "tags": ["v1"]}

    def inspect(self, repo, revision):
        result = plan()
        result.update(model_id=validate_repo(repo), revision=revision)
        return result

    def stream(self, repo, entry, offset=0):
        self.offsets.append((entry["Path"], offset))
        content = CONTENTS[entry["Path"]]
        if self.corrupt and entry["Path"] == "model.safetensors":
            content = b"X" * len(content)
        slow = self.slow

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def iter_content(self, size):
                for start in range(offset, len(content), 4):
                    if slow:
                        time.sleep(.02)
                    yield content[start:start + 4]
        return Response()


class Actor:
    def __init__(self, config):
        self.config = config

    def close(self):
        pass


def capacity():
    return {"gpu_total_bytes": 24 * GIB, "gpu_name": "test-only", "host_available_bytes": 64 * GIB, "disk_free_bytes": 50 * GIB}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # The transport/capacity fixture models a 50 GiB disk. Match the download
    # worker's live disk boundary as well, independent of the target's free space.
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(total=100 * GIB, used=50 * GIB, free=capacity()["disk_free_bytes"]))
    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path, web_dist=tmp_path / "dist",
                        model_download_dir=tmp_path / "models", deployments=[DeploymentConfig(id="baseline", name="baseline")])
    service = ConsoleService(settings, actor_factory=Actor)
    Source.offsets, Source.corrupt, Source.slow = [], False, False
    hub = ModelHub(settings, service, source_factory=Source, capacity=capacity)
    yield settings, service, hub
    hub.close()
    service.close()


def wait(hub, identity):
    for _ in range(500):
        row = next(item for item in hub.list_jobs()["items"] if item["id"] == identity)
        if row["state"] not in {"queued", "downloading", "verifying", "cancelling"}:
            return row
        time.sleep(.01)
    raise AssertionError("download did not terminate")


def test_success_seals_publisher_hashes_and_registers_without_loading(setup):
    settings, service, hub = setup
    inspected = hub.inspect("Qwen/Test", "master")
    job = hub.start(inspected["plan_sha256"])
    row = wait(hub, job["id"])
    assert row["state"] == "completed", row
    assert row["downloaded_bytes"] == row["total_bytes"]
    assert row["verified_files"] == 4
    assert set(path.name for path in Path(row["snapshot_path"]).iterdir()) == set(CONTENTS)
    assert service.states["baseline"]["state"] == "unloaded"
    registered = hub.register(job["id"])
    assert service.states[registered["deployment_id"]]["state"] == "unloaded"
    assert service.configs[registered["deployment_id"]].kernel_profile == "framework-native"
    assert hub.register(job["id"])["deployment_id"] == registered["deployment_id"]
    assert (hub.root / "deployments.json").exists()
    # Runtime registration survives service restart without mutating YAML.
    hub.close()
    another = ConsoleService(settings, actor_factory=Actor)
    recovered = ModelHub(settings, another, source_factory=Source, capacity=capacity)
    assert registered["deployment_id"] in another.configs
    assert recovered.list_jobs()["items"][0]["deployment_id"] == registered["deployment_id"]
    recovered.close()
    another.close()


def test_duplicate_start_is_idempotent(setup):
    hub = setup[2]
    digest = hub.inspect("Qwen/Test", "master")["plan_sha256"]
    first, second = hub.start(digest), hub.start(digest)
    assert first["id"] == second["id"]
    assert len(hub.list_jobs()["items"]) == 1
    assert wait(hub, first["id"])["state"] == "completed"


def test_partial_progress_cancel_and_resume_preserves_identity(setup):
    hub = setup[2]
    Source.slow = True
    digest = hub.inspect("Qwen/Test", "master")["plan_sha256"]
    first = hub.start(digest)
    for _ in range(200):
        if hub.list_jobs()["items"][0]["downloaded_bytes"] > 0:
            break
        time.sleep(.005)
    hub.cancel(first["id"])
    cancelled = wait(hub, first["id"])
    assert cancelled["state"] == "cancelled"
    Source.slow = False
    resumed = hub.resume(first["id"])
    assert resumed["attempt"] == 2
    assert wait(hub, resumed["id"])["state"] == "completed"
    assert any(offset > 0 for _, offset in Source.offsets)


def test_hash_failure_cannot_publish_or_register(setup):
    hub = setup[2]
    Source.corrupt = True
    first = hub.start(hub.inspect("Qwen/Test", "master")["plan_sha256"])
    failed = wait(hub, first["id"])
    assert failed["state"] == "failed"
    assert "HASH_MISMATCH" in failed["error"]
    assert not (hub.models / first["plan_sha256"]).exists()
    with pytest.raises(ServiceError, match="校验封存"):
        hub.register(first["id"])
    Source.corrupt = False
    hub.resume(first["id"])
    assert wait(hub, first["id"])["state"] == "completed"


def test_live_disk_shortage_cannot_publish_after_capacity_preflight(setup, monkeypatch):
    _, service, hub = setup
    digest = hub.inspect("Qwen/Test", "master")["plan_sha256"]
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    job = hub.start(digest)
    row = wait(hub, job["id"])
    assert row["state"] == "failed"
    assert "MODEL_DISK_BUDGET" in row["error"]
    assert not (hub.models / digest).exists()
    assert service.states["baseline"]["state"] == "unloaded"


def test_restart_marks_active_download_interrupted(setup):
    settings, service, hub = setup
    job = {"id": "download-" + "f" * 32, "state": "downloading", "attempt": 1, "plan_sha256": "a" * 64}
    save_json(hub.root / (job["id"] + ".json"), job)
    recovered = ModelHub(settings, service, source_factory=Source, capacity=capacity)
    assert recovered.jobs[job["id"]]["state"] == "interrupted"
    assert json.loads((hub.root / (job["id"] + ".json")).read_text())["state"] == "interrupted"
    recovered.close()


def test_corrupt_existing_file_is_not_reused(setup):
    hub = setup[2]
    inspected = hub.inspect("Qwen/Test", "master")
    directory = hub.models / ("partial-" + inspected["plan_sha256"])
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("wrong")
    row = wait(hub, hub.start(inspected["plan_sha256"])["id"])
    assert row["state"] == "failed"
    assert "CORRUPT" in row["error"]


def test_plan_expires_and_queue_budget_rechecks(setup):
    hub = setup[2]
    inspected = hub.inspect("Qwen/Test", "master")
    hub.inspections[inspected["plan_sha256"]] = (0, inspected["plan"])
    with pytest.raises(ServiceError, match="重新检查"):
        hub.start(inspected["plan_sha256"])
    hub.inspect("Qwen/Test", "master")
    hub.capacity_source = lambda: {**capacity(), "disk_free_bytes": 0}
    with pytest.raises(ServiceError, match="磁盘"):
        hub.start(inspected["plan_sha256"])


def test_index_search_and_resource_filter_cover_full_publisher(setup):
    hub = setup[2]
    hub.refresh("Qwen")
    for _ in range(100):
        result = hub.catalog("Qwen", "test", True)
        if result["state"] == "completed":
            break
        time.sleep(.01)
    assert result["indexed"] == 1
    assert result["total"] == 1
    assert hub.catalog("Qwen", "missing")["total"] == 0
    with pytest.raises(ServiceError):
        hub.refresh("random-community")


def test_native_text_missing_summary_size_remains_pending_not_admitted(setup):
    hub = setup[2]
    hub.catalogs["google"] = {"publisher": "google", "state": "completed", "error": None,
        "total_upstream": 3, "items": [
            {"model_id": "google/gemma-3-270m-it", "parameters": None, "weight_bytes": None,
             "architectures": ["Gemma3ForCausalLM"]},
            {"model_id": "google/too-large", "parameters": 100_000_000_000, "weight_bytes": None,
             "architectures": ["Gemma3ForCausalLM"]},
            {"model_id": "google/gemma-GGUF", "parameters": None, "weight_bytes": None,
             "architectures": ["Gemma3ForCausalLM"]},
        ]}
    result = hub.catalog("google", only_fit=True)
    assert result["indexed"] == 3
    assert result["total"] == 1
    row = result["items"][0]
    assert row["model_id"] == "google/gemma-3-270m-it"
    assert row["candidate_status"] == "needs_inspection"
    assert row["resource_candidate"] is False
    assert row["estimate"]["estimated_runtime_bytes"] is None


@pytest.mark.parametrize("field,direction,names", [
    ("parameters", "asc", ["B", "E", "A", "C", "D"]),
    ("parameters", "desc", ["A", "B", "E", "C", "D"]),
    ("runtime_bytes", "asc", ["C", "B", "E", "A", "D"]),
    ("runtime_bytes", "desc", ["A", "B", "E", "C", "D"]),
])
def test_catalogue_numeric_sort_is_global_stable_and_unknown_last(setup, field, direction, names):
    hub = setup[2]
    # C has known weight bytes but unknown parameter count. Sorting must follow
    # the selected property's actual value, not labels or admission categories.
    values = [("D", None, None), ("E", 1_000_000_000, None), ("C", None, GIB),
              ("A", 2_000_000_000, None), ("B", 1_000_000_000, None)]
    hub.catalogs["Qwen"] = {"publisher": "Qwen", "state": "completed", "error": None,
        "total_upstream": 5, "items": [{"model_id": "Qwen/" + name, "parameters": params,
            "weight_bytes": weight, "architectures": ["Qwen3ForCausalLM"]} for name, params, weight in values]}
    result = hub.catalog("Qwen", only_fit=True, sort_by=field, sort_order=direction)
    expected = ["Qwen/" + name for name in names]
    assert [row["model_id"] for row in result["items"]] == expected
    assert result["sort_by"] == field and result["sort_order"] == direction
    paged = [row["model_id"] for offset in (0, 2, 4)
             for row in hub.catalog("Qwen", only_fit=True, offset=offset, limit=2,
                                    sort_by=field, sort_order=direction)["items"]]
    assert paged == expected
    assert hub.catalog("Qwen", "missing", sort_by=field, sort_order=direction)["items"] == []


def test_catalogue_rejects_unsupported_sort_parameters(setup):
    hub = setup[2]
    with pytest.raises(ServiceError, match="排序"):
        hub.catalog("Qwen", sort_by="made-up")
    with pytest.raises(ServiceError, match="排序"):
        hub.catalog("Qwen", sort_order="sideways")


def test_refresh_keeps_cached_catalogue_until_replacement_is_ready(setup):
    hub = setup[2]
    entered, release = threading.Event(), threading.Event()
    old = Source().page("Qwen", 1)[0][0]
    new = {**old, "model_id": "Qwen/New"}
    hub.catalogs["Qwen"] = {"publisher": "Qwen", "state": "completed", "items": [old],
                            "total_upstream": 1, "error": None}

    class BlockingSource(Source):
        def page(self, owner, page, page_size=50):
            if page == 1:
                return [new], 2
            entered.set()
            assert release.wait(5)
            return [{**new, "model_id": "Qwen/Another"}], 2

    hub.source_factory = BlockingSource
    try:
        hub.refresh("Qwen")
        assert entered.wait(5)
        result = hub.catalog("Qwen", only_fit=True)
        assert result["state"] == "refreshing"
        assert result["retained_cached"] is True
        assert result["scanned"] == 2
        assert result["items"][0]["model_id"] == "Qwen/Test"
    finally:
        release.set()
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen", only_fit=True)
    assert result["state"] == "completed"
    assert result["retained_cached"] is False
    assert {row["model_id"] for row in result["items"]} == {"Qwen/New", "Qwen/Another"}


def test_duplicate_upstream_pages_report_incomplete_catalogue_reason(setup):
    hub = setup[2]
    row = Source().page("Qwen", 1)[0][0]

    class DuplicatedSource(Source):
        def page(self, owner, page, page_size=50):
            return ([row, row] if page == 1 else [row] if page == 2 else []), 3

    hub.source_factory = DuplicatedSource
    hub.refresh("Qwen")
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen")
    assert result["state"] == "partial"
    assert result["total_upstream"] == 3
    assert result["scanned"] == 5
    assert result["indexed"] == 1
    assert result["duplicates"] == 4
    assert "无法确认完整性" in result["error"]
    assert "重复 4" in result["error"]


def test_single_response_covers_equal_rank_rows_without_page_boundary_loss(setup):
    hub = setup[2]
    template = Source().page("Qwen", 1)[0][0]
    calls = []

    class SnapshotSource(Source):
        def page(self, owner, page, page_size=50):
            calls.append((page, page_size))
            # A split list repeats A and omits C at an equal-ranked boundary.
            if page_size == 50:
                return [{**template, "model_id": "Qwen/A"}], 3
            return [{**template, "model_id": "Qwen/" + name} for name in "ABC"], 3

    hub.source_factory = SnapshotSource
    hub.refresh("Qwen")
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen")
    assert result["state"] == "completed" and result["error"] is None
    assert result["indexed"] == result["total_upstream"] == 3
    assert {row["model_id"] for row in result["items"]} == {"Qwen/A", "Qwen/B", "Qwen/C"}
    assert calls == [(1, 10000)]
    assert result["retrieval"] == "single_response" and result["page_requests"] == 1


def test_capped_snapshot_falls_back_to_pages_and_checks_unique_coverage(setup):
    hub = setup[2]
    template = Source().page("Qwen", 1)[0][0]
    calls = []

    class CappedSource(Source):
        def page(self, owner, page, page_size=50):
            calls.append((page, page_size))
            return [{**template, "model_id": "Qwen/" + ("A" if page == 1 else "B")}], 2

    hub.source_factory = CappedSource
    hub.refresh("Qwen")
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen")
    assert result["state"] == "completed" and result["indexed"] == 2
    assert calls == [(1, 10000), (1, 50), (2, 50)]
    assert result["retrieval"] == "paged" and result["page_requests"] == 3


def test_changing_upstream_count_does_not_claim_complete_snapshot(setup):
    hub = setup[2]
    template = Source().page("Qwen", 1)[0][0]

    class ChangingSource(Source):
        def page(self, owner, page, page_size=50):
            return ([template] if page == 1 else []), (2 if page_size == 10000 else 1)

    hub.source_factory = ChangingSource
    hub.refresh("Qwen")
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen")
    assert result["state"] == "partial"
    assert "总数发生变化" in result["error"]


def test_refresh_failure_retains_previous_usable_models(setup):
    hub = setup[2]
    hub.catalogs["Qwen"] = {"publisher": "Qwen", "state": "completed", "items": Source().page("Qwen", 1)[0],
                            "total_upstream": 1, "error": None}

    class OfflineSource(Source):
        def page(self, owner, page, page_size=50):
            raise OSError("upstream unavailable")

    hub.source_factory = OfflineSource
    hub.refresh("Qwen")
    hub.catalog_pool.shutdown(wait=True)
    result = hub.catalog("Qwen", only_fit=True)
    assert result["state"] == "partial"
    assert "upstream unavailable" in result["error"]
    assert result["items"][0]["model_id"] == "Qwen/Test"


@pytest.mark.parametrize("repo", ["../bad", "Qwen/../../bad", "unknown/weights", "Qwen/x?url=http://localhost", "Qwen/a/b"])
def test_untrusted_repositories_are_rejected(repo):
    with pytest.raises(ValueError):
        validate_repo(repo)


def test_immutable_file_commit_and_safe_url_encoding():
    entry = {"Path": "tokenizer.json", "Revision": "a" * 40}
    assert "Revision=" + "a" * 40 in file_url("Qwen/Test", entry)
    with pytest.raises(ValueError, match="REVISION"):
        file_url("Qwen/Test", {**entry, "Revision": "master"})


@pytest.mark.parametrize("change,reason", [({"gpu_total_bytes": None}, "容量未知"), ({"gpu_total_bytes": 1}, "超过"),
    ({"host_available_bytes": 0}, "主机"), ({"disk_free_bytes": 0}, "磁盘")])
def test_admission_does_not_convert_unknown_or_insufficient_to_success(change, reason):
    report = admission(plan(), {**capacity(), **change}, GIB)
    assert not report["eligible"]
    assert any(reason in item for item in report["reasons"])


def test_routes_use_normal_csrf_guard_and_preserve_other_features(setup):
    settings, service, hub = setup
    app = create_app(settings, service=service, monitor=False)
    with TestClient(app) as client:
        assert client.get(PREFIX + "/deployments").status_code == 200
        assert client.get(PREFIX + "/model-hub/catalog?publisher=Qwen&sort_by=made-up").status_code == 422
        assert client.get(PREFIX + "/model-hub/catalog?publisher=Qwen&sort_order=sideways").status_code == 422
        sorted_catalog = client.get(PREFIX + "/model-hub/catalog?publisher=Qwen&sort_by=parameters&sort_order=desc")
        assert sorted_catalog.status_code == 200
        assert sorted_catalog.json()["sort_by"] == "parameters"
        assert sorted_catalog.json()["sort_order"] == "desc"
        assert client.post(PREFIX + "/model-hub/catalog/refresh?publisher=Qwen", json={}).status_code == 403
        response = client.post(PREFIX + "/model-hub/downloads", json={"plan_sha256":"f"*64}, headers={"X-HQSB-Client":"console"})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "MODEL_INSPECTION_EXPIRED"


def test_recover_crash_between_snapshot_rename_and_journal(setup):
    hub = setup[2]
    first = hub.start(hub.inspect("Qwen/Test", "master")["plan_sha256"])
    completed = wait(hub, first["id"])
    transfers = len(Source.offsets)
    hub._update(first["id"], state="interrupted", snapshot_path=None, manifest_path=None)
    hub.resume(first["id"])
    row = wait(hub, first["id"])
    assert row["state"] == "completed"
    assert row["snapshot_path"] == completed["snapshot_path"]
    assert len(Source.offsets) == transfers


def test_sdk_plan_rejects_unsafe_runtime_formats_and_missing_shards(monkeypatch):
    from hqsb.models.modelscope_source import ModelScopeSource
    monkeypatch.setattr("hqsb.models.modelscope_source.native_causal_types", lambda: {"qwen3"})
    entries = [{**entry, "Type": "blob"} for entry in plan()["files"]]
    source = ModelScopeSource.__new__(ModelScopeSource)

    class API:
        def get_model_files(self, *_args, **_kwargs):
            return entries
    source.api = API()
    source.revisions = lambda _: {"branches": ["master"], "tags": []}
    source.small_json = lambda repo, entry: {"model_type":"qwen3", "auto_map":{"AutoModelForCausalLM":"custom.Model"}, "quantization_config":{"bits":4}} if entry["Path"]=="config.json" else {"chat_template":"{{messages}}"}
    result = source.inspect("Qwen/Test", "master")
    assert len(result["runtime_reasons"]) == 2
    with pytest.raises(ValueError, match="REVISION_UNKNOWN"):
        source.inspect("Qwen/Test", "moved")
    index = b'{}'
    entries.append({"Type":"blob", "Path":"model.safetensors.index.json", "Size":2, "Sha256":hashlib.sha256(index).hexdigest(), "Revision":"a"*40})
    source.small_json = lambda repo, entry: {"weight_map":{"weight":"missing.safetensors"}} if "index" in entry["Path"] else {"model_type":"qwen3", "chat_template":"ok"}
    with pytest.raises(ValueError, match="SHARDS_MISSING"):
        source.inspect("Qwen/Test", "master")


@pytest.mark.parametrize('template', ['missing', 'embedded', 'jinja'])
def test_template_assets_choose_input_mode_without_blocking_base_weights(monkeypatch, template):
    from hqsb.models.modelscope_source import ModelScopeSource
    monkeypatch.setattr('hqsb.models.modelscope_source.native_causal_types', lambda: {'gemma3_text'})
    entries = [{**entry, 'Type':'blob'} for entry in plan()['files']]
    if template == 'jinja':
        entries.append({'Type':'blob','Path':'chat_template.jinja','Size':3,'Sha256':'a'*64,'Revision':'a'*40})
    source = ModelScopeSource.__new__(ModelScopeSource)
    class API:
        def get_model_files(self, *_args, **_kwargs):
            return entries
    source.api = API()
    source.revisions = lambda _: {'branches':['master'],'tags':[]}
    source.small_json = lambda repo, entry: {'model_type':'gemma3_text', 'torch_dtype':'bfloat16'} if entry['Path'] == 'config.json' else ({'chat_template':'{{messages}}'} if template == 'embedded' else {})
    result = source.inspect('google/gemma-3-270m','master')
    assert result['runtime_reasons'] == []
    assert result['input_mode'] == ('completion' if template == 'missing' else 'chat')
    assert result['chat_template_available'] is (template != 'missing')
    assert result['runtime_precision'] == 'bfloat16'
    if template == 'jinja':
        assert any(row['Path'] == 'chat_template.jinja' and row['Sha256'] == 'a'*64 for row in result['files'])


def test_registration_persists_completion_mode(setup, monkeypatch):
    hub, service = setup[2], setup[1]
    original = Source.inspect
    def raw_plan(self, repo, revision):
        return {**original(self, repo, revision), 'input_mode':'completion', 'chat_template_available':False, 'runtime_precision':'bfloat16'}
    monkeypatch.setattr(Source, 'inspect', raw_plan)
    inspection = hub.inspect('Qwen/Test','master')
    job = wait(hub, hub.start(inspection['plan_sha256'])['id'])
    registered = hub.register(job['id'])
    config = service.configs[registered['deployment_id']]
    assert config.input_mode == 'completion'
    assert config.precision == 'bfloat16'
    assert json.loads((hub.root/'deployments.json').read_text())[config.id]['input_mode'] == 'completion'


def test_effective_host_memory_respects_container_limit(tmp_path):
    from hqsb.models.model_admission import effective_host_available
    (tmp_path / "memory.max").write_text(str(8 * GIB))
    (tmp_path / "memory.current").write_text(str(3 * GIB))
    assert effective_host_available(600 * GIB, tmp_path) == 5 * GIB
    assert effective_host_available(2 * GIB, tmp_path) == 2 * GIB
    (tmp_path / "memory.max").write_text("max")
    assert effective_host_available(600 * GIB, tmp_path) == 600 * GIB


def test_registered_model_is_visible_in_kernel_and_benchmark_catalogs(setup):
    settings, service, hub = setup
    job = hub.start(hub.inspect("Qwen/Test", "master")["plan_sha256"])
    assert wait(hub, job["id"])["state"] == "completed"
    deployed = hub.register(job["id"])["deployment_id"]
    app = create_app(settings, service=service, monitor=False)
    with TestClient(app) as client:
        response = client.get(PREFIX + "/kernels/profiles", params={"deployment_id":deployed})
        assert response.status_code == 200, response.text
        profiles = response.json()["profiles"]
        assert next(row for row in profiles if row["profile_id"] == "framework-native")["availability"]["loadable"]
        response = client.get(PREFIX + "/benchmark-catalog", params={"deployment_id":deployed})
        assert response.status_code == 200, response.text
        assert response.json()["schemes"][0]["id"] == "framework-native"


def test_native_engine_mapping_reads_data_without_importing_torch(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from hqsb.models.modelscope_source import native_causal_types
    mapping = tmp_path / "modeling_auto.py"
    mapping.write_text("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([('qwen3','Qwen3ForCausalLM'),('llama','LlamaForCausalLM'),('gemma3_text','Gemma3ForCausalLM'),('glm4','Glm4ForCausalLM')])")
    monkeypatch.setattr("hqsb.models.modelscope_source.distribution", lambda _: SimpleNamespace(locate_file=lambda _: mapping))
    native_causal_types.cache_clear()
    before = "torch" in sys.modules
    supported = native_causal_types()
    assert {"qwen3", "llama", "gemma3_text", "glm4"}.issubset(supported)
    assert ("torch" in sys.modules) == before
    native_causal_types.cache_clear()


def test_empty_upstream_owner_is_an_empty_catalog_not_a_transport_error():
    from types import SimpleNamespace
    from hqsb.models.modelscope_source import ModelScopeSource
    source = ModelScopeSource.__new__(ModelScopeSource)
    source.api = SimpleNamespace(list_models=lambda *args, **kwargs: {"Models":None,"TotalCount":0})
    assert source.page("Qwen", 1) == ([], 0)


def test_resource_filter_excludes_known_unsupported_weight_formats(setup):
    hub = setup[2]
    common = {"parameters":100,"weight_bytes":200,"architectures":["Qwen3ForCausalLM"]}
    hub.catalogs['Qwen'] = {'publisher':'Qwen','state':'completed','total_upstream':2,
        'items':[{'model_id':'Qwen/Test',**common},{'model_id':'Qwen/Test-MLX-4bit',**common}]}
    result = hub.catalog('Qwen', only_fit=True)
    assert [row['model_id'] for row in result['items']] == ['Qwen/Test']
    assert hub.catalog('Qwen', only_fit=False)['total'] == 2


def completed_artifact(hub):
    digest = hub.inspect("Qwen/Test", "master")["plan_sha256"]
    return wait(hub, hub.start(digest)["id"])


def test_delete_registered_snapshot_preserves_history_and_can_download_again(setup):
    settings, service, hub = setup
    row = completed_artifact(hub)
    registered = hub.register(row["id"])
    deployment = registered["deployment_id"]
    historical, _ = service.store.create("generate", {"deployment_id": deployment, "model": "Qwen/Test"}, "history", "digest")
    provenance = Path(row["manifest_path"]).with_suffix(".provenance.json")
    assert registered["deletion"]["allowed"]
    result = hub.delete(row["id"], row["plan_sha256"])
    assert result["state"] == "deleted"
    assert not Path(row["snapshot_path"]).exists()
    assert deployment not in service.configs
    assert deployment not in service.actors
    assert deployment not in json.loads((hub.root / "deployments.json").read_text())
    assert service.store.get(historical["id"]) == historical
    assert Path(row["manifest_path"]).exists() and provenance.exists()
    assert hub.delete(row["id"], row["plan_sha256"])["state"] == "deleted"
    hub.close()
    another = ConsoleService(settings, actor_factory=Actor)
    recovered = ModelHub(settings, another, source_factory=Source, capacity=capacity)
    try:
        assert deployment not in another.configs
        new = completed_artifact(recovered)
        assert new["id"] != row["id"]
        assert new["state"] == "completed"
        assert recovered.register(new["id"])["deployment_id"] == deployment
    finally:
        recovered.close()
        another.close()


@pytest.mark.parametrize("state,epoch", [("ready", "loaded"), ("loading", None), ("unloaded", "stale")])
def test_delete_requires_unload_with_direct_deployment_link(setup, state, epoch):
    _, service, hub = setup
    row = completed_artifact(hub)
    deployment = hub.register(row["id"])["deployment_id"]
    service.states[deployment].update(state=state, epoch=epoch)
    status = hub.list_jobs()["items"][0]["deletion"]
    assert not status["allowed"]
    assert status["next"]["to"] == f"/devices#deployment-{deployment}"
    with pytest.raises(ServiceError):
        hub.delete(row["id"], row["plan_sha256"])
    assert Path(row["snapshot_path"]).exists()
    service.states[deployment].update(state="unloaded", epoch=None)
    assert hub.delete(row["id"], row["plan_sha256"])["state"] == "deleted"


@pytest.mark.parametrize("busy", ["pending", "benchmark_owner"])
def test_delete_waits_for_node_tasks(setup, busy):
    _, service, hub = setup
    row = completed_artifact(hub)
    setattr(service, busy, 1)
    try:
        with pytest.raises(ServiceError):
            hub.delete(row["id"], row["plan_sha256"])
        assert Path(row["snapshot_path"]).exists()
    finally:
        setattr(service, busy, 0 if busy == "pending" else None)


def test_delete_static_alias_is_blocked(setup):
    settings, service, hub = setup
    row = completed_artifact(hub)
    config = DeploymentConfig(id="static-alias", name="Static", provider="pytorch", model_path=row["snapshot_path"])
    settings.deployments.append(config)
    service.register_deployment(config)
    with pytest.raises(ServiceError):
        hub.delete(row["id"], row["plan_sha256"])
    assert Path(row["snapshot_path"]).exists()


def test_delete_cleans_cancelled_and_quarantined_bytes(setup):
    _, _, hub = setup
    Source.slow = True
    first = hub.start(hub.inspect("Qwen/Test", "master")["plan_sha256"])
    with pytest.raises(ServiceError):
        hub.delete(first["id"], first["plan_sha256"])
    hub.cancel(first["id"])
    row = wait(hub, first["id"])
    partial = hub.models / ("partial-" + row["plan_sha256"])
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "model.safetensors.part").write_bytes(b"partial")
    quarantine = hub.models / "quarantine"
    quarantine.mkdir()
    owned = quarantine / (row["id"] + "-" + "a" * 32)
    owned.write_bytes(b"corrupt")
    other = quarantine / ("download-" + "b" * 32 + "-" + "c" * 32)
    other.write_bytes(b"other")
    assert str(owned) in hub.list_jobs()["items"][0]["deletion"]["paths"]
    hub.delete(row["id"], row["plan_sha256"])
    assert not partial.exists() and not owned.exists()
    assert other.read_bytes() == b"other"


def test_delete_rejects_changed_identity_and_root_symlink(setup, tmp_path):
    _, _, hub = setup
    row = completed_artifact(hub)
    with pytest.raises(ServiceError):
        hub.delete(row["id"], "f" * 64)
    snapshot = Path(row["snapshot_path"])
    external = tmp_path / "external"
    snapshot.rename(external)
    snapshot.symlink_to(external, target_is_directory=True)
    with pytest.raises(ServiceError):
        hub.delete(row["id"], row["plan_sha256"])
    assert (external / "model.safetensors").exists()


def test_delete_inner_symlink_preserves_external_file(setup, tmp_path):
    _, _, hub = setup
    row = completed_artifact(hub)
    external = tmp_path / "keep"
    external.write_text("keep")
    (Path(row["snapshot_path"]) / "link").symlink_to(external)
    hub.delete(row["id"], row["plan_sha256"])
    assert external.read_text() == "keep"


def test_failed_delete_can_retry_after_actor_close_failure(setup, monkeypatch):
    _, service, hub = setup
    row = completed_artifact(hub)
    deployment = hub.register(row["id"])["deployment_id"]
    actor = service.actors[deployment]
    with monkeypatch.context() as context:
        def fail():
            raise OSError("fixture close error")
        context.setattr(actor, "close", fail)
        with pytest.raises(ServiceError) as caught:
            hub.delete(row["id"], row["plan_sha256"])
        assert caught.value.code == "MODEL_DELETE_FAILED"
    assert hub.list_jobs()["items"][0]["state"] == "delete_failed"
    assert hub.list_jobs()["items"][0]["deletion"]["allowed"]
    assert hub.delete(row["id"], row["plan_sha256"])["state"] == "deleted"


def test_interrupted_deletion_recovers_without_resurrecting_deployment(setup):
    settings, _, hub = setup
    row = completed_artifact(hub)
    deployment = hub.register(row["id"])["deployment_id"]
    hub._update(row["id"], state="deleting", delete_deployment_ids=[deployment])
    another = ConsoleService(settings, actor_factory=Actor)
    recovered = ModelHub(settings, another, source_factory=Source, capacity=capacity)
    try:
        assert deployment not in another.configs
        assert recovered.jobs[row["id"]]["state"] == "delete_failed"
        assert recovered.delete(row["id"], row["plan_sha256"])["state"] == "deleted"
    finally:
        recovered.close()
        another.close()


def test_delete_route_checks_csrf_identity_and_keeps_readable_history(setup):
    settings, service, hub = setup
    row = completed_artifact(hub)
    app = create_app(settings, service=service, monitor=False)
    path = PREFIX + f"/model-hub/downloads/{row['id']}/delete"
    with TestClient(app) as client:
        assert client.post(path, json={"plan_sha256": row["plan_sha256"]}).status_code == 403
        headers = {"X-HQSB-Client": "console"}
        assert client.post(path, json={"plan_sha256": "f" * 64}, headers=headers).status_code == 409
        response = client.post(path, json={"plan_sha256": row["plan_sha256"]}, headers=headers)
        assert response.status_code == 200
        assert response.json()["state"] == "deleted"
        assert client.get(PREFIX + "/deployments").status_code == 200


def local_inventory(setup):
    from hqsb.console.local_models import LocalModels
    settings, service, hub = setup
    service.configs['baseline'].model_path = str(settings.data_dir / 'configured-model')
    return LocalModels(hub)


def test_local_inventory_merges_download_and_deployment_paths(setup):
    local = local_inventory(setup)
    row = completed_artifact(setup[2])
    deployment = setup[2].register(row['id'])['deployment_id']
    report = local.snapshot()
    matches = [r for r in report['items'] if r['path'] == row['snapshot_path']]
    assert len(matches) == 1
    item = matches[0]
    assert item['kind'] == 'download'
    assert item['logical_bytes'] == sum(map(len, CONTENTS.values()))
    assert item['allocated_bytes'] >= item['logical_bytes']
    assert item['references'][0]['id'] == deployment
    assert item['deletion']['allowed']
    setup[1].states[deployment].update(state='ready', epoch='test')
    # Runtime references update even while filesystem accounting is cached.
    assert not next(r for r in local.snapshot()['items'] if r['id'] == item['id'])['deletion']['allowed']


def test_local_inventory_extra_roots_persist_and_discover_nested_model(setup, tmp_path):
    from hqsb.console.local_models import LocalModels
    local = local_inventory(setup)
    directory = tmp_path / 'other-models' / 'vendor' / 'model'
    directory.mkdir(parents=True)
    (directory / 'config.json').write_text('{"model_type":"llama"}')
    (directory / 'weights.bin').write_bytes(b'x' * 100)
    report = local.add_root(str(directory.parent.parent))
    row = next(r for r in report['items'] if r['path'] == str(directory))
    assert row['model_type'] == 'llama'
    assert row['deletion']['allowed']
    recovered = LocalModels(setup[2])
    assert str(directory.parent.parent) in recovered.snapshot()['roots']


def test_local_inventory_deduplicates_hardlinks_across_models(setup, tmp_path):
    import os
    local = local_inventory(setup)
    parent = tmp_path / 'shared'
    for name in ('first', 'second'):
        directory = parent / name
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
    weight = parent / 'first' / 'model.bin'
    weight.write_bytes(b'a' * 10000)
    os.link(weight, parent / 'second' / 'model.bin')
    report = local.add_root(str(parent))
    rows = [r for r in report['items'] if r['path'].startswith(str(parent))]
    assert len(rows) == 2
    assert report['logical_bytes'] == 10004
    assert sum(r['logical_bytes'] for r in rows) == 20004
    assert len(report['volumes']) == 1


def test_local_delete_checks_snapshot_and_preserves_other_files(setup, tmp_path):
    local = local_inventory(setup)
    directory = tmp_path / 'manual-model'
    directory.mkdir()
    config = directory / 'config.json'
    config.write_text('{}')
    report = local.add_root(str(directory))
    row = next(r for r in report['items'] if r['path'] == str(directory))
    config.write_text('{"changed":true}')
    with pytest.raises(ServiceError) as caught:
        local.delete(row['id'], row['revision'])
    assert caught.value.code == 'MODEL_DIRECTORY_CHANGED'
    row = next(r for r in local.snapshot()['items'] if r['id'] == row['id'])
    sentinel = tmp_path / 'keep'
    sentinel.write_text('keep')
    assert local.delete(row['id'], row['revision'])['state'] == 'deleted'
    assert not directory.exists() and sentinel.read_text() == 'keep'
    assert any(local.hub.root.glob('local-*.cleanup.json'))


def test_local_delete_download_updates_original_journal(setup):
    local = local_inventory(setup)
    job = completed_artifact(setup[2])
    setup[2].register(job['id'])
    row = next(r for r in local.snapshot()['items'] if r.get('download_id') == job['id'])
    local.delete(row['id'], row['revision'])
    assert setup[2].jobs[job['id']]['state'] == 'deleted'
    assert all(r.get('download_id') != job['id'] for r in local.snapshot()['items'])
    assert local.snapshot()['logical_bytes'] == 0


def test_local_inventory_lists_derived_artifact_and_blocks_loaded_reference(setup):
    settings, service, _ = setup
    local = local_inventory(setup)
    directory = settings.data_dir / 'l4-artifacts' / ('l4_' + 'a' * 32)
    directory.mkdir(parents=True)
    (directory / 'l4-manifest.json').write_text(json.dumps({'artifact_id': directory.name}))
    report = local.snapshot()
    row = next(r for r in report['items'] if r['path'] == str(directory))
    assert row['kind'] == 'l4'
    service.states['baseline'].update(state='ready', epoch='fixture', quantization={'artifact_id': directory.name})
    row = next(r for r in local.snapshot()['items'] if r['id'] == row['id'])
    assert row['references'][0]['id'] == 'baseline'
    assert not row['deletion']['allowed']
    with pytest.raises(ServiceError):
        local.delete(row['id'], row['revision'])
    service.states['baseline'].update(state='unloaded', epoch=None)
    assert local.delete(row['id'], row['revision'])['state'] == 'deleted'


def test_local_inventory_keeps_missing_configured_models_visible(setup):
    local = local_inventory(setup)
    row = local.snapshot()['items'][0]
    assert row['kind'] == 'configured' and not row['exists']
    assert row['logical_bytes'] == row['allocated_bytes'] == 0
    assert not row['deletion']['allowed']


def test_local_storage_api_uses_existing_csrf_and_strict_request_contract(setup, tmp_path):
    settings, service, _ = setup
    local_inventory(setup)
    directory = tmp_path / 'extra'
    directory.mkdir()
    (directory / 'config.json').write_text('{}')
    app = create_app(settings, service=service, monitor=False)
    base = PREFIX + '/model-hub/local-models'
    with TestClient(app) as client:
        assert client.get(base).status_code == 200
        assert client.post(base + '/roots', json={'path': str(directory)}).status_code == 403
        headers = {'X-HQSB-Client': 'console'}
        assert client.post(base + '/roots', json={'path': '/', 'extra': 1}, headers=headers).status_code == 422
        assert client.post(base + '/roots', json={'path': '/'}, headers=headers).status_code == 422
        report = client.post(base + '/roots', json={'path': str(directory)}, headers=headers).json()
        row = next(r for r in report['items'] if r['path'] == str(directory))
        assert client.post(base + '/' + row['id'] + '/delete', json={'revision':'f' * 64}, headers=headers).status_code == 409
        assert client.post(base + '/' + row['id'] + '/delete', json={'revision':row['revision']}, headers=headers).status_code == 200
        assert not directory.exists()


def test_local_l5_inventory_recovers_source_name_and_variant_from_run(setup):
    local = local_inventory(setup)
    run, _ = setup[1].store.create('compress', {'deployment': {'model': 'Vendor/Original'}}, 'l5-local-name', 'recipe')
    directory = setup[0].data_dir / 'l5-artifacts' / ('l5_' + 'b' * 32)
    directory.mkdir(parents=True)
    (directory / 'l5-manifest.json').write_text(json.dumps({'artifact_id':directory.name, 'creation_run_id':run['id'], 'recipe':{'method':'low_rank'}}))
    row = next(r for r in local.snapshot()['items'] if r['path'] == str(directory))
    assert row['model_id'] == 'Vendor/Original'
    assert row['variant'] == 'low_rank'


def test_local_inventory_handles_untyped_model_metadata_without_runtime_failure(setup, tmp_path):
    local = local_inventory(setup)
    directory = tmp_path / 'untyped'
    directory.mkdir()
    (directory / 'l4-manifest.json').write_text(json.dumps({'artifact_id':{}, 'model_type':[], 'recipe':{'method':{}}, 'creation_run_id':[]}))
    row = next(r for r in local.add_root(str(directory))['items'] if r['path'] == str(directory))
    assert row.get('artifact_id') is None
    assert row['deletion']['allowed']


def test_local_inventory_refreshes_paths_when_download_changes_state(setup):
    local = local_inventory(setup)
    hub = setup[2]
    completed = completed_artifact(hub)
    destination = Path(completed['snapshot_path'])
    partial = destination.with_name('partial-' + completed['plan_sha256'])
    destination.rename(partial)
    hub._update(completed['id'], state='interrupted', snapshot_path=None)
    cached = next(r for r in local.snapshot()['items'] if r.get('download_id') == completed['id'])
    assert cached['path'] == str(partial)
    partial.rename(destination)
    hub._update(completed['id'], state='completed', snapshot_path=str(destination))
    row = next(r for r in local.snapshot()['items'] if r.get('download_id') == completed['id'])
    assert row['path'] == str(destination)
    assert row['logical_bytes'] == sum(map(len, CONTENTS.values()))
