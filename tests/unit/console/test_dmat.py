"""DMAT keeps exact integers, event ownership, completeness and graph mass."""
import json
import struct

import pytest

from hqsb.core.diagnostics import DiagnosticPlan
from hqsb.observation.dmat_reports import read_frame, aggregate, summarize_memory


def fixture(root, *, capacity=2, overflow=False):
    metadata = root / "kernel"
    metadata.mkdir()
    sites = [{"site": 0, "opcode": "ld.global.u32", "operation": "read", "width": 4},
             {"site": 1, "opcode": "st.global.u32", "operation": "write", "width": 4}]
    (metadata / "memory-sites.json").write_text(json.dumps({"kernel": "add", "capacity": capacity, "sites": sites}))
    data = b"HQDMAT1\n" + struct.pack("<8I", 1, 1, 1, 2, 1, 1, 0, capacity)
    data += struct.pack("<2Q", 3 if overflow else 2, 1)
    data += struct.pack("<QQIIII", 2**60, 2**56 + 4096, 0, 0, 3, 4)
    data += struct.pack("<QQIIII", 2**60 + 32, 2**56 + 8192, 1, 0, 1, 4)
    data += struct.pack("<QQIIII", 2**60 + 16, 2**56 + 4096, 0, 1, 3, 4)
    data += bytes(32)
    path = root / "memory-0.dmat"
    path.write_bytes(data)
    (root / "hqsb-launches.jsonl").write_text(json.dumps({"captured": True, "kernel": "add", "grid": [1, 1, 1], "block": [2, 1, 1], "memory_metadata": str(metadata)}) + "\n")
    (root / "coverage.json").write_text(json.dumps({"captured_launches": 1}))
    return path, sites


def test_exact_addresses_above_JS_precision_and_density_conservation(tmp_path):
    fixture(tmp_path)
    panel = summarize_memory(tmp_path, page_bytes=4096, time_bins=16)
    plot = panel["memory_launches"][0]["plot"]
    assert plot["addresses"] == [hex(2**56 + 4096), hex(2**56 + 8192)]
    assert plot["gpu_start_ns"] == str(2**60)
    assert plot["duration_ns"] == 33
    assert sum(c["events"] for c in plot["cells"]) == 3
    assert sum(c["reads"] for c in plot["cells"]) == 2
    assert sum(c["instruction_bytes"] for c in plot["cells"]) == 12
    assert panel["availability"] == "observed"
    assert (tmp_path / "memory-0.svg").read_text().startswith('<svg')


def test_overflow_preserves_observed_plot_and_counts_missing_events(tmp_path):
    fixture(tmp_path, overflow=True)
    panel = summarize_memory(tmp_path)
    assert panel["availability"] == "partial"
    assert panel["counts"]["dropped_events"] == 1
    assert panel["memory_launches"][0]["plot"]["overflow_threads"] == 1


def test_dense_grid_coarsens_time_without_losing_event_mass(tmp_path):
    import numpy as np
    path, sites = fixture(tmp_path)
    frame = read_frame(path)
    n = 128 * 128
    rows = np.zeros((n, 1), dtype=frame["records"].dtype)
    rows["time"][:, 0] = 1 + np.arange(n) // 128
    rows["address"][:, 0] = 2**56 + (np.arange(n) % 128) * 4096
    rows["thread"][:, 0] = np.arange(n)
    rows["width"] = 4
    rows["mask"] = 0xffffffff
    frame.update(records=rows, counts=np.ones(n, dtype="<u8"), capacity=1, block=[128, 1, 1])
    plot = aggregate(frame, sites, page_bytes=4096, time_bins=512)
    assert len(plot["cells"]) <= 8192
    assert plot["requested_time_bins"] == 512 and plot["time_bins"] == 64
    assert sum(c["events"] for c in plot["cells"]) == n


@pytest.mark.parametrize("cut", [1, 15, 32, 127])
def test_truncation_never_becomes_complete_trace(tmp_path, cut):
    path, _ = fixture(tmp_path)
    path.write_bytes(path.read_bytes()[:-cut])
    with pytest.raises(ValueError, match="DMAT_TRUNCATED"):
        read_frame(path)


def test_ownership_site_width_and_lineage_are_checked(tmp_path):
    path, sites = fixture(tmp_path)
    frame = read_frame(path)
    sites[0]["width"] = 8
    with pytest.raises(ValueError, match="SITE_OR_WIDTH"):
        aggregate(frame, sites)
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 40 + 16 + 20, 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="THREAD_OR_CLOCK"):
        summarize_memory(tmp_path)


def test_memory_plan_uses_its_own_raw_format():
    kwargs = {"source_run_id": "run_" + "a" * 32, "tool": "eretrace", "ere_mode": "memory"}
    with pytest.raises(ValueError, match="raw"):
        DiagnosticPlan(**kwargs)
    assert DiagnosticPlan(**kwargs, ere_codec="raw").memory_capacity == 256


def test_pytorch_source_freezes_input_and_rejects_changed_weights(tmp_path):
    from types import SimpleNamespace
    from hqsb.observation.dmat_source import pytorch_reference
    from hqsb.observation.storage import sha256
    manifest = tmp_path / "weights.sha256"
    manifest.write_text("verified weights")
    deployment = SimpleNamespace(provider="pytorch", manifest=str(manifest), precision="float16")
    source = {"kind": "generate", "state": "completed", "output": "42", "metrics": {
        "optimization_receipt": {"model_identity": {"hash": sha256(manifest)}}, "generated_token_ids": [42]},
        "config": {"prompt": "17+25=", "max_output_tokens": 2,
            "deployment": {"attention": "eager"}, "actual_runtime": {
                "parameter_dtypes": ["float16"], "kernel_plan": {"profile_id": "framework-native"},
                "device_name": "test", "source_environment": {"packages": {"torch": "test"}, "machine": "test"}}}}
    reference = pytorch_reference(source, deployment)
    assert reference["plan"]["request"]["prompt"] == "17+25="
    reference["plan"]["request"]["prompt"] = "other"
    assert source["config"]["prompt"] == "17+25="
    source["config"]["runtime_plan"] = {"compiler": "changed"}
    with pytest.raises(ValueError, match="ADAPTER_REQUIRED"):
        pytorch_reference(source, deployment)
    del source["config"]["runtime_plan"]
    manifest.write_text("different weights")
    with pytest.raises(ValueError, match="MANIFEST_CHANGED"):
        pytorch_reference(source, deployment)


def test_exported_trace_resolves_metadata_inside_its_new_artifact_root(tmp_path):
    import shutil
    original = tmp_path / "original"
    original.mkdir()
    fixture(original)
    metadata = original / "kernel" / "0_source"
    metadata.mkdir()
    shutil.copyfile(original / "kernel/memory-sites.json", metadata / "memory-sites.json")
    sidecar = original / "hqsb-launches.jsonl"
    identity = json.loads(sidecar.read_text())
    identity["memory_metadata"] = str(metadata)
    sidecar.write_text(json.dumps(identity) + "\n")
    exported = tmp_path / "exported"
    shutil.copytree(original, exported)
    shutil.rmtree(original)
    assert summarize_memory(exported)["counts"]["events"] == 3


def test_pytorch_target_runs_public_generation_and_closes_window(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from hqsb.backends import interactive
    from hqsb.observation import dmat_target
    events = []

    class Provider:
        def __init__(self, config):
            self.source_identity = {"hash": "weights"}
            self.source_environment = {"packages": {}, "machine": "test"}

        def load(self, **options):
            events.append("load_and_warmup")
            return {"device_name": "test"}

        def generate(self, request, cancel):
            assert request["prompt"] == "question"
            events.append("generate")
            yield {"kind": "result", "text": "answer", "finish_reason": "length", "metrics": {"generated_token_ids": [1]}}

        def close(self):
            events.append("close")

    def start():
        events.append("start")
        return 0

    def stop():
        events.append("stop")
        return 0

    monkeypatch.setattr(interactive, "InteractivePyTorch", Provider)
    monkeypatch.setattr(dmat_target.ctypes, "CDLL", lambda *_: SimpleNamespace(hqsb_eretrace_start=start, hqsb_eretrace_stop=stop))
    monkeypatch.setenv("NEUTRINO_HOOK_DRIVER", "test")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(dmat_target.signal, "signal", lambda *_: None)
    spec = {"directory": str(tmp_path), "run_id": "r", "config": {}, "options": {"attention": "eager", "load_options": {}},
            "workload": {"request": {"prompt": "question"}}, "plan": {"timeout_s": 30}}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["target", str(path)])
    assert dmat_target.main() == 0
    assert events == ["load_and_warmup", "start", "generate", "stop", "close"]
    receipt = json.loads(next((tmp_path / "attempts").glob("*/result.json")).read_text())
    assert receipt["generated_token_ids"] == [1]
    assert receipt["cleanup"] == "succeeded"


def v2_frame():
    import numpy as np
    dtype = np.dtype([('time', '<u8'), ('clock', '<u8'), ('entry', '<u8'), ('address', '<u8'),
                      ('site', '<u4'), ('thread', '<u4'), ('mask', '<u4'), ('width', '<u4')])
    records = np.zeros((2, 2), dtype=dtype)
    # Different scheduling origins, identical normalized algorithm timing.
    records[0, 0] = (1000, 100, 50, 0x10000, 0, 0, 3, 4)
    records[0, 1] = (1020, 120, 50, 0x11000, 1, 0, 3, 4)
    records[1, 0] = (2000, 900, 850, 0x10004, 0, 1, 3, 4)
    records[1, 1] = (2020, 920, 850, 0x11004, 1, 1, 3, 4)
    return {'grid': [1, 1, 1], 'block': [2, 1, 1], 'capacity': 2, 'counts': np.array([2, 2]),
            'records': records, 'event_bytes': 48}


def test_dual_clocks_use_entry_instead_of_first_access_and_keep_object_mass():
    sites = [{'site': 0, 'operation': 'read', 'width': 4}, {'site': 1, 'operation': 'write', 'width': 4}]
    tensors = [{'name': 'Q', 'role': 'Q', 'base': '65536', 'bytes': 8, 'shape': [2]},
               {'name': 'O', 'role': 'O', 'base': '69632', 'bytes': 8, 'shape': [2]}]
    normalized = aggregate(v2_frame(), sites, page_bytes=4096, time_bins=160, clock='thread_entry_clock64', tensors=tensors)
    global_time = aggregate(v2_frame(), sites, page_bytes=4096, time_bins=160, clock='GPU_globaltimer', tensors=tensors)
    assert normalized['time_unit'] == 'cycles' and normalized['duration'] == 71
    assert len(normalized['cells']) == 2 and all(c['events'] == 2 for c in normalized['cells'])
    assert sorted(c['column'] for c in normalized['cells']) == [50, 70]  # not first-access normalization
    assert len(global_time['cells']) == 4 and global_time['duration'] == 1021
    assert normalized['mapped_events'] == 4 and normalized['unknown_events'] == 0
    assert sum(c['instruction_bytes'] for c in normalized['cells']) == 16
    zoom = aggregate(v2_frame(), sites, page_bytes=4096, time_bins=160, clock='thread_entry_clock64', tensors=tensors, window=(60, 80))
    assert zoom['visible_events'] == 2 and zoom['events'] == 4
    assert zoom['time_start'] == 60 and zoom['cells'][0]['column'] == 10


def test_tensor_alias_and_missing_owner_remain_explicit():
    sites = [{'site': 0, 'operation': 'read', 'width': 4}, {'site': 1, 'operation': 'write', 'width': 4}]
    frame = v2_frame()
    tensors = [{'name': 'storage', 'role': 'storage', 'base': '65536', 'bytes': 8192},
               {'name': 'Q', 'role': 'Q', 'base': '65536', 'bytes': 8}]
    plot = aggregate(frame, sites, tensors=tensors)
    assert next(o['events'] for o in plot['objects'] if o['role'] == 'Q') == 2
    assert next(o['events'] for o in plot['objects'] if o['role'] == 'storage') == 2
    plot = aggregate(frame, sites, tensors=tensors[1:])
    assert plot['unknown_events'] == 2
    frame['records']['clock'][0, 0] = 1
    with pytest.raises(ValueError, match='ENTRY_CLOCK_ORDER'):
        aggregate(frame, sites)


def test_paper_density_quantiles_match_independent_reference():
    import numpy as np
    from hqsb.observation.dmat_grid import grid
    frame = v2_frame()
    rows = frame['records'].reshape(-1)
    sites = [{'site': 0, 'operation': 'read', 'width': 4}, {'site': 1, 'operation': 'write', 'width': 4}]
    plot = grid(frame, rows, sites, page_bytes=4096, time_bins=160, clock='thread_entry_clock64')
    # Original callback: nonzero 20/40/60/80 percentiles, then strict > thresholds.
    values = np.array([c['events'] for c in plot['cells']])
    boundaries = np.percentile(values, [20, 40, 60, 80])
    assert plot['objects'][0]['color_thresholds'] == boundaries.tolist()
    assert [c['level'] for c in plot['cells']] == [1 + sum(v > b for b in boundaries) for v in values]


def test_legacy_has_no_fabricated_thread_entry_clock(tmp_path):
    path, sites = fixture(tmp_path)
    with pytest.raises(ValueError, match='LEGACY_TRACE'):
        aggregate(read_frame(path), sites, clock='thread_entry_clock64')


def test_rebin_query_converts_integer_page_size_before_enum_validation():
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hqsb.console.diagnostic_routes import diagnostic_router
    app = FastAPI()
    jobs = SimpleNamespace(memory_view=lambda identity, launch_index, **kw: kw)
    app.include_router(diagnostic_router(jobs, None))
    with TestClient(app) as client:
        response = client.get('/diagnostics/jobs/run_x/memory/0?clock=GPU_globaltimer&page_bytes=4096&time_bins=160')
        assert response.status_code == 200
        assert response.json()['page_bytes'] == 4096
        assert client.get('/diagnostics/jobs/run_x/memory/0?page_bytes=abc').status_code == 422


def test_source_replay_requires_the_recorded_triton_implementation(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from hqsb.observation.dmat_source import pytorch_reference
    from hqsb.observation.storage import sha256
    from hqsb.core.optimization import fingerprint
    manifest = tmp_path / 'weights.sha256'
    manifest.write_text('same weights')
    deployment = SimpleNamespace(provider='pytorch', manifest=str(manifest), precision='float16')
    implementation = Path(__file__).resolve().parents[3] / 'hqsb/backends/model_kernels/triton_attention.py'
    kernel = {'profile_id': 'qwen3-attention-triton-n32-v1', 'triton_implementation': {'source_sha256': sha256(implementation)}}
    source = {'kind': 'generate', 'state': 'completed', 'output': '42', 'metrics': {
        'optimization_receipt': {'model_identity': {'hash': sha256(manifest)}}}, 'config': {
        'prompt': '17+25=', 'max_output_tokens': 2, 'deployment': {'attention': 'eager'},
        'actual_runtime': {'parameter_dtypes': ['float16'], 'kernel_plan': kernel,
                           'device_name': 'test', 'source_environment': {'packages': {}, 'machine': 'test'}}}}
    assert pytorch_reference(source, deployment)['engine']['kernel_plan_sha256'] == fingerprint(kernel)
    kernel['triton_implementation']['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SOURCE_KERNEL_CHANGED'):
        pytorch_reference(source, deployment)


def test_new_probe_requires_matching_native_record_abi(tmp_path):
    from hqsb.observation.eretrace_tools import launch_environment
    from hqsb.observation.storage import sha256
    driver = tmp_path / 'driver'
    driver.write_bytes(b'driver')
    (tmp_path / 'build').mkdir()
    (tmp_path / 'build/block_sched.toml').write_text('warp probe')
    (tmp_path / 'hqsb-runtime.json').write_text(json.dumps({'driver_library': str(driver),
        'driver_sha256': sha256(driver), 'capabilities': ['warp', 'memory']}))
    plan = DiagnosticPlan(source_run_id='run_' + 'a' * 32, tool='eretrace', ere_mode='memory', ere_codec='raw')
    with pytest.raises(ValueError, match='RUNTIME_ABI_REQUIRED'):
        launch_environment(tmp_path, tmp_path / 'capture', plan.model_dump(), tmp_path / 'cache')


def test_diagnostic_budget_scan_tolerates_jit_tempfile_rename(tmp_path, monkeypatch):
    from pathlib import Path
    from hqsb.console.diagnostic_jobs import directory_bytes
    (tmp_path / 'stable').write_bytes(b'12345')
    (tmp_path / 'tmp.pid_jit').write_bytes(b'large transient object')
    original = Path.stat

    def disappear(path, *args, **kwargs):
        if path.name == 'tmp.pid_jit':
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', disappear)
    assert directory_bytes(tmp_path) == 5


def test_pair_alignment_uses_tensor_origin_despite_different_allocation_page_offsets(tmp_path):
    from hqsb.observation.dmat_compare import compare_memory
    sites = [{'site': 0, 'operation': 'read', 'width': 4}, {'site': 1, 'operation': 'write', 'width': 4}]
    panels, roots = [], []
    for i, shift in enumerate((0, 512)):
        root = tmp_path / str(i)
        root.mkdir()
        frame = v2_frame()
        frame['records']['address'] += shift
        tensors = [{'name': 'Q', 'role': 'Q', 'base': str(0x10000 + shift), 'bytes': 8, 'shape': [2]},
                   {'name': 'O', 'role': 'O', 'base': str(0x11000 + shift), 'bytes': 8, 'shape': [2]}]
        context = {'operation': 'test', 'semantic_shape': {'q': [2]}, 'tensors': tensors}
        plot = aggregate(frame, sites, clock='thread_entry_clock64', page_bytes=4096, tensors=tensors)
        (root / 'memory-0.dmat').write_bytes(b'HQDMAT2\n' + struct.pack('<8I', 1, 1, 1, 2, 1, 1, 0, 2)
            + frame['counts'].astype('<u8').tobytes() + frame['records'].tobytes())
        panels.append({'memory_launches': [{'context': context, 'plot': plot, 'trace': 'memory-0.dmat', 'probe': {'sites': sites}}]})
        roots.append(root)
    result = compare_memory(*panels, roots=roots)
    assert result['comparable'] and not result['density_matrix_changed']
    assert not result['structure_changed'] and result['pairs'][0]['aligned_density_l1'] == 0
    left, right = result['pairs'][0]['baseline'], result['pairs'][0]['candidate']
    assert [r['label'] for r in left['address_rows']] == [r['label'] for r in right['address_rows']]
    assert left['addresses'] != right['addresses']  # true VAs are still retained


def test_legacy_global_time_zoom_rebins_raw_events(tmp_path):
    path, sites = fixture(tmp_path)
    plot = aggregate(read_frame(path), sites, page_bytes=4096, time_bins=16, window=(8, 24))
    assert plot['visible_events'] == 1 and plot['events'] == 3
    assert plot['clock_domain'] == 'GPU_globaltimer' and plot['time_start'] == 8
    assert plot['cells'][0]['column'] == 8


def test_pair_palette_pooling_does_not_treat_timing_jitter_as_access_count_change(tmp_path):
    import numpy as np
    from hqsb.observation.dmat_compare import compare_memory
    sites = [{'site': 0, 'operation': 'read', 'width': 4}, {'site': 1, 'operation': 'write', 'width': 4}]
    panels, roots = [], []
    for i in range(2):
        root = tmp_path / str(i)
        root.mkdir()
        frame = v2_frame()
        if i:
            frame['records']['clock'][1, 1] += 10
        tensors = [{'name': 'Q', 'role': 'Q', 'base': '65536', 'bytes': 8, 'shape': [2]},
                   {'name': 'O', 'role': 'O', 'base': '69632', 'bytes': 8, 'shape': [2]}]
        plot = aggregate(frame, sites, clock='thread_entry_clock64', page_bytes=4096, tensors=tensors)
        (root / 'memory-0.dmat').write_bytes(b'HQDMAT2\n' + struct.pack('<8I', 1, 1, 1, 2, 1, 1, 0, 2)
            + frame['counts'].astype('<u8').tobytes() + frame['records'].tobytes())
        panels.append({'memory_launches': [{'context': {'operation': 'test', 'semantic_shape': {'q': [2]}, 'tensors': tensors},
            'plot': plot, 'trace': 'memory-0.dmat', 'probe': {'sites': sites}}]})
        roots.append(root)
    result = compare_memory(*panels, roots=roots)
    assert result['comparable'] and result['density_matrix_changed']
    assert not result['access_counts_changed'] and not result['structure_changed']
    plots = [result['pairs'][0][key] for key in ('baseline', 'candidate')]
    boundaries = np.percentile([2, 1, 1], [20, 40, 60, 80]).tolist()
    for plot in plots:
        obj = next(o for o in plot['objects'] if o['role'] == 'O')
        assert obj['color_thresholds'] == boundaries
        assert all(c['level'] == 1 + sum(c['events'] > t for t in boundaries) for c in plot['cells'] if c['object'] == obj['id'])


def test_run_indexes_keep_dmat_counts_and_load_full_grids_only_on_detail(tmp_path):
    from hqsb.console.store import Store
    store = Store(tmp_path)
    try:
        run, _ = store.create('diagnostic', {'deployment': {'id': 'x'}}, 'dmat-index', 'digest')
        launches = [{'plot': {'cells': [{'events': i} for i in range(1000)]}}]
        store.update(run['id'], {'metrics': {'diagnostic': {'panel': {
            'mode': 'memory', 'counts': {'events': 1000}, 'memory_launches': launches}}}})
        row = store.list(kind='diagnostic')['items'][0]
        assert row['metrics']['diagnostic']['panel']['counts'] == {'events': 1000}
        assert 'memory_launches' not in row['metrics']['diagnostic']['panel']
        assert len(json.dumps(row)) < 2000
        assert store.get(run['id'])['metrics']['diagnostic']['panel']['memory_launches'] == launches
    finally:
        store.db.close()
