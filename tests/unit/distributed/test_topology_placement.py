"""E10-01 interface tests: topology identity, drift, degraded links, placement."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import placement as pl
from hqsb.distributed import probes as pb
from hqsb.distributed import ranks as rk
from hqsb.distributed import topology as tp


def _manifest(**overrides) -> tp.TopologyManifest:
    scope = tp.ObservationScope(branch="cuda_nccl", node_scope="single_node")
    host = tp.HostIdentity(
        node_id="n0", host_alias="host-a", cpu_arch="x86_64", os_release="22.04", kernel="6.5"
    )
    numa = tp.NumaTopology(
        nodes=(tp.NumaNode(0, 0, (0, 1), 1 << 34, 1 << 33),), core_to_numa={0: 0, 1: 0}
    )
    accelerators = tuple(
        tp.AcceleratorRecord(
            device_id=index, sku="GA102", uuid=f"GPU-{index}", pci_bdf=f"0000:0{index}:00.0"
        )
        for index in range(2)
    )
    nodes = tuple(tp.TopologyNode(f"dev{i}", "accelerator") for i in range(2))
    edges = (
        tp.TopologyEdge(
            src="dev0",
            dst="dev1",
            edge_type="nvlink",
            status="UP",
            confidence="MEASURED",
            source_tool="fixture",
            measured_latency_us=1.0,
            measured_bandwidth_gbps=400.0,
        ),
    )
    payload = {
        "scope": scope,
        "host": host,
        "numa": numa,
        "accelerators": accelerators,
        "nodes": nodes,
        "edges": edges,
        "collected_at": "2026-09-19T00:00:00Z",
    }
    payload.update(overrides)
    return tp.TopologyManifest(**payload)


@pytest.mark.unit
class TestManifestSchema:
    def test_valid_manifest_passes(self):
        assert _manifest().validate() == []

    def test_dangling_edge_is_rejected(self):
        manifest = _manifest(
            edges=(
                tp.TopologyEdge(src="dev0", dst="ghost", edge_type="nvlink", status="UP"),
            )
        )
        assert any("dangles" in error for error in manifest.validate())

    def test_measured_edge_requires_evidence(self):
        manifest = _manifest(
            edges=(
                tp.TopologyEdge(
                    src="dev0", dst="dev1", edge_type="nvlink", status="UP", confidence="MEASURED"
                ),
            )
        )
        errors = manifest.validate()
        assert any("MEASURED" in error for error in errors)

    def test_duplicate_inconsistent_edge_is_rejected(self):
        first = tp.TopologyEdge(src="dev0", dst="dev1", edge_type="nvlink", status="UP")
        second = tp.TopologyEdge(src="dev0", dst="dev1", edge_type="nvlink", status="DOWN")
        errors = _manifest(edges=(first, second)).validate()
        assert any("inconsistent" in error for error in errors)

    def test_duplicate_device_uuid_is_rejected(self):
        accelerators = (
            tp.AcceleratorRecord(device_id=0, sku="GA102", uuid="GPU-x"),
            tp.AcceleratorRecord(device_id=1, sku="GA102", uuid="GPU-x"),
        )
        errors = _manifest(accelerators=accelerators).validate()
        assert any("appears 2 times" in error for error in errors)


@pytest.mark.unit
class TestUnavailable:
    def test_unavailable_marker_requires_reason(self):
        with pytest.raises(ConfigError):
            tp.unavailable("")

    def test_probe_status_requires_reason(self):
        with pytest.raises(ConfigError):
            pb.ProbeOutcome(name="npu-smi", status="UNAVAILABLE")

    def test_p2p_unknown_requires_reason(self):
        with pytest.raises(ConfigError):
            pb.P2PCapability(src_uuid="a", dst_uuid="b", reason="")


@pytest.mark.unit
class TestDrift:
    def test_identity_only_change_keeps_results(self):
        left = _manifest()
        right = _manifest(host=tp.HostIdentity(
            node_id="n0", host_alias="host-a", cpu_arch="x86_64", os_release="22.04",
            kernel="6.5", scheduler_job_id="job-2",
        ))
        report = tp.diff_manifests(left, right)
        assert report["highest_level"] == "IDENTITY_ONLY"
        assert report["reuse_allowed"] is True

    def test_performance_drift_forbids_reuse(self):
        left = _manifest()
        right = _manifest(
            edges=(
                tp.TopologyEdge(
                    src="dev0", dst="dev1", edge_type="nvlink", status="DEGRADED",
                    confidence="DEGRADED", source_tool="fixture",
                ),
            )
        )
        report = tp.diff_manifests(left, right)
        assert report["highest_level"] == "PERFORMANCE_RELEVANT"
        assert report["reuse_allowed"] is False

    def test_branch_change_is_topology_class_change(self):
        left = _manifest()
        right = _manifest(
            scope=tp.ObservationScope(branch="cuda_nccl", node_scope="multi_node")
        )
        report = tp.diff_manifests(left, right)
        assert report["highest_level"] == "TOPOLOGY_CLASS_CHANGE"
        assert report["same_topology_class"] is False


@pytest.mark.unit
class TestDegradedPolicy:
    def test_default_deny_blocks_a_down_edge(self):
        manifest = _manifest(
            edges=(
                tp.TopologyEdge(src="dev0", dst="dev1", edge_type="nvlink", status="DOWN"),
            )
        )
        audit = tp.audit_degraded_edges(manifest, tp.DegradedLinkPolicy(action="DENY"))
        assert audit["ok"] is False
        assert audit["blocked"][0]["action"] == "DENY"

    def test_explicit_allow_requires_reason(self):
        with pytest.raises(ConfigError):
            tp.DegradedLinkPolicy(action="EXPLICIT_ALLOW").validate()

    def test_gate_refuses_unverified_topology(self):
        decision = tp.gate_for_collective_probe(_manifest())
        assert decision.ready is False
        assert any("data path" in blocker for blocker in decision.blockers)


@pytest.mark.unit
class TestPlacement:
    def _plan(self) -> pl.PlacementPlan:
        return pl.PlacementPlan(
            plan_id="p",
            world_size=2,
            node_count=1,
            entries=tuple(
                pl.RankPlacement(
                    global_rank=index,
                    local_rank=index,
                    node_rank=0,
                    coordinate=pl.ParallelCoordinate(tp=index),
                    planned_device_uuid=f"GPU-{index}",
                )
                for index in range(2)
            ),
        )

    def test_valid_plan_and_rank_table(self):
        plan = self._plan()
        assert plan.validate() == []
        table = pl.build_rank_table(plan)
        assert [row["global_rank"] for row in table] == [0, 1]
        assert pl.launcher_config(plan, "torchrun")["world_size"] == 2

    def test_plan_rejects_world_size_mismatch(self):
        plan = self._plan()
        bad = pl.PlacementPlan(
            plan_id="p", world_size=3, node_count=1, entries=plan.entries
        )
        assert any("entries" in error for error in bad.validate())

    def test_planned_actual_mismatch_is_detected(self):
        plan = self._plan()
        identities = (
            rk.RankIdentity(0, 0, 0, node_id="n0", pid=1, device_uuid="GPU-0"),
            rk.RankIdentity(1, 1, 0, node_id="n0", pid=2, device_uuid="GPU-9"),
        )
        report = pl.compare_planned_actual(plan, identities)
        assert report["ok"] is False
        assert report["mismatches"][0]["issue"] == "device mismatch"

    def test_alternative_placement_is_not_a_performance_claim(self):
        plan = self._plan()
        with pytest.raises(ConfigError):
            pl.PlacementSanityExpectation(
                baseline_edges=(), alternative_edges=(), expected_direction="slower_or_equal",
                is_performance_claim=True,
            ).validate()
        _, expectation = pl.alternative_placement(plan, swap_ranks=(0, 1), reason="sanity")
        assert expectation.expected_direction == "no_difference_expected"


@pytest.mark.unit
class TestPreflightFixtures:
    def _plan(self) -> pl.PlacementPlan:
        return pl.PlacementPlan(
            plan_id="p",
            world_size=2,
            node_count=1,
            entries=tuple(
                pl.RankPlacement(
                    global_rank=index,
                    local_rank=index,
                    node_rank=0,
                    coordinate=pl.ParallelCoordinate(tp=index),
                    planned_device_uuid=f"GPU-{index}",
                )
                for index in range(2)
            ),
        )

    def test_duplicate_device_is_rejected_before_init(self):
        plan = self._plan()
        result = pb.preflight_rank_mapping(
            plan,
            pb.make_duplicate_device_identities(plan),
            known_device_uuids=["GPU-0", "GPU-1"],
        )
        assert result.ok is False
        assert result.rejections[0].code == "DUPLICATE_DEVICE"
        assert result.rejections[0].as_dict()["detected_before_communicator_init"] is True

    def test_rank_gap_is_rejected(self):
        plan = self._plan()
        identities = (rk.RankIdentity(0, 0, 0, node_id="n0", pid=1, device_uuid="GPU-0"),)
        result = pb.preflight_rank_mapping(plan, identities)
        assert any(item.code == "RANK_GAP" for item in result.rejections)

    def test_group_order_mismatch_is_rejected(self):
        groups = (
            rk.GroupMembership("tp", "tp", (0, 1), "nccl", "2.27", 0, 60.0, creation_sequence=1),
            rk.GroupMembership("ep", "ep", (0, 1), "nccl", "2.27", 0, 60.0, creation_sequence=0),
        )
        result = pb.preflight_group_order(groups, planned_order=["tp", "ep"])
        assert result.ok is False
        assert result.rejections[0].code == "GROUP_ORDER_MISMATCH"


@pytest.mark.unit
class TestRankInvariants:
    def test_one_rank_per_device(self):
        identities = (
            rk.RankIdentity(0, 0, 0, node_id="n0", pid=1, device_uuid="GPU-0"),
            rk.RankIdentity(1, 1, 0, node_id="n0", pid=2, device_uuid="GPU-0"),
        )
        report = rk.validate_rank_identities(identities, expected_world_size=2)
        assert report["ok"] is False
        assert any("bound by ranks" in issue for issue in report["issues"])

    def test_run_identity_must_match_across_ranks(self):
        base = dict(
            run_id="r", experiment_id="E10-02", job_id="j", world_size=2, node_count=1,
            backend="nccl", backend_version="2.27.3", rank_epoch=0,
        )
        first = rk.RunIdentity(**base)
        second = rk.RunIdentity(**{**base, "backend_version": "2.26.0"})
        report = rk.compare_run_identity([first, second])
        assert report["ok"] is False

    def test_identity_rejects_vague_version(self):
        with pytest.raises(ConfigError):
            rk.RunIdentity(
                run_id="r", experiment_id="E10-02", job_id="j", world_size=2, node_count=1,
                backend="nccl", backend_version="latest",
            )
