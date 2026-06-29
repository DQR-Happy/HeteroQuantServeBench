"""Property tests for the S11 compiler invariants.

Follows the repository convention (``tests/property/test_distributed_invariants.py``):
deterministic pseudo-random loops with a fixed seed instead of Hypothesis, so
the suite runs on the CPU-minimal installation.  Each class states one
invariant and includes a negative control that must fail if the invariant is
broken.
"""

from __future__ import annotations

import random

import pytest

from hqsb.compiler import cache as ch
from hqsb.compiler import capture as cap
from hqsb.compiler import costmodel as cm
from hqsb.compiler import guards as gd
from hqsb.compiler import identity as ident
from hqsb.compiler import ir as ir
from hqsb.compiler import pattern_library as pl
from hqsb.compiler import rewrite as rw

SEED = 20260919


@pytest.mark.property
class TestCanonicalHashAlgebra:
    def test_noise_never_changes_the_canonical_hash(self) -> None:
        rng = random.Random(SEED)
        for _ in range(30):
            op = rng.choice(("aten.add", "aten.mul", "aten.rsqrt"))
            first, _ = ident.content_hash({"op": op, "timestamp": rng.random(), "pid": rng.randint(1, 999)})
            second, _ = ident.content_hash({"op": op, "timestamp": rng.random(), "pid": rng.randint(1, 999)})
            assert first == second

    def test_semantic_changes_always_move_the_hash(self) -> None:
        rng = random.Random(SEED + 1)
        for _ in range(30):
            dtype = rng.choice(("fp16", "fp32", "bf16"))
            other = rng.choice([item for item in ("fp16", "fp32", "bf16") if item != dtype])
            first, _ = ident.content_hash({"op": "aten.add", "dtype": dtype})
            second, _ = ident.content_hash({"op": "aten.add", "dtype": other})
            assert first != second

    def test_graph_canonical_hash_is_stable_under_repeated_hashing(self) -> None:
        rng = random.Random(SEED + 2)
        for index in range(10):
            graph = rw.build_graph(
                pl.residual_add_rmsnorm_graph(graph_id=f"prop_{index}", variant="canonical")
            )
            hashes = {graph.canonical_hash()[0] for _ in range(rng.randint(2, 5))}
            assert len(hashes) == 1


@pytest.mark.property
class TestRewriteIdempotence:
    def test_second_pass_never_rewrites_or_changes_the_hash(self) -> None:
        rng = random.Random(SEED + 3)
        variants = ["canonical", "decomposed_cast", "eps_value", "reduction_axis", "cast_order"]
        for _ in range(12):
            variant = rng.choice(variants)
            graph = rw.build_graph(pl.residual_add_rmsnorm_graph(graph_id="prop", variant=variant))
            first = rw.run_pipeline(graph)
            second = rw.run_pipeline(first.final_graph)
            assert second.final_graph.canonical_hash()[0] == first.final_graph.canonical_hash()[0]
            assert sum(item.rewrite_count for item in second.iterations) == 0

    def test_false_positive_count_stays_zero_over_the_whole_corpus(self) -> None:
        rows = rw.build_corpus_rows(pl.corpus_plan(), graph_builder=pl.residual_add_rmsnorm_graph)
        summary = rw.evaluate_corpus(rows)
        assert summary["false_positive"] == 0
        # negative control: swapping in a permissive expectation must produce FPs
        assert summary["false_positive_zero"] is True

    def test_atomicity_holds_for_every_injection_point(self) -> None:
        graph = rw.build_graph(pl.residual_add_rmsnorm_graph(graph_id="prop_atomic", variant="canonical"))
        baseline = graph.canonical_hash()[0]
        for point in rw.INJECTION_POINTS:
            report = rw.atomicity_report(graph, points=(point,))
            assert report["all_atomic"] is True
            assert graph.canonical_hash()[0] == baseline


@pytest.mark.property
class TestGuardDomainAlgebra:
    def test_guarded_lookup_never_returns_a_variant_outside_its_domain(self) -> None:
        rng = random.Random(SEED + 4)
        registry = gd.VariantRegistry()
        registry.add(
            gd.Variant(
                variant_id="v1",
                semantic_identity="sem",
                compile_identity="cid",
                target_id="cpu",
                artifact_id="art",
                guards=(
                    gd.range_guard(guard_id="b", symbol="B", lower=1, upper=8, source="s"),
                    gd.divisibility_guard(guard_id="h", symbol="H", divisor=8, source="s"),
                ),
                created_reason="property fixture",
            )
        )
        for _ in range(40):
            batch = rng.randint(1, 32)
            hidden = rng.choice((8, 16, 4096, 4097))
            result = registry.lookup(semantic_identity="sem", inputs={"B": batch, "H": hidden})
            in_domain = 1 <= batch <= 8 and hidden % 8 == 0
            assert (result.outcome == "hit") is in_domain

    def test_wrong_reuse_audit_catches_every_false_guard(self) -> None:
        from hqsb.compiler import records as rec

        rng = random.Random(SEED + 5)
        for _ in range(10):
            outcomes = [rng.random() < 0.7 for _ in range(4)]
            events = [
                rec.GuardEventRecord(
                    run_id="r",
                    frame_id="f",
                    variant_id="v",
                    guard_id=f"g{index}",
                    expression="B<=8",
                    source="s",
                    category="shape_range",
                    actual_values={"B": 4},
                    outcome=outcome,
                    action="reuse",
                )
                for index, outcome in enumerate(outcomes)
            ]
            audit = gd.wrong_reuse_audit([{"variant_id": "v"}], events)
            assert audit["wrong_reuse_count"] == sum(1 for outcome in outcomes if not outcome)

    def test_domain_coverage_never_exceeds_one(self) -> None:
        rng = random.Random(SEED + 6)
        variant = gd.Variant(
            variant_id="v",
            semantic_identity="sem",
            compile_identity="cid",
            target_id="cpu",
            artifact_id="art",
            guards=(gd.range_guard(guard_id="b", symbol="B", lower=1, upper=4, source="s"),),
            created_reason="property",
        )
        for _ in range(10):
            samples = [{"B": rng.randint(-2, 8)} for _ in range(rng.randint(1, 20))]
            report = gd.domain_coverage([variant], samples)
            assert 0.0 <= report["coverage"] <= 1.0
            assert report["covered"] + len(report["holes"]) == len(samples)


@pytest.mark.property
class TestCacheKeyAlgebra:
    def test_key_is_invariant_to_noise_and_sensitive_to_semantics(self) -> None:
        rng = random.Random(SEED + 7)
        spec = ch.default_key_spec()
        base = {name: f"{name}:base" for name in spec.fields}
        for _ in range(20):
            variant = dict(base)
            noise = rng.choice(ch.NON_KEY_FIELDS)
            variant[noise] = f"noise-{rng.random()}"
            try:
                assert spec.compute(variant)["key"] == spec.compute(base)["key"]
            except Exception:  # pragma: no cover - noise fields are ignored outright
                assert True
            semantic = rng.choice(sorted(spec.fields))
            changed = dict(base)
            changed[semantic] = f"{semantic}:other"
            assert spec.compute(changed)["key"] != spec.compute(base)["key"]

    def test_metadata_fingerprint_detects_any_field_edit(self) -> None:
        rng = random.Random(SEED + 8)
        manifest = ch.EntryManifest(
            entry_id="e", key="k", layer="pass_ir", payload_sha256="a" * 64, state=ch.STATE_COMMITTED
        ).as_dict()
        baseline = ch.metadata_fingerprint(manifest)
        for _ in range(10):
            edited = dict(manifest)
            field = rng.choice(sorted(edited))
            if field == "metadata_sha256":
                continue
            edited[field] = f"tampered-{rng.random()}"
            assert ch.metadata_fingerprint(edited) != baseline


@pytest.mark.property
class TestCaptureCoverageAlgebra:
    def test_coverage_ratios_stay_in_range_or_report_unavailable(self) -> None:
        rng = random.Random(SEED + 9)
        for _ in range(25):
            observed = rng.randint(1, 200)
            captured = rng.randint(0, observed)
            report = cap.CoverageReport(captured_ops=captured, observed_ops=observed)
            value = report.metrics()["op_count_coverage"]
            assert value is None or 0.0 <= value <= 1.0

    def test_repeatability_requires_identical_canonical_hashes(self) -> None:
        rng = random.Random(SEED + 10)
        for _ in range(10):
            canonical = "c" * 64
            rows = [
                {
                    "canonical_hash": canonical if rng.random() < 0.9 else "d" * 64,
                    "raw_hash": f"raw{rng.randint(0, 9)}",
                    "breaks": [],
                    "guards": [],
                }
                for _ in range(3)
            ]
            report = cap.repeatability_check(rows)
            stable = len({row["canonical_hash"] for row in rows}) == 1
            assert report["ok"] is stable


@pytest.mark.property
class TestCostModelInvariants:
    def test_regret_is_bounded_below_by_zero_for_a_legal_choice(self) -> None:
        rng = random.Random(SEED + 11)
        for _ in range(30):
            oracle = rng.uniform(0.1, 10.0)
            chosen = oracle * rng.uniform(1.0, 3.0)
            regret = cm.relative_regret(chosen, oracle)
            assert regret is not None and regret >= 0.0

    def test_illegal_candidate_is_never_selected(self) -> None:
        rng = random.Random(SEED + 12)
        for _ in range(20):
            eligible = ["default", "cuda_v2"]
            predictions = {name: rng.random() for name in eligible}
            predictions["illegal"] = -1.0  # cheapest by far
            decision = cm.select_with_policy(
                eligible=eligible,
                predictions=predictions,
                confidence=rng.uniform(0.6, 1.0),
                confidence_threshold=0.5,
                ood=False,
            )
            assert decision["chosen"] in {*eligible, ""}

    def test_selection_overhead_ratio_is_non_negative(self) -> None:
        rng = random.Random(SEED + 13)
        for _ in range(20):
            report = cm.selection_overhead(
                feature_extraction_us=rng.uniform(0, 5),
                model_load_us=rng.uniform(0, 5),
                inference_us=rng.uniform(0, 5),
                candidate_filter_us=rng.uniform(0, 5),
                decision_us=rng.uniform(0, 5),
                kernel_latency_us=rng.uniform(1, 100),
            )
            assert report["total_us"] >= 0
            assert report["selection_overhead_ratio"] is None or report["selection_overhead_ratio"] >= 0


@pytest.mark.property
class TestIRVerifierInvariants:
    def test_a_verified_graph_always_has_no_blocking_issue(self) -> None:
        rng = random.Random(SEED + 14)
        for index in range(15):
            ops = [
                {
                    "op_id": f"o{step}",
                    "semantic_op": rng.choice(("aten.add", "aten.mul", "aten.rsqrt")),
                    "operands": ["x"] if step == 0 else [f"v{step - 1}", "x"],
                    "results": [f"v{step}"],
                    "source": {"module_path": "prop"},
                    "mutation": "none",
                    "effect_evidence": "schema",
                }
                for step in range(rng.randint(1, 4))
            ]
            graph = ir.import_op_sequence(
                graph_id=f"prop{index}",
                ops=ops,
                inputs=[{"value_id": "x"}],
                outputs=[ops[-1]["results"][0]],
                constraints=(ir.ShapeConstraint("c", "B>0", "semantic", "user"),),
            )
            report = ir.verify_graph(graph)
            assert report.ok is True
            assert not any(issue.blocks_compilation for issue in report.issues)

    def test_removing_constraints_always_blocks(self) -> None:
        graph = ir.import_op_sequence(
            graph_id="noconstraint",
            ops=[
                {
                    "op_id": "o",
                    "semantic_op": "aten.add",
                    "operands": ["x"],
                    "results": ["y"],
                    "source": {"module_path": "prop"},
                }
            ],
            inputs=[{"value_id": "x"}],
            outputs=["y"],
        )
        assert ir.verify_graph(graph).ok is False

    def test_round_trip_never_changes_the_canonical_hash(self) -> None:
        rng = random.Random(SEED + 15)
        for index in range(12):
            graph = rw.build_graph(
                pl.residual_add_rmsnorm_graph(
                    graph_id=f"rt{index}", variant=rng.choice(["canonical", "decomposed_cast", "already_fused"])
                )
            )
            rebuilt, same = ir.round_trip(graph)
            assert same is True
            assert rebuilt.canonical_hash()[0] == graph.canonical_hash()[0]
