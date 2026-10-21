"""One model-local binding lifetime and bounded, phase/site execution evidence."""
from types import MethodType
from pathlib import Path

from hqsb.core.kernel_profiles import profile
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json

PHASES = ("warmup", "prefill", "decode")


class KernelSession:
    def __init__(self, model, config):
        self.profile = profile(config.get("kernel_profile", "framework-native"))
        self.plan = {**self.profile, "profile_sha256": fingerprint(self.profile)}
        self.handles, self.sites, self.counts = [], [], []
        self.phase = 0
        self.native = None
        self.repeat_peak_bytes = self.repeat_total_bytes = 0
        self.binding = {"schema_version": "l2-binding-v1", "sites": self.sites,
                        "parameter_layout": "original_no_packing_no_weight_copies",
                        "state_dict_preserved": True, "tied_parameters_preserved": True}
        if not self.profile["operators"]:
            self.binding["adapter"] = "framework-native"
            self.binding_hash = fingerprint(self.binding)
            self.plan_hash = fingerprint(self.plan)
            return
        try:
            if any(v.startswith("triton-") for v in self.profile["operators"].values()):
                from importlib.metadata import version
                from hqsb.observation.storage import sha256
                implementation = Path(__file__).parent / "model_kernels/triton_attention.py"
                self.plan["triton_implementation"] = {"version": version("triton"), "source_sha256": sha256(implementation)}
            if any(not variant.startswith(("torch-", "triton-")) for variant in self.profile["operators"].values()):
                manifest = config.get("kernel_manifest")
                if not manifest:
                    raise ValueError("KERNEL_NATIVE_MANIFEST_REQUIRED")
                from hqsb.integration.torch_ops import register_model_ops

                self.native = register_model_ops(manifest)
                self.plan["native_artifact"] = self.native.identity
            from hqsb.backends.model_kernels.qwen3 import bind

            bind(model, self)
            self.binding_hash = fingerprint(self.binding)
            self.plan_hash = fingerprint(self.plan)
        except Exception:
            self.close()
            raise

    def install(self, module, forward):
        # Preserve whether the original forward lived on the instance or class.
        self.handles.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
        module.forward = MethodType(forward, module)

    def site(self, path, op, variant, *, covers=()):
        index = len(self.sites)
        self.sites.append({"site_id": path + ":" + op, "module_path": path,
                           "op": op, "variant": variant, "covers": list(covers)})
        self.counts.append([0] * len(PHASES))
        return index

    def hit(self, site):
        self.counts[site][self.phase] += 1

    def set_phase(self, phase):
        self.phase = PHASES.index(phase)

    def reset(self):
        self.repeat_peak_bytes = self.repeat_total_bytes = 0
        for counts in self.counts:
            for i in range(len(counts)):
                counts[i] = 0

    def summary(self):
        rows = [{**site, "calls": dict(zip(PHASES, counts))} for site, counts in zip(self.sites, self.counts)]
        exercised_groups = {row.get("coverage_group") for row in rows
                            if row["calls"]["prefill"] + row["calls"]["decode"]}
        missing = [row["site_id"] for row in rows
                   if not row["calls"]["prefill"] + row["calls"]["decode"]
                   and not (row.get("coverage_group") and row["coverage_group"] in exercised_groups)]
        return {"schema_version": "l2-summary-v1", "profile_id": self.profile["profile_id"],
                "profile_sha256": self.plan["profile_sha256"], "binding_manifest_sha256": self.binding_hash,
                "kernel_plan_sha256": self.plan_hash,
                "status": "invoked" if rows and not missing else "not_exercised" if rows else "reference",
                "sites": rows, "not_exercised": missing, "fallbacks": [],
                "workspace_bytes": 0, "weight_packing_bytes": 0,
                "workspace_scope": "caller_provided_not_vendor_internal_allocations",
                "linear_plans": self.native.plans() if self.native else [],
                "linear_plans_scope": "resident_worker_cache_including_warmup_and_prior_requests",
                "linear_plan_cache_limit": 256,
                "gqa_repeat_peak_logical_bytes": self.repeat_peak_bytes,
                "gqa_repeat_cumulative_logical_bytes": self.repeat_total_bytes,
                "gqa_repeat_bytes_scope": "tensor_shape_accounting_not_physical_dram_traffic",
                "proof_scope": "successful_logical_site_dispatch_not_gpu_activity_or_instruction_proof",
                "gpu_observed": "not_collected_here", "quality": "requires_separate_quality_result"}

    def save(self, directory):
        if directory:
            root = Path(directory)
            atomic_json(root / "kernel-plan.json", self.plan)
            atomic_json(root / "kernel-bindings.json", self.binding)
            atomic_json(root / "kernel-summary.json", self.summary())

    def close(self):
        for module, owned, old in reversed(self.handles):
            if owned:
                module.forward = old
            else:
                del module.forward
        self.handles.clear()
        if self.native:
            self.native.close()
