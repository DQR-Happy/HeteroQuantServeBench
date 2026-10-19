"""Serial-worker ownership of one cached compile plan and per-request evidence.

Only forward is temporarily rebound; nn.Module/parameter identity and the eager
path are preserved. No global suppress-errors fallback and no CUDA Graph replay.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import time

from hqsb.core.optimization import fingerprint
from hqsb.core.runtime_plan import validate_runtime
from hqsb.backends.runtime_graph import describe_graph, analyze_graphs
from hqsb.observation.storage import atomic_json


class RuntimeSession:
    def __init__(self, model, plan, diagnostic=None):
        import torch

        self.model, self.plan = model, plan
        self.diagnostic = diagnostic
        self.views = []
        self.original = model.forward
        self.had_instance_forward = "forward" in model.__dict__
        self.known_modules = set(dict(model.named_modules()))
        self.graphs, self.graph_count = [], 0
        self.truncated = False
        self.active = None
        self.compiled = None
        if plan["mode"] != "eager":
            self.compiled = torch.compile(self.original, backend=self.backend,
                dynamic={"auto": None, "static": False, "dynamic": True}[plan["shape_policy"]],
                fullgraph=plan["fullgraph"])

    def backend(self, gm, inputs):
        self.graph_count += 1
        graph_id = f"g{self.graph_count}"
        start = time.monotonic()
        graph = None
        # One model can specialize many times. Recording is bounded independently
        # of execution; never substitute eager just because evidence is full.
        if len(self.graphs) < 16:
            graph = describe_graph(gm, self.known_modules, graph_id=graph_id,
                                   phase=self.active["phase"])
            graph.update(first_run_id=self.active["run_id"], invocations=0, status="compiling")
            self.graphs.append(graph)
        else:
            self.truncated = True
        try:
            if self.plan["mode"] == "dynamo_eager":
                fn = gm.forward
            else:
                from torch._inductor import compile as compile_graph
                if self.diagnostic == "inductor" and self.active.get("capture_dir"):
                    from tempfile import TemporaryDirectory
                    from hqsb.backends.compiler_evidence import collect
                    with TemporaryDirectory(prefix="hqsb-inductor-") as temporary:
                        fn = compile_graph(gm, inputs, options={"triton.cudagraphs": False,
                            "trace.enabled": True, "trace.debug_dir": temporary,
                            "fx_graph_cache": False, "trace.save_real_tensors": False})
                        if self.graph_count <= 4:
                            self.views.extend(collect(temporary, self.active["capture_dir"], graph_id,
                                                      self.active["phase"], self.active["run_id"]))
                else:
                    fn = compile_graph(gm, inputs, options={"triton.cudagraphs": False})
            if graph is not None:
                graph["status"] = "ready"
        except Exception as exc:
            if graph is not None:
                graph.update(status="failed", error=f"{type(exc).__name__}:{str(exc)[:1000]}")
            raise
        finally:
            elapsed = (time.monotonic() - start) * 1000
            self.active["backend_compile_ms"] += elapsed
            if graph is not None:
                graph["backend_compile_ms"] = elapsed

        def invoke(*args):
            self.active["graph_invocations"] += 1
            self.active["graph_ids"].add(graph_id)
            if graph is not None:
                graph["invocations"] += 1
            return fn(*args)
        return invoke

    @contextmanager
    def activate(self, request):
        import torch

        from torch._dynamo.utils import counters
        if self.active is not None:
            raise RuntimeError("L3_SESSION_ALREADY_ACTIVE")
        self.active = {"run_id": request.get("capture_id"), "phase": "unspecified",
                       "capture_dir": request.get("capture_dir"), "onnx_attempted": False,
                       "forward_calls": 0, "completed_forwards": 0, "graph_invocations": 0,
                       "graph_ids": set(), "graph_count_before": self.graph_count,
                       "backend_compile_ms": 0.0, "forward_host_ms": 0.0,
                       "shape_signatures": [], "errors": [],
                       "breaks_before": dict(counters["graph_break"])}

        @wraps(self.original)
        def forward(*args, **kwargs):
            a = self.active
            a["forward_calls"] += 1
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            signature = {"phase": a["phase"], "input_shape": list(ids.shape) if ids is not None else None,
                         "mask_shape": list(mask.shape) if mask is not None else None,
                         "kv_class": type(kwargs.get("past_key_values")).__name__}
            if len(a["shape_signatures"]) < 32 and signature not in a["shape_signatures"]:
                a["shape_signatures"].append(signature)
            start = time.monotonic()
            try:
                if self.diagnostic == "onnx" and not a["onnx_attempted"] and a.get("capture_dir"):
                    a["onnx_attempted"] = True
                    from hqsb.backends.onnx_capture import capture
                    self.views = [capture(self.model, self.original, ids, mask, a["capture_dir"])]
                    self.views[0]["first_run_id"] = a["run_id"]
                result = (self.compiled or self.original)(*args, **kwargs)
                a["completed_forwards"] += 1
                return result
            except Exception as exc:
                a["errors"].append(f"{type(exc).__name__}:{str(exc)[:1200]}")
                raise
            finally:
                a["forward_host_ms"] += (time.monotonic() - start) * 1000

        self.model.forward = forward
        try:
            # Per-worker scope; recompile-limit failures must not silently become
            # eager. Stance is available in the pinned Torch 2.8 runtime.
            with torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True):
                yield self
        finally:
            if self.had_instance_forward:
                self.model.forward = self.original
            else:
                del self.model.forward
            self.last = self.active
            self.active = None

    def summary(self):
        from torch._dynamo.utils import counters

        a = self.active or getattr(self, "last", {})
        new = self.graph_count - a.get("graph_count_before", self.graph_count)
        breaks = [{"reason": str(k)[:2000], "count": v - a.get("breaks_before", {}).get(k, 0)}
                  for k, v in counters["graph_break"].items() if v > a.get("breaks_before", {}).get(k, 0)]
        calls, invoked = a.get("completed_forwards", 0), a.get("graph_invocations", 0)
        return {"schema_version": "l3-v1", "plan": self.plan, "plan_sha256": fingerprint(self.plan),
                "status": "failed" if a.get("errors") else "reference" if self.plan["mode"] == "eager"
                else "invoked" if invoked and calls else "not_invoked",
                "lowering": "inductor" if self.plan["mode"] == "inductor" else "none",
                "forward_calls": a.get("forward_calls", 0), "completed_forwards": calls,
                "graph_invocations": invoked, "used_graph_ids": sorted(a.get("graph_ids", [])),
                "new_graphs": new, "session_graphs": self.graph_count,
                "backend_compile_ms": a.get("backend_compile_ms", 0),
                "forward_host_ms": a.get("forward_host_ms", 0),
                "timing_scope": "CPU host spans; backend time excludes Dynamo tracing; no GPU timing",
                "diagnostic": self.diagnostic,
                "sample_class": "diagnostic_capture" if self.diagnostic else "includes_compilation" if new else "no_new_backend_graph_observed",
                "shape_signatures": a.get("shape_signatures", []),
                "graph_break_events": breaks, "errors": a.get("errors", []),
                "cuda_graphs": False, "fallback_policy": "graph_break_eager_islands_allowed" if not self.plan["fullgraph"] else "error_on_graph_break",
                "coverage": "unknown:graph_invocations_are_not_full_forward_coverage"}

    def save(self, provider, request):
        summary = self.summary()
        doc = {**summary, "run_id": request.get("capture_id"), "epoch": request.get("expected_epoch"),
               "model_identity": provider.source_identity, "tokenizer_identity": provider.tokenizer_identity,
               "source_digest": provider.source_environment["source_digest"],
               "graphs": [dict(g) for g in self.graphs], "truncated": self.truncated or any(g["truncated"] for g in self.graphs),
               "limitations": ["Dynamo FX before lowering; fusion, device overlap and CUDA cache hits are not inferred.",
                               "Graphs may contain eager islands; zero new graph breaks does not prove full coverage.",
                               "Historical specializations are retained; used_graph_ids identifies this run.",
                               "No tensor values or addresses are saved; symbolic dimensions are preserved."]}
        # Bound the complete JSON, including source stacks and inherited graphs.
        doc["analysis"] = analyze_graphs(doc["graphs"])
        while len(json.dumps(doc).encode()) > 3 * 1024**2 and doc["graphs"]:
            largest = max(doc["graphs"], key=lambda g: len(g["nodes"]))
            if len(largest["nodes"]) <= 1:
                doc["graphs"] = doc["graphs"][:-1]
                continue
            largest["nodes"] = largest["nodes"][:len(largest["nodes"]) // 2]
            ids = {node["id"] for node in largest["nodes"]}
            largest["edges"] = [edge for edge in largest["edges"] if edge["source"] in ids and edge["target"] in ids]
            largest["truncated"] = True
            doc["truncated"] = True
            doc["analysis"] = analyze_graphs(doc["graphs"])
        if request.get("capture_dir"):
            atomic_json(Path(request["capture_dir"]) / "runtime-graph.json", doc)
            if self.diagnostic:
                from copy import deepcopy
                from shutil import copyfile
                a = self.active or self.last
                # Cached compiler views retain their origin. Copy their published
                # immutable source files to bind each run's own manifest.
                views = deepcopy(self.views)
                for view in views:
                    origin = view.get("first_run_id")
                    name = view.get("artifact")
                    dest = Path(request["capture_dir"]) / name if name else None
                    if dest and not dest.exists() and origin != a["run_id"]:
                        source = dest.parent.parent / origin / name
                        if source.is_file() and not source.is_symlink():
                            copyfile(source, dest)
                        else:
                            view.update(status="not_produced", reason="cached diagnostic source unavailable")
                extra = {"schema_version": "graph-views-v1", "run_id": request.get("capture_id"),
                         "epoch": request.get("expected_epoch"), "source_digest": doc["source_digest"],
                         "model_identity": doc["model_identity"], "diagnostic": True, "views": views,
                         "limitations": ["Diagnostic overhead included; not a performance benchmark.",
                                         "Representations have different granularity; counts are not directly comparable."]}
                while len(json.dumps(extra).encode()) > 3 * 1024**2:
                    candidates = [v for v in views if len(v.get("nodes", [])) > 1 or v.get("text")]
                    if not candidates:
                        break
                    largest = max(candidates, key=lambda v: len(json.dumps(v)))
                    if largest.get("text"):
                        largest["text"] = largest["text"][:len(largest["text"]) // 2]
                        largest["text_truncated"] = True
                    else:
                        largest["nodes"] = largest["nodes"][:len(largest["nodes"]) // 2]
                        kept = {n["id"] for n in largest["nodes"]}
                        largest["edges"] = [e for e in largest["edges"] if e["source"] in kept and e["target"] in kept]
                        largest["truncated"] = True
                atomic_json(Path(request["capture_dir"]) / "graph-views.json", extra)
        return summary


@contextmanager
def execution(provider, request):
    """Existing callers without runtime_plan retain the original eager path."""
    if request.get("runtime_plan") is None:
        yield None
        return
    plan = validate_runtime(request["runtime_plan"], kernel=provider.config.get("kernel_profile", "framework-native"),
        method=provider.precision_plan.method, kv_bits=provider.precision_plan.kv_bits,
        capture=request.get("capture_architecture", False), observation=request.get("observation_mode", "basic"))
    old = getattr(provider, "runtime_session", None)
    diagnostic = (request.get("graph_capture") or {}).get("kind")
    if diagnostic:
        from hqsb.core.graph_views import validate_capture
        validate_capture(request["graph_capture"], runtime=plan,
            native=provider.config.get("kernel_profile", "framework-native") == "framework-native"
            and provider.precision_plan.method == "none" and not provider.precision_plan.kv_bits,
            privileged=False, observation=request.get("observation_mode", "basic"),
            quality=bool(request.get("experiment_role")), tokens=request.get("max_output_tokens", 4))
    if old is None or old.plan != plan or old.diagnostic != diagnostic:
        if old is not None:
            import torch
            torch._dynamo.reset()  # This worker owns one model and serial execution.
        provider.runtime_session = RuntimeSession(provider.model, plan, diagnostic)
    with provider.runtime_session.activate(request) as session:
        provider.active_runtime_session = session
        try:
            yield session
        finally:
            provider.active_runtime_session = None


def set_phase(provider, phase):
    session = getattr(provider, "active_runtime_session", None)
    if session and session.active:
        session.active["phase"] = phase


def receipt(provider, request):
    session = getattr(provider, "active_runtime_session", None)
    return session.save(provider, request) if session else {
        "schema_version": "l3-v1", "plan": {"mode": "eager", "shape_policy": "auto", "fullgraph": False},
        "status": "reference", "lowering": "none", "cuda_graphs": False}
