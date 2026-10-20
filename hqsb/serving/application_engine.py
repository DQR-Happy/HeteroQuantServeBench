"""L10 bounded service workload over narrow endpoint adapters. No HTTP/torch here."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from hqsb.core.optimization import fingerprint
from hqsb.serving.application_cache import ResultCache, result_key


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    i = int(p)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (p - i)


class ApplicationEngine:
    """One owner for routing occupancy, exact cache and managed replica lifetimes.

    Endpoint adapter: snapshot(), load(key), unload(epoch,key), prepare(input,policy),
    generate(payload,key,deadline,cancel), clear_prefix(), acquire(owner), release().
    All device lifecycle waits belong to that adapter and return cleanup ack.
    """

    def __init__(self, plan, endpoints, cancelled, record):
        self.plan, self.endpoints = plan, endpoints
        self.cancelled, self.record = cancelled, record
        self.lock = threading.Condition(threading.RLock())
        self.events = []
        self.owned = {}
        self.snapshots = {}
        self.prefix_scopes = set()
        self.busy = {key: False for key in endpoints}
        self.activating = set()
        self.last_used = {key: time.monotonic() for key in endpoints}
        self.failures = {key: 0 for key in endpoints}
        self.cooldown = {key: 0 for key in endpoints}
        self.sequence = 0
        self.started = time.monotonic()
        self.documents = []
        self.current_document = None
        self.current_policy = None

    def event(self, kind, **detail):
        with self.lock:
            doc = {
                "kind": kind,
                "offset_ms": (time.monotonic() - self.started) * 1000,
                **detail,
            }
            self.events.append(doc)
            self.record(doc)

    def ensure(self, target, variant):
        endpoint = self.endpoints[target.id]
        snap = endpoint.snapshot()
        if snap["state"] != "ready":
            if target.lifecycle != "managed":
                raise RuntimeError("L10_TARGET_NOT_READY: " + target.id)
            with self.lock:
                key = f"{variant}-load-{target.id}-{self.sequence}"
                self.sequence += 1
            self.event("replica.loading", target=target.id)
            loaded = endpoint.load(key)
            with self.lock:
                self.owned[target.id] = {
                    "load_run_id": loaded["id"],
                    "epoch": loaded["epoch"],
                }
            snap = endpoint.snapshot()
            self.event("replica.loaded", target=target.id, **self.owned[target.id])
        if not snap.get("identity") or not snap.get("epoch"):
            raise RuntimeError("L10_VERIFIED_RUNTIME_IDENTITY_REQUIRED")
        with self.lock:
            if self.current_policy and self.current_policy.routing != "length_model":
                signature = lambda d: fingerprint(
                    {
                        k: v
                        for k, v in d["identity"].items()
                        if k != "worker_source_digest"
                    }
                )
                if any(
                    signature(d) != signature(snap) for d in self.snapshots.values()
                ):
                    raise ValueError(
                        "L10_LOAD_BALANCING_REQUIRES_IDENTICAL_MODELS_USE_LENGTH_MODEL_FOR_MODEL_ROUTING"
                    )
            self.snapshots[target.id] = snap
        if (
            self.current_policy
            and self.current_policy.prefix_reuse
            and target.id not in self.prefix_scopes
        ):
            if not snap.get("prefix_supported"):
                raise ValueError("L10_PREFIX_UNSUPPORTED_BY_RESIDENT_MODEL")
            scope = endpoint.begin_prefix()
            self.prefix_scopes.add(target.id)
            self.event("prefix_scope.begin", target=target.id, receipt=scope)
        return snap

    def end_prefix(self, key):
        if key in self.prefix_scopes:
            scope = self.endpoints[key].end_prefix()
            self.prefix_scopes.remove(key)
            self.event("prefix_scope.restored", target=key, receipt=scope)

    def unload(self, key, reason):
        own = self.owned[key]
        self.end_prefix(key)
        self.event("replica.unloading", target=key, reason=reason, **own)
        self.endpoints[key].unload(
            own["epoch"], f"unload-{self.sequence}", own["load_run_id"]
        )
        with self.lock:
            self.sequence += 1
            del self.owned[key]
            self.snapshots.pop(key, None)
        self.event("replica.unloaded", target=key, reason=reason)

    def tick(self, policy):
        if not policy.autoscale:
            return
        for key in list(self.owned):
            with self.lock:
                if len(self.snapshots) <= policy.min_replicas:
                    break
                if (
                    self.busy[key]
                    or (time.monotonic() - self.last_used[key]) * 1000
                    < policy.idle_unload_ms
                ):
                    continue
                self.busy[key] = True
            try:
                self.unload(key, "idle_scale_down")
            finally:
                with self.lock:
                    self.busy[key] = False
                    self.lock.notify_all()

    def choose(self, item, policy, cache, deadline):
        desired = None
        if policy.routing == "length_model":
            text = item.prompt or "".join(m.content for m in item.messages)
            desired = "small" if len(text) <= policy.short_input_chars else "large"
        eligible = [
            t for t in self.plan.targets if desired is None or t.role == desired
        ]
        while True:
            selected = None
            with self.lock:
                if all(
                    getattr(self.endpoints[t.id], "quarantined", False)
                    for t in eligible
                ):
                    raise RuntimeError("L10_NO_CONFIRMED_TARGET_CHECK_CHILD_CLEANUP")
                if self.cancelled() or time.monotonic() >= deadline:
                    raise TimeoutError("L10_ADMISSION_CANCELLED_OR_DEADLINE")
                available = [
                    t
                    for t in eligible
                    if not self.busy[t.id]
                    and self.cooldown[t.id] <= time.monotonic()
                    and not getattr(self.endpoints[t.id], "quarantined", False)
                ]
                # Affinity is exact result-cache affinity, not predicted KV hits.
                if policy.routing == "cache_affinity":
                    available.sort(
                        key=lambda t: (
                            not (
                                t.id in self.snapshots
                                and cache.contains(
                                    result_key(
                                        item,
                                        policy,
                                        self.snapshots[t.id],
                                        self.plan.fixed_output_length,
                                    )
                                )
                            )
                        )
                    )
                elif policy.routing == "round_robin":
                    available.sort(
                        key=lambda t: (
                            (self.plan.targets.index(t) - self.sequence)
                            % len(self.plan.targets)
                        )
                    )
                else:
                    available.sort(key=lambda t: self.last_used[t.id])
                for target in available:
                    cold = target.id not in self.snapshots
                    if (
                        cold
                        and len(set(self.snapshots) | self.activating)
                        >= policy.max_replicas
                        and policy.autoscale
                    ):
                        continue
                    self.busy[target.id] = True
                    if cold:
                        self.activating.add(target.id)
                    self.sequence += 1
                    selected = target
                    break
                if selected is None:
                    self.lock.wait(
                        timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                    )
            if selected is not None:
                try:
                    return selected, self.ensure(selected, "scale-up")
                except Exception:
                    with self.lock:
                        self.busy[selected.id] = False
                        self.cooldown[selected.id] = time.monotonic() + 2
                        self.lock.notify_all()
                    raise
                finally:
                    with self.lock:
                        self.activating.discard(selected.id)

    def request(self, item, policy, cache, variant, origin):
        arrival = origin + item.arrival_ms / 1000
        deadline = min(
            arrival + self.plan.request_deadline_s, self.started + self.plan.timeout_s
        )
        target = None
        row = {
            "id": item.id,
            "tenant": item.tenant,
            "arrival_ms": item.arrival_ms,
            "status": "failed",
            "backend_invoked": False,
            "cache": "disabled",
        }
        try:
            target, snapshot = self.choose(item, policy, cache, deadline)
            row.update(
                target=target.id,
                model_identity=snapshot["identity"],
                epoch=snapshot["epoch"],
                queue_ms=max(0, (time.monotonic() - arrival) * 1000),
            )
            # Refresh epoch even on hits: model reload must invalidate results.
            fresh = self.endpoints[target.id].snapshot()
            if (
                fresh["epoch"] != snapshot["epoch"]
                or fresh["identity"] != snapshot["identity"]
            ):
                raise RuntimeError("L10_TARGET_EPOCH_CHANGED")
            key = result_key(item, policy, snapshot, self.plan.fixed_output_length)
            value = cache.get(key) if policy.result_cache else None
            if value is not None:
                row.update(
                    value,
                    cache="hit",
                    backend_invoked=False,
                    backend_first_token_ms=None,
                    backend_generation_ms=None,
                )
            else:
                endpoint = self.endpoints[target.id]
                prepared = endpoint.prepare(item.model_dump(), policy.model_dump())
                row["input"] = {k: v for k, v in prepared.items() if k != "payload"}
                if policy.prefix_reuse and not snapshot.get("prefix_supported"):
                    raise ValueError("L10_PREFIX_UNSUPPORTED_BY_RESIDENT_MODEL")
                payload = {
                    **prepared["payload"],
                    "expected_epoch": snapshot["epoch"],
                    "max_output_tokens": prepared["effective_output_tokens"],
                    "observation_mode": "basic",
                    "fixed_output_length": self.plan.fixed_output_length,
                    "save_input": False,
                }
                if policy.prefix_reuse:
                    payload["memory_plan"] = {
                        "mode": "dynamic",
                        "prefix_reuse": True,
                        "prefix_namespace": "l10-"
                        + fingerprint([variant, item.tenant])[:24],
                    }
                row["backend_invoked"] = True
                result = endpoint.generate(
                    payload, f"{variant}-{item.id}", deadline, self.cancelled
                )
                metrics = result["metrics"]
                memory = (metrics.get("optimization_receipt") or {}).get(
                    "memory_receipt"
                ) or {}
                projected = {
                    k: metrics.get(k)
                    for k in (
                        "output_tokens_per_s",
                        "decode_tail_tokens_per_s",
                        "runtime_first_token_ms",
                        "runtime_e2e_ms",
                        "memory",
                    )
                }
                projected["memory_mechanism"] = {
                    k: memory.get(k)
                    for k in (
                        "mode",
                        "prefix",
                        "transfers",
                        "forward_calls",
                        "processed_input_positions",
                    )
                }
                row.update(
                    backend_run_id=result["id"],
                    output=result["output"],
                    token_ids=metrics.get("generated_token_ids", []),
                    output_tokens=metrics.get("output_tokens"),
                    input_tokens=metrics.get("input_tokens"),
                    backend_first_token_ms=metrics.get("runtime_first_token_ms"),
                    backend_generation_ms=metrics.get("runtime_e2e_ms"),
                    backend_metrics=projected,
                    cache="miss" if policy.result_cache else "disabled",
                )
                if not row["token_ids"] or row["output_tokens"] != len(
                    row["token_ids"]
                ):
                    raise RuntimeError("L10_ACTUAL_TOKEN_EVIDENCE_REQUIRED")
                if policy.result_cache:
                    cache.put(
                        key,
                        {
                            k: row[k]
                            for k in (
                                "backend_run_id",
                                "output",
                                "token_ids",
                                "output_tokens",
                                "input_tokens",
                                "input",
                            )
                        },
                    )
            row["status"] = "completed"
            self.failures[target.id] = 0
        except Exception as exc:
            row["error"] = str(exc)[:1400]
            if target and row["backend_invoked"]:
                with self.lock:
                    self.failures[target.id] += 1
                    if self.failures[target.id] >= 2:
                        self.cooldown[target.id] = time.monotonic() + 2
                        self.event("circuit.open", target=target.id, cooldown_s=2)
        finally:
            row["gateway_e2e_ms"] = max(0, (time.monotonic() - arrival) * 1000)
            row["deadline_met"] = (
                time.monotonic() <= deadline and row["status"] == "completed"
            )
            with self.lock:
                self.current_document["requests"].append(row)
            if target:
                with self.lock:
                    self.busy[target.id] = False
                    self.last_used[target.id] = time.monotonic()
                    self.lock.notify_all()
            self.event(
                "request.finished",
                id=item.id,
                status=row["status"],
                target=row.get("target"),
                cache=row["cache"],
            )
        return row

    def run(self, policy, variant):
        span_start = time.monotonic()
        self.current_policy = policy
        document = {
            "name": variant,
            "policy": policy.model_dump(),
            "requests": [],
            "cache": {},
            "metrics": {},
        }
        self.documents.append(document)
        self.current_document = document
        cache = ResultCache(policy)
        self.snapshots.clear()
        self.failures = {key: 0 for key in self.endpoints}
        self.cooldown = {key: 0 for key in self.endpoints}
        for target in self.plan.targets:
            if (
                target.lifecycle == "existing"
                or not policy.autoscale
                or len(self.snapshots) < policy.min_replicas
            ):
                self.ensure(target, variant)
        # ensure() installs a private, cold L6 store and preserves the caller's.
        origin, futures = time.monotonic(), []
        try:
            with ThreadPoolExecutor(
                max_workers=self.plan.concurrency, thread_name_prefix="l10-request"
            ) as executor:
                for item in sorted(self.plan.requests, key=lambda r: r.arrival_ms):
                    while time.monotonic() < origin + item.arrival_ms / 1000:
                        if (
                            self.cancelled()
                            or time.monotonic() - self.started > self.plan.timeout_s
                        ):
                            raise TimeoutError("L10_WORKLOAD_CANCELLED_OR_TIMEOUT")
                        self.tick(policy)
                        time.sleep(0.02)
                    futures.append(
                        executor.submit(
                            self.request, item, policy, cache, variant, origin
                        )
                    )
                rows = [f.result() for f in futures]
            elapsed = (time.monotonic() - origin) * 1000
            good = [r for r in rows if r["status"] == "completed"]
            served = sum(r["output_tokens"] for r in good)
            invoked = [r for r in good if r["backend_invoked"]]
            peaks = [
                (r.get("backend_metrics", {}).get("memory") or {}).get(
                    "peak_allocated_bytes"
                )
                for r in invoked
            ]
            document.update(
                requests=rows,
                cache=cache.summary(),
                metrics={
                    "startup_ms": (origin - span_start) * 1000,
                    "workload_ms": elapsed,
                    "requests": len(rows),
                    "completed": len(good),
                    "failed": len(rows) - len(good),
                    "served_output_tokens": served,
                    "computed_output_tokens": sum(
                        r["output_tokens"] for r in good if r["backend_invoked"]
                    ),
                    "backend_invocations": sum(r["backend_invoked"] for r in rows),
                    "gateway_served_token_s": served * 1000 / elapsed
                    if elapsed
                    else None,
                    "request_s": len(good) * 1000 / elapsed if elapsed else None,
                    "p50_e2e_ms": percentile([r["gateway_e2e_ms"] for r in good], 0.5),
                    "p95_e2e_ms": percentile([r["gateway_e2e_ms"] for r in good], 0.95),
                    "p99_e2e_ms": percentile([r["gateway_e2e_ms"] for r in good], 0.99),
                    "peak_observed_allocator_bytes": max(peaks)
                    if peaks and all(x is not None for x in peaks)
                    else None,
                    "submitted_input_tokens": sum(
                        r["input"]["input_tokens_after"] for r in invoked
                    ),
                    "logical_input_tokens_before": sum(
                        r["input"]["input_tokens_before"] for r in good
                    ),
                    "logical_input_tokens_after": sum(
                        r["input"]["input_tokens_after"] for r in good
                    ),
                    "prefix_reused_tokens": sum(
                        (
                            (
                                r.get("backend_metrics", {}).get("memory_mechanism")
                                or {}
                            ).get("prefix")
                            or {}
                        ).get("reused_tokens", 0)
                        for r in invoked
                    ),
                    "deadline_met": sum(r["deadline_met"] for r in rows),
                },
            )
            return document
        finally:
            for key in list(self.prefix_scopes):
                self.end_prefix(key)
            if not document["metrics"]:
                rows = document["requests"]
                document.update(
                    cache=cache.summary(),
                    metrics={
                        "workload_ms": (time.monotonic() - origin) * 1000,
                        "requests": len(self.plan.requests),
                        "completed": sum(r["status"] == "completed" for r in rows),
                        "failed": sum(r["status"] != "completed" for r in rows),
                        "not_observed": len(self.plan.requests) - len(rows),
                    },
                )
            for key in list(self.owned):
                self.unload(key, "variant_drain")
            document["variant_total_ms"] = (time.monotonic() - span_start) * 1000


def compare_variants(a, b):
    left = {r["id"]: r for r in a["requests"]}
    pairs = []
    for r in b["requests"]:
        old = left.get(r["id"], {})
        pairs.append(
            {
                "id": r["id"],
                "complete": old.get("status") == r["status"] == "completed",
                "token_ids_exact": bool(old.get("token_ids"))
                and old.get("token_ids") == r.get("token_ids"),
                "same_effective_input": old.get("input", {}).get(
                    "effective_input_sha256"
                )
                == r.get("input", {}).get("effective_input_sha256"),
                "same_model": old.get("model_identity") == r.get("model_identity"),
                "output_token_delta": (r.get("output_tokens") or 0)
                - (old.get("output_tokens") or 0),
                "e2e_delta_ms": r["gateway_e2e_ms"] - old.get("gateway_e2e_ms", 0),
            }
        )
    return {
        "pairs": pairs,
        "all_tokens_exact": bool(pairs)
        and len(pairs) == len(left)
        and all(r["complete"] and r["token_ids_exact"] for r in pairs),
        "metric_deltas": {
            k: b["metrics"][k] - a["metrics"][k]
            for k in a["metrics"]
            if k in b["metrics"]
            and a["metrics"][k] is not None
            and b["metrics"][k] is not None
        },
        "quality_scope": "fixed-workload output agreement only; compression, budget and model changes need independent task-quality evaluation",
    }
