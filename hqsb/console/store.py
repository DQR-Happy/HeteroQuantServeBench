"""Durable run snapshots and ordered events, independent of transport lifetime."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

TERMINAL = frozenset({"completed", "failed", "cancelled", "timed_out", "interrupted"})

# Project away detailed payloads in SQLite, before allocating Python objects.
# Full inputs, tensors, operator rows and manifests remain available via get().
LIST_DETAIL_PATHS = (
    "$.metrics.experiment_bundle.artifacts",
    "$.result",
    "$.config.messages",
    "$.config.prompt",
    "$.config.requests",
    "$.config.external_requests",
    "$.config.actual_runtime.parameter_inventory",
    "$.config.actual_runtime.source_environment.source_files",
    "$.metrics.observation",
    "$.metrics.kv_inventory",
    "$.metrics.generated_token_ids",
    "$.metrics.parallel_receipt.ranks",
    "$.metrics.parallel_receipt.requests",
    "$.metrics.parallel_receipt.participant_receipts",
    "$.metrics.parallel_receipt.plan.prompts",
    "$.config.parallel_plan.prompts",
    "$.metrics.system_receipt.trials",
    "$.metrics.system_receipt.hardware",
    "$.metrics.system_receipt.plan.prompts",
    "$.config.system_plan.prompts",
    "$.metrics.application_receipt.variants",
    "$.metrics.application_receipt.events",
    "$.metrics.application_receipt.plan.requests",
    "$.config.application_plan.requests",
    "$.metrics.token_itl_ms",
    "$.metrics.quantization.weight_error.per_tensor_metrics",
    "$.metrics.quantization.weight_error.per_tensor_wall_time_s",
)


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            directory / "console.sqlite3", check_same_thread=False
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, body TEXT);
            CREATE TABLE IF NOT EXISTS events (run_id TEXT, seq INTEGER, body TEXT,
                PRIMARY KEY(run_id, seq));
            CREATE TABLE IF NOT EXISTS idempotency (key TEXT PRIMARY KEY, digest TEXT, run_id TEXT);
            CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created DESC);
            CREATE INDEX IF NOT EXISTS runs_kind_created_idx
                ON runs(json_extract(body, '$.kind'), created DESC);
        """)
        self.db.commit()

    def create(
        self, kind: str, config: dict, key: str, digest: str
    ) -> tuple[dict, bool]:
        with self.lock, self.db:
            old = self.db.execute(
                "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
            ).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError("IDEMPOTENCY_CONFLICT")
                return self.get(old[1]), False
            now = time.time()
            run = {
                "id": "run_" + uuid.uuid4().hex,
                "kind": kind,
                "state": "queued",
                "created_at": now,
                "updated_at": now,
                "config": config,
                "output": "",
                "metrics": {},
                "error": None,
                "cleanup": "pending",
                "quality": "not_evaluated",
                "seq": 0,
            }
            self.db.execute(
                "INSERT INTO runs VALUES(?,?,?)", (run["id"], now, json.dumps(run))
            )
            self.db.execute(
                "INSERT INTO idempotency VALUES(?,?,?)", (key, digest, run["id"])
            )
            return self.update(run["id"], {}, "accepted", {}), True

    def get(self, run_id: str) -> dict:
        with self.lock:
            row = self.db.execute(
                "SELECT body FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if not row:
                raise KeyError(run_id)
            return json.loads(row[0])

    def update(
        self,
        run_id: str,
        patch: dict,
        kind: str = "state.changed",
        data: dict | None = None,
    ) -> dict:
        with self.lock, self.db:
            run = self.get(run_id)
            run.update(patch)
            run["updated_at"] = time.time()
            run["seq"] += 1
            event = {
                "event_version": "1",
                "request_id": run_id,
                "seq": run["seq"],
                "kind": kind,
                "time": run["updated_at"],
                "data": data if data is not None else patch,
            }
            self.db.execute(
                "UPDATE runs SET body=? WHERE id=?", (json.dumps(run), run_id)
            )
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?)",
                (run_id, run["seq"], json.dumps(event)),
            )
            return run

    def status(self, run_id: str) -> dict:
        """Small lifecycle poll; never deserialize inputs/tensors/token traces."""
        with self.lock:
            row = self.db.execute("SELECT json_extract(body,'$.state'),json_extract(body,'$.cleanup'),json_extract(body,'$.error') FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return {"id":run_id,"state":row[0],"cleanup":row[1],"error":row[2]}

    def events(self, run_id: str, after: int, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def list(
        self, limit: int = 100, offset: int = 0, *, kind: str | None = None
    ) -> dict:
        with self.lock:
            clause = " WHERE json_extract(body, '$.kind')=?" if kind else ""
            parameters = (kind,) if kind else ()
            rows = self.db.execute(
                "SELECT json_remove(body, "
                + ",".join("?" for _ in LIST_DETAIL_PATHS)
                + ") FROM runs"
                + clause
                + " ORDER BY created DESC LIMIT ? OFFSET ?",
                (*LIST_DETAIL_PATHS, *parameters, limit, offset),
            ).fetchall()
            total = self.db.execute(
                "SELECT count(*) FROM runs" + clause, parameters
            ).fetchone()[0]
            return {
                "items": [json.loads(row[0]) for row in rows],
                "total": total,
                "next_offset": offset + limit if offset + limit < total else None,
            }

    def quality_reference_ids(self, model_hash, source_digest):
        """Small ID projection for reference discovery, independent of UI pages."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM runs WHERE json_extract(body,'$.state')='completed' "
                "AND json_extract(body,'$.cleanup')='succeeded' "
                "AND json_extract(body,'$.metrics.quality_evaluation.status')='complete' "
                "AND json_extract(body,'$.metrics.optimization_receipt.precision_receipt.method')='none' "
                "AND json_extract(body,'$.metrics.optimization_receipt.model_identity.hash')=? "
                "AND json_extract(body,'$.metrics.optimization_receipt.source_digest')=? ORDER BY created,id",
                (model_hash, source_digest),
            ).fetchall()
        return [row[0] for row in rows]

    def recover(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM runs WHERE json_extract(body, '$.state') NOT IN ("
                + ",".join("?" for _ in TERMINAL)
                + ")",
                tuple(TERMINAL),
            ).fetchall()
        for (run_id,) in rows:
            self.update(
                run_id,
                {
                    "state": "interrupted",
                    "cleanup": "unknown",
                    "error": "Service restarted; previous execution was not replayed.",
                },
                "error",
            )

    def close(self):
        with self.lock:
            self.db.close()
