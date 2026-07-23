"""Durable run snapshots and ordered events, independent of transport lifetime."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

TERMINAL = frozenset({"completed", "failed", "cancelled", "timed_out", "interrupted"})


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

    def events(self, run_id: str, after: int, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def list(self, limit: int = 100, offset: int = 0) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM runs ORDER BY created DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = self.db.execute("SELECT count(*) FROM runs").fetchone()[0]
            return {
                "items": [json.loads(row[0]) for row in rows],
                "total": total,
                "next_offset": offset + limit if offset + limit < total else None,
            }

    def recover(self):
        with self.lock:
            rows = self.db.execute("SELECT id,body FROM runs").fetchall()
        for run_id, body in rows:
            if json.loads(body)["state"] not in TERMINAL:
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
