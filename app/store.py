"""Versioned plan store backed by SQLite.

Publishing is a single transaction: compute the next version and insert the
payload.  If the transaction fails, nothing is written and every previously
published version remains queryable -- the "last good plan" guarantee.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    version   INTEGER PRIMARY KEY,
    batch_id  TEXT NOT NULL,
    payload   TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class PlanStore:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        con = self._connect()
        try:
            with con:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute(_SCHEMA)
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def publish(self, payload: dict) -> int:
        """Atomically store a new plan version; returns its version number."""
        with self._lock:
            con = self._connect()
            try:
                with con:  # commits on success, rolls back on any error
                    (version,) = con.execute(
                        "SELECT COALESCE(MAX(version), 0) + 1 FROM plans"
                    ).fetchone()
                    con.execute(
                        "INSERT INTO plans (version, batch_id, payload, created_at)"
                        " VALUES (?, ?, ?, ?)",
                        (
                            version,
                            payload["batch_id"],
                            json.dumps(payload, ensure_ascii=False),
                            self._now(),
                        ),
                    )
                return version
            finally:
                con.close()

    def current_version(self) -> int | None:
        con = self._connect()
        try:
            row = con.execute("SELECT MAX(version) FROM plans").fetchone()
            return row[0]
        finally:
            con.close()

    def get(self, version: int) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT payload, created_at FROM plans WHERE version = ?",
                (version,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        doc = json.loads(row[0])
        doc["version"] = version
        doc["created_at"] = row[1]
        return doc

    def current(self) -> dict | None:
        version = self.current_version()
        return None if version is None else self.get(version)

    def list(self) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT version, batch_id, created_at FROM plans"
                " ORDER BY version DESC"
            ).fetchall()
        finally:
            con.close()
        return [
            {"version": v, "batch_id": b, "created_at": ts} for v, b, ts in rows
        ]
