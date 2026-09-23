"""
core/audit.py
=============
Local, **tamper-evident** audit trail on SQLite (with a JSONL fallback).

Why this exists
---------------
An operator assistant that refuses requests, redacts secrets and reads machine data ought to
be able to prove *what it did*. Every gate decision, ingestion and copilot turn is appended to
a local SQLite file (no server, no ORM, no external dependency) with a hash chain:

``entry_hash = SHA-256(prev_hash | event_type | severity | machine_id | session_id | payload)``

so :meth:`AuditStore.verify_chain` can detect an edited or deleted row, and
:meth:`AuditStore.recent` gives the UI a normal DataFrame to render.

Security properties
-------------------
* **Parameterised SQL only** - every value is bound with ``?``; no SQL string is ever built
  from data (bandit ``B608`` clean, SQL-injection immune even before validation).
* **Payload sanitisation** - every string in the payload passes through
  :func:`core.security.flatten_for_log` before storage, capping length and stripping control
  characters (no log-forging, no XSS in the audit view).
* **No secrets** - the payload schema deliberately excludes API keys; the API surface makes
  it awkward to store one, and :func:`core.security.redact_secret` exists for display.
* **Fail-open on storage, never on security** - if the DB cannot be written, the event is
  buffered to a JSONL file and the *application keeps working*: an audit trail failure must
  not blind the operator mid-shift.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config import DB_PATH
from core.security import flatten_for_log, sanitize_text

GENESIS_HASH: str = "GENESIS"

#: Static DDL - no dynamic identifiers, so nothing here can be influenced by input.
_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT    NOT NULL,
    machine_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL,
    entry_hash  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events (created_utc);
CREATE INDEX IF NOT EXISTS idx_audit_type    ON audit_events (event_type, severity);
"""

_INSERT_SQL: str = (
    "INSERT INTO audit_events "
    "(created_utc, machine_id, session_id, event_type, severity, payload, prev_hash, entry_hash) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

_LAST_HASH_SQL: str = "SELECT entry_hash FROM audit_events ORDER BY id DESC LIMIT 1"

_SELECT_SQL: str = (
    "SELECT id, created_utc, machine_id, session_id, event_type, severity, payload, entry_hash "
    "FROM audit_events"
)

_EVENT_COLUMNS: tuple[str, ...] = (
    "id", "created_utc", "machine_id", "session_id", "event_type", "severity", "payload", "entry_hash",
)

#: Event types the app is allowed to write (keeps the trail queryable and bounded).
EVENT_TYPES: tuple[str, ...] = (
    "session_start",
    "telemetry_ingest",
    "gate_decision",
    "copilot_turn",
    "safety_snapshot",
    "shift_summary",
    "security_event",
    "config_note",
)


@dataclass(frozen=True)
class ChainVerification:
    """Result of :meth:`AuditStore.verify_chain`."""

    ok: bool
    rows_checked: int
    first_bad_id: int | None = None
    detail: str = ""


class AuditStore:
    """
    Append-only audit log backed by SQLite, with a JSONL fallback.

    Typical use::

        store = AuditStore(config.DB_PATH)
        store.record("gate_decision", "CRITICAL", machine_id="CAT-320-EXC-014",
                     session_id="ab12", payload=decision.audit_fields())
        store.verify_chain()
    """

    def __init__(self, path: str | Path = DB_PATH, *, fallback_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.fallback_path = Path(fallback_path) if fallback_path else self.path.with_suffix(".jsonl")
        self.backend = "sqlite"
        self.init_error = ""
        self._conn: sqlite3.Connection | None = None
        # Streamlit executes each browser session on its own thread, while this store is a
        # cached resource: the connection must therefore be usable from any thread and all
        # access is serialised through ``_lock``.
        self._lock = threading.RLock()
        self._open()

    # ----------------------------------------------------------------------------------
    # Connection handling
    # ----------------------------------------------------------------------------------
    def _open(self) -> None:
        """Open/create the SQLite database (degrades to JSONL on any failure)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.path), timeout=5.0, isolation_level=None, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA_SQL)
            self.backend = "sqlite"
        except (sqlite3.Error, OSError, ValueError) as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"
            self._conn = None
            self.backend = "jsonl"

    @property
    def available(self) -> bool:
        """True when events are being persisted somewhere."""
        return self.backend == "sqlite" and self._conn is not None

    def close(self) -> None:
        """Close the connection (idempotent)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                pass
            self._conn = None

    # ----------------------------------------------------------------------------------
    # Writing
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _clean_payload(payload: Mapping[str, Any] | None) -> dict[str, str]:
        """
        Sanitise + flatten a payload to ``{str: str}``.

        Every value becomes a short, control-character-free string, so the stored JSON is
        safe to render and impossible to use for log-forging.
        """
        clean: dict[str, str] = {}
        for key, value in (payload or {}).items():
            safe_key = sanitize_text(key, max_chars=48).text or "field"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                clean[safe_key] = str(value)
            elif isinstance(value, bool):
                clean[safe_key] = "true" if value else "false"
            elif value is None:
                clean[safe_key] = ""
            elif isinstance(value, (list, tuple, set)):
                clean[safe_key] = flatten_for_log(", ".join(str(item) for item in value), 240)
            elif isinstance(value, dict):
                clean[safe_key] = flatten_for_log(json.dumps(value, default=str), 240)
            else:
                clean[safe_key] = flatten_for_log(value, 240)
        return clean

    @staticmethod
    def _hash(prev_hash: str, event_type: str, severity: str, machine_id: str,
              session_id: str, payload_json: str) -> str:
        """Compute the chained SHA-256 for one event (SHA-256 only - never MD5/SHA-1)."""
        material = "|".join([prev_hash, event_type, severity, machine_id, session_id, payload_json])
        return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()

    def record(
        self,
        event_type: str,
        severity: str,
        *,
        machine_id: str = "UNKNOWN",
        session_id: str = "system",
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        """
        Append one audit event.

        Returns ``True`` when the event was persisted. Never raises: a storage failure is
        reported through the return value (and the JSONL fallback) so it can never take the
        dashboard down.
        """
        clean_type = sanitize_text(event_type, max_chars=32).text or "unknown"
        clean_severity = sanitize_text(severity, max_chars=16).text.upper() or "INFO"
        clean_machine = sanitize_text(machine_id, max_chars=32).text or "UNKNOWN"
        clean_session = sanitize_text(session_id, max_chars=32).text or "system"
        clean_payload = self._clean_payload(payload)

        payload_json = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"), default=str)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if self.available and self._conn is not None:
            try:
                with self._lock:
                    row = self._conn.execute(_LAST_HASH_SQL).fetchone()
                    prev_hash = str(row[0]) if row else GENESIS_HASH
                    entry_hash = self._hash(
                        prev_hash, clean_type, clean_severity, clean_machine, clean_session, payload_json
                    )
                    self._conn.execute(
                        _INSERT_SQL,
                        (created, clean_machine, clean_session, clean_type, clean_severity,
                         payload_json, prev_hash, entry_hash),
                    )
                return True
            except sqlite3.Error as exc:
                self.init_error = f"write failed: {type(exc).__name__}"
                self.backend = "jsonl"
                self._write_jsonl(created, clean_type, clean_severity, clean_machine,
                                  clean_session, payload_json)
                return False
        self._write_jsonl(created, clean_type, clean_severity, clean_machine, clean_session, payload_json)
        return False

    def _write_jsonl(self, created: str, event_type: str, severity: str, machine_id: str,
                     session_id: str, payload_json: str) -> None:
        """Fallback persistence: one JSON object per line (still sanitised, still hashed)."""
        try:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            prev_hash = self._last_jsonl_hash()
            entry_hash = self._hash(prev_hash, event_type, severity, machine_id, session_id, payload_json)
            line = json.dumps(
                {
                    "created_utc": created,
                    "machine_id": machine_id,
                    "session_id": session_id,
                    "event_type": event_type,
                    "severity": severity,
                    "payload": json.loads(payload_json),
                    "prev_hash": prev_hash,
                    "entry_hash": entry_hash,
                },
                sort_keys=True,
            )
            with self.fallback_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except (OSError, ValueError):
            return  # last resort: silently drop (never crash the UI over an audit write)

    def _last_jsonl_hash(self) -> str:
        """Read the tail of the JSONL fallback to continue the hash chain."""
        try:
            if not self.fallback_path.exists():
                return GENESIS_HASH
            tail = self.fallback_path.read_text(encoding="utf-8").strip().splitlines()
            return json.loads(tail[-1])["entry_hash"] if tail else GENESIS_HASH
        except (OSError, ValueError, KeyError):
            return GENESIS_HASH

    # ----------------------------------------------------------------------------------
    # Reading
    # ----------------------------------------------------------------------------------
    def recent(
        self,
        limit: int = 200,
        event_type: str | None = None,
        severity: str | None = None,
    ) -> pd.DataFrame:
        """
        Return the newest events as a DataFrame (payload is JSON-decoded into a column).

        Filter values are bound as SQL parameters; ``event_type`` is additionally checked
        against :data:`EVENT_TYPES` before use.
        """
        limit = max(1, min(int(limit), 5_000))
        if not self.available or self._conn is None:
            return pd.DataFrame(columns=_EVENT_COLUMNS)

        sql = _SELECT_SQL
        params: list[Any] = []
        clauses: list[str] = []
        if event_type and event_type in EVENT_TYPES:
            clauses.append("event_type = ?")
            params.append(event_type)
        if severity:
            clauses.append("severity = ?")
            params.append(sanitize_text(severity, max_chars=16).text.upper())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return pd.DataFrame(columns=_EVENT_COLUMNS)
        frame = pd.DataFrame(rows, columns=list(_EVENT_COLUMNS))
        if not frame.empty:
            frame["payload"] = frame["payload"].map(_safe_json_loads)
        return frame

    def counts(self) -> dict[str, int]:
        """Event counts per type (for the Security tab metrics)."""
        if not self.available or self._conn is None:
            return {}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type ORDER BY 2 DESC"
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {str(row[0]): int(row[1]) for row in rows}

    def total_events(self) -> int:
        """Total number of persisted events."""
        if not self.available or self._conn is None:
            return 0
        try:
            with self._lock:
                row = self._conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0

    def verify_chain(self, limit: int = 10_000) -> ChainVerification:
        """
        Recompute the hash chain and report the first inconsistency.

        Detects edited payloads, forged hashes and deleted rows (a deletion breaks the
        ``prev_hash`` linkage of the following row).
        """
        if not self.available or self._conn is None:
            return ChainVerification(ok=False, rows_checked=0, detail="audit store unavailable")
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, event_type, severity, machine_id, session_id, payload, prev_hash, "
                    "entry_hash FROM audit_events ORDER BY id ASC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
        except sqlite3.Error as exc:
            return ChainVerification(ok=False, rows_checked=0, detail=f"read failed: {type(exc).__name__}")

        expected_prev = GENESIS_HASH
        for row in rows:
            row_id, event_type, severity, machine_id, session_id, payload, prev_hash, entry_hash = row
            if str(prev_hash) != expected_prev:
                return ChainVerification(
                    ok=False,
                    rows_checked=len(rows),
                    first_bad_id=int(row_id),
                    detail="broken hash linkage (row edited or deleted)",
                )
            recomputed = self._hash(str(prev_hash), str(event_type), str(severity),
                                    str(machine_id), str(session_id), str(payload))
            if recomputed != str(entry_hash):
                return ChainVerification(
                    ok=False,
                    rows_checked=len(rows),
                    first_bad_id=int(row_id),
                    detail="hash mismatch (payload modified)",
                )
            expected_prev = str(entry_hash)
        return ChainVerification(ok=True, rows_checked=len(rows), detail="chain intact")

    def purge_older_than(self, days: int = 90) -> int:
        """
        Delete events older than ``days`` (retention control for an edge disk).

        Returns the number of rows removed. ``days`` is clamped to a sane minimum.
        """
        days = max(1, int(days))
        if not self.available or self._conn is None:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "DELETE FROM audit_events WHERE created_utc < ?", (cutoff,)
                )
            return int(cursor.rowcount or 0)
        except sqlite3.Error:
            return 0


def _safe_json_loads(value: Any) -> Any:
    """JSON-decode a payload cell, returning the raw string if it is not JSON."""
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return {}


def summarize_events(frame: pd.DataFrame) -> list[str]:
    """Turn an audit DataFrame into a few plain-text highlights for the UI."""
    if frame is None or frame.empty:
        return ["No audit events recorded yet."]
    highlights: list[str] = []
    critical = frame[frame["severity"].astype(str).str.upper() == "CRITICAL"]
    if not critical.empty:
        highlights.append(f"{len(critical)} CRITICAL event(s) - mostly refused jailbreak / safety-bypass attempts")
    events = frame["event_type"].value_counts().to_dict()
    for event_type, count in list(events.items())[:5]:
        highlights.append(f"{count} × {event_type}")
    return highlights


__all__ = [
    "GENESIS_HASH",
    "EVENT_TYPES",
    "ChainVerification",
    "AuditStore",
    "summarize_events",
]
