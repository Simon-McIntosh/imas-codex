"""Private archive census and metadata checkpoints, independent of graph ingestion."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event


def metadata_entries(response: dict) -> list[dict]:
    """Represent every answered selector, including explicit archive refusals."""
    entries = list(response["entries"])
    answered = {entry["name"] for entry in entries}
    for failure in response.get("failed", []):
        if failure["name"] not in answered:
            entries.append(
                {
                    "name": failure["name"],
                    "metadata_state": "unavailable",
                    "metadata_error_code": failure.get("error_code", 0),
                    "metadata_error": failure["error"],
                }
            )
    return entries


class ArchiveCatalogue:
    """Retain every listing membership and both metadata interfaces privately.

    One owner commits SQLite batches while independent readers fetch disjoint
    selector ranges. Historical pilot acknowledgements remain untouched.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS census (role TEXT, name TEXT,
                document TEXT, error TEXT, acknowledged INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (role, name));
            CREATE TABLE IF NOT EXISTS receipts (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS written (name TEXT PRIMARY KEY);
            CREATE INDEX IF NOT EXISTS pending_metadata ON census(role) WHERE document IS NULL AND error IS NULL;
            CREATE INDEX IF NOT EXISTS pending_ingest ON census(role) WHERE document IS NOT NULL AND acknowledged=0;
            CREATE INDEX IF NOT EXISTS selector_membership ON census(name);
            CREATE TABLE IF NOT EXISTS range_failures (role TEXT, names TEXT,
                error TEXT, error_type TEXT, attempts INTEGER, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

        columns = {row[1] for row in self.db.execute("PRAGMA table_info(census)")}
        with self.db:
            for name, definition in (
                ("attributes_checked", "INTEGER NOT NULL DEFAULT 0"),
                ("metadata_state", "TEXT"),
                ("error_code", "INTEGER"),
                ("fetch_error", "TEXT"),
                ("fetch_attempts", "INTEGER"),
            ):
                if name not in columns:
                    self.db.execute(
                        f"ALTER TABLE census ADD COLUMN {name} {definition}"
                    )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS pending_attributes ON census(role) WHERE attributes_checked=0"
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS pending_metadata_ranges ON census(role,name) WHERE attributes_checked=0"
            )

    def cached_type_documents(self, role: str, names: list[str]) -> dict:
        """Reuse successful type documents while backfilling full attributes."""
        placeholders = ",".join("?" for _ in names)
        records = self.db.execute(
            f"SELECT name,document FROM census WHERE role=? AND name IN ({placeholders}) AND document IS NOT NULL",
            [role, *names],
        )
        result = {}
        for name, encoded in records:
            document = json.loads(encoded).get("archive_metadata")
            if isinstance(document, dict) and "datatypes" in document:
                result[name] = document
            elif isinstance(document, dict) and isinstance(document.get("type"), dict):
                result[name] = document["type"]
        return result

    def close(self) -> None:
        self.db.close()

    def seed(self, populations: dict[str, Iterable[str]], *, timestamp: str) -> dict:
        """Seal independent listings before metadata retrieval can begin."""
        if self.db.execute("SELECT 1 FROM receipts WHERE key='census'").fetchone():
            return self.census()
        with self.db:
            for role, names in populations.items():
                self.db.executemany(
                    "INSERT OR IGNORE INTO census(role,name) VALUES (?,?)",
                    ((role, name) for name in names),
                )
            counts = dict(
                self.db.execute("SELECT role,count(*) FROM census GROUP BY role")
            )
            union = self.db.execute(
                "SELECT count(DISTINCT name) FROM census"
            ).fetchone()[0]
            overlap = self.db.execute(
                "SELECT count(*) FROM (SELECT name FROM census GROUP BY name HAVING count(*)>1)"
            ).fetchone()[0]
            fingerprint = hashlib.sha256()
            for role, name in self.db.execute(
                "SELECT role,name FROM census ORDER BY role,name"
            ):
                fingerprint.update(json.dumps([role, name]).encode())
            result = {
                "counts": counts,
                "union": union,
                "overlap": overlap,
                "timestamp": timestamp,
                "fingerprint": fingerprint.hexdigest(),
            }
            self.db.execute(
                "INSERT INTO receipts VALUES ('census',?)", (json.dumps(result),)
            )
        return result

    def census(self) -> dict:
        row = self.db.execute(
            "SELECT value FROM receipts WHERE key='census'"
        ).fetchone()
        if row is None:
            raise RuntimeError("a sealed census is required before catalogue writes")
        return json.loads(row[0])

    def record(self, name: str) -> dict:
        """Read one selector with every listing and per-archive metadata outcome."""
        rows = self.db.execute(
            "SELECT role,document,attributes_checked,fetch_error,fetch_attempts FROM census WHERE name=? ORDER BY role",
            (name,),
        ).fetchall()
        if not rows:
            raise KeyError(name)
        return {
            "name": name,
            "listed_archives": [row[0] for row in rows],
            "archives": {
                role: {
                    "completed": bool(checked),
                    "metadata": json.loads(document) if document else None,
                    "fetch_state": "failed"
                    if error
                    else "complete"
                    if checked
                    else "pending",
                    "fetch_error": error,
                    "fetch_attempts": attempts,
                }
                for role, document, checked, error, attempts in rows
            },
        }

    def save_batch(self, role: str, names: list[str], response: dict) -> None:
        """Validate a complete receipt before atomically replacing any metadata."""
        entries = metadata_entries(response)
        failures = response.get("range_failures", [])
        answered = [item["name"] for item in entries] + [
            name for failure in failures for name in failure["names"]
        ]
        if len(answered) != len(names) or set(answered) != set(names):
            raise RuntimeError("metadata receipt does not match requested selectors")
        documents = []
        for entry in entries:
            entry = dict(entry)
            state = entry.get("metadata_state", "available")
            if state == "available":
                document = entry.get("archive_metadata")
                if (
                    not isinstance(document, dict)
                    or not isinstance(document.get("type"), dict)
                    or not isinstance(document.get("full"), list)
                ):
                    raise ValueError(
                        "available metadata requires both interface documents"
                    )
                entry["metadata_state"] = state
                for key in list(entry):
                    value = entry[key]
                    if (
                        value is None
                        or value == ""
                        or value == []
                        or isinstance(value, dict)
                        and value.get("value", object()) is None
                    ):
                        del entry[key]
            elif state == "unavailable":
                if type(entry.get("metadata_error_code")) is not int or not entry.get(
                    "metadata_error"
                ):
                    raise ValueError(
                        "unavailable metadata requires an archive error code and error"
                    )
                entry = {
                    key: value
                    for key, value in entry.items()
                    if key
                    in {
                        "name",
                        "metadata_state",
                        "metadata_error_code",
                        "metadata_error",
                        "metadata_errors",
                        "metadata_interfaces",
                    }
                }
            else:
                raise ValueError("unknown metadata state")
            documents.append(
                (
                    json.dumps(entry),
                    entry.get("metadata_error"),
                    entry.get("metadata_error_code"),
                    state,
                    role,
                    entry["name"],
                )
            )
        with self.db:
            updated = self.db.executemany(
                "UPDATE census SET document=?,error=?,error_code=?,metadata_state=?,attributes_checked=1,fetch_error=NULL,fetch_attempts=NULL WHERE role=? AND name=? AND attributes_checked=0",
                documents,
            )
            if updated.rowcount != len(entries):
                raise RuntimeError(
                    "metadata range includes unknown or completed selectors"
                )
            for failure in failures:
                updated = self.db.executemany(
                    "UPDATE census SET fetch_error=?,fetch_attempts=? WHERE role=? AND name=? AND attributes_checked=0",
                    (
                        (failure["error"], failure["attempts"], role, name)
                        for name in failure["names"]
                    ),
                )
                if updated.rowcount != len(failure["names"]):
                    raise RuntimeError(
                        "failed range includes unknown or completed selectors"
                    )
                self.db.execute(
                    "INSERT INTO range_failures(role,names,error,error_type,attempts) VALUES (?,?,?,?,?)",
                    (
                        role,
                        json.dumps(failure["names"]),
                        failure["error"],
                        failure["error_type"],
                        failure["attempts"],
                    ),
                )

    def summary(self) -> dict:
        """Count the union separately from archive memberships and completed metadata."""
        census = self.census()
        grouped = """SELECT name,count(*) AS memberships,sum(attributes_checked) AS checked,
            max(CASE WHEN metadata_state='available' THEN 1 ELSE 0 END) AS available,
            min(error_code) AS code FROM census GROUP BY name"""
        counts = self.db.execute(
            f"SELECT count(*),sum(checked=memberships) FROM ({grouped})"
        ).fetchone()
        states = dict(
            self.db.execute(
                f"SELECT CASE WHEN available=1 THEN 'available' ELSE 'unavailable' END,count(*) FROM ({grouped}) WHERE checked=memberships GROUP BY available"
            )
        )
        codes = dict(
            self.db.execute(
                f"SELECT code,count(*) FROM ({grouped}) WHERE checked=memberships AND available=0 GROUP BY code"
            )
        )
        by_archive = {
            role: {
                "listed": listed,
                "completed": checked,
                "available": available,
                "unavailable": unavailable,
            }
            for role, listed, checked, available, unavailable in self.db.execute("""SELECT role,count(*),sum(attributes_checked),
                sum(CASE WHEN attributes_checked=1 AND metadata_state='available' THEN 1 ELSE 0 END),
                sum(CASE WHEN attributes_checked=1 AND metadata_state='unavailable' THEN 1 ELSE 0 END)
                FROM census GROUP BY role""")
        }
        fields = (
            "unit",
            "description",
            "data_type",
            "structure_fields",
            "archive_metadata",
            "archive_labels",
            "archive_format",
            "archive_lower_display_limit",
            "archive_upper_display_limit",
            "archive_lower_alarm_limit",
            "archive_upper_alarm_limit",
            "archive_lower_warning_limit",
            "archive_upper_warning_limit",
            "first_sample_ns",
            "last_sample_ns",
            "sampling_mode",
            "sampling_evidence",
            "metadata_interfaces",
        )
        expressions = []
        for field in fields:
            path = "$." + field + (".value" if field in {"unit", "description"} else "")
            expressions.append(
                f"count(DISTINCT CASE WHEN json_extract(document,'{path}') IS NOT NULL AND json_extract(document,'{path}') NOT IN ('','[]','{{}}') THEN name END)"
            )
        coverage = dict(
            zip(
                fields,
                self.db.execute(
                    "SELECT "
                    + ",".join(expressions)
                    + " FROM census WHERE attributes_checked=1"
                ).fetchone(),
                strict=True,
            )
        )
        completed = counts[1] or 0
        return {
            "census": census,
            "variables": counts[0],
            "completed": completed,
            "pending": counts[0] - completed,
            "completed_fraction": completed / counts[0] if counts[0] else 1,
            "complete": completed == counts[0] == census["union"],
            "metadata_states": states,
            "unavailable_codes": codes,
            "archives": by_archive,
            "metadata_fields": coverage,
            "failed_variables": self.db.execute(
                "SELECT count(DISTINCT name) FROM census WHERE fetch_error IS NOT NULL"
            ).fetchone()[0],
            "failed_ranges_recorded": self.db.execute(
                "SELECT count(*) FROM range_failures"
            ).fetchone()[0],
        }

    def scan(
        self,
        fetch: Callable,
        *,
        workers: int = 1,
        batch_size: int = 100,
        max_batches: int | None = None,
        max_seconds: float | None = None,
        stop: Event | None = None,
        progress: Callable | None = None,
    ) -> dict:
        """Fetch disjoint bounded ranges; only this thread writes the private store.

        No graph callback is accepted. Completed ranges are excluded on every
        invocation. On a stop, in-flight batches drain before the receipt returns.
        """
        census = self.census()
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("metadata workers must be between 1 and 8")
        if type(batch_size) is not int or not 1 <= batch_size <= 1000:
            raise ValueError("metadata batch must be between 1 and 1000")
        if (
            max_batches is not None
            and max_batches < 0
            or max_seconds is not None
            and max_seconds <= 0
        ):
            raise ValueError("scan bounds must be positive or a zero batch pause")
        initial_rows = self.db.execute("SELECT count(*) FROM census").fetchone()[0]
        initial_completed = self.db.execute(
            "SELECT count(*) FROM census WHERE attributes_checked=1"
        ).fetchone()[0]
        started = time.monotonic()
        submitted = fetched = batches = type_requests = full_requests = 0
        attempted = failed = timeouts = restarts = submitted_full = submitted_type = 0
        after = dict.fromkeys(census["counts"], "")
        exhausted = set()
        active = {}

        def stopping():
            return (
                bool(stop and stop.is_set())
                or max_seconds is not None
                and time.monotonic() - started >= max_seconds
            )

        def next_range():
            for role in census["counts"]:
                if role in exhausted:
                    continue
                names = [
                    row[0]
                    for row in self.db.execute(
                        "SELECT name FROM census WHERE role=? AND attributes_checked=0 AND name>? ORDER BY name LIMIT ?",
                        (role, after[role], batch_size),
                    )
                ]
                if names:
                    after[role] = names[-1]
                    return role, names
                exhausted.add(role)
            return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while (
                    len(active) < workers
                    and not stopping()
                    and (max_batches is None or submitted < max_batches)
                ):
                    request = next_range()
                    if request is None:
                        break
                    role, names = request
                    cached = self.cached_type_documents(role, names)
                    future = pool.submit(fetch, role, names, cached)
                    active[future] = (role, names, set(cached))
                    submitted += 1
                if not active:
                    break
                ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    role, names, cached_names = active.pop(future)
                    response = future.result()
                    self.save_batch(role, names, response)
                    completed_names = {
                        entry["name"] for entry in metadata_entries(response)
                    }
                    fetched += len(completed_names)
                    attempted += len(names)
                    failed += len(names) - len(completed_names)
                    counters = response.get("request_counts", {})
                    full_requests += counters.get("full_requests", len(completed_names))
                    type_requests += counters.get(
                        "type_requests", len(completed_names - cached_names)
                    )
                    timeouts += counters.get("timeouts", 0)
                    restarts += counters.get("restarts", 0)
                    submitted_full += counters.get(
                        "submitted_full_requests", len(names)
                    )
                    submitted_type += counters.get(
                        "submitted_type_requests", len(set(names) - cached_names)
                    )
                    batches += 1
                    if progress:
                        elapsed = time.monotonic() - started
                        progress(
                            {
                                "metadata_requests": fetched,
                                "metadata_batches": batches,
                                "type_requests": type_requests,
                                "full_requests": full_requests,
                                "attempted_variables": attempted,
                                "failed_variables_this_run": failed,
                                "timeout_events": timeouts,
                                "reader_restarts": restarts,
                                "submitted_full_requests_upper_bound": submitted_full,
                                "submitted_type_requests_upper_bound": submitted_type,
                                "completed_memberships": initial_completed + fetched,
                                "listed_memberships": initial_rows,
                                "completed_membership_fraction": (
                                    initial_completed + fetched
                                )
                                / initial_rows
                                if initial_rows
                                else 1,
                                "elapsed_seconds": elapsed,
                                "variables_per_second": fetched / elapsed,
                            }
                        )
        elapsed = time.monotonic() - started
        result = self.summary()
        result.update(
            metadata_requests=fetched,
            metadata_batches=batches,
            workers=workers,
            type_requests=type_requests,
            full_requests=full_requests,
            elapsed_seconds=elapsed,
            variables_per_second=fetched / elapsed if elapsed else 0,
            attempted_variables=attempted,
            failed_variables_this_run=failed,
            timeout_events=timeouts,
            reader_restarts=restarts,
            submitted_full_requests_upper_bound=submitted_full,
            submitted_type_requests_upper_bound=submitted_type,
            rows_added=self.db.execute("SELECT count(*) FROM census").fetchone()[0]
            - initial_rows,
        )
        return result
