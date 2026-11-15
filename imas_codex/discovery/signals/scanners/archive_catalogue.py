"""Disk-backed archive census, metadata and acknowledged ingest checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from pathlib import Path


class ArchiveCatalogue:
    """Keep bounded batches durable independently of graph acknowledgement.

    Every archive membership is retained. A selector's first successful metadata
    source owns its primary metadata record; all complete source documents remain
    in the checkpoint. Failed requests carry errors and never become graph rows.
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
        """)

    def close(self) -> None:
        self.db.close()

    def seed(self, populations: dict[str, Iterable[str]], *, timestamp: str) -> dict:
        """Seal the independent census before permitting metadata or graph writes."""
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

    def run(
        self,
        *,
        facility: str,
        fetch: Callable,
        ingest: Callable,
        make_signal: Callable,
        batch_size: int = 500,
        max_batches: int | None = None,
        progress: Callable | None = None,
    ) -> dict:
        """Resume without requesting metadata for a completed variable again.

        Metadata is committed before ingest. If acknowledgement is interrupted,
        replaying the same graph identity is safe and avoids another archive call.
        """
        census = self.census()
        if not 1 <= batch_size <= 1000:
            raise ValueError("metadata batch must be between 1 and 1000")
        started = time.monotonic()
        fetched = written = batches = 0
        roles = list(census["counts"])
        for role in roles:
            while max_batches is None or batches < max_batches:
                # Replay fetched but unacknowledged records before new requests.
                pending = self.db.execute(
                    "SELECT name,document FROM census WHERE role=? AND document IS NOT NULL AND acknowledged=0 LIMIT ?",
                    (role, batch_size),
                ).fetchall()
                if not pending:
                    names = [
                        row[0]
                        for row in self.db.execute(
                            "SELECT name FROM census WHERE role=? AND document IS NULL AND error IS NULL LIMIT ?",
                            (role, batch_size),
                        )
                    ]
                    if not names:
                        break
                    response = fetch(role, names)
                    entries = response["entries"]
                    failed = response.get("failed", [])
                    answered = [item["name"] for item in entries + failed]
                    if len(answered) != len(names) or set(answered) != set(names):
                        raise RuntimeError(
                            "metadata receipt does not match requested selectors"
                        )
                    with self.db:
                        self.db.executemany(
                            "UPDATE census SET document=? WHERE role=? AND name=?",
                            (
                                (json.dumps(item), role, item["name"])
                                for item in entries
                            ),
                        )
                        self.db.executemany(
                            "UPDATE census SET error=? WHERE role=? AND name=?",
                            ((item["error"], role, item["name"]) for item in failed),
                        )
                    fetched += len(names)
                    pending = [(item["name"], json.dumps(item)) for item in entries]
                    batches += 1
                signals = []
                for name, document in pending:
                    entry = json.loads(document)
                    entry["listed_archives"] = [
                        f"{facility}:codac_uda:{item[0]}"
                        for item in self.db.execute(
                            "SELECT role FROM census WHERE name=?", (name,)
                        )
                    ]
                    signals.append(make_signal(role, entry, census["fingerprint"]))
                if signals:
                    count = ingest(signals)
                    if count != len(signals):
                        raise RuntimeError(
                            f"incomplete graph batch: acknowledged {count} of {len(signals)}"
                        )
                    with self.db:
                        self.db.executemany(
                            "INSERT OR IGNORE INTO written VALUES (?)",
                            ((name,) for name, _ in pending),
                        )
                        self.db.executemany(
                            "UPDATE census SET acknowledged=1 WHERE role=? AND name=?",
                            ((role, name) for name, _ in pending),
                        )
                    written += count
                if progress:
                    progress(
                        {
                            "fetched": fetched,
                            "written": written,
                            "batches": batches,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
        total_written = self.db.execute("SELECT count(*) FROM written").fetchone()[0]
        failed = self.db.execute(
            "SELECT count(*) FROM census WHERE error IS NOT NULL"
        ).fetchone()[0]
        unrequested = self.db.execute(
            "SELECT count(*) FROM census WHERE document IS NULL AND error IS NULL"
        ).fetchone()[0]
        return {
            "complete": total_written == census["union"]
            and unrequested == 0
            and failed == 0,
            "persisted": True,
            "enumerated": census["union"],
            "written": total_written,
            "newly_written": written,
            "metadata_requests": fetched,
            "failed": failed,
            "metadata_batches": batches,
            "elapsed_seconds": time.monotonic() - started,
        }
