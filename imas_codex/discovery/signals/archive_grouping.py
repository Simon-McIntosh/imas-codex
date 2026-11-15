"""Exact component and field-set grouping in a private, replaceable SQLite store.

The complete prefix identifies a candidate component. A field retains both the
control function and the variable token, including numbers: grouping does not
assert that similarly spelled fields mean the same thing. Metadata remains
per archive and per member; missing metadata does not erase a listed variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

from imas_codex.discovery.signals.variable_names import parse_variable_name
from imas_codex.graph.models import ArchiveGroupingState, SignalNameParseOutcome


def catalogue_records(db: sqlite3.Connection) -> Iterator[dict]:
    """Stream distinct selectors, retaining each archive's metadata outcome."""
    rows = db.execute("""
        SELECT name,role,attributes_checked,metadata_state,
               json_extract(document,'$.data_type'),json_extract(document,'$.unit.value')
        FROM census ORDER BY name,role
    """)
    for name, members in groupby(rows, key=lambda row: row[0]):
        yield {
            "name": name,
            "archives": {
                role: {
                    "completed": bool(checked),
                    "metadata_state": state,
                    "data_type": dtype,
                    "unit": unit,
                }
                for _, role, checked, state, dtype, unit in members
            },
        }


class ArchiveGrouping:
    """Atomically recompute one facility, preserving other facilities' groups.

    The source catalogue is opened read-only. Derived membership lives separately
    so the metadata scanner's lease and checkpoints stay with their owner.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS members (
                facility TEXT, name TEXT, state TEXT NOT NULL, reason TEXT,
                component TEXT, field TEXT, archives TEXT NOT NULL,
                PRIMARY KEY(facility,name));
            CREATE INDEX IF NOT EXISTS members_component ON members(facility,component,field);
            CREATE TABLE IF NOT EXISTS components (
                facility TEXT, id TEXT PRIMARY KEY, code TEXT, pattern TEXT,
                member_count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS patterns (
                facility TEXT, id TEXT PRIMARY KEY, fields TEXT,
                component_count INTEGER NOT NULL, member_count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS grouping_receipts (facility TEXT PRIMARY KEY, receipt TEXT);
        """)

    def close(self) -> None:
        self.db.close()

    def rebuild(self, facility: str, records: Iterable[dict]) -> dict:
        """Replace a complete snapshot transactionally; reject duplicate selectors."""
        if not facility:
            raise ValueError("a facility is required")
        previous_components = {
            row[0]
            for row in self.db.execute(
                "SELECT id FROM components WHERE facility=?", (facility,)
            )
        }
        previous_patterns = {
            row[0]
            for row in self.db.execute(
                "SELECT id FROM patterns WHERE facility=?", (facility,)
            )
        }

        def member_rows():
            for record in records:
                parsed = parse_variable_name(record["name"])
                complete = parsed.outcome == SignalNameParseOutcome.parsed
                yield (
                    facility,
                    parsed.selector,
                    ArchiveGroupingState.grouped.value
                    if complete
                    else ArchiveGroupingState.ungrouped.value,
                    None if complete else parsed.reason,
                    f"{facility}:plant_component:{parsed.plant_component_code}"
                    if complete
                    else None,
                    f"{parsed.control_function}-{parsed.variable_token}"
                    if complete
                    else None,
                    json.dumps(record.get("archives", {}), sort_keys=True),
                )

        with self.db:
            for table in ("members", "components", "patterns"):
                self.db.execute(f"DELETE FROM {table} WHERE facility=?", (facility,))
            self.db.executemany(
                "INSERT INTO members VALUES (?,?,?,?,?,?,?)", member_rows()
            )
            rows = self.db.execute(
                "SELECT component,field FROM members WHERE facility=? AND component IS NOT NULL ORDER BY component,field",
                (facility,),
            )
            for component, fields in groupby(rows, key=lambda row: row[0]):
                field_set = sorted({row[1] for row in fields})
                encoded = json.dumps(field_set, separators=(",", ":"))
                digest = hashlib.sha256(encoded.encode()).hexdigest()
                pattern = f"{facility}:archive_field_pattern:{digest}"
                self.db.execute(
                    "INSERT INTO components VALUES (?,?,?,?,?)",
                    (
                        facility,
                        component,
                        component.split(":plant_component:", 1)[1],
                        pattern,
                        len(field_set),
                    ),
                )
                self.db.execute(
                    "INSERT INTO patterns VALUES (?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET component_count=component_count+1, member_count=member_count+excluded.member_count",
                    (facility, pattern, encoded, 1, len(field_set)),
                )
            receipt = self.summary(facility)
            receipt.update(
                groups_added=len(
                    self._ids("components", facility) - previous_components
                ),
                groups_removed=len(
                    previous_components - self._ids("components", facility)
                ),
                patterns_added=len(self._ids("patterns", facility) - previous_patterns),
                patterns_removed=len(
                    previous_patterns - self._ids("patterns", facility)
                ),
            )
            self.db.execute(
                "INSERT INTO grouping_receipts VALUES (?,?) ON CONFLICT(facility) DO UPDATE SET receipt=excluded.receipt",
                (facility, json.dumps(receipt)),
            )
        return receipt

    def _ids(self, table: str, facility: str) -> set[str]:
        return {
            row[0]
            for row in self.db.execute(
                f"SELECT id FROM {table} WHERE facility=?", (facility,)
            )
        }

    def member(self, facility: str, name: str) -> dict:
        row = self.db.execute(
            "SELECT state,reason,component,field,archives FROM members WHERE facility=? AND name=?",
            (facility, name),
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return {
            "state": row[0],
            "reason": row[1],
            "component": row[2],
            "field": row[3],
            "archives": json.loads(row[4]),
        }

    def component(self, component_id: str) -> dict:
        row = self.db.execute(
            "SELECT facility,code,pattern,member_count FROM components WHERE id=?",
            (component_id,),
        ).fetchone()
        if row is None:
            raise KeyError(component_id)
        return {
            "id": component_id,
            "facility_id": row[0],
            "code": row[1],
            "archive_field_pattern": row[2],
            "archive_member_count": row[3],
        }

    def pattern(self, pattern_id: str) -> dict:
        row = self.db.execute(
            "SELECT facility,fields,component_count,member_count FROM patterns WHERE id=?",
            (pattern_id,),
        ).fetchone()
        if row is None:
            raise KeyError(pattern_id)
        fields = json.loads(row[1])
        return {
            "id": pattern_id,
            "facility_id": row[0],
            "fields": fields,
            "field_count": len(fields),
            "component_count": row[2],
            "member_count": row[3],
        }

    def summary(self, facility: str) -> dict:
        states = dict(
            self.db.execute(
                "SELECT state,count(*) FROM members WHERE facility=? GROUP BY state",
                (facility,),
            )
        )
        sizes = dict(
            self.db.execute(
                "SELECT member_count,count(*) FROM components WHERE facility=? GROUP BY member_count ORDER BY member_count",
                (facility,),
            )
        )
        total = sum(states.values())
        patterns = [
            self.pattern(row[0])
            for row in self.db.execute(
                "SELECT id FROM patterns WHERE facility=? ORDER BY member_count DESC,id LIMIT 20",
                (facility,),
            )
        ]
        return {
            "facility": facility,
            "variables": total,
            "grouped": states.get(ArchiveGroupingState.grouped.value, 0),
            "ungrouped": states.get(ArchiveGroupingState.ungrouped.value, 0),
            "component_groups": sum(sizes.values()),
            "field_patterns": self.db.execute(
                "SELECT count(*) FROM patterns WHERE facility=?", (facility,)
            ).fetchone()[0],
            "group_size_distribution": {
                str(size): count for size, count in sizes.items()
            },
            "singleton_signal_share": sizes.get(1, 0) / total if total else 0,
            "largest_patterns": patterns,
            "group_descriptions_needed": sum(sizes.values()),
            "pattern_descriptions_needed": self.db.execute(
                "SELECT count(*) FROM patterns WHERE facility=?", (facility,)
            ).fetchone()[0],
        }


def group_private_catalogue(
    source: Path,
    grouping: ArchiveGrouping,
    facility: str,
    *,
    graph_signals: Iterable[dict] = (),
) -> dict:
    """Group a snapshot without holding the metadata writer's database lock.

    SQLite's online backup releases the source lock between page batches. The
    long grouping transaction reads only the job-local copy, while the metadata
    scanner keeps committing to its own store.
    """
    extras = {signal["selector"]: signal for signal in graph_signals}
    with (
        TemporaryDirectory(
            prefix="archive-grouping-", dir=os.environ.get("TMPDIR")
        ) as temporary,
        closing(sqlite3.connect(Path(temporary) / "catalogue.sqlite")) as db,
    ):
        with closing(
            sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
        ) as origin:
            origin.backup(db, pages=1024)
        db.execute("BEGIN")
        census_row = db.execute(
            "SELECT value FROM receipts WHERE key='census'"
        ).fetchone()
        if census_row is None:
            raise RuntimeError("grouping requires a sealed archive census")
        census = json.loads(census_row[0])
        stats = {
            "listed_variables": 0,
            "metadata_completed": 0,
            "graph_only_variables": 0,
        }

        def records():
            for record in catalogue_records(db):
                extras.pop(record["name"], None)
                stats["listed_variables"] += 1
                stats["metadata_completed"] += int(
                    all(archive["completed"] for archive in record["archives"].values())
                )
                yield record
            if not census["union"] or stats["listed_variables"] > census["union"]:
                raise RuntimeError(
                    "catalogue census is empty or does not bound the listed variables"
                )
            stats["graph_only_variables"] = len(extras)
            for selector in extras:
                yield {"name": selector}

        result = grouping.rebuild(facility, records())
    return {
        **result,
        **stats,
        "archive_variables": census["union"],
        "archive_fraction": stats["listed_variables"] / census["union"],
        "metadata_completed_fraction": stats["metadata_completed"] / census["union"],
        "metadata_scan_complete": stats["metadata_completed"] == census["union"],
        "census_fingerprint": census["fingerprint"],
    }
