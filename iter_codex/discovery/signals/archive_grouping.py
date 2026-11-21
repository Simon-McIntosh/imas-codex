"""Private component and control-node groups with exact and category field patterns."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

from iter_codex.discovery.signals.variable_names import parse_variable_name
from iter_codex.graph.models import ArchiveGroupingState, SignalNameParseOutcome


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
    """Own private membership storage and atomic recomputation of one facility."""

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
            CREATE TABLE IF NOT EXISTS control_groups (
                facility TEXT, id TEXT PRIMARY KEY, code TEXT, member_count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS grouping_receipts (facility TEXT PRIMARY KEY, receipt TEXT);
        """)
        with self.db:
            for table, columns in {
                "members": ("outcome", "control_node", "category", "discriminator"),
                "components": (
                    "control_node",
                    "category",
                    "sequence",
                    "category_pattern",
                ),
                "patterns": ("category", "key_kind"),
            }.items():
                present = {
                    row[1] for row in self.db.execute(f"PRAGMA table_info({table})")
                }
                for column in columns:
                    if column not in present:
                        self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    def close(self) -> None:
        self.db.close()

    def snapshot_records(self, facility: str) -> list[dict]:
        """Materialize a replayable snapshot without exposing the table layout."""
        return [
            {"name": name, "archives": json.loads(archives)}
            for name, archives in self.db.execute(
                "SELECT name,archives FROM members WHERE facility=? ORDER BY name",
                (facility,),
            )
        ]

    def name_schema_counts(self, facility: str) -> dict:
        """Count control field values, full prefixes and physical category codes.

        A repeated second-level field is counted across all first-level parents.
        Prefix counts retain those parents as a separate, unambiguous view.
        Nonconforming selectors contribute to neither hierarchy nor categories.
        """
        levels = {1: Counter(), 2: Counter()}
        prefixes = {1: Counter(), 2: Counter()}
        for control, count in self.db.execute(
            """
            SELECT control_node,count(*) FROM members
            WHERE facility=? AND control_node IS NOT NULL GROUP BY control_node
        """,
            (facility,),
        ):
            fields = control.split(":control_node:", 1)[1].split("-")
            for level in (1, 2):
                if len(fields) >= level:
                    levels[level][fields[level - 1]] += count
                    prefixes[level]["-".join(fields[:level])] += count

        def counted(values):
            return {
                "distinct_values": len(values),
                "variables": sum(values.values()),
                "values": [
                    {"code": code, "variables": count}
                    for code, count in sorted(
                        values.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            }

        categories = dict(
            self.db.execute(
                """
            SELECT category,count(*) FROM members
            WHERE facility=? AND category IS NOT NULL GROUP BY category
        """,
                (facility,),
            )
        )
        return {
            "control_path_fields": {
                str(level): counted(values) for level, values in levels.items()
            },
            "control_path_prefixes": {
                str(level): counted(values) for level, values in prefixes.items()
            },
            "function_categories": counted(categories),
        }

    def _ids(self, table: str, facility: str) -> set[str]:
        return {
            row[0]
            for row in self.db.execute(
                f"SELECT id FROM {table} WHERE facility=?", (facility,)
            )
        }

    def rebuild(self, facility: str, records: Iterable[dict]) -> dict:
        """Classify every selector and replace all derived rows in one transaction."""
        if not facility:
            raise ValueError("a facility is required")
        previous = {
            table: self._ids(table, facility)
            for table in ("components", "control_groups", "patterns")
        }

        def member_rows():
            for record in records:
                parsed = parse_variable_name(record["name"])
                component = parsed.outcome == SignalNameParseOutcome.component_variable
                function = parsed.outcome == SignalNameParseOutcome.function_variable
                state = (
                    ArchiveGroupingState.component
                    if component
                    else ArchiveGroupingState.control_node
                    if function
                    else ArchiveGroupingState.ungrouped
                )
                yield (
                    facility,
                    parsed.selector,
                    state.value,
                    parsed.reason,
                    f"{facility}:plant_component:{parsed.plant_component_code}"
                    if component
                    else None,
                    parsed.field_code,
                    json.dumps(record.get("archives", {}), sort_keys=True),
                    parsed.outcome.value,
                    f"{facility}:control_node:{parsed.control_path}"
                    if component or function
                    else None,
                    parsed.component_category,
                    parsed.field_discriminator,
                )

        with self.db:
            for table in ("members", "components", "control_groups", "patterns"):
                self.db.execute(f"DELETE FROM {table} WHERE facility=?", (facility,))
            self.db.executemany(
                """INSERT INTO members
                (facility,name,state,reason,component,field,archives,outcome,control_node,category,discriminator)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                member_rows(),
            )
            rows = self.db.execute(
                """SELECT component,control_node,category,field FROM members
                WHERE facility=? AND state=? ORDER BY component,field""",
                (facility, ArchiveGroupingState.component.value),
            )
            for component, members in groupby(rows, key=lambda row: row[0]):
                members = list(members)
                _, control, category, _ = members[0]
                fields = sorted({row[3] for row in members})
                encoded = json.dumps(fields, separators=(",", ":"))
                digest = hashlib.sha256(
                    json.dumps([category, fields], separators=(",", ":")).encode()
                ).hexdigest()
                pattern = f"{facility}:archive_field_pattern:{digest}"
                code = component.split(":plant_component:", 1)[1]
                self.db.execute(
                    """INSERT INTO components
                    (facility,id,code,pattern,member_count,control_node,category,sequence,category_pattern)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        facility,
                        component,
                        code,
                        pattern,
                        len(members),
                        control,
                        category,
                        code.rsplit(":", 1)[1][len(category) :],
                        f"{facility}:archive_category_pattern:{category}",
                    ),
                )
                self.db.execute(
                    """INSERT INTO patterns
                    (facility,id,fields,component_count,member_count,category,key_kind) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET component_count=component_count+1,
                    member_count=member_count+excluded.member_count""",
                    (
                        facility,
                        pattern,
                        encoded,
                        1,
                        len(members),
                        category,
                        "category_fields",
                    ),
                )
            for control, count in self.db.execute(
                """SELECT control_node,count(*) FROM members
                WHERE facility=? AND state=? GROUP BY control_node""",
                (facility, ArchiveGroupingState.control_node.value),
            ):
                self.db.execute(
                    "INSERT INTO control_groups VALUES (?,?,?,?)",
                    (facility, control, control.split(":control_node:", 1)[1], count),
                )
            fields_by_category = self.db.execute(
                """SELECT category,field FROM members
                WHERE facility=? AND state=? GROUP BY category,field ORDER BY category,field""",
                (facility, ArchiveGroupingState.component.value),
            )
            for category, rows in groupby(fields_by_category, key=lambda row: row[0]):
                fields = [row[1] for row in rows]
                count, members = self.db.execute(
                    "SELECT count(*),sum(member_count) FROM components WHERE facility=? AND category=?",
                    (facility, category),
                ).fetchone()
                self.db.execute(
                    """INSERT INTO patterns
                    (facility,id,fields,component_count,member_count,category,key_kind) VALUES (?,?,?,?,?,?,?)""",
                    (
                        facility,
                        f"{facility}:archive_category_pattern:{category}",
                        json.dumps(fields),
                        count,
                        members,
                        category,
                        "category",
                    ),
                )
            receipt = self.summary(facility)
            for table, prefix in (
                ("components", "groups"),
                ("control_groups", "control_groups"),
                ("patterns", "patterns"),
            ):
                current = self._ids(table, facility)
                receipt[prefix + "_added"] = len(current - previous[table])
                receipt[prefix + "_removed"] = len(previous[table] - current)
            self.db.execute(
                "INSERT INTO grouping_receipts VALUES (?,?) ON CONFLICT(facility) DO UPDATE SET receipt=excluded.receipt",
                (facility, json.dumps(receipt)),
            )
        return receipt

    def member(self, facility: str, name: str) -> dict:
        row = self.db.execute(
            """SELECT state,reason,component,field,archives,outcome,control_node,category,discriminator
            FROM members WHERE facility=? AND name=?""",
            (facility, name),
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return dict(
            zip(
                (
                    "state",
                    "reason",
                    "component",
                    "field",
                    "archives",
                    "outcome",
                    "control_node",
                    "category",
                    "discriminator",
                ),
                (*row[:4], json.loads(row[4]), *row[5:]),
                strict=True,
            )
        )

    def component(self, component_id: str) -> dict:
        row = self.db.execute(
            """SELECT facility,code,pattern,member_count,control_node,category,sequence,category_pattern
            FROM components WHERE id=?""",
            (component_id,),
        ).fetchone()
        if row is None:
            raise KeyError(component_id)
        return {
            "id": component_id,
            **dict(
                zip(
                    (
                        "facility_id",
                        "code",
                        "archive_field_pattern",
                        "archive_member_count",
                        "control_node",
                        "component_category",
                        "component_sequence",
                        "archive_category_pattern",
                    ),
                    row,
                    strict=True,
                )
            ),
        }

    def control_group(self, node_id: str) -> dict:
        row = self.db.execute(
            "SELECT facility,code,member_count FROM control_groups WHERE id=?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return {
            "id": node_id,
            "facility_id": row[0],
            "code": row[1],
            "archive_member_count": row[2],
        }

    def pattern(self, pattern_id: str) -> dict:
        row = self.db.execute(
            "SELECT facility,fields,component_count,member_count,category,key_kind FROM patterns WHERE id=?",
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
            "component_category": row[4],
            "key_kind": row[5],
        }

    def _sizes(self, table: str, facility: str) -> dict[str, int]:
        return {
            str(size): count
            for size, count in self.db.execute(
                f"SELECT member_count,count(*) FROM {table} WHERE facility=? GROUP BY member_count ORDER BY member_count",
                (facility,),
            )
        }

    def summary(self, facility: str) -> dict:
        states = dict(
            self.db.execute(
                "SELECT state,count(*) FROM members WHERE facility=? GROUP BY state",
                (facility,),
            )
        )
        outcomes = dict.fromkeys(
            (outcome.value for outcome in SignalNameParseOutcome), 0
        )
        outcomes.update(
            dict(
                self.db.execute(
                    "SELECT outcome,count(*) FROM members WHERE facility=? GROUP BY outcome",
                    (facility,),
                )
            )
        )
        sizes = self._sizes("components", facility)
        control_sizes = self._sizes("control_groups", facility)
        total = sum(states.values())
        component_variables = states.get(ArchiveGroupingState.component.value, 0)
        component_count, control_count = (
            sum(sizes.values()),
            sum(control_sizes.values()),
        )
        counts = dict(
            self.db.execute(
                "SELECT key_kind,count(*) FROM patterns WHERE facility=? GROUP BY key_kind",
                (facility,),
            )
        )
        shared = {}
        largest = {}
        for kind, column in (
            ("category_fields", "pattern"),
            ("category", "category_pattern"),
        ):
            shared_count = self.db.execute(
                f"""SELECT coalesce(sum(c.member_count),0) FROM components c
                JOIN patterns p ON p.id=c.{column} WHERE c.facility=? AND p.component_count>=2""",
                (facility,),
            ).fetchone()[0]
            shared[kind] = (
                shared_count / component_variables if component_variables else 0
            )
            largest[kind] = [
                self.pattern(row[0])
                for row in self.db.execute(
                    "SELECT id FROM patterns WHERE facility=? AND key_kind=? ORDER BY member_count DESC,id LIMIT 20",
                    (facility, kind),
                )
            ]
        return {
            "facility": facility,
            "variables": total,
            "parse_outcomes": outcomes,
            "component_variables": component_variables,
            "function_variables": states.get(
                ArchiveGroupingState.control_node.value, 0
            ),
            "grouped": component_variables
            + states.get(ArchiveGroupingState.control_node.value, 0),
            "ungrouped": states.get(ArchiveGroupingState.ungrouped.value, 0),
            "component_groups": component_count,
            "control_node_groups": control_count,
            "field_patterns": counts.get("category_fields", 0),
            "category_patterns": counts.get("category", 0),
            "group_size_distribution": sizes,
            "control_group_size_distribution": control_sizes,
            "singleton_signal_share": (sizes.get("1", 0) + control_sizes.get("1", 0))
            / total
            if total
            else 0,
            "shared_pattern_component_variable_share": shared,
            "largest_patterns": largest["category_fields"],
            "largest_category_patterns": largest["category"],
            "group_descriptions_needed": component_count + control_count,
            "descriptions_needed": {
                kind: {
                    "component": component_count,
                    "control_node": control_count,
                    "pattern": counts.get(kind, 0),
                    "total": component_count + control_count + counts.get(kind, 0),
                }
                for kind in ("category_fields", "category")
            },
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
