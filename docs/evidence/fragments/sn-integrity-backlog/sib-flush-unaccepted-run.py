#!/usr/bin/env python3
"""Capture a flush drain's before/after state.

Read-only: graph reads only, no LLM call, no graph write.

Everything here is read through an existing instrument's own query so the
numbers reconcile with the records that already publish them:

* the accepted live DD quantity-source count and the per-class census counts
  come from :mod:`sib-unaccepted-census-run` (``source_rows`` and
  ``classify_source``), not a re-derived query;
* the export gate's blocking set and the sourceless live draft count come from
  the export gate's own :class:`ExportReport` (``gate_failures`` and
  ``sourceless_drafts.count``);
* the approved and contested rows are hashed over their full property maps in
  ID order, so a drain that mutates either cohort shows as a changed digest.

Usage::

    PYTHONPATH=. uv run --no-sync python <this-file> --phase before --out before.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CENSUS_INSTRUMENT = HERE / "sib-unaccepted-census-run.py"

GUARDED_STAGES = ("approved", "contested")


def load_census():
    """Load the unaccepted-census instrument so its own queries are reused."""
    spec = importlib.util.spec_from_file_location(
        "sn_unaccepted_census_instrument", CENSUS_INSTRUMENT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load census instrument at {CENSUS_INSTRUMENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_counts(gc: Any) -> dict[str, int]:
    rows = gc.query(
        """
        MATCH (n:StandardName)
        RETURN coalesce(n.name_stage, '') AS stage, count(*) AS count
        ORDER BY stage
        """
    )
    return {str(row["stage"]): int(row["count"]) for row in rows}


def docs_stage_counts(gc: Any) -> dict[str, int]:
    rows = gc.query(
        """
        MATCH (n:StandardName)
        RETURN coalesce(n.docs_stage, '') AS stage, count(*) AS count
        ORDER BY stage
        """
    )
    return {str(row["stage"]): int(row["count"]) for row in rows}


def excluded_source_state(gc: Any, path: str) -> dict[str, Any]:
    """Raw graph state of one DD source the census classifier cannot place.

    Reported whole rather than summarised, because a source that satisfies no
    census class is a state the reader has to judge: its status, attempt count,
    claim and every produced-name edge with that name's stages.
    """
    rows = gc.query(
        """
        MATCH (s:StandardNameSource {source_id: $path})
        OPTIONAL MATCH (s)-[r:PRODUCED_NAME]->(sn:StandardName)
        RETURN s.id AS sid, properties(s) AS props,
               collect(DISTINCT {
                   rel: type(r),
                   id: sn.id,
                   name_stage: sn.name_stage,
                   docs_stage: sn.docs_stage,
                   lifecycle: coalesce(sn.lifecycle_status, sn.lifecycle, '')
               }) AS edges
        """,
        path=path,
    )
    records = []
    for row in rows:
        props = {}
        for key, value in dict(row["props"]).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                props[key] = value if not isinstance(value, str) or len(value) <= 300 else value[:300] + "…"
            elif isinstance(value, (list, tuple)) and len(value) <= 20:
                props[key] = [_jsonable(v) for v in value]
        edges = [
            {
                k: (_jsonable(v) if k != "rel" else v)
                for k, v in dict(edge).items()
            }
            for edge in (row["edges"] or [])
        ]
        records.append({"source_id": path, "sid": row["sid"], "props": props, "edges": edges})
    return {"source_id": path, "matched_sources": len(records), "records": records}


def guarded_rows_digest(
    gc: Any, stages: tuple[str, ...] = GUARDED_STAGES
) -> dict[str, Any]:
    """ID-ordered properties digest over a cohort selected by name stage."""
    rows = gc.query(
        """
        MATCH (n:StandardName)
        WHERE n.name_stage IN $stages
        RETURN n.id AS id, n.name_stage AS stage, properties(n) AS props
        ORDER BY n.id
        """,
        stages=list(stages),
    )
    records = [
        {"id": row["id"], "stage": row["stage"], "props": dict(row["props"])}
        for row in rows
    ]
    per_stage: Counter[str] = Counter(str(r["stage"]) for r in records)
    return {
        "digest": _digest(records),
        "row_count": len(records),
        "per_stage": dict(sorted(per_stage.items())),
    }


def _digest(records: list[dict[str, Any]]) -> str:
    canonical: list[str] = []
    for record in records:
        props = {str(k): _jsonable(v) for k, v in record["props"].items()}
        canonical.append(
            json.dumps({"id": record["id"], "props": props}, sort_keys=True)
        )
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()


def sensitivity_control(gc: Any) -> dict[str, Any]:
    """Positive control for the guarded digest.

    A fresh graph holds no approved or contested rows, so an equal digest over
    that empty population proves nothing. Hash a non-empty cohort and a copy
    with one property changed: a different digest on the copy shows the
    instrument sees content and would notice a mutation, rather than returning
    the empty-string digest on everything.
    """
    rows = gc.query(
        """
        MATCH (n:StandardName {name_stage: 'accepted'})
        RETURN n.id AS id, properties(n) AS props
        ORDER BY n.id
        LIMIT 50
        """
    )
    records = [{"id": r["id"], "props": dict(r["props"])} for r in rows]
    base = _digest(records)
    mutated = [dict(r, props=dict(r["props"])) for r in records]
    if mutated:
        first_id = mutated[0]["id"]
        mutated[0]["props"]["__sensitivity_probe__"] = "changed"
    return {
        "cohort": "accepted",
        "row_count": len(records),
        "digest": base,
        "empty_string_digest": hashlib.sha256(b"").hexdigest(),
        "baseline_differs_from_empty": base != hashlib.sha256(b"").hexdigest(),
        "mutated_digest": _digest(mutated),
        "mutation_detected": _digest(mutated) != base and bool(mutated),
        "mutated_id": first_id if mutated else None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not k.startswith("_")}
    # Neo4j temporal/other scalars: their canonical string form is stable.
    return str(value)


def export_gate() -> dict[str, Any]:
    """Run the export gate and read its blocking set and sourceless drafts."""
    from imas_codex.standard_names.export import run_export

    with tempfile.TemporaryDirectory(prefix="sib-flush-gate-") as staging:
        report = run_export(Path(staging), gate_only=True)
        document = report.to_dict()
    counts = document.get("counts", {})
    sourceless = document.get("sourceless_drafts", {})
    return {
        "gate_failures": int(counts.get("gate_failures", -1)),
        "all_gates_passed": bool(document.get("all_gates_passed", False)),
        "validation_failures": int(counts.get("validation_failures", -1)),
        "sourceless_drafts_count": int(sourceless.get("count", -1)),
        "sourceless_drafts": sourceless.get("names", [])[:50],
    }


def measure(zero_spend: bool) -> dict[str, Any]:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

    census = load_census()
    with GraphClient() as gc:
        sources = census.source_rows(gc)
        accepted = [
            sid
            for sid, rec in sources.items()
            if any(
                b["bound_stage"] == "accepted" or b["produced_stage"] == "accepted"
                for b in rec["bindings"]
            )
        ]
        accepted_set = set(accepted)
        unaccepted = {
            sid: rec for sid, rec in sources.items() if sid not in accepted_set
        }
        source_classes: Counter[str] = Counter()
        excluded: dict[str, dict[str, Any]] = {}
        for rec in unaccepted.values():
            try:
                source_classes[census.classify_source(rec)] += 1
            except Exception as exc:  # a source in a state no census class covers
                excluded[str(rec["path"])] = {
                    "classifier_error": f"{type(exc).__name__}: {exc}",
                    "raw_state": excluded_source_state(gc, str(rec["path"])),
                }
        stages = stage_counts(gc)
        docs_stages = docs_stage_counts(gc)
        guarded = guarded_rows_digest(gc)
        control = sensitivity_control(gc)

    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "acceptance_floor": DEFAULT_MIN_SCORE,
        "live_sources": len(sources),
        "live_sources_accepted": len(accepted),
        "live_sources_unaccepted": len(unaccepted),
        "source_classes": {
            cls: int(source_classes.get(cls, 0)) for cls in census.CLASSES
        },
        "sources_classified": int(sum(source_classes.values())),
        "excluded_sources": excluded,
        "name_stage_counts": stages,
        "docs_stage_counts": docs_stages,
        "guarded_rows": guarded,
        "digest_sensitivity_control": control,
        "export_gate": export_gate() if not zero_spend else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-export-gate", action="store_true")
    args = parser.parse_args()

    document = {"phase": args.phase, **measure(zero_spend=args.skip_export_gate)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
