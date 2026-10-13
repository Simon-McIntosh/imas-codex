"""Measure Jev's grammar-segment selection against the composer on golden A.

Driver for the ``ssog-segment-selection-measure`` node.  It loads golden A
(``golden-name-acceptable.json``) and the human ``sn edit`` rename targets from
the live graph, resolves each target name to its DD source through the name's
``PRODUCED_NAME`` provenance, runs Jev's selection
(:func:`imas_codex.standard_names.segment_selection.select_segments`) once per
source, and reports the figures the plan fixes before the run: exact-name match,
per-slot accuracy, family agreement, the composer's figures, cost per name, and
the composer-against-Jev disagreement AUC with its bootstrap interval.

Every reported metric comes from :mod:`imas_codex.discovery.base.evaluation`;
this driver adds no metric of its own.  Nothing here writes graph state.

Usage::

    python ssog-segment-selection-measure-run.py --out <results.json> [--limit N] [--pilot]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
GOLDEN = (
    REPO / "docs/evidence/fragments/sn-judgment-evaluation/golden-name-acceptable.json"
)

SLOTS = ("physical_base", "subject", "component", "operator", "locus", "process")

#: The seventeen-member boundary-density family the plan names by hand.
BOUNDARY_FAMILY_SIGNATURE = "_density_at_plasma_boundary"


def _load_golden() -> list[dict[str, Any]]:
    from imas_codex.discovery.base.evaluation import load_labelled_fixture

    return load_labelled_fixture(str(GOLDEN))


def _human_edit_targets() -> list[dict[str, Any]]:
    """New spellings of human ``sn edit`` renames, with the pre-edit spelling.

    The pre-edit spelling is the composer's recorded spelling for that source
    (what the human replaced), so it is not circular the way golden A's own
    positives are.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:StandardNameChange {operation: 'human_edit'})
            WHERE c.to_name IS NOT NULL AND c.from_name IS NOT NULL
              AND c.to_name <> c.from_name
            RETURN c.to_name AS to_name, c.from_name AS from_name, c.changed_at AS at
            ORDER BY c.changed_at DESC
            """
        )
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        to_name = r["to_name"]
        if to_name not in seen:
            seen[to_name] = {"target": to_name, "composer": r["from_name"]}
    return [seen[k] for k in sorted(seen)]


def _resolve_sources(names: list[str], batch: int = 200) -> dict[str, dict[str, Any]]:
    """One bounded, indexed read per batch: name -> its first DD source."""
    from imas_codex.graph.client import GraphClient

    out: dict[str, dict[str, Any] | None] = {}
    with GraphClient() as gc:
        for start in range(0, len(names), batch):
            chunk = names[start : start + batch]
            rows = gc.query(
                """
                UNWIND $names AS nm
                MATCH (sn:StandardName {id: nm})
                OPTIONAL MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                RETURN nm AS name,
                       collect({dd_path: sns.dd_path,
                                description: sns.description,
                                enhanced: sns.enhanced_description,
                                documentation: sns.dd_documentation}) AS sources
                """,
                names=chunk,
            )
            for r in rows:
                srcs = [s for s in (r["sources"] or []) if s.get("dd_path")]
                out[r["name"]] = srcs[0] if srcs else None
    return out


def _source_doc(src: dict[str, Any] | None) -> dict[str, Any] | None:
    if not src:
        return None
    return {
        "path": src["dd_path"],
        "description": src.get("enhanced") or src.get("description") or "",
        "documentation": src.get("documentation") or "",
    }


def _gold_segments(name: str) -> dict[str, str | None] | None:
    """Parse a canonical name into its per-slot gold tokens, or None if it does not parse."""
    from imas_codex.standard_names.grammar_adapter import parse_canonical_name

    try:
        ir = parse_canonical_name(name).ir
    except (ValueError, TypeError):
        return None
    quals = [q.token for q in (ir.qualifiers or [])]
    return {
        "physical_base": ir.base.token if ir.base else None,
        "subject": quals[0] if quals else None,
        "component": ir.projection.axis if ir.projection else None,
        "operator": ir.operators[0].op if ir.operators else None,
        "locus": ir.locus.token if ir.locus else None,
        "process": ir.mechanism.token if ir.mechanism else None,
    }


def _family_key(gold: dict[str, str | None]) -> str:
    """Species family = the name's slot tokens with the species (subject) removed."""
    return "|".join("" if s == "subject" else str(gold.get(s)) for s in SLOTS)


def _agreement(a: dict[str, str | None], b: dict[str, str | None]) -> float:
    """Fraction of the six slots where two selections carry the same token."""
    return sum(1 for s in SLOTS if (a.get(s) or None) == (b.get(s) or None)) / len(
        SLOTS
    )


async def _select_chunk(
    sources: list[dict[str, Any]], concurrency: int
) -> tuple[list[Any], float, float]:
    from imas_codex.standard_names.segment_selection import select_segments

    t0 = time.monotonic()
    run = await select_segments(sources, concurrency=concurrency)
    return run.items, run.cost_usd, time.monotonic() - t0


async def measure(limit: int | None, pilot: bool, out_path: str) -> dict[str, Any]:

    rows = _load_golden()
    edits = _human_edit_targets()

    # Cohort: every golden row (positives and negatives; the AUC needs both)
    # plus every human-edit target.
    cohort: list[dict[str, Any]] = []
    for r in rows:
        cohort.append(
            {
                "kind": "golden",
                "target": r["canonical_unit"],
                "composer": r["canonical_unit"],
                "label": int(r["label"]),
                "copy_group": r.get("copy_group"),
            }
        )
    for e in edits:
        cohort.append(
            {
                "kind": "edit",
                "target": e["target"],
                "composer": e["composer"],
                "label": 1,
                "copy_group": None,
            }
        )

    if pilot:
        cohort = cohort[:10]
    elif limit:
        cohort = cohort[:limit]

    names = sorted({c["target"] for c in cohort})
    src_map = _resolve_sources(names)

    records: list[dict[str, Any]] = []
    excluded_no_source = 0
    excluded_no_parse = 0
    calls: list[dict[str, float]] = []

    # Resolve and parse before any paid call, so excluded rows cost nothing.
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, str | None]]] = []
    for c in cohort:
        src = _source_doc(src_map.get(c["target"]))
        if src is None:
            excluded_no_source += 1
            continue
        gold = _gold_segments(c["target"])
        if gold is None:
            excluded_no_parse += 1
            continue
        pending.append((c, src, gold))

    # Resume: a prior partial write keeps every record already measured, so a
    # cut-short run costs nothing to finish.
    if Path(out_path).exists():
        try:
            prior = json.loads(Path(out_path).read_text())
            pr = prior.get("records", [])
            # The artifact omits derived per-record fields, so a compact file
            # cannot seed a resume that needs them.
            if pr and "slot_match" in pr[0]:
                done_targets = {r["target"] for r in pr}
                records.extend(pr)
                calls.extend(prior.get("_calls", []))
                pending = [p for p in pending if p[0]["target"] not in done_targets]
                print(f"resume: {len(records)} records already present", flush=True)
        except (ValueError, KeyError):
            pass

    concurrency = 32
    chunk_size = 32
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        items, cost, latency = await _select_chunk(
            [s for _, s, _ in chunk], concurrency
        )
        calls.append({"cost": cost, "latency_s": latency})
        for (c, src, gold), item in zip(chunk, items, strict=True):
            chosen = item.chosen()
            composer_gold = _gold_segments(c["composer"])
            records.append(
                {
                    "kind": c["kind"],
                    "target": c["target"],
                    "label": c["label"],
                    "copy_group": c["copy_group"],
                    "source_path": src["path"],
                    "jev_name": item.name,
                    "jev_failure": item.failure,
                    "jev_chosen": chosen,
                    "gold": gold,
                    "composer": c["composer"],
                    "composer_segments": composer_gold,
                    "exact_match": (item.name == c["target"]) if item.name else False,
                    "slot_match": {
                        s: (chosen.get(s) or None) == gold.get(s) for s in SLOTS
                    },
                    "agreement_with_composer": (
                        _agreement(chosen, composer_gold) if composer_gold else None
                    ),
                }
            )
        print(
            f"chunk {start // chunk_size + 1}: cost={cost:.6f} records={len(records)}",
            flush=True,
        )
        payload = _aggregate(
            records, calls, excluded_no_source, excluded_no_parse, len(cohort)
        )
        payload["_calls"] = calls
        _write_results(payload, out_path)

    payload = _aggregate(
        records, calls, excluded_no_source, excluded_no_parse, len(cohort)
    )
    payload["_calls"] = calls
    _write_results(payload, out_path)
    return payload


def _slot_accuracy(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return dict.fromkeys(SLOTS, 0.0)
    return {
        s: round(sum(1 for r in records if r["slot_match"][s]) / len(records), 4)
        for s in SLOTS
    }


def _figures(records: list[dict[str, Any]], label: int | None) -> dict[str, Any]:
    subset = records if label is None else [r for r in records if r["label"] == label]
    if not subset:
        return {"n": 0}
    return {
        "n": len(subset),
        "exact_match_rate": round(
            sum(1 for r in subset if r["exact_match"]) / len(subset), 4
        ),
        "slot_accuracy": _slot_accuracy(subset),
    }


def _family_agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Within each species family, do all members receive the same base/operator/locus?"""
    families: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["label"] != 1:
            continue
        families.setdefault(_family_key(r["gold"]), []).append(r)
    multi = {k: v for k, v in families.items() if len(v) > 1}
    agree = 0
    for members in multi.values():
        combos = {
            tuple(
                m["jev_chosen"].get(s) or None
                for s in ("physical_base", "operator", "locus")
            )
            for m in members
        }
        agree += 1 if len(combos) == 1 else 0
    boundary = [r for r in records if r["target"].endswith(BOUNDARY_FAMILY_SIGNATURE)]
    boundary_uniform = None
    if boundary:
        combos = {
            tuple(
                r["jev_chosen"].get(s) or None
                for s in ("physical_base", "operator", "locus")
            )
            for r in boundary
        }
        boundary_uniform = len(combos) == 1
    return {
        "families_with_multiple_members": len(multi),
        "families_uniform": agree,
        "agreement_rate": round(agree / len(multi), 4) if multi else None,
        "boundary_density_family_n": len(boundary),
        "boundary_density_uniform": boundary_uniform,
        "boundary_density_combos": sorted(
            {
                tuple(
                    r["jev_chosen"].get(s) or None
                    for s in ("physical_base", "operator", "locus")
                )
                for r in boundary
            },
            key=str,
        )
        if boundary
        else [],
    }


def _cross_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    from imas_codex.discovery.base.evaluation import bootstrap_interval, paired_auc

    usable = [
        r
        for r in records
        if r["kind"] == "golden" and r["agreement_with_composer"] is not None
    ]
    if not usable:
        return {"n": 0}
    labels = [r["label"] for r in usable]
    scores = [r["agreement_with_composer"] for r in usable]
    groups = [r["copy_group"] for r in usable]
    if len(set(labels)) < 2:
        return {"n": len(usable), "note": "only one label class present"}
    # The cross-check signal is composer-against-Jev agreement.  paired_auc
    # reports it against its mirror (disagreement) over the same units, so the
    # reported disagreement AUC is 1 - agreement AUC; both are the same rank
    # statistic up to sign.
    disagreement = [1.0 - s for s in scores]
    paired = paired_auc(labels, disagreement, scores)
    interval = bootstrap_interval(
        labels, disagreement, copy_groups=groups, n_resamples=1000, seed=123
    )
    return {
        "n": len(usable),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "disagreement_auc": paired["auc_a"],
        "agreement_auc": paired["auc_b"],
        "difference": paired["difference"],
        "bootstrap": interval,
    }


def _aggregate(
    records: list[dict[str, Any]],
    calls: list[dict[str, float]],
    excluded_no_source: int,
    excluded_no_parse: int,
    cohort_size: int,
) -> dict[str, Any]:
    from imas_codex.discovery.base.evaluation import cost_latency_summary

    positives = [r for r in records if r["kind"] == "golden" and r["label"] == 1]
    edits = [r for r in records if r["kind"] == "edit"]
    cost = cost_latency_summary(calls)

    composer_edits = []
    for r in edits:
        cs = r["composer_segments"]
        if cs is None:
            continue
        composer_edits.append(
            {
                **r,
                "slot_match": {
                    s: (cs.get(s) if cs else None) == r["gold"].get(s) for s in SLOTS
                },
            }
        )

    return {
        "cohort_size": cohort_size,
        "measured": len(records),
        "excluded_no_dd_source": excluded_no_source,
        "excluded_no_parse": excluded_no_parse,
        "jev_exact_match_positives": _figures(positives, None),
        "jev_exact_match_edits": _figures(edits, None),
        "composer_exact_match_edits": {
            "n": len(edits),
            "exact_match_rate": round(
                sum(1 for r in edits if r["composer"] == r["target"]) / len(edits), 4
            )
            if edits
            else 0.0,
            "slot_accuracy": _slot_accuracy(composer_edits),
        },
        "family_agreement": _family_agreement(records),
        "cross_check": _cross_check(records),
        "cost": cost,
        "cost_per_name_usd": round(cost["total_cost"] / len(records), 6)
        if records
        else 0.0,
        "total_cost_usd": cost["total_cost"],
        "records": records,
    }


#: Per-record fields the artifact omits because they are derivable: ``gold`` by
#: parsing ``target``, ``slot_match`` from ``jev_chosen`` against ``gold``, and
#: ``composer_segments`` by parsing ``composer``.  Dropping them keeps the
#: committed results stay under the data-file size the repository allows.
_DERIVED_RECORD_FIELDS = ("gold", "slot_match", "composer_segments")


def _write_results(result: dict[str, Any], out_path: str) -> None:
    slim = dict(result)
    slim["records"] = [
        {k: v for k, v in r.items() if k not in _DERIVED_RECORD_FIELDS}
        for r in result.get("records", [])
    ]
    slim["_note"] = (
        "_note: gold, slot_match and composer_segments are omitted as derivable: "
        "gold by parsing target, slot_match from jev_chosen vs gold, "
        "composer_segments by parsing composer."
    )
    Path(out_path).write_text(json.dumps(slim, separators=(",", ":"), default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    result = asyncio.run(measure(args.limit, args.pilot, args.out))
    _write_results(result, args.out)
    print("WROTE", args.out)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "records"}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
