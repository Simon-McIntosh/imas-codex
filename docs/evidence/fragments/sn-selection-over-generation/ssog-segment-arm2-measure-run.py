"""Measure the second grammar-segment selection arm against the composer on golden A.

Driver for the ``ssog-segment-arm2-measure`` node, copied from the first arm's
driver unchanged except for what the second arm needs.  The second arm is the
one ``select_segments`` now implements: a mandatory base offered over an
embedding shortlist, per-option meaning text, and each later slot offered only
tokens that compose with those already chosen.  It loads golden A
(``golden-name-acceptable.json``) and the human ``sn edit`` rename targets from
the live graph, resolves each target name to its DD source through the name's
``PRODUCED_NAME`` provenance, runs the selection once per source, and reports
the figures the plan fixes before the run: exact-name match, per-slot accuracy,
family agreement, the composer's figures, cost per name, and the
composer-against-selection disagreement AUC with its bootstrap interval.

One arm-2 addition: the source carries ``expected_base``, the target's gold base
token, so the item records whether that base reached the shortlist.  That is the
retrieval miss the arm reports on, and it is a figure only this arm has.

Metrics drawn from :mod:`imas_codex.discovery.base.evaluation`:
``load_labelled_fixture`` (the golden A cohort), ``paired_auc`` and
``bootstrap_interval`` (the cross-check separation), and
``cost_latency_summary`` (cost and latency).  Per-slot accuracy delegates its
slot comparison to :func:`imas_codex.standard_names.benchmark_roles.exact_match_accuracy`,
the project's own match, rather than re-implementing it.  The remaining figures
are driver-local helpers: exact-name match (:func:`_figures`), family agreement
(:func:`_family_agreement`), and the composer's agreement and per-slot accuracy
(:func:`_aggregate`).  Nothing here writes graph state.

Usage::

    python ssog-segment-selection-measure-run.py --out <results.json> [--limit N] [--pilot]
    python ssog-segment-selection-measure-run.py --out <results.json> --figure-only \
        --figure-out <svg>
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
        # The target's own base token, so the arm records whether its shortlist
        # offered it: the retrieval miss, distinct from a judgement miss.
        src = dict(src, expected_base=gold.get("physical_base"))
        pending.append((c, src, gold))

    # Resume: a prior partial write keeps every record already measured, so a
    # cut-short run costs nothing to finish.
    if Path(out_path).exists():
        try:
            prior = json.loads(Path(out_path).read_text())
            pr = prior.get("records", [])
            # A written record carries ``jev_chosen``; the artifact omits the
            # derived ``gold`` and ``composer_segments``, so a resume re-derives
            # them from the written ``target`` and ``composer`` rather than
            # re-paying for the selection.
            if pr and "jev_chosen" in pr[0]:
                for r in pr:
                    _rederive_record(r)
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
                    "base_in_shortlist": item.base_in_shortlist,
                    "shortlist_size": len(item.shortlist),
                    "gold": gold,
                    "composer": c["composer"],
                    "composer_segments": composer_gold,
                    "exact_match": (item.name == c["target"]) if item.name else False,
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


def _rederive_record(r: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a written record's derived fields in place.

    The artifact keeps only the measured fields; ``gold`` and
    ``composer_segments`` are re-derived here from the written ``target`` and
    ``composer`` so a resumed run needs no further paid call.
    """
    r["gold"] = _gold_segments(r["target"])
    r["composer_segments"] = (
        _gold_segments(r["composer"]) if r.get("composer") else None
    )
    return r


#: Sentinel for a prediction that is absent outright -- an unparseable composer
#: spelling supplies no segment at all.  Identity comparison means it equals no
#: token, ``None`` included, so such a spelling is a mismatch on every slot
#: rather than being credited on each slot whose gold token is empty.
_MISSING_PREDICTION = object()


def _slot_accuracy(
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> dict[str, float]:
    """Per-slot accuracy over (predicted, expected) segment-dict pairs.

    Delegates the slot comparison to
    :func:`imas_codex.standard_names.benchmark_roles.exact_match_accuracy`, the
    project's own match, rather than re-implementing it.  Its comparison is
    ``predicted.get(slot) == expected[slot]``, so a predicted ``None`` equals no
    token.  An absent prediction is therefore forwarded as
    :data:`_MISSING_PREDICTION`, not as ``None``: forwarding ``None`` would let
    an unparseable spelling match every slot whose gold token is itself empty.
    """
    from imas_codex.standard_names.benchmark_roles import exact_match_accuracy

    if not pairs:
        return dict.fromkeys(SLOTS, 0.0)
    out: dict[str, float] = {}
    for s in SLOTS:
        hits = [
            exact_match_accuracy(
                {s: (p.get(s) if p is not None else _MISSING_PREDICTION)},
                {s: (g.get(s) if g else None)},
            )[0]
            for p, g in pairs
        ]
        out[s] = round(sum(hits) / len(hits), 4)
    return out


def _figures(records: list[dict[str, Any]], label: int | None) -> dict[str, Any]:
    subset = records if label is None else [r for r in records if r["label"] == label]
    if not subset:
        return {"n": 0}
    return {
        "n": len(subset),
        "exact_match_rate": round(
            sum(1 for r in subset if r["exact_match"]) / len(subset), 4
        ),
        "slot_accuracy": _slot_accuracy([(r["jev_chosen"], r["gold"]) for r in subset]),
    }


def _family_agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Within each species family, does every member get the same base/operator/locus?

    Reported for the two cohorts the record needs.  ``golden_positives`` is the
    plan's "positives"; ``label_1_all`` is every record labelled positive, which
    is the golden positives plus the human-edit targets.  The plan's generator
    rule is stated over "the positives", so it reads ``golden_positives``; the
    wider cohort is reported because it is the natural reading of "label == 1"
    and the two give different family counts.
    """

    def _stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        families: dict[str, list[dict[str, Any]]] = {}
        for r in subset:
            families.setdefault(_family_key(r["gold"]), []).append(r)
        multi = {k: v for k, v in families.items() if len(v) > 1}
        agree = sum(
            1
            for members in multi.values()
            if len(
                {
                    tuple(
                        m["jev_chosen"].get(s) or None
                        for s in ("physical_base", "operator", "locus")
                    )
                    for m in members
                }
            )
            == 1
        )
        return {
            "records": len(subset),
            "families": len(families),
            "families_with_multiple_members": len(multi),
            "families_uniform": agree,
            "agreement_rate": round(agree / len(multi), 4) if multi else None,
        }

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
        "golden_positives": _stats(
            [r for r in records if r["kind"] == "golden" and r["label"] == 1]
        ),
        "label_1_all": _stats([r for r in records if r["label"] == 1]),
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


def _retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the target base survived the arm's embedding shortlist.

    A base the shortlist never offered is a retrieval miss, told apart from a
    judgement miss (offered, not chosen) by the item's own ``base_in_shortlist
    `` flag.  ``None`` means the item carried no expected base to check.
    """
    scored = [r for r in records if r.get("base_in_shortlist") is not None]
    hit = sum(1 for r in scored if r["base_in_shortlist"])
    return {
        "n_checked": len(scored),
        "in_shortlist": hit,
        "retrieval_miss": len(scored) - hit,
        "retrieval_hit_rate": round(hit / len(scored), 4) if scored else None,
        "unchecked": len(records) - len(scored),
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

    # The composer comparator runs over the SAME edit targets Jev selects over.
    # A composer spelling that does not parse canonically supplies no segment, so
    # it matches no slot: those rows stay in the denominator rather than being
    # dropped, and the parsable subset is reported beside the full set. The
    # missing prediction is passed as _MISSING_PREDICTION so it is a mismatch even
    # on a slot whose gold token is empty, which a plain None would have matched.
    composer_pairs = [(r["composer_segments"], r["gold"]) for r in edits]
    composer_pairs_parsable = [(p, g) for p, g in composer_pairs if p is not None]

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
            "slot_accuracy": _slot_accuracy(composer_pairs),
            "n_parsed": len(composer_pairs_parsable),
            "n_unparsed": len(composer_pairs) - len(composer_pairs_parsable),
            "slot_accuracy_parsable": _slot_accuracy(composer_pairs_parsable),
        },
        "family_agreement": _family_agreement(records),
        "cross_check": _cross_check(records),
        "retrieval": _retrieval(records),
        "cost": cost,
        "cost_per_name_usd": round(cost["total_cost"] / len(records), 6)
        if records
        else 0.0,
        "total_cost_usd": cost["total_cost"],
        "records": records,
    }


#: Per-record fields the artifact omits because each is derivable from a written
#: field: ``gold`` by parsing ``target``, and ``composer_segments`` by parsing
#: ``composer``.  Omitting them keeps the committed artifact under the data-file
#: size the repository allows.
_DERIVED_RECORD_FIELDS = ("gold", "composer_segments")


def _write_results(result: dict[str, Any], out_path: str) -> None:
    slim = dict(result)
    slim["records"] = [
        {k: v for k, v in r.items() if k not in _DERIVED_RECORD_FIELDS}
        for r in result.get("records", [])
    ]
    slim["_note"] = (
        "_note: gold and composer_segments are omitted as derivable: "
        "gold by parsing target, composer_segments by parsing composer."
    )
    Path(out_path).write_text(json.dumps(slim, separators=(",", ":"), default=str))


def _write_figure(
    results_path: str, out_path: str, arm1_results_path: str | None = None
) -> None:
    """Render the per-slot accuracy figure from written results JSONs.

    Reads the committed artifacts, so the SVG regenerates without re-running any
    selection.  One grouped bar per grammar slot, on the human-edit targets:
    the second arm (this run), the first arm (read from its own results JSON),
    and the composer on the same edit targets.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("data-ink")
    d = json.loads(Path(results_path).read_text())
    n2 = d["jev_exact_match_edits"]["n"]
    arm2_edt = [d["jev_exact_match_edits"]["slot_accuracy"][s] for s in SLOTS]
    cmp_edt = [d["composer_exact_match_edits"]["slot_accuracy"][s] for s in SLOTS]
    if arm1_results_path:
        a1 = json.loads(Path(arm1_results_path).read_text())
        n1 = a1["jev_exact_match_edits"]["n"]
        arm1_edt = [a1["jev_exact_match_edits"]["slot_accuracy"][s] for s in SLOTS]
    else:
        n1 = n2
        arm1_edt = [0.0] * len(SLOTS)
    labels = ["physical\nbase", "subject", "component", "operator", "locus", "process"]
    series = (
        (arm2_edt, "#3b6ea5", f"second arm, human-edit targets (n={n2})"),
        (arm1_edt, "#8fb0d4", f"first arm, human-edit targets (n={n1})"),
        (cmp_edt, "#c1663f", f"composer, human-edit targets (n={n2})"),
    )
    x = np.arange(len(SLOTS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 6.4), dpi=100)
    for i, (vals, colour, label) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colour, label=label)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=9,
                ha="center",
                va="bottom",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("per-slot accuracy")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=12,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.14),
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--figure-out", default=None)
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="render --figure-out from the existing --out and skip Jev entirely",
    )
    ap.add_argument(
        "--arm1-results",
        default=None,
        help="the first arm's committed results JSON, for the comparison figure",
    )
    args = ap.parse_args()

    if args.figure_only:
        if not args.figure_out:
            ap.error("--figure-only requires --figure-out")
        _write_figure(args.out, args.figure_out, args.arm1_results)
        print("WROTE", args.figure_out)
        return 0

    result = asyncio.run(measure(args.limit, args.pilot, args.out))
    _write_results(result, args.out)
    print("WROTE", args.out)
    if args.figure_out:
        _write_figure(args.out, args.figure_out, args.arm1_results)
        print("WROTE", args.figure_out)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "records"}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
