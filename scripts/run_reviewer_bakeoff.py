"""Reviewer-selection bake-off for Standard Name quality review.

Reviewing existing text against a rubric is a *judgment* task, not
generation — so cheap, vendor-diverse models may match frontier
discrimination at a fraction of the cost. This runner rates each
candidate review model on the gold-labelled SN set:

* **positives** — known-GOOD ISN names (the ``expected_name``). A good
  reviewer scores these HIGH.
* **negatives** — known-BAD anti-pattern names (``candidate_name``). A
  good reviewer scores these LOW.

The headline metric is **discrimination** = mean(positive) - mean(negative).
We also report false-accept (negative scored >= 0.5), false-reject
(positive scored < 0.8), pairwise Spearman agreement (to pick a
complementary blind pair), and cost/call. The reasoning-effort sweep
answers "can a cheap model at high effort match a pricey one at low?".

Models come from ``[tool.imas-codex.sn-benchmark].reviewer-models``.
This selects, it does NOT switch the production ``[sn-review.*]`` chain.

Outputs (under ``--output``):

- ``reviewer-bakeoff.scores.jsonl``  — per (model, effort, item) row
  (score, verdict, comment, reasoning) for independent assessment.
- ``reviewer-bakeoff.summary.json``  — per (model, effort) metrics +
  pairwise agreement matrix.
- ``reviewer-bakeoff-summary.md``    — ranked table.

Usage::

    uv run python scripts/run_reviewer_bakeoff.py \\
        --cost-cap 50 --efforts low,high --batch-size 15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.settings import get_sn_benchmark_reviewer_models

logger = logging.getLogger(__name__)

# Models with no exposed reasoning control — run at provider default only.
NO_REASONING = ("owl-alpha",)


# ── Response models ────────────────────────────────────────────────────


class ReviewScore(BaseModel):
    identifier: str = Field(..., description="The item id echoed back.")
    standard_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["pass", "revise", "fail"]
    comment: str = Field(..., max_length=300)


class ReviewBatch(BaseModel):
    scores: list[ReviewScore]


# ── Prompt ─────────────────────────────────────────────────────────────

REVIEW_SYSTEM = (
    "You are an IMAS Standard Name reviewer. For each (DD path context, "
    "proposed name) pair, score the NAME on a continuous [0, 1] rubric:\n"
    "  1.0 — grammar-compliant, controlled vocabulary, correct quantity.\n"
    "  0.8 — correct quantity, minor vocabulary slip.\n"
    "  0.5 — correct quantity but grammar/style violation (abbreviation, "
    "symbol, misplaced component token, unit baked into the name, "
    "tautological/duplicated base, missing required object qualifier).\n"
    "  0.2 — wrong quantity or identity.\n"
    "  0.0 — nonsense, IDS/symbol leakage, empty.\n"
    "Grammar reminders: component tokens are PREFIXES "
    "(toroidal_ion_velocity, not ion_velocity_toroidal); units are "
    "authoritative from the DD and must NEVER appear in the name; symbols "
    "(t_e, ip) and abbreviations are forbidden; object-scoped quantities "
    "need an of_<object> qualifier.\n"
    "Emit verdict: pass (>=0.8), revise (0.5-0.79), fail (<0.5). Echo the "
    "given identifier verbatim in the 'identifier' field."
)


def _user_message(items: list[dict]) -> str:
    lines = [
        "Score each proposed Standard Name. Return JSON "
        "`{scores: [{identifier, standard_name, score, verdict, comment}]}`. "
        "Comment <= 30 words, citing the specific reason.\n"
    ]
    for i, it in enumerate(items, 1):
        desc = (it.get("description") or "").strip()[:200] or "(no description)"
        lines.append(
            f"{i}. identifier: {it['id']}\n"
            f"   dd_path: {it.get('dd_path', '—')}    unit: {it.get('unit', '—')}    "
            f"physics_domain: {it.get('physics_domain', '—')}\n"
            f"   description: {desc}\n"
            f"   proposed_name: {it['name']}\n"
        )
    return "\n".join(lines)


# ── Gold-set loading + enrichment ──────────────────────────────────────


def _load_items(eval_path: Path) -> list[dict]:
    data = json.loads(eval_path.read_text())
    items: list[dict] = []
    for p in data.get("positives", []):
        if p.get("dd_path") in ("TODO", None, ""):
            continue
        items.append(
            {
                "id": p["dd_path"],
                "name": p["expected_name"],
                "kind": "pos",
                "dd_path": p["dd_path"],
                "unit": p.get("expected_unit", "—"),
                "physics_domain": p.get("physics_domain", ""),
                "description": "",
            }
        )
    for n in data.get("negatives", []):
        items.append(
            {
                "id": n["candidate_name"],
                "name": n["candidate_name"],
                "kind": "neg",
                "dd_path": n.get("dd_path", "—"),
                "unit": "—",
                "physics_domain": n.get("physics_domain", ""),
                "description": "",
                "anti_pattern_category": n.get("anti_pattern_category", ""),
            }
        )
    return items


def _enrich(items: list[dict]) -> None:
    """Best-effort: pull description/unit from the graph for positives."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            for it in items:
                if it["kind"] != "pos":
                    continue
                rows = list(
                    gc.query(
                        """
                        MATCH (n:IMASNode {id: $pid})
                        OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                        RETURN coalesce(n.description, '') AS description,
                               coalesce(u.id, '') AS unit
                        """,
                        pid=it["dd_path"],
                    )
                )
                if rows:
                    it["description"] = rows[0]["description"]
                    if rows[0]["unit"]:
                        it["unit"] = rows[0]["unit"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph enrichment skipped (%s); using gold-set fields.", exc)


# ── Metrics ────────────────────────────────────────────────────────────


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rho via Pearson on ranks (no scipy dependency)."""
    if len(a) < 3 or len(a) != len(b):
        return None

    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if va == 0 or vb == 0:
        return None
    return round(cov / (va * vb), 3)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-set",
        type=Path,
        default=Path("tests/standard_names/eval_sets/benchmark.json"),
    )
    ap.add_argument("--output", type=Path, default=Path("research/reviewer-bakeoff"))
    ap.add_argument("--cost-cap", type=float, default=50.0)
    ap.add_argument(
        "--efforts",
        type=str,
        default="low,high",
        help="Comma list of reasoning efforts to sweep (reasoning-capable models).",
    )
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Override comma list of models (default: [sn-benchmark].reviewer-models).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    models = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else get_sn_benchmark_reviewer_models()
    )
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]

    items = _load_items(args.eval_set)
    _enrich(items)
    n_pos = sum(1 for it in items if it["kind"] == "pos")
    n_neg = sum(1 for it in items if it["kind"] == "neg")
    logger.info(
        "Loaded %d items (%d pos, %d neg); models=%d efforts=%s",
        len(items),
        n_pos,
        n_neg,
        len(models),
        efforts,
    )

    def chunks(xs: list[dict], k: int) -> list[list[dict]]:
        return [xs[i : i + k] for i in range(0, len(xs), k)]

    scores_path = args.output / "reviewer-bakeoff.scores.jsonl"
    score_fh = scores_path.open("w")
    total_cost = 0.0
    summary: list[dict] = []
    # per (model,effort) -> {item_id: score} for agreement at the first effort
    score_maps: dict[str, dict[str, float]] = {}

    for model in models:
        model_efforts = [None] if any(tag in model for tag in NO_REASONING) else efforts
        for effort in model_efforts:
            if total_cost >= args.cost_cap:
                logger.warning("Cost cap $%.2f reached — stopping.", args.cost_cap)
                break
            tag = f"{model}@{effort or 'default'}"
            by_id: dict[str, ReviewScore] = {}
            cost = 0.0
            t0 = time.time()
            failed = False
            for batch in chunks(items, args.batch_size):
                if total_cost + cost >= args.cost_cap:
                    logger.warning("Cost cap hit mid-model for %s", tag)
                    failed = True
                    break
                try:
                    parsed, c, _ = call_llm_structured(
                        model=model,
                        messages=[
                            {"role": "system", "content": REVIEW_SYSTEM},
                            {"role": "user", "content": _user_message(batch)},
                        ],
                        response_model=ReviewBatch,
                        service="standard-names",
                        max_tokens=6000,
                        temperature=0.0,
                        reasoning_effort=effort,
                    )
                    cost += c
                    for s in parsed.scores:
                        by_id[s.identifier] = s
                except Exception as exc:  # noqa: BLE001
                    # One flaky/empty batch must not void the whole model —
                    # record partial coverage and keep going. (Reliability is
                    # itself signal: incomplete models are flagged below.)
                    logger.error("%s batch failed (continuing): %s", tag, exc)
                    failed = True
                    continue
            total_cost += cost
            latency = time.time() - t0

            # Match scores back to items; record rows.
            pos_scores, neg_scores = [], []
            smap: dict[str, float] = {}
            for it in items:
                s = by_id.get(it["id"])
                if s is None:
                    continue
                smap[it["id"]] = s.score
                (pos_scores if it["kind"] == "pos" else neg_scores).append(s.score)
                score_fh.write(
                    json.dumps(
                        {
                            "model": model,
                            "effort": effort or "default",
                            "id": it["id"],
                            "kind": it["kind"],
                            "anti_pattern": it.get("anti_pattern_category"),
                            "score": s.score,
                            "verdict": s.verdict,
                            "comment": s.comment,
                        }
                    )
                    + "\n"
                )
            score_fh.flush()
            score_maps[tag] = smap

            mp = sum(pos_scores) / len(pos_scores) if pos_scores else None
            mn = sum(neg_scores) / len(neg_scores) if neg_scores else None
            disc = (mp - mn) if (mp is not None and mn is not None) else None
            false_accept = (
                sum(1 for s in neg_scores if s >= 0.5) / len(neg_scores)
                if neg_scores
                else None
            )
            false_reject = (
                sum(1 for s in pos_scores if s < 0.8) / len(pos_scores)
                if pos_scores
                else None
            )
            row = {
                "model": model,
                "effort": effort or "default",
                "n_pos": len(pos_scores),
                "n_neg": len(neg_scores),
                "mean_pos": round(mp, 3) if mp is not None else None,
                "mean_neg": round(mn, 3) if mn is not None else None,
                "discrimination": round(disc, 3) if disc is not None else None,
                "false_accept": round(false_accept, 3)
                if false_accept is not None
                else None,
                "false_reject": round(false_reject, 3)
                if false_reject is not None
                else None,
                "cost_usd": round(cost, 4),
                "latency_s": round(latency, 1),
                "incomplete": failed,
            }
            summary.append(row)
            logger.info(
                "%-44s disc=%s fa=%s fr=%s cost=$%.4f (total=$%.2f)",
                tag,
                row["discrimination"],
                row["false_accept"],
                row["false_reject"],
                cost,
                total_cost,
            )
        else:
            continue
        break  # cost cap hit in inner loop

    score_fh.close()

    # ── Pairwise agreement (Spearman) at the first swept effort ──────────
    first_effort = efforts[0] if efforts else "default"
    agree_tags = [
        t
        for t in score_maps
        if t.endswith(f"@{first_effort}") or t.endswith("@default")
    ]
    agreement: dict[str, dict[str, float | None]] = {}
    for ta in agree_tags:
        agreement[ta] = {}
        for tb in agree_tags:
            if ta >= tb:
                continue
            common = sorted(set(score_maps[ta]) & set(score_maps[tb]))
            a = [score_maps[ta][i] for i in common]
            b = [score_maps[tb][i] for i in common]
            agreement[ta][tb] = _spearman(a, b)

    out = {
        "eval_set": str(args.eval_set),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "efforts": efforts,
        "total_cost_usd": round(total_cost, 4),
        "rows": sorted(
            summary,
            key=lambda r: (r["discrimination"] is None, -(r["discrimination"] or 0)),
        ),
        "agreement_spearman": agreement,
    }
    (args.output / "reviewer-bakeoff.summary.json").write_text(
        json.dumps(out, indent=2)
    )

    # ── Markdown ─────────────────────────────────────────────────────────
    md = ["# Reviewer bake-off — summary", ""]
    md.append(
        f"- gold set: {n_pos} positives / {n_neg} negatives  "
        f"| efforts: {efforts}  | total cost: ${total_cost:.2f}"
    )
    md.append("")
    md.append(
        "Ranked by discrimination = mean(positive) − mean(negative). "
        "Lower false-accept and false-reject are better."
    )
    md.append("")
    md.append(
        "| Model | effort | disc | mean_pos | mean_neg | false_accept | false_reject | $/run | done |"
    )
    md.append(
        "|-------|--------|-----:|---------:|---------:|-------------:|-------------:|------:|:----:|"
    )
    for r in out["rows"]:
        md.append(
            f"| `{r['model']}` | {r['effort']} | {r['discrimination']} | "
            f"{r['mean_pos']} | {r['mean_neg']} | {r['false_accept']} | "
            f"{r['false_reject']} | {r['cost_usd']} | {'✗' if r['incomplete'] else '✓'} |"
        )
    md.append("")
    md.append("## Pairwise Spearman agreement (blind-pair selection)")
    md.append("")
    md.append(
        "Lower agreement between two ACCURATE models = more complementary "
        "blind pair (disagreement meaningfully triggers the breaker)."
    )
    md.append("")
    for ta, row in agreement.items():
        for tb, rho in row.items():
            md.append(f"- `{ta}` vs `{tb}`: ρ = {rho}")
    md.append("")
    (args.output / "reviewer-bakeoff-summary.md").write_text("\n".join(md))

    logger.info("Wrote %s", args.output / "reviewer-bakeoff-summary.md")
    logger.info("Total cost: $%.4f", total_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
