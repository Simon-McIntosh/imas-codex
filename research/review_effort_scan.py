"""Review-effort scan: does a lower reviewer reasoning budget still catch bad names?

Loads a FIXED set of composed candidates plus the physics-judge faithfulness
verdicts (the accuracy oracle) from an existing benchmark report, then scores
those same candidates with a FIXED reviewer model at several reasoning-effort
levels. Because the candidates and the oracle are held constant, the only
variable is the reviewer's effort — so the discrimination delta is attributable
to effort alone (no compose confound, no local-vLLM contention since the
reviewer is a paid model).

Discrimination metric: the gap between the reviewer's mean score on
physics-FAITHFUL names and on physics-UNFAITHFUL names. A good reviewer scores
faithful names higher than unfaithful ones; a larger gap = better separation.
Also reports how many unfaithful names the reviewer would REJECT at a 0.6
threshold (bad-name catching) plus cost and wall-time per effort level.

Usage:
    uv run python research/review_effort_scan.py <benchmark.json> \
        [--reviewer openrouter/qwen/qwen3.7-max] \
        [--efforts none,low,medium,high] [--out research/review_effort_scan.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anyio

from imas_codex.standard_names.benchmark import score_with_reviewer

_PASS_THRESHOLD = 0.6


def _load_candidates_and_oracle(report_path: str) -> tuple[list[dict], dict[str, bool]]:
    """Return (candidates, {name: faithful}) from a benchmark report JSON."""
    d = json.loads(Path(report_path).read_text())
    results = d.get("results") or []
    # Use the first model result that carries both candidates and verdicts.
    for r in results:
        cands = r.get("candidates") or []
        verdicts = {
            v.get("name"): bool(v.get("faithful"))
            for v in (r.get("physics_verdicts") or [])
            if v.get("name")
        }
        if cands and verdicts:
            return cands, verdicts
    # Fall back to candidates only (no oracle) from the first non-empty result.
    for r in results:
        if r.get("candidates"):
            return r["candidates"], {}
    raise SystemExit(f"No candidates found in {report_path}")


def _resolve_name(cand: dict) -> str:
    return cand.get("standard_name") or cand.get("id") or ""


async def _scan(
    candidates: list[dict],
    oracle: dict[str, bool],
    reviewer: str,
    efforts: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for effort in efforts:
        t0 = time.monotonic()
        reviews, cost = await score_with_reviewer(
            candidates, reviewer, target="names", reasoning_effort=effort
        )
        elapsed = round(time.monotonic() - t0, 1)
        by_name = {r.get("name"): r.get("score") for r in reviews}
        faithful_scores: list[float] = []
        unfaithful_scores: list[float] = []
        caught = 0
        unfaithful_total = 0
        for cand in candidates:
            nm = _resolve_name(cand)
            sc = by_name.get(nm)
            if sc is None or nm not in oracle:
                continue
            if oracle[nm]:
                faithful_scores.append(sc)
            else:
                unfaithful_scores.append(sc)
                unfaithful_total += 1
                if sc < _PASS_THRESHOLD:
                    caught += 1

        def _mean(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 3) if xs else 0.0

        rows.append(
            {
                "effort": effort,
                "n_reviewed": len(by_name),
                "mean_faithful": _mean(faithful_scores),
                "mean_unfaithful": _mean(unfaithful_scores),
                "discrimination": round(
                    _mean(faithful_scores) - _mean(unfaithful_scores), 3
                ),
                "bad_caught": f"{caught}/{unfaithful_total}",
                "cost_usd": round(cost, 4),
                "elapsed_s": elapsed,
            }
        )
        print(
            f"  effort={effort:7} mean_faithful={rows[-1]['mean_faithful']:.3f} "
            f"mean_unfaithful={rows[-1]['mean_unfaithful']:.3f} "
            f"discrim={rows[-1]['discrimination']:+.3f} "
            f"bad_caught={rows[-1]['bad_caught']} ${cost:.4f} {elapsed}s"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="Benchmark report JSON with candidates + verdicts")
    ap.add_argument("--reviewer", default="openrouter/qwen/qwen3.7-max")
    ap.add_argument("--efforts", default="none,low,medium,high")
    ap.add_argument("--out", default="research/review_effort_scan.json")
    args = ap.parse_args()

    candidates, oracle = _load_candidates_and_oracle(args.report)
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    print(
        f"Review-effort scan: {len(candidates)} candidates, "
        f"{sum(1 for v in oracle.values() if not v)} unfaithful in oracle, "
        f"reviewer={args.reviewer}"
    )
    rows = anyio.run(_scan, candidates, oracle, args.reviewer, efforts)
    Path(args.out).write_text(
        json.dumps(
            {
                "reviewer": args.reviewer,
                "source_report": args.report,
                "n_candidates": len(candidates),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
