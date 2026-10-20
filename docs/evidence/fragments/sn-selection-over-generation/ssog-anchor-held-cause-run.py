"""Attribute the 0.849 documentation hold on each rewritten anchor member.

The anchor A/B of the Jev-anchor campaign recorded a documentation score of
0.849 for one member in most arms. That value is the acceptance floor minus one
thousandth, reachable by two production paths: the documentation-review hold
clamp (``_documentation_conformance_note``, ``_documentation_banned_prose_note``
and ``_documentation_contradiction_note`` in ``imas_codex/standard_names/
workers.py``) and the error-severity lint cap
(``_cap_docs_scores_for_lint_errors`` in ``imas_codex/standard_names/review/
pipeline.py``).

This driver reads every arm record whose ``scores_after`` holds 0.849, takes the
member that held, and reports:

* the hold note the production guard actually emitted, read from the arm's own
  ``apply-*.log`` beside the ``persist_reviewed_docs`` line that recorded the
  0.849 score;
* whether that hold is the documentation really breaking the rule or an
  artefact of the member's shape, decided by re-running the three note
  functions through their own code paths on the member's prompt-bearing item;
* whether the lint cap fired at all, counted from its own log line.

Read-only against the graph: no write, no product-code change. The only spend
is the DD-context enrichment the review path itself performs.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

ARM_DIR = REPO / "docs/evidence/fragments/sn-selection-over-generation/ssog-anchor-ab"
RUNS = Path.home() / ".config/reckon/crew/runs"
HELD = 0.849


def held_members() -> list[dict]:
    """Every arm record whose scores_after holds a member at the 0.849 floor."""
    out = []
    for path in sorted(glob.glob(str(ARM_DIR / "*.json"))):
        if os.path.basename(path) == "selection.json":
            continue
        try:
            rec = json.load(open(path))
        except json.JSONDecodeError:
            continue
        after = rec.get("scores_after") or {}
        for member, score in after.items():
            if score == HELD:
                out.append(
                    {
                        "arm_record": os.path.basename(path),
                        "family": rec.get("family"),
                        "arm": rec.get("arm"),
                        "anchor": rec.get("anchor"),
                        "member": member,
                        "score_before": (rec.get("scores_before") or {}).get(member),
                        "score_after": score,
                        "lint_after_classes": rec.get("lint_after_classes"),
                        "run_ids": rec.get("run_ids"),
                    }
                )
    return out


def arm_logs() -> list[str]:
    """Every apply log belonging to an anchor A/B arm, by its run directory."""
    logs = []
    for pattern in ("*ssog*/**/apply-*.log", "*ssog*/apply-*.log"):
        logs.extend(glob.glob(str(RUNS / pattern), recursive=True))
    return sorted(set(logs))


HELD_RE = re.compile(r"review_docs: (\S+) held from acceptance — (.*)")
PERSIST_RE = re.compile(
    r"persist_reviewed_docs: (\S+) → docs_stage=\S+ \(score=([0-9.]+)"
)


def recorded_note(logs: list[str], member: str) -> dict:
    """The hold note the guard emitted for *member*, and the score it persisted.

    Pairs each ``held from acceptance`` line with the ``persist_reviewed_docs``
    line that follows it for the same member, and keeps the pair whose persisted
    score is the 0.849 floor — the value the arm record reports.
    """
    for log in logs:
        last_note = None
        with open(log, errors="replace") as fh:
            for line in fh:
                m = HELD_RE.search(line)
                if m and m.group(1) == member:
                    last_note = m.group(2).strip()
                    continue
                p = PERSIST_RE.search(line)
                if p and p.group(1) == member and float(p.group(2)) == HELD:
                    if last_note:
                        return {
                            "log": log,
                            "note": last_note,
                            "persisted_score": float(p.group(2)),
                        }
    return {"log": None, "note": None, "persisted_score": None}


def lint_cap_firings(logs: list[str]) -> dict:
    """Count the lint cap's own log line across every arm log.

    ``_cap_docs_scores_for_lint_errors`` writes ``... by lint error: ...`` only
    when an error-severity finding stands against the entry. Zero occurrences
    across all arms means the 0.849 values did not come from that path.
    """
    total = 0
    per_log = {}
    for log in logs:
        with open(log, errors="replace") as fh:
            n = sum("by lint error" in line for line in fh)
        if n:
            per_log[log] = n
            total += n
    return {"total": total, "per_log": per_log}


class _RefusingBudget:
    """Stand-in budget that refuses every reservation.

    A hold note that needs no LLM call returns before this is reached, so the
    note a member gets is the one its binding shape decides, not one the stub
    could manufacture.
    """

    def reserve(self, *a, **k):
        return None


async def reproduce(member: str) -> dict:
    """Reconstruct the member's review item and run the three notes on it."""
    from imas_codex.standard_names.prose_policy import banned_prose_findings
    from imas_codex.standard_names.review.pipeline import _fetch_review_dd_context
    from imas_codex.standard_names.workers import (
        _bound_docs_review_dd_paths,
        _documentation_banned_prose_note,
        _documentation_conformance_note,
        _documentation_contradiction_note,
        _load_docs_review_source_item,
    )

    try:
        item = _load_docs_review_source_item(member)
    except Exception as exc:  # member absent from the live graph
        return {"member": member, "loaded": False, "error": f"{type(exc).__name__}: {exc}"}

    item["source_paths"] = _bound_docs_review_dd_paths(item)
    if item["source_paths"]:
        await asyncio.to_thread(_fetch_review_dd_context, [item])

    bindings = item.get("source_bindings") or []
    dd_paths = _bound_docs_review_dd_paths(item)
    text = f"{item.get('description') or ''}\n{item.get('documentation') or ''}"

    conformance = _documentation_conformance_note(item)
    prose_gate = banned_prose_findings(text)
    contradiction = await _documentation_contradiction_note(item, _RefusingBudget())
    prose = None
    if any(prose_gate.values()):
        prose = await _documentation_banned_prose_note(item, _RefusingBudget())

    # Control: the same call with the non-DD binding removed. If the note is
    # caused by that binding's presence, this copy must not reproduce it.
    control = {k: v for k, v in item.items() if k != "source_bindings"}
    control["source_bindings"] = [
        b for b in bindings if b.get("source_type") == "dd"
    ]
    control_note = await _documentation_contradiction_note(control, _RefusingBudget())

    return {
        "member": member,
        "loaded": True,
        "origin": item.get("origin"),
        "source_paths": list(item.get("source_paths") or []),
        "binding_types": [b.get("source_type") for b in bindings],
        "dd_paths": dd_paths,
        "dd_source_docs": len(item.get("dd_source_docs") or []),
        "dd_documentation_present": bool(item.get("dd_documentation")),
        "check_conformance": conformance,
        "check_prose_gate_nonempty": any(prose_gate.values()),
        "check_prose_note": prose,
        "check_contradiction": contradiction,
        "control_contradiction_without_nondd_binding": control_note,
        "control_changed": control_note != contradiction,
    }


def main() -> int:
    arms = held_members()
    logs = arm_logs()
    cap = lint_cap_firings(logs)
    print(f"held members: {len(arms)}")
    print(f"arm apply logs: {len(logs)}")
    print(f"lint-cap firings across all arms: {cap['total']}")

    rows = []
    for a in arms:
        rec = recorded_note(logs, a["member"])
        rows.append({**a, "recorded": rec})

    loop = asyncio.new_event_loop()
    try:
        for row in rows:
            row["reproduce"] = loop.run_until_complete(reproduce(row["member"]))
    finally:
        loop.close()

    for row in rows:
        r = row["reproduce"]
        print(
            f"\n{row['family']} [{row['arm']}] member={row['member']} "
            f"{row['score_before']} -> {row['score_after']}"
        )
        print(f"  recorded note: {str(row['recorded']['note'])[:200]}")
        print(f"  log: {row['recorded']['log']}")
        if r.get("loaded"):
            print(
                f"  origin={r['origin']} binding_types={r['binding_types']} "
                f"dd_paths={len(r['dd_paths'])} dd_source_docs={r['dd_source_docs']}"
            )
            print(f"  re-run conformance: {r['check_conformance']}")
            print(f"  re-run contradiction: {r['check_contradiction']}")
            print(f"  control (non-DD binding dropped): {r['control_contradiction_without_nondd_binding']}")
        else:
            print(f"  live load failed: {r.get('error')}")

    payload = {
        "node": "ssog-anchor-held-cause",
        "held_members": len(arms),
        "lint_cap_firings_total": cap["total"],
        "lint_cap_per_log": cap["per_log"],
        "rows": rows,
    }
    out = Path(__file__).parent / "ssog-anchor-held-cause.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
