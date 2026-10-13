"""Score frozen, hand-reviewed signal roles through the paid Choice judge.

Run with ``python -m tests.llm.measure_signal_role``. The fixture carries the
exact text sent to the judge; this module neither reads nor writes graph state.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from collections import Counter
from pathlib import Path

from imas_codex.discovery.signals.role import ROLE_CRITERIA, judge_signal_roles

FIXTURE = Path(__file__).parent / "fixtures" / "signal_role_labels.json"
REPORT = (
    Path(__file__).parents[2]
    / "docs/evidence/fragments/signal-search-finds-the-measured-signal/signal-role-prototype.html"
)
MAX_SPEND_USD = 1.0


def load_fixture() -> list[dict]:
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["signals"]
    roles = set(ROLE_CRITERIA)
    if len(rows) != 300 or len({row["id"] for row in rows}) != 300:
        raise ValueError("fixture must contain 300 distinct signals")
    if Counter(row["hand_label"] for row in rows) != dict.fromkeys(roles, 50):
        raise ValueError("fixture must contain 50 hand labels per role")
    if any(
        not all(key in row for key in ("name", "description", "unit", "glossary"))
        for row in rows
    ):
        raise ValueError("fixture lacks stored judge text")
    return rows


async def measure(rows: list[dict]) -> tuple[list[dict], float]:
    judged: list[dict] = []
    spent = 0.0
    for offset in range(0, len(rows), 10):
        batch = rows[offset : offset + 10]
        accepted, cost, failed = await judge_signal_roles(batch, concurrency=8)
        spent += cost
        if failed or len(accepted) != len(batch):
            ids = [row["id"] for row in failed]
            raise RuntimeError(
                f"unanswered role choices at batch {offset // 10}: {ids}"
            )
        judged.extend(accepted)
        print(f"judged={len(judged)}/300 spent_usd={spent:.6f}", flush=True)
        if offset == 0 and spent <= 0:
            raise RuntimeError("pilot returned no positive response-usage cost")
        if spent >= MAX_SPEND_USD:
            raise RuntimeError(f"role judgments reached USD {spent:.6f} spend ceiling")
        if offset == 0 and spent * (len(rows) / len(batch)) >= MAX_SPEND_USD:
            raise RuntimeError("pilot cost predicts spend above USD 1")
    if abs(sum(row["cost_usd"] for row in judged) - spent) > 1e-8:
        raise ValueError("per-response usage does not reconcile with batch cost")
    return judged, spent


def summarize(rows: list[dict], judged: list[dict], spent: float) -> dict:
    roles = list(ROLE_CRITERIA)
    by_id = {row["id"]: row for row in judged}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("judge results do not cover the frozen fixture")
    confusion = {expected: dict.fromkeys(roles, 0) for expected in roles}
    for row in rows:
        confusion[row["hand_label"]][by_id[row["id"]]["signal_role"]] += 1
    correct = sum(confusion[role][role] for role in roles)
    return {
        "agreement": correct / len(rows),
        "correct": correct,
        "per_role": {role: confusion[role][role] / 50 for role in roles},
        "confusion": confusion,
        "cost_usd": spent,
        "cost_per_signal_usd": spent / len(rows),
    }


def render(rows: list[dict], judged: list[dict], result: dict) -> str:
    escape = html.escape
    roles = list(ROLE_CRITERIA)
    by_id = {row["id"]: row for row in judged}
    header = "".join(f"<th>{escape(role)}</th>" for role in roles)
    confusion_rows = "\n".join(
        "<tr><th>"
        + escape(expected)
        + "</th>"
        + "".join(
            f"<td>{result['confusion'][expected][predicted]}</td>"
            for predicted in roles
        )
        + "</tr>"
        for expected in roles
    )
    role_rows = "\n".join(
        f"<tr><th>{escape(role)}</th><td>{result['confusion'][role][role]}/50</td>"
        f"<td>{result['per_role'][role]:.1%}</td></tr>"
        for role in roles
    )
    item_rows = "\n".join(
        "<tr><td>"
        + escape(row["id"])
        + "</td><td>"
        + escape(row["hand_label"])
        + "</td><td>"
        + escape(by_id[row["id"]]["signal_role"])
        + "</td><td>"
        + f"{by_id[row['id']]['cost_usd']:.7f}"
        + "</td></tr>"
        for row in rows
    )
    return f"""<section id="ssfms-role-prototype-results" data-reckon="fragment">
  <h2 id="ssfms-role-prototype-results-title">Signal role Choice prototype</h2>
  <p id="ssfms-role-prototype-results-sample">300 eligible JT-60SA signals; 50 hand-reviewed labels per role, frozen before the judge ran. The judge read the fixture's stored name, description, unit and matching glossary entries. No graph property was written.</p>
  <p id="ssfms-role-prototype-results-sampling">Candidate selection searched for role words and reviewers checked each text, so this balanced sample favours signals whose role is explicit. Agreement here estimates performance on that sample; it does not establish population-wide accuracy.</p>
  <p id="ssfms-role-prototype-results-boundaries">Disagreements include control-method references and commands, timestamps, uncertainty flags, and commanded positions. Those boundaries need label review before a backfill decision.</p>
  <p id="ssfms-role-prototype-results-summary">Overall agreement: <strong>{result["correct"]}/300 ({result["agreement"]:.1%})</strong>. Jev response-usage spend: <strong>USD {result["cost_usd"]:.6f}</strong>, or USD {result["cost_per_signal_usd"]:.7f} per signal.</p>
  <h3 id="ssfms-role-prototype-results-per-role">Agreement by hand label</h3>
  <table><thead><tr><th>Role</th><th>Correct</th><th>Agreement</th></tr></thead><tbody>{role_rows}</tbody></table>
  <h3 id="ssfms-role-prototype-results-confusion">Confusion: hand label by judged role</h3>
  <table><thead><tr><th>Hand label ↓ / judged →</th>{header}</tr></thead><tbody>{confusion_rows}</tbody></table>
  <h3 id="ssfms-role-prototype-results-items">Individual judgments and response cost</h3>
  <table><thead><tr><th>Signal ID</th><th>Hand label</th><th>Judged role</th><th>Cost [USD]</th></tr></thead><tbody>{item_rows}</tbody></table>
</section>\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    rows = load_fixture()
    judged, spent = asyncio.run(measure(rows))
    result = summarize(rows, judged, spent)
    args.output.write_text(render(rows, judged, result))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "confusion"},
            indent=2,
        )
    )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
