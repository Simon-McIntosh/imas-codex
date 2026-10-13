"""Score a uniform, hand-reviewed signal sample through the paid Choice judge.

Run with ``python -m tests.llm.measure_signal_role``. The fixture carries the
exact text sent to the judge; this module neither reads nor writes graph state.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
from collections import Counter
from pathlib import Path

from imas_codex.discovery.signals.role import ROLE_CRITERIA, judge_signal_roles

FIXTURE = Path(__file__).parent / "fixtures" / "signal_role_random_labels.json"
REPORT = (
    Path(__file__).parents[2]
    / "docs/evidence/fragments/signal-search-finds-the-measured-signal/signal-role-random.html"
)
MAX_SPEND_USD = 0.5


def load_fixture() -> dict:
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["signals"]
    if len(rows) != 200 or len({row["id"] for row in rows}) != 200:
        raise ValueError("fixture must contain 200 distinct signals")
    if (
        fixture["selection"]["method"]
        != "random.Random(seed).sample(sorted(eligible_ids), 200)"
    ):
        raise ValueError("fixture must declare uniform sampling")
    if any(row.get("hand_label") not in ROLE_CRITERIA for row in rows):
        raise ValueError("fixture has a missing or invalid hand label")
    if any(
        not all(key in row for key in ("name", "description", "unit", "glossary"))
        for row in rows
    ):
        raise ValueError("fixture lacks stored judge text")
    return fixture


def load_receipt(path: Path, rows: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    accepted = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = {row["id"] for row in rows}
    if len({row["id"] for row in accepted}) != len(accepted) or any(
        row["id"] not in ids
        or row["signal_role"] not in ROLE_CRITERIA
        or row["cost_usd"] <= 0
        for row in accepted
    ):
        raise ValueError("receipt has duplicate, foreign, or uncosted judgments")
    return accepted


async def measure(rows: list[dict], receipt: Path) -> tuple[list[dict], float]:
    judged = load_receipt(receipt, rows)
    spent = sum(row["cost_usd"] for row in judged)
    seen = {row["id"] for row in judged}
    remaining = [row for row in rows if row["id"] not in seen]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(remaining), 10):
        batch = remaining[offset : offset + 10]
        if spent >= MAX_SPEND_USD:
            raise RuntimeError(f"response-usage spend reached USD {spent:.6f} ceiling")
        accepted, cost, failed = await judge_signal_roles(batch, concurrency=8)
        if accepted:
            with receipt.open("a") as stream:
                for item in accepted:
                    stream.write(json.dumps(item) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            judged.extend(accepted)
        spent += cost
        if failed or len(accepted) != len(batch):
            ids = [row["id"] for row in failed]
            raise RuntimeError(
                f"unanswered role choices at batch {offset // 10}: {ids}"
            )
        print(f"judged={len(judged)}/200 spent_usd={spent:.6f}", flush=True)
        if spent <= 0:
            raise RuntimeError("pilot returned no positive response-usage cost")
        if spent >= MAX_SPEND_USD:
            raise RuntimeError(f"role judgments reached USD {spent:.6f} spend ceiling")
        if len(judged) == 10 and spent * (len(rows) / len(batch)) >= MAX_SPEND_USD:
            raise RuntimeError("pilot cost predicts spend above USD 0.5")
    if abs(sum(row["cost_usd"] for row in judged) - spent) > 1e-8:
        raise ValueError("per-response usage does not reconcile with batch cost")
    if len(judged) != 200:
        raise ValueError("receipt does not cover all 200 signals")
    return judged, spent


def summarize(rows: list[dict], judged: list[dict], spent: float) -> dict:
    roles = list(ROLE_CRITERIA)
    by_id = {row["id"]: row for row in judged}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("judge results do not cover the frozen fixture")
    distribution = Counter(row["hand_label"] for row in rows)
    confusion = {expected: dict.fromkeys(roles, 0) for expected in roles}
    for row in rows:
        confusion[row["hand_label"]][by_id[row["id"]]["signal_role"]] += 1
    correct = sum(confusion[role][role] for role in roles)
    return {
        "agreement": correct / len(rows),
        "threshold_met": correct / len(rows) >= 0.9,
        "correct": correct,
        "distribution": dict(distribution),
        "per_role": {
            role: confusion[role][role] / distribution[role]
            if distribution[role]
            else None
            for role in roles
        },
        "confusion": confusion,
        "disagreements": len(rows) - correct,
        "cost_usd": spent,
        "cost_per_signal_usd": spent / len(rows),
    }


def render(fixture: dict, judged: list[dict], result: dict) -> str:
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
        f"<tr><th>{escape(role)}</th><td>{result['distribution'].get(role, 0)}</td>"
        f"<td>{result['confusion'][role][role]}</td>"
        f"<td>{result['per_role'][role]:.1%}</td></tr>"
        if result["per_role"][role] is not None
        else f"<tr><th>{escape(role)}</th><td>0</td><td>0</td><td>not sampled</td></tr>"
        for role in roles
    )
    item_rows = "\n".join(
        "<tr><td>"
        + escape(row["id"])
        + "</td><td>"
        + escape(row["name"] or "")
        + "</td><td>"
        + escape(row["description"] or "")
        + "</td><td>"
        + escape(str(row["unit"] or ""))
        + "</td><td>"
        + escape(json.dumps(row["glossary"], ensure_ascii=False))
        + "</td><td>"
        + escape(row["hand_label"])
        + "</td><td>"
        + escape(by_id[row["id"]]["signal_role"])
        + "</td><td>"
        + f"{by_id[row['id']]['cost_usd']:.7f}"
        + "</td></tr>"
        for row in fixture["signals"]
        if row["hand_label"] != by_id[row["id"]]["signal_role"]
    )
    selection = fixture["selection"]
    verdict = "meets" if result["threshold_met"] else "does not meet"
    return f"""<section id="ssfms-role-random-sample-results" data-reckon="fragment">
  <h2 id="ssfms-role-random-sample-title">Uniform random signal-role sample</h2>
  <p id="ssfms-role-random-sample-selection">A seeded uniform draw without role-word selection: 200 of {selection["population_size"]} eligible JT-60SA signals, seed {selection["seed"]}. Eligible means enriched at or after {escape(fixture["eligibility"]["enriched_at_cutover"])}. Labels were frozen at {escape(fixture["labelled_at"])} before the judge ran. The judge read each fixture's name, description, unit and glossary. No signal_role property was written to the graph.</p>
  <p id="ssfms-role-random-sample-rule">{escape(fixture["label_rule"])}</p>
  <p id="ssfms-role-random-sample-summary">Overall agreement: <strong>{result["correct"]}/200 ({result["agreement"]:.1%})</strong>, which <strong>{verdict}</strong> the lead's 90% threshold for wiring the role into enrichment. Response-usage spend: <strong>USD {result["cost_usd"]:.6f}</strong>, or USD {result["cost_per_signal_usd"]:.7f} per signal.</p>
  <h3 id="ssfms-role-random-sample-distribution">Sample distribution and agreement by hand label</h3>
  <table><thead><tr><th>Role</th><th>Sampled</th><th>Correct</th><th>Agreement</th></tr></thead><tbody>{role_rows}</tbody></table>
  <h3 id="ssfms-role-random-sample-confusion">Confusion: hand label by judged role</h3>
  <table><thead><tr><th>Hand label ↓ / judged →</th>{header}</tr></thead><tbody>{confusion_rows}</tbody></table>
  <h3 id="ssfms-role-random-sample-disagreements">Every disagreement ({result["disagreements"]})</h3>
  <table><thead><tr><th>Signal ID</th><th>Name</th><th>Description</th><th>Unit</th><th>Glossary</th><th>Hand label</th><th>Judged role</th><th>Cost [USD]</th></tr></thead><tbody>{item_rows}</tbody></table>
</section>\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(os.environ["TMPDIR"]) / "signal_role_random_receipt.jsonl",
    )
    args = parser.parse_args()
    fixture = load_fixture()
    judged, spent = asyncio.run(measure(fixture["signals"], args.receipt))
    result = summarize(fixture["signals"], judged, spent)
    args.output.write_text(render(fixture, judged, result))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "confusion"},
            indent=2,
        )
    )
    print(f"report={args.output} receipt={args.receipt}")


if __name__ == "__main__":
    main()
