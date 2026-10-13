"""Read live names and write the judged near-duplicate worklist."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.standard_names.near_duplicates import build_worklist, judge_worklist
from imas_codex.standard_names.sources_manifest import load_sources_file

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261013T050740598849-scg-near-duplicate-worklist"
)
SOURCES = ROOT / "imas_codex/standard_names/manifests/west_production_dd_paths.yaml"
PROGRESS = RUN / "worklist-segment-values-progress.json"
PREVIOUS = RUN / "worklist-progress.json"


def read_names(graph: GraphClient) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = graph.query(
            "MATCH (n:StandardName) "
            "WHERE n.name_stage IN ['accepted', 'approved'] "
            "AND n.embedding IS NOT NULL AND n.unit IS NOT NULL "
            "RETURN n.id AS id, n.unit AS unit, n.description AS description, "
            "n.embedding AS embedding ORDER BY n.id SKIP $offset LIMIT 100",
            offset=offset,
        )
        rows.extend(page)
        if len(page) < 100:
            break
        offset += len(page)
    if not rows:
        raise RuntimeError(
            "Live accepted-name query returned no rows; inspect the graph"
        )
    print(f"live_accepted_embedded_names={len(rows)}", flush=True)
    return rows


def read_west(graph: GraphClient) -> set[str]:
    ids = [f"dd:{source}" for source in load_sources_file(SOURCES)]
    rows = graph.query(
        "MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName) "
        "WHERE s.id IN $ids AND n.name_stage IN ['accepted', 'approved'] "
        "RETURN DISTINCT n.id AS id LIMIT 500",
        ids=ids,
    )
    if not rows:
        raise RuntimeError(
            "WEST manifest has no live accepted names; inspect source binding"
        )
    print(f"west_manifest_sources={len(ids)} west_live_names={len(rows)}", flush=True)
    return {row["id"] for row in rows}


async def run() -> None:
    graph = GraphClient()
    names = read_names(graph)
    west_names = read_west(graph)
    worklist = build_worklist(names, west_names=west_names)
    all_pairs = worklist["west"] + worklist["other"]
    print(
        f"west_pairs={len(worklist['west'])} other_pairs={len(worklist['other'])} "
        f"overlaps={len({pair['overlap'] for pair in all_pairs})}",
        flush=True,
    )
    if not all_pairs:
        raise RuntimeError("No pairs found despite the positive accepted-name census")

    model = get_model("sn-judge")
    previous_spend = (
        json.loads(PREVIOUS.read_text())["spent"] if PREVIOUS.exists() else 0.0
    )
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        if progress["pairs"] != [[pair["left"], pair["right"]] for pair in all_pairs]:
            raise RuntimeError(
                "Live pair set changed since the last judgment checkpoint"
            )
        judged = progress["judged"]
        current_pairs = {(pair["left"], pair["right"]): pair for pair in all_pairs}
        for item in judged:
            current = current_pairs[item["left"], item["right"]]
            item["overlap"] = current["overlap"]
            item["overlap_size"] = current["overlap_size"]
        spent = progress["spent"]
        pilot_cost = progress["pilot_cost"]
    else:
        judged = []
        spent = 0.0
        pilot_cost = None

    while len(judged) < len(all_pairs):
        batch = all_pairs[len(judged) : len(judged) + 10]
        if pilot_cost is not None:
            projected = (
                previous_spend
                + spent
                + (pilot_cost / min(10, len(all_pairs))) * len(batch)
            )
            if projected > 1.50:
                print(
                    f"budget_stop spent={spent:.6f} next_projected={projected:.6f}",
                    flush=True,
                )
                raise RuntimeError("Judgment budget would exceed USD 1.50")
        answer, cost = await judge_worklist(batch, model=model)
        if any(item["verdict"] is None for item in answer):
            raise RuntimeError(
                "A decision call returned no verdict; resume from the last checkpoint"
            )
        judged.extend(answer)
        spent += cost
        if pilot_cost is None:
            pilot_cost = cost
            extrapolated = previous_spend + cost / len(batch) * len(all_pairs)
            print(
                f"pilot_pairs={len(batch)} pilot_cost_usd={cost:.6f} extrapolated_usd={extrapolated:.6f}",
                flush=True,
            )
        PROGRESS.write_text(
            json.dumps(
                {
                    "pairs": [(pair["left"], pair["right"]) for pair in all_pairs],
                    "judged": judged,
                    "spent": spent,
                    "pilot_cost": pilot_cost,
                }
            )
        )
        if len(judged) == len(batch):
            changed = [
                value
                for item in answer
                for segment, value in item["segment_judgments"].items()
                if segment in item["divergent_segments"]
            ]
            unchanged = [
                value
                for item in answer
                for segment, value in item["segment_judgments"].items()
                if segment not in item["divergent_segments"]
            ]
            separation = sum(changed) / len(changed) - sum(unchanged) / len(unchanged)
            print(f"pilot_segment_noul_separation={separation:.3f}", flush=True)
            if separation < 0.2:
                raise RuntimeError(
                    "Segment judgments do not distinguish changed from unchanged grammar values"
                )
        if len(judged) == len(batch) and extrapolated > 1.50:
            print("budget_stop after pilot", flush=True)
            raise RuntimeError("Full judgment extrapolates above USD 1.50")
        print(
            f"judged={len(judged)}/{len(all_pairs)} spent_usd={spent:.6f}", flush=True
        )

    west = [
        item
        for item in judged
        if item["left"] in west_names or item["right"] in west_names
    ]
    other = [
        item
        for item in judged
        if item["left"] not in west_names and item["right"] not in west_names
    ]
    order = {"same quantity": 0, "related; needs ruling": 1, "distinct quantities": 2}

    def sort_key(item: dict) -> tuple:
        return (
            order[item["verdict"]],
            -item["same_quantity_probability"],
            item["overlap"],
            item["left"],
        )

    west.sort(key=sort_key)
    other.sort(key=sort_key)
    payload = {
        "threshold": 0.95,
        "live_names": len(names),
        "west_names": len(west_names),
        "west_pairs": len(west),
        "west_overlaps": len({pair["overlap"] for pair in west}),
        "other_pairs": len(other),
        "overlaps": len({pair["overlap"] for pair in judged}),
        "pilot_cost_usd": pilot_cost,
        "total_cost_usd": previous_spend + spent,
        "current_judgment_cost_usd": spent,
        "pairs": other,
    }
    (HERE / "scg-near-duplicate-worklist.json").write_text(
        json.dumps(payload, indent=1) + "\n"
    )
    (RUN / "west-worklist.json").write_text(json.dumps(west, indent=2) + "\n")
    print(
        f"complete west={len(west)} other={len(other)} cost_usd={previous_spend + spent:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(run())
