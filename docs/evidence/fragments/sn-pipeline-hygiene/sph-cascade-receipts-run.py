"""Inventory surviving calibration inputs and their bounded graph provenance."""

import base64
import hashlib
import json
import lzma
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.review import cascade_calibration as calibration

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/evidence/fragments/sn-review-through-jev"


def main():
    thresholds = json.loads(calibration.THRESHOLDS_PATH.read_text())
    for axis in ("name", "docs"):
        path = Path(thresholds[axis]["source_receipts"])
        print(
            json.dumps(
                {
                    "axis": axis,
                    "receipt": str(path),
                    "exists": path.exists(),
                    "source_sha256": thresholds[axis]["source_sha256"],
                    "score_selection_n": thresholds[axis]["score_selection_n"],
                }
            ),
            flush=True,
        )
    source = json.loads((EVIDENCE / "srtj-evidence-supply-results.json").read_text())
    states = json.loads(
        lzma.decompress(base64.b85decode(source["frozen_states"]["data"]))
    )
    assert len(states) == source["frozen_states"]["count"]
    first = next(iter(states.values()))
    print(
        json.dumps(
            {
                "embedded_name_states": len(states),
                "sample_state_fields": list(first),
                "states_with_answers": sum("answers" in s for s in states.values()),
            }
        ),
        flush=True,
    )
    prior = json.loads((EVIDENCE / "srtj-docs-measure-results.json").read_text())
    docs = json.loads(lzma.decompress(base64.b64decode(prior["frozen_lzma_b64"])))
    print(
        json.dumps(
            {
                "embedded_prior_docs_fields": list(docs),
                "embedded_prior_docs_receipts": len(docs["receipts"]),
                "embedded_prior_docs_answered": sum(
                    bool(r.get("answers")) for r in docs["receipts"].values()
                ),
                "sample_docs_receipt_fields": list(
                    next(iter(docs["receipts"].values()))
                ),
            }
        ),
        flush=True,
    )
    scored = json.loads((EVIDENCE / "srtj-docs-score-arm-results.json").read_text())
    print(
        json.dumps(
            {
                "score_arm_frozen_path": scored["frozen_path"],
                "score_arm_frozen_sha256": scored["frozen_sha256"],
                "score_arm_result_fields": list(scored),
            }
        ),
        flush=True,
    )
    graph = GraphClient()
    try:
        for axis, golden in (
            ("name", calibration.GOLDEN_NAME),
            ("docs", calibration.GOLDEN_DOCS),
        ):
            names = sorted(
                {r["canonical_unit"] for r in json.loads(golden.read_text())}
            )
            totals = dict.fromkeys(
                (
                    "names",
                    "found_names",
                    "reviews",
                    "review_ids",
                    "review_axes",
                    "review_dates",
                    "judge_answers",
                    "judge_probs",
                    "historical_answers",
                ),
                0,
            )
            kinds = {}
            for start in range(0, len(names), 50):
                batch = names[start : start + 50]
                rows = list(
                    graph.query(
                        """
                UNWIND $names AS name
                OPTIONAL MATCH (sn:StandardName {id: name})
                OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview {review_axis: $axis})
                RETURN name, sn.id AS found_id, count(r) AS reviews,
                  count(r.id) AS review_ids, count(r.review_axis) AS review_axes,
                  count(r.reviewed_at) AS review_dates,
                  count(r.judge_answers_json) AS judge_answers,
                  count(r.judge_probs_json) AS judge_probs,
                  sum(CASE WHEN r.judge_answers_json IS NOT NULL AND
                    toString(r.reviewed_at) >= '2026-10-15' AND
                    toString(r.reviewed_at) < '2026-10-19' THEN 1 ELSE 0 END) AS historical_answers,
                  collect(DISTINCT r.judge_kind) AS kinds
                LIMIT 50
                """,
                        names=batch,
                        axis=axis,
                    )
                )
                assert len(rows) == len(batch), (axis, len(rows), len(batch))
                for row in rows:
                    totals["names"] += 1
                    totals["found_names"] += row["found_id"] is not None
                    for key in totals.keys() - {"names", "found_names"}:
                        totals[key] += row[key]
                    for kind in row["kinds"]:
                        kinds[kind] = kinds.get(kind, 0) + 1
            assert totals["reviews"] > 0 and totals["review_ids"] > 0
            assert totals["reviews"] == totals["review_axes"]
            print(
                json.dumps(
                    {
                        "axis": axis,
                        "bounded_golden_review_inventory": totals,
                        "names_by_judge_kind": kinds,
                    }
                ),
                flush=True,
            )
    finally:
        graph.close()
    print(
        "THRESHOLDS_SHA256="
        + hashlib.sha256(calibration.THRESHOLDS_PATH.read_bytes()).hexdigest(),
        flush=True,
    )


if __name__ == "__main__":
    main()
