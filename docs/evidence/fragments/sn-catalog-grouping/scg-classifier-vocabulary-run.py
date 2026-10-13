"""Measure DD source domains against reviewed standard-name labels without writes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import (
    classify_tier1_llm,
    gather_classification_context,
)
from imas_codex.llm.prompt_loader import get_prompt_content_hash
from imas_codex.settings import get_model
from imas_codex.standard_names.domain_priority import pick_primary_domain

HERE = Path(__file__).resolve().parent
LABELS = HERE / "scg-domain-label-sheet.json"
RESULTS = HERE / "scg-classifier-vocabulary-results.json"
SOURCE_QUERY = """
UNWIND $names AS name
MATCH (sn:StandardName {id: name})
OPTIONAL MATCH (sn)<-[:PRODUCED_NAME]-(source:StandardNameSource)
              -[:FROM_DD_PATH]->(dd:IMASNode)
RETURN sn.id AS name, sn.physics_domain AS name_domain,
       sn.source_domains AS name_source_domains,
       source.id AS source_id, dd.id AS path,
       dd.physics_domain AS stored_domain, dd.domain_source AS domain_source,
       dd.domain_model AS domain_model,
       toString(dd.domain_classified_at) AS domain_classified_at,
       dd.node_category AS node_category, dd.description AS description,
       dd.unit AS unit
ORDER BY name, path
LIMIT 10000
"""


def _read_sources(gc: GraphClient, names: list[str]) -> list[dict]:
    rows = gc.query(SOURCE_QUERY, names=names)
    if not rows or not any(row["path"] for row in rows):
        raise RuntimeError("source query failed its positive control")
    if len(rows) == 10000:
        raise RuntimeError("source query reached its result bound")
    return rows


def _snapshot(rows: list[dict]) -> dict:
    names = {}
    paths = {}
    for row in rows:
        names[row["name"]] = [row["name_domain"], row["name_source_domains"]]
        if row["path"]:
            paths[row["path"]] = [
                row["stored_domain"],
                row["domain_source"],
                row["domain_model"],
                row["domain_classified_at"],
            ]
    canonical = json.dumps([names, paths], sort_keys=True, default=str)
    return {
        "name_count": len(names),
        "path_count": len(paths),
        "name_domain_distribution": dict(
            sorted(Counter(v[0] or "<none>" for v in names.values()).items())
        ),
        "path_domain_distribution": dict(
            sorted(Counter(v[0] or "<none>" for v in paths.values()).items())
        ),
        "domain_classified_at_max": max(
            (v[3] or "" for v in paths.values()), default=""
        ),
        "property_digest": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _score(entries: list[dict], key: str, *, source_set: str | None = None) -> dict:
    correct = sum(
        row["label"] in row[source_set] if source_set else row[key] == row["label"]
        for row in entries
    )
    confusion = Counter(
        (row["label"], row[key] or "<none>")
        for row in entries
        if not (
            row["label"] in row[source_set] if source_set else row[key] == row["label"]
        )
    )
    return {
        "correct": correct,
        "total": len(entries),
        "accuracy": correct / len(entries) if entries else None,
        "confusion": [
            {"label": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in confusion.most_common()
        ],
    }


async def _fresh(
    gc: GraphClient, paths: dict[str, dict], cache_path: Path
) -> tuple[dict, float, str]:
    model = get_model("sn-classifier")
    cache = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else {"model": model, "spend_usd": 0.0, "domains": {}}
    )
    if cache["model"] != model:
        raise RuntimeError("cached model differs from current classifier model")
    domains = cache["domains"]
    pending = [paths[key] for key in sorted(paths) if key not in domains]
    if pending:
        context = gather_classification_context(gc, [p["id"] for p in pending])
        context_ids = {p["id"] for p in context}
        if context_ids != {p["id"] for p in pending}:
            raise RuntimeError("classification context missed live source paths")
    for start in range(0, len(pending), 20):
        if cache["spend_usd"] >= 2.75:
            raise RuntimeError("spend reserve reached before all paths were classified")
        chunk = pending[start : start + 20]
        charges = []
        result = await classify_tier1_llm(
            gc, chunk, model=model, on_cost=charges.append
        )
        result_ids = {item["id"] for item in result}
        if len(result) != len(chunk) or result_ids != {item["id"] for item in chunk}:
            raise RuntimeError("classifier omitted or duplicated a source path")
        cost = sum(charges)
        cache["spend_usd"] += cost
        if cache["spend_usd"] > 2.80:
            raise RuntimeError(
                f"USD 2.80 spend ceiling exceeded: {cache['spend_usd']:.6f}"
            )
        domains.update({item["id"]: item["physics_domain"] for item in result})
        cache_path.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n")
        print(
            f"classified={len(domains)}/{len(paths)} spend_usd={cache['spend_usd']:.6f}",
            flush=True,
        )
    return domains, cache["spend_usd"], model


def _derive(
    label_rows: list[dict], graph_rows: list[dict], fresh: dict
) -> tuple[list[dict], dict]:
    by_name = defaultdict(list)
    for row in graph_rows:
        if row["path"]:
            by_name[row["name"]].append(row)
    result = []
    for label in label_rows:
        sources = by_name[label["name"]]
        stored = sorted(
            {row["stored_domain"] for row in sources if row["stored_domain"]}
        )
        current = sorted({fresh[row["path"]] for row in sources})
        result.append(
            {
                "name": label["name"],
                "label": label["label"],
                "label_source": label.get("label_source", "coordinator"),
                "split": label.get("split"),
                "sources": [
                    {
                        "path": row["path"],
                        "stored": row["stored_domain"],
                        "fresh": fresh[row["path"]],
                        "domain_source": row["domain_source"],
                    }
                    for row in sources
                ],
                "stored_domains": stored,
                "fresh_domains": current,
                "stored_primary": pick_primary_domain(stored) if stored else None,
                "fresh_primary": pick_primary_domain(current) if current else None,
                "derived_domain": pick_primary_domain(current) if current else None,
            }
        )
    scores = {}
    for cohort, rows in (
        ("lead", result[:20]),
        ("coordinator", result[20:]),
    ):
        scores[cohort] = {
            "stored": _score(rows, "stored_primary", source_set="stored_domains"),
            "fresh": _score(rows, "fresh_primary", source_set="fresh_domains"),
            "derived": _score(rows, "derived_domain"),
            "with_live_sources": sum(bool(row["sources"]) for row in rows),
        }
    return result, scores


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    sheet = json.loads(LABELS.read_text())
    label_rows = sheet["reference"] + sheet["sample"]
    names = [row["name"] for row in label_rows]
    if len(names) != 170 or len(set(names)) != 170:
        raise RuntimeError("expected 170 distinct labelled names")
    with GraphClient() as gc:
        before_rows = _read_sources(gc, names)
        before = _snapshot(before_rows)
        paths = {
            row["path"]: {
                "id": row["path"],
                "node_category": row["node_category"],
                "description": row["description"],
                "units": row["unit"],
            }
            for row in before_rows
            if row["path"]
        }
        print("BEFORE " + json.dumps(before, sort_keys=True), flush=True)
        if args.fresh:
            fresh, spend, model = await _fresh(gc, paths, args.cache)
        else:
            cache = json.loads(args.cache.read_text())
            fresh, spend, model = cache["domains"], cache["spend_usd"], cache["model"]
        if set(fresh) != set(paths):
            raise RuntimeError("cached classifications do not cover live source paths")
        after_rows = _read_sources(gc, names)
        after = _snapshot(after_rows)
        print("AFTER " + json.dumps(after, sort_keys=True), flush=True)
        if before != after:
            raise RuntimeError(
                "graph source or name properties changed during read-only run"
            )
    rows, scores = _derive(label_rows, after_rows, fresh)
    payload = {
        "model": model,
        "prompt_hash": get_prompt_content_hash("imas/domain_classifier"),
        "spend_usd": spend,
        "source_path_count": len(paths),
        "snapshot_before": before,
        "snapshot_after": after,
        "scores": scores,
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print("SCORES " + json.dumps(scores, sort_keys=True), flush=True)
    print(
        f"SPEND_USD={spend:.6f} PROMPT_HASH={payload['prompt_hash']} "
        f"RESULT_SHA256={hashlib.sha256(RESULTS.read_bytes()).hexdigest()}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
