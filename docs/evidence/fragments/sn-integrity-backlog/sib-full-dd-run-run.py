"""Read the graph state around a whole-catalog standard-name run."""

import json
import sys

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.export import (
    _RELEASE_IDENTITY_HOLDS,
    _SHIPPING_NAME_STAGES,
)
from imas_codex.standard_names.graph_store.graph_supersession import (
    classify_sourceless_standard_names,
)
from imas_codex.standard_names.source_invariant import find_sourceless_live_names


def read_back() -> dict:
    with GraphClient() as graph:
        graph_nodes = graph.query("MATCH (n) RETURN count(n) AS total")[0]["total"]
        rows = graph.query(
            "MATCH (n:StandardName) RETURN count(n) AS total, "
            "count(n.id) AS with_id, "
            "sum(CASE WHEN n.name_stage = 'accepted' THEN 1 ELSE 0 END) AS accepted"
        )
        counts = rows[0]
        if not counts["total"] or counts["with_id"] != counts["total"]:
            raise RuntimeError(f"StandardName count control failed: {counts}")

        ids = [
            row["id"] for row in graph.query("MATCH (n:StandardName) RETURN n.id AS id")
        ]
        sourceless_ids = set(find_sourceless_live_names(graph, ids))
        audit = classify_sourceless_standard_names(graph)
        population = [
            row for row in audit if row["population"] in {"dropped_edge", "never_bound"}
        ]
        drafts = [
            row["id"]
            for row in population
            if row["name_stage"] != "exhausted"
            and row["name_stage"] not in _SHIPPING_NAME_STAGES
            and row["id"] in sourceless_ids
        ]
        blocked = [
            row["id"]
            for row in population
            if row["name_stage"] in _SHIPPING_NAME_STAGES
            and row["id"] not in _RELEASE_IDENTITY_HOLDS
        ]
        return {
            "graph_nodes_total": graph_nodes,
            "standard_names_total": counts["total"],
            "accepted": counts["accepted"],
            "sourceless_live_drafts": sorted(drafts),
            "export_source_audit_blocking_set": sorted(blocked),
            "sourceless_invariant_total": len(sourceless_ids),
            "source_audit_population_total": len(population),
        }


def transactions() -> list[dict]:
    with GraphClient() as graph:
        return graph.query(
            "SHOW TRANSACTIONS YIELD transactionId, currentQuery, status, username "
            "RETURN transactionId, currentQuery, status, username"
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["transactions"]:
        print(json.dumps(transactions(), indent=2, sort_keys=True))
    elif sys.argv[1:] == ["read-back"]:
        print(json.dumps(read_back(), indent=2, sort_keys=True))
    else:
        raise SystemExit("usage: run.py transactions|read-back")
