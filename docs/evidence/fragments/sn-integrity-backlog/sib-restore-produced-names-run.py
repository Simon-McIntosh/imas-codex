"""Measure and restore sourced names with no recorded supersession."""

from __future__ import annotations

import json
from collections import Counter

from imas_codex.graph.client import GraphClient


def read_cohort() -> None:
    with GraphClient() as client:
        rows = client.query(
            """
            MATCH (sn:StandardName {name_stage: 'superseded'})
            WHERE NOT (sn)-[:HAS_SUCCESSOR]->()
              AND NOT ()-[:REFINED_FROM]->(sn)
            OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            WITH sn, collect(DISTINCT change.operation) AS operations
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN sn.id AS id, sn.name_stage AS name_stage,
                   sn.superseded_from_stage AS superseded_from_stage,
                   sn.status AS status, sn.validation_status AS validation_status,
                   sn.description AS description, sn.origin AS origin,
                   collect(DISTINCT source.id) AS source_ids, operations
            ORDER BY id
            """
        )
        population = client.query(
            "MATCH (sn:StandardName {name_stage: 'superseded'}) "
            "RETURN count(sn) AS count"
        )[0]["count"]
        prior_stage_population = client.query(
            """
            MATCH (sn:StandardName {name_stage: 'superseded'})
            WHERE sn.superseded_from_stage IS NOT NULL
            RETURN sn.superseded_from_stage AS stage, count(sn) AS count
            ORDER BY stage
            """
        )
        source_states = client.query(
            """
            MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->
                  (sn:StandardName {name_stage: 'superseded'})
            WHERE sn.id IN $ids
            RETURN sn.id AS id, collect(DISTINCT {
                id: source.id, status: source.status
            }) AS sources
            ORDER BY id
            """,
            ids=[row["id"] for row in rows if row["source_ids"]],
        )
        candidates = [
            row
            for row in rows
            if row["source_ids"]
            and set(row["operations"]) <= {"realign_grammar_segments"}
        ]
        print(
            json.dumps(
                {
                    "population": population,
                    "prior_stage_population": prior_stage_population,
                    "successorless": len(rows),
                    "unrecorded_produced": len(candidates),
                    "prior_stage_count": sum(
                        row["superseded_from_stage"] is not None for row in candidates
                    ),
                    "candidate_operations": {
                        ", ".join(operations): count
                        for operations, count in Counter(
                            tuple(row["operations"]) for row in candidates
                        ).items()
                    },
                    "all_sourced_prior_stage": [
                        {
                            "id": row["id"],
                            "prior_stage": row["superseded_from_stage"],
                            "operations": row["operations"],
                        }
                        for row in rows
                        if row["source_ids"]
                        and row["superseded_from_stage"] is not None
                    ],
                    "source_states": source_states,
                    "rows": candidates,
                },
                indent=2,
                default=str,
            )
        )
        prior_count = sum(
            row["superseded_from_stage"] is not None for row in candidates
        )
        if len(candidates) != 25 or prior_count != 9:
            raise RuntimeError(
                "the recorded sourced cohort has moved: "
                f"{len(candidates)} names, {prior_count} with a prior stage; "
                "expected 25 names, 9 with a prior stage"
            )


if __name__ == "__main__":
    read_cohort()
