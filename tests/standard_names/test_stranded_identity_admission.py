"""State admission and its Cypher rendering agree on persisted identities."""

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)
from imas_codex.standard_names.graph_store.graph_catalog import (
    docs_review_eligibility_params,
)
from imas_codex.standard_names.pools import (
    classify_identity_state_groups,
    identity_state_groups,
    standard_name_pool_predicates,
)

_STATE_FIELDS = (
    "name_stage",
    "docs_stage",
    "validation_status",
    "status",
    "origin",
    "description_state",
    "name_score_state",
    "docs_score_state",
    "name_attempts_under_cap",
    "docs_attempts_under_cap",
    "edit_mode",
    "rename_resubmit_capped",
    "review_quorum_shortfall",
    "docs_review_quorum_shortfall",
    "has_live_child",
    "has_winning_docs_review",
    "claim_active",
)
# Each row records a distinct observed state and its population count.
_STATE_ROWS = (
    "accepted|pending|valid|draft|derived|placeholder|missing|missing|True|True|None|False|False|False|False|False|False|6",
    "accepted|pending|valid|draft|derived|substantive|missing|missing|True|True|None|False|False|False|False|False|False|1",
    "accepted|reviewed|quarantined|draft|pipeline|substantive|passing|below|True|True|None|False|False|True|False|False|False|1",
    "accepted|reviewed|valid|draft|None|substantive|passing|below|True|True|None|False|False|False|False|False|False|19",
    "accepted|reviewed|valid|draft|None|substantive|passing|below|True|True|None|False|False|False|True|False|False|1",
    "accepted|reviewed|valid|draft|None|substantive|passing|below|True|True|hint|False|False|False|False|False|False|3",
    "accepted|reviewed|valid|draft|None|substantive|passing|below|True|True|hint|False|False|False|True|False|False|1",
    "accepted|reviewed|valid|draft|None|substantive|passing|passing|True|True|None|False|False|False|False|False|False|13",
    "accepted|reviewed|valid|draft|None|substantive|passing|passing|True|True|hint|False|False|False|False|False|False|2",
    "accepted|reviewed|valid|draft|derived|substantive|missing|below|True|True|None|False|False|False|True|False|False|15",
    "accepted|reviewed|valid|draft|derived|substantive|missing|passing|True|True|None|False|False|False|True|False|False|9",
    "accepted|reviewed|valid|draft|derived|substantive|missing|passing|True|True|None|False|False|True|True|True|False|1",
    "accepted|reviewed|valid|draft|pipeline|substantive|missing|below|True|True|None|False|False|False|True|False|False|1",
    "accepted|reviewed|valid|draft|pipeline|substantive|missing|below|True|True|hint|False|False|False|True|True|False|2",
    "accepted|reviewed|valid|draft|pipeline|substantive|missing|passing|True|True|hint|False|False|False|True|True|False|1",
    "accepted|reviewed|valid|draft|pipeline|substantive|passing|passing|True|True|hint|False|False|False|False|False|False|1",
    "accepted|reviewed|valid|draft|pipeline|substantive|passing|passing|True|True|hint|False|False|False|False|True|False|2",
    "drafted|accepted|pending|draft|None|substantive|missing|missing|True|True|None|False|False|False|False|False|False|6",
    "drafted|accepted|pending|draft|None|substantive|missing|missing|True|True|hint|False|False|False|False|False|False|3",
    "drafted|accepted|pending|draft|None|substantive|missing|missing|True|True|hint|False|False|False|True|False|False|1",
    "drafted|accepted|pending|draft|None|substantive|missing|passing|False|True|None|False|False|False|False|True|False|1",
    "drafted|accepted|pending|draft|None|substantive|missing|passing|False|True|hint|False|False|False|False|True|False|1",
    "drafted|accepted|pending|draft|None|substantive|missing|passing|True|True|None|False|False|False|False|True|False|12",
    "drafted|accepted|pending|draft|None|substantive|missing|passing|True|True|hint|False|False|False|False|True|False|9",
    "drafted|accepted|pending|draft|None|substantive|missing|passing|True|True|hint|False|False|False|True|True|False|5",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|missing|False|True|hint|False|False|False|True|False|False|1",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|missing|True|True|None|False|False|False|False|False|False|1",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|missing|True|True|hint|False|False|False|True|False|False|1",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|passing|False|True|hint|False|False|False|False|True|False|1",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|passing|True|True|None|False|False|False|False|True|False|1",
    "drafted|accepted|pending|draft|pipeline|substantive|missing|passing|True|True|hint|False|False|False|False|True|False|3",
    "drafted|accepted|quarantined|draft|pipeline|substantive|missing|missing|False|True|rename|False|False|False|False|False|False|1",
    "drafted|exhausted|quarantined|draft|pipeline|substantive|missing|missing|False|False|rename|False|False|False|False|False|False|1",
    "drafted|pending|pending|draft|None|substantive|missing|below|True|True|hint|False|False|False|False|True|False|1",
    "drafted|pending|pending|draft|None|substantive|missing|missing|False|True|None|False|False|False|False|False|False|1",
    "drafted|pending|pending|draft|None|substantive|missing|missing|False|True|hint|False|False|False|False|False|False|2",
    "drafted|pending|pending|draft|None|substantive|missing|missing|True|True|None|False|False|False|False|False|False|23",
    "drafted|pending|pending|draft|None|substantive|missing|missing|True|True|hint|False|False|False|False|False|False|42",
    "drafted|pending|pending|draft|None|substantive|missing|missing|True|True|hint|False|False|False|True|False|False|2",
    "drafted|pending|pending|draft|None|substantive|missing|missing|True|True|hint|False|False|False|True|True|False|1",
    "drafted|pending|pending|draft|pipeline|substantive|missing|missing|False|True|None|False|False|False|False|False|False|5",
    "drafted|pending|pending|draft|pipeline|substantive|missing|missing|False|True|hint|False|False|False|False|False|False|4",
    "drafted|pending|pending|draft|pipeline|substantive|missing|missing|True|True|None|False|False|False|False|False|False|9",
    "drafted|pending|pending|draft|pipeline|substantive|missing|missing|True|True|hint|False|False|False|False|False|False|14",
    "drafted|pending|pending|draft|pipeline|substantive|missing|missing|True|True|hint|False|False|False|True|False|False|2",
    "drafted|pending|quarantined|draft|None|substantive|missing|missing|True|True|None|False|False|False|False|False|False|6",
    "drafted|pending|quarantined|draft|None|substantive|missing|missing|True|True|None|False|False|False|True|False|False|1",
    "drafted|pending|quarantined|draft|None|substantive|missing|missing|True|True|hint|False|False|False|False|False|False|2",
    "drafted|pending|quarantined|draft|pipeline|substantive|missing|missing|False|True|hint|False|False|False|False|False|False|1",
    "drafted|pending|quarantined|draft|pipeline|substantive|missing|missing|False|True|rename|False|False|False|False|False|False|1",
    "drafted|pending|quarantined|draft|pipeline|substantive|missing|missing|True|True|hint|False|False|False|False|False|False|1",
    "drafted|pending|quarantined|draft|pipeline|substantive|missing|missing|True|True|rename|False|False|False|False|False|False|3",
    "pending|None|None|draft|derived|missing|missing|missing|True|True|None|False|False|False|True|False|False|2",
    "pending|None|pending|draft|None|missing|missing|missing|True|True|None|False|False|False|False|False|False|1",
    "pending|pending|pending|draft|None|missing|missing|missing|True|True|None|False|False|False|False|False|False|3",
    "reviewed|accepted|valid|draft|pipeline|substantive|below|missing|True|True|hint|False|True|False|False|False|False|4",
    "reviewed|accepted|valid|draft|pipeline|substantive|below|missing|True|True|rename|False|True|False|False|False|False|3",
    "reviewed|accepted|valid|draft|pipeline|substantive|below|passing|True|True|None|False|True|False|False|True|False|1",
    "reviewed|accepted|valid|draft|pipeline|substantive|below|passing|True|True|hint|False|True|False|False|True|False|1",
    "reviewed|accepted|valid|draft|pipeline|substantive|passing|missing|True|True|hint|False|True|False|False|False|False|1",
    "reviewed|accepted|valid|draft|pipeline|substantive|passing|passing|True|True|hint|False|False|False|False|True|False|2",
    "reviewed|pending|quarantined|draft|None|substantive|below|missing|True|True|hint|False|True|False|False|False|False|1",
    "reviewed|pending|quarantined|draft|None|substantive|passing|missing|True|True|hint|False|True|False|False|False|False|1",
    "reviewed|pending|quarantined|draft|pipeline|substantive|below|missing|True|True|rename|True|False|False|False|False|False|1",
    "reviewed|pending|valid|draft|None|substantive|below|missing|False|True|None|False|False|False|False|False|False|1",
    "reviewed|pending|valid|draft|None|substantive|below|missing|True|True|None|False|True|False|False|False|False|17",
    "reviewed|pending|valid|draft|None|substantive|below|missing|True|True|hint|False|True|False|False|False|False|16",
    "reviewed|pending|valid|draft|None|substantive|passing|missing|True|True|None|False|True|False|False|False|False|5",
    "reviewed|pending|valid|draft|None|substantive|passing|missing|True|True|hint|False|True|False|False|False|False|11",
    "reviewed|pending|valid|draft|None|substantive|passing|missing|True|True|hint|False|True|False|True|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|below|missing|True|True|None|False|True|False|False|False|False|10",
    "reviewed|pending|valid|draft|pipeline|substantive|below|missing|True|True|hint|False|True|False|False|False|False|10",
    "reviewed|pending|valid|draft|pipeline|substantive|below|missing|True|True|rename|False|True|False|False|False|False|2",
    "reviewed|pending|valid|draft|pipeline|substantive|below|missing|True|True|rename|False|True|False|True|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|below|missing|True|True|rename|True|False|False|False|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|below|passing|True|True|None|False|True|False|True|True|False|2",
    "reviewed|pending|valid|draft|pipeline|substantive|below|passing|True|True|hint|False|True|False|False|True|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|below|passing|True|True|hint|False|True|False|True|True|False|3",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|missing|False|True|None|False|True|False|False|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|missing|False|True|rename|False|True|False|False|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|missing|True|True|None|False|True|False|False|False|False|2",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|missing|True|True|hint|False|False|False|True|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|missing|True|True|hint|False|True|False|False|False|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|passing|True|True|hint|False|False|False|True|True|False|1",
    "reviewed|pending|valid|draft|pipeline|substantive|passing|passing|True|True|hint|False|True|False|False|True|False|3",
)


def recorded_states():
    rows = []
    for index, encoded in enumerate(_STATE_ROWS):
        *values, count = encoded.split("|")
        state = dict(
            zip(
                _STATE_FIELDS,
                (
                    {"True": True, "False": False, "None": None}.get(value, value)
                    for value in values
                ),
                strict=True,
            )
        )
        state.update(
            identities=[f"admission_fixture_{index}"],
            count=1,
            recorded_count=int(count),
        )
        rows.append(state)
    return rows


def control_states():
    base = dict(
        recorded_states()[0],
        origin="pipeline",
        name_score_state="passing",
        docs_stage="pending",
        description_state="substantive",
        has_live_child=False,
    )
    overrides = {
        "review_name": {"name_stage": "drafted", "name_score_state": "missing"},
        "refine_name": {"name_stage": "reviewed", "name_score_state": "below"},
        "generate_docs": {"name_stage": "accepted"},
        "review_docs": {"name_stage": "accepted", "docs_stage": "drafted"},
        "refine_docs": {
            "name_stage": "accepted",
            "docs_stage": "reviewed",
            "docs_score_state": "below",
            "has_winning_docs_review": True,
        },
        "enrich_parents": {
            "name_stage": "pending",
            "origin": "derived",
            "description_state": "placeholder",
            "has_live_child": True,
            "name_score_state": "missing",
        },
    }
    return {
        pool: dict(base, **values, identities=[f"control_{pool}"])
        for pool, values in overrides.items()
    }


def test_recorded_classes_preserve_population_and_admission():
    states = recorded_states()
    assert len(states) == 85
    assert sum(state["recorded_count"] for state in states) == 360
    result = classify_identity_state_groups(states, standard_name_pool_predicates())
    assert result["stranded_count"] == 85
    assert result["overlap_count"] == 0


@pytest.mark.parametrize("pool", list(control_states()))
def test_each_pool_has_a_positive_control(pool):
    state = control_states()[pool]
    assert [
        predicate.pool
        for predicate in standard_name_pool_predicates()
        if predicate.admits(state)
    ] == [pool]


def _plant_states(graph, states):
    nodes = []
    for state in states:
        properties = {
            key: state[key]
            for key in (
                "name_stage",
                "docs_stage",
                "validation_status",
                "status",
                "origin",
                "edit_mode",
            )
        }
        properties.update(
            id=state["identities"][0],
            description={
                "missing": None,
                "placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
                "substantive": "A measured physical quantity.",
            }[state["description_state"]],
            reviewer_score_name={"missing": None, "below": 0.5, "passing": 0.9}[
                state["name_score_state"]
            ],
            reviewer_score_docs={"missing": None, "below": 0.5, "passing": 0.9}[
                state["docs_score_state"]
            ],
            refine_attempts=0 if state["name_attempts_under_cap"] else 3,
            docs_chain_length=0 if state["docs_attempts_under_cap"] else 3,
            review_resubmit_count=3 if state["rename_resubmit_capped"] else 0,
            review_quorum_shortfall="review chain incomplete"
            if state["review_quorum_shortfall"]
            else None,
            docs_review_quorum_shortfall="review chain incomplete"
            if state["docs_review_quorum_shortfall"]
            else None,
            claim_token="owned" if state["claim_active"] else None,
        )
        nodes.append(
            {
                "properties": properties,
                "child": state["has_live_child"],
                "review": state["has_winning_docs_review"],
            }
        )
    graph.query(
        """
        UNWIND $nodes AS row
        CREATE (sn:StandardName) SET sn = row.properties
        FOREACH (_ IN CASE WHEN row.child THEN [1] ELSE [] END |
          CREATE (child:StandardName {id: sn.id + '_child', name_stage: 'accepted', docs_stage: 'accepted', status: 'draft'})
          CREATE (child)-[:HAS_PARENT]->(sn))
        FOREACH (_ IN CASE WHEN row.review THEN [1] ELSE [] END |
          CREATE (review:StandardNameReview {id: sn.id + '_review', review_axis: 'docs', review_group_id: sn.id, resolution_method: 'quorum_consensus', score: 0.9, reviewed_at: datetime()})
          CREATE (sn)-[:HAS_REVIEW]->(review))
        RETURN count(sn) AS planted
    """,
        nodes=nodes,
    )


@pytest.mark.graph
@pytest.mark.timeout(120)
def test_rendered_admission_matches_planted_classes(disposable_neo4j_empty):
    controls = control_states()
    states = recorded_states() + list(controls.values())
    ids = {state["identities"][0] for state in states}
    params = dict(
        min_score=0.75,
        rotation_cap=3,
        parent_desc_placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        **docs_review_eligibility_params(),
    )
    with GraphClient() as graph:
        _plant_states(graph, states)
        coverage = graph.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids RETURN count(sn) AS n",
            ids=sorted(ids),
        )
        assert coverage == [{"n": 91}]
        groups = identity_state_groups(gc=graph)
        projected = {
            identity: group
            for group in groups
            for identity in group["identities"]
            if identity in ids
        }
        assert set(projected) == ids
        for state in states:
            actual = projected[state["identities"][0]]
            assert {field: actual[field] for field in _STATE_FIELDS} == {
                field: state[field] for field in _STATE_FIELDS
            }
        for predicate in standard_name_pool_predicates():
            rows = graph.query(
                f"MATCH (sn:StandardName) WHERE sn.id IN $ids AND ({predicate.cypher()}) RETURN sn.id AS id",
                ids=sorted(ids),
                **params,
            )
            expected = {
                state["identities"][0] for state in states if predicate.admits(state)
            }
            assert expected == {f"control_{predicate.pool}"}
            assert {row["id"] for row in rows} == expected
