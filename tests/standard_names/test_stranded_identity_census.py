"""The worker-pool state census distinguishes stranded identities."""

from datetime import UTC, datetime

from imas_codex.standard_names.loop import RunSummary, summary_table
from imas_codex.standard_names.pools import (
    classify_identity_state_groups,
    identity_state_groups,
    standard_name_pool_predicates,
)


def _group(identity: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name_stage": "reviewed",
        "docs_stage": "pending",
        "validation_status": "valid",
        "status": "draft",
        "origin": "generated",
        "description_state": "substantive",
        "name_score_state": "below",
        "docs_score_state": "missing",
        "name_attempts_under_cap": True,
        "docs_attempts_under_cap": True,
        "edit_mode": None,
        "rename_resubmit_capped": False,
        "review_quorum_shortfall": False,
        "docs_review_quorum_shortfall": False,
        "has_live_child": False,
        "has_winning_docs_review": False,
        "claim_active": False,
        "identities": [identity],
        "count": 1,
    }
    row.update(overrides)
    return row


def _classify(*groups: dict[str, object]) -> dict[str, object]:
    return classify_identity_state_groups(list(groups), standard_name_pool_predicates())


def test_review_quorum_shortfall_is_reported_with_its_state() -> None:
    result = _classify(_group("reviewed_shortfall", review_quorum_shortfall=True))

    assert result["stranded_count"] == 1
    assert result["stranded_ids"] == ["reviewed_shortfall"]
    assert list(result["by_state"].values()) == [1]
    state = next(iter(result["by_state"]))
    assert "name_stage=reviewed" in state
    assert "review_quorum_shortfall=true" in state


def test_drafted_quarantined_identity_is_reported() -> None:
    result = _classify(
        _group(
            "drafted_quarantined",
            name_stage="drafted",
            validation_status="quarantined",
            name_score_state="missing",
        )
    )

    assert result["stranded_count"] == 1
    assert result["stranded_ids"] == ["drafted_quarantined"]


def test_identity_admitted_by_a_pool_is_not_reported() -> None:
    result = _classify(
        _group(
            "drafted_valid",
            name_stage="drafted",
            name_score_state="missing",
        )
    )

    assert result["population_count"] == 1
    assert result["claimable_count"] == 1
    assert result["stranded_count"] == 0
    assert result["stranded_ids"] == []


def test_terminal_identity_is_not_reported() -> None:
    result = _classify(_group("accepted_name", name_stage="accepted"))

    assert result["terminal_count"] == 1
    assert result["stranded_count"] == 0


def test_grouped_query_and_run_summary_preserve_the_census() -> None:
    rows = [
        _group("first", review_quorum_shortfall=True),
        _group("second", review_quorum_shortfall=True),
    ]
    rows[0]["identities"] = ["first", "second"]
    rows[0]["count"] = 2
    rows.pop()

    class Graph:
        query_text = ""

        def query(self, statement: str, **params: object) -> list[dict[str, object]]:
            self.query_text = statement
            assert params["min_score"] == 0.75
            assert params["rotation_cap"] == 3
            return rows

    graph = Graph()
    groups = identity_state_groups(gc=graph)
    result = classify_identity_state_groups(groups, standard_name_pool_predicates())

    assert "collect(sn.id) AS identities" in graph.query_text
    assert "sn.name_stage AS name_stage" in graph.query_text
    assert "sn.docs_stage AS docs_stage" in graph.query_text
    assert "sn.validation_status AS validation_status" in graph.query_text
    assert "sn.status AS status" in graph.query_text
    assert result["stranded_count"] == 2

    summary = RunSummary(
        run_id="run",
        turn_number=1,
        started_at=datetime.now(UTC),
        stranded_count=2,
        stranded_by_state=result["by_state"],
        stranded_ids=result["stranded_ids"],
    )
    rendered = summary_table(summary)
    assert rendered["stranded_count"] == 2
    assert rendered["stranded_by_state"] == result["by_state"]
    assert rendered["stranded_ids"] == ["first", "second"]
