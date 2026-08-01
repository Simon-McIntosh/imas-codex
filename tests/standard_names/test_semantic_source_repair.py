"""Exact-scope semantic source-invariant repair tests."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    repair_semantic_source_invariants,
)


def _row(
    source_id: str,
    *,
    source_type: str = "dd",
    scalar: str | None,
    produced: list[str],
    live: list[str],
    mapped: list[str],
    target_states: list[dict] | None = None,
    dd_backings: list[str] | None = None,
    signal_backings: list[str] | None = None,
) -> dict:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "status": "composed",
        "produced_sn_id": scalar,
        "produced_targets": produced,
        "live_targets": live,
        "target_states": target_states or [],
        "dd_backings": dd_backings if dd_backings is not None else [source_id[3:]],
        "signal_backings": signal_backings or [],
        "mapped_ids": mapped,
    }


def test_dry_run_classifies_authority_without_opening_a_transaction() -> None:
    rows = [
        _row(
            "dd:sole",
            scalar="stale",
            produced=["authoritative"],
            live=["authoritative"],
            mapped=["stale"],
        ),
        _row(
            "dd:multiple",
            scalar="selected",
            produced=["discarded", "selected"],
            live=["discarded", "selected"],
            mapped=["discarded"],
        ),
        _row(
            "dd:ambiguous",
            scalar="absent",
            produced=["left", "right"],
            live=["left", "right"],
            mapped=["left"],
        ),
        _row(
            "dd:clean",
            scalar="clean_name",
            produced=["clean_name"],
            live=["clean_name"],
            mapped=["clean_name"],
        ),
    ]
    gc = MagicMock()
    gc.query.return_value = rows

    result = repair_semantic_source_invariants(
        gc,
        ["dd:sole", "dd:multiple", "dd:ambiguous", "dd:clean"],
        reason="repair reviewed exact bindings",
    )

    assert [row["source_id"] for row in result["planned"]] == [
        "dd:multiple",
        "dd:sole",
    ]
    assert result["planned"][0]["authoritative_target"] == "selected"
    assert result["planned"][0]["removed_targets"] == ["discarded"]
    assert [row["source_id"] for row in result["ambiguous"]] == ["dd:ambiguous"]
    assert [row["source_id"] for row in result["already_clean"]] == ["dd:clean"]
    assert result["repaired"] == []
    gc.session.assert_not_called()
    assert "DELETE" not in gc.query.call_args.args[0]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            _row(
                "derived:parent",
                source_type="derived",
                scalar="parent",
                produced=["parent"],
                live=["parent"],
                mapped=[],
                dd_backings=[],
            ),
            "unsupported semantic source kinds",
        ),
        (
            _row(
                "dd:two_backings",
                scalar="quantity",
                produced=["quantity"],
                live=["quantity"],
                mapped=["quantity"],
                dd_backings=["one", "two"],
            ),
            "backing projection is ambiguous",
        ),
    ],
)
def test_preflight_fails_loudly_on_unsupported_topology(
    row: dict, message: str
) -> None:
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(ValueError, match=message):
        repair_semantic_source_invariants(
            gc,
            [row["source_id"]],
            reason="refuse unsupported topology",
        )

    gc.session.assert_not_called()


def test_multi_live_lower_authority_scalar_requires_explicit_override() -> None:
    row = _row(
        "dd:policy_conflict",
        scalar="reviewed_candidate",
        produced=["reviewed_candidate", "accepted_candidate"],
        live=["reviewed_candidate", "accepted_candidate"],
        mapped=["reviewed_candidate"],
        target_states=[
            {
                "id": "reviewed_candidate",
                "stage": "reviewed",
                "validation_status": "valid",
            },
            {
                "id": "accepted_candidate",
                "stage": "accepted",
                "validation_status": "valid",
            },
        ],
    )
    gc = MagicMock()
    gc.query.return_value = [row]

    refused = repair_semantic_source_invariants(
        gc,
        ["dd:policy_conflict"],
        reason="inspect lifecycle authority",
    )
    assert refused["planned"] == []
    assert refused["ambiguous"][0]["classification"] == "policy_conflict"
    assert refused["ambiguous"][0]["accepted_valid_competitors"] == [
        "accepted_candidate"
    ]

    overridden = repair_semantic_source_invariants(
        gc,
        ["dd:policy_conflict"],
        reason="apply reviewed authority adjudication",
        authority_overrides={"dd:policy_conflict": "accepted_candidate"},
    )
    assert overridden["planned"][0]["authoritative_target"] == "accepted_candidate"
    assert overridden["planned"][0]["authority_basis"] == "explicit_authority_override"

    with pytest.raises(ValueError, match="not exactly one current live target"):
        repair_semantic_source_invariants(
            gc,
            ["dd:policy_conflict"],
            reason="reject stale authority adjudication",
            authority_overrides={"dd:policy_conflict": "absent_candidate"},
        )


def _transactional_client(
    before: list[dict], after: list[dict], *, mutation_rows: list[dict]
) -> tuple[MagicMock, MagicMock]:
    gc = MagicMock()
    transaction = MagicMock()
    inspection_count = 0

    def run(cypher: str, **_params):
        nonlocal inspection_count
        if "source.produced_sn_id AS produced_sn_id" in cypher:
            inspection_count += 1
            return before if inspection_count == 1 else after
        if "repair_semantic_source_binding" in cypher:
            return mutation_rows
        if "SET sn.source_paths = paths" in cypher:
            return []
        raise AssertionError(f"unexpected query: {cypher}")

    transaction.run.side_effect = run
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    gc.session.return_value.__enter__.return_value = session
    return gc, transaction


def test_apply_is_atomic_audited_and_rebuilds_complete_name_caches() -> None:
    before = [
        _row(
            "dd:path",
            scalar="selected",
            produced=["discarded", "selected"],
            live=["discarded", "selected"],
            mapped=["discarded"],
        )
    ]
    after = [
        _row(
            "dd:path",
            scalar="selected",
            produced=["selected"],
            live=["selected"],
            mapped=["selected"],
        )
    ]
    gc, transaction = _transactional_client(
        before,
        after,
        mutation_rows=[
            {
                "source_id": "dd:path",
                "target": "selected",
                "change_id": "sn-change:test",
            }
        ],
    )

    result = repair_semantic_source_invariants(
        gc,
        ["dd:path"],
        reason="repair exact binding",
        dry_run=False,
        run_id="run:test",
    )

    assert result["repaired"][0]["change_id"] == "sn-change:test"
    transaction.commit.assert_called_once()
    transaction.rollback.assert_not_called()
    mutation = next(
        call
        for call in transaction.run.call_args_list
        if "repair_semantic_source_binding" in call.args[0]
    )
    assert "WHERE size(current_targets) = size($before_targets)" in mutation.args[0]
    assert "FOREACH (edge IN stale_edges | DELETE edge)" in mutation.args[0]
    assert "HAS_INTERNAL_CHANGE" in mutation.args[0]
    assert '"removed_targets":["discarded"]' in mutation.kwargs["audit_reason"]
    rebuild = next(
        call
        for call in transaction.run.call_args_list
        if "SET sn.source_paths = paths" in call.args[0]
    )
    assert rebuild.kwargs["name_ids"] == ["discarded", "selected"]
    assert "MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)" in rebuild.args[0]
    assert "source.id" in rebuild.args[0]


def test_apply_rolls_back_complete_batch_when_compare_check_fails() -> None:
    before = [
        _row(
            "dd:path",
            scalar="stale",
            produced=["selected"],
            live=["selected"],
            mapped=["stale"],
        )
    ]
    gc, transaction = _transactional_client(before, before, mutation_rows=[])

    with pytest.raises(RuntimeError, match="changed during repair"):
        repair_semantic_source_invariants(
            gc,
            ["dd:path"],
            reason="repair exact binding",
            dry_run=False,
        )

    transaction.commit.assert_not_called()
    transaction.rollback.assert_called_once()


@pytest.mark.graph
def test_live_repair_is_exact_fail_closed_audited_and_idempotent(graph_client) -> None:
    token = uuid4().hex
    prefix = f"semantic_source_repair_test_{token}"
    names = {
        "discarded": f"{prefix}_discarded",
        "selected": f"{prefix}_selected",
        "outside": f"{prefix}_outside",
    }
    paths = {
        key: f"{prefix}/{key}"
        for key in ["scoped", "collateral", "ambiguous", "outside", "malformed"]
    }
    sources = {key: f"dd:{path}" for key, path in paths.items()}

    graph_client.query(
        """
        UNWIND $names AS name
        CREATE (:StandardName {
          id: name, name_stage: 'drafted', source_paths: ['stale/cache']
        })
        """,
        names=list(names.values()),
    )
    graph_client.query(
        """
        UNWIND $paths AS path
        CREATE (:IMASNode {id: path})
        """,
        paths=[path for key, path in paths.items() if key != "malformed"],
    )
    try:
        graph_client.query(
            """
            MATCH (discarded:StandardName {id: $discarded}),
                  (selected:StandardName {id: $selected}),
                  (outside:StandardName {id: $outside}),
                  (scoped_dd:IMASNode {id: $scoped_path}),
                  (collateral_dd:IMASNode {id: $collateral_path}),
                  (ambiguous_dd:IMASNode {id: $ambiguous_path}),
                  (outside_dd:IMASNode {id: $outside_path})
            CREATE (scoped:StandardNameSource {
              id: $scoped_source, source_id: $scoped_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $selected
            })
            CREATE (collateral:StandardNameSource {
              id: $collateral_source, source_id: $collateral_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $selected
            })
            CREATE (ambiguous:StandardNameSource {
              id: $ambiguous_source, source_id: $ambiguous_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $outside
            })
            CREATE (outside_source:StandardNameSource {
              id: $outside_source, source_id: $outside_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $discarded
            })
            CREATE (malformed:StandardNameSource {
              id: $malformed_source, source_id: $malformed_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $selected
            })
            CREATE (scoped)-[:FROM_DD_PATH]->(scoped_dd)
            CREATE (collateral)-[:FROM_DD_PATH]->(collateral_dd)
            CREATE (ambiguous)-[:FROM_DD_PATH]->(ambiguous_dd)
            CREATE (outside_source)-[:FROM_DD_PATH]->(outside_dd)
            CREATE (scoped)-[:PRODUCED_NAME]->(discarded)
            CREATE (scoped)-[:PRODUCED_NAME]->(selected)
            CREATE (collateral)-[:PRODUCED_NAME]->(selected)
            CREATE (ambiguous)-[:PRODUCED_NAME]->(discarded)
            CREATE (ambiguous)-[:PRODUCED_NAME]->(selected)
            CREATE (outside_source)-[:PRODUCED_NAME]->(discarded)
            CREATE (malformed)-[:PRODUCED_NAME]->(selected)
            CREATE (scoped_dd)-[:HAS_STANDARD_NAME]->(discarded)
            CREATE (collateral_dd)-[:HAS_STANDARD_NAME]->(selected)
            CREATE (ambiguous_dd)-[:HAS_STANDARD_NAME]->(discarded)
            CREATE (outside_dd)-[:HAS_STANDARD_NAME]->(discarded)
            """,
            discarded=names["discarded"],
            selected=names["selected"],
            outside=names["outside"],
            **{f"{key}_path": value for key, value in paths.items()},
            **{f"{key}_source": value for key, value in sources.items()},
        )

        with pytest.raises(ValueError, match="backing projection is ambiguous"):
            repair_semantic_source_invariants(
                graph_client,
                [sources["scoped"], sources["malformed"]],
                reason=f"{prefix} fail closed",
                dry_run=False,
            )
        untouched = graph_client.query(
            """
            MATCH (:StandardNameSource {id: $source})-[:PRODUCED_NAME]->(sn)
            RETURN collect(sn.id) AS targets
            """,
            source=sources["scoped"],
        )[0]
        assert sorted(untouched["targets"]) == sorted(
            [names["discarded"], names["selected"]]
        )

        result = repair_semantic_source_invariants(
            graph_client,
            [sources["scoped"], sources["ambiguous"]],
            reason=f"{prefix} exact repair",
            dry_run=False,
        )
        assert [row["source_id"] for row in result["repaired"]] == [sources["scoped"]]
        assert [row["source_id"] for row in result["ambiguous"]] == [
            sources["ambiguous"]
        ]

        state = graph_client.query(
            """
            UNWIND $source_ids AS source_id
            MATCH (source:StandardNameSource {id: source_id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            WITH source, collect(target.id) AS targets
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
            OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(mapped:StandardName)
            RETURN source.id AS source_id, source.produced_sn_id AS scalar,
                   targets, collect(mapped.id) AS mapped
            ORDER BY source_id
            """,
            source_ids=[sources["scoped"], sources["ambiguous"], sources["outside"]],
        )
        by_source = {row["source_id"]: row for row in state}
        assert by_source[sources["scoped"]]["targets"] == [names["selected"]]
        assert by_source[sources["scoped"]]["mapped"] == [names["selected"]]
        assert sorted(by_source[sources["ambiguous"]]["targets"]) == sorted(
            [names["discarded"], names["selected"]]
        )
        assert by_source[sources["ambiguous"]]["scalar"] == names["outside"]
        assert by_source[sources["outside"]]["targets"] == [names["discarded"]]

        caches = graph_client.query(
            """
            MATCH (sn:StandardName) WHERE sn.id IN $names
            RETURN sn.id AS id, sn.source_paths AS paths
            """,
            names=[names["discarded"], names["selected"]],
        )
        cache_by_name = {row["id"]: sorted(row["paths"]) for row in caches}
        assert cache_by_name[names["selected"]] == sorted(
            [
                f"dd:{paths['scoped']}",
                f"dd:{paths['collateral']}",
                f"dd:{paths['ambiguous']}",
                sources["malformed"],
            ]
        )
        assert cache_by_name[names["discarded"]] == sorted(
            [f"dd:{paths['ambiguous']}", f"dd:{paths['outside']}"]
        )

        changes = graph_client.query(
            """
            MATCH (sn:StandardName {id: $selected})
                  -[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            WHERE change.reason CONTAINS $prefix
            RETURN change.operation AS operation, change.reason AS reason
            """,
            selected=names["selected"],
            prefix=prefix,
        )
        assert len(changes) == 1
        assert changes[0]["operation"] == "repair_semantic_source_binding"
        assert sources["scoped"] in changes[0]["reason"]
        assert names["discarded"] in changes[0]["reason"]

        repeated = repair_semantic_source_invariants(
            graph_client,
            [sources["scoped"], sources["ambiguous"]],
            reason=f"{prefix} idempotence check",
            dry_run=False,
        )
        assert repeated["repaired"] == []
        assert [row["source_id"] for row in repeated["already_clean"]] == [
            sources["scoped"]
        ]
        changes_after = graph_client.query(
            """
            MATCH (:StandardName {id: $selected})
                  -[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            WHERE change.reason CONTAINS $prefix
            RETURN count(change) AS count
            """,
            selected=names["selected"],
            prefix=prefix,
        )[0]["count"]
        assert changes_after == 1
    finally:
        graph_client.query(
            """
            MATCH (change:StandardNameChange)
            WHERE change.reason CONTAINS $prefix
            DETACH DELETE change
            """,
            prefix=prefix,
        )
        graph_client.query(
            """
            MATCH (node) WHERE node.id IN $ids
            DETACH DELETE node
            """,
            ids=list(names.values()) + list(paths.values()) + list(sources.values()),
        )
