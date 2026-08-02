"""Exact-scope semantic source-invariant repair tests."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    repair_semantic_source_invariants,
)


def test_dd_projection_reconcile_uses_full_attachment_guard() -> None:
    from imas_codex.standard_names.graph_ops import reconcile_standard_name_dd_edges

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "dd_path": "pf_active/coil/element/geometry/rectangle/r",
                "sn_id": "radial_coordinate_of_line_of_sight",
                "name": "radial_coordinate_of_line_of_sight",
                "sn_unit": "m",
                "dd_unit": "m",
                "existing_dd_paths": [],
            },
            {
                "dd_path": (
                    "bremsstrahlung_visible/channel/line_of_sight/first_point/r"
                ),
                "sn_id": "radial_coordinate_of_line_of_sight",
                "name": "radial_coordinate_of_line_of_sight",
                "sn_unit": "m",
                "dd_unit": "m",
                "existing_dd_paths": [],
            },
        ],
        [],
    ]

    result = reconcile_standard_name_dd_edges(gc=gc)

    assert result == {"edges_created": 1, "pairs_dropped": 1}
    assert gc.query.call_args_list[1].kwargs["pairs"] == [
        {
            "dd_path": ("bremsstrahlung_visible/channel/line_of_sight/first_point/r"),
            "sn_id": "radial_coordinate_of_line_of_sight",
        }
    ]


def _row(
    source_id: str,
    *,
    source_type: str = "dd",
    scalar: str | None,
    produced: list[str],
    live: list[str],
    mapped: list[str],
    semantic_id: str | None = None,
    target_states: list[dict] | None = None,
    dd_backings: list[str] | None = None,
    signal_backings: list[str] | None = None,
    backing_owner_ids: list[str] | None = None,
) -> dict:
    if semantic_id is None:
        prefix = "dd:" if source_type == "dd" else "signals:"
        semantic_id = source_id.removeprefix(prefix)
    if dd_backings is None:
        dd_backings = [semantic_id] if source_type == "dd" else []
    if signal_backings is None:
        signal_backings = [semantic_id] if source_type == "signals" else []
    return {
        "source_id": source_id,
        "semantic_id": semantic_id,
        "source_type": source_type,
        "status": "composed",
        "produced_sn_id": scalar,
        "produced_targets": produced,
        "live_targets": live,
        "target_states": target_states or [],
        "dd_backings": dd_backings,
        "signal_backings": signal_backings,
        "mapped_ids": mapped,
        "backing_owner_ids": backing_owner_ids or [source_id],
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
            "backing identity or ownership is invalid",
        ),
        (
            _row(
                "dd:canonical/path",
                scalar="quantity",
                produced=["quantity"],
                live=["quantity"],
                mapped=["quantity"],
                dd_backings=["wrong/path"],
            ),
            "backing identity or ownership is invalid",
        ),
        (
            _row(
                "signals:west:diagnostic/channel",
                source_type="signals",
                scalar="quantity",
                produced=["quantity"],
                live=["quantity"],
                mapped=["quantity"],
                signal_backings=["west:other/channel"],
            ),
            "backing identity or ownership is invalid",
        ),
        (
            _row(
                "dd:shared/path",
                scalar="quantity",
                produced=["quantity"],
                live=["quantity"],
                mapped=["quantity"],
                backing_owner_ids=["dd:shared/path", "dd:other/source"],
            ),
            "backing identity or ownership is invalid",
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
    before: list[dict],
    after: list[dict],
    *,
    mutation_rows: list[dict],
    path_rows: list[dict] | None = None,
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
        if "AS hsn_paths" in cypher:
            return path_rows or [
                {
                    "id": "discarded",
                    "current": ["derived:discarded_parent", "stale/path"],
                    "hsn_paths": ["dd:direct/discarded"],
                    "produced_paths": [],
                },
                {
                    "id": "selected",
                    "current": ["derived:selected_parent", "stale/path"],
                    "hsn_paths": ["dd:direct/selected"],
                    "produced_paths": ["dd:path", "catalog:selected"],
                },
            ]
        if "SET sn.source_paths = update.paths" in cypher:
            return _params["updates"]
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
    assert "source.source_id = $semantic_id" in mutation.args[0]
    assert "size(backing_owner_ids) = size($backing_owner_ids)" in mutation.args[0]
    assert "FOREACH (edge IN stale_edges | DELETE edge)" in mutation.args[0]
    assert "HAS_INTERNAL_CHANGE" in mutation.args[0]
    assert '"removed_targets":["discarded"]' in mutation.kwargs["audit_reason"]
    inspection = next(
        call
        for call in transaction.run.call_args_list
        if "AS hsn_paths" in call.args[0]
    )
    assert inspection.kwargs["name_ids"] == ["discarded", "selected"]
    assert "MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn)" in inspection.args[0]
    assert "source.source_type <> 'derived'" in inspection.args[0]
    rebuild = next(
        call
        for call in transaction.run.call_args_list
        if "SET sn.source_paths = update.paths" in call.args[0]
    )
    assert rebuild.kwargs["updates"] == [
        {
            "id": "discarded",
            "paths": ["dd:direct/discarded", "derived:discarded_parent"],
        },
        {
            "id": "selected",
            "paths": [
                "catalog:selected",
                "dd:direct/selected",
                "dd:path",
                "derived:selected_parent",
            ],
        },
    ]


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


def test_postflight_compares_complete_ambiguous_source_state() -> None:
    planned = _row(
        "dd:path",
        scalar="selected",
        produced=["discarded", "selected"],
        live=["discarded", "selected"],
        mapped=["discarded"],
    )
    ambiguous = _row(
        "dd:ambiguous",
        scalar="outside",
        produced=["left", "right"],
        live=["left", "right"],
        mapped=["left"],
    )
    repaired = _row(
        "dd:path",
        scalar="selected",
        produced=["selected"],
        live=["selected"],
        mapped=["selected"],
    )
    changed_ambiguous = {**ambiguous, "mapped_ids": ["right"]}
    gc, transaction = _transactional_client(
        [planned, ambiguous],
        [repaired, changed_ambiguous],
        mutation_rows=[
            {
                "source_id": "dd:path",
                "target": "selected",
                "change_id": "sn-change:test",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="ambiguous semantic source set changed"):
        repair_semantic_source_invariants(
            gc,
            ["dd:path", "dd:ambiguous"],
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
        for key in [
            "scoped",
            "collateral",
            "ambiguous",
            "outside",
            "malformed",
            "direct",
            "signal",
            "shared",
        ]
    }
    sources = {
        key: f"dd:{path}"
        for key, path in paths.items()
        if key not in {"direct", "signal"}
    }
    sources.update(
        {
            "signal": f"signals:{paths['signal']}",
            "catalog": f"catalog:{prefix}_catalog",
            "shared_other": f"dd:{paths['shared']}/alias",
        }
    )

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
        paths=[
            path for key, path in paths.items() if key not in {"malformed", "signal"}
        ],
    )
    graph_client.query(
        "CREATE (:FacilitySignal {id: $signal})",
        signal=paths["signal"],
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
                  (outside_dd:IMASNode {id: $outside_path}),
                  (direct_dd:IMASNode {id: $direct_path}),
                  (shared_dd:IMASNode {id: $shared_path}),
                  (signal_node:FacilitySignal {id: $signal_path})
            SET selected.source_paths = ['derived:structural_only', 'stale/cache']
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
            CREATE (signal_source:StandardNameSource {
              id: $signal_source, source_id: $signal_path,
              source_type: 'signals', status: 'attached', produced_sn_id: $selected
            })
            CREATE (catalog:StandardNameSource {
              id: $catalog_source, source_id: $selected,
              source_type: 'catalog', status: 'attached', produced_sn_id: $selected
            })
            CREATE (shared:StandardNameSource {
              id: $shared_source, source_id: $shared_path,
              source_type: 'dd', status: 'composed', produced_sn_id: $selected
            })
            CREATE (shared_other:StandardNameSource {
              id: $shared_other_source, source_id: $shared_path + '/alias',
              source_type: 'dd', status: 'extracted'
            })
            CREATE (scoped)-[:FROM_DD_PATH]->(scoped_dd)
            CREATE (collateral)-[:FROM_DD_PATH]->(collateral_dd)
            CREATE (ambiguous)-[:FROM_DD_PATH]->(ambiguous_dd)
            CREATE (outside_source)-[:FROM_DD_PATH]->(outside_dd)
            CREATE (signal_source)-[:FROM_SIGNAL]->(signal_node)
            CREATE (shared)-[:FROM_DD_PATH]->(shared_dd)
            CREATE (shared_other)-[:FROM_DD_PATH]->(shared_dd)
            CREATE (scoped)-[:PRODUCED_NAME]->(discarded)
            CREATE (scoped)-[:PRODUCED_NAME]->(selected)
            CREATE (collateral)-[:PRODUCED_NAME]->(selected)
            CREATE (ambiguous)-[:PRODUCED_NAME]->(discarded)
            CREATE (ambiguous)-[:PRODUCED_NAME]->(selected)
            CREATE (outside_source)-[:PRODUCED_NAME]->(discarded)
            CREATE (malformed)-[:PRODUCED_NAME]->(selected)
            CREATE (signal_source)-[:PRODUCED_NAME]->(selected)
            CREATE (catalog)-[:PRODUCED_NAME]->(selected)
            CREATE (shared)-[:PRODUCED_NAME]->(discarded)
            CREATE (shared)-[:PRODUCED_NAME]->(selected)
            CREATE (scoped_dd)-[:HAS_STANDARD_NAME]->(discarded)
            CREATE (collateral_dd)-[:HAS_STANDARD_NAME]->(selected)
            CREATE (ambiguous_dd)-[:HAS_STANDARD_NAME]->(discarded)
            CREATE (outside_dd)-[:HAS_STANDARD_NAME]->(discarded)
            CREATE (direct_dd)-[:HAS_STANDARD_NAME]->(selected)
            CREATE (signal_node)-[:HAS_STANDARD_NAME]->(selected)
            CREATE (shared_dd)-[:HAS_STANDARD_NAME]->(discarded)
            """,
            discarded=names["discarded"],
            selected=names["selected"],
            outside=names["outside"],
            **{f"{key}_path": value for key, value in paths.items()},
            **{f"{key}_source": value for key, value in sources.items()},
        )

        shared_before = graph_client.query(
            """
            MATCH (source:StandardNameSource {id: $source})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            WITH source, collect(target.id) AS targets
            MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
            OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(mapped:StandardName)
            RETURN source.produced_sn_id AS scalar, targets,
                   collect(mapped.id) AS mapped
            """,
            source=sources["shared"],
        )[0]
        with pytest.raises(
            ValueError, match="backing identity or ownership is invalid"
        ):
            repair_semantic_source_invariants(
                graph_client,
                [sources["shared"]],
                reason=f"{prefix} refuse shared backing",
                dry_run=False,
            )
        shared_after = graph_client.query(
            """
            MATCH (source:StandardNameSource {id: $source})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            WITH source, collect(target.id) AS targets
            MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
            OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(mapped:StandardName)
            RETURN source.produced_sn_id AS scalar, targets,
                   collect(mapped.id) AS mapped
            """,
            source=sources["shared"],
        )[0]
        assert shared_after == shared_before

        with pytest.raises(
            ValueError, match="backing identity or ownership is invalid"
        ):
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
                f"dd:{paths['direct']}",
                f"dd:{paths['shared']}",
                sources["malformed"],
                sources["signal"],
                sources["catalog"],
                "derived:structural_only",
            ]
        )
        assert cache_by_name[names["discarded"]] == sorted(
            [
                f"dd:{paths['ambiguous']}",
                f"dd:{paths['outside']}",
                f"dd:{paths['shared']}",
            ]
        )
        assert "stale/cache" not in cache_by_name[names["selected"]]

        shared_projection = graph_client.query(
            """
            MATCH (:IMASNode {id: $path})-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN collect(sn.id) AS targets
            """,
            path=paths["shared"],
        )[0]["targets"]
        assert shared_projection == [names["discarded"]]

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
