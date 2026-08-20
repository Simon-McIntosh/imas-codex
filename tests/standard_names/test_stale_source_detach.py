"""Signed stale-source lifecycle detachment tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.provenance_lifecycle import (
    StaleSourceDetachConflict,
    _load_signed_stale_source_rows,
    detach_signed_stale_source_bindings,
)

AUTHORITY_PATH = (
    Path(__file__).parents[2]
    / "docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json"
)
BLOCKING_SOURCE_IDS = (
    "dd:neutron_diagnostic/detectors/aperture/centre/phi",
    "dd:neutron_diagnostic/detectors/detector/centre/phi",
    "dd:refractometer/channel/frequencies",
)
ALREADY_DETACHED_SOURCE_IDS = frozenset(BLOCKING_SOURCE_IDS)
LAST_PRODUCER_REFUSALS = {
    "dd:ece/channel/t_e_voltage": "voltage_of_diagnostic_antenna",
    "dd:equilibrium/time_slice/profiles_1d/b_average": (
        "flux_surface_average_magnetic_field_magnitude"
    ),
    "derived:neutral_state_energy_convection_velocity": (
        "neutral_state_energy_convection_velocity"
    ),
}
MULTI_BINDING_SOURCE_ID = (
    "dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z"
)
SCALAR_MISMATCH_SOURCE_ID = "dd:bolometer/channel/aperture/surface"
DERIVED_SOURCE_ID = "derived:electron_density"


def _write_authority(tmp_path: Path, authority: dict[str, object]) -> Path:
    path = tmp_path / "stale-source-lifecycle.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path


def _resign_rows(authority: dict[str, object]) -> None:
    rows = authority["rows"]
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    authority["signature"]["digest"] = sha256((canonical + "\n").encode()).hexdigest()


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("stale-source detachment requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("stale-source detachment refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    print(f"GRAPH_ENDPOINT={uri}")
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


def _client(endpoint: tuple[str, str]) -> GraphClient:
    uri, password = endpoint
    return GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="signed-stale-source-detach",
    )


def _authority_rows() -> list[dict[str, object]]:
    authority = json.loads(AUTHORITY_PATH.read_text())
    rows = authority["rows"]
    assert len(rows) == 58
    return rows


def _clear_graph(client: GraphClient) -> None:
    client.query("MATCH (node) DETACH DELETE node")


def _seed_signed_live_rows(client: GraphClient) -> list[dict[str, object]]:
    rows = _authority_rows()
    live_rows = [
        row for row in rows if row["source_id"] not in ALREADY_DETACHED_SOURCE_IDS
    ]
    seeded: list[dict[str, object]] = []
    for row in live_rows:
        binding_target_ids = list(row["live_target_ids"])
        if row["source_id"] == "derived:neutral_state_energy_convection_velocity":
            binding_target_ids.append(row["scalar_target"])
        seeded.append({**row, "binding_target_ids": binding_target_ids})

    target_ids = sorted(
        {target_id for row in seeded for target_id in row["binding_target_ids"]}
    )
    client.query("CREATE (:DDVersion {id: '4.1.1', is_current: true})")
    client.query(
        "UNWIND $target_ids AS target_id "
        "CREATE (:StandardName {id: target_id, name_stage: 'accepted', "
        "status: 'draft', source_paths: []})",
        target_ids=target_ids,
    )
    client.query(
        "UNWIND $rows AS row "
        "CREATE (:StandardNameSource {id: row.source_id, "
        "source_id: CASE WHEN row.source_type = 'dd' "
        "THEN substring(row.source_id, 3) ELSE row.source_id END, "
        "source_type: row.source_type, dd_version: row.source_dd_version, "
        "status: 'stale', produced_sn_id: row.scalar_target})",
        rows=seeded,
    )
    client.query(
        "UNWIND $rows AS row UNWIND row.binding_target_ids AS target_id "
        "MATCH (source:StandardNameSource {id: row.source_id}) "
        "MATCH (target:StandardName {id: target_id}) "
        "CREATE (source)-[:PRODUCED_NAME]->(target) "
        "SET target.source_paths = target.source_paths + row.source_id",
        rows=seeded,
    )
    dd_rows = [row for row in seeded if row["source_type"] == "dd"]
    client.query(
        "UNWIND $rows AS row "
        "MATCH (source:StandardNameSource {id: row.source_id}) "
        "CREATE (backing:IMASNode {id: substring(row.source_id, 3), "
        "lifecycle_status: row.backing_lifecycle_status}) "
        "CREATE (source)-[:FROM_DD_PATH]->(backing)",
        rows=dd_rows,
    )
    client.query(
        "UNWIND $rows AS row UNWIND row.binding_target_ids AS target_id "
        "MATCH (source:StandardNameSource {id: row.source_id})"
        "-[:FROM_DD_PATH]->(backing:IMASNode) "
        "MATCH (target:StandardName {id: target_id}) "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(target)",
        rows=dd_rows,
    )
    preserved_targets = sorted(set(target_ids) - set(LAST_PRODUCER_REFUSALS.values()))
    client.query(
        "UNWIND $target_ids AS target_id "
        "MATCH (target:StandardName {id: target_id}) "
        "CREATE (:StandardName {id: 'structural_child_of_' + target_id, "
        "name_stage: 'accepted', status: 'draft'})-[:HAS_PARENT]->(target)",
        target_ids=preserved_targets,
    )
    return seeded


def _admitted_source_ids() -> list[str]:
    return sorted(
        row["source_id"]
        for row in _authority_rows()
        if row["source_id"] not in ALREADY_DETACHED_SOURCE_IDS
        and row["source_id"] not in LAST_PRODUCER_REFUSALS
    )


def test_committed_authority_signature_selects_exact_blocking_rows() -> None:
    file_hash, rows_hash, rows = _load_signed_stale_source_rows(
        AUTHORITY_PATH, BLOCKING_SOURCE_IDS
    )

    assert (
        file_hash == "f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad"
    )
    assert (
        rows_hash == "316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198"
    )
    assert [row["source_id"] for row in rows] == sorted(BLOCKING_SOURCE_IDS)
    assert {row["disposition"] for row in rows} == {"detach"}
    assert {tuple(row["live_target_ids"]) for row in rows} == {
        ("frequency_of_diagnostic_antenna",),
        ("toroidal_angle_of_measurement_position",),
    }


def test_tampered_signed_rows_are_rejected(tmp_path: Path) -> None:
    authority = json.loads(AUTHORITY_PATH.read_text())
    authority["rows"][0]["scalar_target"] = "tampered"
    path = _write_authority(tmp_path, authority)

    with pytest.raises(ValueError, match="signature does not match"):
        _load_signed_stale_source_rows(path, [authority["rows"][0]["source_id"]])


def test_signed_non_detach_row_is_not_execution_authority(tmp_path: Path) -> None:
    authority = json.loads(AUTHORITY_PATH.read_text())
    selected = next(
        row for row in authority["rows"] if row["source_id"] == BLOCKING_SOURCE_IDS[0]
    )
    selected["disposition"] = "versioned_migration"
    _resign_rows(authority)
    path = _write_authority(tmp_path, authority)

    with pytest.raises(ValueError, match="lacks exact detach authority"):
        _load_signed_stale_source_rows(path, [BLOCKING_SOURCE_IDS[0]])


def test_source_outside_signed_authority_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside signed authority"):
        _load_signed_stale_source_rows(AUTHORITY_PATH, ["dd:not/signed"])


def test_apply_requires_a_preview_manifest_hash() -> None:
    gc = MagicMock()

    with pytest.raises(ValueError, match="requires manifest_sha256"):
        detach_signed_stale_source_bindings(
            gc,
            AUTHORITY_PATH,
            BLOCKING_SOURCE_IDS,
            reason="Detach sources removed from current DD authority.",
            apply=True,
        )

    gc.session.assert_not_called()


@pytest.mark.graph
def test_complete_signed_admission_cohort_applies_and_replays_without_writes(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j)
    _clear_graph(client)
    _seed_signed_live_rows(client)
    admitted_source_ids = _admitted_source_ids()
    assert len(_authority_rows()) == 58
    assert len(ALREADY_DETACHED_SOURCE_IDS) == 3
    assert len(LAST_PRODUCER_REFUSALS) == 3
    assert len(admitted_source_ids) == 52
    reason = "detach every topology-admitted source absent from current authority"
    try:
        preview = detach_signed_stale_source_bindings(
            client,
            AUTHORITY_PATH,
            admitted_source_ids,
            reason=reason,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["would_change"] == 52
        assert preview["receipt_rows"] == 52
        assert preview["bindings_to_remove"] == 53
        assert preview["projections_to_remove"] == 42

        applied = detach_signed_stale_source_bindings(
            client,
            AUTHORITY_PATH,
            admitted_source_ids,
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
        )
        assert applied["outcome"] == "applied"
        assert applied["changed"] == 52
        assert applied["receipt_rows"] == 52
        assert applied["bindings_removed"] == 53
        assert applied["projections_removed"] == 42
        assert applied["StandardNameChange"]["delta"] == 52
        assert applied["LLMCost"]["delta"] == 0

        covered = client.query(
            """
            UNWIND $source_ids AS source_id
            MATCH (source:StandardNameSource {id: source_id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode)
            OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
            RETURN source_id, source.source_type AS source_type,
                   source.produced_sn_id AS scalar,
                   collect(DISTINCT target.id) AS bindings,
                   collect(DISTINCT projected.id) AS projections
            ORDER BY source_id
            """,
            source_ids=[
                DERIVED_SOURCE_ID,
                MULTI_BINDING_SOURCE_ID,
                SCALAR_MISMATCH_SOURCE_ID,
            ],
        )
        assert covered == [
            {
                "source_id": SCALAR_MISMATCH_SOURCE_ID,
                "source_type": "dd",
                "scalar": None,
                "bindings": [],
                "projections": [],
            },
            {
                "source_id": MULTI_BINDING_SOURCE_ID,
                "source_type": "dd",
                "scalar": None,
                "bindings": [],
                "projections": [],
            },
            {
                "source_id": DERIVED_SOURCE_ID,
                "source_type": "derived",
                "scalar": None,
                "bindings": [],
                "projections": [],
            },
        ]

        before_replay = client.query(
            "MATCH (node) RETURN count(node) AS nodes, "
            "COUNT { (:StandardNameChange) } AS changes, "
            "COUNT { ()-[relationship]->() } AS relationships"
        )
        replay = detach_signed_stale_source_bindings(
            client,
            AUTHORITY_PATH,
            admitted_source_ids,
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
        )
        after_replay = client.query(
            "MATCH (node) RETURN count(node) AS nodes, "
            "COUNT { (:StandardNameChange) } AS changes, "
            "COUNT { ()-[relationship]->() } AS relationships"
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert after_replay == before_replay
    finally:
        _clear_graph(client)


@pytest.mark.graph
@pytest.mark.parametrize(
    ("source_id", "bindings", "projections"),
    [
        pytest.param(DERIVED_SOURCE_ID, 1, 0, id="derived-source"),
        pytest.param(MULTI_BINDING_SOURCE_ID, 2, 2, id="multi-binding-dd"),
        pytest.param(SCALAR_MISMATCH_SOURCE_ID, 1, 1, id="scalar-mismatch"),
    ],
)
def test_each_widened_signed_shape_releases_its_complete_source_closure(
    disposable_neo4j: tuple[str, str],
    source_id: str,
    bindings: int,
    projections: int,
) -> None:
    client = _client(disposable_neo4j)
    _clear_graph(client)
    _seed_signed_live_rows(client)
    reason = "release one widened stale-source shape under its signed authority"
    try:
        preview = detach_signed_stale_source_bindings(
            client, AUTHORITY_PATH, [source_id], reason=reason
        )
        assert preview["outcome"] == "would_apply"
        assert preview["bindings_to_remove"] == bindings
        assert preview["projections_to_remove"] == projections
        applied = detach_signed_stale_source_bindings(
            client,
            AUTHORITY_PATH,
            [source_id],
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
        )
        assert applied["outcome"] == "applied"
        assert applied["changed"] == 1
        assert applied["bindings_removed"] == bindings
        assert applied["projections_removed"] == projections
    finally:
        _clear_graph(client)


@pytest.mark.graph
@pytest.mark.parametrize(
    ("source_id", "target_id"),
    sorted(LAST_PRODUCER_REFUSALS.items()),
)
def test_signed_last_producer_rows_are_refused_by_target_name(
    disposable_neo4j: tuple[str, str], source_id: str, target_id: str
) -> None:
    client = _client(disposable_neo4j)
    _clear_graph(client)
    _seed_signed_live_rows(client)
    try:
        with pytest.raises(
            StaleSourceDetachConflict,
            match=f"detach would orphan target {target_id}",
        ):
            detach_signed_stale_source_bindings(
                client,
                AUTHORITY_PATH,
                [source_id],
                reason="preserve every target's final producing authority",
            )
        assert client.query(
            "MATCH (:StandardNameChange) RETURN count(*) AS changes"
        ) == [{"changes": 0}]
    finally:
        _clear_graph(client)


@pytest.mark.graph
def test_unsigned_live_binding_drift_refuses_before_mutation(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j)
    _clear_graph(client)
    _seed_signed_live_rows(client)
    try:
        client.query(
            "MATCH (source:StandardNameSource {id: $source_id}) "
            "CREATE (unsigned:StandardName {id: 'unsigned_stale_target', "
            "name_stage: 'accepted', status: 'draft'}) "
            "CREATE (source)-[:PRODUCED_NAME]->(unsigned)",
            source_id=DERIVED_SOURCE_ID,
        )
        with pytest.raises(
            StaleSourceDetachConflict,
            match=f"signed source closure changed for {DERIVED_SOURCE_ID}",
        ):
            detach_signed_stale_source_bindings(
                client,
                AUTHORITY_PATH,
                [DERIVED_SOURCE_ID],
                reason="refuse binding identities outside the signed row",
            )
        assert client.query(
            "MATCH (:StandardNameChange) RETURN count(*) AS changes"
        ) == [{"changes": 0}]
    finally:
        _clear_graph(client)
