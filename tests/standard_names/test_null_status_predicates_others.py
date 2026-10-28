"""What a null status means at each status guard outside the signed manifest.

In Cypher ``x.status <> 'v'`` is null, not true, when ``x.status`` is null, so
a row carrying no status drops out of the comparison. Each guard below is
pinned by a row that carries no status at all, so the case is decided rather
than left to the null comparison.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.code import workers
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import attachment_audit

pytestmark = pytest.mark.graph

_PATH = "equilibrium/time_slice/profiles_1d/psi"
_SOURCE_ID = f"dd:{_PATH}"


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="null-status-predicates",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def test_migration_moves_a_status_less_source_with_its_cohort(
    client: GraphClient,
) -> None:
    """A source with no status is not a stale tombstone; it migrates."""
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )

    client.query(
        """
        CREATE (old:StandardName {id: 'null_status_old', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', source_paths: []})
        CREATE (new:StandardName {id: 'null_status_new', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', source_paths: []})
        CREATE (dd:IMASNode {id: $path})
        CREATE (source:StandardNameSource {id: $source_id, source_type: 'dd',
          source_id: $path, produced_sn_id: 'null_status_old'})
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        CREATE (source)-[:PRODUCED_NAME]->(old)
        """,
        path=_PATH,
        source_id=_SOURCE_ID,
    )

    moved = retarget_standard_name_sources(
        client,
        "null_status_old",
        "null_status_new",
        source_ids=[_SOURCE_ID],
        expected_current_bindings={_SOURCE_ID: "null_status_old"},
        record_change=False,
        enforce_consistency=False,
    )

    assert moved == 1
    bound = client.query(
        "MATCH (:StandardNameSource {id: $id})-[:PRODUCED_NAME]->(n:StandardName) "
        "RETURN collect(n.id) AS ids",
        id=_SOURCE_ID,
    )
    assert bound[0]["ids"] == ["null_status_new"]


def test_reset_refuses_a_source_whose_status_the_manifest_cannot_name(
    client: GraphClient,
) -> None:
    """The manifest declares the status it expects, so null is excluded loudly."""
    from imas_codex.standard_names.provenance_lifecycle import (
        reset_standard_name_sources,
    )

    client.query(
        """
        CREATE (name:StandardName {id: 'null_status_held', name_stage: 'drafted',
          validation_status: 'valid', status: 'draft', source_paths: []})
        CREATE (source:StandardNameSource {id: $source_id, source_type: 'dd',
          source_id: $path, produced_sn_id: 'null_status_held'})
        CREATE (source)-[:PRODUCED_NAME]->(name)
        """,
        path=_PATH,
        source_id=_SOURCE_ID,
    )

    with pytest.raises(RuntimeError, match="compare-and-set failed"):
        reset_standard_name_sources(
            client,
            [
                {
                    "source_id": _SOURCE_ID,
                    "expected_status": "attached",
                    "expected_scalar": "null_status_held",
                    "expected_bindings": ["null_status_held"],
                }
            ],
            manifest_id="null-status-reset",
            reason="the manifest names a status the source does not carry",
        )

    kept = client.query(
        "MATCH (s:StandardNameSource {id: $id}) RETURN s.status AS status",
        id=_SOURCE_ID,
    )
    assert kept[0]["status"] is None


def test_structural_candidate_reads_a_status_less_source_as_not_stale(
    client: GraphClient,
) -> None:
    """A structural source with no status is no tombstone, so it is reseedable."""
    from imas_codex.standard_names.graph_store import graph_parent_lifecycle

    parent = "null_status_structural_parent"
    client.query(
        """
        CREATE (parent:StandardName {id: $parent, origin: 'pipeline',
          status: 'draft'})
        CREATE (child:StandardName {id: 'null_status_structural_child',
          name_stage: 'accepted', status: 'draft'})
        CREATE (child)-[:HAS_PARENT]->(parent)
        CREATE (child_source:StandardNameSource {id: 'dd:child/path',
          source_type: 'dd', source_id: 'child/path', status: 'attached'})
        CREATE (child_source)-[:PRODUCED_NAME]->(child)
        CREATE (structural:StandardNameSource {id: 'derived:' + $parent,
          source_type: 'derived', source_id: $parent})
        """,
        parent=parent,
    )

    candidates = graph_parent_lifecycle.find_orphan_parent_source_candidates(
        client, services=MagicMock()
    )

    assert parent in [row["parent_id"] for row in candidates]


def test_seeding_rebinds_a_status_less_structural_source(
    client: GraphClient,
) -> None:
    """Seeding is withheld only by a proven stale tombstone."""
    from imas_codex.standard_names.graph_store import graph_parent_lifecycle

    parent = "null_status_seed_parent"
    client.query(
        """
        CREATE (parent:StandardName {id: $parent, origin: 'pipeline',
          status: 'draft'})
        CREATE (existing:StandardNameSource {id: 'derived:' + $parent,
          source_type: 'derived', source_id: $parent})
        """,
        parent=parent,
    )
    services = MagicMock()
    services._derived_parent_source_metadata.return_value = {
        "source_node_id": f"derived:{parent}",
        "source_type": "derived",
        "source_id": parent,
        "batch_key": "null-status-seed",
    }

    seeded = graph_parent_lifecycle.reconcile_orphan_parent_sources(
        client,
        classification={"repairable": [{"parent_id": parent}], "rejected_derived": []},
        services=services,
    )

    assert seeded == 1
    rows = client.query(
        "MATCH (:StandardNameSource {id: $id})-[:PRODUCED_NAME]->"
        "(n:StandardName) RETURN collect(n.id) AS ids",
        id=f"derived:{parent}",
    )
    assert rows[0]["ids"] == [parent]


def test_compose_hint_refuses_a_source_with_no_status(client: GraphClient) -> None:
    """Only a source proven to sit at extracted may take a compose steer."""
    from imas_codex.standard_names.graph_store import graph_source_bindings

    client.query(
        """
        CREATE (source:StandardNameSource {id: $source_id, source_type: 'dd',
          source_id: $path, attempt_count: 0})
        """,
        path=_PATH,
        source_id=_SOURCE_ID,
    )
    services = MagicMock()
    services._MAX_COMPOSE_CLAIM_ATTEMPTS = 3

    decision = graph_source_bindings._source_compose_hint_decision(
        client, services=services, source_id=_SOURCE_ID, replace=False
    )

    assert decision["decision"] == "source_not_extracted"


def test_attachment_audit_counts_a_status_less_producer_as_live(
    client: GraphClient,
) -> None:
    """A status-less producer keeps its name from reading as sourceless."""
    name = "null_status_audited_name"
    client.query(
        """
        CREATE (sn:StandardName {id: $name, name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', source_paths: []})
        CREATE (dd:IMASNode {id: $path})
        CREATE (src:StandardNameSource {id: $source_id, source_type: 'dd',
          source_id: $path})
        CREATE (src)-[:FROM_DD_PATH]->(dd)
        CREATE (src)-[:PRODUCED_NAME]->(sn)
        """,
        name=name,
        path=_PATH,
        source_id=_SOURCE_ID,
    )

    rows = client.query(
        attachment_audit._attachments_query(attachment_audit._ONE_NAME_SCOPE),
        historical=sorted(attachment_audit._HISTORICAL_NAME_STAGES),
        sn_id=name,
    )

    assert [row["live_producers"] for row in rows] == [[_SOURCE_ID]]


def test_dismissal_accepts_a_hold_with_no_status(client: GraphClient) -> None:
    """A hold with no status has not been dismissed, so it can be dismissed."""
    from imas_codex.standard_names.review import holds

    client.query(
        """
        CREATE (sn:StandardName {id: 'null_status_hold_name',
          name_stage: 'drafted', status: 'draft'})
        CREATE (hold:ReviewHold {id: 'null-status-hold', axis: 'name',
          kind: 'name_description'})
        CREATE (sn)-[:HAS_HOLD]->(hold)
        """
    )

    dismissed = holds.dismiss_hold(
        "null-status-hold", "triager", "the finding was withdrawn", graph=client
    )

    assert dismissed is True
    rows = client.query(
        "MATCH (h:ReviewHold {id: 'null-status-hold'}) RETURN h.status AS status"
    )
    assert rows[0]["status"] == "dismissed"


def test_ingested_mark_accepts_a_file_with_no_status(client: GraphClient) -> None:
    """Only a recorded failure withholds the ingested mark."""
    client.query("CREATE (f:CodeFile {id: 'null-status-file'})")

    updated = workers._mark_files_ingested(["null-status-file"])

    assert updated == 1
    rows = client.query(
        "MATCH (f:CodeFile {id: 'null-status-file'}) RETURN f.status AS status"
    )
    assert rows[0]["status"] == "ingested"