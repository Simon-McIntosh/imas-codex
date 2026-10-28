"""Null-status semantics of the signed manifest's source-status predicates.

In Cypher, ``x.status <> 'stale'`` is null, not true, when ``x.status`` is
null, so a status-less row drops out of the filter entirely. Every predicate in
``signed_manifest.py`` that asks whether a source is *live* must therefore
coalesce: a source with no status is unclassified, not retired, and the module
already counts it as a live producer everywhere it evaluates that in Python.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    apply_adjudicated_source_dispositions,
    reconcile_lifecycleless_standard_name_stubs,
)
from imas_codex.standard_names.graph_store import graph_lineage
from imas_codex.standard_names.graph_store.graph_lineage import (
    LifecyclelessStubConflict,
)
from imas_codex.standard_names.signed_manifest import (
    _apply_ordinary_source_migration,
    _Authority,
    _LoadedRow,
    _out_of_allowlist_source_hash,
    _verify_postconditions,
)
from tests.standard_names.test_signed_source_dispositions import (
    _adjudication,
    _cleanup,
    _row,
    _seed,
)


class _ClientQuery:
    """The module's ``_Query`` protocol over a real graph client."""

    def __init__(self, client: GraphClient) -> None:
        self._client = client

    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        return list(self._client.query(cypher, **params))


def _client(endpoint: tuple[str, str], name: str) -> GraphClient:
    uri, password = endpoint
    return GraphClient(uri=uri, username="neo4j", password=password, graph_name=name)


def _clear(client: GraphClient, ids: list[str]) -> None:
    client.query(
        "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
        ids=ids,
    )


@pytest.mark.graph
def test_stale_detach_postcondition_counts_a_status_less_producer(
    disposable_neo4j: tuple[str, str],
) -> None:
    target_id = "null_status_target"
    detached_id = "dd:null_status/detached"
    producer_id = "signals:null_status/remaining"
    client = _client(disposable_neo4j, "null-status-detach-postcondition")
    query = _ClientQuery(client)
    try:
        client.query(
            "CREATE (target:StandardName {id: $target_id}) "
            "CREATE (detached:StandardNameSource {id: $detached_id, "
            "source_type: 'dd', source_id: $detached_id, status: 'stale'}) "
            "CREATE (producer:StandardNameSource {id: $producer_id, "
            "source_type: 'signals', source_id: $producer_id}) "
            "CREATE (producer)-[:PRODUCED_NAME]->(target)",
            target_id=target_id,
            detached_id=detached_id,
            producer_id=producer_id,
        )
        # The producer carries no status property at all: the exact shape the
        # null-unsafe filter used to discard.
        assert client.query(
            "MATCH (source:StandardNameSource {id: $id}) "
            "RETURN source.status AS status",
            id=producer_id,
        ) == [{"status": None}]
        signed_rows = [
            {"source_id": detached_id, "live_target_ids": [], "scalar_target": None}
        ]
        authority = _Authority(
            data={
                "adapter": "stale-source-lifecycle",
                "signed_rows": signed_rows,
                "stale_actions": [{"target_ids": [target_id]}],
                "out_of_allowlist": dict(
                    zip(
                        ("count", "sha256"),
                        _out_of_allowlist_source_hash(query, [detached_id]),
                        strict=True,
                    )
                ),
            },
            operation_id="null-status-postcondition",
            rows=(),
            receipt_policy={},
            file_sha256="0" * 64,
            payload_sha256="0" * 64,
        )

        _verify_postconditions(
            query,
            authority,
            [detached_id],
            "0" * 64,
            actions=[],
        )

        assert authority.data["target_post"] == [
            {"target_id": target_id, "live_producers": 1, "live_children": 0}
        ]
    finally:
        _clear(client, [target_id, detached_id, producer_id])


@pytest.mark.graph
def test_ordinary_source_migration_admits_a_status_less_source(
    disposable_neo4j: tuple[str, str],
) -> None:
    source_id = "dd:null_status/migration"
    path = "null_status/migration"
    old_id = "null_status_migration_old"
    new_id = "null_status_migration_new"
    client = _client(disposable_neo4j, "null-status-source-migration")
    query = _ClientQuery(client)
    try:
        client.query(
            "CREATE (source:StandardNameSource {id: $source_id, "
            "source_type: 'dd', source_id: $path, produced_sn_id: $old_id}) "
            "CREATE (old:StandardName {id: $old_id}) "
            "CREATE (new:StandardName {id: $new_id}) "
            "CREATE (backing:IMASNode {id: $path}) "
            "CREATE (source)-[:PRODUCED_NAME]->(old) "
            "CREATE (source)-[:FROM_DD_PATH]->(backing) "
            "CREATE (backing)-[:HAS_STANDARD_NAME]->(old)",
            source_id=source_id,
            path=path,
            old_id=old_id,
            new_id=new_id,
        )
        element_ids = client.query(
            "MATCH (source:StandardNameSource {id: $source_id}) "
            "MATCH (old:StandardName {id: $old_id}) "
            "MATCH (new:StandardName {id: $new_id}) "
            "MATCH (source)-[binding:PRODUCED_NAME]->(old) "
            "MATCH (source)-[origin:FROM_DD_PATH]->(backing:IMASNode) "
            "MATCH (backing)-[projection:HAS_STANDARD_NAME]->(old) "
            "RETURN elementId(source) AS source, elementId(old) AS old, "
            "elementId(new) AS new, elementId(binding) AS binding, "
            "elementId(origin) AS origin, elementId(backing) AS backing, "
            "elementId(projection) AS projection",
            source_id=source_id,
            old_id=old_id,
            new_id=new_id,
        )[0]
        assert client.query(
            "MATCH (source:StandardNameSource {id: $id}) "
            "RETURN source.status AS status",
            id=source_id,
        ) == [{"status": None}]
        row = _LoadedRow(
            id="null-status-migration",
            identity={"source_id": source_id, "target_id": new_id},
            participants=(),
            mutations=(
                {"participant_id": source_id},
                {"participant_id": old_id},
                {"participant_id": new_id},
            ),
            guards=(),
            orphan_policy="",
        )
        action = {
            "row": row,
            "participant_snapshots": {
                source_id: {"element_id": element_ids["source"]},
                new_id: {"element_id": element_ids["new"]},
                "binding": {
                    "relationship_type": "PRODUCED_NAME",
                    "element_id": element_ids["binding"],
                    "end_element_id": element_ids["old"],
                },
            },
            "ordinary_source_closure": {
                "backings": [
                    {
                        "origin_id": element_ids["origin"],
                        "backing_element_id": element_ids["backing"],
                        "projection_id": element_ids["projection"],
                    }
                ],
                "projections": [{"projection_id": element_ids["projection"]}],
            },
        }

        assert _apply_ordinary_source_migration(query, action) == len(row.mutations)

        assert client.query(
            "MATCH (source:StandardNameSource {id: $source_id})"
            "-[:PRODUCED_NAME]->(target:StandardName) "
            "RETURN source.produced_sn_id AS scalar, collect(target.id) AS targets",
            source_id=source_id,
        ) == [{"scalar": new_id, "targets": [new_id]}]
    finally:
        _clear(client, [source_id, old_id, new_id, path])


@pytest.mark.graph
def test_catalog_source_disposition_keeps_status_less_authority(
    disposable_neo4j: tuple[str, str],
) -> None:
    prefix = "nullstatussource"
    client = _client(disposable_neo4j, "null-status-source-disposition")
    ids = _seed(client, prefix, scalar=None, preserve_removed_target=True)
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    preview = None
    try:
        # The surviving producer of the removed target carries no status.
        client.query(
            "MATCH (source:StandardNameSource {id: $id}) REMOVE source.status",
            id=ids["catalog_anchor"],
        )
        assert client.query(
            "MATCH (source:StandardNameSource {id: $id}) "
            "RETURN source.status AS status",
            id=ids["catalog_anchor"],
        ) == [{"status": None}]
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="a status-less producer still holds a removed target's authority",
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"]["admitted"] == 1

        applied = apply_adjudicated_source_dispositions(
            adjudication,
            reason="a status-less producer still holds a removed target's authority",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )

        assert applied["outcome"] == "applied"
        assert applied["sources_reconciled"] == 1
        assert client.query(
            "MATCH (anchor:StandardNameSource {id: $anchor})"
            "-[:PRODUCED_NAME]->(target:StandardName {id: $catalog}) "
            "RETURN anchor.status AS status, target.id AS target_id",
            anchor=ids["catalog_anchor"],
            catalog=ids["catalog"],
        ) == [{"status": None, "target_id": ids["catalog"]}]
    finally:
        _cleanup(client, [ids], (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_lifecycleless_stub_cohort_change_refuses_before_deletion(
    disposable_neo4j: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_id = "dead_link_endpoint_without_lifecycle"
    client = _client(disposable_neo4j, "lifecycleless-stub-cohort-change")
    original = graph_lineage._materialize_derived_parent_rows
    try:
        client.query("CREATE (:StandardName {id: $id})", id=stub_id)
        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        assert preview["counts"]["delete-as-dead-link-stub"] == 1

        def _mutate_cohort_then_materialize(
            gc: object, parents: object, **kwargs: object
        ) -> int:
            _ClientQuery(gc).query(
                "MATCH (stub:StandardName {id: $id}) SET stub.origin = 'manual'",
                id=stub_id,
            )
            return original(gc, parents, **kwargs)

        monkeypatch.setattr(
            graph_lineage,
            "_materialize_derived_parent_rows",
            _mutate_cohort_then_materialize,
        )

        with pytest.raises(
            LifecyclelessStubConflict,
            match="stub placeholder cohort changed before deletion",
        ):
            reconcile_lifecycleless_standard_name_stubs(
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )

        # The transaction rolled back: the stub is still the placeholder it was
        # admitted over, so nothing was half-deleted.
        assert client.query(
            "MATCH (stub:StandardName {id: $id}) "
            "RETURN stub.origin AS origin, stub.name_stage AS name_stage",
            id=stub_id,
        ) == [{"origin": None, "name_stage": None}]
    finally:
        _clear(client, [stub_id])
