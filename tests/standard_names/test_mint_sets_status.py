"""Live catalog-status invariants for the batch standard-name mint path."""

from uuid import uuid4

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    reconcile_lifecycleless_standard_name_stubs,
)
from imas_codex.standard_names.graph_store.graph_catalog import write_standard_names

_CANDIDATE_IDENTITIES = (
    "ratio_of_electron_temperature_to_ion_temperature",
    "maximum_of_ratio_of_electron_temperature_to_ion_temperature",
    "minimum_of_ratio_of_electron_temperature_to_ion_temperature",
    "time_averaged_ratio_of_electron_temperature_to_ion_temperature",
    "maximum_of_electron_temperature",
)


def _fresh_identity(graph: GraphClient) -> str:
    existing = {
        row["id"]
        for row in graph.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids RETURN sn.id AS id",
            ids=list(_CANDIDATE_IDENTITIES),
        )
    }
    available = [name for name in _CANDIDATE_IDENTITIES if name not in existing]
    assert available, "no reserved standard-name identity is available for minting"
    return available[uuid4().int % len(available)]


@pytest.mark.graph
def test_batch_mint_sets_draft_and_remint_preserves_active_status() -> None:
    with GraphClient() as graph:
        identity = _fresh_identity(graph)
        try:
            assert write_standard_names([{"id": identity}], gc=graph) == 1

            created = graph.query(
                "MATCH (sn:StandardName {id: $id}) "
                "RETURN sn.status AS status, sn.name_stage AS name_stage, "
                "sn.origin AS origin",
                id=identity,
            )
            assert created == [{"status": "draft", "name_stage": None, "origin": None}]

            preview = reconcile_lifecycleless_standard_name_stubs(gc=graph)
            preview_ids = {
                row["id"]
                for rows in preview["manifest"]["rows"].values()
                for row in rows
            }
            assert identity in preview_ids

            graph.query(
                "MATCH (sn:StandardName {id: $id}) SET sn.status = 'active'",
                id=identity,
            )
            assert write_standard_names([{"id": identity}], gc=graph) == 1

            reminted = graph.query(
                "MATCH (sn:StandardName {id: $id}) RETURN sn.status AS status",
                id=identity,
            )
            assert reminted == [{"status": "active"}]
        finally:
            graph.query(
                "MATCH (sn:StandardName {id: $id}) DETACH DELETE sn",
                id=identity,
            )
