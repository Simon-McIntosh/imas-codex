"""Identity writes preserve a producing source for every live name."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.source_invariant import (
    MissingStandardNameSource,
    _is_sourceless_name,
    require_producing_sources,
)


def test_source_invariant_classifies_touched_names() -> None:
    class Query:
        def __init__(self, *, origin=None, stage=None, sources=0, children=0):
            self.origin = origin
            self.stage = stage
            self.sources = sources
            self.children = children

        def query(self, cypher, **params):
            assert "PRODUCED_NAME" in cypher
            assert "HAS_PARENT" in cypher
            assert params["ids"] == ["electron_temperature"]
            missing = (
                (self.stage != "superseded" or not params["retired_is_exempt"])
                and self.sources == 0
                and not (self.origin == "derived" and self.children)
            )
            return [{"id": "electron_temperature"}] if missing else []

    require_producing_sources(
        Query(origin="derived", children=1), ["electron_temperature"]
    )
    require_producing_sources(Query(stage="superseded"), ["electron_temperature"])
    with pytest.raises(MissingStandardNameSource):
        require_producing_sources(
            Query(stage="superseded"),
            ["electron_temperature"],
            retired_is_exempt=False,
        )
    require_producing_sources(Query(sources=1), ["electron_temperature"])
    with pytest.raises(MissingStandardNameSource, match="electron_temperature"):
        require_producing_sources(Query(), ["electron_temperature"])


def test_invariant_query_keeps_names_with_missing_origin() -> None:
    class CaptureQuery:
        statement = ""

        def query(self, cypher, **params):
            self.statement = cypher
            assert params["ids"] == ["electron_temperature"]
            return []

    capture = CaptureQuery()
    require_producing_sources(capture, ["electron_temperature"])
    assert "coalesce(sn.origin, '') = 'derived'" in capture.statement
    assert "sn.origin = 'derived'" not in capture.statement
    assert _is_sourceless_name(None, has_live_child=False, producers=0)
    assert _is_sourceless_name(None, has_live_child=True, producers=0)
    assert _is_sourceless_name("derived", has_live_child=False, producers=0)
    assert not _is_sourceless_name("derived", has_live_child=True, producers=0)
    assert not _is_sourceless_name(None, has_live_child=False, producers=1)


@pytest.mark.parametrize("stage", ["drafted", "superseded"])
def test_rename_refuses_a_sourceless_predecessor(stage: str) -> None:
    from imas_codex.standard_names.edit import apply_edit
    from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph

    class SourcelessGraph(FakeGraph):
        def query(self, cypher, **params):
            if "SOURCE_INVARIANT" in cypher:
                assert params["retired_is_exempt"] is False
                return [{"id": "plasma_current"}]
            return super().query(cypher, **params)

    graph = SourcelessGraph()
    graph.add_node("plasma_current", name_stage=stage)
    with _patched_graph(graph), pytest.raises(MissingStandardNameSource):
        apply_edit(
            target="plasma_current",
            rename="toroidal_plasma_current",
            reason="clarify the component",
            gc=graph,
        )
    assert "toroidal_plasma_current" not in graph.nodes


def test_rename_transaction_rolls_back_an_unbound_successor() -> None:
    from imas_codex.standard_names.graph_ops import persist_refined_name
    from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph

    class SourcelessGraph(FakeGraph):
        def query(self, cypher, **params):
            if "SOURCE_INVARIANT" in cypher:
                assert params["ids"] == ["toroidal_plasma_current"]
                return [{"id": "toroidal_plasma_current"}]
            return super().query(cypher, **params)

    graph = SourcelessGraph()
    graph.add_node("plasma_current", name_stage="drafted")
    with _patched_graph(graph), pytest.raises(MissingStandardNameSource):
        persist_refined_name(
            old_name="plasma_current",
            new_name="toroidal_plasma_current",
            description="plasma current",
            model="sn-edit",
            reason="clarify component",
            edit_mode="rename",
            expected_old_stage="drafted",
        )
    assert graph.nodes["plasma_current"]["name_stage"] == "drafted"
    assert "toroidal_plasma_current" not in graph.nodes


def test_name_creation_refuses_an_unbound_identity() -> None:
    from imas_codex.standard_names import graph_ops

    graph = MagicMock()

    def query(cypher, **params):
        if "SOURCE_INVARIANT" in cypher:
            return [{"id": "electron_temperature"}]
        return []

    graph.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.protection._fetch_catalog_edit_names",
            return_value=set(),
        ),
        patch.object(graph_ops, "_write_grammar_decomposition", return_value=[]),
        patch.object(graph_ops, "_write_standard_name_edges", return_value=[]),
        pytest.raises(MissingStandardNameSource),
    ):
        graph_ops.write_standard_names([{"id": "electron_temperature"}], gc=graph)


def test_consistency_reconciliation_refuses_last_source_removal() -> None:
    from imas_codex.standard_names import attachment_audit as audit

    graph = MagicMock()

    def query(cypher, **params):
        if cypher == audit._DETACH_QUERY:
            return [{"detached": 1}]
        if "SOURCE_INVARIANT" in cypher:
            return [{"id": "electron_temperature"}]
        return []

    graph.query.side_effect = query
    verdict = audit.AttachmentVerdict(
        source_node_id="dd:core_profiles/profiles_1d/electrons/temperature",
        dd_path="core_profiles/profiles_1d/electrons/temperature",
        sn_id="electron_temperature",
        name_stage="drafted",
        reason="wrong pairing",
    )
    with (
        patch.object(
            audit,
            "audit_attachments",
            return_value=audit.AttachmentAuditResult(checked=1, rejected=[verdict]),
        ),
        pytest.raises(MissingStandardNameSource),
    ):
        audit.reconcile_attachment_consistency(graph)


def test_signed_detachment_refuses_nonderived_parent_without_source() -> None:
    from imas_codex.standard_names import signed_manifest as signed

    authority = SimpleNamespace(
        data={
            "adapter": signed._STALE_SOURCE_ADAPTER,
            "signed_rows": [
                {
                    "source_id": "dd:core_profiles/profiles_1d/electrons/temperature",
                    "live_target_ids": ["electron_temperature"],
                    "scalar_target": "electron_temperature",
                }
            ],
            "stale_actions": [{"target_ids": ["electron_temperature"]}],
            "out_of_allowlist": {"count": 0, "sha256": "empty"},
        }
    )
    graph = MagicMock()

    def query(cypher, **params):
        if "SOURCE_INVARIANT" in cypher:
            return [{"id": "electron_temperature"}]
        if "collect(DISTINCT projected.id) AS projections" in cypher:
            return [
                {
                    "source_id": authority.data["signed_rows"][0]["source_id"],
                    "scalar": None,
                    "bindings": [],
                    "projections": [],
                }
            ]
        if "live_producers" in cypher:
            return [
                {
                    "target_id": "electron_temperature",
                    "live_producers": 0,
                    "live_children": 1,
                }
            ]
        return []

    graph.query.side_effect = query
    with (
        patch.object(
            signed, "_out_of_allowlist_source_hash", return_value=(0, "empty")
        ),
        pytest.raises(MissingStandardNameSource),
    ):
        signed._verify_postconditions(graph, authority, [], "manifest")
