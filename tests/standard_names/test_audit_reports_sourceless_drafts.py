"""A sourceless live draft is reported on the audit and does not stop a release.

A draft never reaches the export population, so its missing source is a fact
about work in progress rather than a reason to refuse the cut. A name in a
stage that ships is a different matter: that one would ship without a producer
and still blocks.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_invariant import find_sourceless_live_names

_DRAFT = "audit_unsourced_draft"
_SHIPPING = "audit_unsourced_shipping"
_SOURCED = "audit_sourced_live"


@pytest.fixture()
def planted_graph(disposable_neo4j_uri: str) -> Iterator[GraphClient]:
    """One sourceless live draft, nothing else."""
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="audit-drafts-fixture",
    )
    client.query(
        """CREATE (:StandardName {id:$draft, name_stage:'drafted',
          origin:'pipeline', source_paths:[], status:'draft'})""",
        draft=_DRAFT,
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'audit_' DETACH DELETE node"
        )
        client.close()


def _export_report(staging_dir: Path, graph: GraphClient):
    from imas_codex.standard_names.export import run_export
    from tests.standard_names.test_source_audit_gate import _candidate

    with (
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=nullcontext(graph),
        ),
        patch(
            "imas_codex.standard_names.export._load_waived_source_paths",
            return_value=set(),
        ),
        patch(
            "imas_codex.standard_names.export._fetch_export_population",
            return_value=[_candidate(_SOURCED)],
        ),
        patch(
            "imas_codex.standard_names.export._validate_entry",
            side_effect=lambda entry: entry,
        ),
        patch(
            "imas_codex.standard_names.export._fetch_deprecation_stubs",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
            return_value=([], set()),
        ),
        patch("imas_codex.standard_names.export._write_domain_yaml"),
    ):
        return run_export(staging_dir, skip_gate=True, force=True)


@pytest.mark.graph
def test_sourceless_draft_is_reported_and_does_not_block(
    planted_graph: GraphClient, tmp_path: Path
) -> None:
    # The audit's own notion of sourceless, asserted first so a report that
    # lists nothing cannot pass on an empty population.
    assert find_sourceless_live_names(planted_graph, [_DRAFT]) == [_DRAFT]

    report = _export_report(tmp_path / "draft", planted_graph)

    assert report.all_gates_passed
    assert "source_audit" not in {gate.gate for gate in report.gate_results}
    assert report.to_dict()["sourceless_drafts"] == {
        "count": 1,
        "names": [
            {
                "name": _DRAFT,
                "name_stage": "drafted",
                "population": "never_bound",
            }
        ],
    }


@pytest.mark.graph
def test_sourceless_shipping_name_still_blocks(
    planted_graph: GraphClient, tmp_path: Path
) -> None:
    planted_graph.query(
        "MATCH (name:StandardName {id:$draft}) SET name.name_stage = 'accepted'",
        draft=_DRAFT,
    )
    report = _export_report(tmp_path / "shipping", planted_graph)

    assert not report.all_gates_passed
    gate = next(gate for gate in report.gate_results if gate.gate == "source_audit")
    assert {issue["name"] for issue in gate.issues} == {_DRAFT}
    assert report.to_dict()["sourceless_drafts"]["count"] == 0
