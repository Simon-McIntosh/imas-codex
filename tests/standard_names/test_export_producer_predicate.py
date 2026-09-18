"""A cut admits only names a source is reachable for.

The population query already projects producing-source topology onto every
candidate -- a derived producer, a non-derived producer, and a structural child
-- and the eligibility classifier decides what a cut carries. A name with no
PRODUCED_NAME edge of either kind and no live child is entailed by nothing, and
leaving it in the population lets a cut publish an identity that no evidence
produced. The exception is the source-free parent: a hierarchy node whose own
edge *is* the hierarchy, entailed by the live children that sit under it.

Reused export-surface symbols, cited per
``docs/evidence/unbound-source-backlog/export-surface-reuse-map.md``: the
identity universe query ``_fetch_export_population`` with its
``_has_derived_producer`` / ``_has_non_derived_producer`` / ``_is_parent``
projections (``export.py`` population query), the eligibility partition
``_classify_export_population`` (``export.py`` ten-clause classifier), the cut
entry point ``run_export``, and the canonical live-child stage set
(``pools.py`` ``has_live_child``).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

from imas_codex.standard_names.export import (
    _fetch_export_population,
    _has_producing_source,
    run_export,
)

# The stages a child is no longer live in; the same three the population query
# and graph_ops.structural_accept_derived_parents exclude.
_RETIRED_CHILD_STAGES = ("superseded", "exhausted", "contested")


class _CapturingGraphClient:
    """A read-only client that records the queries it was handed."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def __enter__(self) -> _CapturingGraphClient:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def query(self, cypher: str, **_params):
        self.queries.append(cypher)
        return []


def _candidate(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
    }
    candidate.update(overrides)
    return candidate


def _run_fixture_export(staging_dir: Path, population: list[dict]):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=population,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_CapturingGraphClient(),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=lambda entry: entry,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_deprecation_stubs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            )
        )
        return run_export(staging_dir, skip_gate=True, force=True)


def _population() -> list[dict]:
    return [
        # No producer of either kind reaches this name and nothing sits under
        # it, so no evidence entails it.
        _candidate(
            "unentailed_leaf",
            _has_dd_source_binding=False,
            _has_derived_producer=False,
            _has_non_derived_producer=False,
            _has_live_child=False,
            _is_parent=False,
        ),
        # A Data Dictionary extraction reached this one directly.
        _candidate(
            "extracted_name",
            _has_dd_source_binding=True,
            _has_derived_producer=False,
            _has_non_derived_producer=True,
            _has_live_child=False,
            _is_parent=False,
        ),
        # Source-free parent: no producer edge of its own, but a live child
        # carries the evidence for it.
        _candidate(
            "source_free_parent",
            _has_dd_source_binding=False,
            _has_derived_producer=False,
            _has_non_derived_producer=False,
            _has_live_child=True,
            _is_parent=True,
        ),
    ]


def test_export_refuses_name_with_no_producing_source(tmp_path: Path) -> None:
    """The cut drops the unentailed leaf and keeps both entailed names."""
    report = _run_fixture_export(tmp_path, _population())
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}
    emitted = yaml.safe_load(
        (tmp_path / "standard_names" / "general.yml").read_text(encoding="utf-8")
    )
    sidecar = yaml.safe_load((tmp_path / "catalog.yml").read_text(encoding="utf-8"))

    assert report.all_gates_passed
    assert report.total_candidates == 3
    assert report.exported_names == ["extracted_name", "source_free_parent"]
    assert rows["no_producing_source"]["identities"] == ["unentailed_leaf"]
    records = {row.standard_name_id: row for row in report.exclusion_records}
    assert records["unentailed_leaf"].reason == "no_producing_source"
    assert records["unentailed_leaf"].stage == "eligibility"
    # The refusal is real at the artifact, not only in the ledger: the name is
    # in neither the domain entries nor the manifest sidecar.
    assert [entry["name"] for entry in emitted] == [
        "extracted_name",
        "source_free_parent",
    ]
    assert set(sidecar["names"]) == {"extracted_name", "source_free_parent"}
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 3

    attestation = json.loads(
        (tmp_path / ".export_report.json").read_text(encoding="utf-8")
    )
    assert "unentailed_leaf" not in attestation["emitted_identities"]


def test_refusal_disappears_when_the_predicate_is_reverted(tmp_path: Path) -> None:
    """The refusal above is attributable to the predicate, not to the fixture.

    Reverting the predicate to its permissive form -- every candidate entailed
    -- must let the unentailed leaf back into the cut. If it does not, the test
    above is passing for some other reason and proves nothing about the gate.
    """
    with patch(
        "imas_codex.standard_names.export._has_producing_source",
        return_value=True,
    ):
        report = _run_fixture_export(tmp_path, _population())

    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}
    assert "no_producing_source" not in rows
    assert "unentailed_leaf" in report.exported_names


def test_producer_predicate_reads_the_topology_flags() -> None:
    """The predicate is a read of the three flags the query already projects."""
    # A caller's own pre-filtered projection carries no flag and is left alone.
    assert _has_producing_source({"id": "projection_only"}) is True
    # Producers of either kind entail a name; so does a live child.
    assert (
        _has_producing_source(
            {"_has_derived_producer": True, "_has_non_derived_producer": False}
        )
        is True
    )
    assert (
        _has_producing_source(
            {
                "_has_live_child": True,
                "_has_derived_producer": False,
                "_has_non_derived_producer": False,
            }
        )
        is True
    )
    assert (
        _has_producing_source(
            {
                "_has_live_child": False,
                "_has_derived_producer": False,
                "_has_non_derived_producer": False,
            }
        )
        is False
    )


def test_retired_children_do_not_entail_a_parent(tmp_path: Path) -> None:
    """A parent whose only children are retired is not source-free."""
    population = [
        _candidate(
            "abandoned_parent",
            _has_dd_source_binding=False,
            _has_derived_producer=False,
            _has_non_derived_producer=False,
            _has_live_child=False,
            _is_parent=True,
        )
    ]

    report = _run_fixture_export(tmp_path, population)
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}

    assert rows["no_producing_source"]["identities"] == ["abandoned_parent"]
    assert report.exported_count == 0


def test_population_query_projects_live_child_evidence() -> None:
    """The flag the predicate reads is carried by the identity universe query.

    Driven through ``_fetch_export_population`` itself rather than by reading
    the source text, so the assertion is about the query the gate sends.
    """
    client = _CapturingGraphClient()
    with patch("imas_codex.graph.client.GraphClient", return_value=client):
        _fetch_export_population(require_docs_review=False)

    assert client.queries, "the population fetch sent no query"
    cypher = client.queries[0]
    assert "AS has_live_child" in cypher
    assert "_has_live_child: has_live_child" in cypher
    assert "MATCH (child:StandardName)-[:HAS_PARENT]->(sn)" in cypher
    for stage in _RETIRED_CHILD_STAGES:
        assert f"'{stage}'" in cypher
