"""Physics-domain resolution for export when the graph node stores none.

The export never fabricates a synthetic bucket: when the graph node carries
no physics_domain, the domain is derived from the identity's producing
sources, and a name with genuinely no resolvable domain is reported in the
exclusion ledger rather than silently emitted.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.export import (
    _graph_node_to_entry_dict,
    run_export,
)


def _candidate(name: str, **overrides) -> dict:
    """A candidate eligible at every stage except physics-domain resolution.

    Deliberately omits the physics_domain key, matching the graph projection
    for a name whose node stores none.
    """
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
        "unit": "m",
        "links": [],
    }
    candidate.update(overrides)
    return candidate


class _RecordingValidate:
    """Stand-in for _validate_entry that records what the export built."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def __call__(self, entry: dict):
        self.entries.append(entry)
        return entry


class _SourceDomainGraph:
    """Read-only graph double returning one source-domain row per name."""

    def __init__(self, source_domains: dict[str, list[str]] | None = None) -> None:
        self.source_domains = source_domains or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "RETURN DISTINCT src.physics_domain AS domain" in cypher:
            name = params.get("name")
            return [{"domain": domain} for domain in self.source_domains.get(name, [])]
        return []


def _run_fixture_export(tmp_path, population, graph, validate) -> dict:
    from contextlib import ExitStack
    from unittest.mock import patch

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
                return_value=graph,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=validate,
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
        stack.enter_context(
            patch("imas_codex.standard_names.export._write_domain_yaml")
        )
        return run_export(tmp_path, skip_gate=True, force=True)


def test_accepted_name_with_source_domain_never_resolves_to_the_bucket(
    tmp_path,
) -> None:
    """A name whose sources carry a domain emits with that real domain."""
    name = "etendue_of_soft_xray_detector"
    graph = _SourceDomainGraph({name: ["radiation_measurement_diagnostics"]})
    validate = _RecordingValidate()

    report = _run_fixture_export(
        tmp_path,
        [_candidate(name)],
        graph,
        validate,
    )

    assert report.exported_names == [name]
    assert validate.entries, "the entry was built and validated"
    emitted = validate.entries[0]
    assert emitted["physics_domain"] == "radiation_measurement_diagnostics"
    assert "unscoped" not in str(emitted.get("physics_domain"))
    assert not [row for row in report.exclusion_records if row.standard_name_id == name]


def test_name_with_no_resolvable_domain_is_reported_not_emitted(
    tmp_path,
) -> None:
    """A name with no stored or source domain is excluded, not silently minted."""
    name = "unscoped_candidate"
    graph = _SourceDomainGraph({})  # no source carries a domain
    validate = _RecordingValidate()

    report = _run_fixture_export(tmp_path, [_candidate(name)], graph, validate)

    assert report.exported_names == []
    assert not validate.entries, "no entry reached ISN validation"
    matching = [row for row in report.exclusion_records if row.standard_name_id == name]
    assert len(matching) == 1
    assert matching[0].reason == "missing_physics_domain"


def test_entry_converter_refuses_a_domainless_node() -> None:
    """The synthetic bucket is gone: conversion refuses rather than fabricates."""
    with pytest.raises(ValueError, match="physics domain"):
        _graph_node_to_entry_dict(
            {
                "id": "no_domain_name",
                "status": "draft",
            }
        )


def test_entry_converter_still_picks_primary_from_stored_domains() -> None:
    """Domain-bearing nodes keep the existing primary-domain selection."""
    entry = _graph_node_to_entry_dict(
        {
            "id": "electron_temperature",
            "status": "draft",
            "physics_domain": ["transport", "core_plasma_physics"],
        }
    )

    assert entry["physics_domain"] in {"transport", "core_plasma_physics"}
