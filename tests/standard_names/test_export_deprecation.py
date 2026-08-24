"""Released catalog entries exclude internal supersession lineage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.export import (  # noqa: E402
    _graph_node_to_entry_dict,
    run_export,
)

_GC_PATH = "imas_codex.graph.client.GraphClient"


def _candidate(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "_has_docs_review": True,
        "_has_winning_docs_review": True,
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


class _ReadOnlyGraphClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        self.queries.append(cypher)
        return []


def _load_domain_yaml(staging: Path, domain: str) -> list[dict]:
    text = (staging / "standard_names" / f"{domain}.yml").read_text()
    return yaml.safe_load(text)


class TestExportOmitsSupersessionLineage:
    @pytest.fixture()
    def exported(self, tmp_path: Path):
        live = _candidate(
            "core_ion_temperature",
            deprecates="ion_temperature_core",
            superseded_by="future_ion_temperature",
        )
        excluded = _candidate("documentation_pending", docs_stage="pending")
        graph = _ReadOnlyGraphClient()

        with (
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=[live, excluded],
            ),
            patch(_GC_PATH, return_value=graph),
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=lambda entry: entry,
            ),
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            ),
            patch(
                "imas_codex.standard_names.export._get_codex_commit_sha",
                return_value="a" * 40,
            ),
            patch(
                "imas_codex.standard_names.export._manifest_iso_timestamp",
                return_value="2026-08-24T00:00:00Z",
            ),
            patch(
                "imas_codex.standard_names.export.pick_primary_domain",
                side_effect=lambda domains: sorted(domains)[0],
            ),
        ):
            report = run_export(
                tmp_path,
                skip_gate=True,
                force=True,
                include_sources=False,
            )

        entries = _load_domain_yaml(tmp_path, "general")
        return report, entries, graph.queries, tmp_path

    def test_released_entries_contain_no_supersession_lineage(self, exported):
        _, entries, queries, _ = exported

        assert len(entries) == 1
        assert sum(entry.get("status") == "deprecated" for entry in entries) == 0
        assert sum("superseded_by" in entry for entry in entries) == 0
        assert sum("deprecates" in entry for entry in entries) == 0
        assert all("catalog_approved_at IS NOT NULL" not in query for query in queries)

    def test_export_accounting_still_closes(self, exported):
        report, _, _, _ = exported
        excluded_count = sum(
            row["count"] for row in report.to_dict()["exclusion_ledger"]
        )

        assert report.exported_count == 1
        assert excluded_count == 1
        assert report.exported_count + excluded_count == report.total_candidates == 2
        assert report.all_gates_passed

    def test_report_and_manifest_have_no_stub_counter(self, exported):
        report, _, _, staging = exported
        report_json = json.loads((staging / ".export_report.json").read_text())
        manifest = yaml.safe_load((staging / "catalog.yml").read_text())

        assert "deprecated_stub_count" not in vars(report)
        assert "deprecated_stubs" not in report_json["counts"]
        assert "deprecated_stub_count" not in manifest
        assert "deprecated_stubs" not in manifest


def test_active_entry_projection_drops_graph_lifecycle_fields() -> None:
    entry = _graph_node_to_entry_dict(
        _candidate(
            "core_ion_temperature",
            deprecates="ion_temperature_core",
            superseded_by="future_ion_temperature",
        )
    )

    assert entry["status"] == "active"
    assert "deprecates" not in entry
    assert "superseded_by" not in entry
