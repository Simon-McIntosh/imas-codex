"""Catalog export reads and enforces graph-owned lifecycle status."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from imas_codex.standard_names.export import GATE_CATALOG_STATUS, run_export


class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str, *, status: str | None) -> dict:
    return {
        "id": name,
        "name_stage": "accepted",
        "status": status,
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
        "unit": "eV",
        "physics_domain": "general",
        "links": [],
    }


def _run_fixture_export(staging_dir: Path, population: list[dict]):
    with (
        patch(
            "imas_codex.standard_names.export._fetch_export_population",
            return_value=population,
        ),
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=_ReadOnlyGraphClient(),
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
            return_value="2026-09-04T00:00:00Z",
        ),
        patch(
            "imas_codex.standard_names.export.pick_primary_domain",
            side_effect=lambda domains: sorted(domains)[0],
        ),
    ):
        return run_export(
            staging_dir,
            skip_gate=True,
            force=True,
            include_sources=False,
        )


def test_draft_graph_status_is_exported_as_draft(tmp_path: Path) -> None:
    report = _run_fixture_export(
        tmp_path,
        [_candidate("electron_temperature", status="draft")],
    )

    manifest = yaml.safe_load((tmp_path / "catalog.yml").read_text(encoding="utf-8"))

    assert report.all_gates_passed
    assert manifest["names"]["electron_temperature"]["status"] == "draft"


def test_null_graph_status_refuses_the_whole_cut(tmp_path: Path) -> None:
    report = _run_fixture_export(
        tmp_path,
        [
            _candidate("electron_temperature", status="draft"),
            _candidate("ion_temperature", status=None),
        ],
    )

    persisted = json.loads(
        (tmp_path / ".export_report.json").read_text(encoding="utf-8")
    )
    status_gate = next(
        gate for gate in persisted["gates"] if gate["gate"] == GATE_CATALOG_STATUS
    )

    assert not report.all_gates_passed
    assert status_gate["issues"] == [
        {
            "type": "null_catalog_status",
            "name": "ion_temperature",
            "detail": "graph catalog status is null",
        }
    ]
    assert not (tmp_path / "catalog.yml").exists()
    assert not (tmp_path / "standard_names").exists()
