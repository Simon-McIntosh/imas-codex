"""Identity-bearing exclusion accounting for catalog export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from imas_codex.standard_names.export import run_export


class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_resolution_method": "quorum_consensus",
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
        return run_export(
            staging_dir,
            skip_gate=True,
            force=True,
            include_sources=False,
        )


def test_export_ledger_closes_over_fixture_population(tmp_path: Path) -> None:
    population = [
        _candidate("emitted_name"),
        _candidate("invalid_name", validation_status="quarantined"),
        _candidate("docs_pending_name", docs_stage="pending"),
        _candidate("unreviewed_name", reviewer_score_name=None),
    ]

    report = _run_fixture_export(tmp_path, population)
    payload = report.to_dict()
    persisted_payload = json.loads(
        (tmp_path / ".export_report.json").read_text(encoding="utf-8")
    )
    rows = {row["reason"]: row for row in payload["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == 4
    assert report.exported_count == 1
    assert report.exported_names == ["emitted_name"]
    assert payload["emitted_identities"] == report.exported_names
    assert persisted_payload["emitted_identities"] == report.exported_names
    assert {reason: row["count"] for reason, row in rows.items()} == {
        "documentation_not_accepted": 1,
        "invalid_validation_status": 1,
        "unreviewed_name": 1,
    }
    assert rows["documentation_not_accepted"]["identities"] == ["docs_pending_name"]
    assert rows["invalid_validation_status"]["identities"] == ["invalid_name"]
    assert rows["unreviewed_name"]["identities"] == ["unreviewed_name"]
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 4


def test_export_refuses_when_ledger_does_not_close(tmp_path: Path) -> None:
    population = [_candidate("emitted_name"), _candidate("silently_dropped_name")]

    with patch(
        "imas_codex.standard_names.export._classify_export_population",
        return_value=([population[0]], []),
    ):
        report = _run_fixture_export(tmp_path, population)

    accounting_gate = next(
        gate for gate in report.gate_results if gate.gate == "exclusion_accounting"
    )
    assert not accounting_gate.passed
    assert not report.all_gates_passed
    assert any(
        issue["type"] == "unattributed_identity"
        and issue["identities"] == ["silently_dropped_name"]
        for issue in accounting_gate.issues
    )
    assert any(
        issue["type"] == "exclusion_accounting_mismatch"
        and issue["accepted_population"] == 2
        and issue["emitted"] == 1
        and issue["excluded"] == 0
        for issue in accounting_gate.issues
    )
    assert not (tmp_path / "catalog.yml").exists()
