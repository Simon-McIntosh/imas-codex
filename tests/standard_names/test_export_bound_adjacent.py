"""Bound-adjacent score exclusions remain distinct and conserved."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from imas_codex.standard_names.export import (
    DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
    _run_gate_c,
    run_export,
)


class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str, score: float) -> dict:
    return {
        "id": name,
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": score,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
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
            min_score=0.85,
            skip_gate=True,
            force=True,
            include_sources=False,
        )


def test_bound_adjacent_is_distinct_and_ledger_closes(tmp_path: Path) -> None:
    population = [
        _candidate(
            "within_measured_swing",
            0.85 - DEFAULT_BOUND_ADJACENT_HALF_WIDTH + 0.001,
        ),
        _candidate("genuinely_below_bound", 0.70),
        _candidate("clearly_above_bound", 0.90),
    ]

    report = _run_fixture_export(tmp_path, population)
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == 3
    assert report.exported_names == ["clearly_above_bound"]
    assert rows["bound_adjacent"]["identities"] == ["within_measured_swing"]
    assert rows["below_name_score"]["identities"] == ["genuinely_below_bound"]
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 3


def test_bound_adjacent_half_width_is_configurable() -> None:
    candidate = _candidate("outside_narrower_measurement", 0.83)

    gate, filtered, below, unreviewed = _run_gate_c(
        [candidate],
        min_score=0.85,
        include_unreviewed=False,
        min_description_score=None,
        bound_adjacent_half_width=0.01,
    )

    assert not filtered
    assert below == 1
    assert unreviewed == 0
    assert [issue["type"] for issue in gate.issues] == ["below_name_score"]
