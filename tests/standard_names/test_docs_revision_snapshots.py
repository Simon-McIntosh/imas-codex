"""Regression coverage for documentation reset snapshots."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import graph_ops

_COLLISION_SHAPES = (
    (
        "accumulated_total_particle_count_due_to_gas_injection",
        0,
        "accumulated_total_particle_count_due_to_gas_injection#rev-2",
        3,
    ),
    (
        "area_of_poloidal_magnetic_field_probe",
        0,
        "area_of_poloidal_magnetic_field_probe#rev-1",
        2,
    ),
    (
        "area_of_toroidal_magnetic_field_probe",
        0,
        "area_of_toroidal_magnetic_field_probe#rev-1",
        2,
    ),
    ("atomic_mass", 0, "atomic_mass#rev-1", 2),
    ("cold_neutral_fraction", 0, "cold_neutral_fraction#rev-1", 2),
    ("cold_neutral_temperature", 0, "cold_neutral_temperature#rev-1", 2),
    ("coolant_temperature_at_outlet", 0, "coolant_temperature_at_outlet#rev-1", 2),
    ("effective_charge", 0, "effective_charge#rev-1", 2),
    (
        "effective_turn_count_of_passive_loop",
        1,
        "effective_turn_count_of_passive_loop#rev-2",
        3,
    ),
    (
        "electron_density_at_divertor_target",
        0,
        "electron_density_at_divertor_target#rev-1",
        2,
    ),
    (
        "electron_density_at_magnetic_axis",
        0,
        "electron_density_at_magnetic_axis#rev-2",
        3,
    ),
    (
        "electron_density_at_plasma_boundary",
        0,
        "electron_density_at_plasma_boundary#rev-2",
        3,
    ),
    ("electron_temperature", 0, "electron_temperature#rev-1", 2),
    (
        "electron_temperature_at_magnetic_axis",
        0,
        "electron_temperature_at_magnetic_axis#rev-1",
        2,
    ),
    ("elongation_of_flux_surface", 0, "elongation_of_flux_surface#rev-1", 2),
    ("elongation_of_plasma_boundary", 0, "elongation_of_plasma_boundary#rev-1", 2),
    (
        "etendue_of_spectrometer_channel",
        0,
        "etendue_of_spectrometer_channel#rev-1",
        2,
    ),
    ("faraday_angle", 0, "faraday_angle#rev-1", 2),
)


class _DryRunGraph:
    def __enter__(self) -> _DryRunGraph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, int]]:
        assert "RETURN count(sn) AS eligible" in cypher
        assert params == {"sn_ids": ["electron_temperature"]}
        return [{"eligible": 1}]


class _CollisionGraph:
    def __init__(
        self,
        identity: str,
        stale_chain: int,
        expected_revision_id: str,
        expected_chain: int,
    ) -> None:
        self.identity = identity
        self.stale_chain = stale_chain
        self.expected_revision_id = expected_revision_id
        self.expected_chain = expected_chain
        next_revision = int(expected_revision_id.rsplit("#rev-", 1)[1])
        self.revision_ids = {
            f"{identity}#rev-{revision}" for revision in range(next_revision)
        }
        self.docs_stage = "accepted"

    def __enter__(self) -> _CollisionGraph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, int]]:
        assert params["sn_ids"] == [self.identity]
        if "RETURN count(sn) AS eligible" in cypher:
            return [{"eligible": 1}]
        if "RETURN count(sn) AS reset" in cypher:
            old_revision_id = f"{self.identity}#rev-{self.stale_chain}"
            assert old_revision_id in self.revision_ids
            if "OPTIONAL MATCH (prior:DocsRevision)" not in cypher:
                raise RuntimeError(
                    f"deterministic revision id already exists: {old_revision_id}"
                )
            assert "OPTIONAL MATCH (prior:DocsRevision)" in cypher
            assert "prior.id STARTS WITH sn.id + '#rev-'" in cypher
            assert "max(toInteger(split(prior.id, '#rev-')[1]))" in cypher
            assert "CREATE (rev:DocsRevision" in cypher
            assert self.expected_revision_id not in self.revision_ids
            self.revision_ids.add(self.expected_revision_id)
            self.docs_stage = "pending"
            self.stale_chain = self.expected_chain
            return [{"reset": 1}]
        assert "coalesce(sn.docs_stage, 'pending') = 'pending'" in cypher
        assert params["run_id"] == "scoped-docs-refresh"
        return []


def test_empty_reset_is_a_graph_free_no_op(monkeypatch) -> None:
    def unexpected_graph_client() -> None:
        raise AssertionError("an empty reset must not open the graph")

    monkeypatch.setattr(graph_ops, "GraphClient", unexpected_graph_client)

    assert graph_ops.reset_standard_name_docs(sn_ids=[]) == {
        "eligible": 0,
        "reset": 0,
    }


def test_dry_run_reports_eligibility_without_writing(monkeypatch) -> None:
    monkeypatch.setattr(graph_ops, "GraphClient", _DryRunGraph)

    assert graph_ops.reset_standard_name_docs(
        sn_ids=["electron_temperature"], dry_run=True
    ) == {"eligible": 1, "reset": 0}


@pytest.mark.parametrize(
    ("identity", "stale_chain", "expected_revision_id", "expected_chain"),
    _COLLISION_SHAPES,
    ids=[shape[0] for shape in _COLLISION_SHAPES],
)
def test_reset_appends_after_existing_revision_ids(
    monkeypatch,
    identity: str,
    stale_chain: int,
    expected_revision_id: str,
    expected_chain: int,
) -> None:
    graph = _CollisionGraph(
        identity,
        stale_chain,
        expected_revision_id,
        expected_chain,
    )
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)

    result = graph_ops.reset_standard_name_docs(
        sn_ids=[identity], run_id="scoped-docs-refresh"
    )

    assert result == {"eligible": 1, "reset": 1}
    assert graph.docs_stage == "pending"
    assert graph.stale_chain == expected_chain
    assert expected_revision_id in graph.revision_ids
