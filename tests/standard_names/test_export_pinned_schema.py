"""Catalog export stays within the installed ISN entry schema."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import yaml
from imas_standard_names.models import create_standard_name_entry

from imas_codex.standard_names.export import (
    _graph_node_to_entry_dict,
    _validate_entry,
    _write_domain_yaml,
)


def test_graph_entry_carries_physics_domain_into_initial_validation() -> None:
    graph_entry = _graph_node_to_entry_dict(
        {
            "id": "electron_temperature",
            "description": "Temperature of electrons.",
            "documentation": "Temperature of electrons.",
            "kind": "scalar",
            "unit": "eV",
            "physics_domain": "equilibrium",
        }
    )

    assert graph_entry["physics_domain"] == "equilibrium"
    validated = _validate_entry(graph_entry)
    assert validated is not None
    assert validated["physics_domain"] == "equilibrium"


def test_fixture_cohort_round_trips_through_pinned_entry_factory(
    tmp_path: Path,
) -> None:
    assert importlib.metadata.version("imas-standard-names") == "0.8.2"
    cohort = [
        {
            "name": "electron_temperature",
            "description": "Temperature of electrons.",
            "documentation": "Temperature of electrons.",
            "kind": "scalar",
            "unit": "eV",
            "status": "active",
            "links": [],
            "physics_domain": "equilibrium",
            "roles": ["quantity", "parent"],
        },
        {
            "name": "ion_temperature",
            "description": "Temperature of ions.",
            "documentation": "Temperature of ions.",
            "kind": "scalar",
            "unit": "eV",
            "status": "active",
            "links": [],
            "physics_domain": "equilibrium",
            "roles": ["quantity"],
        },
    ]

    output = _write_domain_yaml(tmp_path, "equilibrium", cohort)
    emitted = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert len(emitted) == len(cohort)
    assert all("roles" not in entry for entry in emitted)
    for entry in emitted:
        validated = create_standard_name_entry(entry)
        assert validated.name == entry["name"]
