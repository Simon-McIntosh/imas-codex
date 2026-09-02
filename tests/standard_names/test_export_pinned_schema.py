"""Catalog export stays within the installed ISN entry schema."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import yaml
from imas_standard_names.models import create_standard_name_entry

from imas_codex.standard_names.export import _write_domain_yaml


def test_fixture_cohort_round_trips_through_pinned_entry_factory(
    tmp_path: Path,
) -> None:
    assert importlib.metadata.version("imas-standard-names") == "0.8.0"
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
