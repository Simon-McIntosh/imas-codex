"""Catalog exports use the installed ISN review-friendly YAML format."""

from __future__ import annotations

from pathlib import Path

from imas_standard_names.models import create_standard_name_entry
from imas_standard_names.yaml_store import YamlStore

from imas_codex.standard_names.export import _write_domain_yaml


def test_domain_export_uses_review_friendly_yaml_and_round_trips(
    tmp_path: Path,
) -> None:
    entries = [
        {
            "name": "electron_temperature",
            "physics_domain": "equilibrium",
            "description": "Electron temperature has two lines.\nIt remains literal.",
            "documentation": (
                "Electron temperature follows\n\n"
                "$$\nT_e = p_e / n_e\n$$\n\n"
                "for electron pressure and density."
            ),
            "kind": "scalar",
            "unit": "eV",
        },
        {
            "name": "ion_temperature",
            "physics_domain": "equilibrium",
            "description": "Ion temperature has two lines.\nIt remains literal.",
            "documentation": (
                "Ion temperature follows\n\n"
                "$$\nT_i = p_i / n_i\n$$\n\n"
                "for ion pressure and density."
            ),
            "kind": "scalar",
            "unit": "eV",
        },
    ]

    output = _write_domain_yaml(tmp_path, "equilibrium", entries)
    rendered = output.read_text(encoding="utf-8")

    assert rendered.count("  description: |-") == 2
    assert rendered.count("  documentation: |-") == 2
    assert rendered.count("\n\n- name:") == 1
    assert "\n\n\n" not in rendered
    for symbol in ("e", "i"):
        assert (
            f"temperature follows\n\n"
            f"    $$\n    T_{symbol} = p_{symbol} / n_{symbol}\n    $$\n\n"
            f"    for {symbol}"
        ) in rendered

    loaded = YamlStore(tmp_path / "standard_names").load()
    assert [model.model_dump(mode="json") for model in loaded] == [
        create_standard_name_entry(entry).model_dump(mode="json") for entry in entries
    ]
