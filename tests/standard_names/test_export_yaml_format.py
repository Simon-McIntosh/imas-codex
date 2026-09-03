"""Catalog exports use the installed ISN review-friendly YAML format."""

from __future__ import annotations

from pathlib import Path

import yaml
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
            "description": "Electron temperature, a review-friendly one-liner.",
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
            "description": "Ion temperature, a review-friendly one-liner.",
            "documentation": (
                "Ion temperature follows\n\n"
                "$$\nT_i = p_i / n_i\n$$\n\n"
                "for ion pressure and density."
            ),
            "kind": "scalar",
            "unit": "eV",
        },
    ]

    metadata: dict[str, dict] = {}
    output = _write_domain_yaml(
        tmp_path, "equilibrium", entries, name_metadata=metadata
    )
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

    # The reviewable entry carries only name/description/documentation/unit;
    # the manifest sidecar's per-name block supplies the machine-derived
    # fields (here, kind and physics_domain) a reader must overlay before the
    # installed entry model can discriminate and validate it.
    resolved_dir = tmp_path / "resolved" / "standard_names"
    resolved_dir.mkdir(parents=True)
    resolved_entries = [
        {**entry, **metadata[entry["name"]]} for entry in yaml.safe_load(rendered)
    ]
    (resolved_dir / "equilibrium.yml").write_text(
        yaml.safe_dump(resolved_entries, sort_keys=False), encoding="utf-8"
    )

    loaded = YamlStore(resolved_dir).load()
    assert [model.model_dump(mode="json") for model in loaded] == [
        create_standard_name_entry(entry).model_dump(mode="json") for entry in entries
    ]
