"""Catalog entry discovery reads a named set, never a whole-tree walk.

``check_catalog`` reports the post-copy divergence between the published
catalog and the graph. Its entry discovery must read the catalog's own
per-domain files under ``standard_names/`` and nothing else: a whole-tree
walk of the checkout reaches the virtual environment and any other nested
YAML — the measured cause of 387 spurious divergence rows in one release
publish, after a parse error on ``.venv/.../markdown_it/port.yaml``.

The helper under test is exercised directly rather than through
``check_catalog`` because that function opens a graph connection; discovery
is a pure filesystem question that must not depend on one.

The two fixtures pin both failure directions:
- a per-domain entry file directly under ``standard_names/`` MUST be returned;
- a YAML file nested in a ``.venv`` directory anywhere else under the
  checkout MUST NOT be returned.

So the gate fails if the helper is ever restored to a walk-and-filter
(the nested file would reappear) and fails if it is narrowed to nothing
(the per-domain file would vanish).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from imas_codex.standard_names.catalog_import import _catalog_entry_files

#: A real per-domain entry, written the way the exporter emits it.
_ENTRY = [
    {
        "name": "electron_density",
        "description": "Electron density of the thermal population.",
        "documentation": "Electron density of the thermal population.",
        "kind": "scalar",
        "unit": "m^-3",
        "status": "active",
        "links": [],
    }
]


def _catalog_tree(root: Path) -> Path:
    """Build the published-catalog fixture: flat per-domain files + venv noise."""
    catalog = root / "catalog"
    sn_dir = catalog / "standard_names"
    sn_dir.mkdir(parents=True)
    (sn_dir / "equilibrium.yml").write_text(yaml.safe_dump(_ENTRY), encoding="utf-8")
    # Manifest sidecar sits beside standard_names/ in the published layout.
    (catalog / "catalog.yml").write_text(
        yaml.safe_dump({"catalog_name": "imas-standard-names", "names": {}}),
        encoding="utf-8",
    )
    # The exact noise a whole-tree walk picks up: a YAML file nested in the
    # catalog's own virtual environment. A whole-tree walk would return it.
    venv_pkg = (
        catalog / ".venv" / "lib" / "python3.13" / "site-packages" / "markdown_it"
    )
    venv_pkg.mkdir(parents=True)
    (venv_pkg / "port.yaml").write_text("default: 0.13.7\n", encoding="utf-8")
    return catalog


class TestCatalogEntryDiscovery:
    """Discovery names the per-domain entry set, one level deep."""

    def test_returns_per_domain_entry_and_never_the_venv(self, tmp_path: Path) -> None:
        catalog = _catalog_tree(tmp_path)
        equilibrium = catalog / "standard_names" / "equilibrium.yml"

        found = _catalog_entry_files(catalog)

        # The one real per-domain entry file must be discovered ...
        assert equilibrium in found
        # ... and the virtual-environment YAML that a whole-tree walk would
        # pick up must not be, nor the manifest sidecar.
        assert not any(".venv" in str(p) for p in found)
        assert not any(p.name == "catalog.yml" for p in found)
        # Exact equality: a walk-and-filter resurfaces the venv file and a
        # helper narrowed to nothing drops the entry — both fail this line.
        assert found == [equilibrium]

    def test_missing_standard_names_directory_is_an_empty_set(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        assert _catalog_entry_files(empty) == []

    def test_yaml_and_yml_suffixes_are_both_entry_files(self, tmp_path: Path) -> None:
        catalog = _catalog_tree(tmp_path)
        # A .yaml per-domain sibling is an entry file in either spelling.
        (catalog / "standard_names" / "transport.yaml").write_text(
            yaml.safe_dump(_ENTRY), encoding="utf-8"
        )

        names = [p.name for p in _catalog_entry_files(catalog)]
        assert "equilibrium.yml" in names
        assert "transport.yaml" in names
