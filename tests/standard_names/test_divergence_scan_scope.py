"""Tests for the divergence scan's catalog scope.

The divergence scan must read the catalog's own entry files — the named
per-domain set directly under ``standard_names/`` — and never treat files
from any other directory nested inside the checkout (a virtual environment,
for example) as catalog authorities. And the scan must not be narrowed to
nothing: a catalog entry that has really diverged must still be reported.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.standard_names.export import detect_divergence


def _make_candidate(
    name: str,
    origin: str = "catalog_edit",
    **overrides: object,
) -> dict:
    return {
        "id": name,
        "origin": origin,
        "description": "graph description",
        "documentation": "the catalog documentation",
        "kind": "scalar",
        "links": [],
        "status": "active",
        **overrides,
    }


def _write_entry(
    catalog_root: Path, filename: str, name: str, **fields: object
) -> None:
    """Write one catalog entry into a freshly created ``standard_names/``.

    The entry agrees with the graph candidate on every protected field
    except ``description``, which is the one genuine divergence the fixtures
    assert on.
    """
    sn = catalog_root / "standard_names"
    sn.mkdir(parents=True, exist_ok=True)
    entry = {
        "name": name,
        "description": "catalog description",
        "documentation": "the catalog documentation",
        "kind": "scalar",
        "links": [],
        "status": "active",
        **fields,
    }
    (sn / filename).write_text(yaml.safe_dump([entry]), encoding="utf-8")


class TestDivergenceScanScope:
    """The scan reads a named entry set and never gets narrowed to nothing."""

    def test_genuine_divergence_is_still_reported(self, tmp_path: Path) -> None:
        """A catalog entry whose protected field differs from the graph is found.

        This assertion is what a scan narrowed to nothing breaks: with no
        entry files read, the real ``description`` divergence cannot appear.
        """
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        _write_entry(catalog_root, "general.yml", "temperature")

        findings = detect_divergence(
            [_make_candidate("temperature")],
            catalog_root=catalog_root,
        )

        reported_fields = {finding.field for finding in findings}
        assert "description" in reported_fields
        assert [finding.name for finding in findings] == ["temperature"]

    def test_virtual_environment_beside_entries_is_not_read(
        self, tmp_path: Path
    ) -> None:
        """A YAML file inside the checkout's .venv is never a catalog authority.

        A scan that walks the checkout and filters afterwards would load the
        environment file and compare its rows as real entries; the named set
        must not. The environment-derived name may only surface as absent
        from the catalog, never as a field comparison.
        """
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        _write_entry(catalog_root, "general.yml", "temperature")

        venv_payload = catalog_root / ".venv" / "lib" / "python3.13" / "site-packages"
        (venv_payload / "markdown_it").mkdir(parents=True)
        (venv_payload / "markdown_it" / "port.yaml").write_text(
            yaml.safe_dump(
                [{"name": "markdown_port", "description": "port authority"}]
            ),
            encoding="utf-8",
        )

        findings = detect_divergence(
            [
                _make_candidate("temperature"),
                _make_candidate("markdown_port"),
            ],
            catalog_root=catalog_root,
        )

        # temperature diverges for real; markdown_port is absent from the
        # named catalog set and must never carry a field comparison.
        by_name = {finding.name: finding for finding in findings}
        assert by_name["temperature"].field == "description"
        assert by_name["markdown_port"].field == "name"

    def test_dot_yaml_domain_files_are_part_of_the_named_set(
        self, tmp_path: Path
    ) -> None:
        """Both per-domain extensions the layout accepts are read."""
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        _write_entry(catalog_root, "general.yaml", "temperature")

        findings = detect_divergence(
            [_make_candidate("temperature")],
            catalog_root=catalog_root,
        )

        assert [finding.name for finding in findings] == ["temperature"]
        assert {finding.field for finding in findings} == {"description"}

    def test_pipeline_origin_never_compared_against_catalog(
        self, tmp_path: Path
    ) -> None:
        """Only catalog-edited identities enter the comparison at all."""
        catalog_root = tmp_path / "catalog"
        catalog_root.mkdir()
        _write_entry(catalog_root, "general.yml", "temperature")

        findings = detect_divergence(
            [
                _make_candidate("temperature", origin="pipeline"),
                _make_candidate("temperature", origin="catalog_edit"),
            ],
            catalog_root=catalog_root,
        )

        assert [finding.name for finding in findings] == ["temperature"]
