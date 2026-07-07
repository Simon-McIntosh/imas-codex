"""Tests for the catalog feedback import module (Phase 4).

Tests error handling, SHA resolution, check mode, and field normalization.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_CATALOG_ENTRY = {
    "name": "electron_temperature",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te is measured by Thomson scattering.",
    "kind": "scalar",
    "unit": "eV",
    "links": [],
    "physics_domain": "core_plasma_physics",
    "status": "active",
}

SAMPLE_CATALOG_ENTRY_MINIMAL = {
    "name": "plasma_current",
    "description": "Plasma current",
    "documentation": "Total toroidal plasma current.",
    "kind": "scalar",
    "unit": "A",
    "links": [],
    "physics_domain": "equilibrium",
    "status": "active",
}


@pytest.fixture()
def catalog_dir(tmp_path: Path) -> Path:
    """Create a temporary catalog directory using the per-domain list layout.

    Layout: ``<root>/standard_names/<domain>.yaml`` containing a list of entries.
    Per-file layout (one dict per .yaml) is no longer supported by check_catalog
    (silently skipped — see catalog_import.py line 803).
    """
    d = tmp_path / "catalog"
    sn_dir = d / "standard_names"
    sn_dir.mkdir(parents=True)
    # Group by physics_domain so _derive_domain_from_path resolves correctly.
    (sn_dir / "core_plasma_physics.yaml").write_text(
        yaml.safe_dump([SAMPLE_CATALOG_ENTRY])
    )
    (sn_dir / "equilibrium.yaml").write_text(
        yaml.safe_dump([SAMPLE_CATALOG_ENTRY_MINIMAL])
    )
    return d


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Test graceful error handling."""

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Should report errors for invalid YAML files without crashing."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "bad.yaml").write_text(": : : invalid yaml [[[")

        from imas_codex.standard_names.catalog_import import run_import

        result = run_import(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "bad.yaml" in result.errors[0]

    def test_handles_non_list_yaml(self, tmp_path: Path) -> None:
        """Should report errors for per-domain YAML files that aren't lists."""
        d = tmp_path / "catalog"
        sn_dir = d / "standard_names"
        sn_dir.mkdir(parents=True)
        # Per-domain layout expects a list; a scalar is invalid
        (sn_dir / "core.yaml").write_text(yaml.safe_dump(42))

        from imas_codex.standard_names.catalog_import import run_import

        result = run_import(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "expected a YAML list" in result.errors[0]

    def test_handles_incomplete_entry(self, tmp_path: Path) -> None:
        """Should report errors for entries missing required fields."""
        d = tmp_path / "catalog"
        d.mkdir()
        incomplete = {"name": "test", "kind": "scalar"}  # missing required fields
        (d / "incomplete.yaml").write_text(yaml.safe_dump(incomplete))

        from imas_codex.standard_names.catalog_import import run_import

        result = run_import(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "incomplete.yaml" in result.errors[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty result."""
        d = tmp_path / "empty"
        d.mkdir()

        from imas_codex.standard_names.catalog_import import run_import

        result = run_import(d, dry_run=True)

        assert result.imported == 0
        assert result.skipped == 0
        assert len(result.errors) == 0


# =============================================================================
# SHA resolution
# =============================================================================


class TestResolveCatalogSha:
    """Tests for _resolve_catalog_sha()."""

    def test_returns_sha_in_git_repo(self, tmp_path: Path) -> None:
        """Should return a 40-char SHA when run in a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        # Use the project repo itself as the catalog dir
        project_root = Path(__file__).resolve().parents[2]
        sha = _resolve_catalog_sha(project_root)
        assert sha is not None
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_none_for_non_git_dir(self, tmp_path: Path) -> None:
        """Should return None for a directory that isn't a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        sha = _resolve_catalog_sha(tmp_path)
        assert sha is None

    def test_returns_none_when_git_not_found(self, tmp_path: Path) -> None:
        """Should return None when git binary is missing."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        with patch(
            "imas_codex.standard_names.catalog_import.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            sha = _resolve_catalog_sha(tmp_path)
            assert sha is None


# =============================================================================
# Check mode
# =============================================================================


class TestCheckMode:
    """Tests for check_catalog() — the --check sync comparison."""

    def test_only_in_catalog(self, catalog_dir: Path) -> None:
        """Entries in catalog but not graph should appear in only_in_catalog."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Graph has no entries
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert cr.only_in_catalog == ["electron_temperature", "plasma_current"]
        assert cr.in_sync == 0

    def test_check_empty_catalog(self, tmp_path: Path) -> None:
        """Empty catalog directory should return empty CheckResult."""
        from imas_codex.standard_names.catalog_import import check_catalog

        d = tmp_path / "empty_catalog"
        d.mkdir()

        with patch(
            "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
            return_value=None,
        ):
            cr = check_catalog(catalog_dir=d)

        assert cr.in_sync == 0
        assert cr.only_in_catalog == []
        assert cr.only_in_graph == []
        assert cr.diverged == []


# =============================================================================
# Import write: validation_status defaulting (export eligibility)
# =============================================================================


def _import_merge_cypher(mock_gc: MagicMock) -> str:
    """Return the Cypher from the first MERGE StandardName query call."""
    for call in mock_gc.query.call_args_list:
        cypher = call[0][0]
        if "MERGE (sn:StandardName" in cypher:
            return cypher
    raise AssertionError("No MERGE StandardName query found in calls")


class TestImportSetsValidationStatus:
    """Import-created nodes must be export-eligible.

    ``export._fetch_candidates`` requires ``validation_status = 'valid'``.
    A node created purely by import would otherwise carry a null
    ``validation_status`` and be silently dropped on the next
    export → publish (and deleted from ISNC by the full-scope rmtree).
    """

    def test_merge_defaults_validation_status_valid(self) -> None:
        """The MERGE SET clause must coalesce validation_status to 'valid'."""
        from imas_codex.standard_names.catalog_import import _write_import_entries

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _write_import_entries(
            mock_gc,
            [
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature",
                    "documentation": "Te from Thomson scattering.",
                    "kind": "scalar",
                    "unit": "eV",
                    "links": None,
                    "status": "active",
                    "physics_domain": "core_plasma_physics",
                }
            ],
        )

        cypher = _import_merge_cypher(mock_gc)
        assert (
            "sn.validation_status = coalesce(sn.validation_status, 'valid')" in cypher
        ), (
            "Import MERGE must default validation_status to 'valid' so "
            "import-created nodes satisfy export eligibility.\n\n"
            f"Full MERGE query:\n{cypher}"
        )

    def test_coalesce_preserves_existing_status(self) -> None:
        """coalesce keeps an existing (e.g. quarantined) status untouched.

        This is a contract assertion on the Cypher form: because the
        expression is ``coalesce(sn.validation_status, 'valid')`` — reading
        the node's own property first — a pre-existing non-null status such
        as ``'quarantined'`` is preserved, and only a null status is defaulted.
        """
        from imas_codex.standard_names.catalog_import import _write_import_entries

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _write_import_entries(
            mock_gc,
            [
                {
                    "id": "plasma_current",
                    "description": "Plasma current",
                    "documentation": "Total toroidal plasma current.",
                    "kind": "scalar",
                    "unit": "A",
                    "links": None,
                    "status": "active",
                    "physics_domain": "equilibrium",
                }
            ],
        )

        cypher = _import_merge_cypher(mock_gc)
        # The node property is the first coalesce argument → existing wins.
        assert "coalesce(sn.validation_status, 'valid')" in cypher
        assert "coalesce('valid', sn.validation_status)" not in cypher, (
            "coalesce order must read the node property first so an existing "
            "status (e.g. 'quarantined') is preserved, not clobbered."
        )


# =============================================================================
# Normalize field
# =============================================================================


class TestNormalizeField:
    """Tests for _normalize_field() comparison normalization."""

    def test_none(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(None) is None

    def test_empty_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("") is None
        assert _normalize_field("  ") is None

    def test_normal_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("hello") == "hello"
        assert _normalize_field("  hello  ") == "hello"

    def test_empty_list(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field([]) is None

    def test_list_sorted(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(["b", "a"]) == ("a", "b")
        assert _normalize_field(["a", "b"]) == ("a", "b")

    def test_numeric_passthrough(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(42) == 42
        assert _normalize_field(3.14) == 3.14


# =============================================================================
# Import diff: honest skipped + superseded guard
# =============================================================================


class TestEntryIsUnchanged:
    """Tests for _entry_is_unchanged() — the true no-op detector."""

    def _graph(self, **overrides) -> dict:
        base = {
            "description": "Electron temperature",
            "documentation": "Te from Thomson scattering.",
            "kind": "scalar",
            "unit": "eV",
            "links": [],
            "status": "active",
            "deprecates": None,
            "superseded_by": None,
            "validity_domain": None,
            "constraints": None,
            "physics_domain": "kinetics",
            "source_domains": ["kinetics"],
        }
        base.update(overrides)
        return base

    def test_identical_content_is_unchanged(self) -> None:
        from imas_codex.standard_names.catalog_import import _entry_is_unchanged

        graph = self._graph()
        # A prepared entry normalises empty links to None — still a no-op.
        new = self._graph(links=None)
        assert _entry_is_unchanged(graph, new) is True

    def test_list_reorder_is_unchanged(self) -> None:
        from imas_codex.standard_names.catalog_import import _entry_is_unchanged

        graph = self._graph(links=["name:a", "name:b"])
        new = self._graph(links=["name:b", "name:a"])
        assert _entry_is_unchanged(graph, new) is True

    def test_unit_change_is_a_change(self) -> None:
        """unit is not a protected field but still a real change."""
        from imas_codex.standard_names.catalog_import import _entry_is_unchanged

        graph = self._graph(unit="eV")
        new = self._graph(unit="keV")
        assert _entry_is_unchanged(graph, new) is False

    def test_physics_domain_change_is_a_change(self) -> None:
        from imas_codex.standard_names.catalog_import import _entry_is_unchanged

        graph = self._graph(physics_domain="kinetics")
        new = self._graph(physics_domain="equilibrium")
        assert _entry_is_unchanged(graph, new) is False

    def test_description_change_is_a_change(self) -> None:
        from imas_codex.standard_names.catalog_import import _entry_is_unchanged

        graph = self._graph(description="Electron temperature")
        new = self._graph(description="Electron temperature (revised)")
        assert _entry_is_unchanged(graph, new) is False


class TestFetchGraphStateFields:
    """_fetch_graph_state must return the fields the diff loop reads."""

    def test_query_projects_diff_fields(self) -> None:
        from imas_codex.standard_names.catalog_import import _fetch_graph_state

        captured: list[str] = []

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            side_effect=lambda cypher, **kw: captured.append(cypher) or []
        )
        _fetch_graph_state(mock_gc, ["electron_temperature"])

        cypher = captured[0]
        for field_name in ("name_stage", "unit", "physics_domain", "source_domains"):
            assert f"AS {field_name}" in cypher, (
                f"_fetch_graph_state must project '{field_name}' for the "
                f"import diff.\n\nFull query:\n{cypher}"
            )


class TestImportDiffWriteSet:
    """run_import routes entries to the write set honestly.

    - superseded graph node → never resurrected (skipped, not written)
    - genuinely-unchanged entry → skipped, not written
    - changed entry → written
    """

    _GC_PATCH = "imas_codex.graph.client.GraphClient"

    def _make_catalog(self, tmp_path: Path) -> Path:
        root = tmp_path / "isnc"
        sn_dir = root / "standard_names"
        sn_dir.mkdir(parents=True)
        entries = [
            {
                "name": "superseded_name",
                "kind": "scalar",
                "unit": "eV",
                "description": "Superseded quantity",
                "documentation": "A quantity replaced by a refinement.",
                "links": [],
                "status": "active",
            },
            {
                "name": "unchanged_name",
                "kind": "scalar",
                "unit": "eV",
                "description": "Unchanged quantity",
                "documentation": "Identical to the graph node.",
                "links": [],
                "status": "active",
            },
            {
                "name": "changed_name",
                "kind": "scalar",
                "unit": "eV",
                "description": "New description",
                "documentation": "Description differs from graph.",
                "links": [],
                "status": "active",
            },
        ]
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump(entries))
        return root

    def _graph_rows(self) -> list[dict]:
        common = {
            "kind": "scalar",
            "unit": "eV",
            "links": [],
            "status": "active",
            "deprecates": None,
            "superseded_by": None,
            "validity_domain": None,
            "constraints": None,
            "physics_domain": "kinetics",
            "source_domains": ["kinetics"],
            "origin": "pipeline",
        }
        return [
            {
                "id": "superseded_name",
                "name_stage": "superseded",
                "description": "Superseded quantity",
                "documentation": "A quantity replaced by a refinement.",
                **common,
            },
            {
                "id": "unchanged_name",
                "name_stage": "accepted",
                "description": "Unchanged quantity",
                "documentation": "Identical to the graph node.",
                **common,
            },
            {
                "id": "changed_name",
                "name_stage": "accepted",
                # Graph description differs from the catalog's "New description".
                "description": "Stale description",
                "documentation": "Description differs from graph.",
                **common,
            },
        ]

    def test_write_set_excludes_superseded_and_unchanged(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.catalog_import import run_import

        isnc = self._make_catalog(tmp_path)
        graph_rows = self._graph_rows()
        captured: dict[str, Any] = {}

        def _query(cypher: str, **params):
            if "ImportLock" in cypher and "holder IS NULL" in cypher:
                return [{"acquired": True}]
            if "ImportLock" in cypher:
                return []
            if "UNWIND $ids AS id" in cypher:  # _fetch_graph_state
                return graph_rows
            # Main import MERGE (identified by the name_stage SET); capture once.
            if (
                "MERGE (sn:StandardName" in cypher
                and "sn.name_stage" in cypher
                and "batch" not in captured
            ):
                captured["batch"] = params.get("batch")
                return []
            return []

        gc = MagicMock()
        gc.query = MagicMock(side_effect=_query)

        with patch(self._GC_PATCH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            report = run_import(isnc)

        written_ids = {e["id"] for e in captured.get("batch", [])}
        assert written_ids == {"changed_name"}, (
            "Only the changed entry should be written; superseded and "
            f"unchanged must be excluded. Got: {written_ids}"
        )
        assert report.updated == 1
        assert report.skipped == 2  # superseded + unchanged
        assert report.created == 0
        assert report.imported == 1  # only the written entry counts
