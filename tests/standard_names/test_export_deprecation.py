"""Deprecation architecture: accepted names retired by supersession export
as ``status: deprecated`` stubs pointing at their live successor.

Covers:
  * the ``_fetch_deprecation_stubs`` Cypher contract (accepted-only gate,
    REFINED_FROM chain collapse to the live accepted successor);
  * the Python post-processing (published-successor filter, deterministic
    branch collapse);
  * ``_build_stub_entry`` field mapping;
  * an end-to-end ``run_export`` that emits a stub and reports the count
    without inflating the CLOSED catalog manifest.

All graph access is mocked — no live Neo4j.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.export import (  # noqa: E402
    _build_stub_entry,
    _fetch_deprecation_stubs,
    run_export,
)

_GC_PATH = "imas_codex.graph.client.GraphClient"


# ---------------------------------------------------------------------------
# Cypher contract — accepted-only gate + chain-collapse successor predicate
# ---------------------------------------------------------------------------


class TestFetchDeprecationStubsQueryContract:
    """The stub query must gate on the recorded accepted stage and resolve
    the live successor along the REFINED_FROM chain."""

    def _capture(self) -> str:
        captured: list[str] = []

        def _query(cypher: str, **kw):
            captured.append(cypher)
            return []

        mock_gc = MagicMock()
        mock_gc.query = _query
        with patch(_GC_PATH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            _fetch_deprecation_stubs(published_names=set())
        assert captured, "_fetch_deprecation_stubs issued no query"
        return captured[0]

    def test_gates_on_superseded_stage(self):
        cypher = self._capture()
        assert "name_stage" in cypher and "'superseded'" in cypher

    def test_accepted_only_scope(self):
        """Draft/reviewed churn (superseded_from_stage != 'accepted') is
        excluded by the query — the locked dep-scope decision."""
        cypher = self._capture()
        assert "superseded_from_stage = 'accepted'" in cypher

    def test_collapses_chain_to_live_accepted_successor(self):
        cypher = self._capture()
        assert "REFINED_FROM*1.." in cypher
        assert "succ.name_stage = 'accepted'" in cypher


# ---------------------------------------------------------------------------
# Python post-processing — published filter + deterministic collapse
# ---------------------------------------------------------------------------


def _gc_returning(rows: list[dict]) -> MagicMock:
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=rows)
    return mock_gc


def _run_fetch(rows: list[dict], published: set[str]) -> list[dict]:
    mock_gc = _gc_returning(rows)
    with patch(_GC_PATH) as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        return _fetch_deprecation_stubs(published_names=published)


class TestFetchDeprecationStubsProcessing:
    def test_resolves_single_successor(self):
        rows = [
            {
                "record": {"id": "old_name", "kind": "scalar", "unit": "eV"},
                "successors": ["new_name"],
            }
        ]
        stubs = _run_fetch(rows, published={"new_name"})
        assert len(stubs) == 1
        assert stubs[0]["_successor"] == "new_name"

    def test_skips_unpublished_successor(self):
        """A successor that isn't in the published set would be a dangling
        breaking-change pointer — skip the stub this release."""
        rows = [
            {
                "record": {"id": "old_name", "kind": "scalar", "unit": "eV"},
                "successors": ["new_name"],
            }
        ]
        stubs = _run_fetch(rows, published=set())
        assert stubs == []

    def test_branch_collapse_is_deterministic(self):
        """Multiple live successors collapse to the lexicographically first
        published one."""
        rows = [
            {
                "record": {"id": "old_name", "kind": "scalar", "unit": "eV"},
                "successors": ["z_successor", "a_successor"],
            }
        ]
        stubs = _run_fetch(rows, published={"a_successor", "z_successor"})
        assert stubs[0]["_successor"] == "a_successor"


# ---------------------------------------------------------------------------
# Stub entry construction
# ---------------------------------------------------------------------------


class TestBuildStubEntry:
    def test_scalar_stub_fields(self):
        node = {
            "id": "ion_temperature_core",
            "kind": "scalar",
            "unit": "eV",
            "_successor": "core_ion_temperature",
        }
        entry = _build_stub_entry(node)
        assert entry["name"] == "ion_temperature_core"
        assert entry["status"] == "deprecated"
        assert entry["superseded_by"] == "core_ion_temperature"
        assert entry["kind"] == "scalar"
        assert entry["unit"] == "eV"
        assert "name:core_ion_temperature" in entry["links"]
        assert "core_ion_temperature" in entry["documentation"]

    def test_metadata_stub_has_no_unit(self):
        node = {
            "id": "old_boundary",
            "kind": "metadata",
            "_successor": "plasma_boundary",
        }
        entry = _build_stub_entry(node)
        assert "unit" not in entry
        assert entry["superseded_by"] == "plasma_boundary"


# ---------------------------------------------------------------------------
# End-to-end run_export — stub emitted, counted, manifest untouched
# ---------------------------------------------------------------------------


class _FakeGraphClient:
    """Dispatches gc.query() by Cypher content for a full run_export."""

    def __init__(self, candidates: list[dict], stub_rows: list[dict]):
        self._candidates = candidates
        self._stub_rows = stub_rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **kwargs):
        if "superseded_from_stage = 'accepted'" in cypher:
            return list(self._stub_rows)
        if "sn.name_stage = 'accepted'" in cypher and "OPTIONAL MATCH (sn)" in cypher:
            return [{"record": c} for c in self._candidates]
        # ordering edges / arguments / error_variants / sources / domain
        # priority — none relevant to this fixture.
        return []


def _load_domain_yaml(staging: Path, domain: str) -> list[dict]:
    text = (staging / "standard_names" / f"{domain}.yml").read_text()
    return yaml.safe_load(text)


class TestRunExportEmitsStub:
    @pytest.fixture()
    def exported(self, tmp_path: Path):
        candidate = {
            "id": "core_ion_temperature",
            "reviewer_score_name": 0.9,
            "description": "Core ion temperature.",
            "documentation": "The core ion temperature measured by CXRS.",
            "kind": "scalar",
            "unit": "eV",
            "physics_domain": "core_profiles",
            # A stored draft status that must be overridden to 'active'.
            "status": "draft",
        }
        stub_rows = [
            {
                "record": {
                    "id": "ion_temperature_core",
                    "kind": "scalar",
                    "unit": "eV",
                    "physics_domain": "core_profiles",
                    "name_stage": "superseded",
                    "superseded_from_stage": "accepted",
                },
                "successors": ["core_ion_temperature"],
            }
        ]
        fake = _FakeGraphClient([candidate], stub_rows)
        with patch(_GC_PATH, return_value=fake):
            report = run_export(
                tmp_path,
                skip_gate=True,
                force=True,
                include_sources=False,
                include_unreviewed=True,
            )
        return report, tmp_path

    def test_stub_count_reported(self, exported):
        report, _ = exported
        assert report.deprecated_stub_count == 1

    def test_stub_written_to_yaml(self, exported):
        _, staging = exported
        entries = {e["name"]: e for e in _load_domain_yaml(staging, "core_profiles")}
        assert "ion_temperature_core" in entries
        stub = entries["ion_temperature_core"]
        assert stub["status"] == "deprecated"
        assert stub["superseded_by"] == "core_ion_temperature"

    def test_live_name_exported_active(self, exported):
        _, staging = exported
        entries = {e["name"]: e for e in _load_domain_yaml(staging, "core_profiles")}
        assert entries["core_ion_temperature"]["status"] == "active"

    def test_manifest_has_no_stub_field(self, exported):
        """The catalog manifest model is CLOSED — the stub count lives only in
        the export report, never the manifest."""
        _, staging = exported
        manifest = yaml.safe_load((staging / "catalog.yml").read_text())
        assert "deprecated_stub_count" not in manifest
        assert "deprecated_stubs" not in manifest

    def test_export_report_carries_stub_count(self, exported):
        import json

        _, staging = exported
        report_json = json.loads((staging / ".export_report.json").read_text())
        assert report_json["counts"]["deprecated_stubs"] == 1


# ---------------------------------------------------------------------------
# sn merge round-trip — the PR-diff reader leaves stubs unchanged
# ---------------------------------------------------------------------------


def _stub_entry_yaml() -> dict:
    return {
        "name": "ion_temperature_core",
        "kind": "scalar",
        "status": "deprecated",
        "unit": "eV",
        "superseded_by": "core_ion_temperature",
        "description": "Deprecated: renamed to core_ion_temperature.",
        "documentation": "Use core_ion_temperature instead.",
        "links": ["name:core_ion_temperature"],
    }


class TestMergePreservesStubs:
    def test_parse_entries_preserves_stub_fields(self):
        from imas_codex.standard_names.merge import _parse_entries

        text = yaml.safe_dump([_stub_entry_yaml()])
        parsed = _parse_entries(text)
        stub = parsed["ion_temperature_core"]
        assert stub["status"] == "deprecated"
        assert stub["superseded_by"] == "core_ion_temperature"

    def test_read_pr_changes_ignores_unchanged_stub(self, tmp_path: Path):
        """A stub present unchanged in both base and head yields no
        MergeChange, while a legitimate docs edit on the live name still
        flows — the stub round-trips through merge untouched."""
        import subprocess

        repo = tmp_path / "isnc"
        (repo / "standard_names").mkdir(parents=True)

        def _git(*args: str) -> None:
            subprocess.run(
                ["git", *args],
                cwd=repo,
                check=True,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "t",
                    "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t",
                    "PATH": __import__("os").environ["PATH"],
                },
            )

        rel = "standard_names/core_profiles.yml"
        stub = _stub_entry_yaml()
        live = {
            "name": "core_ion_temperature",
            "kind": "scalar",
            "status": "active",
            "unit": "eV",
            "description": "Core ion temperature.",
            "documentation": "Original documentation.",
        }

        _git("init")
        (repo / rel).write_text(yaml.safe_dump([stub, live]))
        _git("add", rel)
        _git("commit", "-m", "base")

        # Head worktree: stub unchanged, live name gets a docs edit.
        live_edited = dict(live, documentation="Revised documentation.")
        (repo / rel).write_text(yaml.safe_dump([stub, live_edited]))

        from imas_codex.standard_names.merge import read_pr_changes

        changes = read_pr_changes(repo, "HEAD")

        touched = {c.sn_id for c in changes}
        assert "ion_temperature_core" not in touched, (
            "deprecation stub must not be picked up as a merge change"
        )
        assert "core_ion_temperature" in touched, (
            "a legitimate docs edit on the live name must still be detected"
        )
