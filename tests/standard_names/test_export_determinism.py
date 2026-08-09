"""Exports are deterministic, so publish's no-change fast path holds.

Two sources of spurious churn are excluded:
- manifest ``generated_at``/``exported_at`` derive from the source commit, not
  wall-clock ``now()``, so identical content yields identical bytes;
- the per-domain YAML header embeds no codex HEAD sha, which would rewrite
  every domain file on any unrelated codex commit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.export import (  # noqa: E402
    _commit_iso_timestamp,
    _get_codex_commit_sha,
    _manifest_iso_timestamp,
    _source_date_epoch_iso_timestamp,
    _write_domain_yaml,
    _write_manifest,
)

_FIXED = "2026-07-07T12:00:00+02:00"


def _write(dirpath: Path) -> bytes:
    _write_manifest(
        dirpath,
        cocos_convention=17,
        candidate_count=3,
        published_count=2,
        excluded_below_score_count=1,
        excluded_unreviewed_count=0,
        min_score_applied=0.65,
        min_description_score_applied=None,
        include_unreviewed=False,
        source_commit_sha="0123456789abcdef0123456789abcdef01234567",
        export_scope="full",
        domains_included=["equilibrium", "transport"],
    )
    return (dirpath / "catalog.yml").read_bytes()


class TestManifestDeterminism:
    def test_identical_content_identical_bytes(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        with patch(
            "imas_codex.standard_names.export._commit_iso_timestamp",
            return_value=_FIXED,
        ):
            bytes_a = _write(a)
            bytes_b = _write(b)
        assert bytes_a == bytes_b, "manifest bytes must be stable across runs"

    def test_generated_and_exported_derive_from_commit(self, tmp_path: Path) -> None:
        with patch(
            "imas_codex.standard_names.export._commit_iso_timestamp",
            return_value=_FIXED,
        ):
            _write(tmp_path)
        manifest = yaml.safe_load((tmp_path / "catalog.yml").read_text())
        # Both stamps come from the same commit-derived value (may be
        # normalised by the ISN model, but identically for both fields).
        assert manifest["generated_at"] == manifest["exported_at"]

    def test_no_commit_uses_stable_unversioned_timestamp(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            cocos_convention=17,
            candidate_count=0,
            published_count=0,
            excluded_below_score_count=0,
            excluded_unreviewed_count=0,
            min_score_applied=0.65,
            min_description_score_applied=None,
            include_unreviewed=False,
            source_commit_sha=None,
        )
        manifest = yaml.safe_load((tmp_path / "catalog.yml").read_text())
        assert manifest["generated_at"] == "1970-01-01T00:00:00Z"

    def test_required_provenance_fails_clearly_when_unavailable(self) -> None:
        with (
            patch(
                "imas_codex.standard_names.export._commit_iso_timestamp",
                return_value=None,
            ),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="SOURCE_DATE_EPOCH"),
        ):
            _manifest_iso_timestamp(None, require_provenance=True)


class TestCommitTimestamp:
    def test_none_sha_returns_none(self) -> None:
        assert _commit_iso_timestamp(None) is None

    def test_real_sha_returns_iso(self) -> None:
        head = _get_codex_commit_sha()
        if head is None:
            pytest.skip("source commit unavailable outside a VCS/package build")
        ts = _commit_iso_timestamp(head)
        if ts is None:
            pytest.skip("commit object unavailable outside a VCS checkout")
        assert ts is not None and "T" in ts

    def test_archive_uses_packaged_sha_and_source_date_epoch(self) -> None:
        with (
            patch(
                "imas_codex.standard_names.export.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            patch(
                "imas_codex.standard_names.export.importlib.metadata.distribution"
            ) as distribution,
            patch.dict("os.environ", {"SOURCE_DATE_EPOCH": "1783591200"}),
        ):
            distribution.return_value.version = "0.13.dev42+gabc123def"
            distribution.return_value.locate_file.return_value = (
                Path(__file__).parents[2] / "imas_codex/standard_names/export.py"
            )
            sha = _get_codex_commit_sha()
            assert sha == "abc123def"
            assert _manifest_iso_timestamp(sha, require_provenance=True) == (
                "2026-07-09T10:00:00+00:00"
            )

    def test_invalid_source_date_epoch_fails_clearly(self) -> None:
        with (
            patch.dict("os.environ", {"SOURCE_DATE_EPOCH": "not-a-timestamp"}),
            pytest.raises(RuntimeError, match="non-negative Unix timestamp"),
        ):
            _source_date_epoch_iso_timestamp()

    def test_source_archive_rejects_unrelated_installed_metadata(
        self, tmp_path: Path
    ) -> None:
        with (
            patch(
                "imas_codex.standard_names.export.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            patch(
                "imas_codex.standard_names.export.importlib.metadata.distribution"
            ) as distribution,
        ):
            distribution.return_value.version = "0.13.dev42+gabc123def"
            distribution.return_value.locate_file.return_value = (
                tmp_path / "other-install/imas_codex/standard_names/export.py"
            )
            assert _get_codex_commit_sha() is None


class TestDomainHeaderHasNoSha:
    def test_header_omits_catalog_sha(self, tmp_path: Path) -> None:
        entry = {
            "name": "electron_temperature",
            "description": "Electron temperature",
            "documentation": "",
            "kind": "scalar",
            "unit": "eV",
            "status": "draft",
            "links": [],
        }
        _write_domain_yaml(tmp_path, "core_plasma_physics", [entry])
        text = (tmp_path / "standard_names" / "core_plasma_physics.yml").read_text()
        assert "Catalog sha" not in text
        assert "deadbeefcafe" not in text
        assert "# Domain: core_plasma_physics" in text
