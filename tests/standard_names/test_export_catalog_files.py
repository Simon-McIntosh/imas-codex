"""Emitted catalog bytes and the reviewer's route to the exclusion ledger.

Two invariants of a review candidate:

- a per-domain catalog file carries entries only. A preamble restating the
  domain, the entry count or the ordering rule rots as soon as an entry moves
  between domains, and churns every domain file on unrelated edits; the
  manifest (``catalog.yml``) is the one place that state is written.
- the review pull-request body carries an address a reviewer can open to see
  every source path withheld by data dictionary node category together with
  the category that withheld it, pinned to the ledger's own commit so it
  resolves to the bytes the release excluded.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from iter_codex.standard_names.catalog_release import (
    ExclusionLedgerLinkError,
    _write_and_verify_review_preview_link,
    body_with_exclusion_ledger_link,
    exclusion_ledger_path,
)
from iter_codex.standard_names.export import (
    _write_domain_yaml,
    _write_manifest,
    assemble_review_catalog,
)


def _entry(name: str) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name.replace('_', ' ').capitalize()}",
        "documentation": "",
        "kind": "scalar",
        "unit": "eV",
        "status": "draft",
        "links": [],
    }


def _emitted_files(root: Path) -> list[Path]:
    return sorted((root / "standard_names").glob("*.yml"))


def _catalog_manifest(
    root: Path, sources_by_name: dict[str, list[dict[str, str]] | None]
) -> dict[str, object]:
    names = {
        name: {
            "kind": "scalar",
            "status": "draft",
            "physics_domain": "core_plasma_physics",
            "links": [],
            **({"sources": sources} if sources is not None else {}),
        }
        for name, sources in sources_by_name.items()
    }
    _write_manifest(
        root,
        cocos_convention=17,
        candidate_count=len(names),
        published_count=len(names),
        excluded_below_score_count=0,
        excluded_unreviewed_count=0,
        min_score_applied=0.0,
        min_description_score_applied=None,
        include_unreviewed=True,
        names=names,
    )
    manifest = yaml.safe_load((root / "catalog.yml").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    return manifest


def _dd_source(ref: str, version: str) -> dict[str, str]:
    return {"kind": "imas-dd", "ref": ref, "version": version}


class TestCatalogManifestDataDictionaryLineage:
    def test_distinct_source_versions_are_reported_in_numeric_order(
        self, tmp_path: Path
    ) -> None:
        manifest = _catalog_manifest(
            tmp_path,
            {
                "electron_temperature": [
                    _dd_source(
                        "core_profiles/profiles_1d/electrons/temperature", "4.1.1"
                    ),
                    _dd_source("core_profiles/global_quantities/ip", "4.1.0"),
                ],
                "ion_temperature": [
                    _dd_source("core_profiles/profiles_1d/t_i_average", "4.1.1")
                ],
            },
        )

        assert manifest["dd_version_lineage"] == ["4.1.0", "4.1.1"]

    def test_single_source_version_is_not_replaced_by_a_catalog_default(
        self, tmp_path: Path
    ) -> None:
        manifest = _catalog_manifest(
            tmp_path,
            {
                "electron_temperature": [
                    _dd_source(
                        "core_profiles/profiles_1d/electrons/temperature", "4.2.0"
                    )
                ]
            },
        )

        assert manifest["dd_version_lineage"] == ["4.2.0"]

    @pytest.mark.parametrize(
        "sources_by_name",
        [{}, {"radial_coordinate": None}],
        ids=["empty-export", "unsourced-entry"],
    )
    def test_export_without_sourced_entries_has_empty_lineage(
        self,
        tmp_path: Path,
        sources_by_name: dict[str, list[dict[str, str]] | None],
    ) -> None:
        manifest = _catalog_manifest(tmp_path, sources_by_name)

        assert manifest["dd_version_lineage"] == []

    def test_version_order_uses_parsed_numeric_components(self, tmp_path: Path) -> None:
        versions = ["4.1.10", "4.1.0rc65", "4.1.2", "4.1.0rc9"]
        manifest = _catalog_manifest(
            tmp_path,
            {
                f"temperature_{index}": [_dd_source(f"source/{index}", version)]
                for index, version in enumerate(versions)
            },
        )

        assert manifest["dd_version_lineage"] == [
            "4.1.0rc9",
            "4.1.0rc65",
            "4.1.2",
            "4.1.10",
        ]


class TestEmittedFilesCarryEntriesOnly:
    def test_fresh_export_file_has_no_comment_line(self, tmp_path: Path) -> None:
        _write_domain_yaml(
            tmp_path,
            "core_plasma_physics",
            [_entry("electron_temperature"), _entry("ion_temperature")],
        )
        files = _emitted_files(tmp_path)
        assert files
        for path in files:
            lines = path.read_text(encoding="utf-8").splitlines()
            assert lines
            assert not lines[0].startswith("#")
            assert [line for line in lines if line.startswith("#")] == []

    def test_review_assembly_file_has_no_comment_line(self, tmp_path: Path) -> None:
        approved_root = tmp_path / "approved"
        staging_root = tmp_path / "staging"
        _write_domain_yaml(
            approved_root, "core_plasma_physics", [_entry("electron_temperature")]
        )
        _write_domain_yaml(
            staging_root, "core_plasma_physics", [_entry("ion_temperature")]
        )

        assemble_review_catalog(
            approved_root,
            staging_root,
            batch_names=["ion_temperature"],
        )

        files = _emitted_files(staging_root)
        assert files
        for path in files:
            lines = path.read_text(encoding="utf-8").splitlines()
            assert lines
            assert not lines[0].startswith("#")
            assert [line for line in lines if line.startswith("#")] == []

    def test_assembly_still_carries_both_identities(self, tmp_path: Path) -> None:
        """Dropping the preamble must not drop an entry with it."""
        approved_root = tmp_path / "approved"
        staging_root = tmp_path / "staging"
        _write_domain_yaml(
            approved_root, "core_plasma_physics", [_entry("electron_temperature")]
        )
        _write_domain_yaml(
            staging_root, "core_plasma_physics", [_entry("ion_temperature")]
        )

        assemble_review_catalog(
            approved_root,
            staging_root,
            batch_names=["ion_temperature"],
        )

        import yaml

        entries = yaml.safe_load(
            (staging_root / "standard_names" / "core_plasma_physics.yml").read_text(
                encoding="utf-8"
            )
        )
        assert sorted(entry["name"] for entry in entries) == [
            "electron_temperature",
            "ion_temperature",
        ]


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def manifest_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A committed manifest plus its exclusion ledger, with a github origin."""
    root = tmp_path / "codex"
    manifests = root / "iter_codex" / "standard_names" / "manifests"
    manifests.mkdir(parents=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "release@example.invalid", cwd=root)
    _git("config", "user.name", "Release", cwd=root)
    _git(
        "remote",
        "add",
        "origin",
        "git@github.com:test-owner/iter-codex.git",
        cwd=root,
    )
    manifest = manifests / "batch_dd_paths.yaml"
    manifest.write_text("sources:\n  - magnetics/flux_loop/flux\n", encoding="utf-8")
    ledger = manifests / "batch_dd_paths.exclusions.json"
    ledger.write_text(
        json.dumps(
            {
                "excluded_ineligible": [
                    {
                        "path": "magnetics/flux_loop/name",
                        "reason": "excluded_metadata",
                        "category": "metadata",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _git(
        "add",
        "iter_codex/standard_names/manifests/batch_dd_paths.yaml",
        "iter_codex/standard_names/manifests/batch_dd_paths.exclusions.json",
        cwd=root,
    )
    _git("commit", "-q", "-m", "add batch manifest and ledger", cwd=root)
    return root, manifest


class TestReviewBodyCarriesExclusionLedger:
    def test_body_link_resolves_to_the_committed_ledger(
        self, manifest_checkout: tuple[Path, Path]
    ) -> None:
        root, manifest = manifest_checkout
        ledger = exclusion_ledger_path(manifest)
        assert ledger is not None

        body = body_with_exclusion_ledger_link("Review candidate.", manifest)

        relative = "iter_codex/standard_names/manifests/batch_dd_paths.exclusions.json"
        sha = _git("log", "-1", "--format=%H", "--", relative, cwd=root).stdout.strip()
        expected = f"https://github.com/test-owner/iter-codex/blob/{sha}/{relative}"
        expected_link = (
            "[ledger of excluded source paths and their withholding data "
            f"dictionary node categories]({expected})"
        )
        assert (
            "Excluded source paths, each with the data dictionary node category "
            f"that excluded it, are recorded in the {expected_link}." in body
        )
        assert expected not in body.replace(expected_link, "", 1)
        assert body.startswith("Review candidate.")
        # Resolvable: the exact revision and path in the address hold the
        # ledger bytes, so the reviewer's link is not a 404.
        blob = _git("cat-file", "-p", f"{sha}:{relative}", cwd=root).stdout
        rows = json.loads(blob)["excluded_ineligible"]
        assert rows[0]["path"] == "magnetics/flux_loop/name"
        assert rows[0]["category"] == "metadata"

    def test_uncommitted_ledger_refuses_to_publish_an_address(
        self, manifest_checkout: tuple[Path, Path]
    ) -> None:
        root, manifest = manifest_checkout
        other = manifest.with_name("second_dd_paths.yaml")
        other.write_text("sources: []\n", encoding="utf-8")
        other.with_name("second_dd_paths.exclusions.json").write_text(
            "{}", encoding="utf-8"
        )
        assert root.is_dir()

        with pytest.raises(ExclusionLedgerLinkError, match="no commit"):
            body_with_exclusion_ledger_link("Review candidate.", other)

    def test_manifest_without_a_ledger_leaves_the_body_untouched(
        self, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "names_only.yaml"
        manifest.write_text("names: []\n", encoding="utf-8")
        assert exclusion_ledger_path(manifest) is None
        assert body_with_exclusion_ledger_link("Body.", manifest) == "Body."


class TestReviewBodyCarriesPreviewLink:
    def test_preview_address_is_named_and_not_bare(self) -> None:
        class RecordingClient:
            body = ""

            def update_pull_request_body(
                self, *, repo: str, number: int, body: str
            ) -> None:
                assert repo == "test-owner/imas-standard-names-catalog"
                assert number == 7
                self.body = body

            def read_pull_request_body(self, *, repo: str, number: int) -> str:
                assert repo == "test-owner/imas-standard-names-catalog"
                assert number == 7
                return self.body

        client = RecordingClient()
        preview_url = _write_and_verify_review_preview_link(
            client,
            repo="test-owner/imas-standard-names-catalog",
            pr_number=7,
            body="Review candidate.",
        )

        expected = "https://test-owner.github.io/imas-standard-names-catalog/pr-7/"
        expected_link = f"[rendered catalog preview]({expected})"
        assert preview_url == expected
        assert f"Preview: {expected_link}" in client.body
        assert expected not in client.body.replace(expected_link, "", 1)
