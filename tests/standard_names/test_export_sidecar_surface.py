"""The reviewable entry shrinks to four fields; the sidecar carries the rest.

Three surfaces move together and are covered here as one contract: the
exporter writes machine-derived fields into the manifest's per-name block
instead of onto each entry, the fold-back comparison recovers those fields
from that block so it keeps checking them against the graph, and the publish
compatibility gate moves off the first manifest shape stamp in step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from imas_codex.standard_names.catalog_import import check_catalog
from imas_codex.standard_names.export import (
    _ENTRY_REVIEW_FIELDS,
    _SIDECAR_NAME_FIELDS,
    CATALOG_EDGE_MODEL_VERSION,
    _write_domain_yaml,
    _write_manifest,
)
from imas_codex.standard_names.publish import run_publish

_COHORT: list[dict[str, Any]] = [
    {
        "name": "electron_temperature",
        "physics_domain": "equilibrium",
        "description": "Electron temperature of the thermal population.",
        "documentation": "Electron temperature of the thermal population.",
        "kind": "scalar",
        "unit": "eV",
        "status": "active",
        "links": ["name:ion_temperature"],
        "sources": [
            {
                "kind": "imas-dd",
                "ref": "core_profiles/profiles_1d/electrons/temperature",
                "version": "4.0.0",
            }
        ],
    },
    {
        "name": "ion_temperature",
        "physics_domain": "equilibrium",
        "description": "Ion temperature of the thermal population.",
        "documentation": "Ion temperature of the thermal population.",
        "kind": "scalar",
        "unit": "eV",
        "status": "active",
        "links": [],
        "arguments": [
            {
                "name": "electron_temperature",
                "operator": "ratio",
                "operator_kind": "binary",
                "role": "a",
                "separator": "to",
            }
        ],
        "sources": [
            {
                "kind": "imas-dd",
                "ref": "core_profiles/profiles_1d/ion/temperature",
                "version": "4.0.0",
            }
        ],
    },
]


def _export(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write one domain file plus the manifest and return both, parsed."""
    metadata: dict[str, dict[str, Any]] = {}
    written = _write_domain_yaml(
        tmp_path, "equilibrium", _COHORT, name_metadata=metadata
    )
    _write_manifest(
        tmp_path,
        cocos_convention=17,
        candidate_count=len(_COHORT),
        published_count=len(_COHORT),
        excluded_below_score_count=0,
        excluded_unreviewed_count=0,
        min_score_applied=0.65,
        min_description_score_applied=None,
        include_unreviewed=False,
        source_commit_sha=None,
        export_scope="review",
        domains_included=["equilibrium"],
        review_batch=[entry["name"] for entry in _COHORT],
        names=metadata,
    )
    entries = yaml.safe_load(written.read_text(encoding="utf-8"))
    manifest = yaml.safe_load((tmp_path / "catalog.yml").read_text(encoding="utf-8"))
    return entries, manifest


class TestEntryKeepsOnlyTheReviewableFields:
    """The domain file holds name, description, documentation and unit."""

    def test_entry_carries_exactly_the_four_reviewable_fields(
        self, tmp_path: Path
    ) -> None:
        entries, _ = _export(tmp_path)

        assert len(entries) == len(_COHORT)
        for entry in entries:
            assert set(entry) == set(_ENTRY_REVIEW_FIELDS)

    def test_no_machine_derived_field_survives_in_the_entry(
        self, tmp_path: Path
    ) -> None:
        entries, _ = _export(tmp_path)

        for entry in entries:
            assert not set(entry) & set(_SIDECAR_NAME_FIELDS)

    def test_sidecar_holds_a_block_for_every_published_name(
        self, tmp_path: Path
    ) -> None:
        entries, manifest = _export(tmp_path)

        block = manifest["names"]
        assert set(block) == {entry["name"] for entry in entries}

    def test_each_block_carries_the_six_machine_derived_fields(
        self, tmp_path: Path
    ) -> None:
        _, manifest = _export(tmp_path)

        electron = manifest["names"]["electron_temperature"]
        assert electron["kind"] == "scalar"
        assert electron["status"] == "active"
        assert electron["physics_domain"] == "equilibrium"
        assert electron["links"] == ["name:ion_temperature"]
        assert electron["sources"][0]["ref"].endswith("electrons/temperature")
        # sources carries its own per-binding version
        assert electron["sources"][0]["version"] == "4.0.0"

        ion = manifest["names"]["ion_temperature"]
        assert ion["arguments"][0]["name"] == "electron_temperature"

    def test_block_never_repeats_a_reviewable_field(self, tmp_path: Path) -> None:
        _, manifest = _export(tmp_path)

        for block in manifest["names"].values():
            assert not set(block) & {"description", "documentation", "unit"}


class TestFoldBackReadsTheSidecar:
    """The catalog-vs-graph comparison recovers the fields it compares."""

    @staticmethod
    def _catalog(tmp_path: Path) -> Path:
        _export(tmp_path)
        return tmp_path

    @staticmethod
    def _graph_rows(**overrides: Any) -> list[dict[str, Any]]:
        row = {
            "id": "electron_temperature",
            "description": "Electron temperature of the thermal population.",
            "documentation": "Electron temperature of the thermal population.",
            "kind": "scalar",
            "unit": "eV",
            "source_paths": None,
            "validity_domain": None,
            "constraints": None,
            "physics_domain": "equilibrium",
            "catalog_commit_sha": None,
        }
        row.update(overrides)
        return [row]

    def _check(self, tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.query.return_value = rows
        with patch("imas_codex.graph.client.GraphClient", return_value=client):
            return check_catalog(self._catalog(tmp_path))

    def test_kind_and_domain_come_off_the_sidecar_and_agree(
        self, tmp_path: Path
    ) -> None:
        result = self._check(tmp_path, self._graph_rows())

        assert result.diverged == []
        assert result.in_sync == 1

    def test_a_kind_disagreement_is_still_reported(self, tmp_path: Path) -> None:
        result = self._check(tmp_path, self._graph_rows(kind="vector"))

        assert [row["name"] for row in result.diverged] == ["electron_temperature"]
        assert result.diverged[0]["fields"]["kind"] == {
            "catalog": "scalar",
            "graph": "vector",
        }

    def test_a_physics_domain_disagreement_is_still_reported(
        self, tmp_path: Path
    ) -> None:
        result = self._check(
            tmp_path, self._graph_rows(physics_domain="core_plasma_physics")
        )

        assert result.diverged[0]["fields"]["physics_domain"] == {
            "catalog": "equilibrium",
            "graph": "core_plasma_physics",
        }

    def test_the_manifest_is_not_read_as_an_entry_file(self, tmp_path: Path) -> None:
        result = self._check(tmp_path, self._graph_rows())

        assert "catalog" not in result.only_in_catalog
        assert result.only_in_catalog == ["ion_temperature"]


class TestPublishGateMovesWithTheShape:
    """The compatibility gate accepts the new stamp and refuses the first."""

    @staticmethod
    def _staging(tmp_path: Path, edge_model_version: str) -> Path:
        """A staging tree whose only defect is the stamp under test.

        Written through the exporter so every field the installed loader
        requires is present: that leaves the stamp the only thing varying, so
        the accepting direction reaches the whole run rather than stopping at
        some other complaint.
        """
        staging = tmp_path / "staging"
        (staging / "standard_names").mkdir(parents=True)
        (staging / "standard_names" / "transport.yml").write_text(
            yaml.safe_dump(
                [
                    {
                        "name": "particle_flux",
                        "description": "Particle flux.",
                        "documentation": "Particle flux through a surface.",
                        "unit": "m^-2.s^-1",
                    }
                ]
            ),
            encoding="utf-8",
        )
        _write_manifest(
            staging,
            cocos_convention=17,
            candidate_count=1,
            published_count=1,
            excluded_below_score_count=0,
            excluded_unreviewed_count=0,
            min_score_applied=0.65,
            min_description_score_applied=None,
            include_unreviewed=False,
            source_commit_sha=None,
            export_scope="domain",
            domains_included=["transport"],
            names={
                "particle_flux": {
                    "kind": "scalar",
                    "status": "active",
                    "physics_domain": "transport",
                    "links": [],
                    "sources": [],
                }
            },
        )
        manifest_path = staging / "catalog.yml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["edge_model_version"] = edge_model_version
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

        isnc = tmp_path / "isnc"
        (isnc / ".git").mkdir(parents=True)
        return staging

    def test_the_first_shape_stamp_is_refused(self, tmp_path: Path) -> None:
        staging = self._staging(tmp_path, "v1")

        report = run_publish(staging, tmp_path / "isnc", dry_run=True)

        assert any("edge_model_version" in error for error in report.errors)

    def test_an_older_exporters_tree_is_refused_by_its_stamp(
        self, tmp_path: Path
    ) -> None:
        """The stale stamp is the message, not the fields the shape moved.

        This is the tree an operator actually brings: cut by an exporter that
        predates the sidecar, so its manifest carries the first stamp *and*
        none of the fields the installed loader now requires. Both faults are
        real, but only one is actionable, and the loader's dump names the
        entry files rather than the manifest.
        """
        staging = tmp_path / "staging"
        (staging / "standard_names").mkdir(parents=True)
        (staging / "standard_names" / "transport.yml").write_text(
            yaml.safe_dump(
                [
                    {
                        "name": "particle_flux",
                        "description": "Particle flux.",
                        "documentation": "Particle flux through a surface.",
                        "unit": "m^-2.s^-1",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (staging / "catalog.yml").write_text(
            yaml.safe_dump(
                {
                    "catalog_name": "imas-standard-names",
                    "edge_model_version": "v1",
                    "export_scope": "domain",
                    "domains_included": ["transport"],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "isnc" / ".git").mkdir(parents=True)

        report = run_publish(staging, tmp_path / "isnc", dry_run=True)

        assert len(report.errors) == 1
        refusal = report.errors[0]
        assert "edge_model_version" in refusal
        assert refusal.count("\n") == 0
        # The loader's manifest model would report one line per missing field.
        assert "Field required" not in refusal
        assert "structural" not in refusal

    def test_the_current_shape_stamp_passes_the_gate(self, tmp_path: Path) -> None:
        staging = self._staging(tmp_path, CATALOG_EDGE_MODEL_VERSION)

        report = run_publish(staging, tmp_path / "isnc", dry_run=True)

        # The whole run has to be clean, not merely free of a stamp complaint:
        # any earlier refusal returns before the gate is read, which would let
        # this pass without the gate ever having accepted anything.
        assert report.errors == []

    def test_the_exporter_stamps_what_the_gate_requires(self, tmp_path: Path) -> None:
        _, manifest = _export(tmp_path)

        assert manifest["edge_model_version"] == CATALOG_EDGE_MODEL_VERSION
        assert CATALOG_EDGE_MODEL_VERSION != "v1"
