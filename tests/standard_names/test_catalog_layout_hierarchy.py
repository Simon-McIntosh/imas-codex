"""Per-domain catalog layout with graph-hierarchy ordering.

Covered behaviours:
1. Round-trip byte stability
2. Round-trip idempotence
3. Ordering — unary prefix
4. Ordering — projection
5. Ordering — binary
6. Ordering — uncertainty
7. Ordering — mixed
8. Ordering — orphan (cross-domain)
9. Stability under cluster reassignment
10. Stability under Neo4j property permutation
11. Computed-field ignored on import + INFO log
12. Partial-export publish safety + manifest mismatch abort
13. check_catalog + list-root parity
14. Legacy per-file rejection
15. Edge-model version guard
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.standard_names.export import _write_manifest


def _stage_manifest(
    staging: Path,
    name: str,
    domain: str,
    *,
    export_scope: str,
    domains_included: list[str],
    edge_model_version: str | None = None,
) -> None:
    """Write the manifest the exporter emits, overriding the field under test.

    Publish's own gates run behind the installed loader's manifest validation,
    which refuses a sidecar missing any required field. A hand-built partial
    manifest therefore never reaches the gate it was written to exercise, so
    the fixture goes through the writer and overrides only what it asserts on.
    """
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
        export_scope=export_scope,
        domains_included=domains_included,
        names={
            name: {
                "kind": "scalar",
                "status": "active",
                "physics_domain": domain,
                "links": [],
                "sources": [],
            }
        },
    )
    if edge_model_version is None:
        return

    # The writer always stamps the current shape; a stale stamp has to be
    # substituted afterwards to reach the compatibility gate.
    manifest_path = staging / "catalog.yml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["edge_model_version"] = edge_model_version
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _stage_entry(name: str) -> dict[str, str]:
    """One reduced-surface entry: identity, prose and the unit that frames it."""
    prose = name.replace("_", " ").capitalize()
    return {
        "name": name,
        "description": f"{prose}.",
        "documentation": f"{prose} of the thermal plasma.",
        "unit": "m^-2.s^-1",
    }


# ============================================================================
# Test 3: Ordering — unary prefix
# ============================================================================


class TestOrderingUnaryPrefix:
    """Unary prefix: base first, then alpha-sorted wrappers."""

    def test_unary_prefix_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "minimum_of_temperature"},
            {"name": "maximum_of_temperature"},
            {"name": "temperature"},
        ]
        # HAS_PARENT: wrapper -> base (base is ordering-parent)
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_PARENT"),
            ("minimum_of_temperature", "temperature", "HAS_PARENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result.entries]
        assert names == [
            "temperature",
            "maximum_of_temperature",
            "minimum_of_temperature",
        ]


# ============================================================================
# Test 4: Ordering — projection
# ============================================================================


class TestOrderingProjection:
    """Projection: base first, then alpha-sorted components."""

    def test_projection_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "z_magnetic_field"},
            {"name": "magnetic_field"},
            {"name": "x_magnetic_field"},
            {"name": "y_magnetic_field"},
        ]
        edges = [
            ("x_magnetic_field", "magnetic_field", "HAS_PARENT"),
            ("y_magnetic_field", "magnetic_field", "HAS_PARENT"),
            ("z_magnetic_field", "magnetic_field", "HAS_PARENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result.entries]
        assert names == [
            "magnetic_field",
            "x_magnetic_field",
            "y_magnetic_field",
            "z_magnetic_field",
        ]


# ============================================================================
# Test 5: Ordering — binary
# ============================================================================


class TestOrderingBinary:
    """Binary: both args as clean roots, ratio last."""

    def test_binary_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "ratio_of_pressure_to_density"},
            {"name": "pressure"},
            {"name": "density"},
        ]
        # Binary has two HAS_PARENT edges
        edges = [
            ("ratio_of_pressure_to_density", "pressure", "HAS_PARENT"),
            ("ratio_of_pressure_to_density", "density", "HAS_PARENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result.entries]
        # Alpha tie-break: density before pressure
        assert names == [
            "density",
            "pressure",
            "ratio_of_pressure_to_density",
        ]


# ============================================================================
# Test 6: Ordering — uncertainty
# ============================================================================


class TestOrderingUncertainty:
    """Uncertainty: base first, then alpha-sorted error siblings."""

    def test_uncertainty_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "upper_uncertainty_of_temperature"},
            {"name": "temperature"},
            {"name": "uncertainty_index_of_temperature"},
            {"name": "lower_uncertainty_of_temperature"},
        ]
        # HAS_ERROR: base -> error sibling (base is ordering-parent)
        edges = [
            ("temperature", "upper_uncertainty_of_temperature", "HAS_ERROR"),
            ("temperature", "lower_uncertainty_of_temperature", "HAS_ERROR"),
            ("temperature", "uncertainty_index_of_temperature", "HAS_ERROR"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result.entries]
        assert names == [
            "temperature",
            "lower_uncertainty_of_temperature",
            "uncertainty_index_of_temperature",
            "upper_uncertainty_of_temperature",
        ]


# ============================================================================
# Test 7: Ordering — mixed
# ============================================================================


class TestOrderingMixed:
    """Mixed: base first, then alpha-sorted variants + components."""

    def test_mixed_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "upper_uncertainty_of_temperature"},
            {"name": "x_temperature"},
            {"name": "temperature"},
            {"name": "maximum_of_temperature"},
        ]
        edges = [
            ("temperature", "upper_uncertainty_of_temperature", "HAS_ERROR"),
            ("x_temperature", "temperature", "HAS_PARENT"),
            ("maximum_of_temperature", "temperature", "HAS_PARENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result.entries]

        # temperature first (sole root), then all children alpha-sorted
        assert names[0] == "temperature"
        assert set(names[1:]) == {
            "maximum_of_temperature",
            "upper_uncertainty_of_temperature",
            "x_temperature",
        }
        # Alpha-sorted among children
        assert names[1:] == sorted(names[1:])


# ============================================================================
# Test 8: Ordering — orphan (cross-domain)
# ============================================================================


class TestOrderingOrphan:
    """Orphan: cross-domain wrapper lands after all clean-roots."""

    def test_orphan_after_clean_roots(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "alpha_base"},
            {"name": "cross_domain_wrapper"},
            {"name": "beta_base"},
        ]
        # No in-domain edges, but cross_domain_wrapper has a parent outside
        edges: list[tuple[str, str, str]] = []
        cross_domain = {"cross_domain_wrapper"}

        result = order_entries_by_hierarchy(
            entries, edges, cross_domain_parent_ids=cross_domain
        )
        names = [e["name"] for e in result.entries]

        # Clean roots first (alpha_base, beta_base), then orphan
        assert names == ["alpha_base", "beta_base", "cross_domain_wrapper"]


# ============================================================================
# Test 9: Stability under cluster reassignment
# ============================================================================


class TestStabilityClusterReassignment:
    """Ordering is stable when primary_cluster_id changes."""

    def test_cluster_reassignment_no_effect(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries_v1 = [
            {"name": "temperature", "primary_cluster_id": "cluster_A"},
            {"name": "maximum_of_temperature", "primary_cluster_id": "cluster_A"},
        ]
        entries_v2 = [
            {"name": "temperature", "primary_cluster_id": "cluster_B"},
            {"name": "maximum_of_temperature", "primary_cluster_id": "cluster_B"},
        ]
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_PARENT"),
        ]

        result_v1 = [
            e["name"] for e in order_entries_by_hierarchy(entries_v1, edges).entries
        ]
        result_v2 = [
            e["name"] for e in order_entries_by_hierarchy(entries_v2, edges).entries
        ]

        assert result_v1 == result_v2


# ============================================================================
# Test 10: Stability under Neo4j property permutation
# ============================================================================


class TestStabilityPropertyPermutation:
    """Ordering is stable regardless of dict key order."""

    def test_property_permutation_no_effect(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries_v1 = [
            {"name": "temperature", "kind": "scalar", "unit": "eV"},
            {"name": "maximum_of_temperature", "kind": "scalar", "unit": "eV"},
        ]
        # Same data, different insertion order
        entries_v2 = [
            {"unit": "eV", "name": "temperature", "kind": "scalar"},
            {"unit": "eV", "kind": "scalar", "name": "maximum_of_temperature"},
        ]
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_PARENT"),
        ]

        result_v1 = [
            e["name"] for e in order_entries_by_hierarchy(entries_v1, edges).entries
        ]
        result_v2 = [
            e["name"] for e in order_entries_by_hierarchy(entries_v2, edges).entries
        ]

        assert result_v1 == result_v2


# ============================================================================
# Test 1: Round-trip byte stability
# ============================================================================


class TestRoundTripByteStability:
    """Export entries, parse, re-emit → byte-identical."""

    def test_byte_stable_round_trip(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.export import _write_domain_yaml

        entries = [
            {
                "name": "temperature",
                "kind": "scalar",
                "status": "draft",
                "description": "Temperature profile",
                "documentation": "A temperature measurement.",
                "unit": "eV",
                "physics_domain": "core_plasma_physics",
                "links": [],
            },
        ]

        # Write domain file; the sidecar carries the machine-derived fields
        # the reviewable entry no longer does.
        metadata: dict[str, dict] = {}
        _write_domain_yaml(tmp_path, "kinetics", entries, name_metadata=metadata)

        filepath = tmp_path / "standard_names" / "kinetics.yml"
        assert filepath.exists()
        text = filepath.read_text(encoding="utf-8")

        # Parse back and overlay the sidecar, the same resolved shape a
        # consumer folds back into the writer, then re-emit through the same
        # writer used the first time.
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, list)
        resolved = [{**entry, **metadata[entry["name"]]} for entry in parsed]

        second_metadata: dict[str, dict] = {}
        second_path = _write_domain_yaml(
            tmp_path / "second", "kinetics", resolved, name_metadata=second_metadata
        )

        assert second_path.read_text(encoding="utf-8") == text


# ============================================================================
# Test 2: Round-trip idempotence (mock-based)
# ============================================================================


class TestRoundTripIdempotence:
    """Export → parse → re-emit yields identical entries."""

    def test_idempotent_re_emit(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.canonical import (
            canonicalise_entry,
            reorder_entry_dict,
        )
        from imas_codex.standard_names.export import _write_domain_yaml

        entries = [
            {
                "name": "electron_temperature",
                "kind": "scalar",
                "status": "draft",
                "description": "Electron temperature",
                "documentation": "Te from Thomson scattering.",
                "unit": "eV",
                "physics_domain": "core_plasma_physics",
                "links": ["name:ion_temperature"],
                "constraints": ["T_e > 0"],
                "validity_domain": "core plasma",
            },
        ]

        # First write
        metadata: dict[str, dict] = {}
        _write_domain_yaml(tmp_path, "kinetics", entries, name_metadata=metadata)
        fp = tmp_path / "standard_names" / "kinetics.yml"
        first_bytes = fp.read_bytes()

        # Parse back, overlay the sidecar the entry no longer carries, and
        # re-write.
        parsed = yaml.safe_load(fp.read_text(encoding="utf-8"))
        resolved = [{**entry, **metadata[entry["name"]]} for entry in parsed]

        _write_domain_yaml(tmp_path, "kinetics", resolved, name_metadata=metadata)
        second_bytes = fp.read_bytes()

        assert first_bytes == second_bytes


# ============================================================================
# Test 12: Partial-export publish safety + manifest mismatch abort
# ============================================================================


class TestPartialExportPublishSafety:
    """Publish aborts on manifest mismatch."""

    def test_manifest_domain_mismatch_aborts(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()

        # Write one domain file
        (sn_dir / "transport.yml").write_text(
            yaml.safe_dump([_stage_entry("particle_flux")])
        )

        # Manifest claims full scope but only one domain
        _stage_manifest(
            staging,
            "particle_flux",
            "transport",
            export_scope="full",
            domains_included=["transport", "magnetics"],
        )

        # Create a fake ISNC git repo
        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()

        report = run_publish(staging, isnc)
        assert report.errors
        assert any("domain mismatch" in e.lower() for e in report.errors)

    def test_domain_scoped_publish_only_touches_listed(self, tmp_path: Path) -> None:
        """Domain-subset publish only copies listed domain files."""
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()

        (sn_dir / "transport.yml").write_text(
            yaml.safe_dump([_stage_entry("particle_flux")])
        )

        _stage_manifest(
            staging,
            "particle_flux",
            "transport",
            export_scope="domain",
            domains_included=["transport"],
        )

        # Set up ISNC with existing domain files
        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()
        isnc_sn = isnc / "standard_names"
        isnc_sn.mkdir()
        (isnc_sn / "magnetics.yml").write_text("- name: b_field\n")

        # Mock git operations
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_publish(staging, isnc, dry_run=True)

        # magnetics.yml should still exist (not touched)
        assert (isnc_sn / "magnetics.yml").exists()


# ============================================================================
# Test 13: check_catalog + list-root parity
# ============================================================================


class TestCheckCatalogListRoot:
    """check_catalog handles list-root files correctly."""

    def test_check_catalog_list_root(self, tmp_path: Path) -> None:
        pytest.importorskip("imas_standard_names")

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        sn_dir = catalog_dir / "standard_names"
        sn_dir.mkdir()

        entries = [
            {
                "name": "temperature",
                "description": "Temperature",
                "documentation": "A temperature measurement.",
                "kind": "scalar",
                "unit": "eV",
                "status": "active",
                "links": [],
            },
        ]
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump(entries))

        # The reviewable entry no longer carries physics_domain; the manifest
        # sidecar's per-name mapping supplies it before model validation.
        manifest = {"names": {"temperature": {"physics_domain": "core_plasma_physics"}}}
        (catalog_dir / "catalog.yml").write_text(yaml.safe_dump(manifest))

        from imas_codex.standard_names.catalog_import import check_catalog

        # GraphClient is imported inside check_catalog — patch at source module.
        with patch("imas_codex.graph.client.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            mock_gc.query = MagicMock(return_value=[])
            mock_gc_cls.return_value = mock_gc

            result = check_catalog(catalog_dir)

        # Should have parsed the entry
        assert result.only_in_catalog == ["temperature"]


# ============================================================================
# Test 15: Edge-model version guard
# ============================================================================


class TestEdgeModelVersionGuard:
    """Manifest with wrong edge_model_version rejected by publish."""

    def test_wrong_version_rejected(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()
        (sn_dir / "core_plasma_physics.yml").write_text(
            yaml.safe_dump([_stage_entry("electron_temperature")])
        )

        _stage_manifest(
            staging,
            "electron_temperature",
            "core_plasma_physics",
            export_scope="full",
            domains_included=["core_plasma_physics"],
            edge_model_version="v0",
        )

        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()

        report = run_publish(staging, isnc)
        assert report.errors
        assert any("edge_model_version" in e for e in report.errors)


# ============================================================================
# Canonical key order tests
# ============================================================================


class TestCanonicalKeyOrder:
    """CANONICAL_KEY_ORDER and reorder_entry_dict."""

    def test_reorder_known_keys(self) -> None:
        from imas_codex.standard_names.canonical import reorder_entry_dict

        entry = {
            "unit": "eV",
            "name": "temperature",
            "kind": "scalar",
            "status": "draft",
        }
        result = reorder_entry_dict(entry)
        assert list(result.keys()) == ["name", "kind", "status", "unit"]

    def test_unknown_key_raises(self) -> None:
        from imas_codex.standard_names.canonical import (
            UnknownCatalogKeyError,
            reorder_entry_dict,
        )

        entry = {"name": "temperature", "bogus_key": "value"}
        with pytest.raises(UnknownCatalogKeyError, match="bogus_key"):
            reorder_entry_dict(entry)

    def test_missing_keys_omitted(self) -> None:
        from imas_codex.standard_names.canonical import reorder_entry_dict

        entry = {"name": "temperature"}
        result = reorder_entry_dict(entry)
        assert result == {"name": "temperature"}


# ============================================================================
# Ordering error detection
# ============================================================================


class TestOrderingCycleDetection:
    """Cycle participants are named and withheld without losing acyclic rows."""

    def test_two_node_cycle_withholds_only_cycle_participants(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "a"},
            {"name": "b"},
            {"name": "child_of_cycle"},
            {"name": "independent"},
        ]
        edges = [
            ("a", "b", "HAS_PARENT"),
            ("b", "a", "HAS_PARENT"),
            ("child_of_cycle", "a", "HAS_PARENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)

        assert [entry["name"] for entry in result.entries] == [
            "child_of_cycle",
            "independent",
        ]
        assert [exclusion.name for exclusion in result.exclusions] == ["a", "b"]
        assert all(
            exclusion.relationships
            == (
                ("a", "b", "HAS_PARENT"),
                ("b", "a", "HAS_PARENT"),
            )
            for exclusion in result.exclusions
        )

    def test_export_ledgers_cycle_and_writes_packet(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.export import run_export

        population = [
            {
                "id": name,
                "name_stage": "accepted",
                "validation_status": "valid",
                "review_quorum_shortfall": None,
                "docs_stage": "accepted",
                "docs_review_quorum_shortfall": None,
                "reviewer_score_name": 0.95,
                "description": f"Description for {name}.",
                "documentation": f"Documentation for {name}.",
                "kind": "scalar",
                "unit": "1",
                "physics_domain": "general",
                "links": [],
            }
            for name in ("a", "b", "independent")
        ]

        class EmptyGraphClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def query(self, cypher: str, **params):
                return []

        with (
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=population,
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=EmptyGraphClient(),
            ),
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=lambda entry: entry,
            ),
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=(
                    [("a", "b", "HAS_PARENT"), ("b", "a", "HAS_PARENT")],
                    set(),
                ),
            ),
            patch("imas_codex.standard_names.export._write_domain_yaml") as write_yaml,
        ):
            report = run_export(
                tmp_path,
                skip_gate=True,
                force=True,
                include_sources=False,
            )

        assert report.exported_names == ["independent"]
        assert report.exported_count == 1
        cycle_records = [
            record
            for record in report.exclusion_records
            if record.reason == "hierarchy_ordering_cycle"
        ]
        assert [record.standard_name_id for record in cycle_records] == ["a", "b"]
        assert all("a -[HAS_PARENT]-> b" in record.detail for record in cycle_records)
        assert all("b -[HAS_PARENT]-> a" in record.detail for record in cycle_records)
        accounting_gate = next(
            gate for gate in report.gate_results if gate.gate == "exclusion_accounting"
        )
        assert accounting_gate.passed
        assert (tmp_path / "catalog.yml").is_file()
        assert (tmp_path / ".export_report.json").is_file()
        written_entries = write_yaml.call_args.args[2]
        assert [entry["name"] for entry in written_entries] == ["independent"]
