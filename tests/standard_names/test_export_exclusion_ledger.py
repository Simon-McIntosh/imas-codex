"""Identity-bearing exclusion accounting for catalog export."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

from imas_codex.standard_names.export import _entry_model, run_export


class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str, **overrides) -> dict:
    candidate = {
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
    candidate.update(overrides)
    return candidate


def _run_fixture_export(
    staging_dir: Path,
    population: list[dict],
    *,
    validate_entries: bool = False,
    include_sources: bool = False,
    source_bindings: list[dict] | None = None,
    write_domain_yaml: bool = False,
):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=population,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_ReadOnlyGraphClient(),
            )
        )
        if not validate_entries:
            stack.enter_context(
                patch(
                    "imas_codex.standard_names.export._validate_entry",
                    side_effect=lambda entry: entry,
                )
            )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_deprecation_stubs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            )
        )
        if source_bindings is not None:
            stack.enter_context(
                patch(
                    "imas_codex.standard_names.export._fetch_sources_for_entry",
                    return_value=source_bindings,
                )
            )
        if not write_domain_yaml:
            stack.enter_context(
                patch("imas_codex.standard_names.export._write_domain_yaml")
            )
        return run_export(
            staging_dir,
            skip_gate=True,
            force=True,
            include_sources=include_sources,
        )


def test_export_ledger_closes_over_fixture_population(tmp_path: Path) -> None:
    population = [
        _candidate("emitted_name"),
        _candidate("invalid_name", validation_status="quarantined"),
        _candidate("docs_pending_name", docs_stage="pending"),
        _candidate("unreviewed_name", reviewer_score_name=None),
    ]

    report = _run_fixture_export(tmp_path, population)
    payload = report.to_dict()
    persisted_payload = json.loads(
        (tmp_path / ".export_report.json").read_text(encoding="utf-8")
    )
    rows = {row["reason"]: row for row in payload["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == 4
    assert report.exported_count == 1
    assert report.exported_names == ["emitted_name"]
    assert payload["emitted_identities"] == report.exported_names
    assert persisted_payload["emitted_identities"] == report.exported_names
    assert {reason: row["count"] for reason, row in rows.items()} == {
        "documentation_not_accepted": 1,
        "invalid_validation_status": 1,
        "unreviewed_name": 1,
    }
    assert rows["documentation_not_accepted"]["identities"] == ["docs_pending_name"]
    assert rows["invalid_validation_status"]["identities"] == ["invalid_name"]
    assert rows["unreviewed_name"]["identities"] == ["unreviewed_name"]
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 4


def test_export_emits_generic_source_bindings_and_preserves_accounting(
    tmp_path: Path,
) -> None:
    population = [
        _candidate("emitted_name", physics_domain="equilibrium"),
        _candidate("invalid_name", validation_status="quarantined"),
    ]
    copied_dd_content = {
        "dd_path": "equilibrium/time_slice/profiles_1d/psi",
        "dd_version": "4.1.0",
        "dd_documentation_url": "https://example.invalid/dd/psi",
        "dd_documentation": {
            "leaf": "Poloidal magnetic flux.",
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "parent": "One-dimensional equilibrium profiles.",
            "data_type": "FLT_1D",
            "unit": "Wb",
            "coordinates": ["equilibrium/time_slice/profiles_1d/rho_tor_norm"],
        },
        "enhanced_context": {
            "description": "Generated explanatory context.",
            "kind": "llm",
        },
        "semantic_facet": "reconstructed",
    }

    report = _run_fixture_export(
        tmp_path,
        population,
        validate_entries=True,
        include_sources=True,
        source_bindings=[
            copied_dd_content,
            {
                "signal_id": "west:magnetics/ip",
                "version": "62253",
                "semantic_facet": "measured",
            },
        ],
        write_domain_yaml=True,
    )

    emitted = yaml.safe_load(
        (tmp_path / "standard_names" / "equilibrium.yml").read_text(encoding="utf-8")
    )[0]
    assert emitted["sources"] == [
        {
            "kind": "imas-dd",
            "ref": "equilibrium/time_slice/profiles_1d/psi",
            "version": "4.1.0",
        },
        {
            "kind": "west",
            "ref": "west:magnetics/ip",
            "version": "62253",
        },
    ]
    assert all(
        set(binding) == {"kind", "ref", "version"} for binding in emitted["sources"]
    )
    assert emitted["sources"][1]["kind"] != "imas-dd"
    assert not {
        "dd_path",
        "dd_version",
        "signal_id",
        "semantic_facet",
    }.intersection(key for binding in emitted["sources"] for key in binding)

    strict_projection = {
        key: value for key, value in emitted.items() if key != "physics_domain"
    }
    strict_entry = _entry_model(strict_projection)
    assert strict_entry.name == "emitted_name"
    assert strict_entry.model_dump(mode="json")["sources"] == emitted["sources"]

    rows = {row.reason: row for row in report.exclusion_records}
    assert report.exported_count == 1
    assert rows["invalid_validation_status"].standard_name_id == "invalid_name"
    assert report.exported_count + len(report.exclusion_records) == len(population)


def test_export_refuses_when_ledger_does_not_close(tmp_path: Path) -> None:
    population = [_candidate("emitted_name"), _candidate("silently_dropped_name")]

    with patch(
        "imas_codex.standard_names.export._classify_export_population",
        return_value=([population[0]], []),
    ):
        report = _run_fixture_export(tmp_path, population)

    accounting_gate = next(
        gate for gate in report.gate_results if gate.gate == "exclusion_accounting"
    )
    assert not accounting_gate.passed
    assert not report.all_gates_passed
    assert any(
        issue["type"] == "unattributed_identity"
        and issue["identities"] == ["silently_dropped_name"]
        for issue in accounting_gate.issues
    )
    assert any(
        issue["type"] == "exclusion_accounting_mismatch"
        and issue["accepted_population"] == 2
        and issue["emitted"] == 1
        and issue["excluded"] == 0
        for issue in accounting_gate.issues
    )
    assert not (tmp_path / "catalog.yml").exists()


def test_export_withholds_source_free_structural_parent(tmp_path: Path) -> None:
    population = [
        _candidate(
            "radial_coordinate",
            origin=None,
            source_paths=["derived:radial_coordinate"],
            _has_dd_source_binding=False,
            _has_derived_producer=True,
            _has_non_derived_producer=False,
            unit="m",
        )
    ]

    report = _run_fixture_export(tmp_path, population, validate_entries=True)
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == 1
    assert report.exported_count == 0
    assert report.validation_failures == 0
    assert rows["structural_parent"]["identities"] == ["radial_coordinate"]
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 1


def test_export_withholds_hard_catalog_semantic_issue(tmp_path: Path) -> None:
    population = [
        _candidate(
            "radial_coordinate",
            origin=None,
            source_paths=["dd:equilibrium/time_slice/profiles_1d/rho_tor"],
            _has_dd_source_binding=True,
            _has_derived_producer=False,
            _has_non_derived_producer=True,
            unit="m",
        ),
    ]

    report = _run_fixture_export(tmp_path, population, validate_entries=True)
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == 1
    assert report.exported_count == 0
    assert rows["invalid_catalog_entry"]["identities"] == ["radial_coordinate"]
    assert "structural_parent" not in rows
    assert report.exported_count + sum(row["count"] for row in rows.values()) == 1


def test_export_validates_cross_links_against_full_catalog(tmp_path: Path) -> None:
    from imas_standard_names.validation import run_semantic_checks

    population = [
        _candidate(
            "electron_density",
            unit="m^-3",
            links=["name:ion_density"],
        ),
        _candidate(
            "ion_density",
            unit="m^-3",
            links=["name:electron_density"],
        ),
    ]

    with patch(
        "imas_standard_names.validation.run_semantic_checks",
        wraps=run_semantic_checks,
    ) as semantic_checks:
        report = _run_fixture_export(tmp_path, population, validate_entries=True)

    assert semantic_checks.call_count == 1
    assert set(semantic_checks.call_args.args[0]) == {
        "electron_density",
        "ion_density",
    }
    assert report.all_gates_passed
    assert report.total_candidates == 2
    assert report.exported_count == 2
    assert set(report.exported_names) == {"electron_density", "ion_density"}
    assert report.exclusion_records == []


def test_export_keeps_catalog_semantic_advisories(tmp_path: Path) -> None:
    population = [
        _candidate(
            "radial_coordinate_of_line_of_sight",
            origin="derived",
            source_paths=["dd:camera_ir/channel/line_of_sight/first_point/r"],
            _has_dd_source_binding=True,
            _has_derived_producer=False,
            _has_non_derived_producer=True,
            unit="m",
        )
    ]

    with patch(
        "imas_standard_names.validation.run_semantic_checks",
        return_value=[
            "radial_coordinate_of_line_of_sight: WARNING - advisory warning",
            "radial_coordinate_of_line_of_sight: INFO - advisory information",
        ],
    ):
        report = _run_fixture_export(tmp_path, population, validate_entries=True)

    assert report.all_gates_passed
    assert report.exported_names == ["radial_coordinate_of_line_of_sight"]
    assert report.exclusion_records == []
