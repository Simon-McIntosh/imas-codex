"""Registry conformance checks over fixtures and the frozen WEST roster.

The fixtures are written names whose equations, symbols and openings differ
from the conventions registry by exactly one property each, so a passing test
shows the check sees the property and a negative control shows it does not fire
on the canonical form.
"""

from __future__ import annotations

from collections import Counter

import pytest
from imas_standard_names.grammar.terms import (
    AppliedConventions,
    AppliedTemplate,
    select_conventions,
)

from imas_codex.standard_names.docs_gates import (
    HouseStyleFindingClass,
    conformance_findings,
    house_style_findings,
)
from imas_codex.standard_names.review.audits import (
    run_all_audits,
    run_house_style_checks,
)

_SURFACE_FORM = (
    r"$$\langle n\rangle = \oint n\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|$$"
)
_LINE_FORM = r"$$\langle n\rangle = \oint n\,dl/B \,/\, \oint dl/B$$"

_CARBON = "flux_surface_averaged_carbon_density_at_plasma_boundary"
_ARGON = "flux_surface_averaged_argon_density_at_plasma_boundary"
_BERYLLIUM = "flux_surface_averaged_beryllium_density_at_plasma_boundary"

_CANONICAL_OPENING = "Number density of argon ions on the last closed flux surface."


@pytest.fixture
def _roster_graph(disposable_neo4j_uri):
    from pathlib import Path

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.sources_manifest import load_names_file

    roster = Path(
        "imas_codex/standard_names/manifests/reviews/"
        "v0.4.0rc7+west-task-2e.sn_names.yaml"
    )
    names = load_names_file(roster)
    with GraphClient() as graph:
        graph.query(
            """
            UNWIND $names AS name
            MERGE (sn:StandardName {id: name})
            ON CREATE SET sn.test_roster_seed = true
            """,
            names=names,
        )
    yield
    with GraphClient() as graph:
        graph.query("MATCH (sn:StandardName {test_roster_seed: true}) DETACH DELETE sn")


def _documentation(equation: str, *, opening: str = _CANONICAL_OPENING) -> str:
    return (
        f"{opening}\n\n"
        "The flux-surface average of the local argon number density.\n\n"
        f"{equation}\n\n"
        "See [electron_density](name:electron_density) for the related quantity."
    )


def test_line_form_equation_is_a_forbidden_variant() -> None:
    for name in (_CARBON, _ARGON):
        findings = conformance_findings(name, documentation=_documentation(_LINE_FORM))
        assert HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT in {
            finding.finding_class for finding in findings
        }, name


def test_surface_form_equation_is_not_flagged() -> None:
    findings = conformance_findings(
        _BERYLLIUM, documentation=_documentation(_SURFACE_FORM)
    )
    assert HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT not in {
        finding.finding_class for finding in findings
    }


def test_separatrix_gloss_is_flagged_for_a_boundary_name() -> None:
    findings = conformance_findings(
        _BERYLLIUM,
        description="Beryllium density on the separatrix.",
        documentation=_documentation(_SURFACE_FORM),
    )
    assert HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS in {
        finding.finding_class for finding in findings
    }


def test_forbidden_symbol_spelling_is_flagged() -> None:
    documentation = _documentation(_SURFACE_FORM).replace(
        "The flux-surface average of the local argon number density.",
        "The flux-surface average of the local argon density $N_s$.",
    )
    findings = conformance_findings(_ARGON, documentation=documentation)
    assert HouseStyleFindingClass.FORBIDDEN_SYMBOL_SPELLING in {
        finding.finding_class for finding in findings
    }


def test_density_documentation_allows_rho_symbol() -> None:
    documentation = _documentation(_SURFACE_FORM).replace(
        "The flux-surface average of the local argon number density.",
        r"The flux-surface average of the local argon density $\rho$.",
    )
    findings = conformance_findings(_ARGON, documentation=documentation)
    assert HouseStyleFindingClass.FORBIDDEN_SYMBOL_SPELLING not in {
        finding.finding_class for finding in findings
    }


def test_off_template_opening_is_flagged() -> None:
    findings = conformance_findings(
        _ARGON,
        documentation=_documentation(
            _SURFACE_FORM, opening="The argon density at the boundary."
        ),
    )
    assert HouseStyleFindingClass.TEMPLATE_OPENING in {
        finding.finding_class for finding in findings
    }


def test_canonical_entry_yields_zero_findings() -> None:
    findings = conformance_findings(_ARGON, documentation=_documentation(_SURFACE_FORM))
    assert findings == []


def test_findings_name_their_registry_entry() -> None:
    findings = conformance_findings(_ARGON, documentation=_documentation(_LINE_FORM))
    assert findings
    for finding in findings:
        assert finding.finding_class in HouseStyleFindingClass
        assert finding.registry_entry.startswith(("concept:", "notation:", "template:"))


def test_audit_report_carries_the_conformance_findings() -> None:
    names = [
        {
            "id": _CARBON,
            "description": "Flux-surface-averaged carbon density.",
            "documentation": _documentation(_LINE_FORM),
        }
    ]
    report = run_all_audits(names)
    violations = [
        finding
        for finding in report.lint_findings
        if finding.finding_type == "convention_violation"
    ]
    assert any("forbidden_equation_variant" in f.detail for f in violations)


def test_house_style_findings_read_the_selected_entries() -> None:
    conventions = select_conventions(_ARGON)
    findings = house_style_findings(
        conventions, documentation=_documentation(_LINE_FORM)
    )
    assert findings
    assert all(isinstance(f.finding_class, HouseStyleFindingClass) for f in findings)


@pytest.mark.graph
@pytest.mark.timeout(600)
def test_live_roster_census_records_per_class_counts(_roster_graph) -> None:
    from pathlib import Path

    from imas_standard_names.grammar.terms import load_conventions

    from imas_codex.standard_names.sources_manifest import load_names_file

    roster = Path(
        "imas_codex/standard_names/manifests/reviews/"
        "v0.4.0rc7+west-task-2e.sn_names.yaml"
    )
    names = load_names_file(roster)
    assert len(names) == 220

    rows = _live_name_rows(names)
    assert len(rows) == 220, f"roster resolution saw {len(rows)} of 220 names"
    registry = load_conventions()
    counts: Counter[str] = Counter()
    no_entry = 0
    parse_failures = 0
    for row in rows:
        try:
            conventions = select_conventions(row["id"], registry=registry)
        except Exception:
            parse_failures += 1
            continue
        if not (conventions.concepts or conventions.notation or conventions.templates):
            no_entry += 1
            continue
        for finding in house_style_findings(
            conventions,
            description=row.get("description") or "",
            documentation=row.get("documentation") or "",
        ):
            counts[finding.finding_class.value] += 1
    print(
        "census",
        dict(counts),
        "no_entry",
        no_entry,
        "parse_failures",
        parse_failures,
    )
    assert set(counts) <= {member.value for member in HouseStyleFindingClass}
    assert no_entry + parse_failures < len(rows), "selection must reach some names"


def _live_name_rows(names: list[str]) -> list[dict]:
    from imas_codex.graph.client import GraphClient

    rows: list[dict] = []
    with GraphClient() as gc:
        for start in range(0, len(names), 50):
            batch = names[start : start + 50]
            rows.extend(
                gc.query(
                    "MATCH (sn:StandardName) WHERE sn.id IN $ids "
                    "RETURN sn.id AS id, sn.description AS description, "
                    "sn.documentation AS documentation",
                    ids=batch,
                )
            )
    return rows


def test_audit_checks_skip_unparseable_names() -> None:
    assert run_house_style_checks([{"id": "not_a_standard_name!!"}]) == []


def _two_density_templates() -> tuple[AppliedTemplate, AppliedTemplate]:
    return (
        AppliedTemplate(
            base="density",
            segments=("subject",),
            description="",
            documentation="Number density of {subject}",
        ),
        AppliedTemplate(
            base="density",
            segments=("subject",),
            description="",
            documentation="The number density of {subject}",
        ),
    )


def test_opening_matching_one_of_two_templates_is_not_flagged() -> None:
    findings = house_style_findings(
        AppliedConventions(templates=_two_density_templates()),
        documentation=_CANONICAL_OPENING,
    )
    template_findings = [
        finding
        for finding in findings
        if finding.finding_class is HouseStyleFindingClass.TEMPLATE_OPENING
    ]
    assert template_findings == []


def test_opening_matching_no_template_is_flagged_once() -> None:
    findings = house_style_findings(
        AppliedConventions(templates=_two_density_templates()),
        documentation="The argon density at the boundary.",
    )
    template_findings = [
        finding
        for finding in findings
        if finding.finding_class is HouseStyleFindingClass.TEMPLATE_OPENING
    ]
    assert len(template_findings) == 1


_NOTE_NAME = "total_thermal_electron_density"
_NOTE_OPENING = "Number density of electron"


def _selected_template_note(name: str) -> str:
    """The note sentence the grammar selects for *name*."""
    from imas_standard_names.grammar.terms import (
        select_conventions,
        template_sentences,
    )

    openings = {
        template.documentation for template in select_conventions(name).templates
    }
    notes = [s for s in template_sentences(name) if s not in openings]
    assert notes, f"{name} must select a note for this check"
    return notes[0]


def _note_template_findings(name: str, note: str | None) -> list:
    opening = f"{_NOTE_OPENING}."
    documentation = opening if note is None else f"{opening}\n\n{note}"
    return [
        finding
        for finding in conformance_findings(name, documentation=documentation)
        if finding.finding_class is HouseStyleFindingClass.TEMPLATE_NOTE
    ]


def test_missing_template_note_is_flagged() -> None:
    findings = _note_template_findings(_NOTE_NAME, None)
    assert len(findings) == 1
    assert findings[0].registry_entry == "template:density"


def test_template_note_present_is_not_flagged() -> None:
    assert (
        _note_template_findings(_NOTE_NAME, _selected_template_note(_NOTE_NAME)) == []
    )


_SIGN_SENSITIVE_NAME = "poloidal_magnetic_flux"


def _flux_documentation(*, sign_convention: bool) -> str:
    sign_paragraph = (
        "\n\nSign convention: Positive when the poloidal flux increases "
        "outward from the magnetic axis."
    )
    return (
        "Poloidal magnetic flux through a surface spanning the magnetic axis."
        "\n\nThe poloidal magnetic flux $\\psi_p$ is the magnetic flux across a "
        "poloidal cross-section, defined by the surface integral of the "
        "poloidal field."
        "\n\nSee [magnetic_field_poloidal](name:magnetic_field_poloidal) for "
        "the related field." + (sign_paragraph if sign_convention else "")
    )


def _flux_name(*, sign_convention: bool) -> dict:
    return {
        "id": _SIGN_SENSITIVE_NAME,
        "description": "Poloidal magnetic flux through a surface.",
        "documentation": _flux_documentation(sign_convention=sign_convention),
        "cocos_transformation_type": "psi_like",
        "unit": "Wb",
    }


def _gate_finding_types(report) -> set[str]:
    return {finding.finding_type for finding in report.lint_findings}


def test_audit_report_carries_the_sign_convention_gate() -> None:
    report = run_all_audits([_flux_name(sign_convention=False)])
    assert "sign_convention" in _gate_finding_types(report)


def test_sign_convention_gate_is_silent_when_the_paragraph_is_present() -> None:
    report = run_all_audits([_flux_name(sign_convention=True)])
    assert "sign_convention" not in _gate_finding_types(report)


def test_fixture_census_counts_each_class_once() -> None:
    written = [
        {"name": _CARBON, "documentation": _documentation(_LINE_FORM)},
        {
            "name": _BERYLLIUM,
            "description": "Beryllium density on the separatrix.",
            "documentation": _documentation(_SURFACE_FORM),
        },
        {
            "name": _ARGON,
            "documentation": _documentation(
                _SURFACE_FORM, opening="The argon density at the boundary."
            ),
        },
        {"name": _ARGON, "documentation": _documentation(_SURFACE_FORM)},
    ]
    counts: Counter[str] = Counter()
    for entry in written:
        for finding in conformance_findings(
            entry["name"],
            description=entry.get("description", ""),
            documentation=entry["documentation"],
        ):
            counts[finding.finding_class.value] += 1
    assert counts == Counter(
        {
            "forbidden_equation_variant": 1,
            "forbidden_locus_gloss": 1,
            "template_opening": 1,
        }
    )
