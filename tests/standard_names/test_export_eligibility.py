"""Export eligibility of names whose only producers are derived.

A hierarchy parent carries no Data Dictionary source binding: its producers
are the derived children that were folded into it. Withholding it for that
topology alone published the children while their shared parent stayed
invisible to the catalog, so eligibility is decided by the review lifecycle
that governs every other name.
"""

from __future__ import annotations

from imas_codex.standard_names.export import _classify_export_population


def _derived_parent(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "m",
        "physics_domain": "general",
        "links": [],
        "source_paths": [f"derived:{name}_from_child"],
        "_has_dd_source_binding": False,
        "_has_derived_producer": True,
        "_has_non_derived_producer": False,
    }
    candidate.update(overrides)
    return candidate


def _classify(population: list[dict]):
    return _classify_export_population(population, domain=None, names_only=False)


def test_accepted_derived_parent_is_eligible() -> None:
    eligible, excluded = _classify([_derived_parent("radial_coordinate")])

    assert [row["id"] for row in eligible] == ["radial_coordinate"]
    assert excluded == []


def test_unaccepted_derived_parent_is_still_withheld() -> None:
    eligible, excluded = _classify(
        [_derived_parent("radial_coordinate", name_stage="drafted")]
    )

    assert eligible == []
    assert [(row.standard_name_id, row.reason) for row in excluded] == [
        ("radial_coordinate", "name_not_accepted")
    ]


def test_no_identity_is_withheld_for_lacking_a_source_binding() -> None:
    population = [
        _derived_parent("radial_coordinate"),
        _derived_parent("poloidal_angle", name_stage="drafted"),
    ]

    _, excluded = _classify(population)

    assert all(row.reason != "structural_parent" for row in excluded)
