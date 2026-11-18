"""Names-mode composition renders the enclosing-structure lineage.

The extraction stage attaches ``ancestor_context`` — the statements of the DD
structures a leaf sits under — to every item it emits, and the full-mode
template has always rendered that block. Names mode rendered none of it, so a
component, geometric reference or locus stated only by the enclosing structure
never reached the composer that writes the identity: ``current_of_passive_
structure`` lost the ``axisymmetric`` the enclosing ``pf_passive_current``
statement states, and ``elongation_of_plasma_boundary`` never saw the enclosing
``equilibrium/time_slice/boundary`` statement that lets the boundary be chosen.

These tests render the two templates over the same items and require the same
lineage, so names mode cannot fall behind full mode again.
"""

from __future__ import annotations

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.context import build_compose_context

#: An equilibrium member: the leaf states the geometric property, and the
#: enclosing ``equilibrium/time_slice/boundary`` structure is the only place
#: that states the boundary is selectable.
_BOUNDARY_LEAF = "equilibrium/time_slice/boundary/minor_radius"
_BOUNDARY_STATEMENT = (
    "Description of the plasma boundary used by fixed-boundary codes and "
    "typically chosen at psi_norm = 99.x% of the separatrix"
)

#: A second member: the enclosing passive-structure statement states the
#: conductor scope (``axisymmetric``) that the leaf description drops.
_PASSIVE_LEAF = "equilibrium/time_slice/constraints/pf_passive_current"
_PASSIVE_STATEMENT = (
    "Current in a set of axisymmetric passive conductors used as a constraint "
    "in MHD equilibrium reconstruction"
)


def _item(path: str, statement: str | None) -> dict:
    item = {
        "path": path,
        "description": "Minor radius of the plasma boundary",
        "unit": "m",
    }
    if statement is not None:
        item["ancestor_context"] = [{"path": path.rsplit("/", 1)[0], "text": statement}]
    return item


def _render(template: str, items: list[dict]) -> str:
    return render_prompt(
        template,
        {**build_compose_context(), "items": items, "ids_name": "equilibrium"},
    )


def test_names_mode_renders_the_enclosing_statement() -> None:
    rendered = _render(
        "sn/generate_name_dd_names", [_item(_BOUNDARY_LEAF, _BOUNDARY_STATEMENT)]
    )

    assert "DD path lineage" in rendered
    assert "`equilibrium/time_slice/boundary`:" in rendered
    assert "psi_norm = 99.x% of the separatrix" in rendered


def test_names_mode_renders_lineage_for_every_item_that_carries_it() -> None:
    items = [
        _item(_BOUNDARY_LEAF, _BOUNDARY_STATEMENT),
        _item(_PASSIVE_LEAF, _PASSIVE_STATEMENT),
    ]
    rendered = _render("sn/generate_name_dd_names", items)

    for item in items:
        for ancestor in item["ancestor_context"]:
            assert ancestor["text"] in rendered
            assert ancestor["path"] in rendered


def test_names_mode_agrees_with_full_mode_on_the_enclosing_lineage() -> None:
    """Both templates must carry the same enclosing statements."""
    items = [
        _item(_BOUNDARY_LEAF, _BOUNDARY_STATEMENT),
        _item(_PASSIVE_LEAF, _PASSIVE_STATEMENT),
    ]
    names_mode = _render("sn/generate_name_dd_names", items)
    full_mode = _render("sn/generate_name_dd", items)

    for item in items:
        for ancestor in item["ancestor_context"]:
            assert ancestor["text"] in full_mode
            assert ancestor["text"] in names_mode


def test_names_mode_invents_no_lineage_block_without_one() -> None:
    """A leaf with no enclosing statement renders no empty lineage header."""
    rendered = _render(
        "sn/generate_name_dd_names",
        [_item("equilibrium/time_slice/profiles_1d/psi", None)],
    )

    assert "DD path lineage" not in rendered
