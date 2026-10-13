"""Tests for self-link detection and the per-link label judgement state."""

from __future__ import annotations

from imas_codex.standard_names.doc_links import (
    find_name_links,
    find_self_links,
    link_judgement_state,
    link_label_question,
)

DOCUMENTING = "flux_surface_averaged_beryllium_density_at_plasma_boundary"
_NEIGHBOUR = "beryllium_density"

#: A page that links back to the documenting name and to a different neighbour.
_DOCUMENTATION = (
    f"See [beryllium density at the plasma boundary](name:{DOCUMENTING}) "
    f"and [beryllium density](name:{_NEIGHBOUR})."
)


def test_self_link_is_flagged_while_a_neighbour_link_is_not() -> None:
    self_links = find_self_links(_DOCUMENTATION, DOCUMENTING)

    assert [link.target for link in self_links] == [DOCUMENTING]
    assert [link.label for link in self_links] == [
        "beryllium density at the plasma boundary"
    ]


def test_all_name_links_are_returned_with_their_labels_and_targets() -> None:
    links = find_name_links(_DOCUMENTATION)

    assert [(link.label, link.target) for link in links] == [
        ("beryllium density at the plasma boundary", DOCUMENTING),
        ("beryllium density", _NEIGHBOUR),
    ]


def test_a_link_to_a_non_name_target_is_not_a_name_link() -> None:
    assert find_name_links("[plot](figure.png) and [docs](https://example.test)") == ()


def test_link_judgement_state_carries_the_target_description() -> None:
    link = find_name_links(f"[beryllium density](name:{_NEIGHBOUR})")[0]

    state = link_judgement_state(
        link,
        documenting_name=DOCUMENTING,
        target_description="The number density of beryllium ions.",
    )

    assert state == {
        "documenting_name": DOCUMENTING,
        "link_text": "beryllium density",
        "target_name": _NEIGHBOUR,
        "target_description": "The number density of beryllium ions.",
    }


def test_link_question_is_a_noul_naming_the_target_description() -> None:
    question = link_label_question()

    ((name, definition),) = question.items()
    assert name == "label_describes_target"
    assert definition["type"] == "noul"
    assert set(definition["criteria"]) == {"true", "false"}
    assert "target_description" in definition["instructions"]
