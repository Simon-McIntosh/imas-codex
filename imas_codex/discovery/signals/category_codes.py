"""Measure which plant codes a pulse category name carries, and how.

A category identity is an archive-declared string. This module reports the ways
such a string can name a plant code already held in the graph, without
interpreting any code and without writing a link: a whole delimited part equal
to a control-node code, a whole delimited part equal to a plant-component code,
and a code that appears only inside a longer word.

Delimited parts are the maximal runs of letters and digits, so a hyphen, colon,
underscore, space or punctuation character separates them. A control-node code
is a hyphen-joined path but every prefix of it is also a control node, so the
single-field control codes are the ones a delimited part can equal exactly.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_DELIMITER = re.compile(r"[^A-Za-z0-9]")

CONTROL_NODE = "control_node"
PLANT_COMPONENT = "plant_component"


def delimited_parts(name: str) -> tuple[str, ...]:
    """Return the name's maximal alphanumeric runs, uppercased."""
    return tuple(part.upper() for part in _DELIMITER.split(name) if part)


@dataclass(frozen=True)
class CategoryCodeCarriage:
    """The codes one category name carries, separated by how it carries them."""

    category_id: str
    delimited_control: tuple[str, ...]
    delimited_component: tuple[str, ...]
    inside_word: tuple[str, ...]

    @property
    def codes(self) -> frozenset[str]:
        """Every distinct code the name carries, however it carries it."""
        return (
            frozenset(self.delimited_control)
            | frozenset(self.delimited_component)
            | frozenset(self.inside_word)
        )

    @property
    def link_count(self) -> int:
        return len(self.codes)


def carried_codes(
    category_id: str,
    name: str,
    control_codes: Iterable[str],
    component_codes: Iterable[str],
) -> CategoryCodeCarriage:
    """Classify each code a name carries without inferring what a code means."""
    parts = delimited_parts(name)
    part_set = set(parts)

    def split_matches(codes: Iterable[str]) -> tuple[set[str], set[str]]:
        delimited: set[str] = set()
        inside: set[str] = set()
        for raw in codes:
            code = raw.upper()
            if code in part_set:
                delimited.add(code)
            elif any(code in part and part != code for part in parts):
                inside.add(code)
        return delimited, inside

    control_delimited, control_inside = split_matches(control_codes)
    component_delimited, component_inside = split_matches(component_codes)
    return CategoryCodeCarriage(
        category_id=category_id,
        delimited_control=tuple(sorted(control_delimited)),
        delimited_component=tuple(sorted(component_delimited)),
        inside_word=tuple(sorted(control_inside | component_inside)),
    )


def summarize(
    carriages: Iterable[CategoryCodeCarriage],
    pulse_counts: Mapping[str, int],
) -> dict:
    """Count the categories and pulses in each way and each whole-number group.

    Every returned code collection is the private measurement itself: the counts
    are publishable, the code values are not.
    """
    carriages = list(carriages)
    delimited_control = [
        carriage for carriage in carriages if carriage.delimited_control
    ]
    delimited_component = [
        carriage for carriage in carriages if carriage.delimited_component
    ]
    inside_word = [carriage for carriage in carriages if carriage.inside_word]

    def distinct(values: Iterable[str]) -> list[str]:
        return sorted(set(values))

    groups = {"one": [], "more": [], "none": []}
    for carriage in carriages:
        count = carriage.link_count
        buckets = groups["none" if count == 0 else "one" if count == 1 else "more"]
        buckets.append(carriage)

    def group_summary(members: list[CategoryCodeCarriage]) -> dict:
        return {
            "categories": len(members),
            "pulses": sum(
                pulse_counts.get(member.category_id, 0) for member in members
            ),
        }

    return {
        "categories_read": len(carriages),
        "ways": {
            CONTROL_NODE: {
                "categories": len(delimited_control),
                "codes": distinct(
                    code
                    for carriage in delimited_control
                    for code in carriage.delimited_control
                ),
            },
            PLANT_COMPONENT: {
                "categories": len(delimited_component),
                "codes": distinct(
                    code
                    for carriage in delimited_component
                    for code in carriage.delimited_component
                ),
            },
            "inside_word": {
                "categories": len(inside_word),
                "codes": distinct(
                    code for carriage in inside_word for code in carriage.inside_word
                ),
            },
        },
        "groups": {group: group_summary(members) for group, members in groups.items()},
        "codes_are_private": (
            "each way's code list names real graph codes; keep it with the private "
            "per-category table and out of tracked files"
        ),
    }


def select_hand_sample(
    carriages: Iterable[CategoryCodeCarriage],
    *,
    size: int = 20,
    seed: int,
) -> list[str]:
    """Choose a hand-reading sample deterministically, stratified by carried code.

    Only categories the delimited-part rule would link are eligible: the question
    the hand reading answers is whether that rule ever names the wrong system, so
    a category it would not link, including one whose only match is inside a
    longer word, would answer nothing. Category
    identities are drawn by a seeded shuffle so the same seed selects the same
    sample, and the selection is written before any verdict is recorded.
    """
    eligible = [
        carriage
        for carriage in carriages
        if carriage.delimited_control or carriage.delimited_component
    ]
    if len(eligible) < size:
        raise ValueError("fewer delimited-linkable categories than the sample size")

    by_code: dict[str, list[CategoryCodeCarriage]] = {}
    for carriage in eligible:
        for code in {*carriage.delimited_control, *carriage.delimited_component}:
            by_code.setdefault(code, []).append(carriage)

    def rank(carriage: CategoryCodeCarriage) -> bytes:
        return hashlib.sha256(
            f"category-code-sample:{seed}:{carriage.category_id}".encode()
        ).digest()

    strata = sorted(by_code, key=lambda code: min(rank(item) for item in by_code[code]))

    selected: list[str] = []
    seen: set[str] = set()

    def take(carriage: CategoryCodeCarriage) -> None:
        if carriage.category_id not in seen:
            seen.add(carriage.category_id)
            selected.append(carriage.category_id)

    # One from every carried code first, so no code is left unrepresented.
    for code in strata:
        if len(selected) >= size:
            break
        take(min(by_code[code], key=rank))

    for carriage in sorted(eligible, key=rank):
        if len(selected) >= size:
            break
        take(carriage)

    return selected
