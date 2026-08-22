"""Dimensional unit agreement with explicit same-dimension distinctions."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.dd_resolutions import load_dd_resolution_manifest
from imas_codex.units import dd_unit_exceptions


def test_equivalence_table_is_unnecessary_for_dimensional_agreement(
    monkeypatch,
) -> None:
    exceptions = dd_unit_exceptions.load_exceptions()
    old_equivalences = exceptions["unit_equivalences"]
    monkeypatch.setattr(
        dd_unit_exceptions,
        "load_exceptions",
        lambda: {**exceptions, "unit_equivalences": []},
    )

    assert len(old_equivalences) == 3
    assert dd_unit_exceptions.units_agree("Hz", "s^-1", "any/path")
    assert dd_unit_exceptions.units_agree("N.m^-2", "kg.m^-1.s^-2", "any/path")
    assert all(
        dd_unit_exceptions.units_agree(*pair, "any/path") for pair in old_equivalences
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ("J", "N.m"),
        ("Hz", "Bq"),
        ("Gy", "Sv"),
    ],
)
def test_registered_same_dimension_meanings_remain_distinct(
    left: str, right: str
) -> None:
    assert not dd_unit_exceptions.units_agree(left, right, "any/path")


def test_every_recorded_dd_bug_pair_still_agrees() -> None:
    exceptions = dd_unit_exceptions.load_exceptions()
    manifest = load_dd_resolution_manifest().model_copy(update={"resolutions": ()})

    assert exceptions["dd_unit_bugs"]
    for entry in exceptions["dd_unit_bugs"]:
        example_path = str(entry["path"]).replace("[xyz]", "x").replace("*", "sample")
        assert dd_unit_exceptions.units_agree(
            str(entry["correct_unit"]),
            str(entry["dd_unit"]),
            example_path,
            manifest=manifest,
        ), entry["path"]
