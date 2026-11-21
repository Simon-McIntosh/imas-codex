"""Category names carrying plant codes, measured against invented codes and names."""

import pytest

from imas_codex.discovery.signals.category_codes import (
    carried_codes,
    delimited_parts,
    select_hand_sample,
    summarize,
)


def test_delimited_parts_split_on_every_non_alphanumeric_run():
    assert delimited_parts("unit ab-12_x:yz") == ("UNIT", "AB", "12", "X", "YZ")


def test_whole_delimited_part_equal_to_a_control_code_is_a_match():
    carriage = carried_codes("cat-1", "UNIT-FAST discharge", {"UNIT"}, set())
    assert carriage.delimited_control == ("UNIT",)
    assert carriage.inside_word == ()
    assert carriage.link_count == 1


def test_whole_delimited_part_equal_to_a_component_code_is_a_match():
    carriage = carried_codes("cat-2", "draw AB12 ramp", set(), {"AB12"})
    assert carriage.delimited_component == ("AB12",)
    assert carriage.link_count == 1


def test_code_only_inside_a_longer_word_is_never_a_delimited_match():
    carriage = carried_codes("cat-3", "UNITWIDE run", {"UNIT"}, set())
    assert carriage.delimited_control == ()
    assert carriage.inside_word == ("UNIT",)
    assert carriage.link_count == 1


def test_two_delimited_codes_count_as_more_than_one():
    carriage = carried_codes("cat-4", "UNIT-PAIR alpha", {"UNIT", "PAIR"}, set())
    assert carriage.delimited_control == ("PAIR", "UNIT")
    assert carriage.link_count == 2


def test_a_code_that_is_delimited_anywhere_is_not_also_inside_a_word():
    carriage = carried_codes("cat-5", "UNITX and UNIT", {"UNIT"}, set())
    assert carriage.delimited_control == ("UNIT",)
    assert carriage.inside_word == ()


def test_codes_match_regardless_of_case():
    carriage = carried_codes("cat-6", "unit-Fast", {"UNIT"}, {"FAST"})
    assert carriage.delimited_control == ("UNIT",)
    assert carriage.delimited_component == ("FAST",)


def test_summary_counts_each_way_and_partitions_every_category():
    carriages = [
        carried_codes("c1", "UNIT-one", {"UNIT"}, set()),
        carried_codes("c2", "PAIR-two", {"PAIR", "TWO"}, set()),
        carried_codes("c3", "nothing-here", {"UNIT"}, set()),
        carried_codes("c4", "DEEPWORD-three", {"DEEP"}, set()),
    ]
    summary = summarize(carriages, {"c1": 5, "c2": 7, "c3": 11, "c4": 13})
    ways = summary["ways"]
    assert ways["control_node"] == {
        "categories": 2,
        "codes": ["PAIR", "TWO", "UNIT"],
    }
    assert ways["inside_word"] == {"categories": 1, "codes": ["DEEP"]}
    groups = summary["groups"]
    assert groups["one"] == {"categories": 2, "pulses": 18}
    assert groups["more"] == {"categories": 1, "pulses": 7}
    assert groups["none"] == {"categories": 1, "pulses": 11}
    assert summary["categories_read"] == sum(
        group["categories"] for group in groups.values()
    )


def test_summary_reports_a_whole_word_match_in_the_inside_word_way_only():
    summary = summarize(
        [carried_codes("c1", "UNITWIDE run", {"UNIT", "WIDE"}, set())], {"c1": 2}
    )
    assert summary["ways"]["control_node"] == {"categories": 0, "codes": []}
    assert summary["ways"]["inside_word"] == {
        "categories": 1,
        "codes": ["UNIT", "WIDE"],
    }


def test_summary_counts_pulses_for_a_category_absent_from_the_count_map_as_zero():
    summary = summarize([carried_codes("c1", "UNIT-one", {"UNIT"}, set())], {})
    assert summary["groups"]["one"]["pulses"] == 0


def test_hand_sample_is_deterministic_for_a_recorded_seed():
    carriages = [
        carried_codes(f"cat-{index}", f"UNIT-{index}", {"UNIT"}, set())
        for index in range(30)
    ]
    first = select_hand_sample(carriages, size=10, seed=7)
    second = select_hand_sample(list(reversed(carriages)), size=10, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 10


def test_hand_sample_represents_every_carried_code():
    carriages = [
        carried_codes("a", "UNIT-one", {"UNIT"}, set()),
        carried_codes("b", "PAIR-two", {"PAIR"}, set()),
        carried_codes("c", "THIRD-three", {"THIRD"}, set()),
    ]
    assert set(select_hand_sample(carriages, size=3, seed=1)) == {"a", "b", "c"}


def test_hand_sample_takes_only_categories_the_delimited_rule_would_link():
    carriages = [
        carried_codes("linked", "UNIT-one", {"UNIT"}, set()),
        carried_codes("unlinked", "nothing", {"UNIT"}, set()),
    ]
    with pytest.raises(ValueError):
        select_hand_sample(carriages, size=2, seed=1)


def test_hand_sample_excludes_a_category_matched_only_inside_a_word():
    carriages = [
        carried_codes("linked", "UNIT-one", {"UNIT"}, set()),
        carried_codes("inside", "UNITWIDE-two", {"UNIT"}, set()),
    ]
    assert select_hand_sample(carriages, size=1, seed=1) == ["linked"]
    with pytest.raises(ValueError):
        select_hand_sample(carriages, size=2, seed=1)


def test_carriage_of_an_unmatched_name_has_no_codes():
    carriage = carried_codes("none", "plain words", {"UNIT"}, set())
    assert carriage.codes == frozenset()
    assert carriage.link_count == 0
