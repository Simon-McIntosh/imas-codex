"""The plant-wide reading sample and the reading it admits.

The sample is drawn before the reading and digested, and a file of verdicts
that a script produced is refused. Every identity here is invented.
"""

from __future__ import annotations

import json

import pytest

from iter_codex.remote.signal_vocabulary import (
    sample_for_reading,
    validate_readings,
)


def _item(signal_id: str, source: str, value: str = "state") -> dict:
    return {
        "id": signal_id,
        "axis": "role",
        "value": value,
        "source": source,
        "field_code": "QX",
        "component_category": "MF",
        "description": "an invented signal",
        "description_source": "controlsmetadata:/invented",
        "unit": None,
    }


class TestTheSampleIsDrawnAndDigested:
    def test_ten_are_taken_per_credited_source(self):
        items = [_item(f"inv:{index:03d}", "field_pattern") for index in range(40)]
        sample = sample_for_reading(items)
        assert len(sample["sample"]) == 10
        assert sample["counts"] == {"field_pattern": 10}

    def test_a_source_with_fewer_than_ten_contributes_all(self):
        items = [_item(f"inv:a{index}", "signal_list_row") for index in range(3)]
        sample = sample_for_reading(items)
        assert len(sample["sample"]) == 3
        assert sample["counts"] == {"signal_list_row": 3}

    def test_the_digest_changes_with_the_sample(self):
        first = sample_for_reading(
            [_item(f"inv:{i:03d}", "field_pattern") for i in range(40)]
        )
        second = sample_for_reading(
            [_item(f"inv:{i:03d}", "field_pattern") for i in range(41)]
        )
        assert first["digest"] != second["digest"]

    def test_the_draw_is_deterministic(self):
        items = [_item(f"inv:{i:03d}", "field_pattern") for i in range(40)]
        assert (
            sample_for_reading(items)["digest"] == sample_for_reading(items)["digest"]
        )


class TestAReadingMustBeAReading:
    def _sample_ids(self, count: int = 3) -> set[str]:
        return {f"inv:{index}" for index in range(count)}

    def test_a_written_verdict_and_reason_per_item_is_admitted(self):
        entries = [
            {
                "id": "inv:0",
                "verdict": "right",
                "reason": "metadata names a measured temperature",
            },
            {
                "id": "inv:1",
                "verdict": "wrong",
                "reason": "metadata names a threshold, not a state",
            },
            {
                "id": "inv:2",
                "verdict": "right",
                "reason": "control text calls it a command",
            },
        ]
        summary = validate_readings(entries, self._sample_ids())
        assert summary == {
            "items": 3,
            "right": 2,
            "wrong": 1,
            "unverifiable": 0,
            "reason_variety": 3,
        }

    def test_one_identical_reason_for_every_item_is_a_script_not_a_reading(self):
        entries = [
            {"id": f"inv:{index}", "verdict": "right", "reason": "agrees"}
            for index in range(3)
        ]
        with pytest.raises(ValueError, match="script"):
            validate_readings(entries, self._sample_ids())

    def test_an_empty_reason_is_refused(self):
        entries = [
            {"id": "inv:0", "verdict": "right", "reason": "a measured value"},
            {"id": "inv:1", "verdict": "right", "reason": "  "},
            {"id": "inv:2", "verdict": "wrong", "reason": "a threshold"},
        ]
        with pytest.raises(ValueError, match="no written reason"):
            validate_readings(entries, self._sample_ids())

    def test_an_omitted_item_is_refused(self):
        entries = [
            {"id": "inv:0", "verdict": "right", "reason": "a measured value"},
            {"id": "inv:1", "verdict": "right", "reason": "a switched state"},
        ]
        with pytest.raises(ValueError, match="omits"):
            validate_readings(entries, self._sample_ids())

    def test_a_verdict_outside_the_three_is_refused(self):
        entries = [
            {"id": "inv:0", "verdict": "probably", "reason": "a measured value"},
            {"id": "inv:1", "verdict": "right", "reason": "a switched state"},
            {"id": "inv:2", "verdict": "right", "reason": "an alarm flag"},
        ]
        with pytest.raises(ValueError, match="verdict"):
            validate_readings(entries, self._sample_ids())

    def test_a_scripted_reading_file_round_trips_as_json_and_is_still_refused(self):
        scripted = [
            {"id": f"inv:{index}", "verdict": "right", "reason": "agrees"}
            for index in range(3)
        ]
        reloaded = json.loads(json.dumps(scripted))
        with pytest.raises(ValueError, match="script"):
            validate_readings(reloaded, self._sample_ids())
