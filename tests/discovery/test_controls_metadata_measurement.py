"""Invented feed and catalogue controls for offline coverage measurement."""

import json
import sqlite3
import sys

import pytest

from scripts import measure_controls_metadata as measurement


@pytest.mark.parametrize(
    ("body", "cause"),
    [
        ([], "no entries"),
        ({"entries": []}, "no entries"),
        ({"message": "unavailable"}, "exactly one entry list"),
        ("not a list", "entry list"),
        ({"a": [], "b": []}, "exactly one entry list"),
        (["not an entry"], "must be an object"),
        ([{"description": "No selector"}], "string name"),
        ([{"name": "INVENTED", "unit": {}}], "non-string unit"),
    ],
    ids=[
        "empty-list",
        "empty-wrapper",
        "wrong-wrapper",
        "wrong-root",
        "ambiguous-wrapper",
        "wrong-entry",
        "missing-name",
        "wrong-unit",
    ],
)
def test_unmeasurable_response_is_refused(body, cause):
    with pytest.raises(measurement.MeasurementError, match=cause):
        measurement.parse_entries(body)


def test_installed_client_key_variants_and_wrapper():
    rows = measurement.parse_entries(
        {"variables": [{"variable": "INVENTED", "units": " K ", "data type": "float"}]}
    )
    assert rows == [
        {"name": "INVENTED", "description": "", "unit": "K", "type": "float"}
    ]


def make_catalogue(path, names):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE census (role TEXT, name TEXT)")
        connection.executemany(
            "INSERT INTO census VALUES ('slow', ?)", [(name,) for name in names]
        )


def test_counts_use_archive_denominators_and_separate_normalisations(tmp_path):
    catalogue = tmp_path / "catalogue.sqlite"
    control = "SYS-COLD-COMP:TEMP"
    make_catalogue(
        catalogue,
        [
            control,
            control,
            "SYS-COLD-COMP:FLOW",
            "SYS-WARM-COMP:PRESS",
            "ALT-WARM-COMP-TEMP",
            "DOT-WARM-COMP.TEMP",
            "SEP-WARM-COMP:TEMP_A",
        ],
    )
    entries = measurement.parse_entries(
        [
            {"name": name, "description": "Cryogenic inlet temperature", "unit": "K"}
            for name in (
                control,
                " SYS-COLD-COMP:FLOW ",
                "sys-warm-comp:press",
                "ALT-WARM-COMP:TEMP",
                "DOT-WARM-COMP:TEMP",
                "sep.warm.comp:temp-a",
            )
        ]
    )
    before = catalogue.read_bytes()
    result = measurement.compare_catalogue(entries, catalogue, control, "COLD")
    assert catalogue.read_bytes() == before
    assert result["catalogue_names"] == 6
    assert result["positive_control_found"] is True
    expected = {
        "exact": 1,
        "outer_whitespace": 2,
        "casefold": 2,
        "colon_to_hyphen": 2,
        "colon_to_dot": 2,
        "separator_fold_casefold": 6,
    }
    for key, count in expected.items():
        row = result["normalisations"][key]
        assert row["matched_feed_names"] == count
        assert row["covered_archive_names"] == count
        assert row["archive_coverage"] == count / 6
    assert result["cohort"]["catalogue_names"] == 2
    assert result["cohort"]["covered_archive_names"]["exact"] == 1
    assert sum(row["catalogue"] for row in result["systems"]) == 6


def test_missing_known_archive_control_is_failure_not_zero_coverage(tmp_path):
    catalogue = tmp_path / "catalogue.sqlite"
    make_catalogue(catalogue, ["SYS-WARM-COMP:TEMP"])
    entries = measurement.parse_entries([{"name": "SYS-COLD-COMP:TEMP"}])
    with pytest.raises(measurement.MeasurementError, match="positive control"):
        measurement.compare_catalogue(entries, catalogue, "ABSENT", "COLD")


def test_normalisation_collisions_remain_visible(tmp_path):
    catalogue = tmp_path / "catalogue.sqlite"
    make_catalogue(catalogue, ["SYS-COLD:TEMP", "sys-cold:temp"])
    entries = measurement.parse_entries(
        [{"name": "SYS-COLD:TEMP"}, {"name": "sys-cold:temp"}]
    )
    result = measurement.compare_catalogue(entries, catalogue, "SYS-COLD:TEMP", "COLD")
    assert result["normalisations"]["casefold"]["feed_collision_keys"] == 1
    assert result["normalisations"]["casefold"]["matched_archive_collision_keys"] == 1
    assert result["normalisations"]["casefold"]["covered_archive_names"] == 2


def test_quality_distribution_keeps_empty_fields_and_boilerplate():
    entries = measurement.parse_entries(
        [
            {"name": "INVENTED-A", "description": "Common label", "unit": "K"},
            {"name": "INVENTED-B", "description": "Common label", "data_type": "float"},
            {"name": "INVENTED-C", "description": "  "},
        ]
    )
    result = measurement.feed_quality(entries)
    assert result["entries"] == 3
    assert result["nonempty"] == {"description": 2, "unit": 1, "type": 1}
    assert result["distinct_nonempty_descriptions"] == 1
    assert result["ten_most_common_share"] == 2 / 3
    assert result["description_length_bins"]["empty"] == 1
    assert result["description_length_quantiles"]["0"] == 0
    assert result["description_length_quantiles"]["50"] == 12


def test_invalid_json_fails_before_opening_catalogue(tmp_path):
    response = tmp_path / "response.json"
    response.write_text("<html>unavailable</html>")
    with pytest.raises(measurement.MeasurementError, match="not valid JSON"):
        measurement.measure(response, tmp_path / "absent.sqlite", {})
    assert not (tmp_path / "absent.sqlite").exists()


def test_empty_response_fails_before_catalogue_comparison(tmp_path):
    response = tmp_path / "response.json"
    response.write_text(json.dumps([]))
    with pytest.raises(measurement.MeasurementError, match="no entries"):
        measurement.measure(response, tmp_path / "absent.sqlite", {})
    assert not (tmp_path / "absent.sqlite").exists()


@pytest.mark.parametrize(
    ("body", "cause"),
    [([], "no entries"), ({"message": "unavailable"}, "exactly one entry list")],
    ids=["empty-response", "wrong-shape"],
)
def test_cli_reports_measurement_failure_without_coverage(
    tmp_path, monkeypatch, capsys, body, cause
):
    response = tmp_path / "response.json"
    response.write_text(json.dumps(body))
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    output = tmp_path / "measurement.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure-controls",
            "--response",
            str(response),
            "--catalogue",
            str(tmp_path / "absent.sqlite"),
            "--settings",
            str(settings),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        measurement.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "measurement failed: response" in captured.err
    assert cause in captured.err
    assert captured.out == ""
    assert not output.exists()
