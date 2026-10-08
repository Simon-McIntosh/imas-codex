"""Facility access calls reach both discovery judges as counted evidence."""

from __future__ import annotations

import re
import shutil

import pytest

from imas_codex.discovery.base.facility import get_facility, list_facilities
from imas_codex.discovery.code.scanner import _get_pattern_categories
from imas_codex.discovery.code.scorer import build_triage_state
from imas_codex.discovery.paths.enrichment import _build_enrich_patterns
from imas_codex.discovery.paths.scorer import build_path_judgment_state
from imas_codex.remote.scripts.discover_files import _batch_pattern_counts
from imas_codex.remote.scripts.enrich_directories import count_category_matches
from imas_codex.remote.scripts.enrich_files import batch_pattern_counts


@pytest.mark.parametrize("facility", sorted(list_facilities()))
def test_facility_access_patterns_reach_path_and_code_states(facility: str) -> None:
    config = get_facility(facility)
    access = config.get("data_access_patterns") or {}
    patterns = _build_enrich_patterns(facility)
    assert _get_pattern_categories(facility) == patterns

    expected = {
        f"{prefix}{value}": re.escape(value)
        for source, prefix in (
            ("key_tools", "facility_tool:"),
            ("code_import_patterns", "facility_import:"),
        )
        for value in access.get(source) or []
        if value
    }
    assert expected
    assert all(patterns.get(key) == regex for key, regex in expected.items())

    # The matched row models the counted output of either remote matcher.
    key, regex = next(iter(expected.items()))
    assert re.search(regex, key.split(":", 1)[1])
    counts = {key: 2}
    path_state = build_path_judgment_state(
        {"path": "/source", "pattern_categories": counts}, facility, config
    )
    file_row = {"path": "/source/reader.py", "pattern_categories": counts}
    names_state = build_triage_state(file_row, facility, config)
    content_state = build_triage_state(file_row, facility, config, with_content=True)
    assert (
        path_state["directory"]["enrichment"]["facility_data_access_matches"] == counts
    )
    assert names_state["file"]["facility_data_access_matches"] == counts
    assert content_state["file"]["facility_data_access_matches"] == counts
    assert content_state["file"]["pattern_evidence"]["categories"] == counts

    empty_path = build_path_judgment_state(
        {"path": "/empty", "pattern_categories": {}}, facility, config
    )
    empty_file = build_triage_state(
        {"path": "/empty/reader.py", "pattern_categories": {}},
        facility,
        config,
        with_content=True,
    )
    assert not empty_path["directory"]["enrichment"]["facility_data_access_matches"]
    assert not empty_file["file"]["facility_data_access_matches"]

    other_facility_keys = {
        f"{prefix}{value}"
        for other in list_facilities()
        if other != facility
        for source, prefix in (
            ("key_tools", "facility_tool:"),
            ("code_import_patterns", "facility_import:"),
        )
        for value in (get_facility(other).get("data_access_patterns") or {}).get(source)
        or []
    }
    assert not (other_facility_keys - expected.keys()) & patterns.keys()


def test_remote_matchers_count_access_calls_in_one_scan(tmp_path) -> None:
    if not shutil.which("rg"):
        pytest.skip("Remote matchers require rg")
    source = tmp_path / "reader.py"
    source.write_text("getseldata()\neddbreadTime()\ngetseldata()\n")
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("print('nothing matched')\n")
    categories = {
        "facility_tool:getseldata": re.escape("getseldata"),
        "facility_import:eddbreadTime": re.escape("eddbreadTime"),
    }
    expected = {
        "facility_tool:getseldata": 2,
        "facility_import:eddbreadTime": 1,
    }
    files = [str(source), str(unrelated)]
    assert count_category_matches(str(tmp_path), categories) == expected
    assert _batch_pattern_counts(files, categories) == {str(source): expected}
    assert batch_pattern_counts(files, categories) == {str(source): expected}
