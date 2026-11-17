"""Controls feed validation refuses ambiguous or ungrounded pilot imports."""

import json

import pytest

from imas_codex.discovery.signals.controls_metadata import (
    catalogue_members,
    read_controls_feed,
)
from imas_codex.graph.signal_descriptions import description_is_name
from imas_codex.remote.archive_catalogue import ArchiveCatalogue


def entry(**changes):
    return {
        "name": "DEMO:TEMPERATURE",
        "description": "Helium inlet temperature",
        "unit": "K",
        "data_type": "FLOAT",
        "update_time": "2026-01-02 03:04:05",
        **changes,
    }


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        [entry(), entry()],
        [entry(update_time="")],
        [entry(update_time="tomorrow")],
        [entry(unit=None)],
    ],
)
def test_invalid_snapshot_is_refused(tmp_path, body):
    path = tmp_path / "feed.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        read_controls_feed(path)


def test_source_timestamp_and_wording_are_preserved(tmp_path):
    path = tmp_path / "feed.json"
    path.write_text(json.dumps([entry(description="  Helium inlet temperature  ")]))
    rows, digest = read_controls_feed(path)
    assert len(digest) == 64
    assert rows[0].description == "  Helium inlet temperature  "
    assert rows[0].update_time == "2026-01-02 03:04:05"


def test_catalogue_requires_known_member_and_does_not_change_store(tmp_path):
    path = tmp_path / "catalogue.sqlite"
    catalogue = ArchiveCatalogue(path)
    catalogue.seed({"slow": ["DEMO:TEMPERATURE"]}, timestamp="2026-01-01")
    catalogue.close()
    before = path.read_bytes()
    result = catalogue_members(
        path, ["DEMO:TEMPERATURE", "DEMO:MISSING"], positive_control="DEMO:TEMPERATURE"
    )
    assert result == {"DEMO:TEMPERATURE": [("slow", None)]}
    with pytest.raises(ValueError, match="positive control"):
        catalogue_members(path, ["DEMO:MISSING"], positive_control="DEMO:MISSING")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "description, expected",
    [
        ("DEMO:TEMPERATURE", True),
        ("TEMPERATURE", True),
        (" TEMPERATURE ", True),
        ("Helium inlet temperature", False),
        ("", False),
    ],
)
def test_name_only_description_eligibility(description, expected):
    assert (
        description_is_name(description, "DEMO:TEMPERATURE", "DEMO:TEMPERATURE")
        is expected
    )


def test_description_writer_refuses_a_missing_precomputed_receipt():
    from unittest.mock import Mock

    from imas_codex.discovery.signals.parallel import write_signal_descriptions

    graph = Mock()
    with pytest.raises(ValueError, match="cover every description"):
        write_signal_descriptions(
            [{"id": "demo:temperature", "description": "Helium inlet temperature"}],
            gc=graph,
            precomputed_embeddings={},
        )
    graph.query.assert_not_called()
