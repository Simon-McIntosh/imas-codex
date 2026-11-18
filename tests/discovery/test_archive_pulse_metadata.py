"""Pulse metadata and category-kind evidence use invented archive identities."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imas_codex.config.codac_uda import create_archive_pulse
from imas_codex.discovery.signals.archive_pulses import (
    PulseKindRule,
    classify_categories,
    pulse_metadata_counts,
)
from imas_codex.graph.models import PulseCategory
from imas_codex.remote.codac_uda import ArchiveReader
from imas_codex.remote.scripts import codac_uda as native


def category(name="DEMO:coil-fast-discharge"):
    return PulseCategory(
        id=f"demo:codac_uda:slow:{name}",
        facility_id="demo",
        archive="demo:codac_uda:slow",
        name=name,
    )


def pulse(description="", status="finished"):
    return create_archive_pulse(
        facility_id="demo",
        archive="demo:codac_uda:slow",
        name=f"{category().name}/example",
        start_ns=1,
        end_ns=2,
        categories=[category().id],
        description=description,
        status=status,
    )


def rule(source="category_name", pattern=r"fast[- ]discharge", kind="fast discharge"):
    return PulseKindRule(
        kind=kind, source=source, pattern=pattern, reference="invented-source"
    )


def test_unsupported_kind_like_name_is_unlabelled_and_counted():
    labelled = classify_categories(
        [category()], [pulse()], [rule(pattern="current ramp")]
    )
    assert labelled[0].kind is labelled[0].kind_source is None
    counts = pulse_metadata_counts(labelled, [pulse()])
    assert counts["categories_without_kind"] == 1
    assert counts["categories_with_kind_by_source"] == counts["kinds"] == {}
    supported = classify_categories([category()], [pulse()], [rule()])
    assert supported[0].kind == "fast discharge"
    assert supported[0].kind_source == "category_name:invented-source"


@pytest.mark.parametrize("description", ["", None])
def test_empty_pulse_description_is_absent(description):
    record = pulse(description)
    assert record.description is None
    assert record.status == "finished"


def test_pulse_description_support_is_membership_scoped_and_ambiguity_abstains():
    records = [pulse("Fast discharge exercise")]
    rules = [rule("pulse_description")]
    other = category("DEMO:unrelated")
    labelled = classify_categories([category(), other], records, rules)
    assert labelled[0].kind_source == "pulse_description:invented-source"
    assert labelled[1].kind is None
    conflicting = classify_categories(
        [category()], records, [*rules, rule(kind="current ramp")]
    )
    assert conflicting[0].kind is conflicting[0].kind_source is None
    stale = category().model_copy(
        update={"kind": "unsupported", "kind_source": "document:stale"}
    )
    assert classify_categories([stale], records, [])[0].kind is None


def test_document_mapping_requires_explicit_match_and_reference():
    labelled = classify_categories([category()], [], [rule("document")])
    assert labelled[0].kind_source == "document:invented-source"
    with pytest.raises(ValueError):
        PulseKindRule(
            kind="invented kind",
            source="document",
            pattern="invented evidence",
            reference=" ",
        )
    with pytest.raises(ValueError):
        rule(pattern="[")


def test_metadata_reader_preserves_status_and_normalizes_empty_description():
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getPulseCategories.return_value = [category().name]
    client.getPulses2.return_value = [
        SimpleNamespace(
            pulseID=pulse().name,
            timeFrom=1,
            timeTo=2,
            description="",
            status="finished",
        )
    ]
    result = native.pulses(client, {"pattern": category().name})
    assert result["pulses"][0]["description"] is None
    assert result["pulses"][0]["status"] == "finished"
    reader = object.__new__(ArchiveReader)
    reader.facility, reader.archive = "demo", category().archive
    reader._request = Mock(return_value=result)
    assert reader.pulses(category().name)[0].status == "finished"
    client.fetchData.assert_not_called()
