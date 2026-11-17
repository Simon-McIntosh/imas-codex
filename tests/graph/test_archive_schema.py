"""Archive and non-numeric pulse round trips through generated graph models."""

import pytest
from pydantic import ValidationError

from imas_codex.config.codac_uda import create_archive_pulse
from imas_codex.graph.models import ArchiveEndpoint, DataSource, Pulse, PulseCategory
from imas_codex.graph.schema import GraphSchema


def test_pulse_and_categories_round_trip_without_losing_nanoseconds() -> None:
    archive = ArchiveEndpoint(
        id="demo:codac_uda:slow",
        facility_id="demo",
        data_system="codac_uda",
        role="slow",
        excluded=False,
        description="Synthetic slow archive",
    )
    category = PulseCategory(
        id=f"{archive.id}:DEMO:CAMPAIGN",
        facility_id="demo",
        archive=archive.id,
        name="DEMO:CAMPAIGN",
        description="Synthetic campaign category",
    )
    pulse_name = "DEMO:CAMPAIGN/commissioning-alpha"
    pulse = create_archive_pulse(
        facility_id="demo",
        archive=archive.id,
        name=pulse_name,
        start_ns=1_731_600_000_000_000_001,
        end_ns=1_731_600_002_000_000_003,
        categories=[category.id],
        description="Synthetic pulse with exact epoch nanosecond bounds",
    )
    for record in (archive, category, pulse):
        assert type(record).model_validate_json(record.model_dump_json()) == record
    assert pulse.id == f"{pulse.archive}:{pulse.name}"
    assert pulse.name.endswith("commissioning-alpha")
    assert pulse.end_ns - pulse.start_ns == 2_000_000_002


def test_same_pulse_segment_in_distinct_categories_has_distinct_ids() -> None:
    archive = "demo:codac_uda:slow"
    pulses = []
    for category_name in ("DEMO:CAMPAIGN-ALPHA", "DEMO:CAMPAIGN-BETA"):
        category = PulseCategory(
            id=f"{archive}:{category_name}",
            facility_id="demo",
            archive=archive,
            name=category_name,
        )
        name = f"{category.name}/commissioning-alpha"
        pulses.append(
            create_archive_pulse(
                facility_id="demo",
                archive=archive,
                name=name,
                categories=[category.id],
                start_ns=1_731_600_000_000_000_001,
                end_ns=1_731_600_002_000_000_003,
            )
        )
    restored = [Pulse.model_validate_json(pulse.model_dump_json()) for pulse in pulses]
    assert {pulse.name.rsplit("/", 1)[-1] for pulse in restored} == {
        "commissioning-alpha"
    }
    assert len({pulse.id for pulse in restored}) == 2
    assert len({pulse.categories[0] for pulse in restored}) == 2
    assert {pulse.id for pulse in restored} == {
        "demo:codac_uda:slow:DEMO:CAMPAIGN-ALPHA/commissioning-alpha",
        "demo:codac_uda:slow:DEMO:CAMPAIGN-BETA/commissioning-alpha",
    }


@pytest.mark.parametrize(
    "supplied_id",
    [
        "demo:codac_uda:slow:commissioning-alpha",
        "demo:codac_uda:fast:DEMO:CAMPAIGN/commissioning-alpha",
    ],
    ids=["final-segment-only", "different-archive"],
)
def test_pulse_construction_refuses_mismatched_id(supplied_id: str) -> None:
    with pytest.raises(ValueError, match="id disagrees with archive and full"):
        create_archive_pulse(
            id=supplied_id,
            facility_id="demo",
            archive="demo:codac_uda:slow",
            name="DEMO:CAMPAIGN/commissioning-alpha",
            start_ns=1_731_600_000_000_000_001,
            end_ns=1_731_600_002_000_000_003,
        )


def test_pulse_construction_accepts_matching_supplied_id() -> None:
    pulse = create_archive_pulse(
        id="demo:codac_uda:slow:DEMO:CAMPAIGN/commissioning-alpha",
        facility_id="demo",
        archive="demo:codac_uda:slow",
        name="DEMO:CAMPAIGN/commissioning-alpha",
        start_ns=1_731_600_000_000_000_001,
        end_ns=1_731_600_002_000_000_003,
    )
    assert pulse.id == "demo:codac_uda:slow:DEMO:CAMPAIGN/commissioning-alpha"


@pytest.mark.parametrize("name", ["commissioning-alpha", "/pulse", "DEMO:CAMPAIGN/"])
def test_pulse_construction_requires_category_qualified_name(name: str) -> None:
    with pytest.raises(ValueError, match="include its archive category and pulse path"):
        create_archive_pulse(
            facility_id="demo",
            archive="demo:codac_uda:slow",
            name=name,
            start_ns=1_731_600_000_000_000_001,
            end_ns=1_731_600_002_000_000_003,
        )


def test_archive_relationships_are_schema_driven() -> None:
    schema = GraphSchema()
    expected = {
        "ArchiveEndpoint": {("facility_id", "Facility", "AT_FACILITY")},
        "PulseCategory": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("archive", "ArchiveEndpoint", "IN_ARCHIVE"),
        },
        "Pulse": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("archive", "ArchiveEndpoint", "IN_ARCHIVE"),
            ("categories", "PulseCategory", "IN_PULSE_CATEGORY"),
        },
        "DataSource": {("archive_endpoint", "ArchiveEndpoint", "HOSTED_BY_ARCHIVE")},
    }
    for label, relationships in expected.items():
        declared = {
            (rel.slot_name, rel.to_class, rel.cypher_type)
            for rel in schema.get_relationships_from(label)
        }
        assert relationships <= declared


def test_archive_data_source_does_not_require_mdsplus_server() -> None:
    source = DataSource(
        id="demo:archive",
        name="archive",
        facility_id="demo",
        source_type="uda",
        archive_endpoint="demo:codac_uda:fast",
    )
    assert source.server_hostname is None
    assert DataSource.model_validate_json(source.model_dump_json()) == source


@pytest.mark.parametrize("field", ["name", "start_ns", "archive"])
def test_pulse_requires_identity_and_bounds(field: str) -> None:
    values = {
        "id": "demo:codac_uda:slow:DEMO:CAMPAIGN/pulse-alpha",
        "facility_id": "demo",
        "archive": "demo:codac_uda:slow",
        "name": "DEMO:CAMPAIGN/pulse-alpha",
        "start_ns": 1_731_600_000_000_000_001,
        "end_ns": 1_731_600_002_000_000_003,
    }
    values.pop(field)
    with pytest.raises(ValidationError):
        Pulse.model_validate(values)
