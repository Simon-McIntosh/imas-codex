"""Archive and non-numeric pulse round trips through generated graph models."""

import pytest
from pydantic import ValidationError

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
        id=f"{archive.id}:campaign",
        facility_id="demo",
        archive=archive.id,
        name="DEMO:CAMPAIGN",
        description="Synthetic campaign category",
    )
    pulse = Pulse(
        id=f"{archive.id}:commissioning-alpha",
        facility_id="demo",
        archive=archive.id,
        name="DEMO:CAMPAIGN/commissioning-alpha",
        start_ns=1_731_600_000_000_000_001,
        end_ns=1_731_600_002_000_000_003,
        categories=[category.id],
        description="Synthetic pulse with exact epoch nanosecond bounds",
    )
    for record in (archive, category, pulse):
        assert type(record).model_validate_json(record.model_dump_json()) == record
    assert pulse.name.endswith("commissioning-alpha")
    assert pulse.end_ns - pulse.start_ns == 2_000_000_002


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


@pytest.mark.parametrize("field", ["name", "start_ns", "end_ns", "archive"])
def test_pulse_requires_identity_and_bounds(field: str) -> None:
    values = {
        "id": "demo:archive:pulse-alpha",
        "facility_id": "demo",
        "archive": "demo:codac_uda:slow",
        "name": "pulse-alpha",
        "start_ns": 1_731_600_000_000_000_001,
        "end_ns": 1_731_600_002_000_000_003,
    }
    values.pop(field)
    with pytest.raises(ValidationError):
        Pulse.model_validate(values)
