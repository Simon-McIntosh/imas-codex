"""Pulse snapshot imports against a disposable Neo4j, including absent data."""

import pytest

from iter_codex.config.codac_uda import create_archive_pulse
from iter_codex.discovery.signals.archive_pulses import import_pulses
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import ArchiveEndpoint, PulseCategory

pytestmark = pytest.mark.graph


def records():
    facility = "demo-pulse-import"
    archive = ArchiveEndpoint(
        id=f"{facility}:codac_uda:slow",
        facility_id=facility,
        data_system="codac_uda",
        role="slow",
        excluded=False,
    )
    categories = [
        PulseCategory(
            id=f"{archive.id}:{name}",
            facility_id=facility,
            archive=archive.id,
            name=name,
        )
        for name in ["DEMO:ACTIVE", "DEMO:EMPTY"]
    ]
    pulse = create_archive_pulse(
        facility_id=facility,
        archive=archive.id,
        name="DEMO:ACTIVE/open-alpha",
        start_ns=1731600000000000001,
        end_ns=None,
        categories=[categories[0].id],
        description="Demonstration unfinished pulse",
    )
    return archive, categories, pulse


def seed_facility(graph, archive):
    graph.query(
        "CREATE (:Facility {id: $id, name: 'Demonstration'})", id=archive.facility_id
    )


def test_open_ended_pulse_is_stored_without_invented_end(disposable_graph):
    archive, categories, pulse = records()
    with GraphClient() as graph:
        seed_facility(graph, archive)
        closed = pulse.model_copy(update={"end_ns": pulse.start_ns + 2})
        import_pulses(graph, archive, categories, [closed])
        receipt = import_pulses(graph, archive, categories, [pulse])
        rows = graph.query(
            "MATCH (p:Pulse {id: $id}) RETURN p.start_ns AS start, p.end_ns AS end, p.description AS description",
            id=pulse.id,
        )
        assert rows == [
            {"start": pulse.start_ns, "end": None, "description": pulse.description}
        ]
        assert receipt["with_end"] == 0
        assert receipt["without_end"] == 1
        assert receipt["nodes_created"] == 0
        assert graph.query(
            "MATCH (p:Pulse {id:$id})-[:IN_PULSE_CATEGORY]->(c:PulseCategory)-[:IN_ARCHIVE]->(a:ArchiveEndpoint)-[:AT_FACILITY]->(f:Facility) RETURN p.categories AS categories, c.id AS category, a.id AS archive, f.id AS facility",
            id=pulse.id,
        ) == [
            {
                "categories": [categories[0].id],
                "category": categories[0].id,
                "archive": archive.id,
                "facility": archive.facility_id,
            }
        ]


def test_empty_category_is_present_linked_and_counted(disposable_graph):
    archive, categories, pulse = records()
    with GraphClient() as graph:
        seed_facility(graph, archive)
        receipt = import_pulses(graph, archive, categories, [pulse], batch_size=1)
        assert receipt == {
            "categories": 2,
            "pulses": 1,
            "pulses_per_category": {0: 1, 1: 1},
            "empty_categories": 1,
            "with_end": 0,
            "without_end": 1,
            "nodes_created": 4,
            "nodes_changed": 3,
        }
        rows = graph.query(
            "MATCH (c:PulseCategory {id: $id})-[:AT_FACILITY]->(f:Facility) MATCH (c)-[:IN_ARCHIVE]->(a:ArchiveEndpoint) OPTIONAL MATCH (p:Pulse)-[:IN_PULSE_CATEGORY]->(c) RETURN count(p) AS pulses, f.id AS facility, a.id AS archive",
            id=categories[1].id,
        )
        assert rows == [
            {"pulses": 0, "facility": archive.facility_id, "archive": archive.id}
        ]
        assert import_pulses(graph, archive, categories, [pulse])["nodes_created"] == 0


def test_missing_facility_refuses_write_without_orphans(disposable_graph):
    archive, categories, pulse = records()
    with GraphClient() as graph:
        with pytest.raises(RuntimeError, match="not acknowledged"):
            import_pulses(graph, archive, categories, [pulse])
        assert graph.query(
            "MATCH (a:ArchiveEndpoint {id:$id}) RETURN count(a) AS n", id=archive.id
        ) == [{"n": 0}]


def test_wrong_membership_refused_before_graph_write(disposable_graph):
    archive, categories, pulse = records()
    pulse.categories = [f"{archive.id}:DEMO:MISSING"]
    with GraphClient() as graph:
        with pytest.raises(ValueError, match="listed categories"):
            import_pulses(graph, archive, categories, [pulse])


def test_empty_description_is_removed_status_retained_and_repeat_changes_nothing(
    disposable_graph,
):
    archive, categories, pulse = records()
    with GraphClient() as graph:
        seed_facility(graph, archive)
        import_pulses(graph, archive, categories, [pulse])
        empty = pulse.model_copy(update={"description": "", "status": "finished"})
        receipt = import_pulses(graph, archive, categories, [empty])
        assert receipt["nodes_changed"] == 1
        assert graph.query(
            "MATCH (p:Pulse {id:$id}) RETURN p.description AS description, p.status AS status",
            id=pulse.id,
        ) == [{"description": None, "status": "finished"}]
        repeated = import_pulses(graph, archive, categories, [empty])
        assert repeated["nodes_created"] == repeated["nodes_changed"] == 0
