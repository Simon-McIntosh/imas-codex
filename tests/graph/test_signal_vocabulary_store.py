"""Exercise the vocabulary write against a rolled-back synthetic facility.

Every identity, code, description and unit here is invented. The transaction is
rolled back, so nothing this module creates can reach the project graph.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from imas_codex.discovery.signals.vocabulary_graph import (
    COHORT_QUERY,
    COHORT_ROWS_QUERY,
    VOCABULARY_KEYS,
    WRITE_QUERY,
    apply_assignment,
    assign_cohort,
    verify_assignment,
)
from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph

SIGNALS = {
    # Decided by the linked row's own words, with the archive unit.
    "row": {
        "accessor": "DEMO-COIL-CTRL:ZZ0001-TT",
        "description": "Invented analogue channel",
        "description_source": None,
        "unit": None,
        "field_code": "TT",
        "archive_metadata": json.dumps(
            {"full": [{"name": "Units", "value": "K"}]}
        ),
    },
    # Decided by the controls metadata the signal carries.
    "controls": {
        "accessor": "DEMO-COIL-CTRL:ZZ0002-PP",
        "description": "Invented water temperature",
        "description_source": "controlsmetadata:/invented-feed",
        "unit": "degC",
        "field_code": "PP",
        "archive_metadata": None,
    },
    # Two sources disagree on the unit: a conflict, nothing stored.
    "conflict": {
        "accessor": "DEMO-COIL-CTRL:ZZ0003-QQ",
        "description": "Invented water temperature",
        "description_source": "controlsmetadata:/invented-feed",
        "unit": "K",
        "field_code": "QQ",
        "archive_metadata": json.dumps(
            {"full": [{"name": "Units", "value": "bar"}]}
        ),
    },
    # No row, no metadata, a field code no other signal shares: undecided.
    "bare": {
        "accessor": "DEMO-COIL-CTRL:ZZ0004-VV",
        "description": None,
        "description_source": None,
        "unit": None,
        "field_code": "VV",
        "archive_metadata": None,
    },
}

ROW_TEXT = "Signal purpose: Absolute pressure at the invented cold box outlet",
ROW_SIGNAL = "row"


class _View:
    """The transaction as the writer sees a graph."""

    def __init__(self, tx):
        self.tx = tx

    def query(self, statement, **parameters):
        return self.tx.run(statement, **parameters).data()


@pytest.fixture
def vocabulary_transaction():
    facility = "synthetic_vocabulary_" + uuid4().hex
    with GraphClient() as graph, graph.session() as session:
        tx = session.begin_transaction()
        try:
            tx.run(
                "CREATE (f:Facility {id:$facility, name:'Invented facility'})",
                facility=facility,
            ).consume()
            for suffix, properties in SIGNALS.items():
                signal_id = f"{facility}:general/demo/{suffix}"
                tx.run(
                    """
                    CREATE (s:FacilitySignal {id:$id, facility_id:$facility,
                        accessor:$accessor, description:$description,
                        description_source:$description_source, unit:$unit,
                        field_code:$field_code, archive_metadata:$archive_metadata,
                        plant_context_state:'derived', name:$suffix})
                    WITH s MATCH (f:Facility {id:$facility})
                    CREATE (s)-[:AT_FACILITY]->(f)
                    """,
                    id=signal_id,
                    facility=facility,
                    suffix=suffix,
                    **properties,
                ).consume()
            tx.run(
                """
                CREATE (d:PlantContextDocument {id:$facility + ':document:demo',
                    facility_id:$facility, role:'signal_list'})
                CREATE (r:PlantSignalRow {id:$facility + ':row:demo',
                    facility_id:$facility, description:$text})
                CREATE (l:SignalContextLink {id:$facility + ':link:demo',
                    facility_id:$facility, match_kind:'derived'})
                WITH d, r, l
                MATCH (s:FacilitySignal {id:$signal})
                CREATE (s)-[:HAS_PLANT_CONTEXT]->(l)
                CREATE (l)-[:CONTEXT_SOURCE_ROW]->(r)
                CREATE (l)-[:CONTEXT_SIGNAL]->(s)
                """,
                facility=facility,
                text=ROW_TEXT,
                signal=f"{facility}:general/demo/{ROW_SIGNAL}",
            ).consume()
            yield tx, _View(tx), facility
        finally:
            tx.rollback()


def test_every_vocabulary_statement_parses_against_the_live_graph(
    vocabulary_transaction,
):
    tx, _, facility = vocabulary_transaction
    for statement in (COHORT_QUERY, COHORT_ROWS_QUERY, WRITE_QUERY):
        tx.run(
            "EXPLAIN " + statement,
            facility=facility,
            limit=10,
            rows=[],
        ).consume()


def test_a_signal_with_no_row_metadata_or_shared_code_stays_undecided(
    vocabulary_transaction,
):
    tx, view, facility = vocabulary_transaction
    signals, rows = _snapshot(view, facility)
    result = assign_cohort(signals, rows)
    record = {item["id"]: item for item in result["ledger"]}[
        f"{facility}:general/demo/bare"
    ]
    for axis in ("quantity", "unit", "role"):
        assert record[f"vocabulary_{axis}"] is None
        assert record[f"vocabulary_{axis}_state"] == "undecided"


def test_sources_decide_and_a_unit_disagreement_stores_nothing(
    vocabulary_transaction,
):
    tx, view, facility = vocabulary_transaction
    signals, rows = _snapshot(view, facility)
    result = assign_cohort(signals, rows)
    by_id = {item["id"]: item for item in result["ledger"]}

    row_record = by_id[f"{facility}:general/demo/{ROW_SIGNAL}"]
    assert row_record["vocabulary_quantity"] == "pressure"
    assert row_record["vocabulary_quantity_source"] == "signal_list_row"
    assert row_record["vocabulary_unit"] == "kelvin"
    assert row_record["vocabulary_unit_source"] == "archive_unit"

    controls = by_id[f"{facility}:general/demo/controls"]
    assert controls["vocabulary_quantity"] == "temperature"
    assert controls["vocabulary_unit"] == "degree_celsius"

    conflict = by_id[f"{facility}:general/demo/conflict"]
    assert conflict["vocabulary_unit_state"] == "conflict"
    assert conflict["vocabulary_unit"] is None


def test_the_write_lands_every_value_and_a_repeat_changes_nothing(
    vocabulary_transaction,
):
    tx, view, facility = vocabulary_transaction
    signals, rows = _snapshot(view, facility)
    first = assign_cohort(signals, rows)
    assert first["changed"] == len(signals)
    applied = apply_assignment(view, facility, first["pending"])
    assert applied["written"] == len(signals)

    verified = verify_assignment(view, facility, first["ledger"])
    assert verified["mismatched"] == 0
    assert verified["assigned_values"] > 0

    second = assign_cohort(*_snapshot(view, facility))
    assert second["changed"] == 0
    assert second["pending"] == []

    rows_written = tx.run(
        """
        MATCH (s:FacilitySignal {facility_id:$facility})
        WHERE s.plant_context_state IS NOT NULL
        RETURN s.id AS id, [k IN $keys WHERE s[k] IS NOT NULL] AS stored,
               EXISTS { MATCH (s)-[:AT_FACILITY]->(:Facility {id:$facility}) } AS scoped
        """,
        facility=facility,
        keys=list(VOCABULARY_KEYS),
    ).data()
    assert all(row["scoped"] for row in rows_written)
    bare = next(row for row in rows_written if row["id"].endswith("/bare"))
    assert bare["stored"] == [
        "vocabulary_quantity_state",
        "vocabulary_unit_state",
        "vocabulary_role_state",
    ]


def _snapshot(view, facility):
    from imas_codex.discovery.signals.vocabulary_graph import snapshot_cohort

    return snapshot_cohort(view, facility)
