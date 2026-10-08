"""Live graph checks for signal rescan and status reconciliation."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from imas_codex.discovery.signals import parallel
from imas_codex.graph import GraphClient
from imas_codex.graph.models import FacilitySignalStatus

pytestmark = pytest.mark.graph


class _TransactionClient:
    """Keep discovery queries in a transaction that the test rolls back."""

    def __init__(self, transaction):
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def query(self, statement, **params):
        return [dict(row) for row in self.transaction.run(statement, **params)]


def test_rescan_preserves_checked_signal_status():
    signal_id = f"status-check:{uuid4()}"
    facility_id = f"status-check:{uuid4()}"
    with GraphClient() as graph, graph.session() as session:
        transaction = session.begin_transaction()
        try:
            transaction.run("MERGE (:Facility {id: $id})", id=facility_id).consume()
            transaction.run(
                """CREATE (s:FacilitySignal {
                    id: $id, facility_id: $facility, name: 'prior',
                    status: $checked, checked_at: datetime(), enriched_at: datetime()
                })""",
                id=signal_id,
                facility=facility_id,
                checked=FacilitySignalStatus.checked.value,
            ).consume()
            scanned = {
                "id": signal_id,
                "facility_id": facility_id,
                "name": "rescanned",
                "status": FacilitySignalStatus.discovered.value,
            }
            with patch.object(
                parallel, "GraphClient", return_value=_TransactionClient(transaction)
            ):
                assert parallel.ingest_discovered_signals([scanned]) == 1
            result = transaction.run(
                "MATCH (s:FacilitySignal {id: $id}) "
                "RETURN s.name AS name, s.status AS status, s.checked_at AS checked_at",
                id=signal_id,
            ).single()
            assert result["name"] == "rescanned"
            assert result["checked_at"] is not None
            assert result["status"] == FacilitySignalStatus.checked.value
        finally:
            transaction.rollback()


def test_reconcile_uses_stage_timestamps_without_reopening_terminal_signals():
    facility_id = f"status-check:{uuid4()}"
    rows = [
        ("checked-lag", "discovered", True, True, "checked"),
        ("enriched-lag", "discovered", False, True, "enriched"),
        ("insufficient", "underspecified", False, True, "underspecified"),
        ("skipped", "skipped", True, True, "skipped"),
    ]
    with GraphClient() as graph, graph.session() as session:
        transaction = session.begin_transaction()
        try:
            transaction.run(
                """UNWIND $rows AS row
                CREATE (:FacilitySignal {
                    id: $facility + ':' + row[0], facility_id: $facility,
                    status: row[1],
                    checked_at: CASE WHEN row[2] THEN datetime() ELSE null END,
                    enriched_at: CASE WHEN row[3] THEN datetime() ELSE null END
                })""",
                facility=facility_id,
                rows=rows,
            ).consume()
            with patch.object(
                parallel, "GraphClient", return_value=_TransactionClient(transaction)
            ):
                preview = parallel.reconcile_signal_statuses(facility_id)
                assert preview["updated"] == 0
                assert sum(row["count"] for row in preview["candidates"]) == 2
                with pytest.raises(ValueError, match="facility is required"):
                    parallel.reconcile_signal_statuses(apply=True)
                repair = parallel.reconcile_signal_statuses(facility_id, apply=True)
                assert repair["updated"] == 2
            actual = transaction.run(
                "MATCH (s:FacilitySignal {facility_id: $facility}) "
                "RETURN s.id AS id, s.status AS status",
                facility=facility_id,
            )
            assert {row["id"].split(":")[-1]: row["status"] for row in actual} == {
                name: expected for name, _, _, _, expected in rows
            }
        finally:
            transaction.rollback()
