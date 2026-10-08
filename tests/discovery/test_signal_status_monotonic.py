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
