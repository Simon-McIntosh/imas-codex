"""One transaction-shaped graph-client double, shared across the suites.

The persistence paths these suites exercise run their compare-and-set
statements inside ``gc.session().begin_transaction()`` and read their rows back
from the same statement. A double that answers only ``gc.query`` therefore
records nothing and returns nothing for those paths.

:class:`TransactionDouble` answers both surfaces from one recorder: the
client's ``query`` is a ``MagicMock``, and the session transaction's ``run``
forwards to it, so ``double.query.call_args_list`` holds every statement the
code sent in order whichever surface it used. ``rows`` seeds the answers in
call order; a test that needs per-statement behaviour drives
``double.transaction.run`` itself.

The class extends :class:`~tests.recording_graph_client.RecordingGraphClient`,
so the seeded path also fills ``calls`` with the plain ``(cypher, params)``
pairs that base class promises.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from tests.recording_graph_client import RecordingGraphClient

__all__ = ["TransactionDouble", "transaction_double"]


class TransactionDouble(RecordingGraphClient):
    """A ``GraphClient`` double answering ``query`` and ``session``.

    ``query`` is a ``MagicMock`` so a test can assert on ``call_args_list``.
    ``session()`` returns a context manager whose ``begin_transaction()``
    returns :attr:`transaction`; that transaction's ``run`` forwards to
    ``query`` unless the test replaces it, which keeps one ordered statement
    log for both surfaces.
    """

    def __init__(
        self,
        rows: list[Any] | None = None,
        *,
        by_statement: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._rows: list[Any] | None = None if rows is None else list(rows)
        self._by_statement = dict(by_statement or {})
        self.query = MagicMock(side_effect=self._answer)
        self.transaction = MagicMock()
        self.transaction.run.side_effect = self._run_in_transaction
        self.transaction.commit = MagicMock()
        self.transaction.rollback = MagicMock()
        self.transaction.close = MagicMock()
        self._session = MagicMock()
        self._session.begin_transaction = MagicMock(return_value=self.transaction)
        self._session.__enter__ = MagicMock(return_value=self._session)
        self._session.__exit__ = MagicMock(return_value=False)

    def __enter__(self) -> "TransactionDouble":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def session(self):
        """Return the one session, so every with-block shares its transaction."""
        return self._session

    def _answer(self, cypher: str, **params: Any) -> list[dict]:
        self.calls.append((cypher, params))
        for marker, answer in self._by_statement.items():
            if marker in cypher:
                return list(answer) if not callable(answer) else list(answer(cypher, params))
        if not self._rows:
            return []
        return self._rows.pop(0)

    def _run_in_transaction(self, cypher: str, **params: Any) -> list[dict]:
        return self.query(cypher, **params)


def transaction_double(
    rows: list[Any] | None = None, *, by_statement: dict[str, Any] | None = None
) -> TransactionDouble:
    """Build a :class:`TransactionDouble` seeded with ``rows`` in call order."""
    return TransactionDouble(rows, by_statement=by_statement)