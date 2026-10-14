"""One statement-capturing graph-client double, shared across test suites.

Several suites need to assert on the exact Cypher a graph client receives —
a predicate's null semantics, a clause separator, a conditional assignment —
without a live server. Each suite previously carried its own recording double,
so a change to how a statement is captured had to be repeated in every copy.
This module holds the one double. A suite that needs the recorded statements
to drive a model of graph state subclasses :class:`RecordingGraphClient` and
overrides :meth:`~RecordingGraphClient.query`, keeping its fixture logic on top
of the shared capture rather than beside it.
"""

from __future__ import annotations

from typing import Any


class RecordingGraphClient:
    """Record every ``query`` and answer with no rows.

    ``calls`` holds one ``(cypher, params)`` pair per statement a client
    receives, in order, so a test can assert on the rendered statement and the
    parameters bound to it. The default answer is an empty result, which is
    what a capture-only test wants; a test that needs rows to flow back
    subclasses this and overrides :meth:`query`.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any) -> list[dict]:
        self.calls.append((cypher, params))
        return []
