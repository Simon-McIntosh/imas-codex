"""GraphClient.create_nodes upsert versus create-only write semantics.

create_nodes is the node-creation owner for additive callers: given an id that
already exists it must be able to leave the stored row untouched, so a seed
that names an existing root does not reset that row's status or scores. The
default stays an upsert.

The fake session models the two Cypher forms the method emits
(``MERGE ... SET`` for upsert, ``MERGE ... ON CREATE SET`` for create-only)
so the assertions drive the real method's query choice against Neo4j's MERGE
semantics rather than a re-implementation of the method.
"""

import re

from imas_codex.graph.client import GraphClient
from imas_codex.graph.schema import get_schema

_MERGE_RE = re.compile(
    r"MERGE \(n:(\w+) \{id: item\.id\}\)\s+"
    r"(ON CREATE SET|SET) n \+= item"
)


class _Result:
    """Stand-in for a Neo4j result, exposing the aggregate row to ``single``."""

    def __init__(self, row: dict | None) -> None:
        self._row = row

    def single(self) -> dict | None:
        return self._row


class FakeSession:
    """Minimal in-memory model of ``MERGE (n) SET`` and ``ON CREATE SET``."""

    def __init__(self, store: dict) -> None:
        self._store = store
        self.queries: list[str] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def run(self, query: str, **params: object) -> _Result:
        self.queries.append(query)
        match = _MERGE_RE.search(query)
        created = 0
        if match:
            label, verb = match.group(1), match.group(2)
            create_only = verb == "ON CREATE SET"
            for item in params["batch"]:  # type: ignore[union-attr]
                key = (label, item["id"])
                existed = key in self._store
                if not existed:
                    self._store[key] = {}
                    created += 1
                if create_only and existed:
                    continue
                self._store[key].update(item)
        # The create-only form aggregates the rows it created into a RETURN.
        if "RETURN" in query:
            return _Result({"created": created})
        return _Result(None)


def _client(store: dict) -> GraphClient:
    """A GraphClient whose session writes into an in-memory store."""
    client = object.__new__(GraphClient)
    client._schema = get_schema()
    client._driver = None

    sessions: list[FakeSession] = []

    def make_session() -> FakeSession:
        sess = FakeSession(store)
        sessions.append(sess)
        return sess

    client.session = make_session  # type: ignore[method-assign]
    client._sessions = sessions  # type: ignore[attr-defined]
    return client


def test_default_updates_existing_node():
    store = {
        ("FacilityPath", "jt-60sa:/analysis/src/SAeqfame"): {
            "id": "jt-60sa:/analysis/src/SAeqfame",
            "status": "scored",
            "triage_composite": 0.60,
        }
    }
    client = _client(store)

    client.create_nodes(
        "FacilityPath",
        [
            {
                "id": "jt-60sa:/analysis/src/SAeqfame",
                "facility_id": "jt-60sa",
                "status": "discovered",
            }
        ],
    )

    node = store[("FacilityPath", "jt-60sa:/analysis/src/SAeqfame")]
    assert node["status"] == "discovered"
    assert node["triage_composite"] == 0.60  # untouched property preserved


def test_create_only_leaves_existing_node_unchanged():
    store = {
        ("FacilityPath", "jt-60sa:/analysis/src/SAeqfame"): {
            "id": "jt-60sa:/analysis/src/SAeqfame",
            "status": "scored",
            "depth": 0,
            "triage_composite": 0.60,
            "discovered_at": "2026-10-05T10:00:00+00:00",
        }
    }
    client = _client(store)

    client.create_nodes(
        "FacilityPath",
        [
            {
                "id": "jt-60sa:/analysis/src/SAeqfame",
                "facility_id": "jt-60sa",
                "status": "discovered",
                "depth": 0,
                "discovered_at": "2026-10-06T07:45:00+00:00",
            }
        ],
        create_only=True,
    )

    node = store[("FacilityPath", "jt-60sa:/analysis/src/SAeqfame")]
    assert node["status"] == "scored"
    assert node["triage_composite"] == 0.60
    assert node["discovered_at"] == "2026-10-05T10:00:00+00:00"


def test_create_only_creates_new_node_with_props():
    store: dict = {}
    client = _client(store)

    client.create_nodes(
        "FacilityPath",
        [
            {
                "id": "jt-60sa:/analysis/src/getseldata_v4.2",
                "facility_id": "jt-60sa",
                "status": "discovered",
                "depth": 0,
            }
        ],
        create_only=True,
    )

    node = store[("FacilityPath", "jt-60sa:/analysis/src/getseldata_v4.2")]
    assert node["status"] == "discovered"
    assert node["depth"] == 0


def test_query_form_differs_by_option():
    store: dict = {}
    client = _client(store)

    client.create_nodes("FacilityPath", [{"id": "jt-60sa:/a", "status": "x"}])
    default_query = client._sessions[-1].queries[-1]

    client.create_nodes(
        "FacilityPath", [{"id": "jt-60sa:/b", "status": "x"}], create_only=True
    )
    create_only_query = client._sessions[-1].queries[-1]

    assert "SET n += item" in default_query
    assert "ON CREATE SET" not in default_query
    assert "ON CREATE SET n += item" in create_only_query


def test_create_only_reports_created_count():
    store = {
        ("FacilityPath", "jt-60sa:/a"): {"id": "jt-60sa:/a", "status": "scored"},
        ("FacilityPath", "jt-60sa:/b"): {"id": "jt-60sa:/b", "status": "scored"},
    }
    client = _client(store)

    result = client.create_nodes(
        "FacilityPath",
        [
            {"id": "jt-60sa:/a", "status": "discovered"},
            {"id": "jt-60sa:/b", "status": "discovered"},
            {"id": "jt-60sa:/c", "status": "discovered"},
        ],
        create_only=True,
    )

    assert result["created"] == 1
    assert result["processed"] == 3


def test_default_reports_processed_and_no_created_count():
    store = {
        ("FacilityPath", "jt-60sa:/a"): {"id": "jt-60sa:/a", "status": "scored"},
    }
    client = _client(store)

    result = client.create_nodes(
        "FacilityPath",
        [
            {"id": "jt-60sa:/a", "status": "discovered"},
            {"id": "jt-60sa:/b", "status": "discovered"},
        ],
    )

    assert result["processed"] == 2
    assert "created" not in result


def test_create_only_empty_items_reports_zero_created():
    client = _client({})

    result = client.create_nodes("FacilityPath", [], create_only=True)

    assert result["created"] == 0
    assert result["processed"] == 0
