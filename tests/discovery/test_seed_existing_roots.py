"""Root seeding is additive: an existing root keeps its stored state.

seed_facility_roots is called with --root on every scoped run, and the same
root is often already scored or triaged from a prior run. Reseeding it must
create only the rows that do not yet exist and leave an existing row's status,
depth, scores and timestamps unchanged; reprocessing is what --reset-to is for.

The fake session models Neo4j's ``MERGE ... ON CREATE SET`` so the assertions
exercise the real GraphClient.create_nodes write the seed performs, against an
in-memory store that starts pre-populated with the scored row.
"""

import re

import pytest

from imas_codex.discovery.paths.frontier import seed_facility_roots
from imas_codex.graph.client import GraphClient
from imas_codex.graph.schema import get_schema

_MERGE_RE = re.compile(
    r"MERGE \(n:(\w+) \{id: item\.id\}\)\s+"
    r"(ON CREATE SET|SET) n \+= item"
)


class FakeSession:
    """Minimal in-memory model of ``MERGE (n) SET`` and ``ON CREATE SET``."""

    def __init__(self, store: dict) -> None:
        self._store = store
        self.queries: list[str] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def run(self, query: str, **params: object) -> None:
        self.queries.append(query)
        match = _MERGE_RE.search(query)
        if not match:
            return
        label, verb = match.group(1), match.group(2)
        create_only = verb == "ON CREATE SET"
        for item in params["batch"]:  # type: ignore[union-attr]
            key = (label, item["id"])
            existed = key in self._store
            if not existed:
                self._store[key] = {}
            if create_only and existed:
                continue
            self._store[key].update(item)


class _NoExclusion:
    def should_exclude(self, path: str) -> tuple[bool, str | None]:
        return False, None


@pytest.fixture
def graph(monkeypatch):
    """Patch the seed's collaborators and return the in-memory node store."""
    store: dict = {}
    sessions: list[FakeSession] = []

    def make_session() -> FakeSession:
        sess = FakeSession(store)
        sessions.append(sess)
        return sess

    def factory(*_: object, **__: object) -> GraphClient:
        client = object.__new__(GraphClient)
        client._schema = get_schema()
        client._driver = None
        client.session = make_session  # type: ignore[method-assign]
        return client

    monkeypatch.setattr("imas_codex.graph.GraphClient", factory)
    monkeypatch.setattr(
        "imas_codex.discovery.paths.frontier._dedupe_paths_by_inode",
        lambda facility, paths: (list(paths), []),
    )
    monkeypatch.setattr(
        "imas_codex.config.discovery_config.get_exclusion_config_for_facility",
        lambda facility: _NoExclusion(),
    )
    store["_sessions"] = sessions
    return store


def test_seed_existing_root_keeps_status(graph):
    facility = "jt-60sa"
    path = "/analysis/src/SAeqfame"
    graph[("FacilityPath", f"{facility}:{path}")] = {
        "id": f"{facility}:{path}",
        "facility_id": facility,
        "path": path,
        "status": "scored",
        "depth": 0,
        "triage_composite": 0.60,
        "discovered_at": "2026-10-05T10:00:00+00:00",
    }

    seed_facility_roots(facility, root_paths=[path])

    node = graph[("FacilityPath", f"{facility}:{path}")]
    assert node["status"] == "scored"
    assert node["triage_composite"] == 0.60
    assert node["depth"] == 0
    assert node["discovered_at"] == "2026-10-05T10:00:00+00:00"


def test_seed_new_root_created_discovered_at_depth_zero(graph):
    facility = "jt-60sa"
    path = "/analysis/src/getseldata_v4.2"

    seed_facility_roots(facility, root_paths=[path])

    node = graph[("FacilityPath", f"{facility}:{path}")]
    assert node["status"] == "discovered"
    assert node["depth"] == 0


def test_seed_writes_create_only(graph):
    seed_facility_roots("jt-60sa", root_paths=["/analysis/src/SAeqfame"])

    queries = [q for sess in graph["_sessions"] for q in sess.queries]
    assert any("ON CREATE SET n += item" in q for q in queries)
