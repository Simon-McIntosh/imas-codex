"""Bare ``[name]`` bracket normalization on the docs accept path.

The docs LLM sometimes writes a related-name mention as a bare ``[name]``
bracket despite the prompt rule; Markdown renders these broken. A
per-rotation reconcile sweep cannot close this on its own — a doc written by
a late in-flight task finalizes AFTER the sweep has run and reaches
``accepted`` still carrying a bare bracket. Normalization therefore happens
at acceptance (:func:`persist_reviewed_docs`), which no in-flight write can
outrun, with the post-drain reconcile retained as a belt-and-suspenders net.

Covers:
- scoped ``_normalize_bare_doc_links(gc, sn_id=...)`` links live names and
  strips dead/unknown tokens (single-node, no full-catalogue scan)
- ``persist_reviewed_docs`` invokes the scoped normalize ONLY when the doc
  promotes to ``accepted`` (not on ``reviewed`` / ``exhausted``)
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_store import (
    graph_audits as _owning_graph_audits,
    graph_catalog as _owning_graph_catalog,
    graph_doc_links as _owning_graph_doc_links,
    graph_foundation as _owning_graph_foundation,
    graph_review_persistence as _owning_graph_review_persistence,
)

_GC_PATH = "imas_codex.standard_names.graph_store.graph_foundation.GraphClient"


class _RecordingGC:
    """Fake GraphClient capturing every ``query(cypher, **params)`` call.

    ``responses`` is consumed in order, one per ``query`` call. The captured
    ``calls`` list lets a test assert what was written.
    """

    def __init__(self, responses: list[list[dict]]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params: Any) -> list[dict]:
        self.calls.append((cypher, params))
        if self._responses:
            return self._responses.pop(0)
        return []


def test_scoped_normalize_links_live_and_strips_dead():
    """A bare [live] becomes a link; a bare [dead]/[unknown] is stripped."""
    from imas_codex.standard_names.graph_store.graph_doc_links import (
        _normalize_bare_doc_links,
    )

    doc = "Related to [poloidal_flux] and the legacy [dead_name] term."
    gc = _RecordingGC(
        responses=[
            # fetch the single node's doc
            [{"id": "area_at_plasma_boundary", "docs": doc}],
            # liveness of the candidate tokens — only poloidal_flux is live
            [{"id": "poloidal_flux"}],
            # the UNWIND write (result unused)
            [],
        ]
    )

    n = _normalize_bare_doc_links(gc, sn_id="area_at_plasma_boundary")

    assert n == 1
    # The liveness query must be scoped to ONLY the doc's tokens (no full scan).
    liveness_cypher, liveness_params = gc.calls[1]
    assert "s.id IN $toks" in liveness_cypher
    assert set(liveness_params["toks"]) == {"poloidal_flux", "dead_name"}
    # The write must link the live name and strip the dead one to plain text.
    write_cypher, write_params = gc.calls[2]
    written = write_params["items"][0]["doc"]
    assert "[poloidal_flux](name:poloidal_flux)" in written
    assert "[dead_name]" not in written
    assert "legacy dead_name term" in written


def test_scoped_normalize_noop_when_no_brackets():
    """A doc with no bare brackets produces no write."""
    from imas_codex.standard_names.graph_store.graph_doc_links import (
        _normalize_bare_doc_links,
    )

    gc = _RecordingGC(responses=[[]])  # fetch returns nothing (no '[' docs)
    assert _normalize_bare_doc_links(gc, sn_id="plasma_current") == 0


def test_the_shared_fixture_asks_for_two_distinct_ports(monkeypatch):
    """The port pair must be two ports held at once, not one asked for twice.

    The kernel is free to hand an ephemeral port back once the socket that
    held it is closed, so two single-port calls in a row can return the same
    number twice. The fake pool below models exactly that: a port becomes
    available again only when its socket closes. A pair chosen one socket at
    a time therefore collides in the model, while a pair chosen with both
    sockets still open does not -- so this test fails if ``_free_port_pair``
    reverts to two sequential single-port calls.
    """
    from types import SimpleNamespace

    from tests.standard_names import conftest as sn_conftest

    class _PoolSocket:
        _live: set = set()

        def __init__(self) -> None:
            self._port: int | None = None

        def bind(self, _address) -> None:
            port = 49152
            while port in {sock._port for sock in _PoolSocket._live}:
                port += 1
            self._port = port
            _PoolSocket._live.add(self)

        def getsockname(self):
            return ("127.0.0.1", self._port)

        def close(self) -> None:
            _PoolSocket._live.discard(self)

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            self.close()

    def two_ports_one_socket_at_a_time() -> list[int]:
        ports = []
        for _ in range(2):
            with _PoolSocket() as sock:
                sock.bind(("127.0.0.1", 0))
                ports.append(sock.getsockname()[1])
        return ports

    # The model reproduces the hazard: the sequential pattern it penalises
    # really does return one port twice. Without this, a distinct pair would
    # prove nothing about the fixture's order of operations, because the fake
    # could be handing out distinct ports regardless.
    assert two_ports_one_socket_at_a_time() == [49152, 49152]

    monkeypatch.setattr(sn_conftest, "socket", SimpleNamespace(socket=_PoolSocket))
    bolt_port, http_port = sn_conftest._free_port_pair()

    assert bolt_port != http_port


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_normalize_prose_against_disposable_graph(disposable_neo4j_empty):
    """The real Cypher reads exact live ids and writes only changed documentation."""
    from imas_codex.standard_names.graph_store.graph_doc_links import (
        _normalize_bare_doc_links,
    )

    assert os.environ["IMAS_CODEX_TEST_NEO4J_EPHEMERAL"] == "1"
    assert os.environ["IMAS_CODEX_TEST_NEO4J_URI"] == disposable_neo4j_empty
    doc = (
        "Related [poloidal flux], [unknown phrase], and [dead name]. "
        "The [documented name](name:documented_name) is this quantity. "
        "Math $[a b]$ and units [10^20 m^-3], [0.1 s], [2 GeV], "
        "[m^-3], [K], and [m^2 s^-1] remain; [co-passing] is prose."
    )
    with GraphClient() as graph:
        assert graph.uri == disposable_neo4j_empty
        graph.query(
            "UNWIND $names AS item CREATE (n:StandardName) "
            "SET n.id = item.id, n.name_stage = item.stage, "
            "n.documentation = item.docs",
            names=[
                {"id": "documented_name", "stage": "accepted", "docs": doc},
                {"id": "poloidal_flux", "stage": "reviewed", "docs": None},
                {"id": "dead_name", "stage": "superseded", "docs": None},
            ],
        )
        assert (
            graph.query("MATCH (n:StandardName) RETURN count(n) AS total")[0]["total"]
            == 3
        )
        changes = _normalize_bare_doc_links(
            graph, sn_id="documented_name", dry_run=True
        )
        assert len(changes) == 1
        assert changes[0]["before"] == doc
        assert (
            graph.query(
                "MATCH (n:StandardName {id: 'documented_name'}) RETURN n.documentation AS doc"
            )[0]["doc"]
            == doc
        )
        after = changes[0]["after"]
        assert "[poloidal flux](name:poloidal_flux)" in after
        assert "unknown phrase" in after and "[unknown phrase]" not in after
        assert "dead name" in after and "[dead name]" not in after
        assert "documented name is this quantity" in after
        assert "$[a b]$" in after
        for notation in (
            "[10^20 m^-3]",
            "[0.1 s]",
            "[2 GeV]",
            "[m^-3]",
            "[K]",
            "[m^2 s^-1]",
        ):
            assert notation in after
        assert "co-passing is prose" in after
        assert {action["action"] for action in changes[0]["actions"]} == {
            "linked",
            "unbracketed",
            "self-link removed",
        }
        assert _normalize_bare_doc_links(graph, sn_id="documented_name") == 1
        assert (
            graph.query(
                "MATCH (n:StandardName {id: 'documented_name'}) RETURN n.documentation AS doc"
            )[0]["doc"]
            == after
        )
        assert _normalize_bare_doc_links(graph, sn_id="documented_name") == 0
        assert (
            _normalize_bare_doc_links(graph, sn_id="documented_name", dry_run=True)
            == []
        )


def _patch_persist_gc(gc):
    return patch(_GC_PATH, return_value=gc)


def test_persist_reviewed_docs_normalizes_on_accept():
    """On accept, persist_reviewed_docs invokes the scoped bare-link normalize."""
    import imas_codex.standard_names.graph_ops as g
    from imas_codex.standard_names.review import holds

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    transaction = MagicMock()
    transaction.run = MagicMock(
        side_effect=[
            [{"docs_chain_length": 0, "documentation": None, "edit_status": None}],
            [{"id": "area_at_plasma_boundary"}],  # SET committed → accepted
        ]
    )
    gc.session.return_value.__enter__.return_value.begin_transaction.return_value = (
        transaction
    )

    with (
        _patch_persist_gc(gc),
        patch.object(_owning_graph_doc_links, "_normalize_bare_doc_links") as norm,
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_audits, "bump_sn_run_counter"),
        patch.object(_owning_graph_foundation, "docs_accepted", return_value=True),
        patch.object(holds, "clear_holds_not_raised", return_value=0),
    ):
        result = _owning_graph_review_persistence.persist_reviewed_docs(
            sn_id="area_at_plasma_boundary",
            claim_token="tok-1",
            score=0.95,
            model="test/model",
            min_score=0.85,
            rotation_cap=3,
            skip_review_node=True,
            resolution_method="quorum_consensus",
            reviewer_chain_size=3,
        )

    assert result == "accepted"
    norm.assert_called_once()
    # scoped to this node
    _args, kwargs = norm.call_args
    assert kwargs.get("sn_id") == "area_at_plasma_boundary"


def test_persist_reviewed_docs_skips_normalize_when_not_accepted():
    """A below-threshold review (→ reviewed) does NOT run the accept normalize."""
    import imas_codex.standard_names.graph_ops as g

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    transaction = MagicMock()
    transaction.run = MagicMock(
        side_effect=[
            [{"docs_chain_length": 0, "documentation": None, "edit_status": None}],
            [{"id": "area_at_plasma_boundary"}],  # SET committed → reviewed
        ]
    )
    gc.session.return_value.__enter__.return_value.begin_transaction.return_value = (
        transaction
    )

    with (
        _patch_persist_gc(gc),
        patch.object(_owning_graph_doc_links, "_normalize_bare_doc_links") as norm,
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_audits, "bump_sn_run_counter"),
    ):
        result = _owning_graph_review_persistence.persist_reviewed_docs(
            sn_id="area_at_plasma_boundary",
            claim_token="tok-1",
            score=0.40,
            model="test/model",
            min_score=0.85,
            rotation_cap=3,
            skip_review_node=True,
            resolution_method="quorum_consensus",
            reviewer_chain_size=3,
        )

    assert result == "reviewed"
    norm.assert_not_called()
