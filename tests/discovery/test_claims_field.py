"""Assert the generic claim release honours its claimed-field parameters.

``release_claim`` and ``release_claims_batch`` clear ``claimed_at`` by default,
which is what every existing caller relies on. The candidate stage claims
sources on ``mapping_claimed_at`` and its token on ``mapping_claim_token``, so
both routines take a ``claimed_field`` (and an optional ``token_field``). These
tests pin both halves: the default query still clears ``claimed_at`` and no
token, and a named field clears exactly the field it was given.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base.claims import release_claim, release_claims_batch


class _FakeGraph:
    """Graph client that records the rendered statement and bound parameters."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.queries: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        self.queries.append((" ".join(cypher.split()), kwargs))
        return self.rows


@pytest.fixture
def fake_graph(monkeypatch):
    graph = _FakeGraph([{"released": 2}])
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    return graph


def _last_statement(graph) -> str:
    return graph.queries[-1][0]


def test_release_claim_defaults_to_claimed_at(fake_graph):
    release_claim("SignalSource", "src-1")

    statement = _last_statement(fake_graph)
    assert "SET n.claimed_at = null" in statement
    assert "mapping_claimed_at" not in statement
    assert "mapping_claim_token" not in statement


def test_release_claim_honours_claimed_and_token_fields(fake_graph):
    release_claim(
        "SignalSource",
        "src-1",
        claimed_field="mapping_claimed_at",
        token_field="mapping_claim_token",
    )

    statement = _last_statement(fake_graph)
    assert "SET n.mapping_claimed_at = null, n.mapping_claim_token = null" in statement
    assert fake_graph.queries[-1][1]["id"] == "src-1"


def test_release_claims_batch_defaults_to_claimed_at(fake_graph):
    released = release_claims_batch("SignalSource", ["src-1", "src-2"])

    assert released == 2
    statement = _last_statement(fake_graph)
    assert "n.claimed_at IS NOT NULL" in statement
    assert "SET n.claimed_at = null" in statement
    assert "mapping_claimed_at" not in statement
    assert fake_graph.queries[-1][1]["ids"] == ["src-1", "src-2"]


def test_release_claims_batch_honours_claimed_field(fake_graph):
    release_claims_batch(
        "SignalSource",
        ["src-1"],
        claimed_field="mapping_claimed_at",
        token_field="mapping_claim_token",
    )

    statement = _last_statement(fake_graph)
    assert "n.mapping_claimed_at IS NOT NULL" in statement
    assert "SET n.mapping_claimed_at = null, n.mapping_claim_token = null" in statement


def test_release_claims_batch_empty_list_returns_zero(fake_graph):
    assert release_claims_batch("SignalSource", []) == 0
    assert fake_graph.queries == []
