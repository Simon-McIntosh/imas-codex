"""The stored domain decision and the one writer that sets a name's domain.

The candidate derivation, the confidence floor, a failed call, a correction and
a derived parent are exercised through a stub graph, so no live database is
needed. The writer's Cypher itself is executed against a disposable Neo4j the
test starts empty and plants its own fixture nodes in — it must pass, not skip.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import domain_decision as dd
from tests.standard_names.test_source_target_reconciliation import disposable_neo4j


class _StubGraph:
    """A graph stand-in: canned rows for the read, recording for the write."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.reads: list[tuple[str, dict[str, Any]]] = []
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN sn.physics_domain AS physics_domain" in cypher:
            self.reads.append((cypher, params))
            return [dict(self.row)]
        self.writes.append((cypher, params))
        return [{"sn_id": params.get("sn_id")}]

    def close(self) -> None:
        pass


def _row(
    *,
    physics_domain: str | None = None,
    own_domains: list[str] | None = None,
    child_dd_domains: list[str] | None = None,
    child_stored_domains: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "physics_domain": physics_domain,
        "own_domains": own_domains or [],
        "child_ids": [],
        "child_dd_domains": child_dd_domains or [],
        "child_stored_domains": child_stored_domains or [],
    }


def _answer(choice: str, probability: float) -> dict[str, Any]:
    return {dd.QUESTION: {"choice": choice, "probabilities": {choice: probability}}}


def _fake_batch(payload: Any) -> Any:
    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        return [(payload, 0.0)], 0.0

    return decide


IDS_MAP = {"equilibrium": "equilibrium", "core_profiles": "core_plasma_physics"}


def test_candidate_domains_carry_umbrella_and_general() -> None:
    """A path's domain, its IDS's prior, the band umbrella and general are offered."""
    sources = {"dd:equilibrium/time_slice/profiles_1d/psi": "equilibrium"}
    found = dd.candidate_domains(sources, IDS_MAP)
    assert "equilibrium" in found
    assert "core_plasma_physics" in found  # the core_physics band umbrella
    assert found[-1] == "general"
    assert dd.band_umbrella("equilibrium") == "core_plasma_physics"


def test_question_carries_each_options_scope_text() -> None:
    question = dd.build_question(("equilibrium", "general"))
    criteria = question[dd.QUESTION]["criteria"]
    assert set(criteria) == {"equilibrium", "general"}
    assert "flux-surface" in criteria["equilibrium"]
    assert question[dd.QUESTION]["type"] == "choice"


def test_single_candidate_takes_it_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One option is taken directly, and no batch call is made."""
    calls = {"n": 0}

    async def boom(*_a, **_k):  # noqa: ANN002
        calls["n"] += 1
        raise AssertionError("no call expected")

    monkeypatch.setattr(dd, "decide_batch", boom)
    decision = dd.decide_domain_for(
        "solo", "", {"dd:general/x/y": "general"}, {}, model="m"
    )
    assert decision.decided and decision.choice == "general"
    assert calls["n"] == 0


def test_below_floor_is_undecided_and_keeps_its_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probability below the floor is undecided and never overwrites the scalar."""
    monkeypatch.setattr(dd, "decide_batch", _fake_batch(_answer("equilibrium", 0.5)))
    sources = {"dd:equilibrium/time_slice/profiles_1d/psi": "equilibrium"}
    decision = dd.decide_domain_for("n", "", sources, IDS_MAP, model="m")
    assert not decision.decided
    assert decision.undecided_reason and "below the floor" in decision.undecided_reason

    gc = _StubGraph(_row(physics_domain="transport", own_domains=["equilibrium"]))
    written = dd.write_name_domain(gc, "sn:n", decision)
    assert written.physics_domain == "transport"
    assert not written.decided


def test_failed_call_is_undecided(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that returns nothing is undecided, with the reason named."""
    monkeypatch.setattr(dd, "decide_batch", _fake_batch(None))
    sources = {"dd:equilibrium/time_slice/profiles_1d/psi": "equilibrium"}
    decision = dd.decide_domain_for("n", "", sources, IDS_MAP, model="m")
    assert not decision.decided
    assert "no answer" in (decision.undecided_reason or "")


def test_choice_outside_candidates_is_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dd, "decide_batch", _fake_batch(_answer("not_a_domain", 0.99)))
    sources = {"dd:equilibrium/time_slice/profiles_1d/psi": "equilibrium"}
    decision = dd.decide_domain_for("n", "", sources, IDS_MAP, model="m")
    assert not decision.decided
    assert "not one of the candidates" in (decision.undecided_reason or "")


def test_correction_overrides_the_decision() -> None:
    """A recorded correction stands above the stored decision and keeps its reason."""
    decision = dd.DomainDecision(
        name="n",
        candidates=("general", "equilibrium"),
        choice="equilibrium",
        probability=0.9,
        model="m",
        decided_at="2026-10-20T00:00:00+00:00",
    )
    gc = _StubGraph(_row(physics_domain="equilibrium", own_domains=["transport"]))
    written = dd.write_name_domain(
        gc,
        "sn:n",
        decision,
        correction=dd.DomainCorrection("transport", "measured vs model"),
    )
    assert written.physics_domain == "transport"
    assert written.correction == "transport"
    _, params = gc.writes[-1]
    assert params["correction"] == "transport"
    assert params["correction_reason"] == "measured vs model"


def test_derived_parent_unions_its_children() -> None:
    """A name with no DD source takes the union of its children's domains."""
    gc = _StubGraph(
        _row(
            physics_domain=None,
            child_dd_domains=["equilibrium", "transport"],
            child_stored_domains=[["fast_particles"]],
        )
    )
    decision = dd.DomainDecision(
        name="parent",
        candidates=("equilibrium", "general", "transport"),
        choice="equilibrium",
        probability=0.8,
        model="m",
        decided_at="2026-10-20T00:00:00+00:00",
    )
    written = dd.write_name_domain(gc, "sn:parent", decision)
    assert set(written.source_domains) == {"equilibrium", "transport", "fast_particles"}
    _, params = gc.writes[-1]
    assert params["source_domains"] == sorted(written.source_domains)
    assert params["edge_domains"] == sorted(written.source_domains)


@pytest.fixture
def graph(disposable_neo4j: tuple[str, str]) -> Any:
    from imas_codex.graph.client import GraphClient

    uri, password = disposable_neo4j
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="name-domain-writer",
    )
    yield client
    client.close()


@pytest.mark.graph
def test_writer_sets_the_three_together_on_a_disposable_graph(graph: Any) -> None:
    """The scalars and the edges land together, and a stale edge is rewritten."""
    graph.query(
        """
        CREATE (sn:StandardName {id: 'sn:e_temp', physics_domain: null,
                                 source_domains: null})
        CREATE (src:StandardNameSource {id: 'src:1'})
        CREATE (dd:IMASNode {id: 'dd:core_profiles/profiles_1d/electrons/temperature',
                             physics_domain: 'core_plasma_physics'})
        CREATE (stale:PhysicsDomain {id: 'general'})
        CREATE (src)-[:PRODUCED_NAME]->(sn)
        CREATE (src)-[:FROM_DD_PATH]->(dd)
        CREATE (sn)-[:HAS_PHYSICS_DOMAIN]->(stale)
        """
    )
    decision = dd.DomainDecision(
        name="electron_temperature",
        candidates=("core_plasma_physics", "general"),
        choice="core_plasma_physics",
        probability=0.97,
        model="test-model",
        decided_at="2026-10-20T00:00:00+00:00",
    )
    written = dd.write_name_domain(graph, "sn:e_temp", decision)
    assert written.physics_domain == "core_plasma_physics"
    assert set(written.source_domains) == {"core_plasma_physics"}

    rows = graph.query(
        """
        MATCH (sn:StandardName {id: 'sn:e_temp'})
        OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(d:PhysicsDomain)
        RETURN sn.physics_domain AS physics_domain,
               sn.source_domains AS source_domains,
               collect(DISTINCT d.id) AS edge_domains,
               sn.domain_decision_choice AS choice,
               sn.domain_decision_probability AS probability
        """
    )
    assert len(rows) == 1, "the fixture name must still be in the graph"
    row = rows[0]
    assert row["physics_domain"] == "core_plasma_physics"
    assert set(row["source_domains"]) == {"core_plasma_physics"}
    assert set(row["edge_domains"]) == {"core_plasma_physics"}, (
        "the stale general edge must be rewritten to source_domains"
    )
    assert row["choice"] == "core_plasma_physics"
    assert row["probability"] == pytest.approx(0.97)
