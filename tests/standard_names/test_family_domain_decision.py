"""One domain decision per family, read back by every member.

A family holds a quantity, operator, process, entity and locus fixed, so its
members name one concept and must receive one domain. The decision offers the
union of every member's candidate domains, asks one Choice, stores one record
and writes it to every member through the single domain writer.

The read-back runs against a disposable Neo4j seeded with the eight published
species families, so it must pass rather than skip.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import domain_decision as dd, families as fam
from tests.standard_names.conftest import (
    disposable_neo4j as disposable_neo4j,  # noqa: F401
)


def _answer(choice: str, probability: float) -> dict[str, Any]:
    return {dd.QUESTION: {"choice": choice, "probabilities": {choice: probability}}}


def _fake_batch(payload: Any, counter: list[int] | None = None) -> Any:
    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        if counter is not None:
            counter.append(len(states))
        return [(payload, 0.0)], 0.0

    return decide


IDS_MAP: dict[str, str] = {}


# ---------------------------------------------------------------------------
# The family grouping: locus fixed, species and component free
# ---------------------------------------------------------------------------


def test_domain_family_key_holds_the_locus_fixed() -> None:
    """Species and component vary inside a family; the locus does not."""
    electron = {
        "physical_base": "density",
        "geometric_base": "",
        "transformation": "",
        "process": "",
        "object": "",
        "position": "pedestal_top",
        "subject": "electron",
        "component": "",
    }
    argon = dict(electron, subject="argon")
    axis = dict(electron, position="magnetic_axis")
    assert fam.domain_family_key(electron) == fam.domain_family_key(argon), (
        "two species of one quantity at one locus name the same concept"
    )
    assert fam.domain_family_key(electron) != fam.domain_family_key(axis), (
        "the same quantity at a different locus is a different concept"
    )


def test_family_options_are_the_union_across_members() -> None:
    """A member's unusual source neither narrows nor widens the choice alone."""
    members = [
        {"id": "electron_density", "sources": {"dd:core_profiles/x": "transport"}},
        {
            "id": "argon_density_at_pedestal_top",
            "sources": {"dd:edge/y": "edge_plasma_physics"},
        },
    ]
    options = dd.family_candidate_domains(members, IDS_MAP)
    assert {"transport", "edge_plasma_physics", "general"} <= set(options)


def test_one_judgment_is_paid_per_family_not_per_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three members are decided by one Choice call."""
    calls: list[int] = []
    monkeypatch.setattr(
        dd, "decide_batch", _fake_batch(_answer("transport", 0.9), calls)
    )
    members = [
        {
            "id": f"name_{index}",
            "description": "",
            "sources": {f"dd:a/{index}": "transport"},
        }
        for index in range(3)
    ]
    decision = asyncio.run(
        dd.decide_family_domain("family-key", members, IDS_MAP, model="test-model")
    )
    assert decision.decided and decision.choice == "transport"
    assert calls == [1], "one family, one state offered to the model"


def test_a_recorded_correction_applies_to_the_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correction on any member settles the family, and no call is made."""
    calls: list[int] = []
    monkeypatch.setattr(
        dd, "decide_batch", _fake_batch(_answer("transport", 0.9), calls)
    )
    members = [
        {"id": "a", "sources": {"dd:a": "transport"}},
        {
            "id": "b",
            "sources": {"dd:b": "edge_plasma_physics"},
            "domain_correction": "equilibrium",
        },
    ]
    decision = asyncio.run(
        dd.decide_family_domain("key", members, IDS_MAP, model="test-model")
    )
    assert decision.choice == "equilibrium"
    assert decision.corrected is True
    assert calls == []


# ---------------------------------------------------------------------------
# The read-back on a disposable graph
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph(disposable_neo4j: tuple[str, str]) -> Any:
    uri, password = disposable_neo4j
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="family-domain-decision",
    )
    yield client
    client.close()


#: The eight species families the release candidate split, as (invariant and held
#: segments, member subjects). Members are planted with disjoint source domains,
#: which is exactly the state the per-name path cannot make agree.
SPLIT_FAMILIES: list[tuple[dict[str, str], list[str]]] = [
    ({"physical_base": "density"}, ["electron", "argon", "krypton"]),
    (
        {
            "physical_base": "density",
            "transformation": "ratio_of_volume_averaged",
        },
        ["electron", "argon", "krypton"],
    ),
    (
        {"physical_base": "density", "position": "internal_transport_barrier"},
        ["electron", "argon", "krypton"],
    ),
    (
        {"physical_base": "density", "position": "pedestal_top"},
        ["electron", "argon", "krypton"],
    ),
    (
        {"physical_base": "density", "transformation": "line_averaged"},
        ["electron", "argon", "krypton"],
    ),
    (
        {"physical_base": "density", "position": "plasma_boundary"},
        ["electron", "argon", "krypton"],
    ),
    (
        {
            "physical_base": "density",
            "transformation": "flux_surface_averaged",
            "position": "plasma_boundary",
        },
        ["electron", "argon", "krypton"],
    ),
    (
        {
            "physical_base": "velocity",
            "geometric_base": "toroidal",
            "position": "magnetic_axis",
        },
        ["electron", "argon", "krypton"],
    ),
]

#: Source domains each member of a planted family carries, one per member so
#: the family has to be reconciled rather than inherited.
MEMBER_DOMAINS = [
    "transport",
    "edge_plasma_physics",
    "radiation_measurement_diagnostics",
]


def _plant_family(
    graph: GraphClient,
    segments: dict[str, str],
    subjects: list[str],
    token: str,
    *,
    domains: list[str] | None = None,
) -> list[str]:
    """Plant one family whose members' sources carry disjoint domains."""
    member_ids: list[str] = []
    for index, subject in enumerate(subjects):
        sn_id = f"{token}:{subject}"
        member_ids.append(sn_id)
        row = {"id": sn_id, "subject": subject}
        row.update(segments)
        graph.query(
            """
            CREATE (sn:StandardName {
              id: $id, subject: $subject, physical_base: $physical_base,
              geometric_base: $geometric_base, transformation: $transformation,
              process: $process, object: $object, position: $position,
              geometry: $geometry, region: $region, device: '',
              name_stage: 'accepted', physics_domain: 'general'})
            """,
            id=row["id"],
            subject=row["subject"],
            physical_base=row.get("physical_base", ""),
            geometric_base=row.get("geometric_base", ""),
            transformation=row.get("transformation", ""),
            process=row.get("process", ""),
            object=row.get("object", ""),
            position=row.get("position", ""),
            geometry=row.get("geometry", ""),
            region=row.get("region", ""),
        )
        domain = (domains or MEMBER_DOMAINS)[index % len(domains or MEMBER_DOMAINS)]
        graph.query(
            """
            MATCH (sn:StandardName {id: $id})
            CREATE (src:StandardNameSource {id: $src})
            CREATE (dd:IMASNode {id: $dd, physics_domain: $domain})
            CREATE (src)-[:PRODUCED_NAME]->(sn)
            CREATE (src)-[:FROM_DD_PATH]->(dd)
            """,
            id=sn_id,
            src=f"{sn_id}:src",
            dd=f"dd:{sn_id}/quantity",
            domain=domain,
        )
    return member_ids


@pytest.mark.graph
def test_every_member_reads_the_one_stored_family_decision(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight seeded families, one judgment each, one domain for every member."""
    calls: list[int] = []
    monkeypatch.setattr(
        dd, "decide_batch", _fake_batch(_answer("transport", 0.9), calls)
    )
    seeds = [
        _plant_family(graph, segments, subjects, f"sn:e2e{index}")[0]
        for index, (segments, subjects) in enumerate(SPLIT_FAMILIES)
    ]
    assert len(seeds) == 8
    for seed in seeds:
        dd.decide_and_write_family(graph, seed, IDS_MAP, model="test-model")
    assert len(calls) == 8, (
        f"one judgment per family, not per member: paid {len(calls)}"
    )
    members = [m for seed in seeds for m in _family_ids(graph, seed)]
    assert len(members) == 24
    decisions = {dd.read_family_decision(graph, member) for member in members}
    assert len(decisions) == 8, "each family stores exactly one record"
    assert {d.choice for d in decisions} == {"transport"}
    domains = _physics_domains(graph, members)
    assert set(domains.values()) == {"transport"}, (
        "no member is filed under a different domain from its siblings"
    )


def _family_ids(graph: GraphClient, seed: str) -> list[str]:
    return [
        row["id"]
        for row in graph.query(
            """
            MATCH (seed:StandardName {id: $id})
            MATCH (m:StandardName)
            WHERE coalesce(m.physical_base, '') = coalesce(seed.physical_base, '')
              AND coalesce(m.geometric_base, '') = coalesce(seed.geometric_base, '')
              AND coalesce(m.transformation, '') = coalesce(seed.transformation, '')
              AND coalesce(m.position, '') = coalesce(seed.position, '')
              AND coalesce(m.geometry, '') = coalesce(seed.geometry, '')
              AND coalesce(m.region, '') = coalesce(seed.region, '')
            RETURN m.id AS id
            """,
            id=seed,
        )
    ]


def _physics_domains(graph: GraphClient, ids: list[str]) -> dict[str, str]:
    rows = graph.query(
        """
        UNWIND $ids AS id
        MATCH (sn:StandardName {id: id})
        RETURN sn.id AS id, sn.physics_domain AS physics_domain
        """,
        ids=ids,
    )
    return {row["id"]: row["physics_domain"] for row in rows}


@pytest.mark.graph
def test_a_planted_disjoint_family_receives_one_domain(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two members whose sources carry disjoint domains receive one domain.

    The comparison the per-name decision makes on the same two names is two
    domains, which is the state this decision retires.
    """
    members = _plant_family(
        graph,
        {"physical_base": "counts_per_volume"},
        ["electron", "krypton"],
        "sn:e2e-disjoint",
        domains=["transport", "edge_plasma_physics"],
    )
    monkeypatch.setattr(dd, "decide_batch", _fake_batch(_answer("transport", 0.9)))
    dd.decide_and_write_family(graph, members[0], IDS_MAP, model="test-model")
    family_domains = _physics_domains(graph, members)
    assert set(family_domains.values()) == {"transport"}, (
        f"the family receives one domain, got {family_domains}"
    )

    per_name = _plant_family(
        graph,
        {"physical_base": "counts_per_volume", "position": "somewhere_else"},
        ["electron", "krypton"],
        "sn:e2e-pername",
        domains=["transport", "edge_plasma_physics"],
    )
    for member in per_name:
        decision = asyncio.run(
            dd.decide_domain(
                member,
                "",
                {f"dd:{member}/quantity": _source_domain(graph, member)},
                IDS_MAP,
                model="test-model",
            )
        )
        dd.write_name_domain(graph, member, decision)
    per_name_domains = _physics_domains(graph, per_name)
    assert len(set(per_name_domains.values())) == 2, (
        "the per-name path files the same two names under two domains: "
        f"{per_name_domains}"
    )


def _source_domain(graph: GraphClient, sn_id: str) -> str | None:
    rows = graph.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName {id: $id})
        MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
        RETURN dd.physics_domain AS domain
        """,
        id=sn_id,
    )
    return rows[0]["domain"] if rows else None


OWN_DOMAIN = {
    "electron": MEMBER_DOMAINS[0],
    "argon": MEMBER_DOMAINS[1],
    "krypton": MEMBER_DOMAINS[2],
}


def _split_batch(seen: list[str]) -> Any:
    """The family question falls below the floor; each per-name question decides.

    Each member is answered with the domain its own source sits in, which is
    what makes the split a per-name decision rather than a second family one.
    """
    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        family_question = "family of" in str(questions)
        seen.append("family" if family_question else "name")
        if family_question:
            return [(_answer("transport", 0.4), 0.0)], 0.0
        options = list(questions[dd.QUESTION]["criteria"])
        subject = str(states[0]["name"]).rsplit(":", 1)[-1]
        wanted = OWN_DOMAIN[subject]
        return [(_answer(wanted if wanted in options else options[0], 0.9), 0.0)], 0.0

    return decide


@pytest.mark.graph
def test_a_family_below_the_floor_is_held_split_across_its_members(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family the floor refuses is split: each member is asked on its own."""
    members = _plant_family(
        graph,
        {"physical_base": "ion_temperature"},
        ["electron", "argon", "krypton"],
        "sn:e2e-floor",
        domains=MEMBER_DOMAINS,
    )
    seen: list[str] = []
    recorded: list[tuple[str, str | None, tuple[str, ...]]] = []

    class _Write:
        def __init__(self, sn_id: str, domain: str | None) -> None:
            self.sn_id = sn_id
            self.physics_domain = domain

    def _record(gc, sn_id, decision, **_kwargs):  # noqa: ANN001
        recorded.append((sn_id, decision.choice, tuple(decision.candidates)))
        return _Write(sn_id, decision.choice)

    monkeypatch.setattr(dd, "decide_batch", _split_batch(seen))
    monkeypatch.setattr(dd, "write_name_domain", _record)
    outcome = dd.decide_and_write_family(graph, members[0], IDS_MAP, model="test-model")
    assert outcome.decision.decided is False, (
        "the family itself is still below the floor; the split is held, not decided"
    )
    record = dd.read_family_decision(graph, members[0])
    assert record is not None and record.choice is None
    assert "floor" in (record.undecided_reason or "")
    assert seen.count("name") == len(members), (
        f"each member is put to the per-name form once: {seen}"
    )
    assert set(outcome.unwritten) == set(), (
        f"no member is left unwritten: {outcome.unwritten}"
    )
    assert len(recorded) == len(members), recorded
    offered = {
        sn_id: candidates for sn_id, _choice, candidates in recorded
    }
    for member in members:
        own = _source_domain(graph, member)
        assert own in offered[member], (
            "each member is offered the domain its own sources carry, not the "
            f"family's union: {member} offered {offered[member]}"
        )
    for sn_id, choice, _candidates in recorded:
        assert choice == OWN_DOMAIN[sn_id.rsplit(":", 1)[-1]], (
            "each member is decided on its own sources rather than the "
            f"family's single answer: {sn_id} settled on {choice}"
        )


@pytest.mark.graph
def test_removing_the_split_leaves_the_members_unwritten(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failing control: without the split a held family writes nothing."""
    members = _plant_family(
        graph,
        {"physical_base": "ion_heat_flux"},
        ["electron", "argon", "krypton"],
        "sn:e2e-floor-control",
    )
    monkeypatch.setattr(dd, "decide_batch", _split_batch([]))
    monkeypatch.setattr(dd, "split_undecided_family", lambda *a, **k: ())
    outcome = dd.decide_and_write_family(graph, members[0], IDS_MAP, model="test-model")
    assert outcome.decision.decided is False
    assert outcome.writes == ()
    assert set(outcome.unwritten) == set(members)
    assert set(_physics_domains(graph, members).values()) == {"general"}, (
        "with the split removed every member keeps its stored domain"
    )


@pytest.mark.graph
def test_the_fan_out_goes_through_the_one_writer(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every member's domain is written by ``write_name_domain``, once each."""
    members = _plant_family(
        graph,
        {"physical_base": "pressure"},
        ["electron", "argon", "krypton"],
        "sn:e2e-writer",
    )
    monkeypatch.setattr(dd, "decide_batch", _fake_batch(_answer("transport", 0.9)))
    seen: list[str] = []
    original = dd.write_name_domain

    def recording(gc, sn_id, decision, **kwargs):  # noqa: ANN001
        seen.append(sn_id)
        return original(gc, sn_id, decision, **kwargs)

    monkeypatch.setattr(dd, "write_name_domain", recording)
    dd.decide_and_write_family(graph, members[0], IDS_MAP, model="test-model")
    assert sorted(seen) == sorted(members)


@pytest.mark.graph
def test_the_census_groups_by_the_same_signature_as_the_seeded_read(
    graph: GraphClient,
) -> None:
    """Three planted families of three members enumerate as three families."""
    seeds = [
        _plant_family(
            graph,
            {"physical_base": f"census_base_{index}"},
            ["electron", "argon", "krypton"],
            f"sn:census{index}",
        )[0]
        for index in range(3)
    ]
    families = fam.fetch_domain_families(graph, min_size=3)
    by_key = {family["key"]: family for family in families}
    seeded = [fam.fetch_domain_family(seed, graph) for seed in seeds]
    assert {family["key"] for family in seeded} <= set(by_key), (
        "the census misses families it must group"
    )
    for seed in seeds:
        read = fam.fetch_domain_family(seed, graph)
        assert len(read["members"]) == 3
        census = by_key[read["key"]]
        assert census["segments"] == read["segments"], (
            "the two reads must agree on the family's signature"
        )
        assert {m["id"] for m in census["members"]} >= {
            m["id"] for m in read["members"]
        }, "the census must carry every member the seeded read finds"
        assert all(len(m["sources"]) == 1 for m in census["members"]), (
            "a census member carries its own sources"
        )
