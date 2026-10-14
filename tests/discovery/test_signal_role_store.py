"""A signal's role is stored after its text is written, and left role-less on failure.

Enrichment judges a signal's role from the text a searcher reads, once that
text is stored, and writes ``signal_role``. The sweep backfills a signal that
carries a description but no role, selecting its rows with the same predicate
its census counts — so a signal the sweep wrote is a signal the census had
named as missing.

Two behaviours the store must hold to, and this is the only place asserting
them: a judgement the model could not answer, or that fell outside the six
roles, stores no role at all and leaves the signal in the sweep's population for
a later run; and the reset to ``discovered`` clears the stored role so a
reprocessed signal is re-judged on its new description.

The decision call is mocked throughout, so the tests exercise the wiring from
the judgment layer through the store to the reset, and never touch the network.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.discovery.base import judgment, reset as base_reset
from imas_codex.discovery.signals import parallel

_STATUS_IN_RE = re.compile(r"s\.status IN \[([^\]]*)\]")


def _where_clause(cypher: str) -> str:
    """The ``WHERE`` fragment a candidate read carries, between ``WHERE`` and ``RETURN``
    (or the end of the statement when the read has no trailing clause)."""
    match = re.search(r"\bWHERE\b(.*?)(?:\bRETURN\b|$)", cypher, re.S)
    assert match, f"candidate read has no WHERE: {cypher[:140]}"
    return match.group(1)


def _selects(where: str, node: dict) -> bool:
    """Evaluate a candidate-read predicate over one signal node.

    The predicate is read from the Cypher the caller issued rather than fixed
    here, so the double selects exactly what that caller selects. A sweep and a
    census that drifted onto different predicates would then disagree, which the
    test can see — a selection rule the double owns would mask the drift.

    Recognises the conjunctive forms the role predicate is written in and fails
    loudly on a predicate it does not understand, so an unrecognised rewrite
    cannot quietly select every node.
    """
    text = " ".join(where.split())
    recognised = False
    status = _STATUS_IN_RE.search(text)
    if status:
        recognised = True
        allowed = re.findall(r"'([^']*)'", status.group(1))
        if node.get("status") not in allowed:
            return False
    if "s.description IS NOT NULL" in text:
        recognised = True
        if node.get("description") is None:
            return False
    if "trim(s.description) <> ''" in text:
        recognised = True
        if not (node.get("description") or "").strip():
            return False
    if "s.signal_role IS NULL" in text:
        recognised = True
        if node.get("signal_role") is not None:
            return False
    assert recognised, f"predicate not understood: {where!r}"
    return True


class _RoleGraph:
    """A stateful model of FacilitySignal status, description and role fields.

    Recognises the three statements the store and sweep issue — the candidate
    read, the population count and the role writer — and mutates the model the
    way the Cypher would, so a signal's stored role follows whatever the writer
    set and the next read sees it. The candidate read is answered from the
    predicate the statement itself carries, so the double never selects by a
    rule of its own.
    """

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "SET s.signal_role = item.signal_role" in cypher:
            for item in params["items"]:
                self.nodes[item["id"]]["signal_role"] = item["signal_role"]
            return [{"updated": len(params["items"]) if params["items"] else 0}]

        if "RETURN count(s) AS n" in cypher:
            facility = params.get("facility")
            return [
                {
                    "n": sum(
                        1
                        for node in self.nodes.values()
                        if facility is None or node.get("facility_id") == facility
                    )
                }
            ]

        if "RETURN s.id AS id, s.name AS name" in cypher:
            where = _where_clause(cypher)
            facility = params.get("facility")
            return [
                {
                    "id": nid,
                    "name": node.get("name"),
                    "description": node.get("description"),
                    "unit": node.get("unit"),
                    "facility_id": node.get("facility_id"),
                }
                for nid, node in self.nodes.items()
                if _selects(where, node)
                and (facility is None or node.get("facility_id") == facility)
            ]

        raise AssertionError(f"unexpected query: {cypher[:140]}")


def _answer(role: str):
    async def fake_decision(model, state, questions, *, service):
        assert questions["signal_role"]["type"] == "choice"
        return {"signal_role": {"choice": role}}, 0.003

    return fake_decision


def _enriched_row(signal_id: str) -> dict:
    return {
        "id": signal_id,
        "name": "Plasma current",
        "description": "Measured plasma current",
        "unit": "A",
    }


def test_enrichment_stores_a_role_for_a_described_signal(monkeypatch):
    """A described signal judged as measured carries that role afterwards.

    The judgement runs after the description is written, so the role describes
    the text a searcher reads. A store that never wires the judgement leaves
    the signal role-less, which is the negative control's failing arm.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "unit": "A",
                "status": "enriched",
                "signal_role": None,
            }
        }
    )
    monkeypatch.setattr(judgment, "acall_decisions", _answer("measured"))

    written, cost = parallel._run_coro(
        parallel.judge_and_store_signal_roles([_enriched_row("jt-60sa:ip")], gc=graph)
    )

    assert written == 1, (written, cost)
    assert graph.nodes["jt-60sa:ip"]["signal_role"] == "measured"
    # The stub prices each call, so spend is the response usage, not a guess.
    assert cost == pytest.approx(0.003), cost


def test_failed_judgement_stores_no_role(monkeypatch):
    """A decision call that faults writes nothing and the signal is left role-less.

    A failed call is reported to the judgment layer as ``None`` rather than
    raised, so no answer reaches the store. The signal keeps no role and the
    sweep's population still names it, ready for the next run.
    """

    async def failing_decision(model, state, questions, *, service):
        raise RuntimeError("decision endpoint refused")

    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "status": "enriched",
                "signal_role": None,
            }
        }
    )
    monkeypatch.setattr(judgment, "acall_decisions", failing_decision)

    written, _cost = parallel._run_coro(
        parallel.judge_and_store_signal_roles([_enriched_row("jt-60sa:ip")], gc=graph)
    )

    assert written == 0
    assert graph.nodes["jt-60sa:ip"]["signal_role"] is None
    assert parallel.census_signal_roles(gc=graph)["missing"] == 1


def test_out_of_set_judgement_stores_no_role(monkeypatch):
    """A choice outside the six roles stores nothing rather than a bad value.

    The judge raises on a role it does not recognise; the caller catches that
    and writes nothing, so an unrecognised answer can never enter the graph as
    a ``signal_role``.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "status": "enriched",
                "signal_role": None,
            }
        }
    )
    monkeypatch.setattr(judgment, "acall_decisions", _answer("not_a_role"))

    written, _cost = parallel._run_coro(
        parallel.judge_and_store_signal_roles([_enriched_row("jt-60sa:ip")], gc=graph)
    )

    assert written == 0
    assert graph.nodes["jt-60sa:ip"]["signal_role"] is None


def test_sweep_and_census_share_one_predicate(monkeypatch):
    """The sweep judges exactly the rows the census counts, through one predicate.

    A described signal with no role is in both populations: the census names it
    and the sweep judges and stores it. After the sweep the census reads zero,
    which can only happen if both use the same predicate — a sweep that
    selected by its own copy could leave a counted row unwritten. An
    underspecified signal is a candidate too, so a predicate that dropped the
    underspecified status would leave it counted and unwritten.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "status": "enriched",
                "signal_role": None,
            },
            # Underspecified, but described: still a role candidate.
            "jt-60sa:low": {
                "name": "Edge density",
                "description": "Line-integrated density, context thin",
                "status": "underspecified",
                "signal_role": None,
            },
            # No description yet: not a candidate.
            "jt-60sa:undescribed": {
                "name": "TODO",
                "description": None,
                "status": "discovered",
                "signal_role": None,
            },
            # Already has a role: not a candidate.
            "jt-60sa:done": {
                "name": "Loop voltage",
                "description": "Loop voltage",
                "status": "enriched",
                "signal_role": "measured",
            },
        }
    )
    monkeypatch.setattr(judgment, "acall_decisions", _answer("measured"))

    before = parallel.census_signal_roles(gc=graph)
    assert before["missing"] == 2, before
    assert before["missing_ids"] == ["jt-60sa:ip", "jt-60sa:low"], before

    result = parallel.backfill_signal_roles(gc=graph)
    assert result["candidates"] == 2, result
    assert result["written"] == 2, result

    assert parallel.census_signal_roles(gc=graph)["missing"] == 0


def test_reset_to_discovered_clears_the_role():
    """The reset to ``discovered`` clears ``signal_role`` so it is re-judged.

    A reprocessed signal gets a new description, and the role must be judged
    against that text rather than kept from the old one. Clearing the slot in
    the reset is what makes an enriched signal selectable by the sweep again.
    """
    cleared = base_reset.SIGNAL_RESET_SPECS["discovered"].clear_fields
    assert "signal_role" in cleared


def test_underspecified_signal_is_a_role_candidate(monkeypatch):
    """An underspecified signal with a description is a role candidate and is stored.

    It carries the description a searcher reads, so a ``role=`` search must find
    it; the predicate admits the underspecified status beside enriched and
    checked. A predicate without it leaves the signal role-less and unreachable
    by role — the drop this test exists to catch.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:low": {
                "name": "Edge density",
                "description": "Line-integrated density, context thin",
                "unit": "m^-2",
                "status": "underspecified",
                "signal_role": None,
            }
        }
    )
    monkeypatch.setattr(judgment, "acall_decisions", _answer("measured"))

    assert parallel.census_signal_roles(gc=graph)["missing_ids"] == ["jt-60sa:low"]

    result = parallel.backfill_signal_roles(gc=graph)
    assert result["candidates"] == 1, result
    assert result["written"] == 1, result
    assert graph.nodes["jt-60sa:low"]["signal_role"] == "measured"
    assert parallel.census_signal_roles(gc=graph)["missing"] == 0


def _claimed_signal(signal_id: str) -> dict:
    return {
        "id": signal_id,
        "accessor": "eddbreadTime('E101173', 'MMSYS', 'curEF1LKAT', t1, t2)",
        "name": "MMSYS/curEF1LKAT",
        "data_source_name": "edas",
        "data_source_path": "MMSYS/curEF1LKAT",
        "discovery_source": "edas",
        "unit": None,
        "description": "",
        "facility_id": "jt-60sa",
    }


@pytest.mark.asyncio
async def test_enrichment_judges_an_underspecified_signal():
    """The enrich worker judges the role of a signal it marks underspecified.

    An underspecified signal carries the text a searcher reads, so enrichment
    must judge its role exactly as it does for an enriched one. A worker that
    judges only the enriched branch leaves every underspecified signal
    role-less until a later sweep, which is the gap this test closes.
    """
    from unittest.mock import patch

    from imas_codex.discovery.signals.models import (
        SignalEnrichmentBatch,
        SignalEnrichmentResult,
    )

    signals = [_claimed_signal("jt-60sa:general/mmsys_curef1lkat")]
    state = parallel.DataDiscoveryState(facility="jt-60sa", scanner_types=["edas"])
    claims = iter([signals])
    judged: list[list[dict]] = []

    def claim(*args, **kwargs):
        batch = next(claims, None)
        if batch is None:
            state.stop_requested = True
            return []
        return batch

    async def llm(*, messages, **kwargs):
        return (
            SignalEnrichmentBatch(
                results=[
                    SignalEnrichmentResult(
                        signal_index=1,
                        physics_domain="general",
                        name="EF1 coil current",
                        description="Current in EF coil 1, context thin",
                        context_quality="low",
                    )
                ]
            ),
            0.0,
            0,
        )

    async def judge(entries, **kwargs):
        judged.append(list(entries))
        return 1, 0.0

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "prepare_signal_sources", return_value=(0, 0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched"),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch.object(parallel, "release_signal_claim"),
        patch.object(parallel, "judge_and_store_signal_roles", side_effect=judge),
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    assert judged, "an underspecified signal is judged by enrichment"
    assert judged[0][0]["id"] == signals[0]["id"]
