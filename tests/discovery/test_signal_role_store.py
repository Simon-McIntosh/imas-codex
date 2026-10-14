"""A signal's role is stored after its text is written, and left role-less on failure.

Enrichment judges a signal's role from the text a searcher reads, once that
text is stored, and writes ``signal_role``. The sweep backfills a signal that
carries a description but no role, selecting its rows with the same predicate
its census counts — so a signal the sweep wrote is a signal the census had
named as missing.

Three behaviours the store must hold to, and this is the only place asserting
them: a judgement the model could not answer, or that fell outside the six
roles, stores no role at all and leaves the signal in the sweep's population for
a later run; the reset to ``discovered`` clears the stored role so a reprocessed
signal is re-judged on its new description; and marking a signal underspecified
clears any stored role, because a role judged from a description the worker has
just rejected must not survive the status change.

An underspecified signal is not a role candidate at all: too thin a description
is what made the worker mark it underspecified, and a role judged from it
disagreed with a blind hand-read too often to store. So the predicate that
selects and counts role candidates admits only the enriched and checked statuses.

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

    Recognises the statements the store, sweep and underspecified-marker issue —
    the candidate read, the population count, the role writer and the status
    change — and mutates the model the way the Cypher would, so a signal's
    stored role follows whatever the writer set and the next read sees it. The
    candidate read is answered from the predicate the statement itself carries,
    so the double never selects by a rule of its own.
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

        if "SET s.status = $status" in cypher and "s.signal_role = null" in cypher:
            for sig in params["signals"]:
                node = self.nodes[sig["id"]]
                node["status"] = params["status"]
                node["signal_role"] = None
                if sig.get("description") is not None:
                    node["description"] = sig["description"]
            return []

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
    underspecified signal is in neither population: a role judged from too thin
    a description is not trustworthy, so its status is not admitted by the
    shared bound.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "status": "enriched",
                "signal_role": None,
            },
            # Underspecified: not a candidate, however well described.
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
    assert before["missing"] == 1, before
    assert before["missing_ids"] == ["jt-60sa:ip"], before

    result = parallel.backfill_signal_roles(gc=graph)
    assert result["candidates"] == 1, result
    assert result["written"] == 1, result

    assert graph.nodes["jt-60sa:low"]["signal_role"] is None
    assert parallel.census_signal_roles(gc=graph)["missing"] == 0


def test_reset_to_discovered_clears_the_role():
    """The reset to ``discovered`` clears ``signal_role`` so it is re-judged.

    A reprocessed signal gets a new description, and the role must be judged
    against that text rather than kept from the old one. Clearing the slot in
    the reset is what makes an enriched signal selectable by the sweep again.
    """
    cleared = base_reset.SIGNAL_RESET_SPECS["discovered"].clear_fields
    assert "signal_role" in cleared


def test_underspecified_signal_is_not_a_role_candidate(monkeypatch):
    """An underspecified signal is not a role candidate, however well described.

    A role judged from a description the worker itself found too thin to enrich
    is not trustworthy, so the predicate does not admit the underspecified
    status and neither the sweep nor the census names such a signal. A predicate
    that re-admitted the status would sweep and store roles whose agreement with
    a blind hand-read is below the threshold worth storing — the change this
    test exists to catch.
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

    assert parallel.census_signal_roles(gc=graph)["missing"] == 0
    assert parallel.census_signal_roles(gc=graph)["missing_ids"] == []

    result = parallel.backfill_signal_roles(gc=graph)
    assert result["candidates"] == 0, result
    assert result["written"] == 0, result
    assert graph.nodes["jt-60sa:low"]["signal_role"] is None


def test_mark_signals_underspecified_clears_a_stored_role(monkeypatch):
    """Marking a signal underspecified clears a role it carried before.

    A signal can be enriched, judged and given a role, then re-processed and
    found underspecified. Its old role was judged from a description the worker
    has now rejected, so it must not survive the status change — the same
    statement that sets the underspecified status clears the role. A statement
    without the clear leaves the stale role in place, which this test catches.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:old": {
                "name": "Edge density",
                "description": "Line-integrated density",
                "status": "enriched",
                "signal_role": "measured",
            }
        }
    )
    monkeypatch.setattr(parallel, "GraphClient", lambda *a, **k: graph)
    monkeypatch.setattr(parallel, "write_signal_descriptions", lambda *a, **k: None)

    written = parallel.mark_signals_underspecified(
        [{"id": "jt-60sa:old", "description": "Edge density, context thin"}]
    )

    assert written == 1
    assert graph.nodes["jt-60sa:old"]["status"] == "underspecified"
    assert graph.nodes["jt-60sa:old"]["signal_role"] is None


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
async def test_enrichment_does_not_judge_an_underspecified_signal():
    """The enrich worker judges no role for a signal it marks underspecified.

    A signal the worker found too thin to enrich has too thin a description for
    the role judge, which disagreed with a blind hand-read on 4 of 30 such
    signals. So the worker marks it underspecified and judges nothing: it does
    not call the role judge at all. A worker that re-added the judgement call
    would store a role the gate rejects, which this test catches.
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
    marked: list[list[dict]] = []

    def claim(*args, **kwargs):
        batch = next(claims, None)
        if batch is None:
            state.stop_requested = True
            return []
        return batch

    def mark(signals, *args, **kwargs):
        marked.append(list(signals))
        return len(signals)

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
        patch.object(parallel, "mark_signals_underspecified", side_effect=mark),
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

    assert marked and marked[0][0]["id"] == signals[0]["id"], (
        "the worker marks the signal underspecified"
    )
    assert not judged, "an underspecified signal is not judged for a role"
