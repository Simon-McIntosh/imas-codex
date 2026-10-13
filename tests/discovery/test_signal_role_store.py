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

import pytest

from imas_codex.discovery.base import judgment, reset as base_reset
from imas_codex.discovery.signals import parallel


def _is_role_candidate(node: dict) -> bool:
    """The signal population the sweep judges, in the predicate's own terms."""
    return (
        node.get("status") in ("enriched", "checked")
        and bool((node.get("description") or "").strip())
        and node.get("signal_role") is None
    )


class _RoleGraph:
    """A stateful model of FacilitySignal status, description and role fields.

    Recognises the three statements the store and sweep issue — the candidate
    read, the population count and the role writer — and mutates the model the
    way the Cypher would, so a signal's stored role follows whatever the writer
    set and the next read sees it.
    """

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "RETURN s.id AS id, s.name AS name" in cypher:
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
                if _is_role_candidate(node)
                and (facility is None or node.get("facility_id") == facility)
            ]

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

        if "SET s.signal_role = item.signal_role" in cypher:
            for item in params["items"]:
                self.nodes[item["id"]]["signal_role"] = item["signal_role"]
            return [{"updated": len(params["items"])}]

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
    selected by its own copy could leave a counted row unwritten.
    """
    graph = _RoleGraph(
        {
            "jt-60sa:ip": {
                "name": "Plasma current",
                "description": "Measured plasma current",
                "status": "enriched",
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

    assert parallel.census_signal_roles(gc=graph)["missing"] == 0


def test_reset_to_discovered_clears_the_role():
    """The reset to ``discovered`` clears ``signal_role`` so it is re-judged.

    A reprocessed signal gets a new description, and the role must be judged
    against that text rather than kept from the old one. Clearing the slot in
    the reset is what makes an enriched signal selectable by the sweep again.
    """
    cleared = base_reset.SIGNAL_RESET_SPECS["discovered"].clear_fields
    assert "signal_role" in cleared
