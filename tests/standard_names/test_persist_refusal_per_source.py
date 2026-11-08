"""A refused compose winner costs its own source, not the whole pool batch.

The rich name writer refuses a candidate by name (the grammar gate and the
unit-conflict filter both drop it before it mints anything). An exact-claim
winner refused that way stays unbound and its claim still belongs to this
write, so the batch reports it and releases it instead of aborting every other
winner alongside it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.graph_store import (
    graph_foundation as foundation,
    graph_source_bindings as bindings,
)
from tests.standard_names.graph_transaction_doubles import TransactionDouble

TOKEN = "source-owner"
SEQ = 9

ACCEPTED = [
    ("plasma_energy", "summary/global_quantities/energy"),
    ("plasma_current", "summary/global_quantities/current"),
]
REFUSED_PATH = "diagnostics/fiber_bundle/surface_area"
REFUSED = "surface_area_of_fiber_bundle"


def _candidate(name: str, path: str) -> dict:
    return {
        "id": name,
        "source_id": path,
        "source_types": ["dd"],
        "description": "Composed from the DD path.",
        "unit": "m^2",
        "source_claim_token": TOKEN,
        "source_claim_seq": SEQ,
    }


class RefusalGraph(TransactionDouble):
    """Record what the batch asked for and apply the failure writer's outcome."""

    def __init__(self, *, refused: tuple[str, ...] = (REFUSED,)) -> None:
        super().__init__()
        self.refused = set(refused)
        self.persisted: list[dict] = []
        self.released: list[dict] = []
        self.unwound: list[dict] = []
        self.commit = self.transaction.commit
        self.claims = {
            f"dd:{path}": TOKEN for _name, path in ACCEPTED
        } | {f"dd:{REFUSED_PATH}": TOKEN}
        self.statuses = {source: "extracted" for source in self.claims}
        self.attempts = {source: 0 for source in self.claims}
        # The refused source has already spent its retries: the writer parks it.
        self.attempts[f"dd:{REFUSED_PATH}"] = 2
        self.errors: dict[str, str] = {}
        self.query.side_effect = self.answer

    def answer(self, cypher: str, **params):
        if "AS outcome" in cypher:
            rows = []
            for item in params["batch"]:
                rows.append(
                    {
                        "id": item["sns_id"],
                        "outcome": "winner",
                        "binding_kind": "owned_reservation",
                        "candidate_id": item["sn_id"],
                        "target_stage": params["pending_stage"],
                        "attempt_count": self.attempts.get(item["sns_id"], 0),
                        "successors": [],
                    }
                )
            return rows
        if "DELETE reservation" in cypher:
            self.unwound.extend(params["batch"])
            return [{"id": item["sns_id"]} for item in params["batch"]]
        if "SET sn.name_stage = 'drafted'" in cypher:
            rows = []
            for item in params["batch"]:
                self.persisted.append(item)
                self.claims[item["sns_id"]] = None
                self.statuses[item["sns_id"]] = "composed"
                rows.append({"id": item["sns_id"]})
            return rows
        if "sns.last_error = $error" in cypher:
            affected = 0
            for source_id in params["source_ids"]:
                if self.claims.get(source_id) != params["token"]:
                    continue
                attempt = self.attempts.get(source_id, 0) + 1
                self.attempts[source_id] = attempt
                self.statuses[source_id] = (
                    "failed" if attempt >= params["max_attempts"] else "extracted"
                )
                self.claims[source_id] = None
                self.errors[source_id] = params["error"]
                self.released.append(
                    {"source_id": source_id, "error": params["error"]}
                )
                affected += 1
            return [{"affected": affected}]
        return []


def _services(graph: RefusalGraph, written: list[str]) -> SimpleNamespace:
    services = SimpleNamespace(
        NameStage=foundation.NameStage,
        ManifestDrainConflict=RuntimeError,
        GraphClient=lambda: graph,
        _TransactionQuery=foundation._TransactionQuery,
        _guard_existing_target_pairings=MagicMock(return_value={}),
        normalize_name_id=lambda name: name,
        write_standard_names=MagicMock(return_value=written),
        bump_sn_run_counter=MagicMock(),
        _backfill_cluster_from_sources=MagicMock(),
        supersede_prior_source_names=MagicMock(),
    )
    services._complete_source_fences = lambda items: bindings._complete_source_fences(
        items, services=services
    )
    services._require_failure_reason = (
        lambda source_id, reason: bindings._require_failure_reason(
            source_id, reason, services=services
        )
    )
    services._lock_claimed_name_bindings = lambda *args, **kwargs: (
        bindings._lock_claimed_name_bindings(*args, services=services, **kwargs)
    )
    services.mark_sources_failed = lambda *args, **kwargs: bindings.mark_sources_failed(
        *args, services=services, **kwargs
    )
    return services


def _persist(graph: RefusalGraph, written: list[str]):
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_winners,
    )

    services = _services(graph, written)
    candidates = [_candidate(name, path) for name, path in ACCEPTED] + [
        _candidate(REFUSED, REFUSED_PATH)
    ]
    return persist_generated_name_winners(
        candidates, compose_model="test", services=services
    ), services


def test_one_refused_winner_leaves_the_others_persisted():
    graph = RefusalGraph()
    result, services = _persist(
        graph, [name for name, _path in ACCEPTED]
    )

    assert result == [f"dd:{path}" for _name, path in ACCEPTED]
    assert [item["sns_id"] for item in graph.persisted] == result
    assert services.write_standard_names.call_count == 1
    assert len(services.write_standard_names.call_args.args[0]) == 3
    graph.commit.assert_called_once()
    graph.transaction.rollback.assert_not_called()


def test_refused_source_is_released_with_the_reason_naming_candidate_and_refusal():
    graph = RefusalGraph()
    _persist(graph, [name for name, _path in ACCEPTED])

    refused_source = f"dd:{REFUSED_PATH}"
    assert [item["source_id"] for item in graph.released] == [refused_source]
    error = graph.errors[refused_source]
    assert REFUSED in error
    assert "refused" in error
    assert graph.statuses[refused_source] == "failed"
    assert graph.statuses[f"dd:{ACCEPTED[0][1]}"] == "composed"
    assert graph.claims[refused_source] is None


def test_refused_source_loses_its_provisional_reservation_only():
    graph = RefusalGraph()
    _persist(graph, [name for name, _path in ACCEPTED])

    assert [item["sns_id"] for item in graph.unwound] == [f"dd:{REFUSED_PATH}"]
    assert all(item["claim_token"] == TOKEN for item in graph.unwound)


def test_every_winner_refused_releases_each_source_and_mints_nothing():
    graph = RefusalGraph(refused={name for name, _path in ACCEPTED} | {REFUSED})
    result, _services_ = _persist(graph, [])

    assert result == []
    assert graph.persisted == []
    assert len(graph.released) == 3
    assert all("refused" in item["error"] for item in graph.released)


def test_a_winner_the_writer_accepted_but_did_not_finalize_still_raises():
    """A name minted for another identity has no source to release."""

    graph = RefusalGraph()
    services = _services(graph, [name for name, _path in ACCEPTED] + [REFUSED])
    # Retargeting a winner to an existing identity leaves the write's accepted
    # identity unbound to any finalize entry, which is not a per-source refusal.
    services._lock_claimed_name_bindings = lambda *args, **kwargs: [
        {
            "id": f"dd:{path}",
            "outcome": "winner",
            "binding_kind": "owned_reservation",
            "candidate_id": name,
            "resolved_id": "plasma_energy_existing"
            if name == ACCEPTED[0][0]
            else None,
            "target_stage": foundation.NameStage.pending.value,
            "attempt_count": 0,
            "successors": [],
        }
        for name, path in ACCEPTED + [(REFUSED, REFUSED_PATH)]
    ]

    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_winners,
    )

    with pytest.raises(RuntimeError, match="exact-claim winner"):
        persist_generated_name_winners(
            [_candidate(name, path) for name, path in ACCEPTED]
            + [_candidate(REFUSED, REFUSED_PATH)],
            compose_model="test",
            services=services,
        )
    assert graph.released == []
    graph.transaction.rollback.assert_not_called()