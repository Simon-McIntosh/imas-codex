"""Claim-ending writers retain fenced receipts and actionable refusal reasons."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from imas_codex.standard_names.graph_store import (
    graph_catalog as catalog,
    graph_claim_release as release,
    graph_foundation as foundation,
    graph_source_bindings as bindings,
)
from imas_codex.standard_names.worker_tasks import worker_common
from tests.standard_names.test_claim_outcome_receipts import (
    _claim,
    _FakeGraphClient,
    _services,
)
from tests.standard_names.test_persist_refusal_per_source import (
    REFUSED,
    REFUSED_PATH,
    SEQ,
    TOKEN,
    RefusalGraph,
    _candidate,
    _services as persistence_services,
)


class ReceiptGraph(_FakeGraphClient):
    """Apply the emitted predicate, including Cypher's null comparisons."""

    def dispatch(self, cypher, params):
        compact = " ".join(cypher.split())
        if "StandardNameSourceClaimAttempt" in cypher:
            self.queries.append((cypher, params))
            entries = []
            for entry in params["entries"]:
                source = self.claims.get(entry["source_id"])
                if source is None:
                    continue
                token = entry.get("claim_token")
                fenced = (
                    token is not None
                    and source["claim_token"] == token
                    and entry.get("claim_seq") is not None
                    and source["claim_seq"] == entry["claim_seq"]
                )
                # Model the predicate actually emitted, including its unsafe
                # null-token shortcut when that shortcut is present.
                shortcut = "WHERE entry.claim_token IS NULL OR" in compact
                unfenced = (
                    "entry.outcome = 'capped'" in compact
                    and "sns.claim_token IS NULL" in compact
                    and entry["outcome"] == "capped"
                    and source["claim_token"] is None
                    and token is None
                )
                if fenced or (token is None and shortcut) or unfenced:
                    entries.append(entry)
            return [{"written": self._append(entries)}]
        if "$ids" in cypher and "StandardNameSource" in cypher:
            self.queries.append((cypher, params))
            sources = [
                (sid, self.claims[sid])
                for sid in params["ids"]
                if sid in self.claims
                and self.claims[sid]["claim_token"] == params["token"]
            ]
            if "n.claimed_at = null" in cypher:
                for _, source in sources:
                    source["claim_token"] = None
                return [{"released": len(sources)}]
            return [{"id": sid, **source} for sid, source in sources]
        return super().dispatch(cypher, params)


@pytest.mark.parametrize(
    "writer",
    [release.release_generate_name_claims, release.release_generate_name_failed_claims],
)
def test_release_appends_receipt_before_consuming_claim(writer):
    graph = ReceiptGraph(
        {
            "dd:held": {
                **_claim("owner", 7, attempt=4),
                "run_id": "run",
                "batch_key": "batch",
            },
            "dd:other": _claim("other-owner", 8),
        }
    )
    services = _services(graph)
    services.release_generate_name_claims = lambda **kw: (
        release.release_generate_name_claims(**kw, services=services)
    )

    assert (
        writer(
            source_ids=["dd:held", "dd:other"], claim_token="owner", services=services
        )
        == 1
    )
    (receipt,) = graph.held_by("dd:held")
    assert receipt["outcome"] in {"released", "requeued"}
    assert receipt["attempt"] == 4
    assert receipt["run_id"] == "run"
    assert receipt["batch_key"] == "batch"
    assert graph.held_by("dd:other") == []
    assert graph.claims["dd:held"]["claim_token"] is None
    assert graph.claims["dd:other"]["claim_token"] == "other-owner"
    statements = [query for query, _ in graph.queries]
    assert next(
        i for i, q in enumerate(statements) if "StandardNameSourceClaimAttempt" in q
    ) < next(i for i, q in enumerate(statements) if "n.claimed_at = null" in q)


@pytest.mark.parametrize("outcome", ["refused", "released", "requeued", "capped"])
def test_missing_token_cannot_append_to_claimed_source(outcome):
    graph = ReceiptGraph({"dd:held": _claim("owner", 7)})
    entry = foundation.claim_attempt_entry("dd:held", outcome=outcome, claim_seq=7)
    assert foundation.append_claim_attempts(graph, [entry]) == 0
    assert graph.held_by("dd:held") == []
    entry["claim_token"] = "owner"
    assert foundation.append_claim_attempts(graph, [entry]) == 1
    assert len(graph.held_by("dd:held")) == 1


@pytest.mark.parametrize(
    "token,seq", [("stale-owner", 7), ("owner", 6), ("owner", None)]
)
def test_wrong_fence_does_not_append(token, seq):
    graph = ReceiptGraph({"dd:held": _claim("owner", 7)})
    assert (
        foundation.append_claim_attempts(
            graph,
            [
                foundation.claim_attempt_entry(
                    "dd:held", outcome="released", claim_token=token, claim_seq=seq
                )
            ],
        )
        == 0
    )
    assert graph.receipts == {}


def test_only_capped_sweep_can_append_to_unclaimed_source():
    from imas_codex.standard_names.orphan_sweep import _capped_claim_entries

    graph = ReceiptGraph({"dd:parked": _claim(None, 7, attempt=5)})
    denied = foundation.claim_attempt_entry(
        "dd:parked", outcome="released", claim_seq=7
    )
    assert foundation.append_claim_attempts(graph, [denied]) == 0
    entries = _capped_claim_entries(
        [{"id": "dd:parked", "claim_seq": 7, "attempt_count": 5}]
    )
    assert foundation.append_claim_attempts(graph, entries) == 1
    assert graph.held_by("dd:parked")[0]["outcome"] == "capped"


@pytest.mark.parametrize(
    "status,outcome",
    [("failed", "refused"), ("skipped", "skipped"), ("extracted", "requeued")],
)
def test_claimed_outcome_stores_run_and_batch_metadata(status, outcome):
    graph = ReceiptGraph({"dd:path": _claim("owner", 7, attempt=4)})
    item = {
        "path": "path",
        "claim_token": "owner",
        "claim_seq": 7,
        "run_id": "run",
        "batch_key": "batch",
    }
    result = worker_common._claimed_source_outcome(
        [item],
        "path",
        source_type="dd",
        status=status,
        last_error="source-specific reason",
    )
    assert bindings.persist_claimed_source_outcomes(
        [result], services=_services(graph)
    ) == ["dd:path"]
    (receipt,) = graph.held_by("dd:path")
    assert receipt["outcome"] == outcome
    assert receipt["run_id"] == "run"
    assert receipt["batch_key"] == "batch"
    assert receipt["reason"] == "source-specific reason"


@pytest.mark.parametrize("cause", ["grammar", "unit"])
def test_persist_refusal_retains_actual_writer_reason(cause, monkeypatch):
    monkeypatch.setattr(
        "imas_codex.standard_names.protection._fetch_catalog_edit_names",
        lambda *args, **kwargs: set(),
    )

    class WriterRefusalGraph(RefusalGraph):
        def answer(self, cypher, **params):
            if "sn.unit AS existing_unit" in cypher:
                return [
                    {
                        "name": "plasma_current",
                        "existing_unit": "A",
                        "incoming_unit": "m^2",
                    }
                ]
            return super().answer(cypher, **params)

    graph = WriterRefusalGraph()
    services = persistence_services(graph, [])
    services.write_standard_names = lambda names, **kwargs: (
        catalog.write_standard_names(names, services=services, **kwargs)
    )
    name = REFUSED if cause == "grammar" else "plasma_current"
    assert (
        catalog.persist_generated_name_winners(
            [_candidate(name, REFUSED_PATH)], compose_model="test", services=services
        )
        == []
    )
    error = graph.errors[f"dd:{REFUSED_PATH}"]
    assert name in error
    if cause == "grammar":
        assert "grammar parse failed" in error
        assert "fiber_bundle" in error
    else:
        assert "unit conflict" in error
        assert "A" in error and "m^2" in error
    assert graph.claims[f"dd:{REFUSED_PATH}"] is None
