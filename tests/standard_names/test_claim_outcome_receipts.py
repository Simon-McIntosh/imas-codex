"""Every compose claim that ends records its outcome on the source.

A claim ends in exactly one of five ways. Each ending writer appends a
``StandardNameSourceClaimAttempt`` entry on the source it ends, so a failure
census can read one entry per attempt instead of a single overwritten
``last_error``. These tests pin the entry contract, the append stanza, and the
fact that each ending writer carries its own outcome.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from imas_codex.standard_names.graph_store import graph_cost_and_claims as gcc


class _FakeGraphClient:
    """Graph client double that applies the entry fence in Python.

    Only the fence and the MERGE-on-id identity are modelled, which is what
    the writers rely on: an entry whose ``claim_token`` does not match the
    source's live claim is not written.
    """

    def __init__(self, claims: dict[str, dict[str, Any]]) -> None:
        self.claims = claims
        self.entries: dict[str, list[dict[str, Any]]] = {}
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.__enter__ = lambda: self  # noqa: E731
        self.__exit__ = lambda *a: False  # noqa: E731

    def __enter__(self) -> _FakeGraphClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((cypher, params))
        written = 0
        for entry in params.get("entries") or []:
            claim = self.claims.get(entry["source_id"])
            if entry.get("claim_token") is not None:
                if claim is None or (
                    claim["claim_token"] != entry["claim_token"]
                    or claim["claim_seq"] != entry["claim_seq"]
                ):
                    continue
            bucket = self.entries.setdefault(entry["source_id"], [])
            if any(existing["id"] == entry["id"] for existing in bucket):
                continue
            bucket.append(dict(entry))
            written += 1
        return [{"written": written}]


class _FakeServices:
    def __init__(self, gc: _FakeGraphClient) -> None:
        self.GraphClient = lambda: gc


def test_claim_attempt_entry_names_each_outcome() -> None:
    """Each of the five outcomes yields a distinct, well-formed entry."""
    entries = [
        gcc.claim_attempt_entry(
            "dd:equilibrium/time_slice/profiles_1d/psi",
            attempt=1,
            outcome=outcome,
            reason=f"{outcome} reason",
            claim_seq=1,
        )
        for outcome in sorted(gcc.CLAIM_ATTEMPT_OUTCOMES)
    ]

    assert {entry["outcome"] for entry in entries} == set(gcc.CLAIM_ATTEMPT_OUTCOMES)
    assert len({entry["id"] for entry in entries}) == 5
    for entry in entries:
        assert entry["source_id"] == "dd:equilibrium/time_slice/profiles_1d/psi"
        assert entry["attempt"] == 1
        assert entry["reason"] == f"{entry['outcome']} reason"


def test_claim_attempt_entry_rejects_unknown_outcome() -> None:
    """A writer cannot invent an outcome the census does not know."""
    with pytest.raises(ValueError, match="unknown compose claim outcome"):
        gcc.claim_attempt_entry("dd:x", attempt=1, outcome="produced_later")


def test_claim_attempt_append_binds_the_calling_variable() -> None:
    """The shared stanza names the caller's source variable, not a fixed one."""
    stanza = gcc.claim_attempt_append_cypher(sns_var="s")
    assert "MATCH (s:StandardNameSource {id: entry.source_id})" in stanza
    assert "MERGE (s)-[:HAS_CLAIM_ATTEMPT]->(attempt_row)" in stanza
    assert "StandardNameSourceClaimAttempt" in stanza


def test_source_at_attempt_cap_holds_one_entry_per_attempt() -> None:
    """A source failed after five attempts carries five entries, not one."""
    claim = {"claim_token": "tok", "claim_seq": 5}
    gc = _FakeGraphClient({"dd:some/path": claim})
    services = _FakeServices(gc)

    for attempt in range(1, 6):
        gcc.record_claim_attempts(
            [
                gcc.claim_attempt_entry(
                    "dd:some/path",
                    attempt=attempt,
                    outcome="refused",
                    reason=f"refusal {attempt}",
                    claim_seq=attempt,
                )
            ],
            services=services,
        )

    held = gc.entries["dd:some/path"]
    assert len(held) == 5
    assert [entry["attempt"] for entry in held] == [1, 2, 3, 4, 5]
    assert [entry["reason"] is not None for entry in held] == [True] * 5


def test_record_claim_attempts_fences_out_a_foreign_claim() -> None:
    """An entry whose token and sequence do not match the live claim is refused."""
    gc = _FakeGraphClient(
        {"dd:some/path": {"claim_token": "live-token", "claim_seq": 7}}
    )
    services = _FakeServices(gc)
    entry = gcc.claim_attempt_entry(
        "dd:some/path", attempt=2, outcome="batch_failed", claim_seq=6
    )
    entry["claim_token"] = "stale-token"

    written = gcc.record_claim_attempts([entry], services=services)

    assert written == 0
    assert gc.entries == {}


def test_record_claim_attempts_is_idempotent_under_retry() -> None:
    """A deadlock retry re-appends the same entry without duplicating it."""
    gc = _FakeGraphClient({"dd:some/path": {"claim_token": "t", "claim_seq": 1}})
    services = _FakeServices(gc)
    entry = gcc.claim_attempt_entry("dd:some/path", attempt=1, outcome="released")

    gcc.record_claim_attempts([entry], services=services)
    gcc.record_claim_attempts([entry], services=services)

    assert len(gc.entries["dd:some/path"]) == 1


def test_batch_failure_records_every_claimed_source(monkeypatch) -> None:
    """A raised batch leaves one batch_failed receipt per claimed source."""
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    recorded: list[list[dict[str, Any]]] = []

    def _capture(entries, *, services):
        recorded.append(entries)
        return len(entries)

    monkeypatch.setattr(gcc, "record_claim_attempts", _capture)
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_foundation.get_services",
        lambda: object(),
    )

    wcb._record_batch_failure(
        [
            {
                "id": "dd:a",
                "claim_token": "tok-a",
                "claim_seq": 3,
                "attempt_count": 3,
            },
            {"id": "dd:b", "claim_token": 0, "claim_seq": 1, "attempt_count": 1},
            {"id": None},
        ],
        "batch-key",
        "generate_name",
    )

    assert len(recorded) == 1
    entries = recorded[0]
    assert [entry["source_id"] for entry in entries] == ["dd:a", "dd:b"]
    assert {entry["outcome"] for entry in entries} == {"batch_failed"}
    assert {entry["batch_key"] for entry in entries} == {"batch-key"}
    assert "generate_name" in entries[0]["reason"]


def test_batch_failure_never_masks_the_batch_error(monkeypatch) -> None:
    """A receipt write that fails does not replace the exception that ended it."""
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    def _explode(entries, *, services):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(gcc, "record_claim_attempts", _explode)
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_foundation.get_services",
        lambda: object(),
    )

    wcb._record_batch_failure(
        [{"id": "dd:a", "claim_token": "t", "claim_seq": 1, "attempt_count": 1}],
        "batch-key",
        "generate_name",
    )


@pytest.mark.parametrize(
    ("module_path", "function_name", "outcome"),
    [
        (
            "imas_codex.standard_names.graph_store.graph_source_bindings",
            "persist_claimed_source_outcomes",
            "refused",
        ),
        (
            "imas_codex.standard_names.orphan_sweep",
            "_orphan_sweep_tick",
            "capped",
        ),
    ],
)
def test_ending_writer_carries_its_outcome(
    module_path: str, function_name: str, outcome: str
) -> None:
    """Each claim-ending writer appends its own outcome, not a generic one."""
    import importlib

    module = importlib.import_module(module_path)
    source = inspect.getsource(module)
    assert "HAS_CLAIM_ATTEMPT" in source
    if outcome == "refused":
        assert "coalesce(b.claim_outcome, 'refused')" in source
    else:
        assert "attempt_row.outcome = 'capped'" in source


def test_compose_batch_records_the_receipt_before_it_reraises() -> None:
    """The batch failure path itself writes the receipt, not a caller.

    The helper is only half the requirement: what makes the receipt reach a
    real source is the compose batch's own failure path invoking it before it
    re-raises. This asserts the call is present in that path, so removing the
    append is a test failure rather than a silent gap in the record.
    """
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    source = inspect.getsource(wcb.compose_batch)
    except_index = source.rindex("except Exception:")
    failure_path = source[except_index:]
    assert "_record_batch_failure(batch, group_key, phase_tag)" in failure_path
    assert failure_path.index("_record_batch_failure") < failure_path.index(
        "\n        raise\n"
    )
