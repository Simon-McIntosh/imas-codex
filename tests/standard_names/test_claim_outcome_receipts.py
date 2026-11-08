"""Every compose claim that ends records its outcome on the source.

A claim ends in exactly one of seven ways. Each ending writer appends a
``StandardNameSourceClaimAttempt`` entry on the source it ends, so a failure
census can read one entry per attempt instead of a single overwritten
``last_error``.

These tests drive the real writer for each outcome through a graph double that
implements the append statement's fence and its create-once identity. A stored
receipt is what is asserted, never the entry a writer built, so an entry that
omits the claim sequence -- and is therefore fenced out of the graph -- fails
here rather than passing on the strength of its own shape.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from imas_codex.standard_names.graph_store import (
    graph_foundation as claims,
    graph_source_bindings as gsb,
)


class _FakeGraphClient:
    """Graph double that executes the append statement's fence and identity.

    Sources are seeded with their live claim. An entry is written only when the
    statement would write it: the source must match, and when the entry carries
    a token the source's live token and sequence must equal it. A missing
    ``claim_seq`` therefore compares against a null and writes nothing, exactly
    as Cypher does.
    """

    def __init__(self, claims: dict[str, dict[str, Any]]) -> None:
        self.claims = claims
        self.receipts: dict[str, dict[str, Any]] = {}
        self.links: list[tuple[str, str]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.status_writes: list[dict[str, Any]] = []
        self.parking_rows: list[dict[str, Any]] = []

    def __enter__(self) -> _FakeGraphClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((cypher, params))
        if "$max_compose_attempts" in cypher:
            return [dict(row) for row in self.parking_rows]
        if "StandardNameSourceClaimAttempt" in cypher:
            return [{"written": self._append(params.get("entries") or [])}]
        if "SET sns.status = b.status" in cypher:
            matched = []
            for item in params.get("batch") or []:
                claim = self.claims.get(item["sns_id"])
                if claim is None:
                    continue
                if (
                    claim["claim_token"] != item["claim_token"]
                    or claim["claim_seq"] != item["claim_seq"]
                ):
                    continue
                self.status_writes.append(item)
                matched.append({"id": item["sns_id"]})
            return matched
        return [{"n": 0}]

    def _append(self, entries: list[dict[str, Any]]) -> int:
        written = 0
        for entry in entries:
            claim = self.claims.get(entry["source_id"])
            if claim is None:
                continue
            if entry.get("claim_token") is not None and (
                claim["claim_token"] != entry["claim_token"]
                or claim["claim_seq"] != entry["claim_seq"]
            ):
                continue
            if entry["id"] in self.receipts:
                self.links.append((entry["source_id"], entry["id"]))
                continue
            self.receipts[entry["id"]] = {
                "id": entry["id"],
                "source_id": entry["source_id"],
                "attempt": entry["attempt"],
                "outcome": entry["outcome"],
                "reason": entry["reason"],
                "batch_key": entry["batch_key"],
                "run_id": entry["run_id"],
            }
            self.links.append((entry["source_id"], entry["id"]))
            written += 1
        return written

    def held_by(self, source_id: str) -> list[dict[str, Any]]:
        return [
            receipt
            for receipt in self.receipts.values()
            if receipt["source_id"] == source_id
        ]


def _services(gc: _FakeGraphClient, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(GraphClient=lambda: gc, **extra)


def _claim(claim_token: str = "tok", claim_seq: int = 1) -> dict[str, Any]:
    return {"claim_token": claim_token, "claim_seq": claim_seq}


# ---------------------------------------------------------------------------
# The entry contract
# ---------------------------------------------------------------------------


def test_claim_attempt_entry_names_each_outcome() -> None:
    """Every outcome the census knows produces a distinct, keyed entry."""
    entries = [
        claims.claim_attempt_entry(
            "dd:equilibrium/time_slice/profiles_1d/psi",
            attempt=1,
            outcome=outcome,
            reason=f"{outcome} reason",
            claim_seq=4,
        )
        for outcome in sorted(claims.CLAIM_ATTEMPT_OUTCOMES)
    ]

    assert {entry["outcome"] for entry in entries} == set(claims.CLAIM_ATTEMPT_OUTCOMES)
    assert len({entry["id"] for entry in entries}) == len(entries)
    for entry in entries:
        assert entry["source_id"] == "dd:equilibrium/time_slice/profiles_1d/psi"
        assert entry["claim_seq"] == 4
        assert entry["attempt"] == 1
        assert entry["reason"] == f"{entry['outcome']} reason"


def test_claim_attempt_entry_requires_a_claim_sequence() -> None:
    """A fenced writer must state the sequence it is closing."""
    with pytest.raises(TypeError, match="claim_seq"):
        claims.claim_attempt_entry("dd:x", attempt=1, outcome="refused")


def test_an_unclaimed_source_keys_its_receipt_on_the_attempt() -> None:
    """The sweep that parks a source at the cap holds no sequence to key on."""
    entry = claims.claim_attempt_entry(
        "dd:x", attempt=5, outcome="capped", claim_seq=None
    )

    assert entry["claim_seq"] is None
    assert entry["id"] == "claim-attempt:dd:x:attempt-5:capped"


def test_claim_attempt_entry_rejects_unknown_outcome() -> None:
    """A writer cannot invent an outcome the census does not know."""
    with pytest.raises(ValueError, match="unknown compose claim outcome"):
        claims.claim_attempt_entry(
            "dd:x", attempt=1, outcome="produced_later", claim_seq=1
        )


# ---------------------------------------------------------------------------
# The outcome mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("failed", "refused"),
        ("skipped", "skipped"),
        ("extracted", "requeued"),
        ("vocab_gap", "refused"),
    ],
)
def test_status_maps_to_its_own_outcome(status: str, outcome: str) -> None:
    """A skipped source and a requeued batch are never recorded as refusals."""
    assert claims.claim_outcome_for_status(status) == outcome


# ---------------------------------------------------------------------------
# Each ending writer, driven through the graph double
# ---------------------------------------------------------------------------


def test_produced_writer_stores_a_produced_receipt() -> None:
    """A bound candidate leaves a produced receipt carrying the claim's run."""
    gc = _FakeGraphClient({"dd:a": _claim("tok-a", 3)})
    batch = [
        {
            "sns_id": "dd:a",
            "claim_token": "tok-a",
            "claim_seq": 3,
            "attempt_count": 3,
            "batch_key": "bk",
            "run_id": "run-1",
        }
    ]

    claims.append_claim_attempts(gc, gsb._produced_claim_entries(batch))

    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == "produced"
    assert held[0]["attempt"] == 3
    assert held[0]["batch_key"] == "bk"
    assert held[0]["run_id"] == "run-1"
    assert held[0]["reason"] is None


def test_released_writer_stores_a_released_receipt_with_its_reason() -> None:
    """A binding refusal returns the claim with the collision it hit."""
    gc = _FakeGraphClient({"dd:a": _claim("tok-a", 2)})
    batch = [
        {
            "sns_id": "dd:a",
            "sn_id": "name_candidate",
            "unit": "m",
            "claim_token": "tok-a",
            "claim_seq": 2,
            "attempt_count": 2,
        }
    ]

    claims.append_claim_attempts(gc, gsb._released_claim_entries(batch))

    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == "released"
    assert "name_candidate" in held[0]["reason"]


def test_attachment_refusal_reason_names_the_attachment_error() -> None:
    """An attachment refusal records the rejection, not a generic collision."""
    gc = _FakeGraphClient({"dd:a": _claim("t", 1)})
    batch = [
        {
            "sns_id": "dd:a",
            "sn_id": "cand",
            "attachment_error": "is already attached to another source",
            "claim_token": "t",
            "claim_seq": 1,
        }
    ]

    claims.append_claim_attempts(gc, gsb._released_claim_entries(batch))

    assert "already attached" in gc.held_by("dd:a")[0]["reason"]


def test_batch_failure_writer_stores_a_receipt_per_claimed_source() -> None:
    """A raised batch leaves one batch_failed receipt per source it carried."""
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    gc = _FakeGraphClient({"dd:a": _claim("tok-a", 3), "dd:b": _claim("tok-b", 3)})
    entries = wcb._batch_failure_claim_entries(
        [
            {
                "id": "dd:a",
                "claim_token": "tok-a",
                "claim_seq": 3,
                "attempt_count": 3,
                "run_id": "run-9",
            },
            {
                "id": "dd:b",
                "claim_token": "tok-b",
                "claim_seq": 3,
                "attempt_count": 3,
            },
            {"id": None},
        ],
        "batch-key",
        "generate_name",
    )

    claims.append_claim_attempts(gc, entries)

    assert len(gc.held_by("dd:a")) == 1
    assert len(gc.held_by("dd:b")) == 1
    assert {r["outcome"] for r in gc.receipts.values()} == {"batch_failed"}
    assert gc.held_by("dd:a")[0]["run_id"] == "run-9"
    assert "generate_name" in gc.held_by("dd:a")[0]["reason"]


def test_stored_status_writer_records_refused_for_a_failed_source() -> None:
    """A source parked failed by pre-validation or a persist refusal is refused."""
    gc = _FakeGraphClient({"dd:a": _claim("tok-a", 2)})
    batch = [
        {
            "sns_id": "dd:a",
            "status": "failed",
            "last_error": "candidate pre-validation failed (grammar): bad_name",
            "claim_token": "tok-a",
            "claim_seq": 2,
            "attempt_count": 2,
        }
    ]

    claims.append_claim_attempts(gc, gsb._stored_outcome_claim_entries(batch))

    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == "refused"
    assert "grammar" in held[0]["reason"]


@pytest.mark.parametrize(
    ("status", "outcome"), [("skipped", "skipped"), ("extracted", "requeued")]
)
def test_stored_status_writer_never_records_a_refusal_for_a_parked_or_requeued_source(
    status: str, outcome: str
) -> None:
    """Skipped and requeued ends are their own outcomes, not refusals."""
    gc = _FakeGraphClient({"dd:a": _claim("tok", 1)})
    batch = [
        {
            "sns_id": "dd:a",
            "status": status,
            "skip_reason_detail": "bare timestamp",
            "claim_token": "tok",
            "claim_seq": 1,
        }
    ]

    claims.append_claim_attempts(gc, gsb._stored_outcome_claim_entries(batch))

    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == outcome


def test_capped_writer_stores_a_capped_receipt() -> None:
    """The attempt-cap sweep leaves a capped receipt on each row it parks."""
    from imas_codex.standard_names import orphan_sweep

    gc = _FakeGraphClient({"dd:a": _claim("tok", 5)})
    entries = orphan_sweep._capped_claim_entries(
        [
            {
                "id": "dd:a",
                "attempt_count": 5,
                "claim_seq": 5,
                "last_error": "compose claim-attempt cap reached",
                "batch_key": "bk",
            }
        ]
    )

    claims.append_claim_attempts(gc, entries)

    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == "capped"
    assert held[0]["attempt"] == 5
    assert held[0]["batch_key"] == "bk"
    assert "cap reached" in held[0]["reason"]


def test_the_sweep_tick_issues_the_capped_append(monkeypatch) -> None:
    """The tick itself appends: deleting the call leaves no capped receipt."""
    from imas_codex.standard_names import orphan_sweep
    from imas_codex.standard_names.graph_store import graph_source_bindings

    gc = _FakeGraphClient({"dd:a": _claim("tok", 5)})
    gc.parking_rows = [
        {
            "id": "dd:a",
            "attempt_count": 5,
            "claim_seq": 5,
            "last_error": "compose claim-attempt cap reached",
            "batch_key": "bk",
        }
    ]
    monkeypatch.setattr(orphan_sweep, "GraphClient", lambda: gc)
    monkeypatch.setattr(
        graph_source_bindings,
        "record_dispositions_for_undisposed_parks",
        lambda **_: {"written": 1, "counts": {}},
    )

    counts = orphan_sweep._orphan_sweep_tick(timeout_s=300)

    assert counts["compose_attempt_cap"] == 1
    held = gc.held_by("dd:a")
    assert len(held) == 1
    assert held[0]["outcome"] == "capped"


# ---------------------------------------------------------------------------
# The fence and the record's bounds
# ---------------------------------------------------------------------------


def test_a_stale_claim_writes_no_receipt() -> None:
    """A delayed duplicate cannot append to a claim it no longer holds."""
    gc = _FakeGraphClient({"dd:a": _claim("live-token", 7)})
    entry = claims.claim_attempt_entry(
        "dd:a",
        attempt=2,
        outcome="batch_failed",
        claim_seq=6,
        claim_token="stale-token",
    )

    assert claims.append_claim_attempts(gc, [entry]) == 0
    assert gc.receipts == {}


def test_a_source_at_the_attempt_cap_holds_one_receipt_per_attempt() -> None:
    """A source that fails five times carries five receipts, not one."""
    gc = _FakeGraphClient({"dd:a": _claim("tok", 1)})
    written = 0
    for attempt in range(1, 6):
        # Each re-claim bumps the sequence, so the source's live sequence is
        # the one the attempt it is recording was made under.
        gc.claims["dd:a"] = _claim("tok", attempt)
        written += claims.append_claim_attempts(
            gc,
            [
                claims.claim_attempt_entry(
                    "dd:a",
                    attempt=attempt,
                    outcome="refused",
                    reason=f"refusal {attempt}",
                    claim_seq=attempt,
                    claim_token="tok",
                )
            ],
        )

    held = gc.held_by("dd:a")
    assert written == 5
    assert len(held) == 5
    assert sorted(receipt["attempt"] for receipt in held) == [1, 2, 3, 4, 5]


def test_the_written_count_is_receipts_created_not_claims_matched() -> None:
    """A retried write of an existing receipt reports nothing new."""
    gc = _FakeGraphClient({"dd:a": _claim("tok", 1)})
    entry = claims.claim_attempt_entry(
        "dd:a",
        attempt=1,
        outcome="released",
        claim_seq=1,
        claim_token="tok",
    )

    assert claims.append_claim_attempts(gc, [entry]) == 1
    assert claims.append_claim_attempts(gc, [entry]) == 0
    assert len(gc.receipts) == 1
    assert gc.links == [("dd:a", entry["id"]), ("dd:a", entry["id"])]


def test_every_receipt_names_its_source() -> None:
    """The schema requires source_id on the receipt, so no writer may omit it."""
    gc = _FakeGraphClient({"dd:a": _claim("tok", 1)})
    claims.append_claim_attempts(
        gc,
        gsb._produced_claim_entries(
            [{"sns_id": "dd:a", "claim_token": "tok", "claim_seq": 1}]
        ),
    )

    assert gc.receipts[next(iter(gc.receipts))]["source_id"] == "dd:a"


# ---------------------------------------------------------------------------
# The batch failure path reaches the writer
# ---------------------------------------------------------------------------


def test_compose_batch_records_the_receipt_before_it_reraises() -> None:
    """The batch failure path itself writes the receipt, not a caller.

    The entry builder is only half the requirement: what puts a receipt on a
    real source is the compose batch's own failure path invoking it before it
    re-raises.
    """
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    source = inspect.getsource(wcb.compose_batch)
    failure_path = source[source.rindex("except Exception:") :]
    assert "_record_batch_failure(batch, group_key, phase_tag)" in failure_path
    assert failure_path.index("_record_batch_failure") < failure_path.index(
        "\n        raise\n"
    )


def test_batch_failure_never_masks_the_batch_error(monkeypatch) -> None:
    """A receipt write that fails does not replace the exception that ended it."""
    from imas_codex.standard_names.worker_tasks import worker_compose_batches as wcb

    def _explode(entries, *, services):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(claims, "record_claim_attempts", _explode)
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_foundation.get_services",
        lambda: object(),
    )

    wcb._record_batch_failure(
        [{"id": "dd:a", "claim_token": "t", "claim_seq": 1, "attempt_count": 1}],
        "batch-key",
        "generate_name",
    )


# ---------------------------------------------------------------------------
# The append statement against the live graph
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_the_append_statement_creates_and_links_a_receipt_in_the_live_graph() -> None:
    """The statement the double models behaves the same way in Neo4j.

    A double proves the writers build entries the fence accepts; only the real
    statement proves the fence, the create-once identity and the relationship
    are what the double claims. The probe source is created and removed here,
    so it leaves nothing behind.
    """
    from imas_codex.graph.client import GraphClient

    source_id = "dd:__claim_receipt_probe__/equilibrium/time_slice/psi"
    with GraphClient() as gc:
        gc.query(
            """
            MERGE (s:StandardNameSource {id: $id})
            SET s.claim_token = 'probe-token', s.claim_seq = 3,
                s.status = 'extracting'
            """,
            id=source_id,
        )
        try:
            entry = claims.claim_attempt_entry(
                source_id,
                attempt=3,
                outcome="batch_failed",
                reason="probe batch failure",
                claim_seq=3,
                claim_token="probe-token",
                batch_key="probe-batch",
                run_id="probe-run",
            )
            assert claims.append_claim_attempts(gc, [entry]) == 1
            assert claims.append_claim_attempts(gc, [entry]) == 0

            stored = gc.query(
                """
                MATCH (s:StandardNameSource {id: $id})
                      -[r:HAS_CLAIM_ATTEMPT]->(a:StandardNameSourceClaimAttempt)
                RETURN a.id AS id, a.source_id AS source_id, a.attempt AS attempt,
                       a.outcome AS outcome, a.reason AS reason,
                       a.batch_key AS batch_key, a.run_id AS run_id,
                       a.recorded_at IS NOT NULL AS recorded
                """,
                id=source_id,
            )
            assert len(stored) == 1
            row = stored[0]
            assert row["id"] == entry["id"]
            assert row["source_id"] == source_id
            assert row["attempt"] == 3
            assert row["outcome"] == "batch_failed"
            assert row["reason"] == "probe batch failure"
            assert row["batch_key"] == "probe-batch"
            assert row["run_id"] == "probe-run"
            assert row["recorded"]

            stale = claims.claim_attempt_entry(
                source_id,
                attempt=4,
                outcome="released",
                claim_seq=2,
                claim_token="probe-token",
            )
            assert claims.append_claim_attempts(gc, [stale]) == 0
        finally:
            gc.query(
                """
                MATCH (s:StandardNameSource {id: $id})
                OPTIONAL MATCH (s)-[:HAS_CLAIM_ATTEMPT]->(a)
                DETACH DELETE a, s
                """,
                id=source_id,
            )
