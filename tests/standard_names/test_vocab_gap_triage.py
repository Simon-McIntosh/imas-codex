"""Bucket triage over stored vocab-gap nodes.

The triage derives one lifecycle bucket per node from fields earlier passes
wrote (category from the classifier reconcile, the mechanical reuse verdict,
the last-seen stamp) — it never re-derives them itself and never deletes.
"""

from unittest.mock import MagicMock

STALE_BEFORE = "2026-07-28"

ROWS = [
    {
        "id": "vocab_gap:device:lower_hybrid_antenna_module",
        "segment": "device",
        "token": "lower_hybrid_antenna_module",
        "category": "absent",
        "dedup": "reuse_confirmed",
        "last_seen": "2026-06-19T10:00:00Z",
        "n": 3,
    },
    {
        "id": "vocab_gap:position:line_of_sight_second_point",
        "segment": "position",
        "token": "line_of_sight_second_point",
        "category": "rule_violation",
        "dedup": "unchecked",
        "last_seen": "2026-07-28T09:00:00Z",
        "n": 19,
    },
    {
        "id": "vocab_gap:qualifier:inverse_square",
        "segment": "qualifier",
        "token": "inverse_square",
        "category": "decomposable",
        "dedup": "unchecked",
        "last_seen": "2026-07-28T09:00:00Z",
        "n": 4,
    },
    {
        "id": "vocab_gap:physical_base:orbit_integral",
        "segment": "physical_base",
        "token": "orbit_integral",
        "category": "absent",
        "dedup": "unchecked",
        "last_seen": "2026-06-20T10:00:00Z",
        "n": 2,
    },
    {
        "id": "vocab_gap:physical_base:detection_efficiency",
        "segment": "physical_base",
        "token": "detection_efficiency",
        "category": "absent",
        "dedup": "distinct_confirmed",
        "last_seen": "2026-07-29T06:00:00Z",
        "n": 14,
    },
]


def _client() -> MagicMock:
    gc = MagicMock()

    def _query(q: str, **params):
        if "RETURN vg.id AS id" in q:
            return [dict(r) for r in ROWS]
        return []

    gc.query.side_effect = _query
    return gc


def test_each_row_lands_in_exactly_one_bucket() -> None:
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    res = triage_vocab_gaps(_client(), stale_before=STALE_BEFORE, dry_run=True)
    assert res["checked"] == len(ROWS)
    assert sum(res["counts"].values()) == len(ROWS)
    assert res["counts"] == {
        "reuse": 1,
        "rule_violation": 1,
        "composable": 1,
        "retired_stale": 1,
        "genuine": 1,
    }


def test_reuse_verdict_outranks_staleness() -> None:
    """A mechanically-resolved token is reuse even when last seen in June."""
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    res = triage_vocab_gaps(_client(), stale_before=STALE_BEFORE, dry_run=True)
    assert [r["token"] for r in res["genuine"]] == ["detection_efficiency"]


def test_genuine_list_carries_rotation_inputs() -> None:
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    res = triage_vocab_gaps(_client(), stale_before=STALE_BEFORE, dry_run=True)
    entry = res["genuine"][0]
    assert entry["segment"] == "physical_base"
    assert entry["example_count"] == 14


def test_dry_run_writes_nothing() -> None:
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    gc = _client()
    triage_vocab_gaps(gc, stale_before=STALE_BEFORE, dry_run=True)
    assert gc.query.call_count == 1, "only the read ran"


def test_write_stamps_every_node_with_its_bucket() -> None:
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    gc = _client()
    triage_vocab_gaps(gc, stale_before=STALE_BEFORE, dry_run=False)
    assert gc.query.call_count == 2
    (_, ), kwargs = gc.query.call_args
    items = {it["id"]: it["bucket"] for it in kwargs["items"]}
    assert items["vocab_gap:device:lower_hybrid_antenna_module"] == "reuse"
    assert items["vocab_gap:physical_base:orbit_integral"] == "retired_stale"
    assert items["vocab_gap:physical_base:detection_efficiency"] == "genuine"
