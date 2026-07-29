"""Blocking decisions cached on a StandardNameSource must be re-evaluated.

Two independent blocks are written durably onto a ``StandardNameSource`` and,
historically, were never revisited when the cause was fixed upstream:

* **Unit skips.** ``_apply_unit_overrides`` skips a DD row whose effective unit
  fails ``_is_unparseable_dd_unit`` and records
  ``skip_reason='dd_unit_unresolvable'``. Once the resolver learns to parse that
  unit (dimensionless sentinels, count pseudo-units, canonical symbol order),
  the source is still parked. :func:`revive_unit_skipped_sources` re-evaluates
  every unit-derived skip against the CURRENT resolver — the same
  ``resolve_unit`` + ``_is_unparseable_dd_unit`` pair the extractor uses — and
  revives only the ones that now parse.

* **Vocabulary gaps.** ``reconcile_vocab_gaps`` un-parks a ``vocab_gap`` source
  only when the *exact* token it asked for turns out to exist. A composer that
  asked for the wrong spelling of a real capability therefore stays parked
  forever even though an ISN vocabulary addition made the quantity expressible
  another way. :func:`retry_vocab_gap_sources_on_grammar_change` gives every
  parked source ONE retry per ISN vocabulary change, keyed on a digest of the
  vocabulary in force when it was parked.

Both are wired into the ``run_sn_pools`` startup reconcile so every ``sn run``
self-heals. Idempotency is the contract: a second pass at an unchanged
resolver / unchanged vocabulary must revive nothing.

Mocked unit tests run in the default tier; end-to-end guarantees run against a
live graph (``@pytest.mark.graph``).
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentAuditResult

_GO = "imas_codex.standard_names.graph_ops"
_AA = "imas_codex.standard_names.attachment_audit"
_DGO = "imas_codex.graph.dd_graph_ops"
_LOOP = "imas_codex.standard_names.loop"
_PREFIX = "test_source_revival__"


# ---------------------------------------------------------------------------
# Fixtures (prefix-scoped; safe against the shared graph)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"Neo4j not available: {exc}")
    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    def _wipe() -> None:
        # A StandardNameSource id carries the ``dd:`` source-type prefix, so
        # match anywhere in the id rather than only at the start.
        for label in ("StandardNameSource", "IMASNode", "Unit", "VocabGap"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _create_dd_source(
    gc,
    *,
    path: str,
    unit: str | None,
    status: str,
    skip_reason: str | None,
    attempt_count: int = 0,
    lifecycle_status: str = "active",
) -> str:
    """Create an IMASNode (+ optional Unit) and a StandardNameSource on it."""
    sns_id = f"dd:{path}"
    gc.query(
        """
        MERGE (n:IMASNode {id: $path})
        SET n.unit = $unit, n.lifecycle_status = $lifecycle,
            n.node_category = 'quantity'
        MERGE (sns:StandardNameSource {id: $sns_id})
        SET sns.source_type = 'dd',
            sns.source_id = $path,
            sns.status = $status,
            sns.skip_reason = $skip_reason,
            sns.attempt_count = $attempts
        MERGE (sns)-[:FROM_DD_PATH]->(n)
        """,
        path=path,
        unit=unit,
        lifecycle=lifecycle_status,
        sns_id=sns_id,
        status=status,
        skip_reason=skip_reason,
        attempts=attempt_count,
    )
    if unit:
        gc.query(
            """
            MERGE (u:Unit {id: $unit})
            WITH u
            MATCH (n:IMASNode {id: $path})
            MERGE (n)-[:HAS_UNIT]->(u)
            """,
            unit=unit,
            path=path,
        )
    return sns_id


def _park_vocab_gap_source(
    gc,
    *,
    path: str,
    signature: str | None,
    attempt_count: int = 1,
) -> str:
    """Create a StandardNameSource parked at ``vocab_gap`` with a blocking gap."""
    sns_id = f"dd:{path}"
    gap_id = f"{_PREFIX}gap:{path}"
    gc.query(
        """
        MERGE (n:IMASNode {id: $path})
        SET n.node_category = 'quantity', n.lifecycle_status = 'active'
        MERGE (sns:StandardNameSource {id: $sns_id})
        SET sns.source_type = 'dd',
            sns.source_id = $path,
            sns.status = 'vocab_gap',
            sns.skip_reason = 'vocab_gap',
            sns.attempt_count = $attempts,
            sns.vocab_gap_grammar_signature = $signature
        MERGE (sns)-[:FROM_DD_PATH]->(n)
        MERGE (vg:VocabGap {id: $gap_id})
        SET vg.segment = 'physical_base', vg.token = 'not_a_real_token',
            vg.category = 'absent'
        MERGE (sns)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
        """,
        path=path,
        sns_id=sns_id,
        attempts=attempt_count,
        signature=signature,
        gap_id=gap_id,
    )
    return sns_id


def _source(gc, sns_id: str) -> dict:
    rows = gc.query(
        "MATCH (sns:StandardNameSource {id: $id}) RETURN properties(sns) AS p",
        id=sns_id,
    )
    return rows[0]["p"] if rows else {}


# ---------------------------------------------------------------------------
# Part A — stale unit skips are revived against the current resolver
# ---------------------------------------------------------------------------


def test_revive_unit_skipped_sources_returns_counts():
    """Fast return-shape guard: no revivable candidates → zero revived."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])
    result = revive_unit_skipped_sources(gc=mock_gc)
    assert result == {"checked": 0, "revived": 0}


def test_revive_unit_skipped_sources_uses_the_extractor_resolver():
    """The revival verdict comes from the extractor's own parse pair.

    A candidate whose unit the extractor would now accept is revived; the write
    is issued with exactly that source id.
    """
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    calls: list[tuple[str, dict]] = []

    def _query(cypher: str, **params):
        calls.append((cypher, params))
        if "skip_reason" in cypher and "RETURN" in cypher and "SET" not in cypher:
            return [
                {
                    "id": "dd:a/dimensionless",
                    "source_id": "a/dimensionless",
                    "unit_from_rel": "1",
                    "node_unit": "1",
                },
                {
                    "id": "dd:b/corrupt",
                    "source_id": "b/corrupt",
                    "unit_from_rel": "as_parent_level_2",
                    "node_unit": "as_parent_level_2",
                },
            ]
        return [{"revived": 1}]

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(side_effect=_query)

    result = revive_unit_skipped_sources(gc=mock_gc)

    assert result["checked"] == 2
    assert result["revived"] == 1
    write_calls = [c for c in calls if "SET" in c[0]]
    assert len(write_calls) == 1
    assert write_calls[0][1]["ids"] == ["dd:a/dimensionless"]


def test_revive_unit_skipped_sources_no_write_when_nothing_resolves():
    """A candidate whose unit still fails the parse issues no write at all."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    calls: list[str] = []

    def _query(cypher: str, **params):
        calls.append(cypher)
        if "SET" not in cypher:
            return [
                {
                    "id": "dd:b/corrupt",
                    "source_id": "b/corrupt",
                    "unit_from_rel": "kg m",
                    "node_unit": "kg m",
                }
            ]
        return [{"revived": 0}]

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(side_effect=_query)

    result = revive_unit_skipped_sources(gc=mock_gc)

    assert result == {"checked": 1, "revived": 0}
    assert not [c for c in calls if "SET" in c]


@pytest.mark.graph
def test_stale_unit_skip_is_revived(_gc, _clean):
    """A dimensionless unit the resolver now parses returns to 'extracted'."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("dimensionless"),
        unit="1",
        status="skipped",
        skip_reason="dd_unit_unresolvable",
        attempt_count=5,
    )

    result = revive_unit_skipped_sources(gc=_gc)

    props = _source(_gc, sns_id)
    assert props["status"] == "extracted"
    assert props.get("skip_reason") is None
    assert props.get("skip_reason_detail") is None
    assert props.get("claimed_at") is None
    # The prior attempts were spent under the broken resolver — a revived
    # source must be seedable again, so the counter is reset.
    assert props["attempt_count"] == 0
    assert result["revived"] >= 1


@pytest.mark.graph
def test_genuinely_unparseable_unit_is_not_revived(_gc, _clean):
    """A DD-corrupted unit string stays skipped."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("corrupt"),
        unit="as_parent_level_2",
        status="skipped",
        skip_reason="dd_unit_unresolvable",
    )

    revive_unit_skipped_sources(gc=_gc)

    props = _source(_gc, sns_id)
    assert props["status"] == "skipped"
    assert props["skip_reason"] == "dd_unit_unresolvable"


@pytest.mark.graph
def test_permanent_eligibility_skip_is_untouched(_gc, _clean):
    """A non-unit eligibility skip (time coordinate) is a correct exclusion."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("time"),
        unit="s",
        status="skipped",
        skip_reason="non_nameable_coordinate:time",
    )
    temporal_id = _create_dd_source(
        _gc,
        path=_uid("temporal"),
        unit="s",
        status="skipped",
        skip_reason="temporal_coordinate",
    )

    revive_unit_skipped_sources(gc=_gc)

    assert _source(_gc, sns_id)["status"] == "skipped"
    assert _source(_gc, temporal_id)["status"] == "skipped"


@pytest.mark.graph
def test_removed_dd_path_is_not_resurrected(_gc, _clean):
    """A skip whose DD node the current DD removed stays skipped."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("removed"),
        unit="1",
        status="skipped",
        skip_reason="dd_unit_unresolvable",
        lifecycle_status="removed",
    )

    revive_unit_skipped_sources(gc=_gc)

    assert _source(_gc, sns_id)["status"] == "skipped"


@pytest.mark.graph
def test_downstream_status_with_stale_skip_crumb_is_untouched(_gc, _clean):
    """A source already past the skip keeps its status.

    ``skip_reason`` survives on sources that later composed/attached — it is an
    audit crumb there, not a block, and reviving would rewind real progress.
    """
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("attached"),
        unit="1",
        status="attached",
        skip_reason="dd_unit_unresolvable",
    )

    revive_unit_skipped_sources(gc=_gc)

    assert _source(_gc, sns_id)["status"] == "attached"


@pytest.mark.graph
def test_unit_revival_is_idempotent(_gc, _clean):
    """A second pass at the same resolver revives nothing further."""
    from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

    sns_id = _create_dd_source(
        _gc,
        path=_uid("dimensionless"),
        unit="1",
        status="skipped",
        skip_reason="dd_unit_unresolvable",
    )

    first = revive_unit_skipped_sources(gc=_gc)
    assert first["revived"] >= 1
    assert _source(_gc, sns_id)["status"] == "extracted"

    second = revive_unit_skipped_sources(gc=_gc)
    assert second["revived"] == 0


# ---------------------------------------------------------------------------
# Part B — parked vocab_gap sources retry once per vocabulary change
# ---------------------------------------------------------------------------


def test_vocabulary_signature_is_derived_from_isn():
    """The signature is a stable digest of the live ISN vocabulary."""
    from imas_codex.standard_names.graph_ops import isn_vocabulary_signature

    sig = isn_vocabulary_signature()
    assert sig is None or (isinstance(sig, str) and sig == isn_vocabulary_signature())


def test_vocabulary_signature_tracks_token_changes():
    """Adding a token to the vocabulary changes the signature."""
    from imas_codex.standard_names import graph_ops

    base_ctx = {
        "vocabulary_sections": [{"segment": "physical_base", "tokens": ["pressure"]}],
        "grammar": {"vocabularies": {"operators": {"normalized": {}}}},
    }
    bumped_ctx = {
        "vocabulary_sections": [
            {
                "segment": "physical_base",
                "tokens": ["pressure", "confinement_enhancement_factor"],
            }
        ],
        "grammar": {"vocabularies": {"operators": {"normalized": {}}}},
    }

    with patch.object(graph_ops, "_isn_grammar_context", return_value=base_ctx):
        graph_ops.isn_vocabulary_signature.cache_clear()
        before = graph_ops.isn_vocabulary_signature()
    with patch.object(graph_ops, "_isn_grammar_context", return_value=bumped_ctx):
        graph_ops.isn_vocabulary_signature.cache_clear()
        after = graph_ops.isn_vocabulary_signature()
    graph_ops.isn_vocabulary_signature.cache_clear()

    assert before and after and before != after


def test_retry_vocab_gap_sources_no_ops_without_isn():
    """Without a resolvable ISN vocabulary the reconcile is a no-op."""
    from imas_codex.standard_names import graph_ops

    mock_gc = MagicMock()
    with patch.object(graph_ops, "isn_vocabulary_signature", return_value=None):
        result = graph_ops.retry_vocab_gap_sources_on_grammar_change(gc=mock_gc)

    assert result == {"checked": 0, "revived": 0, "skipped": True}
    mock_gc.query.assert_not_called()


@pytest.mark.graph
def test_parked_source_revives_once_on_vocabulary_change(_gc, _clean):
    """A source parked under an older vocabulary retries, then stays put."""
    from imas_codex.standard_names.graph_ops import (
        isn_vocabulary_signature,
        retry_vocab_gap_sources_on_grammar_change,
    )

    sns_id = _park_vocab_gap_source(
        _gc, path=_uid("parked"), signature="stale_signature", attempt_count=5
    )

    first = retry_vocab_gap_sources_on_grammar_change(gc=_gc)

    props = _source(_gc, sns_id)
    assert props["status"] == "extracted"
    assert props["vocab_gap_grammar_signature"] == isn_vocabulary_signature()
    assert props["attempt_count"] == 0
    assert props.get("claimed_at") is None
    assert first["revived"] >= 1
    # The blocking gap edges are dropped so the next attempt rebuilds them.
    remaining = _gc.query(
        "MATCH (s:StandardNameSource {id: $id})-[r:HAS_STANDARD_NAME_VOCAB_GAP]->() "
        "RETURN count(r) AS c",
        id=sns_id,
    )
    assert remaining[0]["c"] == 0


@pytest.mark.graph
def test_unstamped_parked_source_revives_once(_gc, _clean):
    """A source parked before signatures existed counts as a vocabulary change."""
    from imas_codex.standard_names.graph_ops import (
        retry_vocab_gap_sources_on_grammar_change,
    )

    sns_id = _park_vocab_gap_source(_gc, path=_uid("unstamped"), signature=None)

    retry_vocab_gap_sources_on_grammar_change(gc=_gc)

    assert _source(_gc, sns_id)["status"] == "extracted"


@pytest.mark.graph
def test_vocab_gap_retry_is_idempotent_at_a_fixed_vocabulary(_gc, _clean):
    """A second pass at the same vocabulary revives nothing."""
    from imas_codex.standard_names.graph_ops import (
        retry_vocab_gap_sources_on_grammar_change,
    )

    _park_vocab_gap_source(_gc, path=_uid("parked"), signature=None)

    first = retry_vocab_gap_sources_on_grammar_change(gc=_gc)
    assert first["revived"] >= 1
    second = retry_vocab_gap_sources_on_grammar_change(gc=_gc)
    assert second["revived"] == 0


@pytest.mark.graph
def test_source_parked_at_the_current_vocabulary_stays_parked(_gc, _clean):
    """A source parked under the vocabulary in force is not retried."""
    from imas_codex.standard_names.graph_ops import (
        isn_vocabulary_signature,
        retry_vocab_gap_sources_on_grammar_change,
    )

    sig = isn_vocabulary_signature()
    if sig is None:  # pragma: no cover - ISN missing
        pytest.skip("ISN vocabulary unavailable")
    sns_id = _park_vocab_gap_source(_gc, path=_uid("current"), signature=sig)

    retry_vocab_gap_sources_on_grammar_change(gc=_gc)

    assert _source(_gc, sns_id)["status"] == "vocab_gap"


# ---------------------------------------------------------------------------
# Part C — loop wiring
# ---------------------------------------------------------------------------


def _startup_patches(revive_mock: MagicMock, retry_mock: MagicMock) -> list:
    """Patch every graph-backed startup call so run_sn_pools stays graph-free."""
    mock_gc_ctx = MagicMock()
    mock_gc_inst = MagicMock()
    mock_gc_inst.query.return_value = [{"cnt": 1}]
    mock_gc_ctx.__enter__ = MagicMock(return_value=mock_gc_inst)
    mock_gc_ctx.__exit__ = MagicMock(return_value=False)

    return [
        patch(f"{_GO}.reconcile_standard_name_sources", return_value={}),
        patch(f"{_GO}.reconcile_vocab_gaps", return_value={}),
        patch(f"{_GO}.revive_unit_skipped_sources", new=revive_mock),
        patch(f"{_GO}.retry_vocab_gap_sources_on_grammar_change", new=retry_mock),
        patch(f"{_GO}.reconcile_provenance", return_value={}),
        patch(f"{_GO}.reconcile_source_status_liveness", return_value={}),
        patch(f"{_GO}.retire_unreachable_hint_edits", return_value=0),
        patch(f"{_GO}.reconcile_grammar_segments", return_value={}),
        patch(f"{_GO}.reconcile_standard_name_cocos_links", return_value={}),
        patch(
            f"{_GO}.reconcile_standard_name_unit_edges",
            return_value={"names_realigned": 0, "edges_dropped": 0, "edges_created": 0},
        ),
        patch(
            f"{_AA}.reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        ),
        # The DD-unit correction reconcile opens its own GraphClient; stub it.
        patch(
            f"{_DGO}.reconcile_dd_unit_corrections",
            return_value={"checked": 0, "corrected": 0},
        ),
        patch(
            f"{_GO}.reconcile_standard_name_dd_edges",
            return_value={"edges_created": 0, "pairs_dropped": 0},
        ),
        patch(
            f"{_GO}.reconcile_standard_name_source_paths",
            return_value={"names_reconciled": 0},
        ),
        patch(
            f"{_GO}.reconcile_reviewable_name_stage",
            return_value={"names_advanced": 0},
        ),
        patch(f"{_GO}.create_sn_run_open"),
        patch(f"{_GO}.finalize_sn_run"),
        patch(f"{_GO}.release_all_orphan_claims", return_value={"sn": 0, "sns": 0}),
        patch(f"{_GO}.rederive_structural_edges", return_value={}),
        patch(f"{_GO}.seed_parent_sources", return_value=0),
        patch(f"{_GO}.normalize_derived_parent_lifecycle", return_value=0),
        patch(f"{_GO}.structural_accept_derived_parents", return_value=0),
        patch(f"{_GO}.resolve_doc_links", return_value={}),
        patch(f"{_GO}.promote_stranded_reviewed", return_value={"name": 0, "docs": 0}),
        patch(f"{_GO}.mark_orphaned_standard_name_runs_stale", return_value=0),
        patch(
            "imas_codex.standard_names.source_refresh.refresh_drifted_sources",
            return_value={},
        ),
        patch(f"{_LOOP}._seed_all_domains", new=AsyncMock(return_value=0)),
        patch(
            "imas_codex.standard_names.pools.run_pools",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "imas_codex.standard_names.budget.BudgetManager.start",
            new_callable=AsyncMock,
        ),
        patch(
            "imas_codex.standard_names.budget.BudgetManager.drain_pending",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "imas_codex.standard_names.budget.BudgetManager.get_total_spent",
            new_callable=AsyncMock,
            return_value=0.0,
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc_ctx),
    ]


@pytest.mark.asyncio
async def test_run_sn_pools_runs_both_revival_reconciles() -> None:
    """Both revivals run in the startup reconcile every ``sn run`` executes."""
    from imas_codex.standard_names.loop import run_sn_pools

    revive_mock = MagicMock(return_value={"checked": 3, "revived": 2})
    retry_mock = MagicMock(return_value={"checked": 5, "revived": 4})

    patches = _startup_patches(revive_mock, retry_mock)
    for p in patches:
        p.start()
    try:
        stop = asyncio.Event()
        stop.set()
        await run_sn_pools(cost_limit=5.0, domains=(), stop_event=stop)
    finally:
        for p in patches:
            p.stop()

    revive_mock.assert_called_once()
    retry_mock.assert_called_once()
