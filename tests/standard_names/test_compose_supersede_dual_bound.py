"""Compose supersede over a source bound to two live identities.

A compose pass that recomposes a name for a DD source writes the successor's
``PRODUCED_NAME`` binding and then runs ``supersede_prior_source_names`` to
retire the predecessor, restoring the invariant *one source -> at most one live
pipeline name*. A source that is already bound to two live identities — the
successor this pass just wrote plus the predecessor it is meant to retire — is
not the ordinary single-predecessor case.

The fixture graph holds one DD source whose only binding evidence is two live
``PRODUCED_NAME`` edges, predecessor and successor. Driving compose supersede
over it shows how the migration's compare-and-set classifies that state.

The migration's own compare-and-set has a named, adjudicable refusal
(``SourceBindingConflictError``) for a source held by a live identity outside
the migration's names, and raises an anonymous ``RuntimeError`` for every other
ineligible state. The two are treated differently by callers: a named conflict
withholds a retry budget, an anonymous fault spends one. This fixture pins down
which branch a dual-bound source takes.

The header records the observed behaviour at the base revision; the module is
committed as an ordinary test requiring compose supersede to route a dual-bound
source through the adjudicable conflict rather than an anonymous fault.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentPairingGuardResult
from imas_codex.standard_names.graph_store.graph_supersession import (
    supersede_prior_source_names,
)
from imas_codex.standard_names.provenance_lifecycle import (
    SourceBindingConflictError,
    SourceRetiredSuccessorConflictError,
)
from imas_codex.standard_names.worker_tasks import (
    worker_common as _owning_worker_common,
)

_DD_PATH = "dd:example/predicted_path"
_SOURCE_NODE = "dd:example/predicted_path"
_OLD_NAME = "example_old_name"
_NEW_NAME = "example_new_name"


def _preflight_row() -> dict[str, object]:
    """The preflight row naming the recomposed source and its predecessor."""
    return {
        "requested_source_id": _DD_PATH,
        "new_name": _NEW_NAME,
        "requested_source_exists": True,
        "successor_exists": True,
        "trigger_source_id": _SOURCE_NODE,
        "old_name": _OLD_NAME,
        "old_stage": "drafted",
        "judged_source_ids": [_SOURCE_NODE],
        "retained_source_ids": [],
    }


def _dual_bound_migration_row() -> dict[str, object]:
    """One source whose two live bindings are predecessor and successor."""
    bindings = [
        {"id": _OLD_NAME, "name_stage": "drafted"},
        {"id": _NEW_NAME, "name_stage": "drafted"},
    ]
    return {
        "source_id": _SOURCE_NODE,
        "source_exists": True,
        "source_status": "composed",
        "scalar_binding": _NEW_NAME,
        "actively_claimed": False,
        "current_bindings": [entry["id"] for entry in bindings],
        "binding_state": bindings,
        "manifest_recorded": False,
    }


def _retired_migration_row() -> dict[str, object]:
    """One source bound to the live predecessor and a retired successor."""
    bindings = [
        {"id": _OLD_NAME, "name_stage": "drafted"},
        {"id": _NEW_NAME, "name_stage": "superseded"},
    ]
    return {
        "source_id": _SOURCE_NODE,
        "source_exists": True,
        "source_status": "composed",
        "scalar_binding": _OLD_NAME,
        "actively_claimed": False,
        "current_bindings": [entry["id"] for entry in bindings],
        "binding_state": bindings,
        "manifest_recorded": False,
    }


class _Tx:
    """A transaction handle returning the fixture rows for each query."""

    def __init__(self, migration_row: dict[str, object] | None = None) -> None:
        self.queries: list[str] = []
        self._migration_row = migration_row or _dual_bound_migration_row()

    def run(self, cypher: str, **params: object) -> list[dict[str, object]]:
        self.queries.append(cypher)
        if "GENERATED_SUPERSESSION_PREFLIGHT" in cypher:
            return [_preflight_row()]
        if "manifest_recorded" in cypher:
            return [self._migration_row]
        raise AssertionError(
            f"compose supersede reached an unexpected query: {cypher[:120]!r}"
        )

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Session:
    def __init__(self, tx: _Tx) -> None:
        self._tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def begin_transaction(self) -> _Tx:
        return self._tx


class _Client:
    def __init__(self, tx: _Tx) -> None:
        self._tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def session(self) -> _Session:
        return _Session(self._tx)


def test_compose_supersede_over_dual_bound_source_raises_adjudicable_conflict() -> None:
    """A dual-bound source must raise the named conflict, not a transient fault.

    The migration's two refusals are not equivalent: a named
    ``SourceBindingConflictError`` withholds a retry budget because repeating the
    migration changes nothing, while an anonymous ``RuntimeError`` is treated as
    a transient fault and spends one. A source bound to two live identities must
    take the adjudicable branch, naming the source and both bound identities, so
    the hard failure reaches an operator instead of a retry loop.
    """
    tx = _Tx()
    with patch(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings",
        return_value=AttachmentPairingGuardResult((_SOURCE_NODE,), ()),
    ):
        with pytest.raises(SourceBindingConflictError) as raised:
            supersede_prior_source_names(
                [{"new_name": _NEW_NAME, "source_id": _DD_PATH}],
                gc=_Client(tx),
            )

    message = str(raised.value)
    assert _SOURCE_NODE in message
    assert _OLD_NAME in message
    assert _NEW_NAME in message


def test_compose_persist_withholds_a_dual_bound_source_without_spending_an_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A compose pass over a dual-bound source spends no attempt.

    The persist boundary is where a compose failure for a source is recorded.
    A named ``SourceBindingConflictError`` means the source is already held by
    a live identity, so releasing the claim to retry would spend an attempt on
    a decision only an operator can make. The boundary must instead record the
    conflict and leave the source claimed and withheld, returning no winner id
    and releasing no claim.
    """
    from imas_codex.standard_names import workers
    from imas_codex.standard_names.provenance_lifecycle import SourceBindingConflict

    conflict = SourceBindingConflictError(
        [
            SourceBindingConflict(
                source_id=_SOURCE_NODE,
                holding_identity=_NEW_NAME,
                holding_stage="drafted",
                source_status="composed",
                detail="dual bound",
            )
        ]
    )
    candidates = [
        {
            "source_id": _DD_PATH,
            "description": "a composed description for the source",
            "source_claim_token": "claim-token-1",
        }
    ]
    releases: list[dict[str, object]] = []

    def _release(**kwargs: object) -> int:
        releases.append(kwargs)
        return 0

    with (
        caplog.at_level("WARNING"),
        patch(
            "imas_codex.standard_names.graph_store.graph_catalog.persist_generated_name_batch",
            side_effect=conflict,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_claim_release.release_generate_name_failed_claims",
            side_effect=_release,
        ),
    ):
        result = _owning_worker_common._persist_description_checked_candidates(
            candidates,
            source_type="dd",
            phase="generate_name",
            compose_model="test-model",
            dd_version="4.0.0",
            cocos_version=1,
            run_id=None,
        )

    assert result == []
    assert releases == []
    assert any(
        _SOURCE_NODE in record.getMessage() and _NEW_NAME in record.getMessage()
        for record in caplog.records
    )


def test_compose_persist_withholds_a_retired_successor_without_spending_an_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A source bound to the live predecessor and a retired successor is withheld.

    The outcome is produced by compose supersede itself, so this test is
    sensitive to the classification: with the row reverted to the anonymous
    transient fault the first ``pytest.raises`` never fires. A named outcome
    means the source is already held, so the persist boundary must record it and
    leave the source claimed — returning no winner id and releasing no claim.
    """
    from imas_codex.standard_names import workers

    tx = _Tx(_retired_migration_row())
    with patch(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings",
        return_value=AttachmentPairingGuardResult((_SOURCE_NODE,), ()),
    ):
        with pytest.raises(SourceRetiredSuccessorConflictError) as raised:
            supersede_prior_source_names(
                [{"new_name": _NEW_NAME, "source_id": _DD_PATH}],
                gc=_Client(tx),
            )
    conflict = raised.value
    (outcome,) = conflict.conflicts
    assert outcome.source_id == _SOURCE_NODE
    assert (outcome.holding_identity, outcome.retired_identity) == (
        _OLD_NAME,
        _NEW_NAME,
    )

    candidates = [
        {
            "source_id": _DD_PATH,
            "description": "a composed description for the source",
            "source_claim_token": "claim-token-1",
        }
    ]
    releases: list[dict[str, object]] = []

    def _release(**kwargs: object) -> int:
        releases.append(kwargs)
        return 0

    with (
        caplog.at_level("WARNING"),
        patch(
            "imas_codex.standard_names.graph_store.graph_catalog.persist_generated_name_batch",
            side_effect=conflict,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_claim_release.release_generate_name_failed_claims",
            side_effect=_release,
        ),
    ):
        result = _owning_worker_common._persist_description_checked_candidates(
            candidates,
            source_type="dd",
            phase="generate_name",
            compose_model="test-model",
            dd_version="4.0.0",
            cocos_version=1,
            run_id=None,
        )

    assert result == []
    assert releases == []
    assert any(
        _SOURCE_NODE in record.getMessage()
        and _OLD_NAME in record.getMessage()
        and _NEW_NAME in record.getMessage()
        for record in caplog.records
    )
