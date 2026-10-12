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
from imas_codex.standard_names.graph_ops import supersede_prior_source_names
from imas_codex.standard_names.provenance_lifecycle import SourceBindingConflictError

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


class _Tx:
    """A transaction handle returning the fixture rows for each query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, cypher: str, **params: object) -> list[dict[str, object]]:
        self.queries.append(cypher)
        if "GENERATED_SUPERSESSION_PREFLIGHT" in cypher:
            return [_preflight_row()]
        if "manifest_recorded" in cypher:
            return [_dual_bound_migration_row()]
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
