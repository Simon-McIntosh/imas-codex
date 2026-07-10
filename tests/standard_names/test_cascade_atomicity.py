"""Cascade-atomicity tests for edit-rename acceptance.

Locked decision: every descendant id produced by a rename cascade is
round-trip-validated and uniqueness-checked BEFORE the cascade commits; any
failure refuses the acceptance itself — the root is not accepted and no
descendant is renamed (full rollback, nothing persisted).  The auto-commit
graph primitive cannot span the acceptance write and the rename write in a
single Neo4j transaction, so atomicity is realised by gating the acceptance
on a clean cascade preflight (see
:func:`imas_codex.standard_names.graph_ops.persist_reviewed_name`).

Runs against the in-memory ``FakeGraph`` from ``test_edit_engine`` — no live
Neo4j.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.graph_ops import persist_reviewed_name
from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph


def _drafted_subtree_root(
    fake: FakeGraph, *, include_accepted: bool = True
) -> None:
    """Graph state right after a subtree rename's successor was created:
    ``temperature`` superseded, ``density`` drafted+open, two accepted
    children pointing at the new (drafted) root.
    """
    fake.add_node("temperature", name_stage="superseded")
    fake.add_node(
        "density",
        name_stage="drafted",
        edit_status="open",
        edit_scope="subtree",
        edit_mode="rename",
        edit_include_accepted=include_accepted,
        claim_token="tok",
    )
    fake.refined_from["density"] = "temperature"
    fake.add_node("electron_temperature", name_stage="accepted")
    fake.add_node("ion_temperature", name_stage="accepted")
    fake.add_edge("electron_temperature", "density", "electron", "qualifier")
    fake.add_edge("ion_temperature", "density", "ion", "qualifier")


def _accept(fake: FakeGraph) -> str:
    with _patched_graph(fake):
        return persist_reviewed_name(
            sn_id="density",
            claim_token="tok",
            score=0.95,
            model="reviewer/x",
            min_score=0.75,
            rotation_cap=3,
        )


class TestCascadeAtomicity:
    def test_clean_cascade_applies_on_acceptance(self) -> None:
        """Baseline: a conflict-free cascade accepts the root and renames
        every descendant atomically."""
        fake = FakeGraph()
        _drafted_subtree_root(fake)
        stage = _accept(fake)
        assert stage == "accepted"
        assert fake.nodes["density"]["edit_status"] == "applied"
        assert "electron_density" in fake.nodes
        assert "ion_density" in fake.nodes
        assert "electron_temperature" not in fake.nodes
        assert "ion_temperature" not in fake.nodes

    def test_collision_refuses_acceptance_nothing_persisted(self) -> None:
        """A pre-existing id collision on a would-be descendant refuses the
        whole acceptance: the root is NOT accepted and NO descendant id
        changed — full rollback, nothing persisted."""
        fake = FakeGraph()
        _drafted_subtree_root(fake)
        # `electron_density` already exists → the cascade collides.
        fake.add_node("electron_density", name_stage="accepted")
        stage = _accept(fake)
        assert stage == "reviewed"
        # Root not accepted; edit stays open (rides refine).
        assert fake.nodes["density"]["name_stage"] == "reviewed"
        assert fake.nodes["density"]["edit_status"] == "open"
        # No descendant renamed — the original children are untouched.
        assert "electron_temperature" in fake.nodes
        assert "ion_temperature" in fake.nodes
        assert "ion_density" not in fake.nodes
        # The pre-existing collider is untouched too.
        assert fake.nodes["electron_density"]["name_stage"] == "accepted"
        # The refusal reason is recorded for operator follow-up.
        issues = fake.nodes["density"].get("validation_issues") or []
        assert any("edit_cascade" in i for i in issues)

    def test_protected_descendant_without_optin_refuses(self) -> None:
        """An accepted descendant with no ``include_accepted`` opt-in makes
        the preflight conflict, so acceptance is refused atomically rather
        than stranding the accepted child with stale grammar."""
        fake = FakeGraph()
        _drafted_subtree_root(fake, include_accepted=False)
        stage = _accept(fake)
        assert stage == "reviewed"
        assert fake.nodes["density"]["edit_status"] == "open"
        # Nothing renamed.
        assert "electron_temperature" in fake.nodes
        assert "electron_density" not in fake.nodes

    def test_partial_collision_rolls_back_all_siblings(self) -> None:
        """A collision on ONE descendant must roll back the WHOLE cascade —
        the non-colliding sibling is not renamed either (all-or-nothing)."""
        fake = FakeGraph()
        _drafted_subtree_root(fake)
        # Only one of the two targets collides.
        fake.add_node("ion_density", name_stage="accepted")
        stage = _accept(fake)
        assert stage == "reviewed"
        # Neither sibling renamed — the clean one is held back with the
        # colliding one.
        assert "electron_temperature" in fake.nodes
        assert "electron_density" not in fake.nodes
        assert "ion_temperature" in fake.nodes


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
