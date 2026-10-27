"""Cascade-atomicity tests for edit-rename acceptance.

Locked decision: every descendant id produced by a rename cascade is
round-trip-validated and uniqueness-checked BEFORE the cascade commits; any
failure refuses the acceptance itself — the root is not accepted and no
descendant is renamed (full rollback, nothing persisted).  The auto-commit
graph primitive cannot span the acceptance write and the rename write in a
single Neo4j transaction, so atomicity is realised by gating the acceptance
on a clean cascade preflight (see
:func:`imas_codex.standard_names.graph_ops.persist_reviewed_name`).

The report-vs-persistence contract closes the gap that gating leaves open:
a cascade that is NOT applied must not read as performed.  ``CascadeResult.
dry_run`` is ``True`` on every non-apply return path and ``False`` only where
the descendant renames were actually persisted, so the two halves of the
contract are (a) an accepted root persists exactly the descendants it
reports, and (b) a root that has not reached ``accepted`` reports its
descendants as awaiting acceptance and changes no id.  The plain-language
deferral wording the surface prints from ``dry_run`` is pinned in
``test_rename_cascade.py``'s ``TestDeferredCascadeIsReportedAsDeferred``.

Runs against the in-memory ``FakeGraph`` from ``test_edit_engine`` — no live
Neo4j.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imas_codex.standard_names.cascade import cascade_descendants_of, rename_cascade
from imas_codex.standard_names.graph_store.graph_review_persistence import (
    persist_reviewed_name,
)
from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph


def _drafted_subtree_root(fake: FakeGraph, *, include_accepted: bool = True) -> None:
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
            resolution_method="quorum_consensus",
            reviewer_chain_size=3,
        )


class TestCascadeAtomicity:
    def test_locus_children_remain_untouched_when_root_accepts(self) -> None:
        fake = FakeGraph()
        fake.add_node("radial_coordinate", name_stage="superseded")
        fake.add_node(
            "radial_outline",
            name_stage="drafted",
            edit_status="open",
            edit_scope="subtree",
            edit_mode="rename",
            edit_include_accepted=True,
            claim_token="tok",
        )
        fake.refined_from["radial_outline"] = "radial_coordinate"
        child = "radial_coordinate_of_control_surface"
        fake.add_node(child, name_stage="accepted")
        fake.add_edge(child, "radial_outline", "control_surface", "locus")

        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="radial_outline",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert stage == "accepted"
        assert child in fake.nodes
        assert "radial_outline_of_control_surface" not in fake.nodes

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


class TestReportMatchesPersistence:
    """Two halves of the cascade report-vs-persistence contract.

    ``CascadeResult.dry_run`` is the discriminator: ``True`` on every return
    that did not write, ``False`` only where the renames in ``renamed`` were
    actually persisted.  A descendant rename may therefore be reported as
    performed only when the root has reached ``accepted``; on any other root
    the same rows are reported as awaiting acceptance and change no id.  This
    is the atomicity proof the acceptance gate itself cannot provide: the
    gate guarantees nothing was written on refusal, and these tests guarantee
    the report says so.
    """

    @staticmethod
    def _exhausted_root_with_children(fake: FakeGraph) -> None:
        """The measured shape: ``emissivity_due_to_fusion`` exhausted at 0.825
        left its two qualifier descendants reported as planned and unchanged.
        The root has NOT reached ``accepted`` — the descendants' renames are
        deferred work under it."""
        fake.add_node("emissivity_due_to_fusion", name_stage="exhausted")
        fake.add_node(
            "deuterium_deuterium_emissivity_due_to_fusion", name_stage="reviewed"
        )
        fake.add_node(
            "deuterium_tritium_emissivity_due_to_fusion", name_stage="reviewed"
        )
        fake.add_edge(
            "deuterium_deuterium_emissivity_due_to_fusion",
            "emissivity_due_to_fusion",
            "deuterium_deuterium",
            "qualifier",
        )
        fake.add_edge(
            "deuterium_tritium_emissivity_due_to_fusion",
            "emissivity_due_to_fusion",
            "deuterium_tritium",
            "qualifier",
        )

    @staticmethod
    def _accepted_successor_with_live_children(fake: FakeGraph) -> None:
        """A subtree rename whose successor HAS reached ``accepted`` and
        already inherited the predecessor's live qualifier children."""
        fake.add_node("temperature", name_stage="superseded")
        fake.add_node(
            "density",
            name_stage="accepted",
            edit_status="applied",
            edit_scope="subtree",
            edit_mode="rename",
            edit_include_accepted=True,
            claim_token="tok",
        )
        fake.refined_from["density"] = "temperature"
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_edge("electron_temperature", "density", "electron", "qualifier")
        fake.add_edge("ion_temperature", "density", "ion", "qualifier")

    def test_unaccepted_root_defers_descendants_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """An exhausted (unaccepted) root reports its descendant renames as
        deferred — never as performed — and changes no id."""
        fake = FakeGraph()
        self._exhausted_root_with_children(fake)
        result = rename_cascade(
            fake,
            "emissivity_due_to_fusion",
            "source_rate",
            dry_run=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert result.conflicts == []
        deferred = {r["from"]: r["to"] for r in result.renamed}
        # The two measured descendants are carried as the deferred plan —
        # the surface can say they await the root's acceptance — and nothing
        # in the graph changed: no id is a successor's, all are the originals.
        assert deferred["deuterium_deuterium_emissivity_due_to_fusion"] == (
            "deuterium_deuterium_source_rate"
        )
        assert deferred["deuterium_tritium_emissivity_due_to_fusion"] == (
            "deuterium_tritium_source_rate"
        )
        assert "deuterium_deuterium_source_rate" not in fake.nodes
        assert "deuterium_tritium_source_rate" not in fake.nodes
        assert "deuterium_deuterium_emissivity_due_to_fusion" in fake.nodes
        assert "deuterium_tritium_emissivity_due_to_fusion" in fake.nodes

    def test_accepted_root_persists_exactly_what_it_reports(
        self, tmp_path: Path
    ) -> None:
        """Once the root reaches ``accepted``, the same apply call persists
        the descendants and reports ``dry_run=False`` — the honest deferral
        could not have quieted the real writes."""
        fake = FakeGraph()
        self._accepted_successor_with_live_children(fake)
        result = cascade_descendants_of(
            fake,
            successor_id="density",
            old_root="temperature",
            new_root="density",
            dry_run=False,
            include_accepted=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is False
        assert result.conflicts == []
        assert {r["from"]: r["to"] for r in result.renamed} == {
            "electron_temperature": "electron_density",
            "ion_temperature": "ion_density",
        }
        # The renames the report carries are the renames the graph holds.
        assert "electron_density" in fake.nodes
        assert "ion_density" in fake.nodes
        assert "electron_temperature" not in fake.nodes
        assert "ion_temperature" not in fake.nodes

    def test_deferral_message_names_what_it_waits_on(self) -> None:
        """The rendered plan uses the deferral vocabulary for the same two
        descendant rows the contract defers, and never calls them done."""
        from rich.console import Console

        from imas_codex.cli import sn as sn_cli
        from imas_codex.standard_names.edit import EditPlan

        plan = EditPlan(
            target="emissivity_due_to_fusion",
            mode="rename",
            axis="name",
            scope="subtree",
            entry="review_name",
            successor="source_rate",
            cascade_deferred=[
                {
                    "from": "deuterium_deuterium_emissivity_due_to_fusion",
                    "to": "deuterium_deuterium_source_rate",
                },
                {
                    "from": "deuterium_tritium_emissivity_due_to_fusion",
                    "to": "deuterium_tritium_source_rate",
                },
            ],
            applied=True,
            run_id="sn-edit-20260910T000000Z",
        )
        recorder = Console(record=True, width=200)
        original = sn_cli.console
        sn_cli.console = recorder
        try:
            sn_cli._render_edit_plan(plan, followup_hint=False)
        finally:
            sn_cli.console = original
        output = recorder.export_text()
        assert "deferred" in output
        assert "not yet applied" in output
        assert "awaiting source_rate reaching" in output
        assert "accepted" in output
        assert "deuterium_tritium_source_rate" in output
        # The two descendant rows must not read as performed work.
        assert "renamed 2 descendant" not in output
        assert "Cascade (planned renames)" not in output


class TestLinearChainPlanConstruction:
    """A strictly linear, non-branching three-level chain plans through its
    whole depth and never reports a descendant as ``unreachable``.

    The historical bug: a middle node leaving the plan at a semantic boundary
    (or under a safety refusal) made the builder classify its own child as an
    unresolvable topology fault — ``no parent in plan`` — which refused the
    whole edit.  Propagation stopping above a child is a skip, not a fault;
    a child whose parent is genuinely absent from the plan is the fault.  The
    fully provable linear chain belongs in the plan outright.
    """

    def test_linear_three_level_chain_plans_without_unreachable(
        self, tmp_path: Path
    ) -> None:
        """``ion_temperature`` ← ``temperature`` ← ``core_ion_temperature``
        (read upstream) — every edge carries a provable qualifier, nothing
        branches, and no level is unreachable."""
        fake = FakeGraph()
        for name in ("temperature", "ion_temperature", "core_ion_temperature"):
            fake.add_node(name)
        fake.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )
        fake.add_edge(
            "core_ion_temperature",
            "ion_temperature",
            operator="core",
            operator_kind="qualifier",
        )
        new_root = "temperature_of_plasma_boundary"
        result = rename_cascade(
            fake,
            "temperature",
            new_root,
            dry_run=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.conflicts == []
        assert not any(
            "unreachable in cascade" in conflict for conflict in result.conflicts
        )
        plan = {r["from"]: r["to"] for r in result.renamed}
        assert plan == {
            "temperature": new_root,
            "ion_temperature": f"ion_{new_root}",
            "core_ion_temperature": f"core_ion_{new_root}",
        }

    def test_subtree_stopped_at_a_boundary_is_skipped_not_a_fault(
        self, tmp_path: Path
    ) -> None:
        """When a middle node leaves the plan at a semantic boundary, its own
        child is not ``unreachable`` — propagation simply stopped above it."""
        fake = FakeGraph()
        for name in (
            "temperature",
            "temperature_of_plasma",
            "ion_temperature_of_plasma",
        ):
            fake.add_node(name)
        fake.add_edge(
            "temperature_of_plasma",
            "temperature",
            operator="of",
            operator_kind="locus",
        )
        fake.add_edge(
            "ion_temperature_of_plasma",
            "temperature_of_plasma",
            operator="ion",
            operator_kind="qualifier",
        )
        result = rename_cascade(
            fake,
            "temperature",
            "electron_temperature",
            dry_run=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.conflicts == []
        stopped = {row["name"]: row["reason"] for row in result.skipped}
        assert set(stopped) == {
            "temperature_of_plasma",
            "ion_temperature_of_plasma",
        }
        assert "semantic boundary" in stopped["temperature_of_plasma"]
        assert (
            "propagation stopped at ancestor 'temperature_of_plasma'"
            in stopped["ion_temperature_of_plasma"]
        )
        assert [r["from"] for r in result.renamed] == ["temperature"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
