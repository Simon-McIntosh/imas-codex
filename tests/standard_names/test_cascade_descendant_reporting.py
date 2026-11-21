"""Report-vs-persistence contract for descendant rename cascades.

A subtree or family rename must report a descendant rename as *performed*
only when the cascade actually persists it.  When the rename root has not
reached ``accepted`` the descendants are deferred work: the report carries
them as a plan that awaits the root's acceptance, and they are never
written.  When the root does reach ``accepted`` the report carries the
persisted renames and they are written exactly as the result describes.

The discriminating field is ``CascadeResult.dry_run``: ``True`` on every
non-apply return path (early refusal, conflict, mismatch, deferred plan) and
``False`` only where the renames were actually persisted.  A caller can
therefore tell planned-and-performed (False) from planned-and-deferred
(True) without reading the graph.  The measured failure this guards: a root
exhausted below the acceptance threshold left its descendants reported as
planned work that was never performed.

The accepted-root assertions pin the persistence behaviour so the
honest-report contract cannot be fixed by silently stopping the writes.

Runs against the in-memory ``FakeGraph`` from ``test_edit_engine`` — no live
Neo4j.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iter_codex.standard_names.cascade import cascade_descendants_of, rename_cascade
from tests.standard_names.test_edit_engine import FakeGraph


def _exhausted_root_with_children(fake: FakeGraph) -> None:
    """The measured shape: an exhausted (unaccepted) root whose two
    qualifier descendants derive from it."""
    fake.add_node("emissivity_due_to_fusion", name_stage="exhausted")
    fake.add_node("deuterium_deuterium_emissivity_due_to_fusion", name_stage="reviewed")
    fake.add_node("deuterium_tritium_emissivity_due_to_fusion", name_stage="reviewed")
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


def _drafted_successor_no_live_children(fake: FakeGraph) -> tuple[str, str, str]:
    """A subtree rename whose successor is still drafted (root has NOT
    reached accepted) and has not inherited live children yet."""
    fake.add_node("temperature", name_stage="superseded")
    fake.add_node(
        "density",
        name_stage="drafted",
        edit_status="open",
        edit_scope="subtree",
        edit_mode="rename",
        edit_include_accepted=True,
        claim_token="tok",
    )
    fake.refined_from["density"] = "temperature"
    return "density", "temperature", "density"


def _accepted_successor_with_live_children(fake: FakeGraph) -> tuple[str, str, str]:
    """A subtree rename whose successor HAS reached accepted and already
    inherited the predecessor's live qualifier children."""
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
    return "density", "temperature", "density"


class TestUnacceptedRootDefersDescendants:
    def test_planning_reports_deferral_and_writes_nothing(self, tmp_path: Path) -> None:
        """The subtree-planning surface (dry-run) reports the descendants
        as deferred, never as performed, and leaves them untouched."""
        fake = FakeGraph()
        _exhausted_root_with_children(fake)
        result = rename_cascade(
            fake,
            "emissivity_due_to_fusion",
            "source_rate",
            dry_run=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert result.conflicts == []
        # ``renamed`` carries the root plus every deferred descendant (root
        # included is the documented contract); the descendants themselves
        # must be present as planned, never as performed.
        deferred = {r["from"]: r["to"] for r in result.renamed}
        assert deferred["emissivity_due_to_fusion"] == "source_rate"
        assert deferred["deuterium_deuterium_emissivity_due_to_fusion"] == (
            "deuterium_deuterium_source_rate"
        )
        assert deferred["deuterium_tritium_emissivity_due_to_fusion"] == (
            "deuterium_tritium_source_rate"
        )
        assert set(deferred) == {
            "emissivity_due_to_fusion",
            "deuterium_deuterium_emissivity_due_to_fusion",
            "deuterium_tritium_emissivity_due_to_fusion",
        }
        # Nothing was persisted — the ids are unchanged after planning.
        assert "deuterium_deuterium_emissivity_due_to_fusion" in fake.nodes
        assert "deuterium_deuterium_source_rate" not in fake.nodes
        assert "deuterium_tritium_source_rate" not in fake.nodes

    def test_apply_request_on_unaccepted_root_reports_deferred(
        self, tmp_path: Path
    ) -> None:
        """A post-acceptance apply call against a root that has NOT reached
        accepted refuses to write and reports ``dry_run=True`` — the report
        cannot claim a persistence the refusal prevented."""
        fake = FakeGraph()
        successor, old_root, new_root = _drafted_successor_no_live_children(fake)
        result = cascade_descendants_of(
            fake,
            successor_id=successor,
            old_root=old_root,
            new_root=new_root,
            dry_run=False,
            include_accepted=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert any("is not accepted" in c for c in result.conflicts)
        assert result.renamed == []

    def test_unaccepted_root_live_children_reported_deferred(
        self, tmp_path: Path
    ) -> None:
        """When an unaccepted root already carries live descendants, the
        dry-run preflight reports them as deferred and writes nothing."""
        fake = FakeGraph()
        _accepted_successor_with_live_children(fake)
        fake.nodes["density"]["name_stage"] = "drafted"
        fake.nodes["density"]["edit_status"] = "open"
        result = cascade_descendants_of(
            fake,
            successor_id="density",
            old_root="temperature",
            new_root="density",
            dry_run=True,
            include_accepted=True,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert result.conflicts == []
        assert {r["from"]: r["to"] for r in result.renamed} == {
            "electron_temperature": "electron_density",
            "ion_temperature": "ion_density",
        }
        assert "electron_temperature" in fake.nodes
        assert "electron_density" not in fake.nodes


class TestAcceptedRootReportsAndPersists:
    def test_apply_persists_descendants_and_reports_dry_run_false(
        self, tmp_path: Path
    ) -> None:
        """Once the root has reached accepted, the same apply call persists
        the descendants and reports ``dry_run=False`` exactly as before — the
        honest-report contract must not have quieted the real writes."""
        fake = FakeGraph()
        successor, old_root, new_root = _accepted_successor_with_live_children(fake)
        result = cascade_descendants_of(
            fake,
            successor_id=successor,
            old_root=old_root,
            new_root=new_root,
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
        # The renaming was actually persisted.
        assert "electron_density" in fake.nodes
        assert "ion_density" in fake.nodes
        assert "electron_temperature" not in fake.nodes
        assert "ion_temperature" not in fake.nodes


class TestNonAppliedPathsReportTrue:
    """Every return path that does not persist must report ``dry_run=True``
    even when the caller requested an apply — a False value is the caller's
    only proof the renames were written."""

    def test_successor_mismatch_is_not_an_apply(self) -> None:
        fake = FakeGraph()
        fake.add_node("density", name_stage="accepted")
        result = cascade_descendants_of(
            fake,
            successor_id="density",
            old_root="temperature",
            new_root="other",
            dry_run=False,
        )
        assert result.dry_run is True
        assert result.conflicts

    def test_unknown_successor_is_not_an_apply(self) -> None:
        fake = FakeGraph()
        result = cascade_descendants_of(
            fake,
            successor_id="missing",
            old_root="temperature",
            new_root="missing",
            dry_run=False,
        )
        assert result.dry_run is True
        assert result.conflicts

    def test_colliding_rename_root_is_not_an_apply(self, tmp_path: Path) -> None:
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="superseded")
        fake.add_node("source_rate", name_stage="accepted")
        result = rename_cascade(
            fake,
            "temperature",
            "source_rate",
            dry_run=False,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert result.conflicts

    def test_noop_rename_is_not_an_apply(self, tmp_path: Path) -> None:
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="superseded")
        result = rename_cascade(
            fake,
            "temperature",
            "temperature",
            dry_run=False,
            audit_log_path=tmp_path / "audit.log",
        )
        assert result.dry_run is True
        assert result.conflicts


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
