"""Tests for the ``sn edit`` engine's scope handling — shared-base guard,
family→parent mapping, subtree cascade planning, and atomic descendant
cascade application on acceptance.

Reuses the ``FakeGraph`` in-memory graph double from ``test_edit_engine.py``
(no live Neo4j — see that module's docstring for how it emulates
``GraphClient``'s query + transaction surface).
"""

from __future__ import annotations

from imas_codex.standard_names.edit import apply_edit
from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph


def _temperature_family(fake: FakeGraph) -> None:
    """``temperature`` parented by ``electron_temperature`` / ``ion_temperature``."""
    fake.add_node("temperature", name_stage="accepted")
    fake.add_node("electron_temperature", name_stage="accepted")
    fake.add_node("ion_temperature", name_stage="accepted")
    fake.add_edge("electron_temperature", "temperature", "electron", "qualifier")
    fake.add_edge("ion_temperature", "temperature", "ion", "qualifier")


class TestSharedBaseGuard:
    def test_leaf_shared_base_edit_remains_local_without_family(self) -> None:
        fake = FakeGraph()
        _temperature_family(fake)
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="rename the underlying quantity",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.successor == "electron_density"
        assert any("keeping the base change local" in action for action in plan.actions)
        assert fake.nodes["temperature"]["name_stage"] == "accepted"

    def test_qualifier_only_change_does_not_trigger_guard(self) -> None:
        """Changing only the qualifier token (electron→upper), leaving the
        shared base (temperature) untouched, needs no family scope even
        though siblings exist."""
        fake = FakeGraph()
        _temperature_family(fake)
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="upper_temperature",
                reason="this measurement is actually the upper qualifier",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.successor == "upper_temperature"

    def test_no_siblings_no_guard(self) -> None:
        """A qualified leaf whose parent has only ONE child is free to
        rename its shared-base segment in place — nothing desyncs."""
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="accepted")
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_edge("electron_temperature", "temperature", "electron", "qualifier")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="rename the underlying quantity",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True


class TestFamilyMapping:
    def test_leaf_family_scope_never_lifts_across_locus_parent(self) -> None:
        """A local outline repair cannot become a radial-coordinate root edit."""
        fake = FakeGraph()
        fake.add_node("radial_coordinate", name_stage="accepted")
        affected = "radial_coordinate_of_control_surface"
        fake.add_node(affected, name_stage="accepted")
        fake.add_edge(
            affected,
            "radial_coordinate",
            "control_surface",
            "locus",
        )
        unrelated = [
            "radial_coordinate_at_inboard_midplane",
            "radial_coordinate_at_outboard_midplane",
            "radial_coordinate_of_antenna_strap",
            "radial_coordinate_of_aperture",
            "radial_coordinate_of_camera",
            "radial_coordinate_of_closest_wall_point",
            "radial_coordinate_of_coil_conductor_element",
            "radial_coordinate_of_conductor_cross_section",
            "radial_coordinate_of_current_center",
            "radial_coordinate_of_detector_pixel",
            "radial_coordinate_of_dr_dz_zero_point",
            "radial_coordinate_of_electron_cyclotron_launcher_mirror",
            "radial_coordinate_of_ferritic_element",
            "radial_coordinate_of_ferritic_element_centroid",
            "radial_coordinate_of_filter_window",
            "radial_coordinate_of_flux_loop",
            "radial_coordinate_of_geometric_axis",
            "radial_coordinate_of_iron_core_segment",
            "radial_coordinate_of_line_of_sight",
            "radial_coordinate_of_lower_hybrid_antenna_row",
            "radial_coordinate_of_magnetic_axis",
            "radial_coordinate_of_measurement_position",
            "radial_coordinate_of_neutral_beam_injector",
            "radial_coordinate_of_pellet_path",
            "radial_coordinate_of_poloidal_field_coil",
            "radial_coordinate_of_poloidal_magnetic_field_probe",
            "radial_coordinate_of_reflector",
            "radial_coordinate_of_rogowski_coil",
            "radial_coordinate_of_shattering_position",
            "radial_coordinate_of_shunt",
            "radial_coordinate_of_spectrometer_channel",
            "radial_coordinate_of_strike_point",
            "radial_coordinate_of_thomson_scattering_laser",
            "radial_coordinate_of_x_point",
        ]
        assert len(unrelated) == 34
        for child in unrelated:
            fake.add_node(child, name_stage="accepted")
            fake.add_edge(
                child,
                "radial_coordinate",
                child.removeprefix("radial_coordinate_of_"),
                "locus",
            )

        with _patched_graph(fake):
            plan = apply_edit(
                target=affected,
                rename="radial_outline_of_control_surface",
                reason="the source represents an outline coordinate",
                include_accepted=True,
                gc=fake,
            )

        assert plan.blocked is None
        assert plan.applied is True
        assert plan.successor == "radial_outline_of_control_surface"
        assert plan.cascade_deferred == []
        assert fake.nodes["radial_coordinate"]["name_stage"] == "accepted"
        assert all(child in fake.nodes for child in unrelated)
        assert all(
            fake.edges_by_child[child][0]["parent_id"] == "radial_coordinate"
            for child in unrelated
        )

    def test_family_scope_leaf_base_change_refuses_parent_promotion(self) -> None:
        fake = FakeGraph()
        _temperature_family(fake)
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="rename the underlying quantity",
                scope="family",
                # siblings are accepted — the family cascade needs the opt-in
                include_accepted=True,
                gc=fake,
            )
        assert plan.blocked is not None
        assert "leaf" in plan.blocked
        assert "parent" in plan.blocked
        assert plan.applied is False
        assert plan.successor is None
        assert plan.cascade_deferred == []
        assert "density" not in fake.nodes
        assert fake.nodes["temperature"]["name_stage"] == "accepted"

    def test_family_mapping_dry_run_plans_without_writing(self) -> None:
        fake = FakeGraph()
        _temperature_family(fake)
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="rename the underlying quantity",
                scope="family",
                include_accepted=True,
                dry_run=True,
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is not None
        assert plan.run_id is None
        assert plan.cascade_deferred == []
        # No graph mutation at all.
        assert "density" not in fake.nodes
        assert fake.nodes["temperature"]["name_stage"] == "accepted"
        assert (
            fake.edges_by_child["electron_temperature"][0]["parent_id"] == "temperature"
        )


class TestSubtreeCascadePlanning:
    def test_heterogeneous_cohort_refuses_before_review_staging(self) -> None:
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="accepted")
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_node("maximum_of_temperature", name_stage="accepted")
        fake.add_edge("ion_temperature", "temperature", "ion", "qualifier")
        fake.add_edge(
            "maximum_of_temperature",
            "temperature",
            "maximum",
            "unary_prefix",
        )

        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="density",
                reason="replace the parent quantity",
                include_accepted=True,
                gc=fake,
            )

        assert plan.blocked is not None
        assert "heterogeneous" in plan.blocked
        assert plan.applied is False
        assert plan.successor is None
        assert "density" not in fake.nodes
        assert fake.nodes["temperature"]["name_stage"] == "accepted"

    def test_subtree_rename_on_parent_target(self) -> None:
        fake = FakeGraph()
        _temperature_family(fake)
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="density",
                reason="rename the underlying quantity",
                include_accepted=True,  # accepted descendants — opt in
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.scope == "subtree"  # default for a parent target
        assert plan.successor == "density"
        planned_pairs = {(r["from"], r["to"]) for r in plan.cascade_deferred}
        assert ("electron_temperature", "electron_density") in planned_pairs
        assert ("ion_temperature", "ion_density") in planned_pairs

    def test_cascade_conflict_refuses_the_whole_edit(self) -> None:
        """A pre-existing collision on a would-be cascade descendant id
        refuses the edit all-or-nothing — nothing is written."""
        fake = FakeGraph()
        _temperature_family(fake)
        # electron_density already exists as an unrelated node — the
        # cascade's collision check must catch this.
        fake.add_node("electron_density", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="density",
                reason="rename the underlying quantity",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "cascade plan conflict" in plan.blocked
        assert plan.applied is False
        # All-or-nothing: nothing was written, not even the root rename.
        assert "density" not in fake.nodes
        assert fake.nodes["temperature"]["name_stage"] == "accepted"
        assert "electron_temperature" in fake.nodes


class TestAcceptanceCascadeApplication:
    def _drafted_subtree_edit(self, fake: FakeGraph) -> None:
        """Simulate the graph state right after a subtree-scoped rename's
        successor was created by persist_refined_name: the old parent is
        superseded, the new parent is drafted+open, and both children's
        HAS_PARENT edges already point at the new (drafted) parent id."""
        fake.add_node("temperature", name_stage="superseded")
        fake.add_node(
            "density",
            name_stage="drafted",
            edit_status="open",
            edit_scope="subtree",
            edit_mode="rename",
            # the attach-time opt-in that authorised the accepted descendants
            edit_include_accepted=True,
            claim_token="tok",
        )
        fake.refined_from["density"] = "temperature"
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_edge("electron_temperature", "density", "electron", "qualifier")
        fake.add_edge("ion_temperature", "density", "ion", "qualifier")

    def test_acceptance_applies_descendant_renames_atomically(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        self._drafted_subtree_edit(fake)
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="density",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert stage == "accepted"
        assert fake.nodes["density"]["edit_status"] == "applied"
        # Descendants renamed atomically alongside the root's acceptance.
        assert "electron_density" in fake.nodes
        assert "ion_density" in fake.nodes
        assert "electron_temperature" not in fake.nodes
        assert "ion_temperature" not in fake.nodes
        assert fake.edges_by_child["electron_density"][0]["parent_id"] == "density"

    def test_acceptance_time_conflict_refuses_acceptance_atomically(self) -> None:
        """Cascade atomicity: a descendant-id collision detected at
        acceptance time refuses the acceptance itself — the root is NOT
        accepted and NOTHING is renamed (full rollback, nothing persisted).
        """
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        self._drafted_subtree_edit(fake)
        # A collision appears between attach-time and acceptance-time: the
        # would-be cascade target `electron_density` already exists.
        fake.add_node("electron_density", name_stage="accepted")
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="density",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        # The acceptance is refused — the reviewed decision cannot land while
        # its atomic cascade is invalid.
        assert stage == "reviewed"
        assert fake.nodes["density"]["name_stage"] == "reviewed"
        # Edit stays open (rides refine) — it was NOT applied.
        assert fake.nodes["density"]["edit_status"] == "open"
        # Nothing renamed: descendants and the pre-existing collider untouched.
        assert "electron_temperature" in fake.nodes
        assert "ion_temperature" in fake.nodes
        assert "ion_density" not in fake.nodes
        # The refusal reason is recorded for operator follow-up.
        issues = fake.nodes["density"].get("validation_issues") or []
        assert any("edit_cascade" in issue for issue in issues)
