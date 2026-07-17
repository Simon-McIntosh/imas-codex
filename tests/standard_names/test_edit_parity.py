"""Validation-parity tests for the ``sn edit`` engine.

The locked decision for the edit engine is *full-pipeline-parity*: every
edit-origin artifact clears exactly the gates a pipeline-generated name
clears, and a quarantined edit result can never reach ``accepted`` — there
is no privileged path.  These tests pin that guarantee for each edit mode
(rename / docs / hint), pairing every edit assertion with the equivalent
pipeline-gate assertion (:func:`validate_name_candidate`) so the two are
demonstrably the SAME gate.

Runs against the in-memory ``FakeGraph`` from ``test_edit_engine`` — no live
Neo4j.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.edit import apply_edit
from imas_codex.standard_names.workers import validate_name_candidate
from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph

# =============================================================================
# The shared pipeline gate — the reference every edit mode is compared against
# =============================================================================


class TestPipelineGateBaseline:
    def test_gate_quarantines_grammar_invalid(self) -> None:
        _issues, _summary, status = validate_name_candidate({"id": "NOTaname!!"})
        assert status == "quarantined"

    def test_gate_quarantines_bad_unit(self) -> None:
        _issues, _summary, status = validate_name_candidate(
            {"id": "electron_temperature", "kind": "scalar", "unit": "not_a_unit"}
        )
        assert status == "quarantined"

    def test_gate_passes_clean_name(self) -> None:
        _issues, _summary, status = validate_name_candidate(
            {"id": "electron_temperature", "kind": "scalar", "unit": "eV"}
        )
        assert status == "valid"


class TestDerivedParentGateScoping:
    """The full-name parse gate is scoped for derived family parents.

    A derived parent is a deliberately partial name (a grammar peel). It must
    be validated structurally — children exist + the peel generalises them —
    NOT rejected merely for failing the standalone full-name round-trip.
    """

    def test_partial_derived_parent_with_children_is_valid(self) -> None:
        """``internal_state_energy_flux`` cannot parse standalone (its species
        subject is peeled off) but is a valid parent of species children."""
        issues, _summary, status = validate_name_candidate(
            {
                "id": "internal_state_energy_flux",
                "origin": "derived",
                "children": [
                    "deuterium_internal_state_energy_flux",
                    "tungsten_internal_state_energy_flux",
                ],
            }
        )
        assert status == "valid", issues

    def test_orphan_derived_parent_quarantines(self) -> None:
        """A derived parent that fails the parse AND has no children is a
        structurally-broken residue — the missed-gate signal is preserved."""
        issues, _summary, status = validate_name_candidate(
            {
                "id": "internal_state_energy_flux",
                "origin": "derived",
                "children": [],
            }
        )
        assert status == "quarantined"
        assert any("no HAS_PARENT children" in i for i in issues)

    def test_inconsistent_peel_derived_parent_quarantines(self) -> None:
        """A derived parent whose tokens do not generalise any child is a bad
        peel — quarantined with a structural finding, not silently admitted."""
        issues, _summary, status = validate_name_candidate(
            {
                "id": "internal_state_energy_flux",
                "origin": "derived",
                "children": ["electron_temperature"],
            }
        )
        assert status == "quarantined"
        assert any("not a token-generalisation" in i for i in issues)

    def test_non_derived_partial_name_still_fails_full_parse(self) -> None:
        """Scoping is derived-only: a non-derived name that fails the full-name
        parse is still quarantined for the parse error."""
        issues, _summary, status = validate_name_candidate(
            {"id": "internal_state_energy_flux", "origin": "pipeline"}
        )
        assert status == "quarantined"
        assert any("parse_error" in i or "grammar:ambiguity" in i for i in issues)


# =============================================================================
# rename mode
# =============================================================================


class TestRenameParity:
    def test_grammar_invalid_blocked_at_attach(self) -> None:
        """A grammar-invalid rename is refused up front by the same ISN
        round-trip that quarantines the equivalent pipeline candidate."""
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="accepted", unit="eV")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="NOTaname!!",
                reason="typo",
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is not None
        assert "round-trip" in plan.blocked
        # Same defect quarantines the equivalent pipeline candidate.
        assert validate_name_candidate({"id": "NOTaname!!"})[2] == "quarantined"

    def test_semantically_invalid_successor_quarantined(self) -> None:
        """A rename that round-trips but carries a semantically-invalid field
        (bad unit) is stamped ``validation_status='quarantined'`` on its
        successor by the SAME gate the pipeline runs — so it cannot be
        accepted."""
        fake = FakeGraph()
        fake.add_node(
            "temperature",
            name_stage="accepted",
            unit="not_a_unit",
            kind="scalar",
            description="temp",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="electron_temperature",
                reason="qualify by species",
                gc=fake,
            )
        assert plan.applied is True
        assert plan.successor == "electron_temperature"
        succ = fake.nodes["electron_temperature"]
        assert succ["validation_status"] == "quarantined"
        # Parity: the pipeline gate quarantines the equivalent candidate.
        assert (
            validate_name_candidate(
                {"id": "electron_temperature", "kind": "scalar", "unit": "not_a_unit"}
            )[2]
            == "quarantined"
        )

    def test_clean_rename_successor_is_valid(self) -> None:
        """Control: a clean rename passes the gate (validation_status=valid),
        proving the quarantine above is the field defect, not the edit path."""
        fake = FakeGraph()
        fake.add_node(
            "temperature",
            name_stage="accepted",
            unit="eV",
            kind="scalar",
            description="temp",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="electron_temperature",
                reason="qualify by species",
                gc=fake,
            )
        assert plan.applied is True
        assert fake.nodes["electron_temperature"]["validation_status"] == "valid"

    def test_quarantined_successor_cannot_reach_accepted(self) -> None:
        """The quarantined successor rides the review gate exactly like a
        quarantined pipeline candidate: the review worker persists a 0.0
        score, which cannot reach the acceptance threshold."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "temperature",
            name_stage="accepted",
            unit="not_a_unit",
            kind="scalar",
            description="temp",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="electron_temperature",
                reason="qualify by species",
                gc=fake,
            )
            assert fake.nodes[plan.successor]["validation_status"] == "quarantined"
            # The review worker's quarantine path persists score=0.0.
            fake.nodes[plan.successor]["claim_token"] = "rtok"
            stage = persist_reviewed_name(
                sn_id=plan.successor,
                claim_token="rtok",
                score=0.0,
                model="(skipped: quarantined)",
                min_score=0.75,
                rotation_cap=3,
            )
        assert stage != "accepted"
        assert fake.nodes[plan.successor]["name_stage"] != "accepted"


# =============================================================================
# docs mode
# =============================================================================


def _accepted_docs_node(fake: FakeGraph, sn_id: str, **extra) -> None:
    fields = {
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "description": "desc",
        "documentation": "old documentation",
        "kind": "scalar",
        "unit": "eV",
    }
    fields.update(extra)
    fake.add_node(sn_id, **fields)


class TestDocsParity:
    def test_requires_settled_docs_stage(self) -> None:
        """Docs edits require the docs axis to have settled — mid-flight docs
        (drafting/refining) cannot be steered (races the docs pool)."""
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature", docs_stage="drafted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is not None
        assert "docs_stage" in plan.blocked

    def test_exhausted_docs_stage_allowed(self) -> None:
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature", docs_stage="exhausted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                gc=fake,
            )
        assert plan.applied is True
        assert plan.entry == "review_docs"

    def test_catalog_edit_blocked_without_override(self) -> None:
        """Editing the documentation of a catalog-edited name runs the same
        protection filter the pipeline writers run — blocked without the
        explicit override."""
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature", origin="catalog_edit")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is not None
        assert "catalog" in plan.blocked.lower()

    def test_catalog_edit_allowed_with_override(self) -> None:
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature", origin="catalog_edit")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                override_edits=True,
                gc=fake,
            )
        assert plan.applied is True
        assert plan.entry == "review_docs"

    def test_pipeline_origin_docs_not_protected(self) -> None:
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature", origin="pipeline")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                gc=fake,
            )
        assert plan.applied is True

    def test_docs_edit_enters_review_not_preapproved(self) -> None:
        """Parity: an accepted docs edit re-enters the docs REVIEW gate with
        an open edit — the replacement is scored, never pre-accepted."""
        fake = FakeGraph()
        _accepted_docs_node(fake, "electron_temperature")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new documentation body",
                reason="clarify",
                gc=fake,
            )
        assert plan.entry == "review_docs"
        assert fake.nodes["electron_temperature"]["docs_stage"] == "drafted"
        assert fake.nodes["electron_temperature"]["edit_status"] == "open"


# =============================================================================
# hint mode
# =============================================================================


class TestHintParity:
    def test_hint_reenters_generate_for_regeneration(self) -> None:
        """A name-axis hint resets the producing source(s) so the name is
        REGENERATED through the generate_name pool — it therefore rides the
        pipeline's own admission gate by construction (parity, no edit-only
        validation path)."""
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted", unit="eV")
        fake.add_source("dd:some/path", sn_id="electron_temperature")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="prefer the species-qualified form",
                reason="steer",
                gc=fake,
            )
        assert plan.applied is True
        assert plan.entry == "generate"
        # The producing source was reset so the generate pool re-composes it.
        assert fake.sources["dd:some/path"]["status"] == "extracted"

    def test_name_hint_without_source_blocked(self) -> None:
        """A derived/structural name has no producing source — a name-axis
        hint cannot regenerate it and is refused (no silent no-op)."""
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted", unit="eV")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="steer the name",
                axis="name",
                reason="steer",
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is not None
        assert "producing" in plan.blocked


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
