"""A rename moves an identity; it must not silently unpublish its document.

``persist_refined_name`` mints the successor of every renamed StandardName and
copies the predecessor's description onto it. Documentation is the other half
of the same prose and was dropped, so a re-rendering of an already-documented
quantity landed as an undocumented node with its docs axis reset — the text
was written about a meaning that had not changed.

These tests pin both halves of the rule. Documentation and the whole docs
lifecycle travel when the two spellings parse to the same grammar IR, and they
do not travel otherwise: a name that says something else must not inherit an
accepted document asserting what nobody wrote about it.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_GC_PATH = "imas_codex.standard_names.graph_store.graph_foundation.GraphClient"

# Measured live pairs. The re-rendered spellings are from the cohort a
# rendering-rule migration minted; the meaning-changing ones are refinements
# that also rewrote the description.
RERENDERED = [
    ("maximum_of_energy_flux_at_limiter", "energy_flux_maximum_at_limiter"),
    ("accumulated_neutral_count_at_wall", "neutral_count_accumulated_at_wall"),
    (
        "wave_current_of_antenna_strap_amplitude",
        "wave_current_amplitude_of_antenna_strap",
    ),
]
REMEANT = [
    ("beta", "toroidal_plasma_beta"),
    ("etendue", "viewing_etendue_of_spectrometer_channel"),
    ("psi_star", "difference_of_magnetic_flux_and_toroidal_magnetic_flux"),
]
QUALIFIER_ADDITION = (
    "power_due_to_ion_cyclotron_heating",
    "net_power_due_to_ion_cyclotron_heating",
)
PROJECTION_NARROWING = (
    "power_due_to_ion_cyclotron_heating",
    "perpendicular_power_due_to_ion_cyclotron_heating",
)


def _graph_client_double(tx: MagicMock) -> MagicMock:
    """Return a context-managed graph client using one transaction double."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc


def _capture_persist(
    old_name: str,
    new_name: str,
    **persist_kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    """Run the persist against a mocked transaction; return its Cypher + params.

    The preflight cannot commit against a mock, so it refuses — the statement
    it issued and the parameters bound to it are what these tests read.
    """
    from imas_codex.standard_names.graph_store.graph_refinement import (
        RefinedNamePersistenceRefusal,
        persist_refined_name,
    )

    tx = MagicMock()
    tx.closed = False
    tx.run.return_value = []
    gc = _graph_client_double(tx)

    with patch(_GC_PATH, return_value=gc):
        with pytest.raises(RefinedNamePersistenceRefusal):
            persist_refined_name(
                old_name=old_name,
                new_name=new_name,
                description="unchanged prose about one quantity",
                **persist_kwargs,
            )

    call = tx.run.call_args_list[0]
    return " ".join(call.args[0].split()), call.kwargs


def _successful_persist_with_assertion() -> tuple[MagicMock, MagicMock]:
    """Run the persistence path far enough to observe both change receipts."""
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )

    old_name, new_name = QUALIFIER_ADDITION
    tx = MagicMock()
    tx.closed = False
    tx.run.side_effect = [
        [
            {
                "old_name": old_name,
                "new_name": new_name,
                "source_documentation": "accepted predecessor document",
            }
        ],
        [{"old_name": old_name, "new_name": new_name}],
    ]
    gc = _graph_client_double(tx)
    record_path = (
        "imas_codex.standard_names.provenance_lifecycle.record_standard_name_change"
    )
    with (
        patch(_GC_PATH, return_value=gc),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=SimpleNamespace(
                accepted_source_ids=[],
                rejected=[],
            ),
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=0,
        ),
        patch(record_path, return_value="sn-change:recorded") as record_change,
        patch("imas_codex.standard_names.graph_store.graph_audits.bump_sn_run_counter"),
    ):
        persist_refined_name(
            old_name=old_name,
            new_name=new_name,
            description="unchanged prose about one quantity",
            edit_mode="rename",
            expected_old_stage="accepted",
            meaning_preservation_reason=(
                "The source bindings and physical definition are unchanged; "
                "net makes the existing sign convention explicit."
            ),
            meaning_preservation_asserted_by="reviewer@example.org",
        )

    return tx, record_change


class TestTheGrammarDecidesWhetherTheMeaningMoved:
    """IR equality is the test for a rename that changed only the spelling."""

    @pytest.mark.parametrize(("old", "new"), RERENDERED)
    def test_a_re_rendering_preserves_meaning(self, old: str, new: str):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        assert rename_preserves_meaning(old, new) is True

    @pytest.mark.parametrize(("old", "new"), REMEANT)
    def test_a_different_quantity_does_not(self, old: str, new: str):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        assert rename_preserves_meaning(old, new) is False

    def test_an_unparseable_spelling_answers_no(self):
        """Absence of evidence is not evidence of sameness."""
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        assert rename_preserves_meaning("beta", "!!! not a name !!!") is False
        assert rename_preserves_meaning("!!! not a name !!!", "beta") is False


class TestExplicitMeaningAuthority:
    """Grammar remains conservative; an attributed assertion may override it."""

    @pytest.mark.parametrize("pair", [QUALIFIER_ADDITION, PROJECTION_NARROWING])
    def test_a_changed_grammar_field_is_not_automatically_equivalent(
        self,
        pair: tuple[str, str],
    ):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        assert rename_preserves_meaning(*pair) is False

    def test_the_measured_qualifier_addition_needs_explicit_authority(self):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        assert (
            rename_preserves_meaning(
                *QUALIFIER_ADDITION,
                assertion_reason=(
                    "The source bindings and physical definition are unchanged; "
                    "net makes the existing sign convention explicit."
                ),
                asserted_by="reviewer@example.org",
            )
            is True
        )

    @pytest.mark.parametrize(
        ("reason", "asserted_by", "message"),
        [
            (None, "reviewer@example.org", "substantive reason"),
            ("", "reviewer@example.org", "substantive reason"),
            ("physics definition unchanged", None, "asserting actor"),
            ("physics definition unchanged", "", "asserting actor"),
        ],
    )
    def test_an_incomplete_assertion_is_refused(
        self,
        reason: str | None,
        asserted_by: str | None,
        message: str,
    ):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            rename_preserves_meaning,
        )

        with pytest.raises(ValueError, match=message):
            rename_preserves_meaning(
                *QUALIFIER_ADDITION,
                assertion_reason=reason,
                asserted_by=asserted_by,
            )

    def test_persistence_binds_the_explicit_assertion_to_the_existing_carry(self):
        _, params = _capture_persist(
            *QUALIFIER_ADDITION,
            meaning_preservation_reason="physics definition unchanged",
            meaning_preservation_asserted_by="reviewer@example.org",
        )
        assert params["meaning_preserved"] is True

    def test_persistence_records_a_distinct_content_addressed_receipt(self):
        import hashlib

        _, record_change = _successful_persist_with_assertion()
        receipt = record_change.call_args_list[-1]
        expected_digest = hashlib.sha256(b"accepted predecessor document").hexdigest()

        assert receipt.args[1] == (
            "standard-name-document:power_due_to_ion_cyclotron_heating:"
            f"sha256:{expected_digest}"
        )
        assert receipt.args[2] == "net_power_due_to_ion_cyclotron_heating"
        assert receipt.kwargs == {
            "operation": "semantics_preserving_rename",
            "reason": (
                "The source bindings and physical definition are unchanged; "
                "net makes the existing sign convention explicit."
            ),
            "origin": "reviewer@example.org",
            "run_id": None,
        }


class TestTheSuccessorInheritsTheDocumentation:
    """A re-rendering carries the text and the docs lifecycle with it."""

    def test_documentation_is_read_from_the_predecessor(self):
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert "new.documentation = CASE WHEN $meaning_preserved" in cypher
        assert "THEN old.documentation" in cypher

    def test_the_docs_stage_is_the_predecessors_not_pending(self):
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert (
            "new.docs_stage = CASE WHEN $meaning_preserved "
            "AND coalesce(old.documentation, '') <> '' "
            "THEN coalesce(old.docs_stage, 'pending') ELSE 'pending' END" in cypher
        )

    @pytest.mark.parametrize(
        "field", ["docs_chain_length", "docs_model", "docs_generated_at"]
    )
    def test_the_docs_provenance_travels_with_the_text(self, field: str):
        """Text without its provenance would misreport who wrote it and when."""
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert f"new.{field} = CASE WHEN $meaning_preserved" in cypher

    @pytest.mark.parametrize(("old", "new"), RERENDERED)
    def test_the_bound_parameter_is_true_for_a_re_rendering(self, old: str, new: str):
        _, params = _capture_persist(old, new)
        assert params["meaning_preserved"] is True


class TestAChangedMeaningStartsTheDocsAxisOver:
    """The negative half: no accepted document may follow a new meaning."""

    @pytest.mark.parametrize(("old", "new"), REMEANT)
    def test_the_bound_parameter_is_false_for_a_new_meaning(self, old: str, new: str):
        _, params = _capture_persist(old, new)
        assert params["meaning_preserved"] is False

    def test_the_false_branch_writes_pending_and_no_text(self):
        """Every docs field's ELSE branch must be the undocumented state."""
        cypher, _ = _capture_persist(*REMEANT[0])
        for fragment in (
            "new.docs_stage = CASE WHEN $meaning_preserved "
            "AND coalesce(old.documentation, '') <> '' "
            "THEN coalesce(old.docs_stage, 'pending') ELSE 'pending' END",
            "new.documentation = CASE WHEN $meaning_preserved "
            "THEN old.documentation ELSE null END",
            "new.docs_model = CASE WHEN $meaning_preserved "
            "THEN old.docs_model ELSE null END",
            "new.docs_generated_at = CASE WHEN $meaning_preserved "
            "THEN old.docs_generated_at ELSE null END",
            "new.docs_chain_length = CASE WHEN $meaning_preserved "
            "THEN coalesce(old.docs_chain_length, 0) ELSE 0 END",
        ):
            assert fragment in cypher

    def test_an_undocumented_predecessor_leaves_the_stage_pending(self):
        """Carrying an 'accepted' stage over empty text would claim a document
        that does not exist, so the stage guard reads the text and not the
        stage."""
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert "coalesce(old.documentation, '') <> ''" in cypher


class TestAnExistingRenameCanReceiveItsAcceptedDocument:
    """The post-rename repair is atomic, attributed, and idempotent."""

    def test_the_repair_writes_only_the_docs_axis_and_a_receipt(self):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            carry_accepted_documentation_across_rename,
        )

        old_name, new_name = QUALIFIER_ADDITION
        tx = MagicMock()
        tx.closed = False
        tx.run.side_effect = [
            [
                {
                    "predecessor_name_stage": "superseded",
                    "predecessor_docs_stage": "accepted",
                    "predecessor_documentation": "accepted predecessor document",
                    "predecessor_docs_chain_length": 2,
                    "predecessor_docs_model": "provider/model",
                    "predecessor_docs_generated_at": "2026-09-01T21:49:51Z",
                    "successor_docs_stage": "pending",
                    "successor_documentation": None,
                    "successor_docs_chain_length": 0,
                    "successor_docs_model": None,
                    "successor_docs_generated_at": None,
                    "successor_name_stage": "accepted",
                    "successor_status": "draft",
                }
            ],
            [
                {
                    "changed": True,
                    "name_stage": "accepted",
                    "status": "draft",
                    "docs_stage": "accepted",
                    "documentation_length": 29,
                    "docs_model": "provider/model",
                    "docs_generated_at": "2026-09-01T21:49:51Z",
                }
            ],
        ]
        gc = _graph_client_double(tx)
        with patch(_GC_PATH, return_value=gc):
            result = carry_accepted_documentation_across_rename(
                predecessor_name=old_name,
                successor_name=new_name,
                reason="physics definition unchanged",
                asserted_by="reviewer@example.org",
            )

        mutation = tx.run.call_args_list[1]
        cypher = " ".join(mutation.args[0].split())
        assert "new.docs_stage = old.docs_stage" in cypher
        assert "new.documentation = old.documentation" in cypher
        assert "new.docs_model = old.docs_model" in cypher
        assert "new.docs_generated_at = old.docs_generated_at" in cypher
        assert "MERGE (new)-[:HAS_INTERNAL_CHANGE]->(change)" in cypher
        assert "new.name_stage =" not in cypher
        assert "new.status =" not in cypher
        assert mutation.kwargs["operation"] == "semantics_preserving_rename"
        assert mutation.kwargs["reason"] == "physics definition unchanged"
        assert mutation.kwargs["asserted_by"] == "reviewer@example.org"
        assert result["changed"] is True
        assert result["status"] == "draft"
        tx.commit.assert_called_once_with()

    @pytest.mark.parametrize(
        ("reason", "asserted_by", "message"),
        [
            ("", "reviewer@example.org", "substantive reason"),
            ("physics definition unchanged", "", "asserting actor"),
        ],
    )
    def test_the_repair_refuses_an_unproven_declaration(
        self,
        reason: str,
        asserted_by: str,
        message: str,
    ):
        from imas_codex.standard_names.graph_store.graph_refinement import (
            carry_accepted_documentation_across_rename,
        )

        with pytest.raises(ValueError, match=message):
            carry_accepted_documentation_across_rename(
                predecessor_name=QUALIFIER_ADDITION[0],
                successor_name=QUALIFIER_ADDITION[1],
                reason=reason,
                asserted_by=asserted_by,
            )
