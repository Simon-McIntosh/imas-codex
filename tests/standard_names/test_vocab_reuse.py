"""Mechanical reuse detection must override an embedding's distinct verdict.

The semantic dedup check scores a proposed token against the registered
vocabulary by cosine similarity and lets the composer confirm a near neighbour
as genuinely distinct.  Two classes of proposal are reuse *by construction* and
must never reach that adjudication:

- a registered token plus a DD structural subdivision noun
  (``lower_hybrid_antenna_module`` for the registered ``lower_hybrid_antenna``);
- the same words in a different order (``methane_deuterated`` for the
  registered ``deuterated_methane``).

Both were stamped ``distinct_confirmed`` at 0.96–0.97 in the stored population,
which is what made the dedup verdict untrustworthy.  A stated physical reason —
an isotope distinction such as ``lithium_6`` against ``lithium`` — is the case
these checks must NOT claim, and it is pinned here as a negative.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.segments import grammar_token_index


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar.constants  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)


@requires_isn
class TestStructuralSuffix:
    """``<registered>_<DD subdivision noun>`` reuses the registered token."""

    @pytest.mark.parametrize(
        ("segment", "token", "target"),
        [
            ("device", "lower_hybrid_antenna_module", "lower_hybrid_antenna"),
            (
                "device",
                "ion_cyclotron_heating_antenna_module",
                "ion_cyclotron_heating_antenna",
            ),
            ("geometric_base", "unit_vector_component", "unit_vector"),
            ("physical_base", "volume_element", "volume"),
            ("object", "vessel_element", "vessel"),
            ("device", "mse_channel", "mse"),
            ("device", "hard_xray_channel", "hard_xray"),
        ],
    )
    def test_suffix_hit_names_the_registered_target(self, segment, token, target):
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        assert target in grammar_token_index(), f"fixture stale: {target} unregistered"
        finding = registered_reuse(segment, token)
        assert finding is not None, f"{token} should resolve to {target}"
        assert finding.target == target
        assert finding.mechanism == "structural_suffix"

    def test_an_unregistered_stem_is_not_a_reuse_claim(self):
        """The suffix alone proves nothing — the stem must be registered."""
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        assert registered_reuse("device", "wibble_frobnicator_module") is None


@requires_isn
class TestWordOrderVariant:
    """The same words in another order are one token, not two."""

    def test_order_variant_names_the_registered_spelling(self):
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        finding = registered_reuse("subject", "methane_deuterated")
        assert finding is not None
        assert finding.target == "deuterated_methane"
        assert finding.mechanism == "word_order"

    def test_the_check_holds_whatever_segment_it_was_reported_against(self):
        """The stored population carries the same token under two segments."""
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        finding = registered_reuse("qualifier", "methane_deuterated")
        assert finding is not None
        assert finding.target == "deuterated_methane"

    def test_registered_vocabulary_contains_no_order_collisions(self):
        """Two registered tokens sharing a word multiset would make this check ambiguous."""
        from collections import defaultdict

        by_words: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for token in grammar_token_index():
            words = token.split("_")
            if len(words) > 1:
                by_words[tuple(sorted(words))].append(token)
        collisions = {k: v for k, v in by_words.items() if len(v) > 1}
        assert not collisions, f"order-variant check is ambiguous for {collisions}"


@requires_isn
class TestSettledSynonym:
    """A synonym settled in review is named back to the composer."""

    def test_optical_depth_points_at_opacity(self):
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        finding = registered_reuse("physical_base", "optical_depth")
        assert finding is not None
        assert finding.target == "opacity"
        assert finding.mechanism == "settled_synonym"

    def test_every_synonym_target_is_registered(self):
        """A mapping whose target ISN dropped must fall silent, not mislead."""
        from imas_codex.standard_names.vocab_reuse import settled_synonyms

        for source, target in settled_synonyms().items():
            assert target in grammar_token_index(), (
                f"synonym {source} -> {target} names an unregistered target"
            )


@requires_isn
class TestNegatives:
    """A stated physical reason keeps a near neighbour distinct."""

    def test_an_isotope_is_not_reuse_of_its_element(self):
        """``lithium_6`` is a genuine isotope distinction, not a spelling of ``lithium``."""
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        assert "lithium" in grammar_token_index()
        assert registered_reuse("subject", "lithium_6") is None

    @pytest.mark.parametrize(
        "token", ["substrate", "wibble_frobnicator", "detection_efficiency"]
    )
    def test_genuine_demand_is_not_reclassified_as_reuse(self, token):
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        assert registered_reuse("physical_base", token) is None

    def test_a_registered_token_is_not_reuse_of_itself(self):
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        assert registered_reuse("physical_base", "density") is None


@requires_isn
class TestStoredVerdictAudit:
    """The re-audit entry point the graph-side reconcile calls."""

    RECORDS = [
        {
            "id": "vocab_gap:device:lower_hybrid_antenna_module",
            "segment": "device",
            "token": "lower_hybrid_antenna_module",
            "dedup": "distinct_confirmed",
            "nearest": "lower_hybrid_antenna",
            "sim": 0.9702,
        },
        {
            "id": "vocab_gap:subject:methane_deuterated",
            "segment": "subject",
            "token": "methane_deuterated",
            "dedup": "distinct_confirmed",
            "nearest": "deuterated_methane",
            "sim": 0.9611,
        },
        {
            "id": "vocab_gap:object:vessel_element",
            "segment": "object",
            "token": "vessel_element",
            "dedup": "unchecked",
            "nearest": None,
            "sim": None,
        },
        {
            "id": "vocab_gap:subject:lithium_6",
            "segment": "subject",
            "token": "lithium_6",
            "dedup": "distinct_confirmed",
            "nearest": "lithium",
            "sim": 0.9123,
        },
        {
            "id": "vocab_gap:physical_base:detection_efficiency",
            "segment": "physical_base",
            "token": "detection_efficiency",
            "dedup": "unchecked",
            "nearest": None,
            "sim": None,
        },
    ]

    def test_audit_finds_the_reuse_cases_and_leaves_the_rest(self):
        from imas_codex.standard_names.vocab_reuse import audit_stored_verdicts

        audits = audit_stored_verdicts(self.RECORDS)
        assert {a.token for a in audits} == {
            "lower_hybrid_antenna_module",
            "methane_deuterated",
            "vessel_element",
        }

    def test_audit_carries_the_stored_decision_it_overrides(self):
        from imas_codex.standard_names.vocab_reuse import audit_stored_verdicts

        by_token = {a.token: a for a in audit_stored_verdicts(self.RECORDS)}
        assert by_token["lower_hybrid_antenna_module"].stored_decision == (
            "distinct_confirmed"
        )
        assert by_token["vessel_element"].stored_decision == "unchecked"
        assert by_token["lower_hybrid_antenna_module"].finding.target == (
            "lower_hybrid_antenna"
        )

    def test_audit_reads_a_records_dedup_under_either_key(self):
        """The graph projects the property as ``dedup_decision``; the dump abbreviates it."""
        from imas_codex.standard_names.vocab_reuse import audit_stored_verdicts

        audits = audit_stored_verdicts(
            [
                {
                    "segment": "object",
                    "token": "vessel_element",
                    "dedup_decision": "distinct_confirmed",
                }
            ]
        )
        assert [a.stored_decision for a in audits] == ["distinct_confirmed"]

    def test_audit_is_read_only_over_its_input(self):
        from imas_codex.standard_names.vocab_reuse import audit_stored_verdicts

        before = [dict(r) for r in self.RECORDS]
        audit_stored_verdicts(self.RECORDS)
        assert self.RECORDS == before
