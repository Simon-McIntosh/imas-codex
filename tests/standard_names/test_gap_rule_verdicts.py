"""A gap verdict must name what to do, and stay honest when it cannot.

Three classes of proposal the grammar already settles were reaching the composer
as ``absent`` — "no such token, ask ISN for one":

- an ordinal sample of a repeated structure (``line_of_sight_second_point``),
  which no vocabulary addition can ever legitimise because ordinality never
  enters a name;
- a DD coordinate-slot field name (``x1``, ``x2``), which carries no physics at
  all — two tokens behind 106 proposals in the stored population;
- a quotient written as one word with symbol shorthand in an operand
  (``gradient_rho_squared_over_B_squared``), which is a binary composition of
  registered operands once the shorthand is read.

The counterweight is pinned just as hard: a genuinely novel quantity stays
``absent``, a division is never folded into a single base, and a registered
token is never called a rule violation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_codex.standard_names.segments import (
    NON_ACTIONABLE_GAP_CATEGORIES,
    classify_gap,
    dd_indexed_field_words,
    describe_gap,
    grammar_tokens_by_segment,
    is_actionable_gap,
    ordinal_form,
)


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar.constants  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)


def _settled_fold_decisions() -> list[dict[str, object]]:
    """Load the reviewed folds that constrain gap classification."""
    evidence_path = (
        Path(__file__).parents[2] / "docs/evidence/sn-vocabulary-adjudication.json"
    )
    evidence = json.loads(evidence_path.read_text())
    return [
        decision for decision in evidence["decisions"] if decision["decision"] == "fold"
    ]


def _public_advisory_aliases() -> dict[tuple[str, str], str]:
    """Return the segment-scoped spelling guidance published by ISN."""
    from imas_standard_names.grammar import get_grammar_context

    aliases = get_grammar_context()["grammar"]["advisory_aliases"]
    return {
        (segment, token): str(rule["canonical"])
        for segment, segment_aliases in aliases.items()
        for token, rule in segment_aliases.items()
    }


@requires_isn
class TestSettledFoldCoverage:
    """A settled fold is grammar-derived or carries a complete reviewed resolution."""

    @pytest.mark.parametrize(
        ("token", "expected_segments", "expected_parts"),
        [
            ("heat_flux", {"channel", "physical_base"}, {"heat", "flux"}),
            ("particle_flux", {"channel", "physical_base"}, {"particle", "flux"}),
            (
                "poloidal_magnetic_flux",
                {"component", "coordinate", "physical_base"},
                {"poloidal", "magnetic_flux"},
            ),
        ],
    )
    def test_registered_modifier_and_base_compose_without_an_atomic_exception(
        self,
        token: str,
        expected_segments: set[str],
        expected_parts: set[str],
    ):
        category, segments = classify_gap("physical_base", token)
        assert category == "decomposable"
        assert expected_segments <= set(segments)

        verdict = describe_gap("physical_base", token)
        for part in expected_parts:
            assert f"'{part}'" in verdict.guidance
        assert not is_actionable_gap("physical_base", token)

    def test_reviewed_fold_artifact_has_complete_targets_and_rationales(self):
        decisions = _settled_fold_decisions()
        assert len(decisions) == 27

        for decision in decisions:
            target = decision["canonical_target"]
            rationale = decision["rationale"]
            assert isinstance(target, str) and target.strip()
            assert isinstance(rationale, str) and rationale.strip().endswith(".")

    def test_only_grammar_resolved_folds_are_non_actionable(self):
        decisions = _settled_fold_decisions()
        decision_keys = {
            (str(decision["segment"]), str(decision["token"])) for decision in decisions
        }
        expected = {
            ("physical_base", "heat_flux"): "decomposable",
            ("position", "detector"): "ambiguous_known_token",
        }
        expected.update(
            {
                alias: "reuse"
                for alias in _public_advisory_aliases()
                if alias in decision_keys
            }
        )
        actual: dict[tuple[str, str], str] = {}
        for decision in decisions:
            segment = str(decision["segment"])
            token = str(decision["token"])
            category, _segments = classify_gap(segment, token)
            if category != "absent":
                actual[(segment, token)] = category
                assert category in NON_ACTIONABLE_GAP_CATEGORIES
                assert not is_actionable_gap(segment, token)

        assert actual == expected

    def test_public_advisory_aliases_match_reviewed_fold_targets(self):
        """Each alias with a reviewed fold decision must match its target.

        A settled fold is grammar-derived or carries a complete reviewed
        resolution (see class docstring). An alias with no matching decision
        — the grammar retiring a spelling on its own authority, not from a
        human-reviewed fold — is verified below instead: it must still
        resolve as a legitimate reuse target.
        """
        reviewed_targets = {
            (str(decision["segment"]), str(decision["token"])): str(
                decision["canonical_target"]
            )
            for decision in _settled_fold_decisions()
        }
        aliases = _public_advisory_aliases()
        assert aliases
        reviewed_aliases = {
            alias: target
            for alias, target in aliases.items()
            if alias in reviewed_targets
        }
        assert reviewed_aliases
        assert {
            alias: reviewed_targets[alias] for alias in reviewed_aliases
        } == reviewed_aliases

        for (segment, token), target in reviewed_aliases.items():
            verdict = describe_gap(segment, token)
            assert verdict.category == "reuse"
            assert verdict.reuse_target == target
            assert not is_actionable_gap(segment, token)

    def test_contextual_folds_remain_explicitly_unresolved_and_actionable(self):
        mechanically_resolved = {
            ("physical_base", "heat_flux"),
            ("position", "detector"),
        }
        mechanically_resolved.update(_public_advisory_aliases())
        unresolved = 0
        decisions = _settled_fold_decisions()
        decision_keys = {
            (str(decision["segment"]), str(decision["token"])) for decision in decisions
        }
        for decision in decisions:
            segment = str(decision["segment"])
            token = str(decision["token"])
            if (segment, token) in mechanically_resolved:
                continue
            assert classify_gap(segment, token) == ("absent", [])
            assert is_actionable_gap(segment, token)
            unresolved += 1

        resolved_in_decisions = mechanically_resolved & decision_keys
        assert unresolved == len(decisions) - len(resolved_in_decisions)


@requires_isn
class TestOrdinalSamples:
    """Ordinality never enters a name, so the verdict says so instead of 'absent'."""

    @pytest.mark.parametrize(
        ("segment", "token", "target"),
        [
            ("position", "line_of_sight_second_point", "line_of_sight"),
            ("position", "line_of_sight_first_point", "line_of_sight"),
            ("position", "third_point_of_line_of_sight", "line_of_sight"),
            ("position", "lines_of_sight_second_point", "line_of_sight"),
            ("position", "conductor_start_point", "conductor"),
            ("position", "conductor_element_intermediate_point", None),
            ("position", "thick_line_first_point", None),
            ("position", "arc_of_circle_start_point", None),
            ("position", "first_point", None),
            ("position", "starting_position", "position"),
            ("component", "second_coordinate", "coordinate"),
        ],
    )
    def test_ordinal_gap_is_a_rule_violation_not_a_missing_token(
        self, segment, token, target
    ):
        category, _ = classify_gap(segment, token)
        assert category == "rule_violation"
        verdict = describe_gap(segment, token)
        assert verdict.reuse_target == target
        assert "ordinality never enters a standard name" in verdict.guidance
        if target is not None:
            assert f"'{target}'" in verdict.guidance

    def test_the_ordinal_word_is_named_so_the_composer_can_drop_it(self):
        verdict = describe_gap("position", "line_of_sight_second_point")
        assert "'second'" in verdict.guidance

    def test_an_ordinal_inside_a_registered_token_is_not_an_ordinal(self):
        """``first_wall`` is grammar vocabulary; its ``first`` is not an index."""
        assert "first_wall" in grammar_tokens_by_segment()["geometry"]
        assert ordinal_form("first_wall_point") is None
        assert classify_gap("position", "first_wall_point")[0] != "rule_violation"

    def test_an_ordinal_without_a_sampled_feature_does_not_fire(self):
        """``final`` describing a state is not an index into a repeated structure."""
        assert ordinal_form("final_state_energy") is None

    def test_the_rule_fires_on_no_registered_token(self):
        """The rule must never fire on the grammar's own vocabulary.

        Asserted against the rule directly rather than through
        :func:`classify_gap`, which would resolve each token through the ISN
        parser — tens of milliseconds each, and the whole vocabulary is 700.
        """
        offenders = sorted(
            {
                token
                for tokens in grammar_tokens_by_segment().values()
                for token in tokens
                if ordinal_form(token) is not None or dd_indexed_field_words(token)
            }
        )
        assert not offenders


@requires_isn
class TestDdCoordinateFieldNames:
    """``x1``/``x2`` name a DD slot, not a quantity — they never become gaps."""

    @pytest.mark.parametrize(
        ("segment", "token"),
        [
            ("component", "x1"),
            ("component", "x2"),
            ("geometric_base", "x1_width"),
            ("geometric_base", "x2_width"),
            ("physical_base", "x2_curvature"),
        ],
    )
    def test_indexed_field_is_a_rule_violation(self, segment, token):
        assert classify_gap(segment, token)[0] == "rule_violation"
        guidance = describe_gap(segment, token).guidance
        assert "DD coordinate slot" in guidance

    def test_a_rule_violation_never_mints_a_gap_or_retires_its_source(self):
        """Both are keyed on the category, so the write path drops it by itself."""
        assert "rule_violation" in NON_ACTIONABLE_GAP_CATEGORIES
        assert not is_actionable_gap("component", "x1")

    def test_an_isotope_suffix_is_not_a_dd_field_name(self):
        """``lithium_6`` is a genuine isotope distinction — it stays a real request."""
        assert dd_indexed_field_words("lithium_6") == ()
        assert classify_gap("subject", "lithium_6")[0] == "absent"

    def test_a_bare_index_is_never_offered_as_a_token_to_request(self):
        assert "'6'" not in describe_gap("subject", "lithium_6").guidance


@requires_isn
class TestReuseReachesTheRetryGuidance:
    """A mechanically-resolved proposal is told which registered token to use."""

    @pytest.mark.parametrize(
        ("segment", "token", "target"),
        [
            ("device", "lower_hybrid_antenna_module", "lower_hybrid_antenna"),
            ("subject", "methane_deuterated", "deuterated_methane"),
            ("physical_base", "optical_depth", "opacity"),
            ("geometric_base", "unit_vector_component", "unit_vector"),
        ],
    )
    def test_verdict_names_the_registered_target(self, segment, token, target):
        category, found = classify_gap(segment, token)
        assert category == "reuse"
        assert found  # the target's classes, so the composer knows the slot
        verdict = describe_gap(segment, token)
        assert verdict.reuse_target == target
        assert f"'{target}'" in verdict.guidance

    def test_reuse_never_mints_a_gap_or_retires_its_source(self):
        assert "reuse" in NON_ACTIONABLE_GAP_CATEGORIES
        assert not is_actionable_gap("physical_base", "optical_depth")

    def test_reuse_outranks_decomposition(self):
        """A word-order variant covers as a compound; 'compose it' keeps the wrong order."""
        assert classify_gap("subject", "methane_deuterated")[0] == "reuse"


@requires_isn
class TestSymbolShorthandInAQuotient:
    """Shorthand read as its registered token, with the division preserved."""

    @pytest.mark.parametrize(
        ("token", "left", "right"),
        [
            (
                "gradient_rho_squared_over_B_squared",
                "gradient_toroidal_flux_coordinate_squared",
                "magnetic_field_squared",
            ),
            (
                "grad_rho_squared_over_R_squared",
                "gradient_toroidal_flux_coordinate_squared",
                "major_radius_squared",
            ),
            ("velocity_over_b_field", "velocity", "magnetic_field"),
            ("vorticity_over_r", "vorticity", "major_radius"),
        ],
    )
    def test_quotient_resolves_to_two_registered_operands(self, token, left, right):
        from imas_codex.standard_names.segments import operator_composition

        comp = operator_composition(token)
        assert comp is not None, f"{token} should resolve as a quotient"
        assert comp.binary_operator is not None
        assert comp.operands == (left, right)
        assert comp.symbol_expansions

        verdict = describe_gap("physical_base", token)
        assert verdict.category == "decomposable"
        assert f"'{left}'" in verdict.guidance
        assert f"'{right}'" in verdict.guidance
        assert comp.binary_operator in verdict.guidance

    def test_the_expansion_is_shown_so_the_composer_can_check_it(self):
        guidance = describe_gap(
            "physical_base", "gradient_rho_squared_over_B_squared"
        ).guidance
        assert "'rho' as 'toroidal_flux_coordinate'" in guidance
        assert "'B' as 'magnetic_field'" in guidance

    @pytest.mark.parametrize(
        "token", ["velocity_per_magnetic_field", "vorticity_per_major_radius"]
    )
    def test_per_is_the_other_spelling_of_the_same_division(self, token):
        from imas_codex.standard_names.segments import operator_composition

        comp = operator_composition(token)
        assert comp is not None
        assert comp.binary_operator is not None

    def test_a_division_word_inside_a_registered_operator_is_not_a_division(self):
        """``per_toroidal_mode`` is one registered operator, not a quotient."""
        from imas_codex.standard_names.segments import operator_composition

        comp = operator_composition("density_per_toroidal_mode")
        assert comp is not None
        assert comp.binary_operator is None
        assert "per_toroidal_mode" in comp.operators

    def test_an_unexpandable_operand_keeps_the_gap_honest(self):
        """A quotient with a genuinely missing operand must still ask for it."""
        verdict = describe_gap("physical_base", "velocity_over_wibble_frobnicator")
        assert verdict.category == "absent"
        assert "'wibble_frobnicator'" in verdict.guidance

    def test_an_unresolved_division_is_never_folded_into_one_base(self):
        guidance = describe_gap("physical_base", "field_per_current_coupling").guidance
        assert "never fold the quotient into one base token" in guidance
        assert "'over'" not in guidance and "'per'" not in guidance

    def test_every_symbol_expansion_targets_a_registered_token(self):
        from imas_codex.standard_names.segments import (
            _symbol_expansions,
            is_known_token,
        )

        for symbol, target in _symbol_expansions().items():
            assert is_known_token(target), f"{symbol} -> {target} is unregistered"


#: Stored gap records, one per mechanism, replayed to check the shipped verdicts.
#: Drawn verbatim from the stored population (id, segment and token as written)
#: and annotated with the mechanism each one exercises.
STORED_GAP_FIXTURES: tuple[tuple[str, str, str], ...] = (
    # ordinality and DD-primitive structure
    ("ordinal", "position", "line_of_sight_second_point"),
    ("ordinal", "position", "line_of_sight_third_point"),
    ("ordinal", "position", "thick_line_second_point"),
    ("ordinal", "position", "coil_conductor_element_start_point"),
    ("ordinal", "position", "element_intermediate_point"),
    ("ordinal", "position", "laser_end_point"),
    ("ordinal", "position", "second_point"),
    ("ordinal", "object", "arc_start_point"),
    ("ordinal", "qualifier", "first_coordinate"),
    ("dd_field", "component", "x1"),
    ("dd_field", "component", "x2"),
    ("dd_field", "geometric_base", "x1_width"),
    # reuse of a registered token
    ("reuse", "device", "lower_hybrid_antenna_module"),
    ("reuse", "device", "ion_cyclotron_heating_antenna_module"),
    ("reuse", "device", "hard_xray_channel"),
    ("reuse", "device", "mse_channel"),
    ("reuse", "object", "vessel_element"),
    ("reuse", "physical_base", "volume_element"),
    ("reuse", "physical_base", "optical_depth"),
    ("reuse", "geometric_base", "unit_vector_component"),
    ("reuse", "subject", "methane_deuterated"),
    # quotients and products written as one word
    ("quotient", "physical_base", "vorticity_over_r"),
    ("quotient", "physical_base", "velocity_over_b_field"),
    ("quotient", "physical_base", "gradient_rho_squared_over_B_squared"),
    ("quotient", "physical_base", "grad_rho_squared_over_R_squared"),
    ("quotient", "physical_base", "velocity_per_magnetic_field"),
    ("quotient", "physical_base", "vorticity_per_major_radius"),
    (
        "quotient",
        "physical_base",
        "flux_surface_averaged_squared_radial_gradient_over_square_magnetic_field",
    ),
    # genuine narrow demand — must stay a real request
    ("genuine", "physical_base", "detection_efficiency"),
    ("added", "physical_base", "focal_length"),
    ("genuine", "physical_base", "groove_density"),
    ("genuine", "physical_base", "substrate"),
    ("genuine", "subject", "lithium_6"),
    ("reuse", "position", "rectangle_centre"),
)


@requires_isn
class TestStoredGapReplay:
    """Replaying stored tokens: every settled mechanism resolves, demand does not."""

    @pytest.mark.parametrize(
        ("mechanism", "segment", "token"),
        [(m, s, t) for m, s, t in STORED_GAP_FIXTURES if m != "genuine"],
    )
    def test_a_settled_mechanism_never_replays_as_absent(
        self, mechanism, segment, token
    ):
        category = classify_gap(segment, token)[0]
        assert category != "absent", f"{token} ({mechanism}) still reads as absent"
        assert category in NON_ACTIONABLE_GAP_CATEGORIES

    @pytest.mark.parametrize(
        ("segment", "token"),
        [(s, t) for m, s, t in STORED_GAP_FIXTURES if m == "genuine"],
    )
    def test_genuine_demand_still_asks_for_a_token(self, segment, token):
        assert classify_gap(segment, token)[0] == "absent"
        assert is_actionable_gap(segment, token)

    @pytest.mark.parametrize(("mechanism", "segment", "token"), STORED_GAP_FIXTURES)
    def test_every_verdict_carries_guidance(self, mechanism, segment, token):
        guidance = describe_gap(segment, token).guidance
        assert guidance and token in guidance

    def test_a_control_token_stays_absent(self):
        """Nothing here may turn an invented word into a resolvable one."""
        for token in ("wibble_frobnicator", "zzz_nonexistent_quantity"):
            assert classify_gap("physical_base", token)[0] == "absent"
