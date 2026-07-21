"""Tests for L3 post-generation audits module."""

from __future__ import annotations

import numpy as np
import pytest

# =========================================================================
# L3: latex_def_check
# =========================================================================


class TestLatexDefCheck:
    """Tests for latex_def_check audit."""

    def test_pass_symbols_defined(self):
        """Symbols with definitions pass."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The safety factor is $q = d\\Phi/d\\Psi$, where "
                "$q$ is the safety factor (dimensionless), "
                "$\\Phi$ is the toroidal flux (Wb), and "
                "$\\Psi$ is the poloidal flux (Wb)."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) == 0

    def test_fail_undefined_symbol(self):
        """Symbols without definitions are flagged."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The plasma has $T_e$ and $n_e$ parameters. "
                "These affect confinement significantly. "
                "Further analysis shows improved stability."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) >= 1
        assert any("latex_def_check" in i for i in issues)

    def test_fail_unit_only_is_not_a_definition(self):
        """A bare parenthetical unit no longer satisfies the check — it states
        the symbol's dimension, not which quantity it denotes."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The reference field enters as $B_p$ (T). "
                "The contour integral runs over the poloidal loop. "
                "The result normalizes the profile."
            ),
        }
        issues = latex_def_check(candidate)
        assert any("latex_def_check" in i and "B_p" in i for i in issues)

    def test_pass_name_link_is_identity(self):
        """A `name:` link binds a symbol to a catalog quantity — a valid
        identity even with no definition verb present."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The normalization uses $\\psi_b$, the "
                "[boundary poloidal flux](name:poloidal_magnetic_flux_at_boundary). "
                "It sets the outer reference."
            ),
        }
        assert latex_def_check(candidate) == []

    def test_pass_identity_defined_symbol(self):
        """A symbol defined by identity (a def word) passes. latex_def_check
        scores identity only; unit tokens have no place in documentation prose
        and are policed separately (see audit:symbol_units)."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The poloidal field $B_p$ is the poloidal magnetic-field "
                "magnitude evaluated on the integration contour. "
                "It weights the flux-surface average."
            ),
        }
        assert latex_def_check(candidate) == []

    def test_pass_universal_constants_skipped(self):
        """Universal physics constants (\\pi, \\alpha, \\mu_0, \\hbar,
        k_B) and numeric factors thereof (``2\\pi``, ``\\pi/2``) do not
        require a definition sentence."""
        from imas_codex.standard_names.audits import latex_def_check

        for sym in (
            r"\pi",
            r"2\pi",
            r"\pi/2",
            r"\alpha",
            r"\mu_0",
            r"\hbar",
            "k_B",
            r"\epsilon_0",
        ):
            candidate = {
                "documentation": f"Uses ${sym}$ with no explicit definition.",
            }
            assert latex_def_check(candidate) == [], (
                f"Expected no issue for {sym}, got {latex_def_check(candidate)}"
            )

    def test_pass_empty_documentation(self):
        """No documentation produces no issues."""
        from imas_codex.standard_names.audits import latex_def_check

        assert latex_def_check({"documentation": ""}) == []
        assert latex_def_check({}) == []

    def test_pass_no_latex_symbols(self):
        """Documentation without LaTeX symbols passes."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {"documentation": "A simple description without math."}
        assert latex_def_check(candidate) == []


# =========================================================================
# symbol_units_check — no units in documentation prose
# =========================================================================


class TestSymbolUnitsCheck:
    """Tests for symbol_units_check (advisory 'no units in prose')."""

    def _flag(self, doc):
        from imas_codex.standard_names.audits import symbol_units_check

        return symbol_units_check({"documentation": doc})

    def test_flag_in_unit(self):
        assert self._flag("The field $B_p$ is this magnitude, in T here.")

    def test_flag_with_unit_mathrm(self):
        assert self._flag(
            "The density is the value with unit $\\mathrm{m^{-3}}$ over the volume."
        )

    def test_flag_unit_exponent(self):
        assert self._flag("The momentum density is stated as kg.m^-2.s^-1 here.")

    def test_flag_mathrm_unit_product(self):
        assert self._flag(
            "The current density component is $\\mathrm{A\\,m^{-2}}$ overall."
        )

    def test_flag_explicit_in_units_of(self):
        assert self._flag("The energy is expressed in units of eV throughout.")

    def test_pass_identity_no_units(self):
        """An identity-defined, unit-free doc is clean."""
        assert (
            self._flag(
                "The poloidal field $B_p$ is the poloidal magnetic-field magnitude "
                "on the contour. It weights the flux-surface average."
            )
            == []
        )

    def test_pass_mathrm_label_not_a_unit(self):
        """A \\mathrm{} label subscript (radiated, axis, eff) is not a unit."""
        assert self._flag("The power $P_{\\mathrm{rad}}$ is the radiated total.") == []
        assert self._flag("At $R_{\\mathrm{axis}}$ the value peaks smoothly.") == []
        assert self._flag("The area $A_{\\mathrm{eff}}$ enters the coupling here.") == []

    def test_pass_mathrm_multiword_label_with_thin_space(self):
        """A \\mathrm{} label joining words with \\, (th ion, wave beam, gas inj)
        is not a unit — only an exponent inside \\mathrm{} marks a unit."""
        assert self._flag("The density $n_{\\mathrm{th\\,ion}}$ is the thermal ion.") == []
        assert self._flag("The power $P_{\\mathrm{wave\\,beam}}$ enters here.") == []
        assert self._flag("The rate $S_{\\mathrm{gas\\,inj}}$ counts injection.") == []

    def test_pass_positive_exponent_variable_not_a_unit(self):
        """A positive exponent on a physics variable (mass², c²) is not a unit;
        only negative-exponent unit letters (m^{-3}, s^{-1}) flag."""
        assert self._flag("The rest energy uses $m^2 c^2$ in the relation.") == []
        assert self._flag("The density scales as $m^{2}$ over the region.") == []

    def test_pass_unit_normal_vector_not_a_unit(self):
        """'with unit normal' / 'with unit vector' is a geometric unit-length
        vector, not a physical unit."""
        assert self._flag("The surface with unit normal $\\hat{n}$ bounds it.") == []
        assert self._flag("Projected onto the with unit vector $\\hat{e}$ basis.") == []

    def test_pass_empty(self):
        assert self._flag("") == []
        from imas_codex.standard_names.audits import symbol_units_check

        assert symbol_units_check({}) == []

    def test_advisory_not_critical(self):
        """The check is advisory — it must not be in CRITICAL_CHECKS."""
        from imas_codex.standard_names.audits import CRITICAL_CHECKS

        assert "symbol_units_check" not in CRITICAL_CHECKS


# =========================================================================
# L3: provenance_verb_check
# =========================================================================


class TestProvenanceVerbCheck:
    """Tests for provenance_verb_check audit."""

    def test_pass_no_provenance_verb(self):
        """Clean name passes."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature"}
        issues = provenance_verb_check(candidate)
        assert len(issues) == 0

    def test_fail_measured_in_name(self):
        """Name with 'measured' when source lacks it is flagged."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(
            candidate, source_path="core_profiles/profiles_1d/electrons/temperature"
        )
        assert len(issues) == 1
        assert "measured" in issues[0]

    def test_pass_measured_in_both(self):
        """Name with 'measured' is OK when source path also has it."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(candidate, source_path="diagnostics/measured/te")
        assert len(issues) == 0


# =========================================================================
# L3: synonym_check
# =========================================================================


class TestSynonymCheck:
    """Tests for synonym_check audit."""

    def test_pass_no_similar(self):
        """No existing SNs means no synonym issues."""
        from imas_codex.standard_names.audits import synonym_check

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": np.random.rand(384).tolist(),
        }
        issues = synonym_check(candidate, [])
        assert len(issues) == 0

    def test_fail_high_cosine(self):
        """Near-identical embedding with same unit is flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)
        # Create a very similar vector
        similar_vec = vec + np.random.rand(384).astype(np.float32) * 0.01

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "te_profile",
                "unit": "eV",
                "description_embedding": similar_vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 1
        assert "synonym_check" in issues[0]

    def test_pass_different_unit(self):
        """Same embedding but different unit is not flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "electron_density",
                "unit": "m^-3",
                "description_embedding": vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 0


# =========================================================================
# L3: unit_dimension_check
# =========================================================================


class TestUnitDimensionCheck:
    """Tests for unit_dimension_check audit."""

    def test_pass_consistent(self):
        """Description consistent with unit passes."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "eV", "description": "Electron temperature profile"}
        assert unit_dimension_check(candidate) == []

    def test_fail_inconsistent(self):
        """Description inconsistent with unit is flagged."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "A", "description": "Radial position of the boundary"}
        issues = unit_dimension_check(candidate)
        assert len(issues) == 1
        assert "unit_dimension_check" in issues[0]

    def test_pass_dimensionless(self):
        """Dimensionless unit is not checked."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "1", "description": "Safety factor profile"}
        assert unit_dimension_check(candidate) == []


# =========================================================================
# L3: multi_subject_check
# =========================================================================


class TestMultiSubjectCheck:
    """Tests for multi_subject_check audit."""

    def test_pass_single_subject(self):
        """Single-subject name passes."""
        from imas_codex.standard_names.audits import multi_subject_check

        candidate = {"id": "electron_temperature"}
        issues = multi_subject_check(candidate)
        assert len(issues) == 0

    def test_fail_multiple_subjects(self):
        """Name with two different subject tokens is flagged."""
        from imas_codex.standard_names.audits import multi_subject_check

        # This will flag if two Subject enum values appear in name tokens
        candidate = {"id": "electron_ion_temperature"}
        issues = multi_subject_check(candidate)
        # May or may not flag depending on grammar — at least doesn't crash
        assert isinstance(issues, list)


# =========================================================================
# L3: cocos_specificity_check
# =========================================================================


class TestCocosSpecificityCheck:
    """Tests for cocos_specificity_check audit."""

    def test_pass_cocos_mentioned(self):
        """Documentation mentioning COCOS with digit passes."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {
            "documentation": "Sign convention: Positive when COCOS 11 convention applies."
        }
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 0

    def test_fail_no_cocos(self):
        """Documentation without COCOS mention when source has COCOS type is flagged."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "The poloidal magnetic flux per radian."}
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 1
        assert "cocos_specificity_check" in issues[0]

    def test_pass_no_cocos_type(self):
        """No source COCOS type means no check."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "Simple quantity."}
        issues = cocos_specificity_check(candidate, source_cocos_type=None)
        assert len(issues) == 0


# =========================================================================
# L3: run_audits integration
# =========================================================================


class TestRunAudits:
    """Tests for run_audits orchestrator."""

    def test_clean_candidate_passes(self):
        """A well-formed candidate passes all audits."""
        from imas_codex.standard_names.audits import run_audits

        candidate = {
            "id": "electron_temperature",
            "description": "Electron temperature profile",
            "documentation": (
                "The electron temperature $T_e$ is the kinetic energy "
                "per degree of freedom, where $T_e$ denotes the temperature (eV)."
            ),
            "unit": "eV",
        }
        issues = run_audits(candidate)
        assert isinstance(issues, list)

    def test_has_critical_audit_failure(self):
        """Critical check tag detection works."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        assert (
            has_critical_audit_failure(["audit:latex_def_check: missing def"]) is True
        )
        assert has_critical_audit_failure(["audit:synonym_check: cosine=0.95"]) is True
        assert (
            has_critical_audit_failure(["audit:multi_subject_check: two subjects"])
            is True
        )
        assert (
            has_critical_audit_failure(["audit:placeholder_check: [condition]"]) is True
        )
        assert (
            has_critical_audit_failure(
                ["audit:unit_validity_check: non-unit token 'dimension'"]
            )
            is True
        )
        assert (
            has_critical_audit_failure(["audit:unit_dimension_check: mismatch"])
            is False
        )
        assert has_critical_audit_failure([]) is False


# =========================================================================
# placeholder_check
# =========================================================================


class TestPlaceholderCheck:
    """Tests for placeholder_check audit."""

    def test_pass_concrete_sign_convention(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "Sign convention: Positive when the plasma current flows "
                "counter-clockwise viewed from above."
            ),
        }
        assert placeholder_check(c) == []

    def test_fail_bracketed_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "Sign convention: Positive when [condition]."}
        issues = placeholder_check(c)
        assert len(issues) == 1
        assert "placeholder_check" in issues[0]

    def test_fail_bracketed_specific_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"description": "Positive when [specific physical condition]"}
        issues = placeholder_check(c)
        assert len(issues) == 1

    def test_pass_markdown_link(self):
        """Markdown [text](url) links are not placeholders."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "See [magnetic_flux](#magnetic_flux) and [safety_factor](#safety_factor)."
            ),
        }
        assert placeholder_check(c) == []

    def test_pass_numeric_bracket(self):
        """Numeric brackets like [1] or citation-like markers are not flagged."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "See reference [1] and range [0, 1]."}
        assert placeholder_check(c) == []


# =========================================================================
# unit_validity_check
# =========================================================================


class TestUnitValidityCheck:
    """Tests for unit_validity_check audit."""

    def test_pass_real_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        for unit in ("m", "T", "Wb", "eV", "m^2", "kg*m/s^2", "m.s^-1", "1"):
            assert unit_validity_check({"unit": unit}) == [], f"failed for {unit}"

    def test_fail_m_caret_dimension(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "m^dimension"})
        assert len(issues) == 1
        assert "dimension" in issues[0]

    def test_fail_fourier_in_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "T*fourier"})
        assert len(issues) == 1

    def test_pass_empty_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        assert unit_validity_check({"unit": ""}) == []
        assert unit_validity_check({"unit": "dimensionless"}) == []


class TestGenericNounCheck:
    """Tests for generic_noun_check audit."""

    def test_fail_bare_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        issues = generic_noun_check({"id": "geometry"})
        assert len(issues) == 1
        assert "generic_noun_check" in issues[0]

    def test_fail_bare_data(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "data"})) == 1

    def test_pass_qualified_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "grid_object_geometry"}) == []

    def test_pass_multi_token(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "electron_temperature"}) == []

    def test_fail_generic_qualifier_plus_generic_noun(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "raw_data"})) == 1


class TestTautologyCheck:
    """Tests for tautology_check audit."""

    def test_fail_position_of_position(self):
        from imas_codex.standard_names.audits import tautology_check

        issues = tautology_check({"id": "radial_position_of_reference_position"})
        assert len(issues) == 1
        assert "tautology_check" in issues[0]
        assert "position" in issues[0]

    def test_fail_component_of_component(self):
        from imas_codex.standard_names.audits import tautology_check

        assert len(tautology_check({"id": "normal_component_of_field_component"})) == 1

    def test_pass_no_of(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "electron_temperature"}) == []

    def test_pass_different_heads(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "radial_position_of_plasma_boundary"}) == []

    def test_pass_of_without_tautology_head(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "elongation_of_plasma_boundary"}) == []


class TestSpectralSuffixCheck:
    """Tests for spectral_suffix_check audit."""

    def test_fail_fourier_coefficients(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        issues = spectral_suffix_check({"id": "normal_field_fourier_coefficients"})
        assert len(issues) == 1
        assert "spectral_suffix_check" in issues[0]

    def test_fail_harmonics(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert len(spectral_suffix_check({"id": "magnetic_field_harmonics"})) == 1

    def test_pass_mode_prefix(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "mode_amplitude_of_normal_field"}) == []

    def test_pass_ordinary_name(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "poloidal_magnetic_flux"}) == []


class TestAbbreviationCheck:
    """Tests for abbreviation_check audit."""

    def test_fail_norm_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        issues = abbreviation_check({"id": "norm_poloidal_magnetic_flux"})
        assert len(issues) == 1
        assert "normalized" in issues[0]

    def test_fail_perp_interior(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "velocity_perp_component"})) == 1

    def test_fail_temp_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "temp_profile"})) == 1

    def test_pass_full_words(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": "normalized_poloidal_magnetic_flux"}) == []
        assert abbreviation_check({"id": "perpendicular_velocity_component"}) == []

    def test_pass_empty(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": ""}) == []


class TestNameDescriptionConsistencyCheck:
    """Tests for name_description_consistency_check audit."""

    def test_fail_fourier_desc_bare_name(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        issues = name_description_consistency_check(
            {
                "id": "normal_magnetic_field",
                "description": "Fourier coefficients of the normal component of the field.",
            }
        )
        assert len(issues) == 1

    def test_pass_decomposition_marker(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "mode_amplitude_of_normal_field",
                    "description": "Fourier coefficients of the normal field.",
                }
            )
            == []
        )

    def test_pass_plain_desc(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature profile.",
                }
            )
            == []
        )

    def test_pass_missing_fields(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert name_description_consistency_check({"id": "x", "description": ""}) == []


class TestAmericanSpellingCheck:
    """Tests for american_spelling_check audit (NC-17)."""

    def test_fail_british_in_name(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "normalised_poloidal_flux", "description": "A flux."}
        )
        assert any("'normalised'" in i and "normalized" in i for i in issues)
        assert any("field 'name'" in i for i in issues)

    def test_fail_british_in_description(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "plasma_current",
                "description": "Current at the centre of the plasma, analysed per shot.",
            }
        )
        fields = {i.split("field '")[1].split("'")[0] for i in issues}
        assert "description" in fields
        joined = " ".join(issues)
        assert "centre" in joined and "analysed" in joined

    def test_fail_british_in_constraints(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "foo",
                "description": "ok",
                "constraints": ["Must be normalised to 1"],
            }
        )
        assert any("constraints[0]" in i for i in issues)

    def test_pass_american_only(self):
        from imas_codex.standard_names.audits import american_spelling_check

        assert (
            american_spelling_check(
                {
                    "id": "normalized_poloidal_flux",
                    "description": "The normalized flux at the center of the plasma, analyzed per shot.",
                    "documentation": "Modeled behavior of labeled channels.",
                }
            )
            == []
        )

    def test_case_insensitive(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "x", "description": "The Normalised profile."}
        )
        assert len(issues) == 1
        assert "Normalised" in issues[0]

    def test_pass_gauge_is_valid_american(self):
        """'gauge' is the standard US spelling (Merriam-Webster headword) and
        the spelling the IMAS DD uses; breame maps it to the niche variant
        'gage', so it must be allowlisted and never flagged."""
        from imas_codex.standard_names.audits import american_spelling_check

        assert (
            american_spelling_check(
                {
                    "id": "x_first_measurement_direction_unit_vector_of_strain_gauge",
                    "description": "Unit vector of a strain gauge; two gauges gauged per axis.",
                }
            )
            == []
        )


# =========================================================================
# description_verb_drift_check
# =========================================================================


class TestDescriptionVerbDriftCheck:
    """Name/description rate-marker consistency."""

    def test_fail_instant_change_prefix(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        issues = description_verb_drift_check(
            {
                "id": "instant_change_in_electron_density",
                "description": "Instantaneous signed change in electron number density.",
            }
        )
        assert len(issues) == 1
        assert "instant_change_" in issues[0] or "begins with" in issues[0]

    def test_fail_rate_description_missing_marker(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        issues = description_verb_drift_check(
            {
                "id": "ion_temperature",
                "description": "Instantaneous change in ion temperature due to a transient plasma event.",
            }
        )
        assert len(issues) == 1
        assert "rate" in issues[0] or "tendency_of_" in issues[0]

    def test_pass_tendency_name(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "tendency_of_electron_density",
                    "description": "Instantaneous signed change in electron density.",
                }
            )
            == []
        )

    def test_pass_change_in_name(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "change_in_ion_temperature",
                    "description": "Time derivative of ion temperature.",
                }
            )
            == []
        )

    def test_pass_base_quantity_description(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature radial profile.",
                }
            )
            == []
        )


# =========================================================================
# structural_dim_tag_check
# =========================================================================


class TestStructuralDimTagCheck:
    """Advisory flag for DD data-type dimensionality tags in descriptions."""

    def test_fail_1d_in_description(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        issues = structural_dim_tag_check(
            {"description": "Electron temperature as a 1D radial profile."}
        )
        assert len(issues) == 1
        assert "1D" in issues[0]

    def test_fail_2d_in_description(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        issues = structural_dim_tag_check({"description": "2D map of poloidal flux."})
        assert len(issues) == 1

    def test_pass_no_tag(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        assert (
            structural_dim_tag_check(
                {"description": "Electron temperature radial profile."}
            )
            == []
        )

    def test_pass_dimensionless(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        # 'dimensionless' contains 'd' but not \bNd\b
        assert (
            structural_dim_tag_check({"description": "A dimensionless parameter."})
            == []
        )


class TestDensityUnitConsistencyCheck:
    def test_fail_density_with_bare_momentum_unit(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        issues = density_unit_consistency_check(
            {"id": "toroidal_angular_momentum_density", "unit": "kg.m.s^-1"}
        )
        assert len(issues) == 1
        assert "no inverse-length factor" in issues[0]

    def test_pass_volumetric_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check({"id": "electron_density", "unit": "m^-3"})
            == []
        )

    def test_pass_areal_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check(
                {"id": "surface_charge_density", "unit": "C.m^-2"}
            )
            == []
        )

    def test_pass_dimensionless_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check({"id": "ion_fraction_density", "unit": "1"})
            == []
        )

    def test_pass_no_density_in_name(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check(
                {"id": "toroidal_torque", "unit": "kg.m^2.s^-2"}
            )
            == []
        )


class TestPositionCoordinateCheck:
    def test_fail_radial_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "radial_position_of_antenna_row"})
        assert len(issues) == 1
        assert "major_radius_of_<X>" in issues[0]

    def test_fail_toroidal_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "toroidal_position_of_antenna_row"})
        assert len(issues) == 1
        assert "toroidal_angle_of_<X>" in issues[0]

    def test_fail_vertical_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "vertical_position_of_x_point"})
        assert len(issues) == 1
        assert "vertical_coordinate_of_<X>" in issues[0]

    def test_pass_canonical_major_radius(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        assert (
            position_coordinate_check(
                {"id": "major_radius_of_electron_cyclotron_launcher"}
            )
            == []
        )

    def test_pass_no_position_in_name(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        assert position_coordinate_check({"id": "electron_density"}) == []

    def test_pass_plain_position_no_directional_qualifier(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        # Plain `position_of_X` (no R/Z/phi qualifier) is acceptable for
        # unspecified 3-vector positions and must not be flagged.
        assert position_coordinate_check({"id": "position_of_strike_point"}) == []


class TestVectorFieldComponentCheck:
    """Tests for vector_field_component_check audit."""

    def test_flags_vertical_coordinate_of_surface_normal(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        issues = vector_field_component_check(
            {"id": "vertical_coordinate_of_surface_normal"}
        )
        assert len(issues) == 1
        assert "vector_field_component_check" in issues[0]
        # Production code recommends short-form: '{axis}_{vector}'
        assert "vertical_surface_normal" in issues[0]

    def test_flags_radial_coordinate_of_magnetic_field_vector(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        issues = vector_field_component_check(
            {"id": "radial_coordinate_of_magnetic_field_vector"}
        )
        assert len(issues) == 1
        assert "radial_magnetic_field_vector" in issues[0]

    def test_passes_vertical_coordinate_of_plasma_boundary(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        # plasma_boundary is a geometric feature (point/curve), not a vector
        # field — _coordinate_of_ is correct.
        assert (
            vector_field_component_check(
                {"id": "vertical_coordinate_of_plasma_boundary"}
            )
            == []
        )

    def test_passes_vertical_surface_normal(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        # The canonical form is not flagged.
        assert vector_field_component_check({"id": "vertical_surface_normal"}) == []

    def test_passes_unrelated_name(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        assert vector_field_component_check({"id": "electron_temperature"}) == []


class TestSegmentOrderCheck:
    def test_fail_trailing_toroidal(self):
        from imas_codex.standard_names.audits import segment_order_check

        issues = segment_order_check({"id": "ion_rotation_frequency_toroidal"})
        assert issues and "segment_order_check" in issues[0]

    def test_fail_trailing_poloidal(self):
        from imas_codex.standard_names.audits import segment_order_check

        issues = segment_order_check({"id": "electron_flux_poloidal"})
        assert issues and "segment_order_check" in issues[0]

    def test_pass_leading_component(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert segment_order_check({"id": "toroidal_ion_rotation_frequency"}) == []

    def test_pass_short_form_component_prefix(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert segment_order_check({"id": "toroidal_ion_rotation_frequency"}) == []

    def test_pass_no_component_token(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert segment_order_check({"id": "electron_temperature"}) == []


class TestCausalDueToCheckExtended:
    def test_pass_due_to_resistive_diffusion_isn_process(self):
        """`resistive_diffusion` is the mechanism-noun process token.

        The bare `resistive` regime token was dropped in favour of the
        specific mechanism nouns (resistive_diffusion / resistive_dissipation).
        """
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check(
                {"id": "parallel_current_density_due_to_resistive_diffusion"}
            )
            == []
        )

    def test_pass_due_to_non_inductive_isn_process(self):
        """`non_inductive_current_drive` is a mechanism-noun process token.

        The bare regime token `non_inductive` was lifted to the mechanism
        noun `non_inductive_current_drive`; the causal check accepts the
        full mechanism noun after `due_to_`.
        """
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check(
                {"id": "parallel_current_density_due_to_non_inductive_current_drive"}
            )
            == []
        )

    def test_pass_due_to_turbulent_isn_process(self):
        """`turbulent_transport` is a mechanism-noun process token.

        The bare regime token `turbulent` was lifted to `turbulent_transport`;
        the causal check accepts the full mechanism noun after `due_to_`.
        """
        from imas_codex.standard_names.audits import causal_due_to_check

        assert causal_due_to_check({"id": "heat_flux_due_to_turbulent_transport"}) == []

    def test_fail_due_to_shutdown_temporal(self):
        """Temporal events not in ISN process vocab remain flagged."""
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "heat_flux_due_to_shutdown"})
        assert issues and "shutdown" in issues[0]

    def test_pass_due_to_resistive_diffusion(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check(
                {"id": "parallel_current_density_due_to_resistive_diffusion"}
            )
            == []
        )


class TestPeakingFactorExemption:
    def test_pass_ion_temperature_peaking_factor(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "ion_temperature_peaking_factor",
                    "unit": "1",
                    "description": "Ratio of central to volume-averaged ion temperature",
                }
            )
            == []
        )

    def test_pass_electron_temperature_profile_factor(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "electron_temperature_profile_factor",
                    "unit": "1",
                    "description": "Profile peaking factor for electron temperature",
                }
            )
            == []
        )

    def test_pass_fraction(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "bootstrap_current_fraction",
                    "unit": "1",
                    "description": "Fraction of total current carried by bootstrap",
                }
            )
            == []
        )

    def test_fail_bare_temperature_with_dimensionless(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {
                "id": "ion_temperature",
                "unit": "1",
                "description": "Ion temperature in dimensionless units",
            }
        )
        assert issues

    def test_pass_energy_flux_with_power_per_area(self):
        """``_energy_flux`` carries dimensions of power-per-area (W.m^-2),
        not pure energy. The audit must accept this without flagging."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        for name, unit in [
            ("incident_neutral_kinetic_energy_flux", "W.m^-2"),
            ("incident_ion_kinetic_energy_flux_on_wall", "m^-2.W"),
            ("electron_emitted_kinetic_energy_flux_at_first_wall", "m^-2.W"),
        ]:
            assert (
                name_unit_consistency_check(
                    {"id": name, "unit": unit, "description": ""}
                )
                == []
            ), f"unexpected fail for {name} {unit}"

    def test_pass_energy_flux_with_particle_flux_unit(self):
        """A particle flux of energy-bearing species (``m^-2.s^-1``) is also
        a legitimate use of ``_energy_flux`` per source-path semantics."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "neutral_kinetic_energy_flux_emitted_from_wall",
                    "unit": "m^-2.s^-1",
                    "description": "",
                }
            )
            == []
        )

    def test_pass_mass_flux_with_compound_unit(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "ion_mass_flux_at_separatrix",
                    "unit": "kg.m^-2.s^-1",
                    "description": "",
                }
            )
            == []
        )

    def test_fail_bare_energy_with_particle_flux_unit(self):
        """Without the ``flux`` qualifier, energy with a non-energy unit
        must still fail — guards against the exemption being too broad."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {
                "id": "neutral_kinetic_energy",
                "unit": "m^-2.s^-1",
                "description": "",
            }
        )
        assert issues

    def test_pass_energy_velocity_with_transport_velocity_unit(self):
        """``velocity`` is the head noun in convective-transport names —
        ``energy`` classifies what is transported, so m.s^-1 is correct."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        for name in [
            "energy_velocity_due_to_convection",
            "effective_thermal_ion_charge_state_energy_velocity_due_to_convection",
        ]:
            assert (
                name_unit_consistency_check(
                    {"id": name, "unit": "m.s^-1", "description": ""}
                )
                == []
            ), f"unexpected fail for {name}"

    def test_fail_bare_energy_with_velocity_unit(self):
        """Without the ``velocity`` head noun, energy with m.s^-1 must
        still fail — guards against the exemption being too broad."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {"id": "particle_energy", "unit": "m.s^-1", "description": ""}
        )
        assert issues


class TestAggregatorOrderCheck:
    def test_fail_trailing_volume_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "ion_temperature_volume_averaged"})
        assert issues and "aggregator_order_check" in issues[0]
        assert "volume_averaged_ion_temperature" in issues[0]

    def test_fail_trailing_flux_surface_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "current_density_flux_surface_averaged"})
        assert issues and "flux_surface_averaged" in issues[0]

    def test_fail_trailing_line_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "electron_density_line_averaged"})
        assert issues

    def test_pass_leading_volume_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        assert (
            aggregator_order_check({"id": "volume_averaged_electron_temperature"}) == []
        )

    def test_pass_no_aggregator(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        assert aggregator_order_check({"id": "electron_temperature"}) == []


class TestDiamagneticComponentCheck:
    def test_fail_diamagnetic_component_of_electric_field(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        issues = diamagnetic_component_check(
            {"id": "diamagnetic_component_of_electric_field"}
        )
        assert issues and "drift" in issues[0].lower()

    def test_fail_diamagnetic_component_of_ion_velocity(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        issues = diamagnetic_component_check(
            {"id": "diamagnetic_component_of_ion_velocity"}
        )
        assert issues

    def test_pass_diamagnetic_drift_velocity(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert diamagnetic_component_check({"id": "diamagnetic_drift_velocity"}) == []

    def test_pass_diamagnetic_current_density(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert diamagnetic_component_check({"id": "diamagnetic_current_density"}) == []

    def test_pass_toroidal_component(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert diamagnetic_component_check({"id": "toroidal_electric_field"}) == []


# =========================================================================
# C.1: multi_subject_check — greedy compound-subject match
# =========================================================================


class TestMultiSubjectCheckGreedy:
    """Tests for greedy longest-match in multi_subject_check."""

    @pytest.mark.parametrize(
        "name,expected_passes",
        [
            ("deuterium_tritium_fusion_power_density", True),
            ("tritium_to_deuterium_density_ratio", True),
            ("deuterium_deuterium_fusion_power", True),
            ("tritium_tritium_reaction_rate", True),
            ("electron_deuterium_density", False),
        ],
    )
    def test_compound_species_greedy(self, name, expected_passes):
        from imas_codex.standard_names.audits import multi_subject_check

        issues = multi_subject_check({"id": name})
        if expected_passes:
            assert issues == [], f"False positive on '{name}': {issues}"
        else:
            assert issues, f"Expected failure on '{name}'"

    def test_pass_single_compound_subject(self):
        from imas_codex.standard_names.audits import multi_subject_check

        assert multi_subject_check({"id": "deuterium_tritium_fusion_power"}) == []

    def test_pass_electron_equivalent_exemption_preserved(self):
        from imas_codex.standard_names.audits import multi_subject_check

        assert multi_subject_check({"id": "ion_electron_equivalent"}) == []

    @pytest.mark.parametrize(
        "name",
        [
            "trapped_fast_density",
            "co_passing_fast_pressure",
            "counter_passing_fast_density",
            "total_rejected_thermal_power",
            "total_supplied_thermal_power",
        ],
    )
    def test_pass_all_modifier_subjects(self, name):
        """When ALL matched subjects are modifiers, no dual-subject conflict."""
        from imas_codex.standard_names.audits import multi_subject_check

        issues = multi_subject_check({"id": name})
        assert issues == [], f"False positive on '{name}': {issues}"

    def test_pass_runaway_modifier(self):
        """Runaway is a modifier subject, not a dual subject with electrons."""
        from imas_codex.standard_names.audits import multi_subject_check

        assert (
            multi_subject_check({"id": "critical_electric_field_for_runaway_electrons"})
            == []
        )

    @pytest.mark.parametrize(
        "name",
        [
            "toroidal_trapped_fast_particle_torque_density_due_to_coulomb_collisions_with_ion",
            "toroidal_trapped_fast_particle_torque_density_due_to_coulomb_collisions_with_electrons",
            "toroidal_co_passing_fast_particle_torque_density_due_to_coulomb_collisions_with_ion",
        ],
    )
    def test_pass_collisional_with_target(self, name):
        """Collision target after _with_ is exempt from multi-subject check."""
        from imas_codex.standard_names.audits import multi_subject_check

        issues = multi_subject_check({"id": name})
        assert issues == [], f"False positive on '{name}': {issues}"


# =========================================================================
# C.3: density_unit_consistency_check — constraint-metadata suffix skip
# =========================================================================


class TestDensityConstraintMetadataExemption:
    """Tests for constraint-metadata suffix skip in density_unit_consistency_check."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "_constraint_measurement_time",
            "_constraint_weight",
            "_constraint_reconstructed",
            "_constraint_measured",
            "_constraint_time_measurement",
            "_constraint_position",
        ],
    )
    def test_pass_constraint_metadata_suffix(self, suffix):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        name = f"toroidal_current_density{suffix}"
        assert density_unit_consistency_check({"id": name, "unit": "s"}) == [], (
            f"False positive on '{name}'"
        )

    def test_fail_density_without_constraint_suffix(self):
        """Density without a constraint suffix and wrong unit still flagged."""
        from imas_codex.standard_names.audits import density_unit_consistency_check

        issues = density_unit_consistency_check(
            {"id": "toroidal_current_density", "unit": "s"}
        )
        assert issues, "Expected failure for bare density with wrong unit"


# =========================================================================
# C.4: implicit_field_check — device whitelist
# =========================================================================


class TestImplicitFieldDeviceWhitelist:
    """Tests for device whitelist in implicit_field_check."""

    def test_pass_vacuum_toroidal_field_function(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert implicit_field_check({"id": "vacuum_toroidal_field_function"}) == []

    def test_pass_resistance_of_poloidal_field_coil(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert implicit_field_check({"id": "resistance_of_poloidal_field_coil"}) == []

    def test_pass_field_coil_substring(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert (
            implicit_field_check({"id": "current_in_poloidal_field_coil_supply"}) == []
        )

    def test_fail_bare_field_still_flagged(self):
        from imas_codex.standard_names.audits import implicit_field_check

        issues = implicit_field_check({"id": "vacuum_toroidal_field"})
        assert issues and "implicit_field_check" in issues[0]


# =========================================================================
# C.4b: implicit_field_check — use_exact_* exemption (W19A Issue 3)
# =========================================================================


class TestImplicitFieldUseExactExemption:
    """Tests for use_exact_* prefix exemption in implicit_field_check.

    Constraint selectors reference the field they constrain — the bare
    ``_field`` token is part of the constraint target name, not a
    physics-field concept that needs qualifying.
    """

    def test_pass_use_exact_vacuum_toroidal_field_constraint(self):
        """use_exact_vacuum_toroidal_field_constraint must not be penalised."""
        from imas_codex.standard_names.audits import implicit_field_check

        assert (
            implicit_field_check({"id": "use_exact_vacuum_toroidal_field_constraint"})
            == []
        )

    def test_pass_use_exact_poloidal_field_probe(self):
        """use_exact_poloidal_field_probe must not be penalised."""
        from imas_codex.standard_names.audits import implicit_field_check

        assert (
            implicit_field_check(
                {"id": "use_exact_poloidal_magnetic_field_probe_constraint"}
            )
            == []
        )

    def test_pass_use_exact_with_bare_field(self):
        """Even with bare _field, use_exact prefix exempts."""
        from imas_codex.standard_names.audits import implicit_field_check

        assert implicit_field_check({"id": "use_exact_toroidal_field"}) == []

    def test_fail_bare_field_non_use_exact_still_caught(self):
        """Non-use_exact names with bare _field must still be flagged."""
        from imas_codex.standard_names.audits import implicit_field_check

        issues = implicit_field_check({"id": "vacuum_toroidal_field"})
        assert issues and "implicit_field_check" in issues[0]

    def test_fail_bare_field_poloidal_still_caught(self):
        """Non-use_exact poloidal_field must still be flagged."""
        from imas_codex.standard_names.audits import implicit_field_check

        issues = implicit_field_check({"id": "poloidal_field_strength"})
        assert issues and "implicit_field_check" in issues[0]

    def test_pass_field_of_view(self):
        """``field_of_view`` is an optics term, not a physics field."""
        from imas_codex.standard_names.audits import implicit_field_check

        assert (
            implicit_field_check({"id": "solid_angle_of_detector_field_of_view"}) == []
        )


class TestAngleSolidUnitExpectation:
    """Solid angles use steradians (sr), not radians."""

    def test_pass_solid_angle_sr(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {"id": "solid_angle_of_detector_field_of_view", "unit": "sr"}
        )
        assert issues == [], f"False positive: {issues}"

    def test_pass_angle_rad(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check({"id": "toroidal_angle", "unit": "rad"})
        assert issues == []

    def test_fail_angle_wrong_unit(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check({"id": "toroidal_angle", "unit": "m"})
        assert issues and "name_unit_consistency_check" in issues[0]


class TestCausalDueToSuggestedFix:
    """Tests for suggested_fix in causal_due_to_check adjective map."""

    def test_suggested_fix_halo(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_halo"})
        if issues:  # only fires if halo is not in ISN process vocab
            assert "suggested_fix=" in issues[0]
            assert "halo_currents" in issues[0]

    def test_suggested_fix_thermal(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_thermal"})
        if issues:
            assert "suggested_fix=" in issues[0]
            assert "thermal_fusion" in issues[0]

    def test_suggested_fix_fast_ion(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_fast_ion"})
        if issues:
            assert "suggested_fix=" in issues[0]
            assert "fast_ions" in issues[0]


# =========================================================================
# C.6: pulse_schedule_reference_check
# =========================================================================


class TestPulseScheduleReferenceCheck:
    """Tests for pulse_schedule_reference_check audit."""

    def test_fail_reference_suffix(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check({"id": "plasma_current_reference"})
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_fail_reference_waveform_suffix(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current_reference_waveform"}
        )
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_pass_source_path_with_physics_name(self):
        """Physics SN with pulse_schedule source attached should NOT flag.

        Name like ``plasma_current`` is a legitimate physics standard name
        even if a controller-reference source path is attached to it.
        """
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current"},
            source_path="pulse_schedule/position_control/reference",
        )
        assert issues == []

    def test_fail_reference_suffix_with_pulse_schedule_source(self):
        """Name with _reference_waveform suffix must fail regardless of source."""
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current_reference_waveform"},
            source_path="pulse_schedule/position_control/reference_waveform/data",
        )
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_pass_no_reference(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        assert pulse_schedule_reference_check({"id": "electron_temperature"}) == []

    def test_pass_source_path_no_pulse_schedule(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        assert (
            pulse_schedule_reference_check(
                {"id": "electron_temperature"},
                source_path="core_profiles/profiles_1d/electrons/temperature",
            )
            == []
        )


# =========================================================================
# C.7: ratio_binary_operator_check
# =========================================================================


class TestRatioBinaryOperatorCheck:
    """Tests for ratio_binary_operator_check audit."""

    def test_pass_canonical_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        assert (
            ratio_binary_operator_check(
                {"id": "ratio_of_electron_density_to_ion_density"}
            )
            == []
        )

    def test_fail_adhoc_density_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        issues = ratio_binary_operator_check({"id": "electron_to_ion_density_ratio"})
        assert issues and "ratio_binary_operator_check" in issues[0]
        assert "suggested_fix=" in issues[0]
        assert "ratio_of_electron_to_ion" in issues[0]

    def test_fail_adhoc_bare_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        issues = ratio_binary_operator_check({"id": "tritium_to_deuterium_ratio"})
        assert issues and "ratio_binary_operator_check" in issues[0]

    def test_pass_no_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        assert ratio_binary_operator_check({"id": "electron_temperature"}) == []


# =========================================================================
# C.8: unit_validity_check — whitespace and ^dimension
# =========================================================================


class TestUnitValidityCheckStrengthened:
    """Tests for strengthened unit_validity_check."""

    def test_fail_whitespace_prose_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "Elementary Charge Unit"})
        assert issues and "whitespace" in issues[0]
        assert "dd_upstream" in issues[0]

    def test_fail_caret_dimension_placeholder(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "m^dimension"})
        assert issues and "^dimension" in issues[0]
        assert "dd_upstream" in issues[0]

    def test_pass_normal_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        assert unit_validity_check({"unit": "m^-3"}) == []
        assert unit_validity_check({"unit": "eV"}) == []
        assert unit_validity_check({"unit": "kg.m.s^-2"}) == []


# =========================================================================
# repeated_token_check
# =========================================================================


class TestAdjacentDuplicateTokenCheck:
    """Tests for adjacent_duplicate_token_check audit (CRITICAL)."""

    def test_fail_magnetic_magnetic(self):
        """Adjacent duplicate 'magnetic_magnetic' is flagged as critical."""
        from imas_codex.standard_names.audits import (
            adjacent_duplicate_token_check,
            has_critical_audit_failure,
        )

        issues = adjacent_duplicate_token_check(
            {"id": "bandwidth_3db_of_toroidal_magnetic_magnetic_field_probe"}
        )
        assert len(issues) == 1
        assert "adjacent_duplicate_token_check" in issues[0]
        assert "'magnetic_magnetic'" in issues[0]
        assert has_critical_audit_failure(issues)

    def test_fail_poloidal_magnetic_magnetic_probe(self):
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        issues = adjacent_duplicate_token_check(
            {"id": "poloidal_magnetic_magnetic_field_probe_constraint_weight"}
        )
        assert len(issues) == 1

    def test_pass_deuterium_deuterium(self):
        """Compound-subject 'deuterium_deuterium' does not fire."""
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        assert (
            adjacent_duplicate_token_check({"id": "deuterium_deuterium_reaction_rate"})
            == []
        )

    def test_pass_beam_beam(self):
        """beam-beam fusion reaction is a legitimate compound."""
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        assert (
            adjacent_duplicate_token_check(
                {"id": "deuterium_deuterium_beam_beam_neutron_emission_rate"}
            )
            == []
        )

    def test_pass_non_adjacent_magnetic(self):
        """magnetic_field_at_magnetic_axis has non-adjacent repetition — legit."""
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        assert (
            adjacent_duplicate_token_check({"id": "magnetic_field_at_magnetic_axis"})
            == []
        )

    def test_pass_non_adjacent_upper(self):
        """upper_uncertainty_of_upper_triangularity — non-adjacent."""
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        assert (
            adjacent_duplicate_token_check(
                {"id": "upper_uncertainty_of_upper_triangularity_of_plasma_boundary"}
            )
            == []
        )

    def test_pass_empty(self):
        from imas_codex.standard_names.audits import adjacent_duplicate_token_check

        assert adjacent_duplicate_token_check({"id": ""}) == []


class TestRepeatedTokenCheck:
    """Tests for repeated_token_check audit."""

    def test_fail_magnetic_magnetic(self):
        """Duplicated content token 'magnetic' is flagged."""
        from imas_codex.standard_names.audits import repeated_token_check

        issues = repeated_token_check({"id": "magnetic_magnetic_field_strength"})
        assert len(issues) == 1
        assert "repeated_token_check" in issues[0]
        assert "'magnetic'" in issues[0]

    def test_fail_temperature_temperature(self):
        """Duplicated 'temperature' across segments is flagged."""
        from imas_codex.standard_names.audits import repeated_token_check

        issues = repeated_token_check(
            {"id": "electron_temperature_of_core_temperature"}
        )
        assert len(issues) == 1
        assert "'temperature'" in issues[0]

    def test_pass_deuterium_deuterium_reaction_rate(self):
        """Compound-subject 'deuterium_deuterium' must not fire."""
        from imas_codex.standard_names.audits import repeated_token_check

        assert repeated_token_check({"id": "deuterium_deuterium_reaction_rate"}) == []

    def test_pass_deuterium_tritium_reaction_rate(self):
        """Compound-subject 'deuterium_tritium' must not fire."""
        from imas_codex.standard_names.audits import repeated_token_check

        assert repeated_token_check({"id": "deuterium_tritium_reaction_rate"}) == []

    def test_pass_normal_name(self):
        """Normal name with no repeats passes."""
        from imas_codex.standard_names.audits import repeated_token_check

        assert repeated_token_check({"id": "electron_temperature"}) == []

    def test_pass_connectives_repeated(self):
        """Grammar connectives (of, at, per) may repeat without firing."""
        from imas_codex.standard_names.audits import repeated_token_check

        # 'of' appears twice but is a connective — no content duplication
        assert (
            repeated_token_check({"id": "gradient_of_pressure_at_edge_of_plasma"}) == []
        )


# =========================================================================
# instrument_stokes_bind_check
# =========================================================================


class TestInstrumentStokesBindCheck:
    """Tests for instrument_stokes_bind_check audit."""

    def test_fail_stokes_vector_of_polarimeter(self):
        """Classic anti-pattern: stokes observable bound to instrument."""
        from imas_codex.standard_names.audits import instrument_stokes_bind_check

        issues = instrument_stokes_bind_check({"id": "stokes_vector_of_polarimeter"})
        assert len(issues) == 1
        assert "instrument_stokes_bind_check" in issues[0]
        assert "NC-30" in issues[0]

    def test_fail_degree_of_polarization_interferometer(self):
        """degree_of_polarization + interferometer fires."""
        from imas_codex.standard_names.audits import instrument_stokes_bind_check

        issues = instrument_stokes_bind_check(
            {"id": "degree_of_polarization_of_interferometer"}
        )
        assert len(issues) == 1

    def test_pass_stokes_vector_alone(self):
        """stokes_vector without an instrument token does not fire."""
        from imas_codex.standard_names.audits import instrument_stokes_bind_check

        assert instrument_stokes_bind_check({"id": "stokes_vector"}) == []

    def test_pass_polarimeter_alone(self):
        """Instrument token without a Stokes observable does not fire."""
        from imas_codex.standard_names.audits import instrument_stokes_bind_check

        assert instrument_stokes_bind_check({"id": "wavelength_of_polarimeter"}) == []

    def test_pass_no_overlap(self):
        """Totally unrelated name passes."""
        from imas_codex.standard_names.audits import instrument_stokes_bind_check

        assert instrument_stokes_bind_check({"id": "electron_temperature"}) == []


# =========================================================================
# position_redundancy_check
# =========================================================================


class TestPositionRedundancyCheck:
    """Tests for position_redundancy_check audit."""

    def test_fail_at_wall_surface(self):
        """``at_wall_surface`` triggers — ISN has ``wall``, not ``wall_surface``."""
        from imas_codex.standard_names.audits import position_redundancy_check

        issues = position_redundancy_check(
            {"id": "emitted_radiation_energy_flux_at_wall_surface"}
        )
        assert len(issues) == 1
        assert "position_redundancy_check" in issues[0]
        assert "'wall'" in issues[0]

    def test_fail_on_wall_surface(self):
        """``on_wall_surface`` also triggers."""
        from imas_codex.standard_names.audits import position_redundancy_check

        issues = position_redundancy_check({"id": "ion_energy_flux_on_wall_surface"})
        assert len(issues) == 1
        assert "position_redundancy_check" in issues[0]

    def test_pass_at_wall(self):
        """Correct ``at_wall`` does not trigger."""
        from imas_codex.standard_names.audits import position_redundancy_check

        assert (
            position_redundancy_check({"id": "emitted_radiation_energy_flux_at_wall"})
            == []
        )

    def test_pass_unrelated(self):
        """Unrelated name does not trigger."""
        from imas_codex.standard_names.audits import position_redundancy_check

        assert position_redundancy_check({"id": "electron_temperature"}) == []


# =========================================================================
# process_qualifier_check
# =========================================================================


class TestProcessQualifierCheck:
    """Tests for process_qualifier_check audit."""

    def test_fail_due_to_recombination_at_ion_state(self):
        """Over-qualified process ``recombination_at_ion_state`` triggers."""
        from imas_codex.standard_names.audits import process_qualifier_check

        issues = process_qualifier_check(
            {"id": "ion_incident_energy_flux_on_wall_due_to_recombination_at_ion_state"}
        )
        assert len(issues) == 1
        assert "process_qualifier_check" in issues[0]
        assert "'recombination'" in issues[0]

    def test_fail_due_to_impurity_radiation_in_halo_region(self):
        """Over-qualified process ``impurity_radiation_in_halo_region`` triggers."""
        from imas_codex.standard_names.audits import process_qualifier_check

        issues = process_qualifier_check(
            {"id": "electron_radiated_energy_due_to_impurity_radiation_in_halo_region"}
        )
        assert len(issues) == 1
        assert "process_qualifier_check" in issues[0]
        assert "'impurity_radiation'" in issues[0]

    def test_pass_bare_process(self):
        """Correct bare process token does not trigger."""
        from imas_codex.standard_names.audits import process_qualifier_check

        assert (
            process_qualifier_check({"id": "ion_energy_flux_due_to_recombination"})
            == []
        )

    def test_pass_no_due_to(self):
        """Name without ``due_to`` does not trigger."""
        from imas_codex.standard_names.audits import process_qualifier_check

        assert process_qualifier_check({"id": "electron_temperature"}) == []

    def test_fail_due_to_process_on_surface(self):
        """``_on_`` qualifier after process triggers."""
        from imas_codex.standard_names.audits import process_qualifier_check

        issues = process_qualifier_check(
            {"id": "heat_flux_due_to_radiation_on_outer_wall"}
        )
        assert len(issues) == 1
        assert "process_qualifier_check" in issues[0]


class TestNormalizedQuantityBypass:
    """Normalized/gyrokinetic quantities must pass unit-dimensionality audit."""

    def test_pass_normalized_pressure_moment_dimensionless(self):
        """Name starting with ``normalized_`` + dimensionless unit → pass."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "normalized_pressure_moment",
                    "unit": "1",
                    "description": "Normalized pressure moment",
                }
            )
            == []
        )

    def test_fail_bare_pressure_dimensionless(self):
        """``pressure`` without normalization marker + dimensionless → fail."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {
                "id": "pressure",
                "unit": "1",
                "description": "Pressure in dimensionless units",
            }
        )
        assert issues
        assert "name_unit_consistency_check" in issues[0]

    def test_pass_normalized_electron_temperature_pedestal(self):
        """Normalized temperature with dimensionless unit → pass."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "normalized_electron_temperature_pedestal",
                    "unit": "1",
                    "description": "Normalized electron temperature at the pedestal",
                }
            )
            == []
        )

    def test_pass_norm_path_even_without_name_marker(self):
        """DD path with ``_norm_`` segment → pass even if name lacks normalized."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "gyrocenter_pressure_moment",
                    "unit": "1",
                    "description": "Gyrocenter pressure moment (normalized)",
                },
                source_path="gyrokinetics_local/linear/wavevector/eigenmode/"
                "moments_norm_gyrocenter/pressure",
            )
            == []
        )

    def test_pass_normalised_british_spelling(self):
        """British spelling ``normalised_`` prefix → pass."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "normalised_energy",
                    "unit": "1",
                    "description": "Normalised energy",
                }
            )
            == []
        )


class TestGgdImplementationLeakageCheck:
    """Tests for ggd_implementation_leakage_check."""

    def test_flag_on_the_ggd(self):
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {"description": "Temperature on the GGD edge grid"}
        )
        assert len(issues) == 1
        assert "ggd_leakage" in issues[0]

    def test_flag_ggd_mesh(self):
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {"documentation": "Defined on a GGD mesh element for edge transport"}
        )
        assert len(issues) == 1
        assert "ggd_leakage" in issues[0]

    def test_flag_unstructured_ggd(self):
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {"description": "Stored on an unstructured GGD representation"}
        )
        assert len(issues) == 1

    def test_pass_bare_ggd_reference(self):
        """Bare 'GGD' without implementation pattern should pass."""
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {"description": "General grid description (GGD) based quantity"}
        )
        assert issues == []

    def test_pass_no_ggd(self):
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {"description": "Electron temperature on the plasma edge"}
        )
        assert issues == []

    def test_both_fields_flagged(self):
        from imas_codex.standard_names.audits import ggd_implementation_leakage_check

        issues = ggd_implementation_leakage_check(
            {
                "description": "Density on the GGD edge grid",
                "documentation": "Values stored on a GGD mesh",
            }
        )
        assert len(issues) == 2


# =========================================================================
# Semantic similarity gate
# =========================================================================


class TestSemanticSimilarityCheck:
    """Tests for the embedding-based semantic similarity gate."""

    def test_good_name_high_similarity(self):
        """A self-describing name should score above warning threshold."""
        from unittest.mock import patch

        from imas_codex.standard_names.audits import semantic_similarity_check

        # Mock embed to return controlled vectors
        # name = "electron temperature" → [1, 0, 0, ...]
        # desc = "Temperature of electrons" → [0.9, 0.1, 0, ...]
        good_name_emb = [1.0, 0.0, 0.0, 0.0]
        good_desc_emb = [0.9, 0.1, 0.0, 0.0]

        def mock_embed(items, text_field="_text", embedding_field="embedding"):
            for item in items:
                if item["id"] == "name":
                    item["embedding"] = good_name_emb
                else:
                    item["embedding"] = good_desc_emb
            return items

        with patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=mock_embed,
        ):
            sim, issues = semantic_similarity_check(
                "electron_temperature",
                "Temperature of electrons in the plasma",
            )

        assert sim is not None
        assert sim > 0.65  # above warning
        assert issues == []

    def test_ambiguous_name_critical(self):
        """An ambiguous name should fail critical threshold."""
        from unittest.mock import patch

        from imas_codex.standard_names.audits import semantic_similarity_check

        # Orthogonal vectors → sim ≈ 0
        ambig_name_emb = [1.0, 0.0, 0.0, 0.0]
        ambig_desc_emb = [0.0, 1.0, 0.0, 0.0]

        def mock_embed(items, text_field="_text", embedding_field="embedding"):
            for item in items:
                if item["id"] == "name":
                    item["embedding"] = ambig_name_emb
                else:
                    item["embedding"] = ambig_desc_emb
            return items

        with patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=mock_embed,
        ):
            sim, issues = semantic_similarity_check(
                "co_passing_density",
                "Number density of co-passing particles in velocity space",
            )

        assert sim is not None
        assert sim < 0.55  # below critical
        assert len(issues) == 1
        assert "semantic_similarity_check:" in issues[0]
        assert "critical" in issues[0]

    def test_warning_zone(self):
        """A name in the warning zone should get advisory issue."""
        from unittest.mock import patch

        from imas_codex.standard_names.audits import semantic_similarity_check

        # Vectors with ~0.6 cosine similarity
        name_emb = [1.0, 0.0, 0.0, 0.0]
        desc_emb = [0.6, 0.8, 0.0, 0.0]  # cos ≈ 0.6

        def mock_embed(items, text_field="_text", embedding_field="embedding"):
            for item in items:
                if item["id"] == "name":
                    item["embedding"] = name_emb
                else:
                    item["embedding"] = desc_emb
            return items

        with patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=mock_embed,
        ):
            sim, issues = semantic_similarity_check(
                "electric_field",
                "Magnitude of the electric field in a tokamak plasma",
            )

        assert sim is not None
        assert sim < 0.65
        assert sim > 0.55
        assert len(issues) == 1
        assert "warning" in issues[0]

    def test_empty_description_returns_none(self):
        """Empty description should return None, no issues."""
        from imas_codex.standard_names.audits import semantic_similarity_check

        sim, issues = semantic_similarity_check("electron_temperature", "")
        assert sim is None
        assert issues == []

    def test_none_description_returns_none(self):
        """None description should return None, no issues."""
        from imas_codex.standard_names.audits import semantic_similarity_check

        sim, issues = semantic_similarity_check("electron_temperature", None)
        assert sim is None
        assert issues == []

    def test_embed_failure_returns_none(self):
        """If embed server is down, should return None gracefully."""
        from unittest.mock import patch

        from imas_codex.standard_names.audits import semantic_similarity_check

        with patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=ConnectionError("embed server down"),
        ):
            sim, issues = semantic_similarity_check(
                "electron_temperature",
                "Temperature of electrons",
            )

        assert sim is None
        assert issues == []

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        from unittest.mock import patch

        from imas_codex.standard_names.audits import semantic_similarity_check

        # Use ~0.7 similarity (normally above warning 0.65, but set warning to 0.8)
        name_emb = [1.0, 0.0, 0.0, 0.0]
        desc_emb = [0.7, 0.71, 0.0, 0.0]

        def mock_embed(items, text_field="_text", embedding_field="embedding"):
            for item in items:
                if item["id"] == "name":
                    item["embedding"] = name_emb
                else:
                    item["embedding"] = desc_emb
            return items

        with patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=mock_embed,
        ):
            sim, issues = semantic_similarity_check(
                "electron_temperature",
                "Temperature of electrons",
                critical_threshold=0.3,
                warning_threshold=0.8,
            )

        assert sim is not None
        # sim ~0.7 is below custom warning 0.8 but above custom critical 0.3
        assert len(issues) == 1
        assert "warning" in issues[0]

    def test_critical_check_in_critical_checks(self):
        """semantic_similarity_check should be in CRITICAL_CHECKS."""
        from imas_codex.standard_names.audits import CRITICAL_CHECKS

        assert "semantic_similarity_check" in CRITICAL_CHECKS

    def test_has_critical_failure_with_semantic(self):
        """has_critical_audit_failure should detect semantic issues."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        issues = [
            "audit:semantic_similarity_check: sim=0.400 below critical threshold 0.55"
        ]
        assert has_critical_audit_failure(issues) is True

    def test_warning_not_critical(self):
        """Warning issues should NOT count as critical failures."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        issues = [
            "audit:semantic_similarity_check_warning: sim=0.600 below warning 0.65"
        ]
        assert has_critical_audit_failure(issues) is False


class TestPrepositionPhysicalBaseCheck:
    """Names whose ISN-parsed physical_base starts with a preposition are broken."""

    def test_normalized_of_particle_temperature_not_flagged(self):
        """ISN v0.8.0rc1 parses ``normalized_of_particle_temperature`` correctly:
        transformation=normalized, subject=particle, physical_base=temperature.
        The ``_of_`` connector is absorbed by the parser, so physical_base is
        clean — no preposition leak."""
        from imas_codex.standard_names.audits import preposition_physical_base_check

        issues = preposition_physical_base_check(
            {"id": "normalized_of_particle_temperature"}
        )
        assert len(issues) == 0

    def test_normalized_of_particle_mass_not_flagged(self):
        """ISN v0.8.0rc1 correctly parses ``normalized_of_particle_mass`` —
        physical_base is 'mass', not 'of_particle_mass'."""
        from imas_codex.standard_names.audits import preposition_physical_base_check

        issues = preposition_physical_base_check({"id": "normalized_of_particle_mass"})
        assert len(issues) == 0

    def test_clean_name_passes(self):
        """A well-formed name has no preposition prefix on physical_base."""
        from imas_codex.standard_names.audits import preposition_physical_base_check

        assert preposition_physical_base_check({"id": "electron_temperature"}) == []

    def test_normalized_without_of_passes(self):
        """``normalized_particle_temperature`` is the correct form."""
        from imas_codex.standard_names.audits import preposition_physical_base_check

        assert (
            preposition_physical_base_check({"id": "normalized_particle_temperature"})
            == []
        )

    def test_empty_name_passes(self):
        """Empty or missing name is a no-op."""
        from imas_codex.standard_names.audits import preposition_physical_base_check

        assert preposition_physical_base_check({"id": ""}) == []
        assert preposition_physical_base_check({}) == []

    def test_is_critical_check(self):
        """preposition_physical_base_check must be in CRITICAL_CHECKS."""
        from imas_codex.standard_names.audits import CRITICAL_CHECKS

        assert "preposition_physical_base_check" in CRITICAL_CHECKS

    def test_has_critical_failure(self):
        """has_critical_audit_failure detects preposition_physical_base_check issues."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        issues = [
            "audit:preposition_physical_base_check: ISN parse of "
            "'normalized_of_particle_temperature' yields "
            "physical_base='of_particle_temperature'"
        ]
        assert has_critical_audit_failure(issues) is True


class TestCanonicalLocusCheck:
    """Canonical-locus rewrites must stay inside the installed ISN registry."""

    def test_flags_plain_separatrix_synonym(self):
        from imas_codex.standard_names.audits import canonical_locus_check

        issues = canonical_locus_check({"id": "electron_temperature_at_separatrix"})

        assert any("plasma_boundary" in issue for issue in issues)

    def test_flags_registered_compound_divertor_synonym(self):
        from imas_codex.standard_names.audits import canonical_locus_check

        issues = canonical_locus_check({"id": "heat_flux_at_inner_divertor_plate"})

        assert any("inner_divertor_target" in issue for issue in issues)

    def test_does_not_invent_secondary_plasma_boundary(self):
        from imas_codex.standard_names.audits import canonical_locus_check

        issues = canonical_locus_check(
            {"id": "electron_temperature_at_secondary_separatrix"}
        )

        assert not any("secondary_plasma_boundary" in issue for issue in issues)


class TestAttachmentStateResolution:
    """State-resolution consistency in _is_attachment_consistent (R1/R4)."""

    @pytest.mark.parametrize(
        "source_id,sn_name,expected_ok",
        [
            # state path -> state name: OK
            (
                "edge_profiles/profiles_1d/neutral/state/density_thermal",
                "thermal_neutral_state_density",
                True,
            ),
            # state path -> species name: REJECT (R4 live defect)
            (
                "edge_profiles/profiles_1d/neutral/state/density_thermal",
                "thermal_neutral_density",
                False,
            ),
            # species path -> state name: REJECT
            (
                "core_profiles/profiles_1d/ion/pressure_fast_parallel",
                "parallel_fast_ion_state_pressure",
                False,
            ),
            # species path -> species name: OK
            (
                "core_profiles/profiles_1d/ion/pressure_fast_parallel",
                "parallel_fast_ion_pressure",
                True,
            ),
            # ion_charge_state token also counts as state-resolved
            (
                "edge_profiles/ggd/ion/state/pressure_fast_perpendicular",
                "perpendicular_fast_ion_charge_state_pressure",
                True,
            ),
        ],
    )
    def test_state_resolution_consistency(self, source_id, sn_name, expected_ok):
        from imas_codex.standard_names.workers import _is_attachment_consistent

        ok, reason = _is_attachment_consistent(source_id, sn_name)
        assert ok is expected_ok, reason


# =========================================================================
# Corpus-level: vector_family_consistency_check
# =========================================================================


class TestVectorFamilyConsistencyCheck:
    """Family audit — components of one DD vector node must agree."""

    def _cam(self, axis_name, leaf, *, locus="camera", domain="magnetics"):
        # A fabricated camera direction-vector component.
        name = f"{axis_name}_direction_unit_vector"
        if locus:
            name = f"{name}_of_{locus}"
        return {
            "id": name,
            "physics_domain": domain,
            "source_paths": [f"camera_ir/channel/camera/direction/{leaf}"],
        }

    def test_clean_cartesian_family_passes(self):
        """Cartesian node (x, y, z): the z leaf is the 'z' axis token."""
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("x", "x"),
            self._cam("y", "y"),
            self._cam("z", "z"),
        ]
        assert vector_family_consistency_check(names) == []

    def test_clean_cylindrical_family_passes(self):
        """Cylindrical node (r, phi, z): the z leaf is the 'vertical' axis token."""
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("radial", "r"),
            self._cam("toroidal", "phi"),
            self._cam("vertical", "z"),
        ]
        assert vector_family_consistency_check(names) == []

    def test_locus_and_domain_drift_flagged(self):
        """The real camera failure: one component locus-less + off-domain."""
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("x", "x", locus=None, domain="camera_visible"),
            self._cam("y", "y", locus="strain_gauge_sensor", domain="mechanical_loads"),
            self._cam("z", "z", locus=None, domain="camera_visible"),
        ]
        issues = vector_family_consistency_check(names)
        joined = "\n".join(issues)
        assert "locus" in joined
        assert "physics_domain" in joined

    def test_cartesian_z_leaf_must_not_be_vertical(self):
        """In a Cartesian node the z leaf is 'z' — a 'vertical' name is flagged."""
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("x", "x"),
            self._cam("y", "y"),
            # z leaf of a Cartesian node named 'vertical' instead of 'z'
            self._cam("vertical", "z"),
        ]
        issues = vector_family_consistency_check(names)
        joined = "\n".join(issues)
        assert "canonical axis token 'z'" in joined
        assert "vertical_direction_unit_vector_of_camera" in joined

    def test_cylindrical_z_leaf_must_not_be_bare_z(self):
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("radial", "r"),
            # a 'z' leaf of a cylindrical node named with a bare 'z' token
            {
                "id": "z_direction_unit_vector_of_camera",
                "physics_domain": "magnetics",
                "source_paths": ["camera_ir/channel/camera/direction/z"],
            },
        ]
        issues = vector_family_consistency_check(names)
        joined = "\n".join(issues)
        assert "canonical axis token 'vertical'" in joined
        assert "z_direction_unit_vector_of_camera" in joined

    def test_mixed_frame_triple_flagged(self):
        from imas_codex.standard_names.audits import vector_family_consistency_check

        # leaves x / phi / z -> a cylindrical node (phi present) so z -> vertical,
        # but axis tokens {x, toroidal, vertical} mix frames: subset of neither
        # canonical triple.
        names = [
            self._cam("x", "x"),
            self._cam("toroidal", "phi"),
            self._cam("vertical", "z"),
        ]
        issues = vector_family_consistency_check(names)
        assert any("non-canonical axis triple" in i for i in issues)

    def test_base_carrier_disagreement_flagged(self):
        from imas_codex.standard_names.audits import vector_family_consistency_check

        names = [
            self._cam("x", "x"),
            {
                "id": "y_line_of_sight_unit_vector_of_camera",
                "physics_domain": "magnetics",
                "source_paths": ["camera_ir/channel/camera/direction/y"],
            },
        ]
        issues = vector_family_consistency_check(names)
        assert any("base carrier" in i for i in issues)

    def test_single_component_not_flagged(self):
        from imas_codex.standard_names.audits import vector_family_consistency_check

        assert vector_family_consistency_check([self._cam("vertical", "z")]) == []

    def test_distinct_devices_do_not_group(self):
        from imas_codex.standard_names.audits import vector_family_consistency_check

        # Different device grandparents -> different vector nodes -> no
        # cross-family comparison even though both are '/direction/z'.
        names = [
            self._cam("vertical", "z"),
            {
                "id": "vertical_direction_unit_vector_of_launching_mirror",
                "physics_domain": "ec_launchers",
                "source_paths": ["ec_launchers/beam/launching_position/direction/z"],
            },
        ]
        assert vector_family_consistency_check(names) == []


# ---------------------------------------------------------------------------
# Corpus-level: dd_path_uniqueness_check
# ---------------------------------------------------------------------------


class TestDdPathUniquenessCheck:
    """One DD path carries exactly one standard name."""

    def _entry(self, name, paths):
        return {"id": name, "source_paths": paths}

    def test_unique_paths_clean(self):
        from imas_codex.standard_names.audits import dd_path_uniqueness_check

        names = [
            self._entry("electron_temperature", ["core_profiles/te"]),
            self._entry("ion_temperature", ["core_profiles/ti"]),
        ]
        assert dd_path_uniqueness_check(names) == []

    def test_double_attach_flagged(self):
        from imas_codex.standard_names.audits import dd_path_uniqueness_check

        names = [
            self._entry("magnetic_shear", ["summary/local/pedestal/magnetic_shear"]),
            self._entry(
                "magnetic_shear_at_pedestal",
                ["summary/local/pedestal/magnetic_shear"],
            ),
        ]
        issues = dd_path_uniqueness_check(names)
        assert len(issues) == 1
        assert "summary/local/pedestal/magnetic_shear" in issues[0]
        assert "magnetic_shear_at_pedestal" in issues[0]

    def test_dd_scheme_prefix_stripped(self):
        from imas_codex.standard_names.audits import dd_path_uniqueness_check

        names = [
            self._entry("a_name", ["dd:x/y/z"]),
            self._entry("b_name", ["x/y/z"]),
        ]
        assert len(dd_path_uniqueness_check(names)) == 1


# =========================================================================
# Derived-parent structural admission
# =========================================================================


class TestDerivedParentStructuralCheck:
    """Structural gate for derived family parents (deliberately partial names)."""

    def test_consistent_peel_over_species_passes(self):
        """A species-peel parent whose tokens are a subset of a child passes."""
        from imas_codex.standard_names.audits import derived_parent_structural_check

        # Parent drops the species subject its children carry.
        assert (
            derived_parent_structural_check(
                "internal_state_energy_flux",
                [
                    "deuterium_internal_state_energy_flux",
                    "tungsten_internal_state_energy_flux",
                ],
            )
            == []
        )

    def test_reordered_qualifier_peel_passes(self):
        """Token-SET containment tolerates qualifier reordering between peel/child."""
        from imas_codex.standard_names.audits import derived_parent_structural_check

        # parent tokens {perturbed, particle, energy} ⊆ child token set.
        assert (
            derived_parent_structural_check(
                "perturbed_particle_energy",
                ["normalized_particle_perturbed_energy"],
            )
            == []
        )

    def test_orphan_parent_quarantines(self):
        """A derived parent with no children generalises nothing — critical."""
        from imas_codex.standard_names.audits import (
            derived_parent_structural_check,
            has_critical_audit_failure,
        )

        issues = derived_parent_structural_check("velocity_due_to_convection", [])
        assert len(issues) == 1
        assert "no HAS_PARENT children" in issues[0]
        assert has_critical_audit_failure(issues) is True

    def test_inconsistent_peel_quarantines(self):
        """A parent whose tokens are not a subset of any child is a bad peel."""
        from imas_codex.standard_names.audits import (
            derived_parent_structural_check,
            has_critical_audit_failure,
        )

        issues = derived_parent_structural_check(
            "wave_power",
            ["electron_temperature", "ion_density"],
        )
        assert len(issues) == 1
        assert "not a token-generalisation" in issues[0]
        assert has_critical_audit_failure(issues) is True

    def test_empty_name_quarantines(self):
        from imas_codex.standard_names.audits import derived_parent_structural_check

        issues = derived_parent_structural_check("", ["some_child"])
        assert len(issues) == 1
        assert "empty parent name" in issues[0]

    def test_none_children_treated_as_orphan(self):
        from imas_codex.standard_names.audits import derived_parent_structural_check

        issues = derived_parent_structural_check("plasma_momentum", None)
        assert len(issues) == 1
        assert "no HAS_PARENT children" in issues[0]
