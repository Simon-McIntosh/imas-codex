"""Regression guards for strict normative standard-name documentation."""

from pathlib import Path

import yaml

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt
from imas_codex.standard_names.prose_policy import banned_prose_findings


def test_derived_from_linked_quantity_is_provenance_not_a_recipe() -> None:
    """ "X is derived from [Y](name:Y)" names a catalogued source quantity — the
    parent/child provenance form the refine seat writes — so it must not flag as
    estimator_recipe.  The exemption is conditioned on the in-sentence
    ``[label](name:id)`` link, NOT on the word "derived" alone.

    Anchored on the two accepted docs whose linked "derived from the local ..."
    provenance sentence tripped the campaign gate as a spurious reintroduction.
    """
    provenance = [
        "It is derived from the local [hydrogen density](name:hydrogen_density) "
        "and excludes neutral atomic hydrogen and heavier hydrogen isotopes.",
        "It is derived from the local [tritium density](name:tritium_density) "
        "and [deuterium density](name:deuterium_density).",
        "This ratio can be derived from the local "
        "[deuterium density](name:deuterium_density).",
    ]
    for text in provenance:
        assert banned_prose_findings(text)["estimator_recipe"] == 0, text


def test_derived_from_a_procedure_still_flags_estimator_recipe() -> None:
    """ "derived from <procedure/measurement>" (no linked quantity) is a genuine
    recipe and must stay flagged — the exemption is only for linked provenance."""
    recipes = [
        # unlinked "derived from <procedure>" — the majority catalog usage
        "It is derived from equilibrium reconstruction by fitting magnetic "
        "field measurements.",
        "This quantity is derived from view-factor integration of the total "
        "radiated power.",
        "The profile is derived from transport models that parameterize "
        "neoclassical fluxes.",
        # other compute verbs are unconditional
        "In practice this quantity is computed by integrating local absorption.",
        "This quantity is obtained by integrating the fast-ion distribution.",
        "This value can be computed with a transport code.",
        "It is derived by differentiating the smoothed profile.",
    ]
    for text in recipes:
        assert banned_prose_findings(text)["estimator_recipe"] >= 1, text


def test_compute_verb_introducing_an_equation_is_a_definition_not_a_recipe() -> None:
    """ "X is obtained/computed by <operation>: $$equation$$" is a mathematical
    definition, not a procedure — the definitional form the refine seat writes
    for derived/integral quantities.  Exempt when a display or inline equation
    follows within ~200 chars.

    Anchored on the two accepted docs whose definitional-integral sentence
    tripped the campaign gate as a spurious reintroduction.
    """
    definitions = [
        # display equation on the next block (current_due_to_ion_cyclotron_...)
        "The current is obtained by integrating the driven toroidal current "
        "density over the plasma cross-section:\n\n$$\nI = \\int_A j\\,dA\n$$\n",
        # inline equation immediately after (perturbed_gyrocenter_pressure)
        "The dimensional counterpart is obtained by omitting the "
        "$1/(n_\\mathrm{ref} T_\\mathrm{ref})$ factor and carries units of "
        "pressure.",
        "The area is computed as the surface integral $$A = \\int dS$$.",
    ]
    for text in definitions:
        assert banned_prose_findings(text)["estimator_recipe"] == 0, text


def test_compute_verb_without_an_equation_still_flags_estimator_recipe() -> None:
    """A compute verb with no nearby equation narrates a measurement/estimation
    procedure and must stay flagged — the equation exemption is definitional
    only."""
    recipes = [
        "On-axis temperature is obtained from Thomson scattering or electron "
        "cyclotron emission radiometry for the electron channel.",
        "In practice it is computed by summing charge-state-resolved iron ion "
        "densities and applying the flux-surface average.",
        "This average is obtained from equilibrium reconstruction by evaluating "
        "the external-source Green's functions.",
    ]
    for text in recipes:
        assert banned_prose_findings(text)["estimator_recipe"] >= 1, text


def test_for_example_introducing_a_linked_child_is_taxonomy_not_padding() -> None:
    """ "for example, [child](name:id)" points at a catalogued specialisation in
    a parent-concept doc — legitimate taxonomy, not procedural padding.  Exempt
    when a name link follows within ~120 chars.

    Anchored on power_of_breeder_blanket, whose linked-child example tripped the
    campaign gate.
    """
    taxonomy = [
        "Child quantities define those restrictions explicitly; for example, "
        "[thermal power of breeder blanket](name:thermal_power_of_breeder_"
        "blanket) denotes a thermal-load specialisation.",
        "For example, the child standard name "
        "[energy flux at wall](name:energy_flux_at_wall_due_to_surface_emission) "
        "restricts the accounting boundary.",
    ]
    for text in taxonomy:
        assert banned_prose_findings(text)["procedural_padding"] == 0, text


def test_for_example_without_a_link_still_flags_procedural_padding() -> None:
    """Illustrative "for example, <formula/text>" with no catalogued child link
    remains procedural padding and keeps flagging."""
    padding = [
        "For example, $10^{20}\\ \\mathrm{m^{-3}\\,s^{-1}}$ changes the balance.",
        "The count is by atoms; for example, deuterium and tritium nuclei in a "
        "mixed hydrogenic core are counted individually.",
    ]
    for text in padding:
        assert banned_prose_findings(text)["procedural_padding"] >= 1, text


def _read(relative: str) -> str:
    return (PROMPTS_DIR / relative).read_text(encoding="utf-8")


def test_generation_and_refinement_prompts_enforce_normative_boundary() -> None:
    generated = render_prompt(
        "sn/generate_docs_user",
        {
            "item": {
                "name": "thermal_plasma_energy",
                "unit": "J",
                "kind": "scalar",
                "physics_domain": "core_plasma_physics",
            },
            "nearby_existing_names": [],
        },
    )
    refined = render_prompt(
        "sn/refine_docs_user",
        {
            "sn_name": "total_power_due_to_ion_cyclotron_heating",
            "unit": "W",
            "kind": "scalar",
            "physics_domain": "auxiliary_heating",
            "description": "Power absorbed by the plasma through ion-cyclotron waves.",
            "dd_paths": [],
            "docs_chain_history": [],
            "docs_chain_length": 0,
            "reviewer_score_docs": None,
            "reviewer_comments_per_dim_docs": None,
        },
    )

    for prompt in (generated, refined):
        assert "strict normative" in prompt.lower()
        assert "generic diagnostics" in prompt.lower()
        assert "estimator" in prompt.lower() and "recipes" in prompt.lower()
        assert "typical machine" in prompt.lower()
        assert "constitutive" in prompt.lower()
        assert "necessary to distinguish" in prompt.lower()


def test_review_prompts_penalize_practical_appendices_instead_of_rewarding_them() -> (
    None
):
    prompt_files = (
        "sn/review.md",
        "sn/review_docs.md",
        "sn/review_docs_system.md",
    )
    for prompt_file in prompt_files:
        text = _read(prompt_file).lower()
        assert "strict normative" in text
        assert "estimator recipes" in text
        assert "typical" in text
        assert "constitutive" in text

    criteria = yaml.safe_load(
        Path("imas_codex/llm/config/sn_review_criteria.yaml").read_text(
            encoding="utf-8"
        )
    )
    rule = criteria["common_issues"]["documentation_quality"][
        "I2.2_tangential_context"
    ]["rule"].lower()
    assert "typical values" in rule and "defects" in rule
    assert "measurement/computation is allowed only" in rule


def test_exemplars_reject_both_expert_review_failure_modes() -> None:
    exemplars = _read("shared/sn/_exemplars_enrich.md")
    assert "thermal_plasma_energy" in exemplars
    assert "total_power_due_to_ion_cyclotron_heating" in exemplars
    assert "diagnostic list, estimator comparison" in exemplars
    assert "estimator recipe, not a canonical definition" in exemplars


def test_shared_format_keeps_normative_positive_content() -> None:
    docs_format = _read("shared/sn/_docs_format.md")
    for required in (
        "Definition paragraph",
        "Governing equation paragraph",
        "Scope / distinction paragraph",
        "Sign convention paragraph",
        "Define every symbol introduced",
    ):
        assert required in docs_format

    assert "Measurement / computation paragraph" not in docs_format
    assert "Typical values paragraph" not in docs_format
