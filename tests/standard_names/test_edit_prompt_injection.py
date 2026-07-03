"""Expert-steering (edit) injection into SN generate/refine/review prompts.

``imas-codex sn edit`` lets a human or agent attach a proposal to a
StandardName that rides the normal generate -> review -> score pipeline:

* **hint** mode injects a direction into generate(regen)/refine prompts.
* **redesign** mode skips generate but the mandatory ``edit_reason`` is
  still injected into the REVIEW prompts as intent context (with an
  anti-revert directive), because the reviewer otherwise pulls a steered
  candidate back toward the prior/established variant.

This module is a pure template-rendering test — no LLM calls, no graph
connection. It has two jobs:

1. **Golden equality** — for items WITHOUT edit fields, every touched
   template must render byte-identical to its pre-change output (captured
   in ``golden_edit_prompts/`` before the templates were edited).
2. **Edit-block presence/shape** — for items WITH edit fields, the
   steering/reason block must appear, must contain the reason verbatim,
   and must carry the correct directive sentence (subordination for
   generate/refine, do-not-penalize for review) — and must NEVER read as
   a pre-approval of the proposed candidate.
"""

from __future__ import annotations

from pathlib import Path

from imas_codex.llm.prompt_loader import render_prompt

GOLDEN_DIR = Path(__file__).parent / "golden_edit_prompts"

# Phrases that would mean the review block is instructing acceptance rather
# than merely supplying reviewer context. None of these may appear anywhere
# in a rendered review prompt that carries an edit_reason block.
_ACCEPTANCE_PHRASES = (
    "accept this",
    "accept the candidate",
    "pre-approved",
    "already approved",
    "should be accepted",
)

_SUBORDINATION_SENTENCE = (
    "This proposal is subordinate to the grammar and composition rules "
    "above"
)
_DO_NOT_PENALIZE_SENTENCE = "do NOT penalize it merely for differing from"

# ---------------------------------------------------------------------------
# Context builders — one pair (no-edit / with-edit) per touched template.
# Kept side-effect free so a golden-capture script can import and call them
# directly (see scratchpad generator used once before the template edits).
# ---------------------------------------------------------------------------


def _generate_name_dd_item(*, with_edit: bool) -> dict:
    item = {
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "ids_name": "equilibrium",
        "description": "Poloidal flux",
        "unit": "Wb",
        "data_type": "FLT_1D",
        "physics_domain": "equilibrium",
        "parent_path": "equilibrium/time_slice/profiles_1d",
        "parent_description": "1D profiles",
        "parent_type": "STRUCTURE",
        "review_feedback": {
            "previous_name": "poloidal_flux_old",
            "previous_description": "Old description text.",
            "previous_documentation": "Multi-paragraph prior doc.",
            "reviewer_score": 0.45,
            "review_tier": "inadequate",
            "reviewer_comments": "Name lacks locus distinguisher.",
            "reviewer_scores": {
                "grammar": 12,
                "semantic": 10,
                "convention": 14,
                "completeness": 9,
            },
            "reviewer_suggested_name": "poloidal_magnetic_flux",
            "reviewer_suggestion_justification": "Cluster siblings use _magnetic_.",
        },
    }
    if with_edit:
        item["review_feedback"]["name_hint"] = "lean on the flux-surface locus"
        item["review_feedback"]["edit_reason"] = (
            "Domain review flagged the locus as ambiguous across devices."
        )
        item["review_feedback"]["edit_origin"] = "human"
    return item


def _generate_name_dd_context(*, with_edit: bool) -> dict:
    return {
        "items": [_generate_name_dd_item(with_edit=with_edit)],
        "ids_contexts": [],
        "reference_exemplars": [],
        "nearby_existing_names": [],
        "existing_names": [],
    }


def _refine_name_context(*, with_edit: bool) -> dict:
    item = {
        "path": "equilibrium/time_slice/boundary/lcfs_electron_temp",
        "ids_name": "equilibrium",
        "description": "Electron temperature at the last closed flux surface",
        "unit": "eV",
        "data_type": "FLT_0D",
        "physics_domain": "equilibrium",
        "parent_path": "equilibrium/time_slice/boundary",
        "parent_description": "Plasma boundary",
    }
    if with_edit:
        item["name_hint"] = "prefer the separatrix locus token"
        item["edit_reason"] = "Prior attempts conflated LCFS with separatrix."
        item["edit_origin"] = "agent"
    return {
        "item": item,
        "hybrid_neighbours": [],
        "chain_history": [],
        "chain_length": 0,
    }


def _refine_docs_context(*, with_edit: bool) -> dict:
    ctx: dict = {
        "sn_name": "electron_temperature_at_lcfs",
        "unit": "eV",
        "kind": "scalar",
        "physics_domain": "equilibrium",
        "description": "Electron temperature at the last closed flux surface",
        "dd_paths": [],
        "docs_chain_history": [],
        "docs_chain_length": 0,
        "reviewer_score_docs": None,
        "reviewer_comments_per_dim_docs": None,
    }
    if with_edit:
        ctx["docs_hint"] = "emphasize the Thomson-scattering measurement basis"
        ctx["edit_reason"] = "Docs omitted the standard measurement method."
        ctx["edit_origin"] = "human"
    return ctx


def _generate_docs_context(*, with_edit: bool) -> dict:
    item = {
        "name": "electron_temperature_at_lcfs",
        "unit": "eV",
        "kind": "scalar",
        "physics_domain": "equilibrium",
    }
    if with_edit:
        item["docs_hint"] = "emphasize the Thomson-scattering measurement basis"
        item["edit_reason"] = "Docs omitted the standard measurement method."
        item["edit_origin"] = "human"
    return {
        "item": item,
        "nearby_existing_names": [],
    }


def _review_names_context(*, with_edit: bool) -> dict:
    item = {
        "id": "core_profiles__electron_temperature",
        "source_id": "core_profiles/profiles_1d/electrons/temperature",
        "standard_name": "electron_temperature",
        "unit": "eV",
        "kind": "scalar",
        "physical_base": "temperature",
        "subject": "electron",
    }
    if with_edit:
        item["edit_reason"] = "Expert redirected the subject qualifier."
        item["edit_origin"] = "human"
    return {
        "items": [item],
        "nearby_existing_names": [],
    }


def _review_docs_context(*, with_edit: bool) -> dict:
    item = {
        "id": "electron_temperature_at_lcfs",
        "unit": "eV",
        "kind": "scalar",
        "physics_domain": "equilibrium",
        "description": "Electron temperature at the last closed flux surface.",
        "documentation": "The electron temperature $T_e$ at the LCFS.",
    }
    if with_edit:
        item["edit_reason"] = "Expert redirected the documentation emphasis."
        item["edit_origin"] = "agent"
    return {"item": item}


def _review_docs_parent_context(*, with_edit: bool) -> dict:
    item = {
        "id": "perturbed_velocity",
        "unit": "m.s^-1",
        "kind": "scalar",
        "physics_domain": "mhd",
        "description": "General perturbed velocity.",
        "documentation": "The perturbed velocity across MHD contexts.",
        "derived_children": [
            {"name": "normalized_parallel_perturbed_velocity", "unit": "m.s^-1"},
        ],
    }
    if with_edit:
        item["edit_reason"] = "Expert redirected the generalization scope."
        item["edit_origin"] = "human"
    return {"item": item}


# name -> (template_path, context_builder)
_TEMPLATES: dict[str, tuple[str, callable]] = {
    "generate_name_dd": ("sn/generate_name_dd", _generate_name_dd_context),
    "generate_name_dd_names": (
        "sn/generate_name_dd_names",
        _generate_name_dd_context,
    ),
    "refine_name_user": ("sn/refine_name_user", _refine_name_context),
    "refine_docs_user": ("sn/refine_docs_user", _refine_docs_context),
    "generate_docs_user": ("sn/generate_docs_user", _generate_docs_context),
    "review_names_user": ("sn/review_names_user", _review_names_context),
    "review_docs_user": ("sn/review_docs_user", _review_docs_context),
    "review_docs_parent_user": (
        "sn/review_docs_parent_user",
        _review_docs_parent_context,
    ),
}

# Templates where the injected block is a subordinate steering hint
# (generate regen path + refine paths).
_STEERING_TEMPLATES = {
    "generate_name_dd",
    "generate_name_dd_names",
    "refine_name_user",
    "refine_docs_user",
    "generate_docs_user",
}

# Templates where the injected block is the reason-only anti-revert context
# fed to reviewers.
_REVIEW_TEMPLATES = {
    "review_names_user",
    "review_docs_user",
    "review_docs_parent_user",
}


def _golden_path(key: str) -> Path:
    return GOLDEN_DIR / f"{key}.txt"


# ---------------------------------------------------------------------------
# (a) No-edit renders are byte-identical to the pre-change golden capture
# ---------------------------------------------------------------------------


def test_golden_files_exist():
    missing = [key for key in _TEMPLATES if not _golden_path(key).exists()]
    assert not missing, (
        f"Missing golden capture(s) for {missing} — run the golden generator "
        "before editing templates."
    )


def test_no_edit_render_matches_golden():
    for key, (template, builder) in _TEMPLATES.items():
        rendered = render_prompt(template, builder(with_edit=False))
        golden = _golden_path(key).read_text()
        assert rendered == golden, (
            f"{template}: no-edit render diverged from golden capture — "
            "the edit-steering change must be a no-op when no edit fields "
            "are present."
        )


# ---------------------------------------------------------------------------
# (b) With-edit renders carry the steering/reason block
# ---------------------------------------------------------------------------


def test_steering_block_present_with_edit_fields():
    for key in _STEERING_TEMPLATES:
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=True))
        assert "Expert steering" in rendered, (
            f"{template}: steering block missing when edit fields are present"
        )
        assert _SUBORDINATION_SENTENCE in rendered, (
            f"{template}: missing subordination sentence in steering block"
        )


def test_steering_block_absent_without_edit_fields():
    for key in _STEERING_TEMPLATES:
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=False))
        assert "Expert steering" not in rendered, (
            f"{template}: steering block leaked with no edit fields present"
        )


def test_steering_block_contains_hint_and_reason_text():
    name_hint_cases = {
        "generate_name_dd": "lean on the flux-surface locus",
        "generate_name_dd_names": "lean on the flux-surface locus",
        "refine_name_user": "prefer the separatrix locus token",
    }
    docs_hint_cases = {
        "refine_docs_user": "emphasize the Thomson-scattering measurement basis",
        "generate_docs_user": "emphasize the Thomson-scattering measurement basis",
    }
    reason_cases = {
        "generate_name_dd": "Domain review flagged the locus as ambiguous across devices.",
        "generate_name_dd_names": "Domain review flagged the locus as ambiguous across devices.",
        "refine_name_user": "Prior attempts conflated LCFS with separatrix.",
        "refine_docs_user": "Docs omitted the standard measurement method.",
        "generate_docs_user": "Docs omitted the standard measurement method.",
    }
    for key, hint in {**name_hint_cases, **docs_hint_cases}.items():
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=True))
        assert hint in rendered, f"{template}: name/docs hint text missing"
        assert reason_cases[key] in rendered, f"{template}: edit_reason text missing"


def test_steering_block_reports_origin():
    template, builder = _TEMPLATES["refine_name_user"]
    rendered = render_prompt(template, builder(with_edit=True))
    assert "agent" in rendered


# ---------------------------------------------------------------------------
# (c) Review templates: reason-only anti-revert block, never pre-approval
# ---------------------------------------------------------------------------


def test_review_reason_block_present_with_edit_reason():
    for key in _REVIEW_TEMPLATES:
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=True))
        assert "deliberately steered" in rendered.lower() or (
            "deliberate expert steering" in rendered.lower()
        ), f"{template}: anti-revert reviewer block missing"
        assert _DO_NOT_PENALIZE_SENTENCE in rendered, (
            f"{template}: missing do-not-penalize directive"
        )


def test_review_reason_block_contains_reason_text():
    reason_cases = {
        "review_names_user": "Expert redirected the subject qualifier.",
        "review_docs_user": "Expert redirected the documentation emphasis.",
        "review_docs_parent_user": "Expert redirected the generalization scope.",
    }
    for key, reason in reason_cases.items():
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=True))
        assert reason in rendered, f"{template}: edit_reason text missing"


def test_review_reason_block_absent_without_edit_reason():
    for key in _REVIEW_TEMPLATES:
        template, builder = _TEMPLATES[key]
        rendered = render_prompt(template, builder(with_edit=False))
        assert "deliberate" not in rendered.lower(), (
            f"{template}: anti-revert reviewer block leaked with no edit_reason"
        )


def test_review_block_never_instructs_acceptance():
    for key in _REVIEW_TEMPLATES:
        template, builder = _TEMPLATES[key]
        rendered_lower = render_prompt(template, builder(with_edit=True)).lower()
        for phrase in _ACCEPTANCE_PHRASES:
            assert phrase not in rendered_lower, (
                f"{template}: review block contains acceptance-instruction "
                f"phrase '{phrase}' — scoring must stay independent"
            )
