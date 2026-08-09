"""Name-review grounding from exact sources and the public grammar IR."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.llm.prompt_loader import render_prompt

ISOTOPE_RATIO = (
    "ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density"
    "_and_neutral_density_of_isotope"
)
SOURCE_PATH = "core_profiles/profiles_1d/neutrals/isotope/density"
SECOND_SOURCE_PATH = "edge_profiles/profiles_1d/neutrals/isotope/density"

SOURCE_AXIS_MARKERS = (
    "subject/object",
    "mechanism/cause",
    "locus/carrier",
    "projection/axis",
    "surface kind",
    "geometry representation",
    "coordinate kind",
    "aggregation",
    "process",
    "state",
    "unit",
    "dd-authoritative transformation/label semantics",
)


def _source_binding() -> dict:
    return {
        "id": f"dd:{SOURCE_PATH}",
        "source_type": "dd",
        "source_id": SOURCE_PATH,
        "dd_path": SOURCE_PATH,
        "dd_version": "4.1.0",
        "dd_documentation": (
            "Density of neutral particles for the array-selected isotope."
        ),
        "dd_snapshot_pinned": True,
        "dd_parent_path": "core_profiles/profiles_1d/neutrals/isotope",
        "dd_parent_documentation": "Set of isotopes.",
        "dd_data_type": "FLT_1D",
        "dd_unit": "m^-3",
        "dd_coordinates": ["core_profiles/time"],
        "dd_lifecycle_status": "active",
        "enhanced_description": "Neutral isotope density profile.",
        "compose_hint": "Use the ratio n_i / (n_total - n_i).",
        "compose_hint_reason": "The source formula fixes nested operand scope.",
    }


def _second_source_binding() -> dict:
    return {
        "id": f"dd:{SECOND_SOURCE_PATH}",
        "source_type": "dd",
        "source_id": SECOND_SOURCE_PATH,
        "dd_path": SECOND_SOURCE_PATH,
        "dd_version": "4.1.0",
        "dd_documentation": "Density of an edge neutral isotope member.",
        "dd_snapshot_pinned": True,
        "dd_parent_path": "edge_profiles/profiles_1d/neutrals/isotope",
        "dd_parent_documentation": "Edge neutral isotope array.",
        "dd_data_type": "FLT_1D",
        "dd_unit": "m^-3",
        "dd_coordinates": ["edge_profiles/time"],
        "dd_lifecycle_status": "active",
        "enhanced_description": "Edge neutral isotope density profile.",
        "compose_hint": "Retain the selected isotope member locus.",
        "compose_hint_reason": "The array index selects the isotope.",
    }


def _repeated_source_binding(index: int) -> dict:
    path = f"shared_profiles/source_{index:02d}/density"
    binding = _source_binding()
    binding.update(
        {
            "id": f"dd:{path}",
            "source_id": path,
            "dd_path": path,
            "dd_documentation": "Shared neutral isotope density definition.",
            "dd_parent_path": f"shared_profiles/source_{index:02d}",
            "dd_parent_documentation": "Shared isotope array definition.",
            "enhanced_description": "Shared neutral isotope density context.",
            "compose_hint": "",
            "compose_hint_reason": "",
        }
    )
    return binding


def _review_item() -> dict:
    return {
        "id": ISOTOPE_RATIO,
        "description": "Neutral isotope density fraction relative to the remainder.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "core_plasma_physics",
        "source_bindings": [_source_binding()],
    }


@pytest.mark.parametrize("prompt_name", ("sn/review", "sn/review_names"))
def test_name_review_prompts_enforce_complete_source_axis_contract(
    prompt_name: str,
) -> None:
    """Both name-review paths reject semantic loss instead of rewarding brevity."""
    rendered = render_prompt(
        prompt_name,
        {
            "items": [],
            "nearby_existing_names": [],
            "review_scored_examples": [],
            "batch_context": "",
        },
    ).lower()

    for marker in SOURCE_AXIS_MARKERS:
        assert marker in rendered
    assert "semantic accuracy" in rendered
    assert "5/20" in rendered
    assert "vocab_gap" in rendered
    assert "poloidal_cross_sectional_area_of_flux_surface" in rendered
    assert "surface_area_of_flux_surface" in rendered
    assert "area_of_flux_surface" in rendered
    assert "ambiguous umbrella" in rendered
    assert "cocos is fixed ddv4 catalog metadata" in rendered
    assert "psi_like" in rendered
    assert "ip_like" in rendered


@pytest.mark.parametrize("prompt_name", ("sn/review", "sn/review_names"))
def test_name_review_prompts_reject_non_line_of_sight_geometry_collapse(
    prompt_name: str,
) -> None:
    """Review catches ordinal removal that changes the geometry representation."""
    rendered = render_prompt(
        prompt_name,
        {
            "items": [],
            "nearby_existing_names": [],
            "review_scored_examples": [],
            "batch_context": "",
        },
    ).lower()

    for counterexample in (
        "thick_line",
        "pellet",
        "gas pipe",
        "shunt",
        "beam path",
        "interpolation",
        "other-object outline",
    ):
        assert counterexample in rendered
    assert "only genuine" in rendered
    assert "line_of_sight" in rendered


@pytest.mark.parametrize("prompt_name", ("sn/review", "sn/review_names"))
def test_name_review_prompts_reject_endpoint_position_identity(
    prompt_name: str,
) -> None:
    """Review never approves a name carrying endpoint ordering."""
    rendered = " ".join(
        render_prompt(
            prompt_name,
            {
                "items": [],
                "nearby_existing_names": [],
                "review_scored_examples": [],
                "batch_context": "",
            },
        )
        .lower()
        .split()
    )

    assert "never emit, propose, approve, or refine" in rendered
    assert "first, second, third, start, end" in rendered
    assert "radial_coordinate_of_arc_of_circle_start_point" in rendered
    assert "same arc-of-circle carrier" in rendered
    assert "vocab_gap" in rendered


@pytest.mark.parametrize("prompt_name", ("sn/review", "sn/review_names"))
def test_name_review_prompts_use_public_flux_area_decomposition(
    prompt_name: str,
) -> None:
    """Reviewers do not treat projection and qualifier tokens as atomic bases."""
    source = render_prompt(
        prompt_name,
        {
            "items": [],
            "nearby_existing_names": [],
            "review_scored_examples": [],
            "batch_context": "",
        },
    )

    assert "projection=`poloidal` + qualifier=`cross_sectional` + base=`area`" in source
    assert "`poloidal_flux` is a lexicalised atomic term" not in source
    assert "Allow lexicalised atomic compounds (`poloidal_flux`" not in source
    assert "Compound `physical_base` tokens like `poloidal_flux`" not in source
    assert "`cross_sectional_area`, `safety_factor`" not in source


def test_review_claim_projects_exact_pinned_source_bindings() -> None:
    """The claim carries enough exact-source state for immutable DD grounding."""
    from imas_codex.standard_names import graph_ops

    with (
        patch.object(graph_ops, "_claim_sn_atomic", return_value=[]) as claim,
        patch.object(graph_ops, "_verify_name_claim_winners", return_value=[]),
    ):
        graph_ops.claim_review_name_batch(batch_size=3)

    projection = claim.call_args.kwargs["extra_return_fields"]
    assert "PRODUCED_NAME" in projection
    assert "source.source_id" in projection
    assert "source.dd_version" in projection
    assert "source.dd_documentation" in projection
    assert "source.dd_snapshot_pinned" in projection
    assert "source.dd_parent_documentation" in projection
    assert "source.compose_hint" in projection


def test_strict_review_context_preserves_recursive_isotope_ratio() -> None:
    """The lossless adapter exposes full fields and nested operator scope."""
    from imas_codex.standard_names.graph_ops import strict_review_grammar_context

    context = strict_review_grammar_context(ISOTOPE_RATIO)
    projection = {
        entry["field"]: entry["value"] for entry in context["grammar_projection"]
    }

    assert context["grammar_round_trip"] == ISOTOPE_RATIO
    assert context["semantic_ir"] == (
        "ratio(neutral_density_of_isotope, "
        "difference(total_neutral_density, neutral_density_of_isotope))"
    )
    assert projection["physical_base"] == "density"
    assert projection["subject"] == "neutral"
    assert projection["object"] == "isotope"
    assert projection["transformation"] == "ratio"


def test_review_enrichment_keeps_pins_and_reuses_compose_context() -> None:
    """Pinned text stays authoritative while compose enrichment adds neighbors."""
    from imas_codex.standard_names import workers

    item = _review_item()

    def enrich(stubs: list[dict]) -> None:
        assert stubs[0]["path"] == SOURCE_PATH
        stubs[0]["identifier_schema"] = "isotope_identifier"
        stubs[0]["identifier_values"] = [
            {"name": "deuterium", "index": 1, "description": "D"}
        ]
        stubs[0]["parent_description"] = "Current graph enrichment."

    with patch.object(workers, "_enrich_batch_items", side_effect=enrich) as canonical:
        workers._enrich_name_review_items([item])

    canonical.assert_called_once()
    assert item["source_id"] == SOURCE_PATH
    assert item["source_paths"] == [SOURCE_PATH]
    assert item["dd_source_docs"][0]["documentation"] == (
        "Density of neutral particles for the array-selected isotope."
    )
    assert item["dd_source_docs"][0]["snapshot_pinned"] is True
    assert item["dd_parent_contexts"][0]["documentation"] == "Set of isotopes."
    assert item["source_hints"][0]["hint"] == ("Use the ratio n_i / (n_total - n_i).")
    assert item["identifier_schema"] == "isotope_identifier"
    assert item["parent_description"] == "Current graph enrichment."
    assert item["description"] == (
        "Neutral isotope density fraction relative to the remainder."
    )


def test_multi_source_review_prompt_renders_complete_context_deterministically() -> (
    None
):
    """The rendered prompt preserves every pinned and canonical context channel."""
    from imas_codex.standard_names import workers

    item = _review_item()
    repeated_bindings = [_repeated_source_binding(index) for index in range(11)]
    item["source_bindings"] = list(
        reversed([_source_binding(), _second_source_binding(), *repeated_bindings])
    )

    def enrich(stubs: list[dict]) -> None:
        assert stubs[0]["path"] == SOURCE_PATH
        stubs[0].update(
            {
                "parent_path": "core_profiles/profiles_1d/neutrals/isotope",
                "parent_description": "Current isotope array description.",
                "ancestor_context": [
                    {
                        "path": "core_profiles/profiles_1d/neutrals",
                        "text": "Neutral population profiles by species.",
                    }
                ],
                "identifier_schema": "isotope_identifier",
                "identifier_schema_doc": "Selects an isotope array member.",
                "identifier_values": [
                    {"name": "deuterium", "index": 1, "description": "D"}
                ],
                "clusters": [
                    {
                        "label": "neutral-density-family",
                        "scope": "global",
                        "description": "Neutral density quantities.",
                        "members": [SECOND_SOURCE_PATH],
                    }
                ],
                "cross_ids_paths": [SECOND_SOURCE_PATH],
                "dd_paths_docs": {
                    "z_profiles/member/density": (
                        "Supplementary zeta member definition."
                    ),
                    "a_profiles/member/density": (
                        "Supplementary alpha member definition."
                    ),
                },
                "hybrid_neighbours": [
                    {
                        "tag": "core_profiles/neutral_density",
                        "unit": "m^-3",
                        "physics_domain": "core_plasma_physics",
                        "doc_short": "Neutral density comparison quantity.",
                        "cocos_label": "",
                    }
                ],
                "related_neighbours": [
                    {
                        "path": SECOND_SOURCE_PATH,
                        "ids": "edge_profiles",
                        "relationship_type": "HAS_CLUSTER",
                        "via": "neutral-density-family",
                    }
                ],
                "error_fields": [f"{SOURCE_PATH}_error_upper"],
                "sibling_fields": [
                    {
                        "path": f"{SOURCE_PATH}_thermal",
                        "description": "Thermal neutral density.",
                        "data_type": "FLT_1D",
                    }
                ],
                "version_history": [
                    {"version": "4.1.0", "change_type": "definition_clarification"}
                ],
            }
        )

    with patch.object(workers, "_enrich_batch_items", side_effect=enrich):
        workers._enrich_name_review_items([item])

    context = {
        "items": [item],
        "batch_context": "",
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
        "nearby_existing_names": [],
        "review_scored_examples": [],
        "prior_reviews": [],
    }
    rendered = render_prompt("sn/review_names_user", context)

    rendered_repeated_paths = [
        f"shared_profiles/source_{index:02d}/density" for index in range(6)
    ]
    assert item["source_paths"] == [
        SOURCE_PATH,
        SECOND_SOURCE_PATH,
        *rendered_repeated_paths,
    ]
    assert item["source_context_omitted"] == 5
    assert SOURCE_PATH in rendered
    assert SECOND_SOURCE_PATH in rendered
    assert rendered.index(
        "Density of neutral particles for the array-selected isotope."
    ) < rendered.index("Density of an edge neutral isotope member.")
    assert "Set of isotopes." in rendered
    assert "Edge neutral isotope array." in rendered
    assert "Density of neutral particles for the array-selected isotope." in rendered
    assert rendered.count("Shared neutral isotope density definition.") == 1
    assert rendered.count("Shared isotope array definition.") == 1
    assert "5 additional exact source binding(s) omitted" in rendered
    assert "shared_profiles/source_05/density" in rendered
    assert "shared_profiles/source_06/density" not in rendered
    assert "subject=neutral" in rendered
    assert "object=isotope" in rendered
    assert "transformation=ratio" in rendered
    assert (
        "ratio(neutral_density_of_isotope, "
        "difference(total_neutral_density, neutral_density_of_isotope))"
    ) in rendered
    assert "intentional\nparameter" in rendered
    assert "Use the ratio n_i / (n_total - n_i)." in rendered
    assert "Current isotope array description." in rendered
    assert "Neutral population profiles by species." in rendered
    assert "isotope_identifier" in rendered
    assert "Selects an isotope array member." in rendered
    assert "neutral-density-family" in rendered
    assert "definition_clarification (v4.1.0)" in rendered
    assert "supplementary and non-authoritative" in rendered
    assert "pinned source snapshots above are the sole definition authority" in rendered
    assert rendered.index("Supplementary alpha member definition.") < rendered.index(
        "Supplementary zeta member definition."
    )
    assert "authoritative clauses for related leaves" not in rendered
    assert "Neutral density comparison quantity." in rendered
    assert "HAS_CLUSTER via neutral-density-family" in rendered
    assert f"{SOURCE_PATH}_error_upper" in rendered
    assert f"{SOURCE_PATH}_thermal" in rendered
    assert "Authoritative Escalation Context" not in rendered


def test_unpinned_source_context_is_explicitly_non_authoritative() -> None:
    """Mutable source text never masquerades as an immutable DD snapshot."""
    from imas_codex.standard_names import workers

    item = _review_item()
    item["source_bindings"][0]["dd_snapshot_pinned"] = False
    with patch.object(workers, "_enrich_batch_items", return_value=None):
        workers._enrich_name_review_items([item])

    context = {
        "items": [item],
        "batch_context": "",
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
        "nearby_existing_names": [],
        "review_scored_examples": [],
        "prior_reviews": [],
    }
    rendered = render_prompt("sn/review_names_user", context)

    assert item["unpinned_source_count"] == 1
    assert item["dd_source_docs"][0]["snapshot_pinned"] is False
    assert "Provenance incomplete" in rendered
    assert "Unpinned DD definition (non-authoritative)" in rendered
    assert "Unpinned parent definition (non-authoritative)" in rendered
    assert "Pinned immutable DD definition (authoritative)" not in rendered
    assert "Pinned immutable parent definition (authoritative)" not in rendered


@pytest.mark.asyncio
async def test_authoritative_escalation_alone_receives_both_critiques() -> None:
    """Blind reviewers stay independent and the escalator sees both critiques."""
    from imas_codex.standard_names.workers import _run_rd_quorum_cycles

    calls: list[dict] = []
    scores = [
        {"grammar": 20, "semantic": 18, "convention": 18, "completeness": 18},
        {"grammar": 8, "semantic": 18, "convention": 18, "completeness": 18},
        {"grammar": 19, "semantic": 19, "convention": 19, "completeness": 19},
    ]
    reasons = ["primary critique", "secondary critique", "resolved critique"]

    async def call_llm(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        values = scores[index]
        score = sum(values.values()) / 80.0
        result = SimpleNamespace(
            scores=SimpleNamespace(score=score, model_dump=lambda: values),
            comments=None,
            reasoning=reasons[index],
        )
        return result, 0.01, 100

    def escalation_prompt(prior_reviews: list[dict]) -> str:
        assert [review["role"] for review in prior_reviews] == [
            "primary",
            "secondary",
        ]
        return "escalation prompt\n" + "\n".join(
            review["reasoning"] for review in prior_reviews
        )

    result = await _run_rd_quorum_cycles(
        sn_id=ISOTOPE_RATIO,
        review_axis="names",
        response_model=object,
        user_prompt="blind prompt",
        system_prompt="static system prompt",
        models=["primary-model", "secondary-model", "escalation-model"],
        disagreement_threshold=0.20,
        rubric_dims=("grammar", "semantic", "convention", "completeness"),
        lease=None,
        phase="review_name",
        acall_llm_structured=call_llm,
        escalation_prompt_factory=escalation_prompt,
    )

    assert result is not None
    assert result["resolution_method"] == "authoritative_escalation"
    prompts = [call["messages"][1]["content"] for call in calls]
    assert prompts[:2] == ["blind prompt", "blind prompt"]
    assert "primary critique" not in prompts[0]
    assert "secondary critique" not in prompts[1]
    assert "primary critique" in prompts[2]
    assert "secondary critique" in prompts[2]
