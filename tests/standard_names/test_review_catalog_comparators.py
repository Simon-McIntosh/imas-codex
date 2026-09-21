"""Catalog-comparator coverage for standalone standard-name review."""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.review.enrichment import build_neighborhood_context


def _catalog_name(
    name_id: str,
    *,
    unit: str,
    physical_base: str,
    name_stage: str = "accepted",
) -> dict:
    return {
        "id": name_id,
        "description": name_id.replace("_", " "),
        "kind": "scalar",
        "unit": unit,
        "physical_base": physical_base,
        "name_stage": name_stage,
        "review_tier": "good",
    }


def test_every_batch_item_contributes_a_vector_query(monkeypatch) -> None:
    queries: list[str] = []

    def search(query: str, *, k: int) -> list[dict]:
        queries.append(query)
        return [
            _catalog_name(
                f"peer_{query.rsplit(maxsplit=1)[-1]}",
                unit="1",
                physical_base="peer",
            )
        ]

    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector", search
    )
    names = [
        _catalog_name(
            f"candidate_{index}",
            unit="1",
            physical_base=f"candidate_{index}",
        )
        for index in range(25)
    ]

    build_neighborhood_context(
        {
            "names": names,
            "cluster": {"cluster_label": "shared cluster"},
        },
        names,
        k=10,
    )

    assert queries == [name["description"] for name in names]


def test_same_unit_different_base_accepted_name_is_a_comparator(
    monkeypatch,
) -> None:
    unit = "m^-2.s^-1.sr^-1"
    brightness = _catalog_name(
        "hard_xray_brightness",
        unit=unit,
        physical_base="brightness",
    )
    photon_radiance = _catalog_name(
        "photon_radiance_of_hard_xray",
        unit=unit,
        physical_base="photon_radiance",
    )
    same_base = _catalog_name(
        "soft_xray_brightness",
        unit=unit,
        physical_base="brightness",
    )
    unaccepted_other_base = _catalog_name(
        "draft_radiance",
        unit=unit,
        physical_base="radiance",
        name_stage="drafted",
    )
    accepted_other_unit = _catalog_name(
        "spectral_photon_radiance",
        unit="m^-2.s^-1.sr^-1.m^-1",
        physical_base="photon_radiance",
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector",
        lambda _query, *, k: [],
    )

    comparators = build_neighborhood_context(
        {"names": [brightness], "cluster": None},
        [
            brightness,
            photon_radiance,
            same_base,
            unaccepted_other_base,
            accepted_other_unit,
        ],
        k=10,
    )

    comparator_ids = {comparator["id"] for comparator in comparators}
    assert "photon_radiance_of_hard_xray" in comparator_ids
    assert "soft_xray_brightness" not in comparator_ids
    assert "draft_radiance" not in comparator_ids
    assert "spectral_photon_radiance" not in comparator_ids


@pytest.mark.asyncio
async def test_standalone_review_prompt_receives_catalog_names(monkeypatch) -> None:
    from imas_codex.discovery.base import llm as llm_module
    from imas_codex.llm import prompt_loader
    from imas_codex.standard_names.review.pipeline import _review_single_batch

    rendered_contexts: list[dict] = []

    def render_prompt(_prompt_name: str, context: dict) -> str:
        rendered_contexts.append(context)
        return "prompt"

    async def call_llm_structured(**kwargs):
        return kwargs["response_model"](reviews=[]), 0.0, 0

    monkeypatch.setattr(prompt_loader, "render_prompt", render_prompt)
    monkeypatch.setattr(llm_module, "acall_llm_structured", call_llm_structured)

    await _review_single_batch(
        names=[],
        model="test-model",
        grammar_enums={},
        compose_ctx={},
        batch_context="",
        neighborhood=[],
        audit_findings=[],
        wlog=logging.LoggerAdapter(logging.getLogger("test"), {}),
        existing_names=["hard_xray_brightness", "photon_radiance_of_hard_xray"],
    )

    user_context = rendered_contexts[-1]
    assert user_context["existing_names"] == [
        "hard_xray_brightness",
        "photon_radiance_of_hard_xray",
    ]
