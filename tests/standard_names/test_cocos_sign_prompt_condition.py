"""Documentation prompts require sign prose only for COCOS-sensitive quantities."""

from __future__ import annotations

import re

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt

_MANDATORY_SIGN_INSTRUCTIONS = (
    "You **MUST** include a sign convention paragraph",
    "Sign convention is REQUIRED for this quantity",
)


def _render_generate_docs(cocos_transformation_type: str) -> str:
    return render_prompt(
        "sn/generate_docs_user",
        {
            "item": {
                "name": "magnetic_field",
                "unit": "T",
                "kind": "vector",
                "physics_domain": "magnetics",
                "cocos_transformation_type": cocos_transformation_type,
                "cocos_guidance": (
                    "Positive when the vacuum toroidal magnetic field points "
                    "toward increasing toroidal angle."
                ),
            },
            "chain_history": [],
            "nearby_existing_names": [],
        },
    )


def test_invariant_quantity_has_no_mandatory_sign_instruction() -> None:
    rendered = _render_generate_docs("one_like")

    assert "COCOS Sign Convention" not in rendered
    for instruction in _MANDATORY_SIGN_INSTRUCTIONS:
        assert instruction not in rendered


def test_sensitive_quantity_retains_mandatory_sign_instruction() -> None:
    rendered = _render_generate_docs("b0_like")

    assert "COCOS Sign Convention" in rendered
    assert "b0_like" in rendered
    for instruction in _MANDATORY_SIGN_INSTRUCTIONS:
        assert instruction in rendered


def test_docs_prompts_do_not_gate_on_bare_cocos_label_truthiness() -> None:
    bare_requirement = re.compile(
        r"{%\s*if\s+(?:[A-Za-z_]\w*\.)?cocos_"
        r"(?:label|transformation_type)\s*%}"
        r"(?:(?!{%\s*endif\s*%}).)*"
        r"(?:\bMUST\b.{0,80}sign convention|"
        r"Sign convention.{0,80}\bREQUIRED\b)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    prompt_paths = sorted((PROMPTS_DIR / "sn").glob("*docs_user.md"))

    offenders = [
        path.name
        for path in prompt_paths
        if bare_requirement.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
