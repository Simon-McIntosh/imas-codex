"""The DD domain classifier prompt must offer the whole PhysicsDomain vocabulary.

The classifier can only return a domain the prompt lists, so the vocabulary is
rendered from the ``PhysicsDomain`` enum rather than hand-typed. These tests
hold the prompt to that: every enum value must appear, and the list must track
the provider rather than a frozen table.
"""

from __future__ import annotations

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.definitions.physics import load_domain_guidance
from imas_codex.llm.prompt_loader import render_prompt


def test_prompt_lists_every_physics_domain_value() -> None:
    prompt = render_prompt("imas/domain_classifier", {"paths": []})
    missing = [member.value for member in PhysicsDomain if member.value not in prompt]
    assert missing == [], f"domains absent from the classifier prompt: {missing}"


def test_vocabulary_follows_the_provider_not_a_frozen_table(monkeypatch) -> None:
    """A domain added upstream must appear without touching the prompt text."""
    from imas_codex.llm import prompt_loader

    delivered = prompt_loader._provide_physics_domains()["physics_domains"]
    added = {
        "value": "sentinel_domain_xyz",
        "description": "",
        "category": "uncategorized",
    }
    monkeypatch.setitem(
        prompt_loader._SCHEMA_PROVIDERS,
        "physics_domains",
        lambda: {"physics_domains": [*delivered, added]},
    )
    prompt = render_prompt("imas/domain_classifier", {"paths": []})
    assert "sentinel_domain_xyz" in prompt


def test_prompt_renders_the_category_block_with_each_scope() -> None:
    """Every category is listed beside the scope its guidance gives it.

    The first judgement is which category a path belongs to, so a category
    rendered without its scope would ask the question with nothing to judge
    against.
    """
    categories = load_domain_guidance()["categories"]
    assert categories, "guidance read empty — the check cannot see its subject"
    prompt = render_prompt("imas/domain_classifier", {"paths": []})
    missing = [
        name
        for name, entry in categories.items()
        if name not in prompt or (entry.get("scope") or "") not in prompt
    ]
    assert missing == [], f"categories absent from the prompt: {missing}"


def test_prompt_renders_each_domains_scope_includes_and_excludes() -> None:
    """Each domain's guidance reaches the prompt: scope, includes, exclusions.

    A domain the prompt lists without its guidance is a bare label, and the
    classifier would separate it from a neighbouring domain by name alone.
    """
    domains = load_domain_guidance()["domains"]
    assert domains, "guidance read empty — the check cannot see its subject"
    prompt = render_prompt("imas/domain_classifier", {"paths": []})
    missing: list[str] = []
    for name, entry in domains.items():
        if entry.get("scope") and entry["scope"] not in prompt:
            missing.append(f"{name}:scope")
        for include in entry.get("includes") or ():
            if include not in prompt:
                missing.append(f"{name}:includes:{include}")
        for exclusion in entry.get("excludes") or ():
            if exclusion["quantity"] not in prompt:
                missing.append(f"{name}:excludes:{exclusion['quantity']}")
    assert missing == [], f"guidance absent from the prompt: {missing}"
