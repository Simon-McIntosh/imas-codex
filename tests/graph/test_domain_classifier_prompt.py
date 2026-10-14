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


def test_category_block_is_rendered_in_precedence_order(monkeypatch) -> None:
    """The categories reach the prompt in precedence order, not file order.

    The template tells the classifier the categories are in precedence order, so
    a category block rendered in file-insertion order would state a rule the
    list itself contradicts. The guidance is supplied here with its categories
    in reverse precedence order: an explicit sort is the only way the rendered
    order can follow precedence, so removing that sort fails this test.
    """
    from imas_codex.definitions import physics as physics_defs
    from imas_codex.graph.dd_domain_classifier import (
        _category_precedence_rank,
        _precedence_rank,
    )
    from imas_codex.llm import prompt_loader

    rank = _category_precedence_rank(_precedence_rank())
    guidance = load_domain_guidance()
    categories = guidance["categories"]
    assert categories and rank, "check cannot see its subject"

    scrambled = dict(
        sorted(
            categories.items(),
            key=lambda item: (-rank.get(item[0], len(rank)), item[0]),
        )
    )
    expected = sorted(scrambled, key=lambda name: (rank[name], name))
    assert expected != list(scrambled), "scramble is a no-op — the test is vacuous"

    def guidance_in_file_order():
        return {**guidance, "categories": scrambled}

    monkeypatch.setattr(physics_defs, "load_domain_guidance", guidance_in_file_order)
    monkeypatch.setattr(
        prompt_loader,
        "_SCHEMA_PROVIDERS",
        {
            **prompt_loader._SCHEMA_PROVIDERS,
            "physics_domains": prompt_loader._provide_physics_domains,
        },
    )

    prompt_loader._provide_physics_domains.cache_clear()
    try:
        rendered = [
            entry["value"]
            for entry in prompt_loader._provide_physics_domains()[
                "physics_domain_categories"
            ]
        ]
        assert rendered == expected
    finally:
        prompt_loader._provide_physics_domains.cache_clear()

    prompt = render_prompt("imas/domain_classifier", {"paths": []})
    positions = [prompt.index(name) for name in expected]
    assert positions == sorted(positions), "categories not ordered by precedence"
