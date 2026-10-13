"""The DD domain classifier prompt must offer the whole PhysicsDomain vocabulary.

The classifier can only return a domain the prompt lists, so the vocabulary is
rendered from the ``PhysicsDomain`` enum rather than hand-typed. These tests
hold the prompt to that: every enum value must appear, and the list must track
the provider rather than a frozen table.
"""

from __future__ import annotations

from imas_codex.core.physics_domain import PhysicsDomain
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
