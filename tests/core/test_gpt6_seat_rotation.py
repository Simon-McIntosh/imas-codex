"""Every live GPT seat resolves to the GPT-6 model the seat rotation assigned.

The seat bench moved each production GPT seat onto a GPT-6 model: the two seats
the bench shows ``gpt-6-luna`` capable for take it, the seats it does not take
``gpt-6.1-sol``, and the seats with no bench move generation for generation.

The rotation is recorded in
``docs/evidence/fragments/llm-seats/llm-gpt6-seat-rotation.html``. This test
reads every rotated seat back through the project's own model accessor, so a
seat that drifts off its assigned model fails here rather than at the first
pipeline call that consumes it.

The scalar seats are read with :func:`imas_codex.settings.get_model`, the docs
reviewer chain with :func:`imas_codex.settings.get_sn_review_docs_models`, the
production names profile with
:func:`imas_codex.settings.get_sn_review_profile_models`, and the top-level
names reviewer chain with
:func:`imas_codex.settings.get_model_source_models`, which is the source the
pipeline consults for the reviewer chain the active profile does not override.
"""

from __future__ import annotations

import pytest

from imas_codex import settings

# Scalar seats the rotation moved, and the GPT-6 model each must resolve to.
_ROTATED_SCALAR_SEATS = {
    "sn-refine": "openrouter/openai/gpt-6-luna",
    "sn-classifier": "openrouter/openai/gpt-6-luna",
    "sn-docs": "openrouter/openai/gpt-6.1-sol",
    "sn-prose-adjudicator": "openrouter/openai/gpt-6-luna",
    "cluster-labels": "openrouter/openai/gpt-6.1-sol",
}

# The names reviewer chain's third seat (the escalator) before the rotation; it
# must be gone from the names model source once that seat moves to GPT-6.
_RETIRED_NAMES_ESCALATOR = "openrouter/openai/gpt-5.5"


@pytest.fixture(autouse=True)
def _isolate_model_settings(monkeypatch):
    """Read the seats from pyproject, not from a lingering env override."""
    for env_var in settings._MODEL_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    settings._load_pyproject_settings.cache_clear()
    yield
    settings._load_pyproject_settings.cache_clear()


@pytest.mark.parametrize(("section", "expected"), sorted(_ROTATED_SCALAR_SEATS.items()))
def test_scalar_seat_resolves_to_its_rotated_model(section, expected):
    assert settings.get_model(section) == expected


def test_docs_reviewer_breaker_is_rotated():
    """The docs axis' context-aware breaker is the third-seat reviewer."""
    assert settings.get_sn_review_docs_models()[2] == "openrouter/openai/gpt-6.1-sol"


def test_default_profile_secondary_is_rotated():
    """The production names profile's blind secondary moved to GPT-6 Luna."""
    assert (
        settings.get_sn_review_profile_models("default")[1]
        == "openrouter/openai/gpt-6-luna"
    )


def test_names_escalator_is_rotated():
    """The top-level names escalator moved off its pre-rotation model."""
    names_source = settings.get_model_source_models("sn-review:names")
    assert "openrouter/openai/gpt-6.1-sol" in names_source
    assert _RETIRED_NAMES_ESCALATOR not in names_source


def test_model_defaults_name_no_gpt5_model():
    """Every fallback names a post-rotation model.

    A seat's pyproject entry shadows its default, so a stale default is
    invisible until a seat key is removed or a section is absent - and then it
    silently returns the pipeline to a superseded model. The fallbacks are
    rotated with the seats so that path stays on the current generation.
    """
    stale = {
        section: model
        for section, model in settings._MODEL_DEFAULTS.items()
        if "gpt-5" in model
    }
    assert stale == {}
