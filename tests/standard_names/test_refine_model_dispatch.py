"""Regression tests for refine-model dispatch.

Pin the contract that ``process_refine_name_batch`` and
``process_refine_docs_batch`` resolve their model via
``get_model("sn-refine")`` — *not* ``get_model("sn-compose")`` (the
bulk-content tier).

This guards the failure mode where a cheap tier refines: a flash-lite or
haiku model lifted reviewed, critiqued names at about 5% against about
42% for the compose tier.  Refine therefore keeps its own pyproject
section, so its model can diverge from compose without coupling, and the
fallback in ``settings._MODEL_DEFAULTS`` holds the same capable tier the
section names; a fallback left behind would silently change the model
wherever the project configuration is absent.
"""

from __future__ import annotations

import pytest

from imas_codex import settings
from imas_codex.standard_names.worker_tasks import (
    worker_refine_docs as _owning_worker_refine_docs,
    worker_refine_names as _owning_worker_refine_names,
)

# ─── Section registration ────────────────────────────────────────────


def test_refine_section_registered() -> None:
    """``"sn-refine"`` is a valid model section (so ``get_model`` accepts it)."""
    assert "sn-refine" in settings.MODEL_SECTIONS


# The lines measured to lift reviewed, critiqued names at the rate refine
# needs. An allow-list, not a denylist of cheap-sounding names: a line that is
# newly served, renamed or re-tiered is refused until someone measures it and
# books it here.
REFINE_CAPABLE_LINES = frozenset({"openai/gpt-6-luna"})


def test_refine_default_holds_the_seated_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refine fallback is a capable tier and agrees with the seat.

    Two ways this guard can fail, both real: a config edit that drops the
    fallback to a cheap tier, which cannot lift critiqued names, and a seat
    edit that leaves the fallback behind, which silently changes the model
    wherever the project configuration is absent.
    """
    monkeypatch.delenv("IMAS_CODEX_SN_REFINE_MODEL", raising=False)
    # Bypass any pyproject override — we want to verify the *default*
    # baked into settings.py protects against accidental downgrades.
    model = settings._MODEL_DEFAULTS["sn-refine"].lower()
    line = model.removeprefix("openrouter/")
    assert line in REFINE_CAPABLE_LINES, (
        f"Default refine model {model!r} is not a line booked as capable of "
        f"refining critiqued names; booked lines are {sorted(REFINE_CAPABLE_LINES)}"
    )
    assert model == settings.get_model("sn-refine").lower(), (
        "the sn-refine fallback and the seat name different models, so the "
        "model changes where the project configuration is absent"
    )


def test_refine_env_override_independent_of_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IMAS_CODEX_SN_REFINE_MODEL`` is honoured without leaking to compose."""
    monkeypatch.setenv("IMAS_CODEX_SN_REFINE_MODEL", "test/refine-only")
    monkeypatch.setenv("IMAS_CODEX_SN_COMPOSE_MODEL", "test/compose-only")
    assert settings.get_model("sn-refine") == "test/refine-only"
    assert settings.get_model("sn-compose") == "test/compose-only"


def test_refine_does_not_alias_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch contract: refine and compose resolve via *different* keys.

    Future regressions where refine is silently routed back through
    ``get_model("sn-compose")`` would re-introduce the E3 failure mode.
    Override only the refine env var and assert compose is unaffected
    (this is the property we care about — distinct dispatch keys).
    """
    monkeypatch.setenv("IMAS_CODEX_SN_REFINE_MODEL", "test/refine-canary")
    monkeypatch.delenv("IMAS_CODEX_SN_COMPOSE_MODEL", raising=False)
    refine = settings.get_model("sn-refine")
    compose = settings.get_model("sn-compose")
    assert refine == "test/refine-canary"
    assert compose != "test/refine-canary"


# ─── Worker-level dispatch ───────────────────────────────────────────


def test_refine_name_worker_imports_refine_section() -> None:
    """``process_refine_name_batch`` resolves its model through the rotation.

    Source-level check: the batch must ask the rotation helper for its model
    rather than naming one itself, and the helper must reference the
    ``"sn-refine"`` settings key (not ``"language"`` or ``get_model("sn-compose")``)
    for the non-escalation branch.  This is cheaper and more stable than
    spinning up an asyncio mock harness with the full GraphClient + fanout
    fakes that the worker requires at runtime.
    """
    import inspect

    from imas_codex.standard_names import workers

    batch_src = inspect.getsource(_owning_worker_refine_names.process_refine_name_batch)
    assert "refine_model_for_attempt(" in batch_src, (
        "process_refine_name_batch must take its model from the rotation helper"
    )

    rotation_src = inspect.getsource(
        _owning_worker_refine_names.refine_model_for_attempt
    )
    assert 'get_model("sn-refine")' in rotation_src, (
        "the refine rotation must resolve the non-escalation model from the "
        "sn-refine seat"
    )
    # And it must NOT silently route through the compose (bulk) tier.
    assert 'get_model("sn-compose")' not in rotation_src, (
        "a compose-tier model refines critiqued names at about 5%"
    )


def test_refine_docs_worker_imports_refine_section() -> None:
    """``process_refine_docs_batch`` resolves its non-escalation model
    via ``get_model("sn-refine")`` (same contract as refine_name)."""
    import inspect

    from imas_codex.standard_names import workers

    src = inspect.getsource(_owning_worker_refine_docs.process_refine_docs_batch)
    assert 'get_model("sn-refine")' in src
    assert 'get_model("sn-compose")' not in src
