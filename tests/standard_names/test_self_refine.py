"""Free local self-refine pass over freshly-composed standard names.

The self-refine step runs AFTER compose + grammar normalization and BEFORE
persist (so before the paid review quorum). It is an **improve-or-no-op**
pass on the LOCAL compose model:

- when OFF (``[sn-compose].self-refine = false``, the default) the compose
  path is byte-identical to the path without the feature — no extra LLM
  call fires;
- when ON it adopts a better suggested name, but only if that name survives
  the grammar round-trip; an ungrammatical "improvement" is discarded and the
  original is kept.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _self_refine_candidate — direct unit tests
# ---------------------------------------------------------------------------


class _Resp:
    """Mimics the SelfRefineResponse pydantic model."""

    def __init__(self, changed: bool, name: str, description: str = "") -> None:
        self.changed = changed
        self.name = name
        self.description = description
        self.segments = None


class _LLMOut:
    """Mimics the LLMResult triple from acall_llm_structured."""

    def __init__(self, result: Any, cost: float = 0.0) -> None:
        self._result = result
        self._cost = cost

    def __iter__(self):
        return iter((self._result, self._cost, 0))


def _grammar_module(good_names: set[str]) -> MagicMock:
    """Build a fake imas_standard_names.grammar where only good_names parse.

    Round-trip is identity for the good names (canonical-order preserved).
    """
    mod = MagicMock()

    def _parse(name: str):
        if name in good_names:
            # NB: MagicMock(name=...) is reserved for the mock repr, so set
            # the attribute explicitly to carry the parsed name through.
            parsed = MagicMock()
            parsed.sn_name = name
            return parsed
        raise ValueError(f"Grammar parse failed for {name}")

    def _compose(parsed):
        # Identity round-trip — echo the name we stashed on the mock.
        return parsed.sn_name

    mod.parse_standard_name = _parse
    mod.compose_standard_name = _compose
    return mod


@pytest.mark.asyncio
async def test_self_refine_adopts_better_name() -> None:
    """An improved, grammatical suggestion is adopted (name + description)."""
    from imas_codex.standard_names.workers import _self_refine_candidate

    orig_name = "co_passing_density"
    better_name = "co_passing_fast_ion_density"
    grammar = _grammar_module({orig_name, better_name})

    async def mock_acall(*_a, **_k):
        return _LLMOut(
            _Resp(
                changed=True,
                name=better_name,
                description="Number density of co-passing fast ions.",
            )
        )

    with (
        patch.dict(sys.modules, {"imas_standard_names.grammar": grammar}),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: f"[{name}]",
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
    ):
        name, desc = await _self_refine_candidate(
            orig_name,
            "Density of co-passing.",
            {"base_token": "density", "qualifiers": ["co_passing"]},
            {"path": "x/y/density", "unit": "m^-3"},
            "local-model",
            mock_acall,
        )

    assert name == better_name
    assert desc == "Number density of co-passing fast ions."


@pytest.mark.asyncio
async def test_self_refine_discards_ungrammatical_suggestion() -> None:
    """An ungrammatical 'improvement' is discarded; the original is kept."""
    from imas_codex.standard_names.workers import _self_refine_candidate

    orig_name = "electron_temperature"
    bad_name = "garbled_not_a_real_token_xyzzy"
    # Only the original parses; the suggestion does not.
    grammar = _grammar_module({orig_name})

    async def mock_acall(*_a, **_k):
        return _LLMOut(
            _Resp(changed=True, name=bad_name, description="some description")
        )

    with (
        patch.dict(sys.modules, {"imas_standard_names.grammar": grammar}),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: f"[{name}]",
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
    ):
        name, desc = await _self_refine_candidate(
            orig_name,
            "Electron temperature.",
            {"base_token": "temperature", "qualifiers": ["electron"]},
            {"path": "core_profiles/electrons/temperature", "unit": "eV"},
            "local-model",
            mock_acall,
        )

    assert name == orig_name, "ungrammatical suggestion must be discarded"
    assert desc == "Electron temperature."


@pytest.mark.asyncio
async def test_self_refine_no_op_when_unchanged() -> None:
    """changed=False echoes the original name and description unchanged."""
    from imas_codex.standard_names.workers import _self_refine_candidate

    orig_name = "ion_temperature"
    grammar = _grammar_module({orig_name})

    async def mock_acall(*_a, **_k):
        return _LLMOut(_Resp(changed=False, name=orig_name, description="ignored"))

    with (
        patch.dict(sys.modules, {"imas_standard_names.grammar": grammar}),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: f"[{name}]",
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
    ):
        name, desc = await _self_refine_candidate(
            orig_name,
            "Ion temperature.",
            None,
            None,
            "local-model",
            mock_acall,
        )

    assert name == orig_name
    assert desc == "Ion temperature."


@pytest.mark.asyncio
async def test_self_refine_keeps_original_on_llm_error() -> None:
    """An LLM/parse exception keeps the original candidate (never blanks it)."""
    from imas_codex.standard_names.workers import _self_refine_candidate

    orig_name = "poloidal_magnetic_field"
    grammar = _grammar_module({orig_name})

    async def mock_acall(*_a, **_k):
        raise RuntimeError("local endpoint down")

    with (
        patch.dict(sys.modules, {"imas_standard_names.grammar": grammar}),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: f"[{name}]",
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
    ):
        name, desc = await _self_refine_candidate(
            orig_name,
            "Poloidal magnetic field.",
            None,
            None,
            "local-model",
            mock_acall,
        )

    assert name == orig_name
    assert desc == "Poloidal magnetic field."


# ---------------------------------------------------------------------------
# compose_batch — OFF is a no-op (no self-refine LLM call), ON invokes it
# ---------------------------------------------------------------------------


def _make_batch_item(
    path: str = "equilibrium/time_slice/profiles_1d/psi", **kw: Any
) -> dict[str, Any]:
    base = {
        "path": path,
        "description": "Poloidal flux",
        "physics_domain": "equilibrium",
        "unit": "Wb",
        "data_type": "FLT_1D",
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }
    base.update(kw)
    return base


class _FakeCandidate:
    def __init__(self, name: str, source_id: str, **kw: Any) -> None:
        self.source_id = source_id
        self.description = kw.get("description", "desc")
        self.kind = kw.get("kind", "scalar")
        self.dd_paths = kw.get("dd_paths", [source_id])
        self.reason = kw.get("reason", "")
        self.base_token = kw.get("base_token", name)
        self.base_kind = kw.get("base_kind", "quantity")
        self.qualifiers = kw.get("qualifiers", [])
        self._name = name

    def compose_name(self) -> str:
        return self._name


class _FakeBatchResult:
    def __init__(self, candidates: list[_FakeCandidate]) -> None:
        self.candidates = candidates
        self.vocab_gaps: list[Any] = []
        self.attachments: list[Any] = []
        self.skipped: list[Any] = []


class _FakeLLMOut:
    def __init__(self, result: Any, cost: float = 0.01, tokens: int = 100) -> None:
        self._result = result
        self._cost = cost
        self._tokens = tokens
        self.input_tokens = tokens
        self.output_tokens = tokens // 2
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0

    def __iter__(self):
        return iter((self._result, self._cost, self._tokens))


class _FakeBudgetManager:
    def __init__(self) -> None:
        self.reserved = 0.0
        self.run_id = "test-run-id"

    def reserve(self, amount: float, phase: str = "") -> Any:
        self.reserved += amount
        return _FakeLease()


class _FakeLease:
    def charge_event(self, cost: float, event: Any) -> Any:
        return MagicMock(overspend=0)

    def release_unused(self) -> None:
        pass


_GOOD_NAME = "electron_temperature"


@pytest.fixture
def _patch_compose_deps():
    """Patch all local imports inside compose_batch so it runs in isolation."""
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query = MagicMock(return_value=[])

    patches = [
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch("imas_codex.settings.get_model", return_value="local-model"),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: "system" if "system" in name else "user",
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_batch_items",
            side_effect=lambda items: None,
        ),
        patch(
            "imas_codex.standard_names.workers._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_ids_context",
            return_value=None,
        ),
        patch(
            "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
            return_value="",
        ),
        patch(
            "imas_codex.standard_names.review.themes.extract_reviewer_themes",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc),
        patch(
            "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
            return_value=1,
        ),
        patch(
            "imas_codex.standard_names.audits.run_audits",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._auto_detect_physical_base_gaps",
            return_value=[],
        ),
    ]
    for p in patches:
        p.start()
    yield
    patch.stopall()


@pytest.mark.asyncio
async def test_compose_batch_self_refine_off_is_noop(_patch_compose_deps) -> None:
    """With self-refine OFF (default), no self-refine call is made."""
    from imas_codex.standard_names import workers
    from imas_codex.standard_names.workers import compose_batch

    result = _FakeBatchResult([_FakeCandidate(_GOOD_NAME, _make_batch_item()["path"])])

    async def mock_llm(*_a, **_k):
        return _FakeLLMOut(result)

    refine_calls = 0

    async def spy_refine(*a, **k):
        nonlocal refine_calls
        refine_calls += 1
        return a[0], a[1]

    with (
        patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
        patch("imas_codex.standard_names.workers._retry_attempts", return_value=1),
        patch("imas_codex.settings.get_compose_self_refine", return_value=False),
        patch.object(workers, "_self_refine_candidate", spy_refine),
    ):
        await compose_batch([_make_batch_item()], _FakeBudgetManager(), asyncio.Event())

    assert refine_calls == 0, "self-refine must not run when disabled"


@pytest.mark.asyncio
async def test_compose_batch_self_refine_on_invokes_pass(_patch_compose_deps) -> None:
    """With self-refine ON, the self-refine pass runs and its result is used."""
    from imas_codex.standard_names import workers
    from imas_codex.standard_names.workers import compose_batch

    result = _FakeBatchResult([_FakeCandidate(_GOOD_NAME, _make_batch_item()["path"])])

    async def mock_llm(*_a, **_k):
        return _FakeLLMOut(result)

    refine_calls = 0
    improved = "core_electron_temperature"

    async def spy_refine(name, description, *a, **k):
        nonlocal refine_calls
        refine_calls += 1
        return improved, description

    persisted: list[list[dict]] = []

    def _capture_persist(cands, **_k):
        persisted.append(cands)
        return len(cands)

    with (
        patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
        patch("imas_codex.standard_names.workers._retry_attempts", return_value=1),
        patch("imas_codex.settings.get_compose_self_refine", return_value=True),
        patch.object(workers, "_self_refine_candidate", spy_refine),
        patch(
            "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
            _capture_persist,
        ),
    ):
        await compose_batch([_make_batch_item()], _FakeBudgetManager(), asyncio.Event())

    assert refine_calls == 1, "self-refine must run once per candidate when enabled"
    assert persisted, "candidates should be persisted"
    assert persisted[0][0]["id"] == improved, "refined name should be persisted"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
