"""Shared fixtures for standard name tests.

The ``mock_llm`` fixture replaces the real LLM call (``acall_llm_structured``)
with a deterministic, scripted response provider. Tests register expected
responses per (stage, model) and the mock returns them in registration order.

Stages map to worker pool names: ``'generate_name'``, ``'review_name'``,
``'refine_name'``, ``'generate_docs'``, ``'review_docs'``, ``'refine_docs'``.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Structural guard — block live GraphClient connections in default-tier tests
# ---------------------------------------------------------------------------

#: Markers that indicate a test intentionally needs a live Neo4j connection.
_LIVE_GRAPH_MARKERS = frozenset(
    {"graph", "graph_mcp", "requires_graph", "fixture_only"}
)


@pytest.fixture(autouse=True)
def _block_live_graph(request):
    """Raise RuntimeError if a default-tier test attempts to open a real
    Neo4j connection.

    This guard patches ``GraphClient.__post_init__`` — the dataclass
    initialiser where the driver connection is established — directly on the
    class object.  Because the patch targets the class itself rather than a
    module-level name binding, it is effective regardless of how
    ``GraphClient`` was imported (``from ... import GraphClient``, local
    import, etc.).

    Tests marked with any live-graph marker (graph, graph_mcp,
    requires_graph, fixture_only) are exempt and receive an unpatched
    ``GraphClient``.
    """
    if any(request.node.get_closest_marker(m) for m in _LIVE_GRAPH_MARKERS):
        # Test is allowed to connect; yield without patching.
        yield
        return

    from imas_codex.graph.client import GraphClient

    def _blocked_post_init(self, *args, **kwargs):
        raise RuntimeError(
            f"GraphClient.__post_init__ called in a default-tier test "
            f"({request.node.nodeid!r}). "
            "Mark the test with @pytest.mark.graph (or requires_graph) if "
            "it intentionally needs a live Neo4j connection, or add a local "
            "mock/patch so it does not reach GraphClient at all."
        )

    with patch.object(GraphClient, "__post_init__", _blocked_post_init):
        yield


@pytest.fixture(scope="session", autouse=True)
def _cache_grammar_context():
    """Memoize ``get_grammar_context()`` for the whole test session.

    ISN's ``get_grammar_context()`` rebuilds and semantic-validates the full
    published catalog on every call (tens of seconds), and the
    ``GrammarSegments`` validator invokes it on every construction. Without a
    cache the suite is effectively unrunnable. The context is a pure function
    of the on-disk vocabularies, so a single build reused for the session is
    equivalent. Patches both the source symbol and the top-level re-export so
    the many local ``from imas_standard_names import get_grammar_context``
    call sites resolve to the cached value.
    """
    import warnings
    from contextlib import ExitStack

    import imas_standard_names as _isn
    import imas_standard_names.grammar.context as _ctx

    try:
        cached = _ctx.get_grammar_context()
    except Exception as exc:
        # A broken ISN pin (e.g. a grammar/catalog internal inconsistency that
        # raises ParseError while building the context) must not error every
        # test in this package at setup. Skip memoization and leave the real
        # function in place: pure-unit tests that never touch the grammar
        # context still run, and tests that genuinely need it fail (fast) on
        # their own rather than taking the whole package down at collection.
        warnings.warn(
            f"grammar context unavailable ({type(exc).__name__}: {exc}); "
            "skipping session cache — grammar-dependent tests may fail "
            "individually until the ISN pin is fixed",
            stacklevel=2,
        )
        yield
        return

    def _cached() -> Any:
        return cached

    with ExitStack() as stack:
        stack.enter_context(patch.object(_ctx, "get_grammar_context", _cached))
        if hasattr(_isn, "get_grammar_context"):
            stack.enter_context(patch.object(_isn, "get_grammar_context", _cached))
        yield


@pytest.fixture()
def sample_standard_names() -> list[dict]:
    """Sample standard name dicts for write_standard_names testing."""
    return [
        {
            "id": "electron_temperature",
            "source_types": ["dd"],
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "physical_base": "temperature",
            "subject": "electron",
            "description": "Electron temperature profile",
            "documentation": "The electron temperature $T_e$ is measured by Thomson scattering.",
            "kind": "scalar",
            "links": ["ion_temperature", "electron_density"],
            "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "core plasma",
            "constraints": ["T_e > 0"],
            "unit": "eV",
            "physics_domain": "core_profiles",
            "model": "test/model",
            "name_stage": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "plasma_current",
            "source_types": ["signals"],
            "source_id": "tcv:ip/measured",
            "physical_base": "current",
            "description": "Plasma current",
            "unit": "A",
            "physics_domain": "magnetics",
            "kind": "scalar",
            "model": "test/model",
            "name_stage": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
    ]


@pytest.fixture()
def mock_graph_client():
    """A mock GraphClient that records query calls."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client


@pytest.fixture()
def graph_client():
    """Real GraphClient; skips if Neo4j is unavailable.

    Used by graph-marked integration tests (e.g. test_physics_domain_multi).
    """
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


# ---------------------------------------------------------------------------
# MockLLM — scripted LLM response fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ScriptedResponse:
    stage: str
    model: str | None  # None = match any model
    response: Any  # parsed pydantic instance OR dict that will be coerced
    cost: float = 0.0
    tokens: dict[str, int] | None = None


class MockLLM:
    """Scripted LLM mock for SN pipeline tests.

    Usage::

        def test_generate_then_review(mock_llm):
            mock_llm.add_response('generate_name', response=GeneratedName(name=...))
            mock_llm.add_response('review_name',   response=ReviewResult(score=0.9))
            await some_pipeline_step(...)
            assert mock_llm.calls_for('generate_name') == 1

    Fallthrough behaviour: if no scripted response matches a call, the mock
    raises ``RuntimeError`` so tests fail loudly rather than silently using
    the real LLM.
    """

    def __init__(self) -> None:
        self._queue: list[_ScriptedResponse] = []
        self._calls: list[dict[str, Any]] = []

    def add_response(
        self,
        stage: str,
        *,
        response: Any,
        model: str | None = None,
        cost: float = 0.0,
        tokens: dict[str, int] | None = None,
    ) -> None:
        """Register a scripted response. FIFO per stage."""
        self._queue.append(
            _ScriptedResponse(
                stage=stage,
                model=model,
                response=response,
                cost=cost,
                tokens=tokens or {"input": 0, "output": 0, "cached": 0},
            )
        )

    def calls_for(self, stage: str) -> int:
        """Return number of calls dispatched for *stage*."""
        return sum(1 for c in self._calls if c["stage"] == stage)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded call records."""
        return list(self._calls)

    async def _dispatch(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        service: str | None = None,
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, float, dict[str, int]]:
        """Replacement coroutine for ``acall_llm_structured``."""
        # Stage priority: explicit kwarg > inferred from messages > 'unknown'
        resolved_stage = stage or _infer_stage(messages, service) or "unknown"
        self._calls.append(
            {
                "stage": resolved_stage,
                "model": model,
                "service": service,
                "messages": messages,
                "response_model": response_model.__name__,
            }
        )

        for i, scripted in enumerate(self._queue):
            if scripted.stage != resolved_stage:
                continue
            if scripted.model is not None and scripted.model != model:
                continue
            self._queue.pop(i)
            payload = scripted.response
            if isinstance(payload, dict):
                payload = response_model(**payload)
            return payload, scripted.cost, scripted.tokens

        raise RuntimeError(
            f"MockLLM: no scripted response for stage={resolved_stage!r} "
            f"model={model!r}. Registered: "
            f"{[(s.stage, s.model) for s in self._queue]}"
        )


def _infer_stage(
    messages: list[dict[str, str]],
    service: str | None,  # noqa: ARG001
) -> str | None:
    """Best-effort stage inference from system-prompt keywords.

    Workers that need precision should pass ``stage=`` explicitly; this
    function is a fallback for callers that don't.
    """
    if not messages:
        return None
    sys_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    s = sys_content.lower()
    # Check "refine" before "review" with specific phrase matching,
    # because review prompts mention "refinement" and refine prompts
    # mention "reviewer feedback" — single-keyword matching is ambiguous.
    if "you are refining" in s:
        if "documentation" in s or "description" in s or "docs" in s:
            return "refine_docs"
        return "refine_name"
    if "review" in s or "critic" in s:
        # Disambiguate review_name vs review_docs.  The name-axis review
        # prompt says "name-only mode" / "name-axis review"; the docs-axis
        # prompt says "evaluating…documentation" / "documentation text quality".
        # A naïve "documentation" check false-matches on the name prompt
        # because it mentions "documentation is filled in by a later pass".
        if "name-only mode" in s or "name-axis" in s:
            return "review_name"
        if (
            "documentation text quality" in s
            or "evaluating" in s
            and "documentation" in s
        ):
            return "review_docs"
        # Fallback: presence of "documentation quality" signals docs axis
        if "documentation quality" in s:
            return "review_docs"
        return "review_name"
    if "documentation" in s or "enrich" in s:
        return "generate_docs"
    if "compose" in s or "standard name" in s:
        return "generate_name"
    return None


@pytest.fixture()
def mock_llm():
    """Patch ``acall_llm_structured`` globally and yield a :class:`MockLLM`.

    All SN workers import ``acall_llm_structured`` lazily (function-local
    ``from imas_codex.discovery.base.llm import acall_llm_structured``), so
    patching the canonical module attribute is sufficient to intercept every
    call.
    """
    mock = MockLLM()

    # Patch at every known import site.  Function-local imports all resolve
    # to the same object in imas_codex.discovery.base.llm, so one target
    # covers the standard_names workers.  Additional sites (discovery, graph)
    # are included so the fixture is safe to use in cross-module tests.
    targets = [
        # Canonical definition — covers all function-local SN imports:
        #   workers.py, enrich_workers.py, benchmark.py, review/pipeline.py
        "imas_codex.discovery.base.llm.acall_llm_structured",
        # Re-export in discovery.base.__init__ (used by some discovery tests)
        "imas_codex.discovery.base.acall_llm_structured",
    ]

    patches = []
    for target in targets:
        try:
            p = patch(target, side_effect=mock._dispatch)
            patches.append(p)
            p.start()
        except (ImportError, AttributeError):
            # Import site may not exist yet (e.g. Phase 2 code not landed).
            pass

    try:
        yield mock
    finally:
        for p in patches:
            p.stop()
