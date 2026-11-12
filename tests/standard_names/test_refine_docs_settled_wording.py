"""The documentation writer's settled shared wording, on the worker's own path.

These tests drive the batch processors the pipeline runs rather than the
checks they call, so a repair that is skipped on the pipeline path fails here.
Each one fails when its own repair is removed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ARGON_CONTRADICTION = (
    "The argon density is summed over all charge states, including neutral "
    "atoms and ions, from neutral argon at Z=0.\n\n"
    r"$$n_{Ar} = \sum_{Z=0}^{18} n_{Ar^{Z+}}$$"
)

ARGON_POPULATION = (
    "The density counts ions summed over all charge states; neutral atoms are excluded."
)

#: The refine worker's own attempt bound: a name is claimed for refinement only
#: while its docs chain is below the rotation cap.
ATTEMPT_BOUND = 3


def _refine_item(sn_id: str = "argon_density") -> dict:
    from tests.standard_names.test_refine_docs_chain import _make_refine_docs_item

    return _make_refine_docs_item(sn_id=sn_id, score=0.4)


def _generate_item(sn_id: str = "argon_density") -> dict:
    from tests.standard_names.test_generate_docs import _make_docs_item

    return _make_docs_item(sn_id=sn_id)


def _refine_patches(answer, *, terminal_stage: dict, calls: list, exhausted: MagicMock):
    """Patch the refine worker's collaborators and the graph it settles into."""

    async def _answer(model, messages, response_model, service, **kwargs):
        calls.append(model)
        return (answer, 0.01, 100)

    def _passthrough(fn, *args, **kwargs):
        async def _run():
            return fn(*args, **kwargs)

        return _run()

    return (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_answer,
        ),
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
        patch("imas_codex.settings.get_model", return_value="model-x"),
        patch(
            "imas_codex.standard_names.graph_store.graph_claim_release."
            "_mark_docs_exhausted",
            exhausted,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence."
            "persist_refined_docs",
            MagicMock(return_value={"docs_chain_length": 1}),
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence."
            "release_refine_docs_failed_claims",
            MagicMock(return_value=1),
        ),
        patch("asyncio.to_thread", new=_passthrough),
    )


@pytest.mark.asyncio
async def test_contradicting_answer_reaches_terminal_state_within_the_bound():
    """An answer that contradicts the settled population exhausts the name.

    The same item is asked the same question on every pass, so an answer that
    contradicts the stored wording on the first pass contradicts it on the
    next. The name must therefore reach a terminal state rather than being
    released to review and re-claimed, and it must do so inside the attempt
    bound the refine pool already enforces. The model calls spent are counted
    rather than asserted to be zero, because the refusal is discovered only
    after the answer arrives.
    """

    from imas_codex.standard_names.models import RefinedDocs
    from imas_codex.standard_names.worker_tasks.worker_refine_docs import (
        process_refine_docs_batch,
    )

    answer = RefinedDocs(
        description=(
            "Argon density summed over all charge states, including neutral atoms."
        ),
        documentation=ARGON_CONTRADICTION,
        links=[],
    )
    stage = {"docs_stage": "refining"}
    calls: list[str] = []
    exhausted = MagicMock(side_effect=lambda **kw: stage.update(docs_stage="exhausted"))
    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    patches = _refine_patches(
        answer, terminal_stage=stage, calls=calls, exhausted=exhausted
    )
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        for _attempt in range(ATTEMPT_BOUND):
            if stage["docs_stage"] == "exhausted":
                break
            await process_refine_docs_batch([_refine_item()], mgr, asyncio.Event())

    assert stage["docs_stage"] == "exhausted", (
        "the name was released to review instead of reaching a terminal state; "
        f"{len(calls)} model calls were spent"
    )
    assert exhausted.call_count == 1
    assert len(calls) <= ATTEMPT_BOUND
    # The bound is what stops the loop, so a model that contradicts on every
    # pass costs at most one call before the name stops being claimed.
    assert len(calls) == 1, f"cost {len(calls)} model calls to reach terminal"


@pytest.mark.asyncio
async def test_refined_description_carries_the_settled_population():
    """A refinement's one-line description states the population it counts."""

    from imas_codex.standard_names.models import RefinedDocs
    from imas_codex.standard_names.worker_tasks.worker_refine_docs import (
        process_refine_docs_batch,
    )

    answer = RefinedDocs(
        description="Number density of argon ions in the plasma.",
        documentation=(
            "Number density of argon ions in the plasma. It is used to "
            f"diagnose impurity transport.\n\n{ARGON_POPULATION}"
        ),
        links=[],
    )
    persist = MagicMock(return_value={"docs_chain_length": 1})
    exhausted = MagicMock()
    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    patches = _refine_patches(answer, terminal_stage={}, calls=[], exhausted=exhausted)

    def _passthrough(fn, *args, **kwargs):
        async def _run():
            return fn(*args, **kwargs)

        return _run()

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence."
            "persist_refined_docs",
            persist,
        ),
        patches[5],
        patch("asyncio.to_thread", new=_passthrough),
    ):
        await process_refine_docs_batch([_refine_item()], mgr, asyncio.Event())

    assert persist.call_count == 1, "the conforming refinement was not stored"
    stored_description = persist.call_args.kwargs["description"]
    assert "neutral atoms excluded" in stored_description
    assert "charge states" in stored_description


@pytest.mark.asyncio
async def test_refused_first_draft_reaches_the_terminal_state():
    """A refused first draft is exhausted, not left to be re-claimed."""

    from imas_codex.standard_names.models import GeneratedDocs
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        process_generate_docs_batch,
    )

    answer = GeneratedDocs(
        description=(
            "Argon density summed over all charge states, including neutral "
            "atoms and ions."
        ),
        documentation=ARGON_CONTRADICTION,
    )
    stage = {"docs_stage": "pending"}
    calls: list[str] = []
    exhausted = MagicMock(side_effect=lambda **kw: stage.update(docs_stage="exhausted"))

    async def _respond(model, messages, response_model, service, **kwargs):
        calls.append(model)
        return (answer, 0.01, 100)

    def _passthrough(fn, *args, **kwargs):
        async def _run():
            return fn(*args, **kwargs)

        return _run()

    persist = MagicMock(return_value="drafted")
    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_respond,
        ),
        patch("imas_codex.settings.get_model", return_value="model-x"),
        patch(
            "imas_codex.standard_names.graph_store.graph_claim_release."
            "_mark_docs_exhausted",
            exhausted,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence."
            "persist_generated_docs",
            persist,
        ),
        patch("asyncio.to_thread", new=_passthrough),
    ):
        for _attempt in range(ATTEMPT_BOUND):
            if stage["docs_stage"] == "exhausted":
                break
            await process_generate_docs_batch([_generate_item()], mgr, asyncio.Event())

    persist.assert_not_called()
    assert stage["docs_stage"] == "exhausted"
    assert exhausted.call_count == 1
    assert len(calls) == 1
