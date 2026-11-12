"""The one docs-exhaust marker, read through the statement it sends.

Both documentation writers end a name at the same terminal docs state, so the
transition has one owner and the two writers differ only in the stage they
fence on and the tag they record. These tests replace the graph client at the
client boundary and read the statement and parameters the marker actually
sends, so a wrong fence, a dropped claim token, a drifted tag or a missing
steering-edit clause fails here rather than silently passing a mocked marker.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

#: (eligible_stage, tag) for each writer that reaches the marker.
WRITERS = [
    ("pending", "docs_generate_refusal"),
    ("refining", "docs_refine_exhaust"),
]


class _FakeGraphClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, statement: str, **params):
        self._calls.append((statement, params))
        return []


class _FakeServices:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def GraphClient(self):  # noqa: N802 - mirrors the services attribute
        return _FakeGraphClient(self.calls)


def _marker(**kwargs):
    from imas_codex.standard_names.graph_store.graph_claim_release import (
        _mark_docs_exhausted,
    )

    return _mark_docs_exhausted


@pytest.mark.parametrize(("eligible_stage", "tag"), WRITERS)
def test_marker_statement_fences_the_writers_stage_and_records_its_tag(
    eligible_stage: str, tag: str
) -> None:
    services = _FakeServices()
    _marker()(
        sn_id="argon_density",
        token="tok-abc",
        error_msg="refused: contradicts the settled wording",
        eligible_stage=eligible_stage,
        tag=tag,
        services=services,
    )

    assert len(services.calls) == 1
    statement, params = services.calls[0]

    # The fence is the stage the calling writer holds, not a literal.
    assert "sn.docs_stage = $eligible_stage" in statement
    assert "sn.docs_stage = 'pending'" not in statement
    assert "sn.docs_stage = 'refining'" not in statement

    # The claim token is part of the fence.
    assert "sn.claim_token = $token" in statement

    # The reason is tagged by the writer that wrote it.
    assert "' [' + $tag + '] '" in statement

    # A docs-axis exhaust closes an open docs-steering edit.
    assert "sn.edit_status = 'open'" in statement
    assert "sn.docs_hint IS NOT NULL" in statement
    assert "THEN 'exhausted'" in statement

    assert params["sn_id"] == "argon_density"
    assert params["token"] == "tok-abc"
    assert params["eligible_stage"] == eligible_stage
    assert params["tag"] == tag
    assert params["error_msg"] == "refused: contradicts the settled wording"


def test_marker_bounds_the_recorded_reason() -> None:
    services = _FakeServices()
    _marker()(
        sn_id="argon_density",
        token="tok-abc",
        error_msg="x" * 900,
        eligible_stage="pending",
        tag="docs_generate_refusal",
        services=services,
    )

    assert len(services.calls[0][1]["error_msg"]) == 300


def _passthrough(fn, *args, **kwargs):
    async def _run():
        return fn(*args, **kwargs)

    return _run()


@pytest.mark.parametrize(
    ("writer", "stage", "tag"),
    [
        ("generate", "pending", "docs_generate_refusal"),
        ("refine", "refining", "docs_refine_exhaust"),
    ],
)
@pytest.mark.asyncio
async def test_writer_paths_reach_the_marker_with_the_graph_client_replaced(
    writer: str, stage: str, tag: str
) -> None:
    """Each writer's refusal path sends the marker query to the real client.

    The marker is not mocked: the writer runs, calls the marker, and the
    statement is captured where the marker opens its graph client. A call site
    that fails to hand the marker its services handle raises here.
    """

    if writer == "generate":
        from imas_codex.standard_names.models import GeneratedDocs
        from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
            process_generate_docs_batch,
        )
        from tests.standard_names.test_generate_docs import _make_docs_item
        from tests.standard_names.test_refine_docs_settled_wording import (
            ARGON_CONTRADICTION,
        )

        answer = GeneratedDocs(
            description=(
                "Argon density summed over all charge states, including "
                "neutral atoms and ions."
            ),
            documentation=ARGON_CONTRADICTION,
        )
        module = "imas_codex.standard_names.graph_store.graph_docs_persistence"
        persist_name = "persist_generated_docs"
        release_name = "release_generate_docs_failed_claims"
        processor = process_generate_docs_batch
        item = _make_docs_item(sn_id="argon_density")
    else:
        from imas_codex.standard_names.models import RefinedDocs
        from imas_codex.standard_names.worker_tasks.worker_refine_docs import (
            process_refine_docs_batch,
        )
        from tests.standard_names.test_refine_docs_chain import _make_refine_docs_item

        answer = RefinedDocs(
            description=(
                "Argon density summed over all charge states, including neutral atoms."
            ),
            documentation=(
                "The argon density is summed over all charge states, including "
                "neutral atoms, from neutral argon at Z=0."
            ),
            links=[],
        )
        module = "imas_codex.standard_names.graph_store.graph_docs_persistence"
        persist_name = "persist_refined_docs"
        release_name = "release_refine_docs_failed_claims"
        processor = process_refine_docs_batch
        item = _make_refine_docs_item(sn_id="argon_density", score=0.4)

    async def _answer(model, messages, response_model, service, **kwargs):
        return (answer, 0.01, 100)

    services = _FakeServices()
    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_answer,
        ),
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
        patch("imas_codex.settings.get_model", return_value="model-x"),
        patch(
            "imas_codex.standard_names.graph_store.graph_foundation.get_services",
            return_value=services,
        ),
        patch(f"{module}.{persist_name}", MagicMock(return_value="drafted")),
        patch(f"{module}.{release_name}", MagicMock(return_value=1)),
        patch("asyncio.to_thread", new=_passthrough),
    ):
        await processor([item], mgr, asyncio.Event())

    mark_calls = [
        (statement, params)
        for statement, params in services.calls
        if "docs_stage = 'exhausted'" in statement
    ]
    assert len(mark_calls) == 1, (
        f"the {writer} writer did not reach the docs-exhaust marker "
        f"({len(services.calls)} queries sent)"
    )
    statement, params = mark_calls[0]
    assert params["eligible_stage"] == stage
    assert params["tag"] == tag
    assert "sn.docs_stage = $eligible_stage" in statement
