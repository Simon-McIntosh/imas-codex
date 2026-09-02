"""Source drift requeues accepted documentation within the active run."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names import loop, source_refresh


class _RefreshGraph:
    """Retain the name fields written by the refresh and its edit helper."""

    def __init__(self) -> None:
        self.name = {
            "id": "plasma_current",
            "name_stage": "accepted",
            "docs_stage": "accepted",
        }

    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        if "SET sn.run_id = $scope_run_id" in cypher:
            self.name["run_id"] = params["scope_run_id"]
        return []


@pytest.mark.asyncio
async def test_drift_refresh_is_claimable_by_docs_pool_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RefreshGraph()
    active_run = "active-run"
    reason = source_refresh._format_reason(
        graph.name["id"],
        [{"field": "documentation", "old": "Old source.", "new": "New source."}],
    )
    drift = {
        "sn_id": graph.name["id"],
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "new_path": "magnetics/ip/0",
        "resolved_dd_context": {},
        "renamed": False,
        "requires_steering": True,
        "deltas": [
            {"field": "documentation", "old": "Old source.", "new": "New source."}
        ],
    }

    monkeypatch.setattr(source_refresh, "stamp_source_snapshots", lambda *a, **k: 1)
    monkeypatch.setattr(source_refresh, "detect_source_drift", lambda **k: [drift])

    def apply_edit(**kwargs: object) -> MagicMock:
        graph.name.update(
            docs_stage="pending",
            edit_reason=kwargs["reason"],
            edit_requested_at="2026-09-02T15:00:00+00:00",
            run_id="private-edit-run",
        )
        return MagicMock(blocked=None)

    monkeypatch.setattr("imas_codex.standard_names.edit.apply_edit", apply_edit)

    refreshed = source_refresh.refresh_drifted_sources(
        scope_run_id=active_run,
        gc=graph,
    )

    assert refreshed["steered"] == 1
    assert graph.name["docs_stage"] == "pending"
    assert graph.name["edit_reason"] == reason
    assert graph.name["edit_requested_at"]
    assert graph.name["run_id"] == active_run

    claim_calls: list[str | None] = []

    def claim_generate_docs_batch(**kwargs: object) -> list[dict[str, object]]:
        scope = kwargs.get("scope_run_id")
        claim_calls.append(scope if isinstance(scope, str) else None)
        if (
            scope == graph.name["run_id"]
            and graph.name["name_stage"] == "accepted"
            and graph.name["docs_stage"] == "pending"
        ):
            return [dict(graph.name, claim_token="claim-token")]
        return []

    with (
        patch(
            "imas_codex.standard_names.graph_ops.claim_generate_docs_batch",
            side_effect=claim_generate_docs_batch,
        ),
        patch("imas_codex.settings.get_pool_replicas", return_value=1),
        patch(
            "imas_codex.standard_names.workers.process_generate_docs_batch",
            new_callable=AsyncMock,
        ),
    ):
        specs = loop._build_pool_specs(
            MagicMock(),
            asyncio.Event(),
            drift_scope_run_id=active_run,
            docs_only=True,
        )
        generate_docs = next(spec for spec in specs if spec.name == "generate_docs")
        claimed = await generate_docs.claim()

    assert claim_calls == [active_run]
    assert claimed is not None
    assert claimed["items"][0]["id"] == graph.name["id"]
