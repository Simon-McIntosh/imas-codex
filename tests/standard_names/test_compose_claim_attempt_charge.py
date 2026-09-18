"""Compose claims spend attempts only on sources they actually acquire."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest


class _StatefulClaimGraph:
    def __init__(self, sources: dict[str, dict[str, Any]]) -> None:
        self.sources = sources

    def __enter__(self) -> _StatefulClaimGraph:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def query(self, _cypher: str, **params: Any) -> list[dict[str, Any]]:
        guarded = "settled_stages" in params
        settled_stages = set(params.get("settled_stages", []))
        rows: list[dict[str, Any]] = []
        for source_id in params["ids"]:
            source = self.sources[source_id]
            holding = [
                dict(binding)
                for binding in source["bindings"]
                if binding["stage"] not in settled_stages
            ]
            withheld = guarded and bool(holding)
            if not withheld:
                source["attempt_count"] += 1
                source["claim_seq"] += 1
                source["claim_token"] = params["token"]
            rows.append(
                {
                    "id": source_id,
                    "source_id": source_id.removeprefix("dd:"),
                    "source_type": "dd",
                    "batch_key": source_id.removeprefix("dd:").split("/", 1)[0],
                    "description": source["description"],
                    "status": source["status"],
                    "claim_token": source.get("claim_token"),
                    "claim_seq": source["claim_seq"],
                    "attempt_count": source["attempt_count"],
                    "claimed": not withheld,
                    "holding": holding,
                }
            )
        return rows


def _source(
    *,
    attempts: int,
    bindings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "attempt_count": attempts,
        "bindings": bindings,
        "claim_seq": 8,
        "claim_token": None,
        "description": "A bounded source used to exercise claim accounting.",
        "status": "extracted",
    }


def test_live_binding_is_withheld_without_charge_while_unbound_source_advances(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from imas_codex.standard_names.graph_ops import (
        claim_explicit_standard_name_sources,
    )

    bound_path = "diagnostic/bound_quantity"
    unbound_path = "diagnostic/unbound_quantity"
    graph = _StatefulClaimGraph(
        {
            f"dd:{bound_path}": _source(
                attempts=4,
                bindings=[{"id": "live_holder", "stage": "reviewed"}],
            ),
            f"dd:{unbound_path}": _source(attempts=2, bindings=[]),
        }
    )
    bound_before = graph.sources[f"dd:{bound_path}"]["attempt_count"]
    unbound_before = graph.sources[f"dd:{unbound_path}"]["attempt_count"]

    with (
        patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=graph,
        ),
        caplog.at_level(logging.WARNING, logger="imas_codex.standard_names.graph_ops"),
    ):
        result = claim_explicit_standard_name_sources([bound_path, unbound_path])

    bound_after = graph.sources[f"dd:{bound_path}"]["attempt_count"]
    unbound_after = graph.sources[f"dd:{unbound_path}"]["attempt_count"]
    assert (bound_before, bound_after) == (4, 4)
    assert (unbound_before, unbound_after) == (2, 3)
    assert [row["source_id"] for row in result] == [unbound_path]
    assert result.withheld == [
        {
            "id": f"dd:{bound_path}",
            "source_id": bound_path,
            "holding_identity": "live_holder",
            "holding_stage": "reviewed",
        }
    ]
    assert f"dd:{bound_path}" in caplog.text
    assert "live_holder" in caplog.text
    assert "reviewed" in caplog.text


@pytest.mark.parametrize("settled_stage", ["superseded", "exhausted"])
def test_source_bound_only_to_settled_name_remains_claimable(
    settled_stage: str,
) -> None:
    from imas_codex.standard_names.graph_ops import (
        claim_explicit_standard_name_sources,
    )

    source_path = f"diagnostic/{settled_stage}_quantity"
    graph = _StatefulClaimGraph(
        {
            f"dd:{source_path}": _source(
                attempts=1,
                bindings=[{"id": f"{settled_stage}_holder", "stage": settled_stage}],
            )
        }
    )
    before = graph.sources[f"dd:{source_path}"]["attempt_count"]

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=graph,
    ):
        result = claim_explicit_standard_name_sources([source_path])

    after = graph.sources[f"dd:{source_path}"]["attempt_count"]
    assert (before, after) == (1, 2)
    assert [row["source_id"] for row in result] == [source_path]
    assert result.withheld == []
