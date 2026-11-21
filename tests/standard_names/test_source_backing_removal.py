"""Sanctioned removal of one source-to-DD-path backing relationship."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

_BETA_NORMAL_SOURCE = "dd:equilibrium/time_slice/global_quantities/beta_normal"
_BETA_NORMAL_PATH = "equilibrium/time_slice/global_quantities/beta_normal"
_BETA_TOR_NORM_SOURCE = "dd:equilibrium/time_slice/global_quantities/beta_tor_norm"
_BETA_TOR_NORM_PATH = "equilibrium/time_slice/global_quantities/beta_tor_norm"
_PRODUCED_NAME = "normalized_toroidal_plasma_beta"
_REASON = "the source already retains its own beta_normal path"


class _SourceBackingGraph:
    """In-memory graph answering only the two source-backing statements."""

    def __init__(self) -> None:
        self.sources: dict[str, dict[str, Any]] = {
            _BETA_NORMAL_SOURCE: {
                "claim_token": None,
                "claimed_at": None,
                "produced_sn_id": _PRODUCED_NAME,
            },
            _BETA_TOR_NORM_SOURCE: {
                "claim_token": None,
                "claimed_at": None,
                "produced_sn_id": _PRODUCED_NAME,
            },
        }
        self.dd_paths = {_BETA_NORMAL_PATH, _BETA_TOR_NORM_PATH}
        self.backings = {
            (_BETA_NORMAL_SOURCE, _BETA_NORMAL_PATH),
            (_BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH),
            (_BETA_TOR_NORM_SOURCE, _BETA_TOR_NORM_PATH),
        }
        self.changes: list[dict[str, Any]] = []
        self.writes: list[str] = []

    def relationship_count(self) -> int:
        return len(self.backings)

    def source_backings(self, source_id: str) -> set[str]:
        return {path for source, path in self.backings if source == source_id}

    def shape(self) -> dict[str, Any]:
        return {
            "sources": deepcopy(self.sources),
            "dd_paths": sorted(self.dd_paths),
            "backings": sorted(self.backings),
            "changes": deepcopy(self.changes),
        }

    def query(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        from iter_codex.standard_names import edit

        if statement == edit._SOURCE_BACKING_REMOVAL_PREFLIGHT_QUERY:
            return [self._preflight(**params)]
        if statement == edit._SOURCE_BACKING_REMOVAL_QUERY:
            return self._remove(**params)
        raise AssertionError(f"unrecognized query: {statement[:100]!r}")

    def close(self) -> None:  # pragma: no cover - caller owns the borrowed graph
        raise AssertionError("the function must not close a borrowed graph")

    def _preflight(self, *, source_id: str, dd_path: str) -> dict[str, Any]:
        source = self.sources.get(source_id, {})
        return {
            "source_exists": source_id in self.sources,
            "dd_path_exists": dd_path in self.dd_paths,
            "claim_token": source.get("claim_token"),
            "claimed_at": source.get("claimed_at"),
            "directed_edges": int((source_id, dd_path) in self.backings),
            "backing_count": len(self.source_backings(source_id)),
        }

    def _remove(
        self,
        *,
        source_id: str,
        dd_path: str,
        change_id: str,
        reason: str,
        changed_at: str,
    ) -> list[dict[str, Any]]:
        row = self._preflight(source_id=source_id, dd_path=dd_path)
        if (
            row["directed_edges"] != 1
            or row["backing_count"] <= 1
            or row["claim_token"] is not None
            or row["claimed_at"] is not None
        ):
            return []
        self.writes.append("remove_source_backing")
        self.backings.remove((source_id, dd_path))
        self.changes.append(
            {
                "id": change_id,
                "from_name": source_id,
                "to_name": dd_path,
                "operation": "remove_source_dd_path_backing",
                "reason": reason,
                "origin": "source_backing_adjudication",
                "changed_at": changed_at,
                "owner": self.sources[source_id]["produced_sn_id"],
            }
        )
        return [
            {
                "change_id": change_id,
                "remaining_backings": row["backing_count"] - 1,
            }
        ]


def test_removes_spurious_beta_backing_and_records_reason() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is True
    assert graph.relationship_count() == before_count - 1
    assert (_BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH) not in graph.backings
    assert (_BETA_NORMAL_SOURCE, _BETA_NORMAL_PATH) in graph.backings
    assert (_BETA_TOR_NORM_SOURCE, _BETA_TOR_NORM_PATH) in graph.backings
    assert len(graph.changes) == 1
    change = graph.changes[0]
    assert change["operation"] == "remove_source_dd_path_backing"
    assert change["from_name"] == _BETA_NORMAL_SOURCE
    assert change["to_name"] == _BETA_TOR_NORM_PATH
    assert change["reason"] == _REASON
    assert change["owner"] == _PRODUCED_NAME


def test_dry_run_reports_same_directed_intent_and_writes_nothing() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before = graph.shape()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_NORMAL_SOURCE,
        _BETA_TOR_NORM_PATH,
        reason=_REASON,
        gc=graph,
        dry_run=True,
    )

    assert result["ok"] is True and result["dry_run"] is True
    assert result["direction"] == (
        f"{_BETA_NORMAL_SOURCE!r} -[:FROM_DD_PATH]-> {_BETA_TOR_NORM_PATH!r}"
    )
    assert graph.relationship_count() == before_count
    assert graph.shape() == before
    assert graph.writes == [] and graph.changes == []


def test_refuses_missing_source_without_changing_relationship_count() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        "dd:missing", _BETA_TOR_NORM_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "source not found" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_refuses_missing_dd_path_without_changing_relationship_count() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_NORMAL_SOURCE, "equilibrium/missing", reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "DD path not found" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_refuses_absent_relationship_without_changing_relationship_count() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_TOR_NORM_SOURCE, _BETA_NORMAL_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "no FROM_DD_PATH relationship" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_refuses_to_leave_source_unbacked_and_keeps_relationship() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_TOR_NORM_SOURCE, _BETA_TOR_NORM_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "zero DD-path backings" in result["reason"]
    assert graph.relationship_count() == before_count
    assert (_BETA_TOR_NORM_SOURCE, _BETA_TOR_NORM_PATH) in graph.backings
    assert graph.writes == [] and graph.changes == []


@pytest.mark.parametrize("claim_field", ["claim_token", "claimed_at"])
def test_refuses_actively_claimed_source_without_changing_relationship_count(
    claim_field: str,
) -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    graph.sources[_BETA_NORMAL_SOURCE][claim_field] = "active-claim"
    before_count = graph.relationship_count()

    result = remove_source_dd_path_backing(
        _BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "actively claimed" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_success_leaves_exactly_one_backing_required_by_fold_guard() -> None:
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()

    result = remove_source_dd_path_backing(
        _BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH, reason=_REASON, gc=graph
    )

    assert result["ok"] is True
    assert result["remaining_backings"] == 1
    assert graph.source_backings(_BETA_NORMAL_SOURCE) == {_BETA_NORMAL_PATH}


def test_cli_exposes_required_reason_and_dry_run() -> None:
    from iter_codex.cli.sn import sn
    from iter_codex.standard_names.edit import remove_source_dd_path_backing

    graph = _SourceBackingGraph()
    before_count = graph.relationship_count()

    def _remove(source: str, dd_path: str, **kwargs: Any) -> dict[str, Any]:
        return remove_source_dd_path_backing(source, dd_path, gc=graph, **kwargs)

    with patch(
        "iter_codex.standard_names.edit.remove_source_dd_path_backing",
        side_effect=_remove,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "remove-source-backing",
                _BETA_NORMAL_SOURCE,
                _BETA_TOR_NORM_PATH,
                "--reason",
                _REASON,
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "would remove" in result.output
    assert "-[:FROM_DD_PATH]->" in result.output
    assert "remaining=1" in result.output
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []

    missing_reason = CliRunner().invoke(
        sn,
        ["remove-source-backing", _BETA_NORMAL_SOURCE, _BETA_TOR_NORM_PATH],
    )
    assert missing_reason.exit_code != 0
    assert "--reason" in missing_reason.output
