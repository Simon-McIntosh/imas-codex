"""Name-scoped grammar-segment repair — one name, any lifecycle stage.

The segment columns are a deterministic projection of the canonical id via
the ISN parser, so a name written by an out-of-grammar path can store
segments that disagree with its own id. The whole-graph sweep repairs that
every rotation; these cover the scoped route an operator reaches for a single
name between rotations, including the dry run and the terminal stages the
sweep's live predicate used to exclude.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_ops import (
    _GRAMMAR_SEGMENT_COLUMNS,
    _parse_grammar,
    realign_grammar_segments_for_name,
)

#: A name whose stored ``physical_base`` has drifted from its own id parse.
DRIFTED_NAME = "electron_density"


class _Graph:
    """Minimal graph double: one StandardName row plus a recorded write."""

    def __init__(
        self,
        *,
        row: dict[str, Any] | None,
        stage: str | None = "accepted",
    ) -> None:
        self.row = row
        self.stage = stage
        self.reads: list[str] = []
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _Graph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN sn.name_stage AS stage" in cypher:
            self.reads.append(cypher)
            if self.row is None:
                return []
            return [{"stage": self.stage, **self.row}]
        self.writes.append((cypher, params))
        return [{"id": params["id"]}]


def _stored(**overrides: Any) -> dict[str, Any]:
    """A segment row that matches the parse except for the overrides given."""
    parsed = _parse_grammar(DRIFTED_NAME)
    row = {column: parsed.get(column) for column in _GRAMMAR_SEGMENT_COLUMNS}
    row.update(overrides)
    return row


def _install(monkeypatch: pytest.MonkeyPatch, graph: _Graph) -> None:
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)


def test_the_route_takes_one_name_and_reports_its_drift_under_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run reads one id, reports stored vs parsed, and writes nothing."""
    graph = _Graph(row=_stored(physical_base="rate"))
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name(DRIFTED_NAME, dry_run=True)

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["noop"] is False
    assert result["drift"] == {"physical_base": {"stored": "rate", "parsed": "density"}}
    assert result["physical_base"] == "density"
    assert graph.writes == []
    assert len(graph.reads) == 1
    # The read is scoped to the one id, not to a stage-filtered population.
    assert "{id: $id}" in graph.reads[0]


def test_a_drifted_physical_base_column_is_realigned_to_the_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying the repair writes every segment column plus a ledger entry."""
    graph = _Graph(row=_stored(physical_base="rate"))
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name(DRIFTED_NAME)

    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["change_id"].startswith("sn-change:")
    assert len(graph.writes) == 1
    cypher, params = graph.writes[0]
    assert "SET sn.physical_base = seg.physical_base" in cypher
    assert "CREATE (change:StandardNameChange" in cypher
    assert "MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)" in cypher
    assert params["segments"]["physical_base"] == "density"
    assert params["operation"] == "realign_grammar_segments"
    assert "'rate' -> 'density'" in params["reason"]


@pytest.mark.parametrize("stage", ["superseded", "exhausted"])
def test_a_terminal_stage_is_repaired_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """The route covers every lifecycle stage, including tombstoned names."""
    graph = _Graph(row=_stored(physical_base="rate"), stage=stage)
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name(DRIFTED_NAME)

    assert result["ok"] is True
    assert result["stage"] == stage
    assert len(graph.writes) == 1
    assert graph.writes[0][1]["segments"]["physical_base"] == "density"


def test_the_whole_graph_sweep_is_gated_to_live_names() -> None:
    """The sweep quarantines tombstones: only a live name is read for repair."""
    import inspect

    source = inspect.getsource(graph_ops.reconcile_grammar_segments)
    assert "LIVE_NAME" in source
    assert "MATCH (sn:StandardName) WHERE" in source


class _StagedGraph:
    """Graph double holding one id whose stage the caller controls.

    Both routes read and write against the same in-memory row, so a test can
    show that the sweep's bulk query never selects a terminal-stage id while
    the name-scoped route reaches it directly.
    """

    def __init__(self, *, stage: str, row: dict[str, Any]) -> None:
        self.stage = stage
        self.row = dict(row)
        self.sweep_writes: list[dict[str, Any]] = []
        self.scoped_writes: list[dict[str, Any]] = []

    def __enter__(self) -> _StagedGraph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "UNWIND $batch AS b" in cypher:
            self.sweep_writes.extend(params["batch"])
            return []
        if "RETURN sn.name_stage AS stage" in cypher:
            return [{"stage": self.stage, **self.row}]
        if "UNWIND [$segments]" in cypher:
            self.scoped_writes.append(params)
            return [{"id": params["id"]}]
        # The whole-graph sweep's bulk read: a real LIVE_NAME predicate
        # excludes a superseded/exhausted row server-side.
        if self.stage in ("superseded", "exhausted"):
            return []
        return [{"id": DRIFTED_NAME, **self.row}]


def test_the_sweep_leaves_a_superseded_identity_untouched_while_the_scoped_route_repairs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quarantine: the sweep skips a tombstone; the name-scoped route reaches it."""
    graph = _StagedGraph(stage="superseded", row=_stored(physical_base="rate"))
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)
    monkeypatch.setattr(
        graph_ops, "reconcile_standard_name_kinds", lambda: {"kinds_realigned": 0}
    )

    sweep_result = graph_ops.reconcile_grammar_segments()

    assert sweep_result == {"names_realigned": 0}
    assert graph.sweep_writes == []

    scoped_result = realign_grammar_segments_for_name(DRIFTED_NAME)

    assert scoped_result["ok"] is True
    assert scoped_result["stage"] == "superseded"
    assert len(graph.scoped_writes) == 1
    assert graph.scoped_writes[0]["segments"]["physical_base"] == "density"


def test_an_aligned_name_is_a_reported_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name already matching its parse triggers no write at all."""
    graph = _Graph(row=_stored())
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name(DRIFTED_NAME)

    assert result == {
        "ok": True,
        "name": DRIFTED_NAME,
        "stage": "accepted",
        "dry_run": False,
        "drift": {},
        "noop": True,
        "physical_base": "density",
    }
    assert graph.writes == []


def test_a_blank_name_is_refused_before_the_graph_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph(row=_stored())
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name("   ")

    assert result == {"ok": False, "reason": "a standard name is required"}
    assert graph.reads == []


def test_an_absent_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _Graph(row=None)
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name("no_such_name")

    assert result["ok"] is False
    assert "not found" in result["reason"]
    assert graph.writes == []


def test_an_unparseable_name_keeps_its_stored_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segments of a name the grammar rejects belong to the quarantine path."""
    row = dict.fromkeys(_GRAMMAR_SEGMENT_COLUMNS)
    row["position"] = "pedestal"
    graph = _Graph(row=row)
    _install(monkeypatch, graph)

    result = realign_grammar_segments_for_name("!!not a grammar name!!")

    assert result["ok"] is False
    assert "cannot parse" in result["reason"]
    assert graph.writes == []


def test_cli_route_passes_the_name_and_dry_run_and_prints_the_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sn realign-segments NAME --dry-run`` is the operator surface."""
    from imas_codex.cli.sn import sn

    calls: list[tuple[str, bool]] = []

    def _route(name: str, *, dry_run: bool = False) -> dict[str, Any]:
        calls.append((name, dry_run))
        return {
            "ok": True,
            "name": name,
            "stage": "superseded",
            "dry_run": dry_run,
            "drift": {"physical_base": {"stored": "rate", "parsed": "density"}},
            "noop": False,
        }

    monkeypatch.setattr(graph_ops, "realign_grammar_segments_for_name", _route)

    result = CliRunner().invoke(sn, ["realign-segments", DRIFTED_NAME, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert calls == [(DRIFTED_NAME, True)]
    assert "would realign" in result.output
    assert "superseded" in result.output
    assert "physical_base: rate" in result.output
    assert "density" in result.output


def test_cli_route_surfaces_a_refusal_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.cli.sn import sn

    monkeypatch.setattr(
        graph_ops,
        "realign_grammar_segments_for_name",
        lambda name, *, dry_run=False: {"ok": False, "reason": "not found"},
    )

    result = CliRunner().invoke(sn, ["realign-segments", "no_such_name"])

    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_route_reports_the_aligned_base_on_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aligned name still shows the value the parse agreed on."""
    from imas_codex.cli.sn import sn

    monkeypatch.setattr(
        graph_ops,
        "realign_grammar_segments_for_name",
        lambda name, *, dry_run=False: {
            "ok": True,
            "name": name,
            "stage": "accepted",
            "dry_run": dry_run,
            "drift": {},
            "noop": True,
            "physical_base": "rate",
        },
    )

    result = CliRunner().invoke(
        sn, ["realign-segments", "neutron_rate_due_to_beam_beam_fusion", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "already match the parse" in result.output
    assert "physical_base=rate" in result.output
