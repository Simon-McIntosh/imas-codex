"""Ownership tests for the SignalSource → IMASNode edge.

One module owns every create and delete of a MAPS_TO_IMAS edge. These tests
pin the consolidation: the count-checked writer raises when the write reports
zero edges, both production writers route through it, ``imas map clear``
deletes through ``graph_ops``, a route string in a candidate judgment is
refused, ``imas map status`` prints candidate routes, and the two census greps
find MAPS_TO_IMAS query sites only in ``imas_codex/ids/graph_ops.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from imas_codex.ids.graph_ops import (
    CandidateWriteError,
    clear_mapping_bindings,
    write_mapping_binding,
)
from imas_codex.ids.models import (
    ValidatedMappingResult,
    ValidatedSignalMapping,
    persist_mapping_result,
)


def _binding() -> ValidatedSignalMapping:
    return ValidatedSignalMapping(
        source_id="jet:magnetics/ip",
        target_id="equilibrium/time_slice/global_quantities/ip",
        confidence=0.9,
        mapping_type="direct",
    )


def _gc_zero_map_write():
    """A graph client whose MAPS_TO_IMAS MERGE reports zero edges written."""
    gc = MagicMock()

    def _query(query, **kwargs):
        if "MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)" in query:
            return [{"written": 0}]
        return []

    gc.query.side_effect = _query
    return gc


# ---------------------------------------------------------------------------
# The count-checked writer
# ---------------------------------------------------------------------------


class TestWriteMappingBinding:
    def test_returns_written_count(self):
        gc = MagicMock()
        gc.query.return_value = [{"written": 1}]

        assert write_mapping_binding(_binding(), gc) == 1
        statement = " ".join(gc.query.call_args.args[0].split())
        assert "MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)" in statement
        assert "RETURN count(r) AS written" in statement

    def test_zero_written_raises(self):
        """A vanished endpoint reports zero written, and the write fails."""
        gc = MagicMock()
        gc.query.return_value = [{"written": 0}]

        with pytest.raises(CandidateWriteError):
            write_mapping_binding(_binding(), gc)


# ---------------------------------------------------------------------------
# Every writer raises on a zero-row write
# ---------------------------------------------------------------------------


class TestWritersRaiseOnZeroRowWrite:
    def test_persist_mapping_result_raises(self):
        result = ValidatedMappingResult(
            facility="jet",
            ids_name="equilibrium",
            dd_version="4.1.0",
            sections=[],
            bindings=[_binding()],
        )

        with pytest.raises(CandidateWriteError):
            persist_mapping_result(result, gc=_gc_zero_map_write(), status="generated")

    def test_run_error_derivation_only_raises(self, monkeypatch):
        import imas_codex.ids.mapping as mapping

        gc = MagicMock()

        def _query(query, **kwargs):
            if "MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)" in query:
                return [{"written": 0}]
            # The direct-binding fetch the derivation reads from.
            return [
                {
                    "source_id": "jet:magnetics/ip",
                    "source_property": "value",
                    "target_id": "equilibrium/time_slice/global_quantities/ip",
                    "transform_expression": "value",
                    "source_units": "A",
                    "target_units": "A",
                    "cocos_label": None,
                    "confidence": 0.95,
                }
            ]

        gc.query.side_effect = _query
        monkeypatch.setattr(
            mapping, "derive_error_mappings", lambda *a, **k: [_binding()]
        )

        with pytest.raises(CandidateWriteError):
            mapping.run_error_derivation_only("jet", "equilibrium", gc=gc)


# ---------------------------------------------------------------------------
# The delete owner
# ---------------------------------------------------------------------------


class TestClearMappingBindings:
    def test_returns_deleted_count(self):
        gc = MagicMock()
        gc.query.return_value = [{"deleted": 4}]

        assert clear_mapping_bindings("jet", "pf_active", gc) == 4
        statement = " ".join(gc.query.call_args.args[0].split())
        assert "DELETE r" in statement
        assert gc.query.call_args.kwargs["ids"] == "pf_active"

    def test_imas_map_clear_deletes_through_graph_ops(self, monkeypatch):
        """The CLI's MAPS_TO_IMAS delete lives in graph_ops, so the CLI issues
        no MAPS_TO_IMAS statement of its own."""
        import imas_codex.cli.map as map_cli
        import imas_codex.graph.client as gclient
        import imas_codex.ids.graph_ops as graph_ops

        called: dict[str, tuple] = {}

        def fake_clear(facility, ids_name, gc):
            called["args"] = (facility, ids_name)
            return 3

        monkeypatch.setattr(graph_ops, "clear_mapping_bindings", fake_clear)
        gc = MagicMock()
        gc.query.return_value = [{"deleted": 1}]
        monkeypatch.setattr(gclient, "GraphClient", lambda *a, **k: gc)

        deleted = map_cli._clear_mapping("jet", "pf_active")

        assert called["args"] == ("jet", "pf_active")
        assert deleted == 1
        for call in gc.query.call_args_list:
            assert "MAPS_TO_IMAS" not in str(call)


# ---------------------------------------------------------------------------
# imas map status prints candidate routes
# ---------------------------------------------------------------------------


def test_map_status_prints_route_counts(monkeypatch):
    import imas_codex.cli.map as map_cli
    import imas_codex.graph.client as gclient
    import imas_codex.ids.graph_ops as graph_ops

    monkeypatch.setattr(
        graph_ops,
        "count_candidates_by_route",
        lambda facility, gc: {"selected": 5, "escalated": 2},
    )
    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "jet:pf_active",
            "ids_name": "pf_active",
            "status": "active",
            "dd_version": "4.1.1",
        }
    ]
    monkeypatch.setattr(gclient, "GraphClient", lambda *a, **k: gc)

    from click.testing import CliRunner

    from imas_codex.cli.map import map_cmd

    runner = CliRunner()
    result = runner.invoke(map_cmd, ["status", "jet"])

    assert result.exit_code == 0
    assert "selected" in result.output
    assert "escalated" in result.output


# ---------------------------------------------------------------------------
# The census: one module owns the edge's query sites
# ---------------------------------------------------------------------------

_WRITE = re.compile(r"MERGE \(sg\)-\[(r|rel):MAPS_TO_IMAS\]")
_DELETE = re.compile(r"\[r:MAPS_TO_IMAS\]->\(:IMASNode\)")


def test_maps_to_imas_query_sites_live_only_in_graph_ops():
    root = Path(__file__).resolve().parents[2] / "imas_codex"
    hits: dict[str, list[str]] = {"write": [], "delete": []}
    for path in root.rglob("*.py"):
        rel = path.relative_to(root.parent).as_posix()
        text = path.read_text(encoding="utf-8")
        if _WRITE.search(text):
            hits["write"].append(rel)
        if _DELETE.search(text):
            hits["delete"].append(rel)

    assert hits["write"], "census found no MAPS_TO_IMAS write site"
    assert hits["delete"], "census found no MAPS_TO_IMAS delete site"
    for kind, paths in hits.items():
        for rel in paths:
            assert rel == "imas_codex/ids/graph_ops.py", (
                f"{kind} site outside graph_ops: {rel}"
            )
