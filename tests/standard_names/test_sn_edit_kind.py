"""Governed structural-kind repairs through ``sn edit``."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.edit import _apply_rename, reclassify_kind


class _KindGraph:
    def __init__(self, *, origin: str = "pipeline", stage: str = "accepted") -> None:
        self.origin = origin
        self.stage = stage
        self.applied: list[dict[str, object]] = []

    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        if "KIND_RECLASSIFY_FETCH" in cypher:
            return [
                {
                    "kind": "metadata",
                    "unit": "s",
                    "origin": self.origin,
                    "name_stage": self.stage,
                    "docs_stage": "accepted",
                    "validation_status": "valid",
                }
            ]
        if "KIND_RECLASSIFY_APPLY" in cypher:
            self.applied.append(params)
            return [
                {
                    "kind": params["to_kind"],
                    "unit": "s",
                    "name_stage": "accepted",
                    "docs_stage": "accepted",
                    "validation_status": "valid",
                    "change_id": params["change_id"],
                }
            ]
        raise AssertionError(cypher)


def test_kind_reclassification_defaults_to_an_exact_zero_write_preview() -> None:
    graph = _KindGraph()

    with patch(
        "imas_codex.standard_names.edit._validate_kind_candidate",
        return_value=[],
    ):
        result = reclassify_kind(
            "breakdown_initial_time",
            "scalar",
            reason="The DD value is a quantitative time in seconds.",
            gc=graph,
        )

    assert result == {
        "ok": True,
        "name": "breakdown_initial_time",
        "from_kind": "metadata",
        "to_kind": "scalar",
        "unit": "s",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "validation_status": "valid",
        "dry_run": True,
        "noop": False,
        "hard_findings": [],
    }
    assert graph.applied == []


def test_kind_reclassification_refuses_a_kind_that_disagrees_with_derivation() -> None:
    graph = _KindGraph()

    result = reclassify_kind(
        "breakdown_initial_time",
        "vector",
        reason="Wrong requested target for the test.",
        gc=graph,
    )

    assert result["ok"] is False
    assert "derive_kind" in str(result["reason"])
    assert graph.applied == []


def test_kind_reclassification_respects_catalog_protection() -> None:
    graph = _KindGraph(origin="catalog_edit")

    result = reclassify_kind(
        "breakdown_initial_time",
        "scalar",
        reason="The DD value is a quantitative time in seconds.",
        apply=True,
        gc=graph,
    )

    assert result["ok"] is False
    assert "--override-edits" in str(result["reason"])
    assert graph.applied == []


def test_kind_reclassification_applies_one_cas_event_without_stage_changes() -> None:
    graph = _KindGraph()

    with patch(
        "imas_codex.standard_names.edit._validate_kind_candidate",
        return_value=[],
    ):
        result = reclassify_kind(
            "breakdown_initial_time",
            "scalar",
            reason="The DD value is a quantitative time in seconds.",
            apply=True,
            gc=graph,
        )

    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["from_kind"] == "metadata"
    assert result["to_kind"] == "scalar"
    assert result["unit"] == "s"
    assert result["name_stage"] == "accepted"
    assert result["docs_stage"] == "accepted"
    assert result["validation_status"] == "valid"
    assert str(result["change_id"]).startswith("sn-change:")
    assert len(graph.applied) == 1
    assert graph.applied[0]["id"] == "breakdown_initial_time"
    assert graph.applied[0]["from_kind"] == "metadata"
    assert graph.applied[0]["to_kind"] == "scalar"


def test_rename_derives_successor_kind_instead_of_copying_predecessor() -> None:
    class _RenameGraph:
        def query(self, cypher: str, **params: object) -> list[dict[str, int]]:
            if "EDIT_CHECK_COLLISION" in cypher:
                return [{"n": 0}]
            raise AssertionError(cypher)

    target_row = {
        "name_stage": "accepted",
        "has_successor": False,
        "description": "Time of initial plasma breakdown.",
        "kind": "metadata",
        "unit": "s",
        "physics_domain": "equilibrium",
        "tags": [],
        "chain_length": 0,
    }
    persisted: dict[str, object] = {}

    def _persist(**kwargs: object) -> dict[str, str]:
        persisted.update(kwargs)
        return {"new_name": str(kwargs["new_name"])}

    with (
        patch(
            "imas_codex.standard_names.edit._isn_round_trip_ok",
            return_value=(True, ""),
        ),
        patch("imas_codex.standard_names.edit._base_token", return_value="time"),
        patch("imas_codex.standard_names.edit._derive_rename_unit", return_value="s"),
        patch("imas_codex.standard_names.edit.persist_refined_name", _persist),
        patch("imas_codex.standard_names.edit._grammar_segment_props", return_value={}),
        patch("imas_codex.standard_names.edit._stamp_successor_validation"),
    ):
        plan = _apply_rename(
            _RenameGraph(),
            target="initial_plasma_breakdown_time",
            target_row=target_row,
            new_name="breakdown_initial_time",
            reason="Canonical quantitative identity.",
            origin="human",
            scope="only_self",
            is_parent=False,
            override_edits=False,
            include_accepted=False,
            dry_run=False,
        )

    assert plan.applied is True
    assert persisted["kind"] == "scalar"


def test_cli_kind_mode_is_preview_first_and_requires_apply_to_mutate() -> None:
    preview = {
        "ok": True,
        "name": "breakdown_initial_time",
        "from_kind": "metadata",
        "to_kind": "scalar",
        "unit": "s",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "validation_status": "valid",
        "dry_run": True,
        "noop": False,
        "hard_findings": [],
    }
    runner = CliRunner()

    with patch(
        "imas_codex.standard_names.edit.reclassify_kind", return_value=preview
    ) as operation:
        result = runner.invoke(
            sn,
            [
                "edit",
                "breakdown_initial_time",
                "--kind",
                "scalar",
                "--reason",
                "The DD value is a quantitative time in seconds.",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "would reclassify" in result.output
    assert "metadata" in result.output and "scalar" in result.output
    assert "unit=s" in result.output
    assert operation.call_args.kwargs["apply"] is False

    applied = {**preview, "dry_run": False, "change_id": "sn-change:one"}
    with patch(
        "imas_codex.standard_names.edit.reclassify_kind", return_value=applied
    ) as operation:
        result = runner.invoke(
            sn,
            [
                "edit",
                "breakdown_initial_time",
                "--kind",
                "scalar",
                "--reason",
                "The DD value is a quantitative time in seconds.",
                "--apply",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "reclassified" in result.output
    assert "sn-change:one" in result.output
    assert operation.call_args.kwargs["apply"] is True
