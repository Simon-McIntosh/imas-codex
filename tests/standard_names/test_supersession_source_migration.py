"""A supersession carries only the source bindings something judged.

``supersede_prior_source_names`` retires the predecessor name left on a source
by a regen pass. The predecessor is frequently shared: many DD paths compose
onto one name. Retargeting that whole population because ONE of its sources
recomposed stamps the recomposed source's new identity onto sources no composer
and no reviewer ever weighed against it — which is how a diagnostic
line-of-sight name came to carry a conductor cross-section, and a
measurement-position angle name a 3D field-map grid axis. Both sides of each
pair are metres and radians respectively, so unit and dimensionality agreement
is silent on it; only per-source judgement separates them.

The rule these tests pin: a source may carry its provenance to the successor
only when a composer already bound it there. Everything else that reaches the
predecessor holds it live, and the supersession is refused rather than
performed on unjudged evidence.

Behavioural contract tests — the emitted Cypher and the call arguments are
captured through a mocked graph client, so no live Neo4j is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.standard_names.attachment_audit import (
    AttachmentPairingGuardResult,
    AttachmentVerdict,
)
from imas_codex.standard_names.graph_ops import supersede_prior_source_names

# graph_ops binds GraphClient at module import time, so patch that namespace.
_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"

_TRIGGER = "dd:spectrometer_x_ray_crystal/channel/lines_of_sight_second_point/r"
_TRIGGER_PATH = "spectrometer_x_ray_crystal/channel/lines_of_sight_second_point/r"
_UNJUDGED = [
    "dd:ferritic/object/axisymmetric/thick_line/first_point/r",
    "dd:pf_active/coil/element/geometry/thick_line/second_point/r",
]


def _preflight_row(
    *,
    judged: list[str],
    retained: list[str],
    old_name: str | None = "radial_coordinate",
    new_name: str = "radial_coordinate_of_line_of_sight",
    trigger_source_id: str | None = _TRIGGER,
) -> dict[str, Any]:
    return {
        "requested_source_id": _TRIGGER_PATH,
        "new_name": new_name,
        "requested_source_exists": True,
        "successor_exists": True,
        "trigger_source_id": trigger_source_id,
        "old_name": old_name,
        "old_stage": "accepted" if old_name else None,
        "judged_source_ids": judged,
        "retained_source_ids": retained,
    }


class _Recorder:
    """Captures every statement and its parameters, replying like the graph."""

    def __init__(self, preflight_rows: list[dict[str, Any]]) -> None:
        self.preflight_rows = preflight_rows
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.committed = False
        self.rolled_back = False

    # -- transaction surface ------------------------------------------------
    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        if "GENERATED_SUPERSESSION_PREFLIGHT" in cypher:
            return self.preflight_rows
        if "GENERATED_SUPERSESSION_FINALIZE" in cypher:
            return [
                {"old_name": plan["old_name"], "new_name": plan["new_name"]}
                for plan in params["plans"]
            ]
        return []

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    # -- client surface -----------------------------------------------------
    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def session(self) -> _Recorder:
        return self

    def begin_transaction(self) -> _Recorder:
        return self

    # -- assertions helpers -------------------------------------------------
    def statements(self, needle: str) -> list[tuple[str, dict[str, Any]]]:
        return [call for call in self.calls if needle in call[0]]


def _drive(
    preflight_rows: list[dict[str, Any]],
    *,
    admitted: tuple[str, ...] | None = None,
    rejected: tuple[AttachmentVerdict, ...] = (),
    moved: int | None = None,
) -> tuple[_Recorder, Any, Any]:
    """Run one supersession against the recorder, returning the spies."""
    recorder = _Recorder(preflight_rows)
    guard_result = AttachmentPairingGuardResult(admitted or (), rejected)
    with (
        patch(_GC_PATH, return_value=recorder),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=guard_result,
        ) as guard,
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            side_effect=lambda *a, **kw: (
                moved if moved is not None else len(kw["source_ids"])
            ),
        ) as retarget,
    ):
        superseded = supersede_prior_source_names(
            [
                {
                    "new_name": "radial_coordinate_of_line_of_sight",
                    "source_id": _TRIGGER_PATH,
                }
            ]
        )
    return recorder, guard, retarget, superseded  # type: ignore[return-value]


class TestUnjudgedSourcesStayBehind:
    def test_only_the_judged_set_is_retargeted(self) -> None:
        """The unjudged population is never handed to the retarget."""
        recorder, guard, retarget, superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=list(_UNJUDGED))],
            admitted=(_TRIGGER,),
        )

        assert retarget.call_args.kwargs["source_ids"] == [_TRIGGER]
        # The attachment guard sees the same set, so no unjudged pairing is
        # even offered to it.
        assert guard.call_args.args[2] == [_TRIGGER]
        assert superseded == 0
        assert recorder.committed is True

    def test_predecessor_is_not_superseded_while_sources_remain(self) -> None:
        recorder, _guard, _retarget, superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=list(_UNJUDGED))],
            admitted=(_TRIGGER,),
        )

        assert superseded == 0
        assert recorder.statements("GENERATED_SUPERSESSION_FINALIZE") == []
        assert recorder.statements("old.name_stage = 'superseded'") == []
        assert recorder.statements("MERGE (new)-[:REFINED_FROM]->(old)") == []

    def test_predecessor_path_projection_is_rebuilt(self) -> None:
        """The retarget empties both projections; a live predecessor gets its
        own back from the bindings it still holds, or it reads as source-less
        to every consumer of the cache."""
        recorder, _guard, _retarget, _superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=list(_UNJUDGED))],
            admitted=(_TRIGGER,),
        )

        rebuilds = recorder.statements("SET old.source_paths =")
        assert len(rebuilds) == 1
        cypher, params = rebuilds[0]
        assert params["names"] == ["radial_coordinate"]
        # Rebuilt from live bindings, not from the parameter list.
        assert "(source:StandardNameSource)-[:PRODUCED_NAME]->(old)" in cypher

    def test_refusal_is_recorded_in_the_change_ledger(self) -> None:
        """A refused supersession leaves a split; it must be enumerable."""
        recorder, _guard, _retarget, _superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=list(_UNJUDGED))],
            admitted=(_TRIGGER,),
        )

        ledger = recorder.statements("CREATE (change:StandardNameChange")
        assert len(ledger) == 1
        params = ledger[0][1]
        assert params["operation"] == "supersession_deferred"
        assert params["from_name"] == "radial_coordinate"
        assert params["to_name"] == "radial_coordinate_of_line_of_sight"
        assert "2 source(s)" in params["reason"]


class TestJudgedSourcesMigrate:
    def test_recomposed_source_carries_across(self) -> None:
        """The source whose recompose produced the successor is judged by that
        composition, so its binding moves and the drained predecessor retires."""
        recorder, _guard, retarget, superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=[])],
            admitted=(_TRIGGER,),
        )

        assert superseded == 1
        assert retarget.call_args.kwargs["source_ids"] == [_TRIGGER]
        finalize = recorder.statements("GENERATED_SUPERSESSION_FINALIZE")
        assert len(finalize) == 1
        assert finalize[0][1]["plans"] == [
            {
                "old_name": "radial_coordinate",
                "new_name": "radial_coordinate_of_line_of_sight",
                "old_stage": "accepted",
                "source_ids": [_TRIGGER],
            }
        ]
        # Nothing was refused, so no split record and no projection rebuild.
        assert recorder.statements("CREATE (change:StandardNameChange") == []
        assert recorder.statements("SET old.source_paths =") == []

    def test_sources_already_on_the_successor_travel_with_it(self) -> None:
        """A source a composer previously bound to the successor is judged too,
        and must stay in the moved set — the retarget rebuilds the successor's
        path projection from exactly that set."""
        sibling = "dd:ece/line_of_sight/first_point/r"
        recorder, _guard, retarget, superseded = _drive(
            [_preflight_row(judged=[_TRIGGER, sibling], retained=[])],
            admitted=(_TRIGGER, sibling),
        )

        assert superseded == 1
        assert retarget.call_args.kwargs["source_ids"] == sorted([_TRIGGER, sibling])
        assert recorder.statements("SET old.source_paths =") == []


class TestFailClosed:
    def test_missing_successor_binding_for_the_recomposed_source_refuses(
        self,
    ) -> None:
        """Without the composer's own edge there is no judged pairing at all."""
        recorder = _Recorder(
            [
                _preflight_row(
                    judged=list(_UNJUDGED), retained=[], trigger_source_id=None
                )
            ]
        )
        with (
            patch(_GC_PATH, return_value=recorder),
            pytest.raises(RuntimeError, match="no successor binding"),
        ):
            supersede_prior_source_names(
                [
                    {
                        "new_name": "radial_coordinate_of_line_of_sight",
                        "source_id": _TRIGGER_PATH,
                    }
                ]
            )

        assert recorder.rolled_back is True
        assert recorder.committed is False
        assert recorder.statements("GENERATED_SUPERSESSION_FINALIZE") == []

    def test_guard_rejection_still_rolls_back_the_refused_pass(self) -> None:
        """The mechanical attachment guard keeps its veto over the judged set."""
        rejected = AttachmentVerdict(
            _TRIGGER,
            _TRIGGER_PATH,
            "radial_coordinate_of_line_of_sight",
            "drafted",
            "unit dimensionality mismatch",
        )
        recorder = _Recorder(
            [_preflight_row(judged=[_TRIGGER], retained=list(_UNJUDGED))]
        )
        with (
            patch(_GC_PATH, return_value=recorder),
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=AttachmentPairingGuardResult((), (rejected,)),
            ),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources"
            ) as retarget,
            pytest.raises(ValueError, match="supersession rolled back"),
        ):
            supersede_prior_source_names(
                [
                    {
                        "new_name": "radial_coordinate_of_line_of_sight",
                        "source_id": _TRIGGER_PATH,
                    }
                ]
            )

        retarget.assert_not_called()
        assert recorder.rolled_back is True
        assert recorder.statements("CREATE (change:StandardNameChange") == []


class TestPreflightContract:
    """The classification lives in the emitted Cypher; pin its shape so a
    later edit cannot quietly restore a whole-population enumeration."""

    def _preflight_cypher(self) -> str:
        recorder, _guard, _retarget, _superseded = _drive(
            [_preflight_row(judged=[_TRIGGER], retained=[])],
            admitted=(_TRIGGER,),
        )
        statements = recorder.statements("GENERATED_SUPERSESSION_PREFLIGHT")
        assert len(statements) == 1
        return " ".join(statements[0][0].split())

    def test_judged_set_is_defined_by_a_successor_binding(self) -> None:
        cypher = self._preflight_cypher()
        assert "EXISTS { (source)-[:PRODUCED_NAME]->(new) } AS judged" in cypher
        assert "CASE WHEN judged THEN source.id END" in cypher

    def test_unjudged_sources_are_enumerated_not_dropped(self) -> None:
        """They must come back so the caller can see the predecessor is held."""
        cypher = self._preflight_cypher()
        assert "AS retained_source_ids" in cypher
        assert "CASE WHEN judged THEN null ELSE source.id END" in cypher

    def test_recomposed_source_is_resolved_through_its_dd_edge(self) -> None:
        """Not by rebuilding the composite id, which would encode the scheme."""
        cypher = self._preflight_cypher()
        assert (
            "OPTIONAL MATCH (trigger:StandardNameSource)-[:FROM_DD_PATH]->(src)"
            in cypher
        )

    def test_the_dd_edge_alone_does_not_single_out_the_recomposed_source(
        self,
    ) -> None:
        """A renamed DD path keeps a source under its superseded spelling beside
        the current one, so both anchor the same node. The successor binding is
        what tells them apart; without it every alias is offered as the
        recomposed source and the pass aborts a batch it should have run."""
        cypher = self._preflight_cypher()
        assert (
            "OPTIONAL MATCH (trigger:StandardNameSource)-[:FROM_DD_PATH]->(src) "
            "WHERE EXISTS { (trigger)-[:PRODUCED_NAME]->(new) }" in cypher
        )

    def test_protected_predecessors_stay_excluded(self) -> None:
        """Published catalog content, terminal stages and structural parents are
        filtered before any source is classified."""
        cypher = self._preflight_cypher()
        assert (
            "NOT coalesce(old.name_stage, '') IN "
            "['superseded', 'exhausted', 'contested', 'approved']" in cypher
        )
        assert "old.catalog_pr_number IS NULL" in cypher
        assert "coalesce(old.origin, 'pipeline') <> 'derived'" in cypher
