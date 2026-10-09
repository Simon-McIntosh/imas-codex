"""A binding targets a data field, never a structure.

imas-ambix refuses a binding whose target is a structure such as
``pf_active/coil/current`` with "target is not a data field". ``validate_mappings``
retargets a value binding to the structure's ``data`` leaf and a time binding to
its ``time`` child, recording the rewrite in the binding's evidence; a structure
carrying no such child is refused as an escalation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.ids.mapping import validate_mappings
from imas_codex.ids.models import (
    EscalationSeverity,
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
)

SOURCE = "jt-60sa:pf-current"
SECTION = "pf_active/coil"


def _sections() -> TargetAssignmentBatch:
    return TargetAssignmentBatch(
        ids_name="pf_active",
        assignments=[
            TargetAssignment(
                source_id=SOURCE,
                imas_target_path=SECTION,
                confidence=0.9,
                reasoning="selected",
            )
        ],
    )


def _batch(entry: SignalMappingEntry) -> SignalMappingBatch:
    return SignalMappingBatch(
        ids_name="pf_active", target_path=SECTION, mappings=[entry]
    )


def _entry(target_id: str, *, source_property: str = "value") -> SignalMappingEntry:
    return SignalMappingEntry(
        source_id=SOURCE,
        source_property=source_property,
        target_id=target_id,
        transform_expression="value",
        confidence=0.9,
        reasoning="r",
    )


# DD shape of the paths the tests exercise: ``current`` is a STRUCTURE with
# ``data`` and ``time`` leaves; ``circuit/current`` is a STRUCTURE with neither.
_DD = {
    "pf_active/coil/current": ("STRUCTURE", True, True),
    "pf_active/circuit/current": ("STRUCTURE", False, False),
    "pf_active/coil/element/geometry/rectangle/r": ("FLT_0D", False, False),
}


def _gc() -> MagicMock:
    gc = MagicMock()

    def _query(statement, **params):
        paths = params.get("paths")
        if paths is None:
            return []
        rows = []
        for path in paths:
            if path in _DD:
                data_type, has_data, has_time = _DD[path]
                rows.append(
                    {
                        "path": path,
                        "data_type": data_type,
                        "has_data": has_data,
                        "has_time": has_time,
                    }
                )
        return rows

    gc.query.side_effect = _query
    return gc


def _validate(batch: SignalMappingBatch):
    passing = MagicMock()
    passing.escalations = []
    passing.duplicate_targets = []
    passing.all_passed = True
    passing.binding_checks = []
    with (
        patch("imas_codex.ids.validation.validate_mapping", return_value=passing),
        patch("imas_codex.ids.validation.check_coverage_threshold", return_value=[]),
        patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
    ):
        return validate_mappings(
            "jt-60sa", "pf_active", "4.1.1", _sections(), [batch], gc=_gc()
        )


def test_value_binding_on_a_structure_retargets_to_its_data_child():
    result = _validate(_batch(_entry("pf_active/coil/current")))

    assert [b.target_id for b in result.bindings] == ["pf_active/coil/current/data"]
    assert "retargeted" in result.bindings[0].evidence
    assert "pf_active/coil/current" in result.bindings[0].evidence


def test_time_binding_on_a_structure_retargets_to_its_time_child():
    result = _validate(_batch(_entry("pf_active/coil/current", source_property="time")))

    assert [b.target_id for b in result.bindings] == ["pf_active/coil/current/time"]


def test_structure_without_a_data_child_is_refused():
    result = _validate(_batch(_entry("pf_active/circuit/current")))

    assert result.bindings == []
    refused = [e for e in result.escalations if e.severity == EscalationSeverity.ERROR]
    assert len(refused) == 1
    assert "pf_active/circuit/current" in refused[0].reason
    assert "no data child" in refused[0].reason


def test_leaf_binding_is_left_unchanged():
    leaf = "pf_active/coil/element/geometry/rectangle/r"
    result = _validate(_batch(_entry(leaf)))

    assert [b.target_id for b in result.bindings] == [leaf]
    assert result.bindings[0].evidence == ""
