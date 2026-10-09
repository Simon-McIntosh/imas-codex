"""A binding that fails a validation check is not persisted.

``validate_mappings`` runs four checks per binding — source existence, target
existence, transform execution and unit compatibility. A failing binding used to
be counted into the ``corrections`` note and returned unchanged, so validation
was a report and only a refused graph write stopped a non-existent target from
reaching the graph. A binding that fails any check is now removed from the
returned ``bindings`` and its failure is recorded as an escalation naming the
failed check, while a passing sibling is kept.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.ids.mapping import validate_mappings
from imas_codex.ids.models import (
    EscalationFlag,
    EscalationSeverity,
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
)
from imas_codex.ids.validation import BindingCheck

SOURCE = "jt-60sa:ec-launcher"
SECTION = "ec_launchers/beam"

# The real fields end in ``/data``; these bindings name ``/value`` instead, so
# the target does not exist. The third target is a passing sibling.
MISSING_TARGET = "ec_launchers/beam/power_launched/value"
UNITS_TARGET = "ec_launchers/beam/current/value"
GOOD_TARGET = "ec_launchers/beam/energy/data"

_TARGETS = [MISSING_TARGET, UNITS_TARGET, GOOD_TARGET]


def _entry(target_id: str) -> SignalMappingEntry:
    return SignalMappingEntry(
        source_id=SOURCE,
        source_property="value",
        target_id=target_id,
        transform_expression="value",
        confidence=0.9,
        reasoning="r",
    )


def _batch() -> SignalMappingBatch:
    return SignalMappingBatch(
        ids_name="ec_launchers",
        target_path=SECTION,
        mappings=[_entry(t) for t in _TARGETS],
    )


def _sections() -> TargetAssignmentBatch:
    return TargetAssignmentBatch(
        ids_name="ec_launchers",
        assignments=[
            TargetAssignment(
                source_id=SOURCE,
                imas_target_path=SECTION,
                confidence=0.9,
                reasoning="selected",
            )
        ],
    )


def _report() -> MagicMock:
    """A validation report where one target is missing, one has bad units."""
    report = MagicMock()
    report.binding_checks = [
        BindingCheck(
            source_id=SOURCE,
            target_id=MISSING_TARGET,
            source_exists=True,
            target_exists=False,
            transform_executes=True,
            units_compatible=True,
            error=f"IMAS path '{MISSING_TARGET}' not found",
        ),
        BindingCheck(
            source_id=SOURCE,
            target_id=UNITS_TARGET,
            source_exists=True,
            target_exists=True,
            transform_executes=True,
            units_compatible=False,
            error="Units incompatible: MW → W",
        ),
        BindingCheck(
            source_id=SOURCE,
            target_id=GOOD_TARGET,
            source_exists=True,
            target_exists=True,
            transform_executes=True,
            units_compatible=True,
            error=None,
        ),
    ]
    report.escalations = [
        EscalationFlag(
            source_id=SOURCE,
            target_id=MISSING_TARGET,
            severity=EscalationSeverity.ERROR,
            reason=f"IMAS path '{MISSING_TARGET}' not found",
        ),
        EscalationFlag(
            source_id=SOURCE,
            target_id=UNITS_TARGET,
            severity=EscalationSeverity.ERROR,
            reason="Units incompatible: MW → W",
        ),
    ]
    report.duplicate_targets = []
    report.all_passed = False
    return report


def _validate():
    gc = MagicMock()
    gc.query.side_effect = lambda statement, **params: []
    with (
        patch("imas_codex.ids.validation.validate_mapping", return_value=_report()),
        patch("imas_codex.ids.validation.check_coverage_threshold", return_value=[]),
        patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
    ):
        return validate_mappings(
            "jt-60sa", "ec_launchers", "4.1.1", _sections(), [_batch()], gc=gc
        )


def test_a_binding_with_a_missing_target_is_removed_and_escalated():
    result = _validate()

    assert MISSING_TARGET not in [b.target_id for b in result.bindings]
    reasons = {
        e.target_id: e.reason
        for e in result.escalations
        if e.severity == EscalationSeverity.ERROR
    }
    assert MISSING_TARGET in reasons
    assert "not found" in reasons[MISSING_TARGET]


def test_a_binding_with_incompatible_units_is_removed_and_escalated():
    result = _validate()

    assert UNITS_TARGET not in [b.target_id for b in result.bindings]
    reasons = {
        e.target_id: e.reason
        for e in result.escalations
        if e.severity == EscalationSeverity.ERROR
    }
    assert UNITS_TARGET in reasons
    assert "incompatible" in reasons[UNITS_TARGET].lower()


def test_the_passing_sibling_is_kept():
    result = _validate()

    assert [b.target_id for b in result.bindings] == [GOOD_TARGET]


def test_the_corrections_note_remains_the_summary_of_what_failed():
    result = _validate()

    assert any("failed validation checks" in c for c in result.corrections)
