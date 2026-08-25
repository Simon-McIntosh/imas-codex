"""Regression coverage for structured and free-text DD-gap evidence rules."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from imas_codex.graph.models import DDGapEvidenceRule
from imas_codex.standard_names.dd_gaps import _prepare_reports, write_dd_gaps

PATH = "equilibrium/time_slice/profiles_1d/pressure"
REVIEWER_RULE = "declared unit must match the defined physical quantity"


def _reviewer_report(*, evidence_rule: str) -> dict[str, str]:
    return {
        "path": PATH,
        "kind": "unit_defect",
        "reason": "The DD declares 1 while the pressure definition uses Pa.",
        "reporter": "review-name",
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": evidence_rule,
    }


def test_reviewer_prose_persists_as_visible_manual_evidence(caplog) -> None:
    graph_client = MagicMock()
    graph_client.query.side_effect = [
        [{"id": PATH}],
        [
            {
                "reported": 1,
                "relationships": 1,
                "observations": 1,
                "ids": [f"dd_gap:{PATH}:unit_defect"],
            }
        ],
    ]
    graph = MagicMock()
    graph.return_value.__enter__.return_value = graph_client
    graph.return_value.__exit__.return_value = False

    with (
        patch("imas_codex.standard_names.dd_gaps.GraphClient", graph),
        caplog.at_level(logging.WARNING, logger="imas_codex.standard_names.dd_gaps"),
    ):
        result = write_dd_gaps(
            [_reviewer_report(evidence_rule=REVIEWER_RULE)],
        )

    assert result["reported"] == 1
    persisted = graph_client.query.call_args_list[1].kwargs["batch"][0]
    assert persisted["evidence_rule"] is None
    assert REVIEWER_RULE in persisted["reason"]
    assert "The DD declares 1" in persisted["reason"]
    assert "persisted as manual evidence" in caplog.text


def test_declared_rule_remains_structured_without_widening_the_enum() -> None:
    batch = _prepare_reports([_reviewer_report(evidence_rule="unit_equals_expected")])

    assert batch[0]["evidence_rule"] == "unit_equals_expected"
    assert [item.value for item in DDGapEvidenceRule] == [
        "unit_equals_expected",
        "declaration_present",
        "value_equals_expected",
        "reference_value_matches",
    ]
