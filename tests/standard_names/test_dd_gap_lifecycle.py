"""Lifecycle and release-reconciliation contracts for DD-gap evidence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from imas_codex.standard_names.dd_gaps import (
    DDGapTransitionConflict,
    _evidence_token,
    reconcile_dd_gaps,
    transition_dd_gap,
)

_ANY_EVIDENCE_TOKEN = "dd-gap-evidence:validation-only"


def test_schema_separates_identity_history_from_lifecycle_history() -> None:
    schema_path = Path(__file__).parents[2] / "imas_codex/schemas/standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text())
    identity = schema["classes"]["DDGapIdentityChange"]["attributes"]
    gap = schema["classes"]["DDGap"]["attributes"]

    assert set(identity) == {
        "id",
        "dd_gap_id",
        "old_id",
        "new_id",
        "old_kind",
        "new_kind",
        "changed_at",
        "changed_by",
        "reason",
    }
    assert identity["id"]["identifier"] is True
    assert identity["old_kind"]["range"] == "DDGapKind"
    assert identity["new_kind"]["range"] == "DDGapKind"
    assert identity["changed_at"]["range"] == "datetime"
    assert gap["identity_changes"]["range"] == "DDGapIdentityChange"
    assert gap["identity_changes"]["annotations"]["relationship_type"] == (
        "HAS_IDENTITY_CHANGE"
    )
    assert gap["state_changes"]["range"] == "DDGapStateChange"


def _transition_row(status: str) -> list[dict[str, str]]:
    return [
        {
            "id": "dd_gap:equilibrium/path:unit_defect",
            "from_status": "triaged",
            "status": status,
        }
    ]


def _transition_fact(status: str) -> dict[str, object]:
    return {
        "id": "dd_gap:equilibrium/path:unit_defect",
        "path": "equilibrium/path",
        "kind": "unit_defect",
        "status": status,
        "example_count": 1,
        "first_seen_at": "2026-08-01T10:00:00Z",
        "last_seen_at": "2026-08-01T10:00:00Z",
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
        "reference_path": None,
        "reference_value": None,
        "registry_backend": None,
        "source_paths": ["equilibrium/path"],
        "affected_name_ids": ["plasma_pressure"],
        "affected_path_count": 1,
    }


def _transition_graph(current_status: str, target_status: str) -> tuple[MagicMock, str]:
    gc = MagicMock()
    fact = _transition_fact(current_status)
    observations = [{"id": "observation:1"}]
    gc.query.side_effect = [[fact], observations, [], _transition_row(target_status)]
    token = _evidence_token({**fact, "observations": observations})
    return gc, token


def test_flagged_to_triaged_uses_expected_status_compare_and_set() -> None:
    gc, token = _transition_graph("flagged", "triaged")

    result = transition_dd_gap(
        "dd_gap:equilibrium/path:unit_defect",
        expected_status="flagged",
        new_status="triaged",
        actor="operator@example.org",
        reason="evidence checked against the DD declaration",
        expected_evidence_token=token,
        gc=gc,
    )

    assert result["status"] == "triaged"
    query = gc.query.call_args.args[0]
    assert "gap.status = $expected_status" in query
    assert "size(current_observation_ids) = size($evidence_observation_ids)" in query
    assert "DDGapStateChange" in query
    assert "HAS_STATE_CHANGE" in query


def test_transition_requires_the_reviewed_evidence_token() -> None:
    gc = MagicMock()
    with pytest.raises(TypeError, match="expected_evidence_token"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="flagged",
            new_status="triaged",
            actor="operator@example.org",
            reason="evidence checked",
            gc=gc,
        )
    gc.query.assert_not_called()


def test_transition_rejects_evidence_changed_since_operator_review() -> None:
    gc, token = _transition_graph("flagged", "triaged")
    with pytest.raises(DDGapTransitionConflict, match="evidence changed"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="flagged",
            new_status="triaged",
            actor="operator@example.org",
            reason="evidence checked",
            expected_evidence_token=token + "-stale",
            gc=gc,
        )
    assert gc.query.call_count == 3


def test_registered_exception_requires_registry_provenance() -> None:
    with pytest.raises(ValueError, match="registry_backend"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="triaged",
            new_status="registered_exception",
            actor="operator@example.org",
            reason="curated exception exists",
            expected_evidence_token=_ANY_EVIDENCE_TOKEN,
            gc=MagicMock(),
        )


def test_upstream_issue_requires_https_url() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="triaged",
            new_status="upstream_issue",
            actor="operator@example.org",
            reason="filed after reproducing the defect",
            expected_evidence_token=_ANY_EVIDENCE_TOKEN,
            upstream_url="http://example.invalid/issue/1",
            gc=MagicMock(),
        )


def test_triaged_to_upstream_issue_records_url() -> None:
    gc, token = _transition_graph("triaged", "upstream_issue")
    result = transition_dd_gap(
        "dd_gap:equilibrium/path:unit_defect",
        expected_status="triaged",
        new_status="upstream_issue",
        actor="operator@example.org",
        reason="upstream maintainers can reproduce it",
        expected_evidence_token=token,
        upstream_url="https://example.invalid/issue/1",
        gc=gc,
    )
    assert result["status"] == "upstream_issue"
    assert gc.query.call_args.kwargs["upstream_url"] == (
        "https://example.invalid/issue/1"
    )


def test_rejected_is_a_human_disposition() -> None:
    gc, token = _transition_graph("flagged", "rejected")
    result = transition_dd_gap(
        "dd_gap:equilibrium/path:unit_defect",
        expected_status="flagged",
        new_status="rejected",
        actor="operator@example.org",
        reason="comparison path represents a different quantity",
        expected_evidence_token=token,
        gc=gc,
    )
    assert result["status"] == "rejected"


def test_resolved_upstream_requires_version_and_validation_evidence() -> None:
    with pytest.raises(ValueError, match="published DD version"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="upstream_issue",
            new_status="resolved_upstream",
            actor="dd-release-reconcile",
            reason="release predicate passed",
            expected_evidence_token=_ANY_EVIDENCE_TOKEN,
            validation_evidence="unit now equals Pa",
            gc=MagicMock(),
        )
    with pytest.raises(ValueError, match="validation_evidence"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="upstream_issue",
            new_status="resolved_upstream",
            actor="dd-release-reconcile",
            reason="release predicate passed",
            expected_evidence_token=_ANY_EVIDENCE_TOKEN,
            resolved_dd_version="4.1.1",
            gc=MagicMock(),
        )


def test_invalid_transition_is_rejected_before_graph_access() -> None:
    gc = MagicMock()
    with pytest.raises(ValueError, match="invalid DD-gap transition"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="rejected",
            new_status="upstream_issue",
            actor="operator@example.org",
            reason="attempted implicit reopen",
            expected_evidence_token=_ANY_EVIDENCE_TOKEN,
            upstream_url="https://example.invalid/issue/1",
            gc=gc,
        )
    gc.query.assert_not_called()


def test_compare_and_set_conflict_is_visible() -> None:
    gc = MagicMock()
    fact = _transition_fact("flagged")
    observations = [{"id": "observation:1"}]
    gc.query.side_effect = [[fact], observations, [], []]
    token = _evidence_token({**fact, "observations": observations})
    with pytest.raises(DDGapTransitionConflict, match="expected 'flagged'"):
        transition_dd_gap(
            "dd_gap:equilibrium/path:unit_defect",
            expected_status="flagged",
            new_status="triaged",
            actor="operator@example.org",
            reason="evidence checked",
            expected_evidence_token=token,
            gc=gc,
        )


def test_release_reconcile_resolves_only_proven_unit_correction() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "4.1.1", "is_current": True}],
        [
            {
                "id": "dd_gap:equilibrium/path:unit_defect",
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "upstream_issue",
                "expected_value": "Pa",
                "evidence_rule": "unit_equals_expected",
                "source_paths": ["equilibrium/path"],
                "registry_backend": None,
            }
        ],
        [{"id": "dd_gap:equilibrium/path:unit_defect"}],
    ]

    result = reconcile_dd_gaps(
        "4.1.1",
        {"equilibrium/path": {"unit": "Pa"}},
        gc=gc,
    )

    assert result["resolved"] == 1
    assert result["manual_required"] == []
    mutation = gc.query.call_args_list[2]
    assert "MATCH (version:DDVersion {id: $dd_version})" in mutation.args[0]
    assert "gap.status = item.expected_status" in mutation.args[0]
    assert mutation.kwargs["dd_version"] == "4.1.1"


def test_registered_exception_resolution_reports_stale_registry_fact() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "4.1.1", "is_current": True}],
        [
            {
                "id": "dd_gap:equilibrium/path:unit_defect",
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "registered_exception",
                "expected_value": "Pa",
                "evidence_rule": "unit_equals_expected",
                "source_paths": ["equilibrium/path"],
                "registry_backend": "dd_unit_exceptions",
            }
        ],
        [{"id": "dd_gap:equilibrium/path:unit_defect"}],
    ]
    result = reconcile_dd_gaps("4.1.1", {"equilibrium/path": {"unit": "Pa"}}, gc=gc)
    assert result["stale_registry_entries"] == ["dd_gap:equilibrium/path:unit_defect"]


def test_unsupported_predicate_remains_manual_and_never_mutates() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "4.1.1", "is_current": True}],
        [
            {
                "id": "dd_gap:equilibrium/path:doc_mismatch",
                "path": "equilibrium/path",
                "kind": "doc_mismatch",
                "status": "upstream_issue",
                "expected_value": "better prose",
                "evidence_rule": "documentation_matches_expected",
                "source_paths": ["equilibrium/path"],
                "registry_backend": None,
            }
        ],
    ]
    result = reconcile_dd_gaps(
        "4.1.1", {"equilibrium/path": {"documentation": "better prose"}}, gc=gc
    )
    assert result["resolved"] == 0
    assert result["manual_required"] == [
        {
            "id": "dd_gap:equilibrium/path:doc_mismatch",
            "reason": "unsupported predicate for doc_mismatch",
        }
    ]
    assert gc.query.call_count == 2


def test_missing_release_path_is_not_proof() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "4.1.1", "is_current": True}],
        [
            {
                "id": "dd_gap:equilibrium/path:unit_defect",
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "upstream_issue",
                "expected_value": "Pa",
                "evidence_rule": "unit_equals_expected",
                "source_paths": ["equilibrium/path"],
                "registry_backend": None,
            }
        ],
    ]
    result = reconcile_dd_gaps("4.1.1", {}, gc=gc)
    assert result["resolved"] == 0
    assert result["manual_required"][0]["reason"] == (
        "release facts missing exact path equilibrium/path"
    )


def test_reconcile_rejects_missing_or_noncurrent_dd_version() -> None:
    missing = MagicMock()
    missing.query.return_value = []
    with pytest.raises(ValueError, match="published DD version"):
        reconcile_dd_gaps("4.1.1", {}, gc=missing)

    stale = MagicMock()
    stale.query.return_value = [{"id": "4.1.0", "is_current": False}]
    with pytest.raises(ValueError, match="not current"):
        reconcile_dd_gaps("4.1.0", {}, gc=stale)


def test_lifecycle_queries_never_mutate_pipeline_behavior_fields() -> None:
    gc, token = _transition_graph("flagged", "triaged")
    transition_dd_gap(
        "dd_gap:equilibrium/path:unit_defect",
        expected_status="flagged",
        new_status="triaged",
        actor="operator@example.org",
        reason="evidence checked",
        expected_evidence_token=token,
        gc=gc,
    )
    query = gc.query.call_args.args[0]
    assert "StandardNameSource" not in query
    assert "StandardName" not in query
    assert "SET node." not in query
