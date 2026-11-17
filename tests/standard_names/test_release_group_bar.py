"""A group release cut refuses a group that does not meet the ruled bar.

Each refusal is exercised on a group planted with that one fault, and a group
planted with none is cut. The bar itself is read from the ruling document, so
an unruled bar refuses rather than guessing a threshold; the tests state their
own test bar and record the group's audit records in a temporary directory.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.standard_names.export import (
    GATE_GROUP_BAR,
    ExclusionRecord,
    GateResult,
    GroupReleaseRuling,
    _group_family_issues,
    _run_group_bar_gate,
    audited_text_digest,
    load_group_audit_record,
    load_group_release_ruling,
    run_export,
)
from tests.standard_names.conftest import clean_source_audit_graph

PLAIN_DOCUMENTATION = "A quantity stated plainly, counted once for the plasma."
GROUP = "equilibrium"


def _member(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "description": f"Description of {name}.",
        "documentation": PLAIN_DOCUMENTATION,
        "kind": "scalar",
        "unit": "1",
        "physics_domain": GROUP,
        "links": [],
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_has_live_child": False,
        "_is_parent": False,
    }
    candidate.update(overrides)
    return candidate


def _ruling(**overrides) -> GroupReleaseRuling:
    """The stated test bar. The lead's ruling replaces these numbers."""
    fields: dict = {"max_defect_rate": 0.2}
    fields.update(overrides)
    return GroupReleaseRuling(**fields)


def _write_audit_record(
    directory: Path,
    candidates: dict[str, dict],
    defects: dict[str, list[str]],
    *,
    group: str = GROUP,
    drop: tuple[str, ...] = (),
    text: dict[str, str] | None = None,
) -> Path:
    """Record one audit of *candidates*, one row per member not dropped."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, candidate in candidates.items():
        if name in drop:
            continue
        documentation = (text or {}).get(name, candidate["documentation"])
        rows.append(
            {
                "name": name,
                "text_sha256": audited_text_digest(
                    candidate["description"], documentation
                ),
                "defects": defects.get(name, []),
            }
        )
    path = directory / f"{group}.json"
    path.write_text(
        json.dumps(
            {
                "group": group,
                "read_at": "2026-11-17T12:00:00+00:00",
                "members": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


def _gate(
    *,
    members: dict[str, str] | None = None,
    candidates: dict[str, dict] | None = None,
    exclusions: list[ExclusionRecord] | None = None,
    sourceless_ids: list[str] | None = None,
    ruling: GroupReleaseRuling | None = None,
    audit_dir: Path | None = None,
) -> GateResult:
    candidates = candidates if candidates is not None else {}
    return _run_group_bar_gate(
        group=GROUP,
        members=members if members is not None else {},
        candidates=candidates,
        exclusions=exclusions or [],
        sourceless_ids=sourceless_ids or [],
        ruling=ruling if ruling is not None else _ruling(),
        bar_problem="",
        audit_dir=audit_dir,
    )


def _causes(result: GateResult) -> list[str]:
    return sorted({str(issue["type"]) for issue in result.issues})


# ---------------------------------------------------------------------------
# A group planted with no fault is cut
# ---------------------------------------------------------------------------


def test_clean_group_is_cut(tmp_path: Path) -> None:
    candidates = {
        "plasma_current": _member("plasma_current"),
        "electron_temperature": _member("electron_temperature"),
    }
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members=dict.fromkeys(candidates, "accepted"),
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert result.gate == GATE_GROUP_BAR
    assert result.passed, result.issues
    assert result.issues == []


# ---------------------------------------------------------------------------
# A member not accepted on name or documentation refuses
# ---------------------------------------------------------------------------


def test_member_not_accepted_on_name_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current", name_stage="reviewed")}
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members={"plasma_current": "reviewed"},
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert not result.passed
    assert _causes(result) == ["group_member_not_accepted"]
    issue = result.issues[0]
    assert issue["name"] == "plasma_current"
    assert issue["cause"] == "name_not_accepted"


def test_member_not_accepted_on_documentation_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members={"plasma_current": "accepted"},
        candidates=candidates,
        exclusions=[
            ExclusionRecord(
                standard_name_id="plasma_current",
                stage="eligibility",
                reason="documentation_not_accepted",
                detail="docs_stage='reviewed'",
            )
        ],
        audit_dir=tmp_path,
    )

    assert not result.passed
    assert _causes(result) == ["group_member_not_accepted"]
    assert result.issues[0]["cause"] == "documentation_not_accepted"


# ---------------------------------------------------------------------------
# A sourceless member refuses; a ruled exception clears it
# ---------------------------------------------------------------------------


def test_sourceless_member_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members={"plasma_current": "accepted"},
        candidates=candidates,
        sourceless_ids=["plasma_current"],
        audit_dir=tmp_path,
    )

    assert not result.passed
    assert _causes(result) == ["group_member_sourceless"]
    assert result.issues[0]["name"] == "plasma_current"


def test_ruled_source_exception_clears_a_sourceless_member(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members={"plasma_current": "accepted"},
        candidates=candidates,
        sourceless_ids=["plasma_current"],
        ruling=_ruling(sourceless_exceptions=frozenset({"plasma_current"})),
        audit_dir=tmp_path,
    )

    assert result.passed, result.issues


# ---------------------------------------------------------------------------
# A member whose shared wording is not its stored sentence refuses
# ---------------------------------------------------------------------------


def test_altered_shared_wording_refuses(tmp_path: Path) -> None:
    candidates = {
        "argon_density": _member(
            "argon_density",
            documentation="The argon density is measured once per discharge.",
        )
    }
    _write_audit_record(tmp_path, candidates, defects={})

    result = _gate(
        members={"argon_density": "accepted"},
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert not result.passed
    assert _causes(result) == ["member_shared_wording_altered"]
    assert result.issues[0]["name"] == "argon_density"


# ---------------------------------------------------------------------------
# A family with a member in another group refuses
# ---------------------------------------------------------------------------


def _family_rows() -> list[tuple[str, str, str]]:
    """One vector family of three paths, split across two groups."""
    return [
        ("equilibrium/foo/radial", "equilibrium_name", GROUP),
        ("equilibrium/foo/toroidal", "equilibrium_name", GROUP),
        ("equilibrium/foo/vertical", "field_systems_name", "magnetic_field_systems"),
    ]


def _family_paths(rows) -> list[dict]:
    return [
        {
            "dd_path": path,
            "unit": "m",
            "sn_id": name,
            "physics_domain": group,
        }
        for path, name, group in rows
    ]


def test_family_with_a_member_in_another_group_refuses() -> None:
    issues = _group_family_issues(_family_paths(_family_rows()), GROUP, _ruling())

    assert [issue["type"] for issue in issues] == ["family_spans_groups"]
    issue = issues[0]
    assert issue["members_in_group"] == ["equilibrium_name"]
    assert issue["other_groups"] == {"magnetic_field_systems": ["field_systems_name"]}


def test_ruled_family_exception_or_a_joint_cut_clears_the_family() -> None:
    family_id = _group_family_issues(_family_paths(_family_rows()), GROUP, _ruling())[
        0
    ]["family"]

    as_exception = _group_family_issues(
        _family_paths(_family_rows()),
        GROUP,
        _ruling(family_exceptions=frozenset({family_id})),
    )
    cut_together = _group_family_issues(
        _family_paths(_family_rows()),
        GROUP,
        _ruling(
            groups_cut_together=frozenset(
                {frozenset({GROUP, "magnetic_field_systems"})}
            )
        ),
    )

    assert as_exception == []
    assert cut_together == []


# ---------------------------------------------------------------------------
# The audited defect rate: above the bar, above a class maximum, unmeasured,
# incomplete, stale
# ---------------------------------------------------------------------------


def _three_members() -> dict[str, dict]:
    return {
        name: _member(name)
        for name in ("plasma_current", "electron_temperature", "ion_temperature")
    }


def test_defect_rate_above_the_bar_refuses(tmp_path: Path) -> None:
    candidates = _three_members()
    # Two of three members defective: 0.667 against a 0.2 bar.
    _write_audit_record(
        tmp_path,
        candidates,
        defects={
            "plasma_current": ["name_omits_source"],
            "electron_temperature": ["name_definition_disagree"],
        },
    )

    result = _gate(
        members=dict.fromkeys(candidates, "accepted"),
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert _causes(result) == ["group_defect_rate_above_bar"]
    issue = result.issues[0]
    assert issue["defective"] == 2
    assert issue["members"] == 3
    assert issue["bar"] == 0.2


def test_class_rate_above_the_class_maximum_refuses(tmp_path: Path) -> None:
    candidates = _three_members()
    # One of three is defective (0.333 under the 0.4 group bar) but the class
    # maximum for that class is 0.1.
    _write_audit_record(
        tmp_path,
        candidates,
        defects={"plasma_current": ["name_contradicts_source"]},
    )

    result = _gate(
        members=dict.fromkeys(candidates, "accepted"),
        candidates=candidates,
        ruling=_ruling(
            max_defect_rate=0.4,
            class_maxima={"name_contradicts_source": 0.1},
        ),
        audit_dir=tmp_path,
    )

    assert _causes(result) == ["group_defect_class_rate_above_bar"]
    issue = result.issues[0]
    assert issue["defect_class"] == "name_contradicts_source"
    assert issue["defective"] == 1


def test_group_never_measured_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}

    result = _gate(
        members={"plasma_current": "accepted"},
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert _causes(result) == ["group_audit_never_measured"]
    assert "no audit record" in result.issues[0]["detail"]


def test_audit_not_covering_every_member_refuses(tmp_path: Path) -> None:
    candidates = {
        "plasma_current": _member("plasma_current"),
        "electron_temperature": _member("electron_temperature"),
    }
    _write_audit_record(
        tmp_path, candidates, defects={}, drop=("electron_temperature",)
    )

    result = _gate(
        members=dict.fromkeys(candidates, "accepted"),
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert _causes(result) == ["group_audit_incomplete"]
    assert result.issues[0]["missing_members"] == ["electron_temperature"]


def test_audit_measured_before_a_text_change_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}
    # The audit read one documentation; the stored text is now another.
    _write_audit_record(
        tmp_path,
        candidates,
        defects={},
        text={"plasma_current": "A quantity written before the repair."},
    )

    result = _gate(
        members={"plasma_current": "accepted"},
        candidates=candidates,
        audit_dir=tmp_path,
    )

    assert _causes(result) == ["group_audit_stale"]
    assert result.issues[0]["names"] == ["plasma_current"]


# ---------------------------------------------------------------------------
# A bar nobody ruled refuses the cut
# ---------------------------------------------------------------------------


def test_unruled_bar_refuses(tmp_path: Path) -> None:
    candidates = {"plasma_current": _member("plasma_current")}
    _write_audit_record(tmp_path, candidates, defects={})

    result = _run_group_bar_gate(
        group=GROUP,
        members={"plasma_current": "accepted"},
        candidates=candidates,
        exclusions=[],
        sourceless_ids=[],
        ruling=None,
        bar_problem="no release bar ruling is configured",
        audit_dir=tmp_path,
    )

    assert not result.passed
    assert _causes(result) == ["release_bar_not_ruled"]
    assert result.issues[0]["detail"] == "no release bar ruling is configured"


def test_ruling_document_is_read_or_reported(tmp_path: Path) -> None:
    ruling_path = tmp_path / "ruling.json"
    ruling_path.write_text(
        json.dumps(
            {
                "max_defect_rate": 0.05,
                "class_maxima": {"name_omits_source": 0.02},
                "sourceless_exceptions": ["plasma_current"],
                "family_exceptions": ["equilibrium/foo|physical_vector|abc12345"],
                "groups_cut_together": [["equilibrium", "magnetohydrodynamics"]],
            }
        ),
        encoding="utf-8",
    )

    ruling, problem = load_group_release_ruling(ruling_path)
    assert problem == ""
    assert ruling is not None
    assert ruling.max_defect_rate == 0.05
    assert ruling.class_maxima == {"name_omits_source": 0.02}
    assert ruling.sourceless_exceptions == frozenset({"plasma_current"})
    assert ruling.cut_with("equilibrium", "magnetohydrodynamics")
    assert not ruling.cut_with("equilibrium", "transport")

    missing, missing_problem = load_group_release_ruling(tmp_path / "absent.json")
    assert missing is None
    assert "no release bar ruling is recorded" in missing_problem

    # A document without the rate is a problem, never a looser bar.
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps({"class_maxima": {}}), encoding="utf-8")
    partial, partial_problem = load_group_release_ruling(partial_path)
    assert partial is None
    assert "max_defect_rate" in partial_problem


# ---------------------------------------------------------------------------
# The group cut refuses through the export, not only at the gate
# ---------------------------------------------------------------------------


@clean_source_audit_graph
class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def test_domain_scoped_cut_refuses_and_writes_no_catalog(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    audit_dir = tmp_path / "audit"
    candidates = {"plasma_current": _member("plasma_current")}
    _write_audit_record(audit_dir, candidates, defects={})
    ruling_path = tmp_path / "ruling.json"
    ruling_path.write_text(json.dumps({"max_defect_rate": 0.2}), encoding="utf-8")

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=list(candidates.values()),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_retired_batch_ids",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_group_live_members",
                return_value={"plasma_current": "reviewed"},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_group_family_paths",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._run_gate_a",
                return_value=GateResult(gate="graph_tests", passed=True, skipped=True),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_ReadOnlyGraphClient(),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.settings.get_sn_group_release_bar",
                return_value=ruling_path,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.settings.get_sn_group_audit_dir",
                return_value=audit_dir,
            )
        )
        report = run_export(
            staging,
            domain=GROUP,
            gate_scope="domain",
        )

    group_bar = next(
        gate for gate in report.gate_results if gate.gate == GATE_GROUP_BAR
    )
    assert not group_bar.passed
    assert "group_member_not_accepted" in _causes(group_bar)
    assert not report.all_gates_passed
    assert not (staging / "catalog.yml").exists()
