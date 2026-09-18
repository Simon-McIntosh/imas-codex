"""Measured updated_at omissions are a debt ledger, not an approval, lowered only after fixing the corresponding writes and updating the count."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from imas_codex.graph.cypher_property_check import audit_standard_name_touch

REPO_ROOT = Path(__file__).resolve().parents[2]
_STANDARD_NAMES_ROOT = REPO_ROOT / "imas_codex" / "standard_names"

# These are observed omissions by package-relative file. Keep an entry when a
# file reaches zero so an unrecorded improvement fails until its debt is
# explicitly lowered. New findings also fail until their file is accounted for.
#
# The two survivors are named, justified entries rather than part of an
# opaque total, so a future finding cannot hide inside the count: the package
# test also asserts that the live (path, properties) identities equal
# _SURVIVING_FINDINGS exactly, and _EXPECTED_FINDINGS is derived from it so the
# counts cannot drift from the names. Each survivor is a write its lane argued
# for leaving unstamped, not a debt awaiting repair.
_SURVIVING_FINDINGS = {
    (
        "catalog_import.py",
        ("catalog_commit_sha", "imported_at"),
    ): (
        "provenance receipt: the import stamps imported_at and "
        "catalog_commit_sha through a positive allow-list under a locked "
        "import-write-authority decision, and a test asserts that exact "
        "assigned-property set; stamping updated_at would expand the import "
        "path's authority over name content."
    ),
    ("orphan_sweep.py", ("drain_scope_claimed_at",)): (
        "drain-scope lease heartbeat: writes only liveness on a name whose "
        "ownership it does not change, so a stamp would make updated_at track "
        "heartbeat cadence instead of name modifications."
    ),
    ("publish.py", ("exported_at",)): (
        "catalog export receipt: applies the published manifest timestamp only "
        "after the catalog commit succeeds, recording delivery rather than a "
        "change to the name content."
    ),
    ("workers.py", ("refine_attempts",)): (
        "claim-accounting correction: returns a charged rotation when a pinned "
        "rename is resubmitted unchanged, without modifying the name content."
    ),
}

_EXPECTED_FINDINGS = Counter(path for path, _properties in _SURVIVING_FINDINGS)

_GATED_PATH = REPO_ROOT / "imas_codex" / "standard_names" / "graph_ops.py"


def _finding_counts(findings: tuple) -> Counter[str]:
    """Return omission counts keyed by package-relative source path."""
    return Counter(
        finding.path.relative_to(_STANDARD_NAMES_ROOT).as_posix()
        for finding in findings
    )


def _assert_findings_match_baseline(counts: Counter[str]) -> None:
    """Require every observed file count to equal its declared debt count."""
    unexpected = sorted(set(counts) - set(_EXPECTED_FINDINGS))
    too_high = {
        path: (counts.get(path, 0), expected)
        for path, expected in _EXPECTED_FINDINGS.items()
        if counts.get(path, 0) > expected
    }
    too_low = {
        path: (counts.get(path, 0), expected)
        for path, expected in _EXPECTED_FINDINGS.items()
        if counts.get(path, 0) < expected
    }
    assert not unexpected, f"findings lack a debt baseline: {unexpected}"
    assert not too_high, f"findings exceed the debt baseline: {too_high}"
    assert not too_low, (
        f"findings fell below the debt baseline; lower the recorded count: {too_low}"
    )


def test_standard_name_package_matches_updated_at_debt_baseline() -> None:
    """Every package module is audited against its measured omission count."""
    findings = audit_standard_name_touch(_STANDARD_NAMES_ROOT)
    _assert_findings_match_baseline(_finding_counts(findings))
    live_identities = {
        (
            finding.path.relative_to(_STANDARD_NAMES_ROOT).as_posix(),
            finding.properties,
        )
        for finding in findings
    }
    assert live_identities == set(_SURVIVING_FINDINGS), (
        "surviving updated_at omissions changed identity; add or rename the "
        "justified entry in _SURVIVING_FINDINGS"
    )


def test_debt_baseline_rejects_unrecorded_count_changes() -> None:
    """Both new omissions and unrecorded repairs require an explicit update."""
    _assert_findings_match_baseline(Counter(_EXPECTED_FINDINGS))

    increased = Counter(_EXPECTED_FINDINGS)
    increased["catalog_import.py"] += 1
    with pytest.raises(AssertionError, match="exceed the debt baseline"):
        _assert_findings_match_baseline(increased)

    decreased = Counter(_EXPECTED_FINDINGS)
    decreased["catalog_import.py"] -= 1
    with pytest.raises(AssertionError, match="fell below the debt baseline"):
        _assert_findings_match_baseline(decreased)


def test_graph_ops_standard_name_writes_have_hard_zero() -> None:
    """The primary graph operation module has no tolerated omissions."""
    findings = audit_standard_name_touch(_GATED_PATH)
    assert not findings, (
        "Cypher writes that modify StandardName without stamping updated_at:\n"
        + "\n".join(str(finding) for finding in findings)
    )


def test_transient_lock_only_write_is_exempt(tmp_path: Path) -> None:
    """A statement that only sets a transient claim/lock marker is exempt."""
    fixture = tmp_path / "lock_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn._refine_claim_release_lock = true\n"
        "REMOVE sn._refine_claim_release_lock\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(fixture)

    assert not findings


def test_new_unstamped_write_in_other_module_fails_gate(tmp_path: Path) -> None:
    """An omission in a non-graph_ops module cannot hide from the package gate."""
    fixture = tmp_path / "other_module.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn.description = $description\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == fixture
    with pytest.raises(AssertionError, match="lack a debt baseline"):
        _assert_findings_match_baseline(Counter({"other_module.py": 1}))


def test_substantive_write_without_updated_at_is_flagged(tmp_path: Path) -> None:
    """A SET clause that changes a real property must stamp updated_at."""
    fixture = tmp_path / "missing_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn.kind = $kind\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(fixture)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.alias == "sn"
    assert finding.properties == ("kind",)
    assert finding.path == fixture


def test_substantive_write_with_updated_at_is_not_flagged(tmp_path: Path) -> None:
    """The same write, restored to stamp updated_at, produces no finding."""
    fixture = tmp_path / "restored_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn.kind = $kind, sn.updated_at = datetime()\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(fixture)

    assert not findings


def test_stamp_without_matching_property_write_is_flagged(tmp_path: Path) -> None:
    """A compare-and-set lock that modifies nothing must not stamp updated_at."""
    fixture = tmp_path / "overstamp_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn.updated_at = datetime(), sn.claimed_at = sn.claimed_at\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(fixture)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.alias == "sn"
    assert finding.properties == ()
    assert finding.stamped is True
    assert finding.path == fixture


def test_stamp_removed_from_self_assignment_only_write_is_not_flagged(
    tmp_path: Path,
) -> None:
    """The same lock, restored to drop the stray stamp, produces no finding."""
    fixture = tmp_path / "overstamp_repaired_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        "MATCH (sn:StandardName {id: $id})\n"
        "SET sn.claimed_at = sn.claimed_at\n"
        "RETURN sn.id AS id\n"
        "'''\n",
        encoding="utf-8",
    )

    findings = audit_standard_name_touch(fixture)

    assert not findings
