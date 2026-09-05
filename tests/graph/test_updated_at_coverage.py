"""Specification for the StandardName.updated_at coverage check."""

from __future__ import annotations

from pathlib import Path

from imas_codex.graph.cypher_property_check import audit_standard_name_touch

REPO_ROOT = Path(__file__).resolve().parents[2]

# graph_ops.py is the write path this coverage gate holds to zero. Every
# other module under imas_codex/standard_names/ still has pre-existing gaps
# (a separately dispatched node owns closing those) — scoping here keeps this
# gate meaningful without failing on write paths this change did not touch.
_GATED_PATH = REPO_ROOT / "imas_codex" / "standard_names" / "graph_ops.py"


def test_repository_standard_name_writes_stamp_updated_at() -> None:
    """Every substantive StandardName write in graph_ops.py sets ``updated_at``."""
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
