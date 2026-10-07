"""Member identifiers carry only the numeric runs that vary across a source.

A signal source groups accessors that share a pattern but differ in one or
more numeric runs. The member identifier is the part that distinguishes one
member from its peers, so a run held constant across every member belongs to
the group and must not appear in the name or description.
"""

from __future__ import annotations

from imas_codex.discovery.signals.parallel import (
    extract_member_identifier,
    individualize_members,
)


def _jt60sa_members():
    return [
        f"eddbreadTime('E101173', 'CryoCP', 'm1TAF{n}', t1, t2)" for n in range(81, 87)
    ]


def test_jt60sa_constant_shot_dropped():
    """A constant shot in every accessor is not part of the member id."""
    members = _jt60sa_members()
    identifiers = [extract_member_identifier(a, members) for a in members]
    assert identifiers == ["81", "82", "83", "84", "85", "86"]


def test_two_varying_runs_join():
    members = [
        "CALIB_GAS_010:PROPERTIES:PARAM_048:LIM",
        "CALIB_GAS_020:PROPERTIES:PARAM_050:LIM",
    ]
    assert (
        extract_member_identifier("CALIB_GAS_010:PROPERTIES:PARAM_048:LIM", members)
        == "010/048"
    )
    assert (
        extract_member_identifier("CALIB_GAS_020:PROPERTIES:PARAM_050:LIM", members)
        == "020/050"
    )


def test_constant_first_run_dropped():
    members = ["FOO_007:BAR_001", "FOO_007:BAR_002"]
    assert extract_member_identifier("FOO_007:BAR_001", members) == "001"
    assert extract_member_identifier("FOO_007:BAR_002", members) == "002"


def test_individualize_members_uses_varying_runs():
    members = [
        {"id": f"s{n}", "accessor": a, "node_description": ""}
        for n, a in enumerate(_jt60sa_members(), start=81)
    ]
    out = individualize_members("Channel {member_id}", "Signal {member_id}", members)
    names = {r["id"]: r["name"] for r in out}
    assert names["s81"] == "Channel 81"
    assert names["s86"] == "Channel 86"
    assert all("101173" not in r["name"] for r in out)
    assert all("101173" not in r["description"] for r in out)
