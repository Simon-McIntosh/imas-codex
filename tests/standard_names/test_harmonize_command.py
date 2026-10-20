"""`sn harmonize` reports pattern families over one scope, writing nothing.

The command is the single read-only family surface: one row per pattern
family carrying its members, documentation anchor, description drift and
lint finding counts by class, then a scope total. Its read-only decision is
the mutation ban, so the tests make the writing audit preflight raise and
show the command still succeeds — and a negative-control test that points
the report at the full audit shows the ban is live rather than vacuous.

Scope is exactly one of ``--batch``, ``--family``, ``--concept`` or
``--all``; anything else is a usage error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

_MEMBERS = [
    {
        "id": "alpha_base_at_boundary",
        "physical_base": "alpha_base",
        "description": "alpha base at plasma boundary",
        "documentation": "Alpha base at the plasma boundary.",
        "docs_stage": "drafted",
        "harmonized_group_signature": None,
        "has_dd_source": True,
    },
    {
        "id": "alpha_base_at_axis",
        "physical_base": "alpha_base",
        "description": "beta gamma delta",
        "documentation": "A different opening entirely.",
        "docs_stage": "accepted",
        "harmonized_group_signature": None,
        "has_dd_source": True,
    },
    {
        "id": "alpha_base_of_probe",
        "physical_base": "alpha_base",
        "description": "epsilon zeta eta",
        "documentation": "Yet another opening.",
        "docs_stage": "drafted",
        "harmonized_group_signature": None,
        "has_dd_source": True,
    },
]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _FamilyGraph:
    """GraphClient stand-in answering the read-only family queries.

    Groups the fixture nodes by their invariant grammar signature, so it is
    the query's own notion of a family, and records every query issued so a
    caller can assert none of them wrote. A ``kind`` other than the plain
    read is never produced by this surface.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = [dict(n) for n in nodes]
        self.queries: list[str] = []

    def __enter__(self) -> _FamilyGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "RETURN m.id AS id" in cypher and "members" not in cypher:
            return [{"id": n["id"]} for n in self.nodes]
        if "RETURN m.id AS id" in cypher or "RETURN sn.id AS id" in cypher:
            return [{"id": n["id"]} for n in self.nodes]
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for node in self.nodes:
            groups.setdefault(_invariant(node), []).append(node)
        seeds = set(params.get("seeds") or [])
        rows = []
        for key, members in groups.items():
            if seeds and not (seeds & {m["id"] for m in members}):
                continue
            row = dict(zip(PATTERN_INVARIANT_SEGMENTS, key, strict=True))
            row["members"] = members
            rows.append(row)
        return rows


def _install(
    monkeypatch: Any, nodes: list[dict[str, Any]] | None = None
) -> _FamilyGraph:
    from imas_codex.graph import client as graph_client

    graph = _FamilyGraph(nodes if nodes is not None else _MEMBERS)
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)
    return graph


def _rows(output: str) -> list[dict[str, Any]]:
    match = output[output.index("[") : output.rindex("]") + 1]
    return json.loads(match)


def test_family_scope_reports_members_drift_anchor_and_total(
    monkeypatch: Any,
) -> None:
    from imas_codex.cli.sn import sn

    _install(monkeypatch)

    result = CliRunner().invoke(
        sn, ["harmonize", "--family", "alpha_base_at_axis", "--json"]
    )

    assert result.exit_code == 0, result.output
    rows = _rows(result.output)
    assert len(rows) == 1
    family = rows[0]
    assert family["physical_base"] == "alpha_base"
    # Written out independently of _MEMBERS: the report's enumeration over the
    # fixture changes the answer, so a stubbed or dropped grouping fails here
    # rather than moving both sides of the comparison together.
    assert family["n"] == 3
    assert set(family["members"]) == {
        "alpha_base_at_boundary",
        "alpha_base_at_axis",
        "alpha_base_of_probe",
    }
    assert family["drift"] == pytest.approx(1.0 - 1.0 / 3)
    assert family["anchor"] is not None


def test_human_report_prints_a_row_per_family_and_a_scope_total(
    monkeypatch: Any,
) -> None:
    from imas_codex.cli.sn import sn

    _install(monkeypatch)

    result = CliRunner().invoke(sn, ["harmonize", "--family", "alpha_base_at_axis"])

    assert result.exit_code == 0, result.output
    assert "family alpha_base" in result.output
    assert "members=3" in result.output
    assert "scope total: families=1  members=3" in result.output


def test_batch_scope_reports_the_roster_families(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from imas_codex.cli.sn import sn

    _install(monkeypatch)
    roster = tmp_path / "roster.sn_names.yaml"
    roster.write_text(
        "kind: sn_names\nschema_version: 1\nname: test-roster\n"
        "names:\n  - alpha_base_at_axis\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(sn, ["harmonize", "--batch", str(roster), "--json"])

    assert result.exit_code == 0, result.output
    assert _rows(result.output)[0]["n"] == 3


def test_all_scope_enumerates_every_family(monkeypatch: Any) -> None:
    from imas_codex.cli.sn import sn

    extra = dict(_MEMBERS[0])
    extra["id"] = "gamma_base"
    extra["physical_base"] = "gamma_base"
    extra["description"] = "gamma base quantity"
    _install(monkeypatch, [*_MEMBERS, extra])

    result = CliRunner().invoke(sn, ["harmonize", "--all", "--json"])

    assert result.exit_code == 0, result.output
    rows = _rows(result.output)
    assert {row["physical_base"] for row in rows} == {"alpha_base", "gamma_base"}


def _domain_node(
    node_id: str, base: str, *, domain: str, position: str
) -> dict[str, Any]:
    """A minimal member whose domain carries one axis-ambiguous position form."""
    node = dict(_MEMBERS[0])
    node.update(
        {
            "id": node_id,
            "physical_base": base,
            "description": f"Docs for {node_id}.",
            "documentation": f"Documentation for {node_id}.",
            "physics_domain": domain,
            "position": position,
        }
    )
    return node


def test_domain_scoped_finding_is_attributed_and_counted_once(
    monkeypatch: Any,
) -> None:
    """A cross-name domain finding reaches the family rows, counted once.

    The lexical lint keys a mixed-position-form finding on ``[domain:<d>]``
    (no member), observed across two families that share physics domain
    ``equilibrium``. Each family row carries it in ``domain_lint``; the scope
    total counts the one distinct finding once, so summing the rows agrees.
    """
    from imas_codex.cli.sn import sn

    _install(
        monkeypatch,
        [
            _domain_node(
                "alpha_base_at_magnetic_axis",
                "alpha_base",
                domain="equilibrium",
                position="at_magnetic_axis",
            ),
            _domain_node(
                "beta_base_on_axis",
                "beta_base",
                domain="equilibrium",
                position="on_axis",
            ),
        ],
    )

    result = CliRunner().invoke(sn, ["harmonize", "--all", "--json"])

    assert result.exit_code == 0, result.output
    rows = _rows(result.output)
    assert {row["physical_base"] for row in rows} == {"alpha_base", "beta_base"}
    # The synthetic-id finding matches no member, so it is attributed to every
    # family whose members belong to the domain.
    assert [row["domain_lint"].get("domain_convention_violation") for row in rows] == [
        1,
        1,
    ]
    # ...and counted in the scope total exactly once.
    assert sum(row["lint"].get("domain_convention_violation", 0) for row in rows) == 1
    # Row lint_totals written out independently: alpha_base carries its two
    # member findings plus the one domain finding (folded in at the first
    # family in report order), beta_base its two member findings only.
    assert [row["lint_total"] for row in rows] == [3, 2]


def test_concept_scope_reports_its_families(monkeypatch: Any) -> None:
    from imas_codex.cli.sn import sn
    from imas_codex.standard_names import harmonize

    _install(monkeypatch)
    monkeypatch.setattr(
        harmonize, "concept_member_ids", lambda concept, gc=None: ["alpha_base_at_axis"]
    )

    result = CliRunner().invoke(sn, ["harmonize", "--concept", "temperature", "--json"])

    assert result.exit_code == 0, result.output
    assert _rows(result.output)[0]["physical_base"] == "alpha_base"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--family", "a", "--all"],
        ["--batch", "x", "--concept", "y"],
    ],
)
def test_scope_must_be_exactly_one(monkeypatch: Any, argv: list[str]) -> None:
    from imas_codex.cli.sn import sn

    _install(monkeypatch)

    result = CliRunner().invoke(sn, ["harmonize", *argv])

    assert result.exit_code == 2, result.output
    assert "exactly one scope" in result.output


def test_report_flag_is_accepted_as_the_default_mode(monkeypatch: Any) -> None:
    from imas_codex.cli.sn import sn

    _install(monkeypatch)

    result = CliRunner().invoke(sn, ["harmonize", "--all", "--report"])

    assert result.exit_code == 0, result.output


def test_report_never_runs_the_writing_preflight(monkeypatch: Any) -> None:
    """The read-only decision: the embedding-writing preflight is never reached."""
    from imas_codex.cli.sn import sn
    from imas_codex.standard_names.review import audits

    graph = _install(monkeypatch)

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("sn harmonize must not run the embedding preflight")

    monkeypatch.setattr(audits, "run_embedding_preflight", _refuse)
    monkeypatch.setattr(audits, "run_all_audits", _refuse)

    result = CliRunner().invoke(sn, ["harmonize", "--all"])

    assert result.exit_code == 0, result.output
    assert not any("CREATE" in q or "MERGE" in q or "SET " in q for q in graph.queries)


def test_the_ban_is_live_when_the_report_is_pointed_at_the_full_audit(
    monkeypatch: Any,
) -> None:
    """Negative control: reaching run_all_audits trips the preflight ban.

    Points the report's lint at the full audit and shows the invocation that
    succeeds above now fails on the writing preflight — the guard in
    ``test_report_never_runs_the_same_preflight`` is not vacuous.
    """
    from imas_codex.cli.sn import sn
    from imas_codex.standard_names.review import audits

    _install(monkeypatch)

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("sn harmonize must not run the embedding preflight")

    monkeypatch.setattr(audits, "run_embedding_preflight", _refuse)

    def _audit_with_writes(names: list[dict[str, Any]]) -> list[Any]:
        return audits.run_all_audits(names).lint_findings

    monkeypatch.setattr(audits, "run_lint_checks", _audit_with_writes)

    result = CliRunner().invoke(sn, ["harmonize", "--all"])

    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, AssertionError)


def test_run_families_flag_is_unknown() -> None:
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["run", "--families", "flux"])

    assert result.exit_code == 2, result.output
    assert "--families" in result.output


def test_review_target_groups_is_unknown() -> None:
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["review", "--target", "groups"])

    assert result.exit_code == 2, result.output
    assert "groups" in result.output


def test_harmonize_calls_none_of_the_apply_helpers(monkeypatch: Any) -> None:
    """The read-only decision is a mutation ban over the apply helpers."""
    from imas_codex.cli.sn import sn
    from imas_codex.standard_names import harmonize

    graph = _install(monkeypatch)

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("sn harmonize must not mutate the graph")

    for name in (
        "mark_members_for_regen",
        "stamp_harmonized",
        "restamp_harmonized_families",
    ):
        monkeypatch.setattr(harmonize, name, _refuse)
        # Negative control: the ban is live, so a reached helper raises.
        with pytest.raises(AssertionError, match="must not mutate"):
            getattr(harmonize, name)()

    result = CliRunner().invoke(sn, ["harmonize", "--all", "--json"])

    assert result.exit_code == 0, result.output
    assert _rows(result.output)
    assert not any("MERGE" in q for q in graph.queries)
