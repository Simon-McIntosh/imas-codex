"""Stubbed tests for the read-only live-name domain audit.

Every test drives the audit through a stub graph client, so no live database is
needed. Each of the four checks gets a fixture that passes it and one that fails
it, a derived parent gets its own fixture (its expected set is its live
children's union, not a DD source's domains), and an order-only difference
between the stored property and the derived set must pass — the audit compares
sets, so arrival order carries no meaning.
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.domain_audit import (
    CHECK_DOMAIN_EDGES_SOURCE_DOMAINS,
    CHECK_PHYSICS_DOMAIN_DECISION,
    CHECK_PHYSICS_DOMAIN_SET,
    CHECK_SOURCE_DOMAINS_EXPECTED,
    audit_domains,
)


class _StubGraph:
    """A graph client stand-in: canned rows, no connection, records calls."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        return list(self.rows)

    def close(self) -> None:
        self.closed = True


def _row(
    sn_id: str,
    *,
    stage: str | None = "accepted",
    physics_domain: str | None = None,
    source_domains: list[str] | None = None,
    dd_paths: list[str] | None = None,
    dd_domains: list[str] | None = None,
    child_ids: list[str] | None = None,
    edge_domains: list[str] | None = None,
    edge_count: int | None = None,
    decision_choice: str | None = None,
    correction: str | None = None,
) -> dict[str, Any]:
    return {
        "sn_id": sn_id,
        "name_stage": stage,
        "physics_domain": physics_domain,
        "source_domains": source_domains,
        "dd_paths": dd_paths or [],
        "dd_domains": dd_domains or [],
        "child_ids": child_ids or [],
        "decision_choice": decision_choice,
        "correction": correction,
        "edge_domains": edge_domains or [],
        "edge_count": len(edge_domains or []) if edge_count is None else edge_count,
    }


def _consistent_name(sn_id: str = "sn:equilibrium") -> dict[str, Any]:
    """A DD-sourced name on which all four checks pass."""
    return _row(
        sn_id,
        physics_domain="equilibrium",
        source_domains=["equilibrium", "transport"],
        dd_paths=["equilibrium/time_slice/profiles_1d/psi"],
        dd_domains=["equilibrium", "transport"],
        edge_domains=["equilibrium", "transport"],
        decision_choice="equilibrium",
    )


ALL_CHECKS = (
    CHECK_PHYSICS_DOMAIN_SET,
    CHECK_PHYSICS_DOMAIN_DECISION,
    CHECK_SOURCE_DOMAINS_EXPECTED,
    CHECK_DOMAIN_EDGES_SOURCE_DOMAINS,
)


class TestRowSet:
    def test_read_is_bounded_to_the_live_stages(self):
        stub = _StubGraph([_consistent_name()])
        result = audit_domains(stub)
        assert result.live_names == 1
        ((_cypher, params),) = stub.calls
        assert set(params["live"]) == {
            "accepted",
            "reviewed",
            "drafted",
            "pending",
        }
        # A supplied client is the caller's to close.
        assert stub.closed is False

    def test_a_clean_name_fails_no_check(self):
        result = audit_domains(_StubGraph([_consistent_name()]))
        assert result.counts() == dict.fromkeys(ALL_CHECKS, 0)

    def test_the_instrument_sees_a_deliberately_broken_row(self):
        # An absence reported by a check is only meaningful if the check can
        # fire at all: this row breaks three of the four and the audit names it.
        broken = _consistent_name("sn:broken")
        broken["physics_domain"] = None
        broken["source_domains"] = []
        broken["edge_domains"] = ["magnetohydrodynamics"]
        result = audit_domains(_StubGraph([broken]))
        assert result.live_names == 1
        assert CHECK_PHYSICS_DOMAIN_SET in result.counts()
        assert result.counts()[CHECK_PHYSICS_DOMAIN_SET] == 1
        assert result.counts()[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS] == 1
        assert set(result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names) == {
            "sn:broken"
        }


class TestPhysicsDomainSet:
    """Check 1: ``physics_domain`` is set."""

    def test_passes_when_set(self):
        result = audit_domains(_StubGraph([_consistent_name()]))
        assert result.checks[CHECK_PHYSICS_DOMAIN_SET].names == ()

    def test_fails_when_unset_for_a_name_with_no_expected_domains(self):
        # No DD source and no child: the derivation's answer is None, so an
        # unset scalar is the only thing wrong here and this check alone fires.
        row = _row("sn:bare", physics_domain=None, source_domains=[])
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_PHYSICS_DOMAIN_SET].names == ("sn:bare",)
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ()
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()


class TestPhysicsDomainDerived:
    """Check 2: ``physics_domain`` equals the selected expected domain."""

    def test_passes_when_it_matches_the_derivation(self):
        result = audit_domains(_StubGraph([_consistent_name()]))
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ()

    def test_fails_when_it_is_a_different_member_of_the_expected_set(self):
        row = _consistent_name("sn:wrong-primary")
        row["physics_domain"] = "transport"  # equilibrium outranks transport
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == (
            "sn:wrong-primary",
        )
        assert result.checks[CHECK_PHYSICS_DOMAIN_SET].names == ()
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()


class TestSourceDomains:
    """Check 3: ``source_domains`` equals the expected set."""

    def test_passes_when_it_matches(self):
        result = audit_domains(_StubGraph([_consistent_name()]))
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()

    def test_fails_when_it_drops_a_source_domain(self):
        row = _consistent_name("sn:short-sources")
        row["source_domains"] = ["equilibrium"]
        row["edge_domains"] = ["equilibrium"]
        row["edge_count"] = 1
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == (
            "sn:short-sources",
        )
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ()
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()

    def test_order_and_duplicates_carry_no_meaning(self):
        # Same set, stored in a different order with a duplicate: passes.
        row = _row(
            "sn:reordered",
            physics_domain="equilibrium",
            source_domains=["transport", "equilibrium", "transport"],
            dd_paths=["equilibrium/time_slice/profiles_1d/psi"],
            dd_domains=["transport", "equilibrium"],
            edge_domains=["transport", "equilibrium"],
        )
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()


class TestDomainEdges:
    """Check 4: ``HAS_PHYSICS_DOMAIN`` edges are exactly ``source_domains``."""

    def test_passes_when_edges_match(self):
        result = audit_domains(_StubGraph([_consistent_name()]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()

    def test_fails_when_an_edge_is_missing(self):
        row = _consistent_name("sn:edge-gap")
        row["edge_domains"] = ["equilibrium"]
        row["edge_count"] = 1
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == (
            "sn:edge-gap",
        )
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()

    def test_fails_when_an_edge_is_extra(self):
        row = _consistent_name("sn:edge-extra")
        row["edge_domains"] = ["equilibrium", "transport", "general"]
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == (
            "sn:edge-extra",
        )

    def test_absent_edges_are_an_empty_set_not_a_placeholder(self):
        # A name with no edge and nothing claimed: absence matches absence, so
        # this check stays quiet and the row's edge set is empty rather than
        # carrying a stand-in for a target the graph could not name.
        row = _row("sn:no-edges", physics_domain=None, source_domains=[])
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == ()
        state = result.state_for("sn:no-edges")
        assert state is not None
        assert state.edge_domains == frozenset()
        assert state.unnamed_edge_count == 0

    def test_fails_when_no_edge_carries_a_claimed_domain(self):
        row = _consistent_name("sn:edge-none")
        row["edge_domains"] = []
        row["edge_count"] = 0
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == (
            "sn:edge-none",
        )

    def test_fails_when_an_edge_names_no_domain(self):
        # One edge, no id on its target: the derivation cannot be checked
        # against it, so it is a finding even though the rest of the row agrees.
        row = _consistent_name("sn:unnamed-edge")
        row["edge_domains"] = []
        row["edge_count"] = 1
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_DOMAIN_EDGES_SOURCE_DOMAINS].names == (
            "sn:unnamed-edge",
        )
        state = result.state_for("sn:unnamed-edge")
        assert state is not None
        assert state.unnamed_edge_count == 1


class TestDerivedParent:
    """A name with no DD source takes its live children's union."""

    def _parent_and_child(
        self,
        *,
        parent_physics_domain: str = "equilibrium",
        parent_source_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        parent = _row(
            "sn:parent",
            physics_domain=parent_physics_domain,
            source_domains=parent_source_domains
            if parent_source_domains is not None
            else ["equilibrium", "transport"],
            dd_paths=[],
            child_ids=["sn:child"],
            edge_domains=["equilibrium", "transport"],
            decision_choice=parent_physics_domain,
        )
        child = _row(
            "sn:child",
            physics_domain="equilibrium",
            decision_choice="equilibrium",
            source_domains=["equilibrium", "transport"],
            dd_paths=["equilibrium/time_slice/profiles_1d/psi"],
            dd_domains=["equilibrium", "transport"],
            edge_domains=["equilibrium", "transport"],
        )
        return [parent, child]

    def test_parent_expected_set_is_its_childrens_union(self):
        result = audit_domains(_StubGraph(self._parent_and_child()))
        parent = result.state_for("sn:parent")
        assert parent is not None
        assert parent.is_derived_parent is True
        assert parent.dd_source_count == 0
        assert parent.expected == frozenset({"equilibrium", "transport"})
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ()
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()

    def test_parent_fails_when_its_scalar_disagrees_with_its_decision(self):
        # The scalar is not derived from the child union any more; it is checked
        # against the decision stored on the name.
        rows = self._parent_and_child(parent_physics_domain="equilibrium")
        rows[0]["physics_domain"] = "transport"
        result = audit_domains(_StubGraph(rows))
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ("sn:parent",)
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ()

    def test_a_recorded_correction_stands_above_the_stored_decision(self):
        row = _consistent_name("sn:corrected")
        row["physics_domain"] = "transport"
        row["decision_choice"] = "equilibrium"
        row["correction"] = "transport"
        row["source_domains"] = ["equilibrium", "transport"]
        result = audit_domains(_StubGraph([row]))
        assert result.checks[CHECK_PHYSICS_DOMAIN_DECISION].names == ()

    def test_an_undecided_name_is_reported_with_its_reason(self):
        row = _consistent_name("sn:undecided")
        row["decision_choice"] = None
        row["correction"] = None
        result = audit_domains(_StubGraph([row]))
        assert [state.sn_id for state in result.undecided] == ["sn:undecided"]
        assert "no stored decision" in result.undecided_reasons()["sn:undecided"]
        assert result.undecided_share == 1.0

    def test_parent_fails_when_its_sources_drop_a_child_domain(self):
        rows = self._parent_and_child(parent_source_domains=["equilibrium"])
        result = audit_domains(_StubGraph(rows))
        assert result.checks[CHECK_SOURCE_DOMAINS_EXPECTED].names == ("sn:parent",)


class TestReportSurface:
    def test_counts_and_ids_are_exposed(self):
        result = audit_domains(
            _StubGraph([_consistent_name(), _consistent_name("sn:second")])
        )
        payload = result.as_dict()
        assert payload["live_names"] == 2
        assert payload["counts"][CHECK_PHYSICS_DOMAIN_SET] == 0
        assert payload["checks"][CHECK_PHYSICS_DOMAIN_SET]["names"] == []

    def test_largest_source_sets_ranks_by_domain_count(self):
        narrow = _consistent_name("sn:narrow")
        wide = _consistent_name("sn:wide")
        wide["dd_domains"] = ["general", "transport", "equilibrium"]
        wide["source_domains"] = ["general", "transport", "equilibrium"]
        wide["edge_domains"] = ["general", "transport", "equilibrium"]
        result = audit_domains(_StubGraph([narrow, wide]))
        assert [state.sn_id for state in result.largest_source_sets(1)] == ["sn:wide"]
