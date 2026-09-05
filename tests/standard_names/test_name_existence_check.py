"""The existence verdict must not depend on how much of the catalogue was read.

Answering existence by enumerating every ``StandardName`` under a row limit
and comparing against what came back makes the answer a function of store
order once the catalogue outgrows the limit: a name past the cut reads as
absent, and the nearest surviving neighbour is offered as a suggestion, both
with no indication that the read was partial. These tests pin the two
properties that remove the dependence — existence resolves by direct id
match, and the enumeration retained for nearest-match scoring is ordered and
refuses when it fills.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.search import (
    SuggestionPoolExhausted,
    check_names,
)


class RecordingGraph:
    """Minimal graph stand-in that honours ``LIMIT`` the way the store does.

    The enumeration is served in insertion order rather than sorted order, so
    a test that passes here cannot be relying on the pool happening to contain
    the name it asks about.
    """

    def __init__(self, ids: list[str]):
        self.ids = list(ids)
        self.queries: list[str] = []

    def query(self, cypher: str, **params):
        self.queries.append(cypher)
        if "sn.id IN $ids" in cypher:
            wanted = set(params["ids"])
            return [{"id": i} for i in self.ids if i in wanted]
        cap = params["cap"]
        return [{"id": i, "physical_base": i} for i in self.ids[:cap]]

    @property
    def enumerations(self) -> int:
        return sum(1 for q in self.queries if "sn.id IN $ids" not in q)


CATALOG = [f"quantity_{i:04d}" for i in range(200)]


def test_name_beyond_the_pool_limit_still_exists():
    """The identity a truncated enumeration would drop is reported present."""
    gc = RecordingGraph(CATALOG)
    beyond = CATALOG[-1]

    (result,) = check_names([beyond], gc=gc, suggestion_pool_limit=10)

    assert result["exists"] is True
    assert result["suggestion"] == ""


def test_verdicts_are_unchanged_when_the_pool_limit_collapses():
    """Existence answers are identical at a cap far below the population."""
    gc = RecordingGraph(CATALOG)

    generous = check_names(CATALOG, gc=gc, suggestion_pool_limit=len(CATALOG) * 10)
    starved = check_names(CATALOG, gc=gc, suggestion_pool_limit=2)

    assert [r["exists"] for r in starved] == [r["exists"] for r in generous]
    assert all(r["exists"] for r in starved)


def test_an_all_present_batch_reads_no_pool():
    """Nothing enumerates the catalogue when every name resolves directly."""
    gc = RecordingGraph(CATALOG)

    check_names(CATALOG[:5], gc=gc, suggestion_pool_limit=10)

    assert gc.enumerations == 0


def test_the_pool_refuses_rather_than_suggesting_from_a_slice():
    """A filled pool raises; it cannot know which candidate is nearest."""
    gc = RecordingGraph(CATALOG)

    with pytest.raises(SuggestionPoolExhausted) as excinfo:
        check_names(["quantity_that_is_absent"], gc=gc, suggestion_pool_limit=10)

    assert "10" in str(excinfo.value)


def test_the_pool_is_ordered_so_the_same_catalogue_gives_the_same_pool():
    gc = RecordingGraph(CATALOG)

    check_names(["quantity_that_is_absent"], gc=gc, suggestion_pool_limit=1000)

    (enumeration,) = [q for q in gc.queries if "sn.id IN $ids" not in q]
    assert "ORDER BY sn.id" in enumeration


def test_an_absent_name_still_gets_its_nearest_match():
    gc = RecordingGraph(CATALOG)

    (result,) = check_names(["quantity_0007x"], gc=gc, suggestion_pool_limit=1000)

    assert result["exists"] is False
    assert result["suggestion"] == "quantity_0007"


def test_blank_and_empty_input_are_dropped():
    gc = RecordingGraph(CATALOG)

    assert check_names([], gc=gc) == []
    assert check_names(["", "   "], gc=gc) == []
    assert gc.queries == []


@pytest.mark.graph
def test_every_identity_a_capped_scan_drops_is_reported_present():
    """Measured against the live catalogue, not asserted.

    The old default read 5000 unordered rows; whatever the store leaves out of
    that slice is exactly the population an existence check has most reason to
    be asked about. Every one of them must resolve.
    """
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    try:
        population = gc.query("MATCH (sn:StandardName) RETURN count(sn) AS c")[0]["c"]
        every = [
            r["id"]
            for r in gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
            if r.get("id")
        ]
        kept = {
            r["id"]
            for r in gc.query(
                "MATCH (sn:StandardName) RETURN sn.id AS id LIMIT $cap", cap=5000
            )
            if r.get("id")
        }
        dropped = [i for i in every if i not in kept]
        if population <= 5000:
            pytest.skip("catalogue no longer exceeds the historic row limit")

        assert dropped, "expected the capped scan to drop part of the population"
        verdicts = check_names(dropped, gc=gc)
        absent = [r["name"] for r in verdicts if not r["exists"]]
        assert absent == [], f"{len(absent)} of {len(dropped)} reported absent"
    finally:
        gc.close()


@pytest.mark.graph
def test_live_verdicts_survive_a_pool_limit_far_below_the_population():
    """The counterfactual: starve the pool and the answers do not move."""
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    try:
        sample = [
            r["id"]
            for r in gc.query(
                "MATCH (sn:StandardName) RETURN sn.id AS id ORDER BY sn.id DESC "
                "LIMIT 40"
            )
            if r.get("id")
        ]
        if not sample:
            pytest.skip("no StandardName nodes in the graph")

        generous = check_names(sample, gc=gc)
        starved = check_names(sample, gc=gc, suggestion_pool_limit=5)

        assert [r["exists"] for r in starved] == [r["exists"] for r in generous]
        assert all(r["exists"] for r in starved)
    finally:
        gc.close()
