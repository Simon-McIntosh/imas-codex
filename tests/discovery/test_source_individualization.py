"""A grouped source's members keep their own description, not the representative's.

``propagate_source_enrichment`` copies a representative signal's description to
every follower in its ``SignalSource``. ``individualize_source_descriptions``
then replaces the copy with one description per member, built from each
member's own accessor identifier and ``SignalNode`` description. The
individualization pass selects only sources whose ``members_described`` flag is
unset, so once a source is individualized it is skipped — and a later
re-enrichment of its representative re-copies the representative's description
back onto the followers through propagation, leaving them as copies that nothing
repairs. The live graph showed twenty JT-60SA sources marked
``members_described`` whose members all carried the same propagated
description.

Propagation therefore clears ``members_described`` as it writes the copy, which
re-arms the individualization pass. These tests drive the real propagation and
individualization functions over a small stateful graph so both halves are
exercised together: the two-member test asserts the members end with distinct
descriptions, and it fails if propagation stops clearing the flag.
"""

from __future__ import annotations

import asyncio

from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalSourceCodeUnwind,
    SignalSourceCodeUnwindBatch,
)

SOURCE_ID = "facility:eddbreadTime('ENNN', 'MDAC', 'magPbTCNNN', t1, t2)"
REP_ID = "facility:mdac_magpbtc10"
DESC_PATTERN = "Field from probe TC-{member_id}. {node_description}"
NAME_PATTERN = "Probe TC-{member_id}"


class _Graph:
    """A stateful model of one SignalSource and its member signals.

    ``query`` recognises the statements ``propagate_source_enrichment`` and
    ``individualize_source_descriptions`` issue, mutating the model the way the
    Cypher would, so the pair can be run without a database.
    """

    def __init__(self, source: dict, members: list[dict]):
        self.source = source
        self.members = members

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "RETURN count(s) AS cnt" in cypher:
            cnt = sum(
                1
                for m in self.members
                if m["id"] != params["rep_id"] and m["status"] == "discovered"
            )
            return [{"cnt": cnt}]

        if "SET sg.status = 'enriched'," in cypher:
            self.source["status"] = "enriched"
            self.source["description"] = params["description"]
            # The regression: without this line the flag stays true and the
            # individualization pass skips the source below.
            if "sg.members_described = false" in cypher:
                self.source["members_described"] = False
            return []

        if "SET s.status = $enriched" in cypher:
            for m in self.members:
                if m["id"] != params["rep_id"] and m["status"] == "discovered":
                    m["status"] = "enriched"
                    m["description"] = params["description"]
                    m["name"] = params["name"]
                    m["enrichment_source"] = "signal_source_propagation"
            return [{"updated": len(self.members) - 1}]

        if "MERGE (d:Diagnostic" in cypher:
            return []

        if "RETURN sg.id AS source_id" in cypher:
            described = self.source.get("members_described")
            if (
                self.source["status"] == "enriched"
                and (described is None or described is False)
                and len(self.members) > 1
            ):
                return [
                    {
                        "source_id": self.source["id"],
                        "group_key": self.source["group_key"],
                        "description": self.source["description"],
                        "representative_id": self.source["representative_id"],
                        "members": [
                            {"id": m["id"], "accessor": m["accessor"]}
                            for m in self.members
                        ],
                    }
                ]
            return []

        if "UNWIND $items AS item" in cypher:
            by_id = {m["id"]: m for m in self.members}
            for item in params["items"]:
                by_id[item["id"]]["name"] = item["name"]
                by_id[item["id"]]["description"] = item["description"]
                by_id[item["id"]]["enrichment_source"] = "individualized"
            return []

        if "SET sg.members_described = true" in cypher:
            self.source["members_described"] = True
            return []

        raise AssertionError(f"unexpected query: {cypher[:80]}")


def _source(members_described=True):
    return {
        "id": SOURCE_ID,
        "group_key": SOURCE_ID.split(":", 1)[1],
        "status": "enriched",
        "description": "Representative probe description.",
        "representative_id": REP_ID,
        "members_described": members_described,
    }


def _members(start, n):
    return [
        {
            "id": f"facility:mdac_magpbtc{i}",
            "accessor": f"eddbreadTime('E101173', 'MDAC', 'magPbTC{i}', t1, t2)",
            "description": "Representative probe description.",
            "name": None,
            "status": "discovered",
            "enrichment_source": "direct",
            "node_description": f"Magnetic probe TC-{i} at R=1.{i} m",
        }
        for i in range(start, start + n)
    ]


def _wire(monkeypatch, graph):
    monkeypatch.setattr(parallel, "GraphClient", lambda: graph)
    monkeypatch.setattr(
        parallel,
        "fetch_source_member_node_descriptions",
        lambda source_id, limit=5: [
            {"accessor": m["accessor"], "node_description": m["node_description"]}
            for m in graph.members[:limit]
        ],
    )
    monkeypatch.setattr(
        parallel,
        "fetch_all_member_node_descriptions",
        lambda source_id: {m["id"]: m["node_description"] for m in graph.members},
    )

    async def _llm(*args, **kwargs):
        batch = SignalSourceCodeUnwindBatch(
            results=[
                SignalSourceCodeUnwind(
                    source_index=1,
                    name_template=NAME_PATTERN,
                    description_template=DESC_PATTERN,
                    variation_field="probe number",
                )
            ]
        )
        return batch, 0.0, 0

    import imas_codex.discovery.base.llm as llm

    monkeypatch.setattr(llm, "acall_llm_structured", _llm)


def test_repropagated_group_members_are_reindividualized(monkeypatch):
    """Two members with different node descriptions end with different ones.

    The source starts already individualized (``members_described=True``) with
    both members holding the representative's copy, which is the state a
    re-enrichment of the representative leaves behind. Propagation must clear
    the flag so individualization runs again and gives each member its own
    description.
    """
    graph = _Graph(_source(members_described=True), _members(10, 2))
    graph.members[0]["id"] = REP_ID
    graph.members[0]["node_description"] = "Magnetic probe TC-10 at R=1.10 m"
    _wire(monkeypatch, graph)

    parallel.propagate_source_enrichment(
        REP_ID,
        {"description": "Representative probe description.", "name": "Rep"},
        batch_cost=0.0,
    )
    assert graph.source["members_described"] is False

    asyncio.run(parallel.individualize_source_descriptions("facility"))

    descriptions = [m["description"] for m in graph.members]
    assert len(set(descriptions)) == 2
    assert "TC-10" in descriptions[0]
    assert "TC-11" in descriptions[1]


def test_singleton_source_is_left_untouched(monkeypatch):
    """A one-member source is excluded by the size gate and keeps its row, not a
    description the pass would compute."""
    graph = _Graph(_source(members_described=False), _members(10, 1))
    graph.members[0]["id"] = REP_ID
    _wire(monkeypatch, graph)

    result = asyncio.run(parallel.individualize_source_descriptions("facility"))

    assert result == 0
    assert graph.members[0]["description"] == "Representative probe description."
    assert graph.source.get("members_described") is False