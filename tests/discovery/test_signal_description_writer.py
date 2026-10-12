"""The signal description writer keeps every stored vector with its text.

A description rewrite that leaves the old embedding in place leaves the vector
describing text the node no longer carries, so search ranks a signal by a
description that is gone. ``write_signal_descriptions`` is the one place a
``FacilitySignal`` description is written: it sets ``description``,
``embedding``, ``embedded_at`` and ``embedded_text_hash`` together, and every
description rewrite in the signals package delegates to it.

The individualization pass replaces each group member's copy of the
representative's description with its own. These tests drive that pass over a
small stateful graph and assert the members end holding embeddings of their own
text rather than the representative's — the check that fails against a writer
that writes the description without re-embedding.
"""

from __future__ import annotations

import asyncio
import hashlib

from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalSourceCodeUnwind,
    SignalSourceCodeUnwindBatch,
)

SOURCE_ID = "facility:eddbreadTime('ENNN', 'MDAC', 'magPbTCNNN', t1, t2)"
REP_ID = "facility:mdac_magpbtc10"
DESC_PATTERN = "Field from probe TC-{member_id}. {node_description}"
NAME_PATTERN = "Probe TC-{member_id}"
REP_DESCRIPTION = "Representative probe description."


def _embedding_of(text: str) -> list[float]:
    """Deterministic text-dependent vector, standing in for the encoder."""
    return [float(len(text)), float(sum(map(ord, text)) % 997)]


def _fake_embed(items, text_field="description", embedding_field="embedding"):
    for item in items:
        item[embedding_field] = _embedding_of(item[text_field])
    return items


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Graph:
    """A stateful model of one individualizable SignalSource and its members.

    ``query`` recognises the statements the individualization pass issues — the
    source selection, the description write and the ``members_described``
    update — and mutates the model the way the Cypher would, so a member's
    stored ``embedding`` and ``embedded_text_hash`` follow whatever the writer
    actually set.
    """

    def __init__(self, source: dict, members: list[dict]):
        self.source = source
        self.members = members

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
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

        if "embedded_text_hash" in cypher:
            by_id = {m["id"]: m for m in self.members}
            for item in params["items"]:
                node = by_id[item["id"]]
                node["description"] = item["description"]
                node["embedding"] = item.get("embedding")
                node["embedded_text_hash"] = item["embedded_text_hash"]
                node["name"] = item["name"]
                node["enrichment_source"] = item["enrichment_source"]
            return [{"updated": len(params["items"])}]

        if "SET sg.members_described = true" in cypher:
            self.source["members_described"] = True
            return []

        raise AssertionError(f"unexpected query: {cypher[:120]}")


def _source(members_described=False):
    return {
        "id": SOURCE_ID,
        "group_key": SOURCE_ID.split(":", 1)[1],
        "status": "enriched",
        "description": REP_DESCRIPTION,
        "representative_id": REP_ID,
        "members_described": members_described,
    }


def _members(start, n):
    return [
        {
            "id": f"facility:mdac_magpbtc{i}",
            "accessor": f"eddbreadTime('E101173', 'MDAC', 'magPbTC{i}', t1, t2)",
            # Every member starts holding the representative's copy and its
            # vector, the state propagation leaves behind.
            "description": REP_DESCRIPTION,
            "embedding": _embedding_of(REP_DESCRIPTION),
            "embedded_text_hash": _text_hash(REP_DESCRIPTION),
            "name": None,
            "status": "discovered",
            "enrichment_source": "signal_source_propagation",
            "node_description": f"Magnetic probe TC-{i} at R=1.{i} m",
        }
        for i in range(start, start + n)
    ]


def _wire(monkeypatch, graph):
    monkeypatch.setattr(parallel, "GraphClient", lambda: graph)
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)
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


def test_individualization_write_reembeds_each_member(monkeypatch):
    """An individualization write leaves each member holding its own vector.

    The members start with the representative's description and its embedding.
    After individualization each member carries a distinct description, so the
    stored ``embedded_text_hash`` and ``embedding`` must follow that description
    rather than the copied one. A writer that sets the description without
    re-embedding leaves the old vector and the old hash, and this test fails.
    """
    graph = _Graph(_source(members_described=False), _members(10, 2))
    graph.members[0]["id"] = REP_ID
    graph.members[0]["node_description"] = "Magnetic probe TC-10 at R=1.10 m"
    _wire(monkeypatch, graph)

    asyncio.run(parallel.individualize_source_descriptions("facility"))

    descriptions = [m["description"] for m in graph.members]
    assert len(set(descriptions)) == 2, descriptions
    for member in graph.members:
        assert member["embedding"] == _embedding_of(member["description"]), (
            f"{member['id']} kept a stale embedding for {member['description']!r}"
        )
        assert member["embedded_text_hash"] == _text_hash(member["description"])


def test_writer_sets_description_embedding_and_hash_together(monkeypatch):
    """The writer stores the description, its embedding and its hash in one write."""
    written: list[dict] = []

    class _Capture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def query(self, cypher: str, **params):
            written.extend(params["items"])
            return [{"updated": len(params["items"])}]

    monkeypatch.setattr(parallel, "GraphClient", _Capture)
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)

    parallel.write_signal_descriptions(
        [{"id": "tcv:ip", "description": "Plasma current"}]
    )

    assert len(written) == 1
    row = written[0]
    assert row["description"] == "Plasma current"
    assert row["embedding"] == _embedding_of("Plasma current")
    assert row["embedded_text_hash"] == _text_hash("Plasma current")


def test_writer_embeds_the_text_the_caller_supplies(monkeypatch):
    """``embed_text`` selects the embedded text, and the hash follows it."""
    written: list[dict] = []

    class _Capture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def query(self, cypher: str, **params):
            written.extend(params["items"])
            return [{"updated": len(params["items"])}]

    monkeypatch.setattr(parallel, "GraphClient", _Capture)
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)

    parallel.write_signal_descriptions(
        [
            {
                "id": "tcv:ip",
                "description": "Plasma current",
                "name": "Ip",
                "unit": "A",
            }
        ],
        embed_text=lambda item: (
            f"{item['name']} — {item['description']} ({item['unit']})"
        ),
    )

    composed = "Ip — Plasma current (A)"
    assert written[0]["embedding"] == _embedding_of(composed)
    assert written[0]["embedded_text_hash"] == _text_hash(composed)
    assert written[0]["description"] == "Plasma current"
