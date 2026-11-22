"""A generated description never lands where the sources' text is stored.

The controls feed's operator wording is a signal's source description; an
enrichment composes its own text from context. Both are wanted, and the defect
this file guards against is the one where writing the second erases the first:
the writer used to set ``description`` and the feed label was gone, with
``description_source`` left behind, so nothing could tell the sources had ever
said anything.

The graph fake here does not assume which properties are written. It reads the
statement the writer actually emitted and applies the assignments it finds, so
a writer that goes back to ``s.description = item.description`` overwrites the
feed label and these tests fail — the separation is exercised through the
Cypher rather than restated in the test.
"""

from __future__ import annotations

import hashlib
import re

from iter_codex.discovery.signals import parallel

FEED_SOURCE = "controlsmetadata:/variable_hmi"
FEED_DESCRIPTION = "Fixture feed label for the demo current input"


def _embedding_of(text: str) -> list[float]:
    return [float(len(text)), float(sum(map(ord, text)) % 997)]


def _fake_embed(items, text_field="embed_text", embedding_field="embedding"):
    for item in items:
        item[embedding_field] = _embedding_of(item[text_field])
    return items


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_ASSIGNMENT = re.compile(
    r"s\.([A-Za-z_][A-Za-z0-9_]*) = item\.([A-Za-z_][A-Za-z0-9_]*)"
)


class _AssigningGraph:
    """Applies the property assignments a writer statement actually carries."""

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        self.statements.append(cypher)
        for item in params["items"]:
            node = self.nodes[item["id"]]
            for prop, key in _ASSIGNMENT.findall(cypher):
                node[prop] = item.get(key)
            if "generated_description_source = coalesce(" in cypher:
                node["generated_description_source"] = (
                    item.get("generated_description_source")
                    or item.get("enrichment_source")
                    or "enrichment"
                )
            if "generated_description_updated_at = datetime()" in cypher:
                node["generated_description_updated_at"] = "now"
            if "embedded_at = datetime()" in cypher:
                node["embedded_at"] = "now"
        return [{"updated": len(params["items"])}]


def _feed_described_signal() -> dict:
    return {
        "id": "facility:general/codac_uda/DEMO:IP",
        "description": FEED_DESCRIPTION,
        "description_source": FEED_SOURCE,
        "description_source_updated_at": "2026-01-02 03:04:05",
        "embedding": _embedding_of(FEED_DESCRIPTION),
        "embedded_text_hash": _text_hash(FEED_DESCRIPTION),
        "embedded_at": "earlier",
    }


def test_generated_description_leaves_the_feed_description_alone(monkeypatch):
    """The source text, its marker and the mechanism are all held separately."""
    nodes = {"facility:general/codac_uda/DEMO:IP": _feed_described_signal()}
    monkeypatch.setattr(parallel, "GraphClient", lambda: _AssigningGraph(nodes))
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)

    composed = "The plasma current, measured by the central solenoid's Rogowski coil."
    parallel.write_signal_descriptions(
        [
            {
                "id": "facility:general/codac_uda/DEMO:IP",
                "description": composed,
                "enrichment_source": "group_composition",
            }
        ]
    )

    node = nodes["facility:general/codac_uda/DEMO:IP"]
    assert node["description"] == FEED_DESCRIPTION
    assert node["description_source"] == FEED_SOURCE
    assert node["description_source_updated_at"] == "2026-01-02 03:04:05"
    assert node["generated_description"] == composed
    assert node["generated_description_source"] == "group_composition"
    assert node["generated_description_updated_at"] == "now"
    assert node["embedded_text_hash"] == _text_hash(composed)
    assert node["embedding"] == _embedding_of(composed)


def test_source_import_is_the_only_path_that_writes_the_source_slot(monkeypatch):
    """A source import writes its own text and marker through the explicit slot."""
    nodes = {"facility:general/codac_uda/DEMO:IP": _feed_described_signal()}
    monkeypatch.setattr(parallel, "GraphClient", lambda: _AssigningGraph(nodes))
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)

    parallel.write_signal_descriptions(
        [
            {
                "id": "facility:general/codac_uda/DEMO:IP",
                "description": "Fixture demo label",
                "description_source": FEED_SOURCE,
                "description_source_updated_at": "2026-02-03 04:05:06",
            }
        ],
        text_property="description",
    )

    node = nodes["facility:general/codac_uda/DEMO:IP"]
    assert node["description"] == "Fixture demo label"
    assert node["description_source"] == FEED_SOURCE
    assert node["description_source_updated_at"] == "2026-02-03 04:05:06"
    assert "generated_description" not in node


def test_embed_text_prefers_the_generated_description():
    """The indexed text is the generated text where a signal has one."""
    assert (
        parallel.signal_embed_text(
            {"description": FEED_DESCRIPTION, "generated_description": "Composed"}
        )
        == "Composed"
    )
    assert parallel.signal_embed_text({"description": FEED_DESCRIPTION}) == (
        FEED_DESCRIPTION
    )
    assert parallel.signal_embed_text({}) == ""


def test_embed_only_rewrites_no_description_property(monkeypatch):
    """A re-embed sweep touches the vector fields and nothing else."""
    nodes = {"facility:general/codac_uda/DEMO:IP": _feed_described_signal()}
    monkeypatch.setattr(parallel, "GraphClient", lambda: _AssigningGraph(nodes))
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)

    parallel.write_signal_descriptions(
        [
            {
                "id": "facility:general/codac_uda/DEMO:IP",
                "description": FEED_DESCRIPTION,
            }
        ],
        embed_only=True,
    )

    node = nodes["facility:general/codac_uda/DEMO:IP"]
    assert node["description"] == FEED_DESCRIPTION
    assert "generated_description" not in node
    assert node["embedded_text_hash"] == _text_hash(FEED_DESCRIPTION)
