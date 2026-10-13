"""The re-embed sweep selects a signal by one predicate and clears it.

A signal's vector only describes its text while ``embedded_text_hash`` matches
the hash of the text the writer embedded. ``reembed_stale_signal_descriptions``
rewrites every row whose stored hash is missing or differs, and
``census_signal_embeddings`` counts exactly those rows, both through
``_SIGNAL_HASH_MISMATCH_WHERE`` — so a row the sweep rewrote is a row the
census had counted.

A null hash is a mismatch: a signal the writer has never stamped must be
selected. The predicate is rendered as Cypher, where ``s.embedded_text_hash <>``
against a null yields null rather than true, so a predicate that drops the
``IS NULL`` disjunct misses every unstamped row. These tests drive the sweep
over a stateful graph and assert a null-hash row is both counted and cleared —
the check that fails against the null-blind predicate.
"""

from __future__ import annotations

import hashlib

import pytest

from imas_codex.discovery.signals import parallel


def _embedding_of(text: str) -> list[float]:
    """Deterministic text-dependent vector, standing in for the encoder."""
    return [float(len(text)), float(sum(map(ord, text)) % 997)]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(items, text_field="embed_text", embedding_field="embedding"):
    """An encoder that answers only for non-empty text, like the real one."""
    for item in items:
        text = item.get(text_field) or ""
        item[embedding_field] = _embedding_of(text) if text.strip() else None
    return items


def _cypher_predicate_matches(cypher: str, stored: str | None, expected: str) -> bool:
    """Evaluate the sweep's rendered predicate with Cypher's null semantics.

    A null compared to a hash with ``<>`` yields null, and a null ``WHERE``
    drops the row — so a null-hash row matches only when the predicate carries
    an explicit ``IS NULL`` disjunct.
    """
    where = cypher.split("WHERE ", 1)[1].split(" RETURN", 1)[0]
    if stored is None:
        return "IS NULL" in where
    return stored != expected


class _SweepGraph:
    """A stateful model of FacilitySignal description and hash fields.

    Recognises the statements the sweep and census issue — the state read, the
    predicate query and the writer's three arms — and mutates the model the way
    the Cypher would, so a signal's stored hash follows whatever the writer set.
    """

    def __init__(self, nodes: dict[str, dict]):
        self.nodes = nodes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "RETURN s.id AS id, s.description AS description" in cypher:
            return [
                {
                    "id": nid,
                    "description": node.get("description"),
                    "embedded_text_hash": node.get("embedded_text_hash"),
                    "facility_id": node.get("facility_id"),
                }
                for nid, node in self.nodes.items()
            ]

        if "UNWIND $rows AS row" in cypher:
            return [
                {"id": row["id"]}
                for row in params["rows"]
                if _cypher_predicate_matches(
                    cypher,
                    self.nodes[row["id"]].get("embedded_text_hash"),
                    row["expected_hash"],
                )
            ]

        if "s.description = item.description" in cypher:
            carries_vector = "s.embedding = item.embedding" in cypher
            carries_hash = "s.embedded_text_hash = item.embedded_text_hash" in cypher
            for item in params["items"]:
                node = self.nodes[item["id"]]
                node["description"] = item["description"]
                if carries_vector:
                    node["embedding"] = item["embedding"]
                    node["embedded_text_hash"] = item["embedded_text_hash"]
                    node["embedded_at"] = "now"
                elif carries_hash:
                    node["description"] = item["description"]
                    node["embedded_text_hash"] = item["embedded_text_hash"]
            return [{"updated": len(params["items"])}]

        raise AssertionError(f"unexpected query: {cypher[:140]}")


def _wire(monkeypatch, graph):
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _fake_embed)


def test_predicate_counts_a_null_hash_row_and_the_sweep_clears_it(monkeypatch):
    """A signal the writer never stamped is selected and then re-embedded.

    The signal carries a description a searcher would find and no stored hash,
    the state every signal is left in before the first re-embed. The sweep must
    count it as a mismatch, rewrite it, and leave its hash equal to the hash of
    its description. A predicate that compares only ``<>`` misses the null row,
    writes nothing, and fails this test.
    """
    graph = _SweepGraph(
        {"tcv:ip": {"description": "Plasma current", "embedded_text_hash": None}}
    )
    _wire(monkeypatch, graph)

    before = parallel.census_signal_embeddings(gc=graph)
    assert before["missing_hash"] == 1, before
    assert before["mismatch_ids"] == ["tcv:ip"], before

    result = parallel.reembed_stale_signal_descriptions(gc=graph)
    assert result == {"population": 1, "mismatches": 1, "written": 1}, result

    node = graph.nodes["tcv:ip"]
    assert node["embedded_text_hash"] == _text_hash("Plasma current")
    assert node["embedding"] == _embedding_of("Plasma current")

    after = parallel.census_signal_embeddings(gc=graph)
    assert after["mismatch"] == 0, after


def test_census_names_a_row_rewritten_after_the_sweep(monkeypatch):
    """A description changed after the sweep leaves its row a named mismatch.

    A pipeline that rewrites a description outside the writer leaves the stored
    hash describing the old text. The census, run with the sweep's own
    predicate, reports that row and names it rather than hiding it in a count.
    """
    graph = _SweepGraph(
        {"tcv:ip": {"description": "Plasma current", "embedded_text_hash": None}}
    )
    _wire(monkeypatch, graph)
    parallel.reembed_stale_signal_descriptions(gc=graph)
    assert parallel.census_signal_embeddings(gc=graph)["mismatch"] == 0

    # A concurrent rewrite: the description moves, the hash does not.
    graph.nodes["tcv:ip"]["description"] = "Plasma current (updated)"

    after = parallel.census_signal_embeddings(gc=graph)
    assert after["differing"] == 1, after
    assert after["mismatch_ids"] == ["tcv:ip"], after


def test_empty_text_signal_records_its_hash_without_a_vector(monkeypatch):
    """A signal with no description records its empty text's hash, no vector.

    A skipped signal carries no text to embed, so the encoder answers nothing
    and no vector will ever arrive. Its empty text's hash is still stored, so
    the row stops reading as a hash merely missing and never re-enters the
    sweep. A writer that leaves the hash untouched by an absent vector fails
    this test.
    """
    graph = _SweepGraph(
        {"jet:skipped": {"description": None, "embedded_text_hash": None}}
    )
    _wire(monkeypatch, graph)

    parallel.reembed_stale_signal_descriptions(gc=graph)

    node = graph.nodes["jet:skipped"]
    assert node["embedded_text_hash"] == _text_hash("")
    assert node.get("embedding") is None
    assert parallel.census_signal_embeddings(gc=graph)["mismatch"] == 0


def test_writer_and_census_share_one_embed_text_derivation(monkeypatch):
    """The writer's stored hash and the census's expected hash come from one function.

    ``signal_embed_text`` is the single derivation of a signal's embed text:
    the description writer takes it as its default, and the census recomputes
    it to decide a stored vector is stale. Pointing it at a composed
    ``name — description (unit)`` derivation moves both the hash the writer
    stores and the hash the census expects. A derivation held privately by
    either side would not move, so the two would disagree and the census would
    report the row the sweep just wrote as a mismatch — this test then fails.
    """

    def composed(row: dict) -> str:
        return (row.get("description") or "") + " (unit)"

    graph = _SweepGraph(
        {"tcv:ip": {"description": "Plasma current", "embedded_text_hash": None}}
    )
    _wire(monkeypatch, graph)
    monkeypatch.setattr(parallel, "signal_embed_text", composed)

    parallel.reembed_stale_signal_descriptions(gc=graph)

    node = graph.nodes["tcv:ip"]
    assert node["embedded_text_hash"] == _text_hash("Plasma current (unit)")
    assert node["embedding"] == _embedding_of("Plasma current (unit)")
    # The census recomputes the same derivation, so the sweep cleared the row.
    assert parallel.census_signal_embeddings(gc=graph)["mismatch"] == 0

    # The writer's own default is the same derivation, not a private copy.
    direct = _SweepGraph(
        {"tcv:ip": {"description": "Plasma current", "embedded_text_hash": None}}
    )
    parallel.write_signal_descriptions(
        [{"id": "tcv:ip", "description": "Plasma current"}], gc=direct
    )
    assert direct.nodes["tcv:ip"]["embedded_text_hash"] == _text_hash(
        "Plasma current (unit)"
    )


@pytest.mark.graph
def test_census_statement_runs_against_the_live_graph():
    """The real read-only census statement runs on the graph and returns counts.

    ``census_signal_embeddings`` opens its own ``GraphClient``, reads every
    ``FacilitySignal`` and runs the hash-mismatch statement over them. Against
    the live graph it must return a population count and a mismatch count whose
    parts add up, with one named id per mismatch. Run with ``-m graph``; the
    default markers exclude it.
    """
    result = parallel.census_signal_embeddings()

    assert isinstance(result["n"], int), result
    assert result["n"] >= 0, result
    assert result["mismatch"] == result["missing_hash"] + result["differing"], result
    assert len(result["mismatch_ids"]) == result["mismatch"], result
