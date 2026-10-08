"""Binding context draws on every searchable source and keeps provenance."""

from unittest.mock import Mock

from imas_codex.ids.slot_context import build_slot_context


def _search_fixture(monkeypatch):
    from imas_codex.ids import slot_context

    hits = {
        "code": [
            {"id": "code-high", "text": "weak code", "source_file": "read.py"},
            {"id": "code-low", "text": "sign is inverted", "source_file": "sign.py"},
        ],
        "wiki": [
            {
                "id": "wiki",
                "text": "positive current",
                "page_id": "page-1",
                "page_title": "Current",
            }
        ],
        "document": [
            {
                "id": "doc",
                "title": "Convention",
                "description": "current direction",
                "url": "https://docs.example/convention",
            }
        ],
        "signal": [
            {
                "id": "signal",
                "name": "Ip",
                "description": "plasma current",
                "node_path": "tree/ip",
            }
        ],
    }
    monkeypatch.setattr(
        slot_context,
        "_vector_search_code_chunks",
        lambda *a: (["code-high", "code-low"], {"code-high": 0.9, "code-low": 0.6}),
    )
    monkeypatch.setattr(slot_context, "_text_search_code_chunks", lambda *a: [])
    monkeypatch.setattr(
        slot_context,
        "_enrich_code_chunks",
        lambda gc, ids: [r for r in hits["code"] if r["id"] in ids],
    )
    monkeypatch.setattr(
        slot_context, "_vector_search_wiki_chunks", lambda *a: (["wiki"], {"wiki": 0.7})
    )
    monkeypatch.setattr(slot_context, "_text_search_wiki_chunks", lambda *a: [])
    monkeypatch.setattr(
        slot_context, "_enrich_wiki_chunks", lambda gc, ids: hits["wiki"]
    )
    monkeypatch.setattr(
        slot_context,
        "_vector_search_documents",
        lambda *a: (hits["document"], {"doc": 0.65}),
    )
    monkeypatch.setattr(slot_context, "_text_search_documents", lambda *a: [])
    monkeypatch.setattr(
        slot_context, "_vector_search_signals", lambda *a: (["signal"], {"signal": 0.8})
    )
    monkeypatch.setattr(slot_context, "_text_search_signals", lambda *a: [])
    monkeypatch.setattr(slot_context, "_enrich_signals", lambda gc, ids: hits["signal"])
    encoder = Mock()
    encoder.embed_texts.return_value = [[0.1, 0.2]]
    return slot_context, encoder


def _context(encoder):
    return build_slot_context(
        "tcv",
        {"id": "source", "description": "plasma current"},
        {"id": "equilibrium/ip", "documentation": "plasma current"},
        "sign",
        gc=Mock(),
        encoder=encoder,
        limit=5,
    )


def test_four_kinds_have_origins_and_shared_judgment_changes_order(monkeypatch):
    slot_context, encoder = _search_fixture(monkeypatch)
    from imas_codex.discovery.base import judgment

    monkeypatch.setattr(judgment, "decisions_key_present", lambda: True)

    async def decide(states, questions, **kwargs):
        assert len(states) == 5
        assert all("sign" in state["query"] for state in states)
        assert questions["relevance_grade"]["type"] == "score"
        scores = [1, 4, 3, 2, 5]
        return [({"relevance_grade": {"score": score}}, 0.0) for score in scores], 0.0

    monkeypatch.setattr(judgment, "decide_batch", decide)
    result = _context(encoder)
    assert [item["id"] for item in result] == [
        "code-low",
        "signal",
        "wiki",
        "doc",
        "code-high",
    ]
    assert {item["kind"] for item in result} == {"code", "wiki", "document", "signal"}
    assert {item["origin"] for item in result} == {
        "read.py",
        "sign.py",
        "page-1",
        "https://docs.example/convention",
        "tree/ip",
    }


def test_empty_kind_is_omitted(monkeypatch):
    slot_context, encoder = _search_fixture(monkeypatch)
    from imas_codex.discovery.base import judgment

    monkeypatch.setattr(slot_context, "_vector_search_documents", lambda *a: ([], {}))
    monkeypatch.setattr(judgment, "decisions_key_present", lambda: False)
    result = _context(encoder)
    assert len(result) == 4
    assert "document" not in {item["kind"] for item in result}


def test_no_key_keeps_retrieval_order(monkeypatch):
    slot_context, encoder = _search_fixture(monkeypatch)
    from imas_codex.discovery.base import judgment

    monkeypatch.setattr(judgment, "decisions_key_present", lambda: False)
    monkeypatch.setattr(
        judgment, "decide_batch", Mock(side_effect=AssertionError("must skip"))
    )
    result = _context(encoder)
    assert [item["id"] for item in result] == [
        "code-high",
        "signal",
        "wiki",
        "doc",
        "code-low",
    ]


def test_lexical_hits_survive_empty_vector_indexes(monkeypatch):
    slot_context, encoder = _search_fixture(monkeypatch)
    from imas_codex.discovery.base import judgment

    monkeypatch.setattr(judgment, "decisions_key_present", lambda: False)
    monkeypatch.setattr(slot_context, "_vector_search_code_chunks", lambda *a: ([], {}))
    monkeypatch.setattr(slot_context, "_vector_search_wiki_chunks", lambda *a: ([], {}))
    monkeypatch.setattr(slot_context, "_vector_search_documents", lambda *a: ([], {}))
    monkeypatch.setattr(slot_context, "_vector_search_signals", lambda *a: ([], {}))
    monkeypatch.setattr(
        slot_context,
        "_text_search_code_chunks",
        lambda *a: [{"id": "code-low", "score": 0.8}],
    )
    monkeypatch.setattr(
        slot_context,
        "_text_search_wiki_chunks",
        lambda *a: [{"id": "wiki", "score": 0.7}],
    )
    monkeypatch.setattr(
        slot_context,
        "_text_search_documents",
        lambda *a: [
            {
                "id": "doc",
                "title": "Convention",
                "description": "current direction",
                "url": "https://docs.example/convention",
                "score": 0.6,
            }
        ],
    )
    monkeypatch.setattr(
        slot_context,
        "_text_search_signals",
        lambda *a: [{"id": "signal", "score": 0.5}],
    )

    result = _context(encoder)
    assert [item["kind"] for item in result] == ["code", "wiki", "document", "signal"]
