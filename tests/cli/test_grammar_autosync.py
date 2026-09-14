"""Regression coverage for content-addressed grammar synchronisation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from imas_codex.cli import sn
from imas_codex.standard_names import grammar_sync


class _GrammarGraph:
    """Minimal graph client that serves one active grammar snapshot."""

    def __init__(self, rows: list[dict[str, str | None]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> _GrammarGraph:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def query(self, cypher: str, **params: object) -> list[dict[str, str | None]]:
        self.queries.append((cypher, params))
        return self.rows


def _write_grammar_inputs(root: Path, vocabulary: str) -> Path:
    """Create a small installed-package-shaped grammar input tree."""
    grammar_dir = root / "grammar"
    vocabulary_dir = grammar_dir / "vocabularies"
    vocabulary_dir.mkdir(parents=True)
    (grammar_dir / "specification.yml").write_text(
        "vocabularies:\n  quantities: !include vocabularies/quantities.yml\n",
        encoding="utf-8",
    )
    vocabulary_path = vocabulary_dir / "quantities.yml"
    vocabulary_path.write_text(vocabulary, encoding="utf-8")
    return vocabulary_path


def _use_grammar_inputs(monkeypatch, root: Path) -> None:
    from imas_standard_names.grammar_codegen import spec as grammar_spec

    monkeypatch.setattr(
        grammar_spec, "_GRAMMAR_SPEC_PATH", root / "grammar/specification.yml"
    )


def test_content_change_triggers_sync_without_version_change(
    tmp_path, monkeypatch
) -> None:
    """A vocabulary edit refreshes the graph even when package metadata is fixed."""
    vocabulary = _write_grammar_inputs(tmp_path, "- temperature\n")
    _use_grammar_inputs(monkeypatch, tmp_path)
    before = grammar_sync.grammar_content_digest()
    vocabulary.write_text("\n# quantity vocabulary\n- temperature\n", encoding="utf-8")
    assert grammar_sync.grammar_content_digest() == before
    vocabulary.write_text("- temperature\n- density\n", encoding="utf-8")
    after = grammar_sync.grammar_content_digest()
    assert after != before

    graph = _GrammarGraph([{"version": "0.9.3", "content_digest": before}])
    sync = Mock()
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    monkeypatch.setattr(grammar_sync, "sync_isn_grammar_to_graph", sync)

    sn._auto_sync_grammar(quiet=True)

    assert sync.call_count == 1
    assert sync.call_args.kwargs == {"gc": graph}


def test_identical_content_keeps_auto_sync_a_no_op(tmp_path, monkeypatch) -> None:
    """An unchanged grammar digest does not rewrite the graph snapshot."""
    _write_grammar_inputs(tmp_path, "- temperature\n")
    _use_grammar_inputs(monkeypatch, tmp_path)
    digest = grammar_sync.grammar_content_digest()
    graph = _GrammarGraph([{"version": "0.9.3", "content_digest": digest}])
    sync = Mock()
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    monkeypatch.setattr(grammar_sync, "sync_isn_grammar_to_graph", sync)

    sn._auto_sync_grammar(quiet=True)

    assert sync.call_count == 0


def test_sync_stores_the_installed_grammar_digest(monkeypatch) -> None:
    """The active graph snapshot records the digest that selected the sync."""
    graph = _GrammarGraph([])
    monkeypatch.setattr(grammar_sync, "grammar_content_digest", lambda: "sha256:known")
    monkeypatch.setattr(
        "imas_standard_names.graph.sync.sync_grammar",
        lambda *_args, **_kwargs: {},
    )

    grammar_sync.sync_isn_grammar_to_graph(gc=graph)

    assert any(
        "SET v.content_digest = $content_digest" in cypher
        and params["content_digest"] == "sha256:known"
        for cypher, params in graph.queries
    )
