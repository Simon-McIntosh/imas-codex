"""The ancestor read's one rule: both fields, collapsed, deduped, nearest first.

The review and the compose prompt read the structures above a DD leaf through
one query and one statement rule, so both views see the same text: documentation
and description, documentation first, whitespace collapsed, an unstated
structure dropped, a repeated statement kept once, and the nearest first.
"""

from __future__ import annotations

from unittest.mock import patch

import imas_codex.graph.client as graph_client
from imas_codex.standard_names.worker_tasks import worker_review_names
from imas_codex.standard_names.worker_tasks.worker_common import (
    _ANCESTOR_CONTEXT_QUERY,
    ancestor_statements,
)


def _row(path: str, *, documentation: str = "", description: str = "") -> dict:
    return {
        "depth": 1,
        "path": path,
        "documentation": documentation,
        "description": description,
        "enrichment_source": None,
    }


def test_documentation_and_description_are_both_read() -> None:
    """A structure states a qualifier in one field and a label in the other."""
    assert ancestor_statements(
        [_row("a", documentation="Total injected gas", description="Gas")]
    ) == [{"path": "a", "text": "Total injected gas Gas"}]


def test_description_answers_when_documentation_is_empty() -> None:
    """A structure with no documentation is still shown, by its description."""
    assert ancestor_statements([_row("a", description="Gas")]) == [
        {"path": "a", "text": "Gas"}
    ]


def test_statement_whitespace_is_collapsed() -> None:
    """A statement is one line: a review is not shown its layout."""
    assert ancestor_statements(
        [_row("a", documentation="Total injected\n\t gas   since breakdown")]
    ) == [{"path": "a", "text": "Total injected gas since breakdown"}]


def test_structure_stating_nothing_is_left_out() -> None:
    """A structure with no documentation and no description is not a statement."""
    assert (
        ancestor_statements([_row("empty"), _row("blank", documentation="   \n")]) == []
    )


def test_repeated_statement_is_kept_once_at_the_nearer_path() -> None:
    """An ancestor repeating a nearer statement adds nothing to the review."""
    assert ancestor_statements(
        [
            _row("near", documentation="Carbon"),
            _row("far", documentation=" Carbon "),
        ]
    ) == [{"path": "near", "text": "Carbon"}]


def test_limit_caps_the_prompt_read_and_none_keeps_every_statement() -> None:
    """A compose prompt carries the nearest few; a review may read them all."""
    rows = [_row(f"a{index}", documentation=f"Statement {index}") for index in range(6)]
    assert len(ancestor_statements(rows)) == 4
    assert len(ancestor_statements(rows, limit=None)) == 6


class _FakeGraphClient:
    """Answer the ancestor query per path, recording the paths asked for."""

    def __init__(self, rows_by_path: dict[str, list[dict]]) -> None:
        self.rows_by_path = rows_by_path
        self.asked: list[tuple[str, dict]] = []

    def __enter__(self) -> _FakeGraphClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, statement: str, **params):
        self.asked.append((statement, params))
        return iter(self.rows_by_path.get(params.get("path"), []))


def test_review_read_asks_the_owner_query_once_per_path() -> None:
    """The review read is the owner's query, not a second walk of the chain."""
    fake = _FakeGraphClient(
        {
            "leaf/b": [_row("leaf", documentation="Carbon value")],
            "leaf/a": [],
        }
    )
    with patch.object(graph_client, "GraphClient", lambda: fake):
        read = worker_review_names.enclosing_structure_statements(["leaf/b", "leaf/a"])

    assert read == {"leaf/b": [{"path": "leaf", "text": "Carbon value"}]}
    assert [statement for statement, _params in fake.asked] == [
        _ANCESTOR_CONTEXT_QUERY,
        _ANCESTOR_CONTEXT_QUERY,
    ]
    assert [params for _statement, params in fake.asked] == [
        {"path": "leaf/a"},
        {"path": "leaf/b"},
    ]


def test_review_read_keeps_every_statement_the_walk_returns() -> None:
    """The review is not the prompt: it reads the whole bounded lineage."""
    rows = [_row(f"a{index}", documentation=f"Statement {index}") for index in range(6)]
    fake = _FakeGraphClient({"leaf": rows})
    with patch.object(graph_client, "GraphClient", lambda: fake):
        read = worker_review_names.enclosing_structure_statements(["leaf"])

    assert len(read["leaf"]) == 6


def test_review_read_of_no_paths_asks_the_graph_nothing() -> None:
    """An item with no bound DD path costs no round trip."""
    fake = _FakeGraphClient({})
    with patch.object(graph_client, "GraphClient", lambda: fake):
        assert worker_review_names.enclosing_structure_statements([]) == {}
    assert fake.asked == []
