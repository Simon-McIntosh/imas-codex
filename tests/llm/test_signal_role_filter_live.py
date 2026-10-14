"""Live check that a ``role=measured`` search keeps commands and references out.

Every signal carries a stored ``signal_role`` judged at enrichment. A signal
search asks for a quantity, and a ``role=measured`` filter should return only
signals that measure it, so a commanded, reference or setpoint value never
reaches the answer. This module runs that filter over the graded questions in
``tests/llm/fixtures/signal_search_questions.json`` against the live graph and
reads its own serving path: ``_search_signals`` with the rerank off, which is
what the ``search_signals`` tool calls.

Each question is searched twice: once with ``role="measured"`` and once with no
role, and the top :data:`MEASUREMENT_REACH_DEPTH` of each is read. The graded
fixture states, per question, whether a candidate is a measurement of the
quantity (grade 3), a commanded or reference value of it or the same quantity at
an adjacent point (grade 2), related hardware (grade 1), or read and judged
irrelevant (grade 0). The four properties this module checks:

* every signal a filtered call returns carries the stored role ``measured``,
  read over the filtered top :data:`TOP_N`;
* no filtered top :data:`TOP_N` holds a candidate the fixture grades 2 whose
  stored role is ``commanded`` or ``reference_or_setpoint``;
* on at least one question the unfiltered top :data:`TOP_N` does hold such a
  grade-2 commanded or reference candidate, so the filter is shown to remove
  something the unfiltered search keeps;
* every question's filtered top :data:`MEASUREMENT_REACH_DEPTH` holds at least
  one candidate the fixture grades 3, so the filter reaches the measurement it
  should rather than returning nothing on most questions. This reach is read
  deeper than the three checks above: it asks whether the filter finds the
  measurement at all, not whether a command crowds it out of the served page.

Marked ``graph``: it needs the live database and the production embedding model
the stored signal vectors were built with.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "signal_search_questions.json"

# The depth the served page is read to. The three control checks read the top
# ``TOP_N``; the reach check reads deeper, to ``MEASUREMENT_REACH_DEPTH``, since
# it asks whether the filter finds the measurement at all rather than whether a
# command crowds it out of the page a user sees.
TOP_N = 5
MEASUREMENT_REACH_DEPTH = 10

_FIXTURE = json.loads(FIXTURE_PATH.read_text())
FACILITY = _FIXTURE["eligibility"]["facility"]

# The stored roles that mark a value as commanded, a reference or a setpoint
# rather than a measurement of the queried quantity.
COMMAND_OR_REFERENCE = frozenset({"commanded", "reference_or_setpoint"})


@pytest.fixture(autouse=True)
def _use_production_embeddings(monkeypatch):
    """Embed with the production model, not the suite's local test model.

    The stored signal vectors in the graph come from the production embedding
    model, so a locally embedded query is dimensionally incompatible with the
    signal description index and every vector search fails before it ranks
    anything. Deleting the two overrides lets ``EncoderConfig`` resolve the
    production backend.
    """
    monkeypatch.delenv("IMAS_CODEX_EMBEDDING_LOCATION", raising=False)
    monkeypatch.delenv("IMAS_CODEX_EMBEDDING_MODEL", raising=False)


def _served_top(
    query: str, role: str | None, k: int = MEASUREMENT_REACH_DEPTH
) -> list[dict]:
    """Call the served signal search with the rerank off; return its ranked rows.

    The served tool body is ``_search_signals``, which the ``search_signals``
    tool wraps; it returns the formatted report. Its ranked, enriched rows are
    captured by wrapping the report formatter, so the order and each row's
    stored ``signal_role`` are read from the call the tool actually made rather
    than re-derived. An error reply is refused here rather than scored, so a
    serving failure cannot read as an empty result.

    Reads ``k`` rows, by default the deeper :data:`MEASUREMENT_REACH_DEPTH`, so
    one call serves both the control checks (which slice to :data:`TOP_N`) and
    the reach check.
    """
    from imas_codex.llm import search_tools

    captured: dict[str, list[dict]] = {}
    real = search_tools.format_signals_report

    def spy(signals, data_nodes, scores, **kwargs):
        captured["rows"] = list(signals)
        return real(signals, data_nodes, scores, **kwargs)

    with mock.patch.object(search_tools, "format_signals_report", spy):
        report = search_tools._search_signals(
            query, FACILITY, role=role, k=k, rerank=False
        )

    assert "rows" in captured, f"search did not reach the report formatter: {report!r}"
    assert "Search error" not in report, report
    return captured["rows"]


def _row(signal: dict, question: dict) -> dict:
    """One ranked candidate with its fixture grade and stored role."""
    return {
        "id": signal["id"],
        "name": signal.get("name") or "",
        "description": signal.get("description") or "",
        "grade": question["relevant"].get(signal["id"]),
        "stored_role": signal.get("signal_role"),
    }


_TABLE: list[dict] | None = None


def _live_table() -> list[dict]:
    """The 22 questions, each with its filtered and unfiltered top ``MEASUREMENT_REACH_DEPTH``."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE

    questions = _FIXTURE["questions"]
    assert len(questions) == 22, f"expected 22 questions, found {len(questions)}"

    table: list[dict] = []
    for question in questions:
        filtered = _served_top(question["query"], role="measured")
        unfiltered = _served_top(question["query"], role=None)
        table.append(
            {
                "id": question["id"],
                "query": question["query"],
                "intent": question.get("intent", ""),
                "filtered": [_row(s, question) for s in filtered],
                "unfiltered": [_row(s, question) for s in unfiltered],
            }
        )
    _TABLE = table
    return table


@pytest.mark.graph
def test_filtered_call_returns_only_measured_signals():
    """Every signal in a ``role=measured`` call's top :data:`TOP_N` carries the stored role."""
    offenders = [
        (entry["id"], row["id"], row["stored_role"])
        for entry in _live_table()
        for row in entry["filtered"][:TOP_N]
        if row["stored_role"] != "measured"
    ]
    assert not offenders, f"filtered search returned non-measured signals: {offenders}"


@pytest.mark.graph
def test_filtered_top_holds_no_grade2_command_or_reference():
    """No filtered top :data:`TOP_N` holds a graded-2 commanded or reference signal."""
    offenders = [
        (entry["id"], row["id"], row["stored_role"])
        for entry in _live_table()
        for row in entry["filtered"][:TOP_N]
        if row["grade"] == 2 and row["stored_role"] in COMMAND_OR_REFERENCE
    ]
    assert not offenders, (
        f"filtered search kept a graded-2 command or reference: {offenders}"
    )


@pytest.mark.graph
def test_unfiltered_control_holds_a_grade2_command_or_reference():
    """The unfiltered top :data:`TOP_N` keeps a graded-2 command or reference on some question."""
    hits = [
        (entry["id"], row["id"], row["stored_role"])
        for entry in _live_table()
        for row in entry["unfiltered"][:TOP_N]
        if row["grade"] == 2 and row["stored_role"] in COMMAND_OR_REFERENCE
    ]
    assert hits, (
        "no question's unfiltered top "
        f"{TOP_N} holds a graded-2 command or reference, so the filter removes "
        "nothing this run measured"
    )


@pytest.mark.graph
def test_every_question_returns_a_grade3_measurement_in_its_filtered_top():
    """Every question's filtered top :data:`MEASUREMENT_REACH_DEPTH` holds a graded-3 measurement.

    A role predicate that returned no signals would satisfy the two offender
    checks above vacuously, and a single graded-3 hit anywhere across the
    questions would let the rest return none, so the reach is checked per
    question: every question's filtered top :data:`MEASUREMENT_REACH_DEPTH` must
    hold at least one candidate the fixture grades 3, a measurement of the
    queried quantity. Read deeper than the control checks' :data:`TOP_N`: this
    asks whether the filter finds the measurement at all. A question with no
    graded-3 hit at that depth is named here, not exempted.
    """
    table = _live_table()
    empty = [entry["id"] for entry in table if not entry["filtered"]]
    assert not empty, f"filtered search returned no signals for: {empty}"

    missing = [
        entry["id"]
        for entry in table
        if not any(row["grade"] == 3 for row in entry["filtered"])
    ]
    assert not missing, (
        f"{len(missing)} of {len(table)} question(s) return no graded-3 "
        f"measurement in their filtered top {MEASUREMENT_REACH_DEPTH}: {missing}"
    )
