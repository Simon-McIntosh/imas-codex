"""A content-stage file is admitted on a strong facet alone.

The content arm writes four facet Scores beside the composite.  A file whose
composite is below the ingest gate can still carry the machine description or
signal processing a mapper needs, so the ingest claim, the has-work check and
the description choice admit it when its strongest facet reaches the facet
threshold.  These tests drive the Python mirror (:func:`content_admits`), the
rendered Cypher clause, and the ``content`` reset target that puts a
content-stage file back in front of the scorer while keeping its enrichment.

The decision seam, the graph stub and the file builder are the ones
``test_code_relevance`` already establishes, so no test opens a live endpoint;
the autouse guard in ``tests/conftest.py`` refuses a real request.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base.reset import (
    _CODE_ENRICH_FIELDS,
    _CODE_SCORE_FIELDS,
    CODE_RESET_SPECS,
    reset_to_status,
)
from imas_codex.discovery.code.scorer import (
    ADMISSION_FACET_FIELDS,
    RELEVANCE_STAGE_CONTENT,
    RELEVANCE_STAGE_NAME,
    _relevance_item,
    content_admits,
    content_facet_relevance,
    relevance_predicate,
)
from tests.discovery.test_code_relevance import (
    FACILITY,
    _answers,
    _file,
    _IngestStub,
    _run_score,
)

# The thresholds the shared test harness pins in ``_stub_common``: the ingest
# gate at 0.6, the facet gate at 0.8.
INGEST = 0.6
FACET = 0.8


# ---------------------------------------------------------------------------
# The Python mirror
# ---------------------------------------------------------------------------


def test_strong_facet_admits_where_composite_does_not():
    """Composite 0.2 is below 0.6, but a full machine description clears 0.8."""
    answers = _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 3.0, 0.0))

    assert content_facet_relevance(answers) == pytest.approx(1.0)
    assert content_admits(answers, INGEST, FACET) is True


def test_weak_facets_and_composite_reject_together():
    """Neither gate admits a file with a weak composite and weak facets."""
    answers = _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 1.0, 0.0))

    assert content_facet_relevance(answers) < FACET
    assert content_admits(answers, INGEST, FACET) is False


def test_facet_below_gate_does_not_admit_a_below_composite_file():
    """A facet between the two gates is not enough: the gate is the facet one."""
    answers = _answers(0.2, 0.1, 0.1, 0.1, facets=(2.0, 1.0, 0.0, 0.0))

    assert content_facet_relevance(answers) == pytest.approx(0.5)
    assert content_admits(answers, INGEST, FACET) is False


# ---------------------------------------------------------------------------
# The recorder fails closed on an unanswered content question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    (
        "data_access_depth",
        "signal_processing_depth",
        "machine_description_depth",
        "imas_mapping_depth",
        "relevance_grade",
    ),
)
def test_relevance_item_refuses_a_content_answer_missing_a_question(missing):
    """An absent content answer is refused, never recorded as a zero.

    The decision layer refuses a response that omits a question it asked, so
    this guard catches the case where the question set itself never asked:
    building the item would otherwise divide an absent answer into a 0.
    """
    answers = _answers(0.75, 0.4, 0.2, 0.1)
    del answers[missing]

    with pytest.raises(ValueError, match=missing):
        _relevance_item("cf-1", answers, stage="content", model="fake-model", cost=0.0)


def test_relevance_item_records_a_complete_content_answer():
    """The guard passes a full content answer and records the facets."""
    item = _relevance_item(
        "cf-1",
        _answers(0.75, 0.4, 0.2, 0.1),
        stage="content",
        model="fake-model",
        cost=0.0,
    )

    assert item["score_data_access"] == 1.0
    assert item["score_machine_description"] == pytest.approx(0.6667, abs=1e-4)


# ---------------------------------------------------------------------------
# The rendered Cypher clause
# ---------------------------------------------------------------------------


def test_facet_clause_names_every_admission_facet():
    """The Cypher clause reads the same four fields the Python value reads."""
    rendered = relevance_predicate(
        "sf", RELEVANCE_STAGE_CONTENT, "$min_relevance", "$min_facet_relevance"
    )

    assert "sf.relevance_stage = 'content'" in rendered
    assert "sf.score_composite >= $min_relevance" in rendered
    for field in ADMISSION_FACET_FIELDS:
        assert f"sf.{field} >= $min_facet_relevance" in rendered


def test_name_stage_predicate_carries_no_facet_clause():
    """The names arm's relevance is not a facet, so its predicate is unchanged."""
    rendered = relevance_predicate("cf", RELEVANCE_STAGE_NAME)

    assert (
        rendered
        == "cf.relevance_stage = 'name' AND cf.score_composite >= $min_relevance"
    )
    assert "facet" not in rendered


# ---------------------------------------------------------------------------
# The ingest claim and the description choice
# ---------------------------------------------------------------------------


def test_ingest_claim_admits_a_facet_only_file(monkeypatch):
    """A file below the composite gate is claimed when its facet clears."""
    from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

    facet_only = _file(
        "/analysis/src/mgset.f",
        status="scored",
        relevance_stage="content",
        score_composite=0.3,
        score_machine_description=0.9,
    )
    weak = _file(
        "/analysis/src/plot.f",
        status="scored",
        relevance_stage="content",
        score_composite=0.3,
        score_machine_description=0.4,
    )
    monkeypatch.setattr(
        "imas_codex.graph.GraphClient", lambda: _IngestStub([facet_only, weak])
    )

    claimed = _claim_code_files_for_ingestion(
        FACILITY, limit=10, min_relevance=INGEST, min_facet_relevance=FACET
    )

    assert [c["path"] for c in claimed] == ["/analysis/src/mgset.f"]


def test_score_worker_describes_a_facet_only_file(monkeypatch):
    """The description choice follows the mirror, not the composite alone."""
    facet_only = "/analysis/src/mgset.f"
    weak = "/analysis/src/plot.f"
    files = [
        _file(facet_only, preview_text="machine geometry setup"),
        _file(weak, preview_text="plt.plot(x, y)"),
    ]
    answers = {
        # Below the composite gate, strong machine description: admitted.
        facet_only: _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 3.0, 0.0)),
        # Below both gates: scored, never described.
        weak: _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 0.0, 0.0)),
    }
    graph, _, _, description_calls = _run_score(monkeypatch, files, answers)

    assert description_calls == [[facet_only]]
    described = {
        item["id"]: item.get("score_reason")
        for item in graph.items_for("sf.status = 'scored'")
    }
    assert described[facet_only] == "analysis helper"
    assert not described[weak]


# ---------------------------------------------------------------------------
# The content reset target
# ---------------------------------------------------------------------------


class _CapturingGraph:
    """Record the query the reset renders and answer with a reset count."""

    def __init__(self, count: int = 1):
        self.queries: list[str] = []
        self._count = count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        self.queries.append(" ".join(cypher.split()))
        return [{"reset_count": self._count}]


@pytest.fixture
def captured(monkeypatch):
    graph = _CapturingGraph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    return graph


def test_content_target_resets_only_content_stage_files(captured):
    """The reset takes a content-stage skipped file as well as a scored one."""
    reset_to_status(CODE_RESET_SPECS["content"], FACILITY)

    (query,) = captured.queries
    assert "n.relevance_stage = 'content' AND n.status = 'skipped'" in query
    assert "n.status IN $source_statuses" in query


def test_content_reset_keeps_enrichment_and_clears_only_the_score_fields(captured):
    """A reset file stays enriched and loses only what the scorer will rewrite."""
    reset_to_status(CODE_RESET_SPECS["content"], FACILITY)

    (query,) = captured.queries
    for field in _CODE_SCORE_FIELDS:
        assert f"n.{field} = null" in query
    assert "n.skip_reason = null" in query
    for field in _CODE_ENRICH_FIELDS:
        assert f"n.{field} = null" not in query, f"{field} must survive the reset"
    # ``is_enriched`` is what the scoring claim reads, so it must survive.
    assert "n.is_enriched = null" not in query


def test_content_reset_returns_file_to_the_scoring_claim_state(captured):
    """After the reset the file is what ``claim_files_for_scoring`` takes."""
    spec = CODE_RESET_SPECS["content"]

    assert spec.target_status == "triaged"
    # The scoring claim takes ``status='triaged' AND is_enriched=true``; the
    # reset sets exactly that status and never touches the enrichment flag.
    assert "is_enriched" not in spec.clear_fields
    assert spec.post_cypher is None
