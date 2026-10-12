"""Reviewer agreement must match the tie-correct Spearman implementation.

The reviewer-agreement table reports a rank correlation between reviewers.
Rank correlation is only invariant to the no-ties shortcut
``1 - 6*sum(d**2)/(n*(n**2-1))`` when no ranks tie; review scores tie
constantly, so the agreement figure must come from the same tie-correct
implementation the rest of the suite uses (``benchmark_roles.spearman_rho``),
not a no-ties approximation. These tests pin the table's value to that
implementation on an input with a tied rank.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from imas_codex.standard_names import benchmark as bm
from imas_codex.standard_names.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    ModelResult,
    ReviewerScores,
)
from imas_codex.standard_names.benchmark_roles import spearman_rho

# A tied reviewer pair: reviewer "A" ties its two lowest scores, and the
# no-ties shortcut diverges from the rank correlation precisely here.
TIED_X = [1.0, 1.0, 2.0, 3.0]
TIED_Y = [1.0, 2.0, 3.0, 4.0]
# The no-ties shortcut applied to the average ranks of this tied input.
SHORTCUT_VALUE = 0.95


def _agreement_table(col_a: list[float], col_b: list[float]) -> str:
    """Render the names reviewer-agreement table for two reviewers."""
    models = [f"m{i}" for i in range(1, len(col_a) + 1)]
    results = [
        ModelResult(
            model=model,
            name_reviewer_results=[
                ReviewerScores(reviewer_model="A", target="names", avg_score=a),
                ReviewerScores(reviewer_model="B", target="names", avg_score=b),
            ],
        )
        for model, a, b in zip(models, col_a, col_b, strict=True)
    ]
    report = BenchmarkReport(
        config=BenchmarkConfig(models=models), results=results, reference_names=[]
    )
    buf = StringIO()
    bm._render_reviewer_agreement(Console(file=buf, width=200), report, "names")
    return buf.getvalue()


def test_tied_input_agreement_matches_rank_correlation():
    rho = spearman_rho(TIED_X, TIED_Y)
    # The two formulas genuinely disagree on this input, so the assertion below
    # has discriminating power.
    assert rho != round(SHORTCUT_VALUE, 4)
    rendered = _agreement_table(TIED_X, TIED_Y)
    assert f"{rho:.3f}" in rendered
    assert f"{SHORTCUT_VALUE:.3f}" not in rendered
