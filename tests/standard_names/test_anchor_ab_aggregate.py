"""Gate for the anchor-adoption aggregate: the required-field refusal, the
computed verdict, the derived population and the figure's text check.

The aggregate is a standalone runner under ``docs/evidence/fragments`` rather
than an installed module, so it is loaded by path. The runner's records and
selection are located from this test file, not from the runner, so the same
tests hold when the runner is exercised from a mutated copy.

The guarded behaviour is made to happen rather than assumed: a record missing
``scores_after``, ``lint_after``, ``batches_run`` or ``batches_total`` is fed
to the aggregate and the refusal asserted, and a figure carrying a direct label
over plotted data is handed to the text check and the refusal asserted, beside
a clean figure the check must accept.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = (
    REPO
    / "docs/evidence/fragments/sn-selection-over-generation"
    / "ssog-anchor-ab-aggregate-run.py"
)
MODULE_PATH = Path(os.environ.get("SSOG_AGGREGATE_MODULE", DEFAULT_MODULE))
RECORDS_DIR = (
    REPO / "docs/evidence/fragments/sn-selection-over-generation/ssog-anchor-ab"
)
SELECTION = RECORDS_DIR / "selection.json"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("ssog_anchor_ab_aggregate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


agg = _load(MODULE_PATH)


def _record(drop: tuple[str, ...] = (), git_head: str | None = None) -> dict:
    record = {
        "family": "family",
        "arm": "heuristic",
        "anchor": "some_anchor",
        "members": ["m1", "m2"],
        "harmonize_summary": {
            "batches_run": 2,
            "batches_total": 2,
            "halted": False,
            "halt_reasons": [],
        },
        "code": {"git_head": git_head or agg.FIXED_TREE},
        "lint_before": 1,
        "lint_after": 0,
        "scores_before": {"m1": 0.9, "m2": 0.9},
        "scores_after": {"m1": 0.95, "m2": 0.95},
        "stamped": True,
        "outcome_status": "complete",
    }
    for field in drop:
        if field in ("batches_run", "batches_total"):
            record["harmonize_summary"].pop(field, None)
        else:
            record.pop(field, None)
    return record


@pytest.mark.parametrize(
    "field", ["scores_after", "lint_after", "batches_run", "batches_total"]
)
def test_missing_required_field_refused(field: str) -> None:
    """A record missing a required field must refuse, not read as an absent arm."""
    with pytest.raises(agg.AggregateRefusal) as excinfo:
        agg.classify_record(
            _record(drop=(field,)), "family", "heuristic", Path("r.json")
        )
    assert field in str(excinfo.value)


def test_required_field_set_is_declared() -> None:
    """The required field set is the one the brief names, on the record and summary."""
    assert set(agg.REQUIRED_TOP) == {"scores_after", "lint_after"}
    assert set(agg.REQUIRED_SUMMARY) == {"batches_run", "batches_total"}


def test_first_round_record_refused() -> None:
    """A record naming a revision other than the fixed tree is refused by name."""
    with pytest.raises(agg.AggregateRefusal):
        agg.classify_record(
            _record(git_head="deadbeefdeadbeef"), "family", "heuristic", Path("r.json")
        )


def test_complete_record_reads() -> None:
    rec = agg.classify_record(_record(), "family", "heuristic", Path("r.json"))
    assert rec["batches_run"] == 2
    assert rec["lint_after"] == 0


def test_population_derived_from_selection() -> None:
    """The agreements and the compared families are read from selection.json."""
    selection = json.loads(SELECTION.read_text())
    expected_agreements = [
        row["seed"]
        for row in selection["selected"]
        if row["heuristic_anchor"] is not None
        and row["heuristic_anchor"] == row["jev_anchor"]
    ]
    expected_compared = [
        row["seed"]
        for row in selection["selected"]
        if row["heuristic_anchor"] != row["jev_anchor"]
    ]
    payload = agg.build(RECORDS_DIR, SELECTION)
    assert payload["population"]["agreements"] == expected_agreements
    assert payload["population"]["compared"] == expected_compared
    assert len(payload["population"]["compared"]) == 3
    assert len(payload["population"]["agreements"]) == 3 + 3


def test_verdict_computed_from_conditions() -> None:
    """The verdict is the conjunction of the two conditions, not a literal."""
    payload = agg.build(RECORDS_DIR, SELECTION)
    conditions = payload["conditions"]
    assert payload["verdict"]["adopt_jev"] == (
        conditions["fewer_lint_total"]["met"]
        and conditions["no_lower_member_scores"]["met"]
    )


def test_verdict_adopts_when_both_conditions_met(tmp_path: Path) -> None:
    """A synthetic pair meeting both conditions adopts."""
    selection = {
        "adoption_rule": {"registered_at": "t", "text": "rule"},
        "source_export": "x",
        "selected": [
            {
                "seed": "diff",
                "kind": "k",
                "physical_base": "",
                "members": ["m1", "m2"],
                "heuristic_anchor": "h",
                "jev_anchor": "j",
                "jev_cost": 0.0,
            }
        ],
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    for arm, lint, scores in (
        ("heuristic", 3, {"m1": 0.9, "m2": 0.9}),
        ("jev", 1, {"m1": 0.95, "m2": 0.95}),
    ):
        rec = _record()
        rec["family"] = "diff"
        rec["arm"] = arm
        rec["lint_after"] = lint
        rec["scores_after"] = scores
        (tmp_path / f"diff-{arm}.json").write_text(json.dumps(rec))
    payload = agg.build(tmp_path, tmp_path / "selection.json")
    assert payload["conditions"]["fewer_lint_total"]["met"] is True
    assert payload["conditions"]["no_lower_member_scores"]["met"] is True
    assert payload["verdict"]["adopt_jev"] is True


def test_verdict_refuses_when_a_member_is_lower(tmp_path: Path) -> None:
    """A single lower member refuses adoption and is named."""
    selection = {
        "adoption_rule": {"registered_at": "t", "text": "rule"},
        "source_export": "x",
        "selected": [
            {
                "seed": "diff",
                "kind": "k",
                "physical_base": "",
                "members": ["m1", "m2"],
                "heuristic_anchor": "h",
                "jev_anchor": "j",
                "jev_cost": 0.0,
            }
        ],
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    for arm, lint, scores in (
        ("heuristic", 3, {"m1": 0.9, "m2": 0.9}),
        ("jev", 1, {"m1": 0.5, "m2": 0.95}),
    ):
        rec = _record()
        rec["family"] = "diff"
        rec["arm"] = arm
        rec["lint_after"] = lint
        rec["scores_after"] = scores
        (tmp_path / f"diff-{arm}.json").write_text(json.dumps(rec))
    payload = agg.build(tmp_path, tmp_path / "selection.json")
    assert payload["conditions"]["fewer_lint_total"]["met"] is True
    assert payload["conditions"]["no_lower_member_scores"]["met"] is False
    assert payload["verdict"]["adopt_jev"] is False
    assert payload["conditions"]["no_lower_member_scores"]["members_lower_in_jev"] == [
        {"family": "diff", "member": "m1", "heuristic": 0.9, "jev": 0.5}
    ]


def _figure_with_label_over_data(*, over_data: bool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh([0], [5], height=0.5, color="#246c78")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if over_data:
        ax.text(2.5, 0.0, "planted label", ha="center", va="center")
    fig.tight_layout(pad=2.0)
    return plt, fig


def test_figure_check_refuses_a_direct_label_over_data() -> None:
    """A direct label over plotted data is refused."""
    plt, fig = _figure_with_label_over_data(over_data=True)
    try:
        with pytest.raises(SystemExit) as excinfo:
            agg._check_text(fig)
        assert "over plotted data" in str(excinfo.value)
    finally:
        plt.close(fig)


def test_figure_check_accepts_a_clean_figure() -> None:
    """The guard is not refusing everything: a clean figure passes."""
    plt, fig = _figure_with_label_over_data(over_data=False)
    try:
        agg._check_text(fig)
    finally:
        plt.close(fig)
