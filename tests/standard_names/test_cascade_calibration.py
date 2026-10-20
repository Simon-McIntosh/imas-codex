"""Joint veto selection holds the union of cutoffs inside its false-veto budget.

The defect this guards is a decision rule built from checks chosen one at a
time: each cutoff is fitted to flag at most a tenth of the accepted items, and
the union of several such cutoffs flags far more than any one of them. The
fixture below is built so that every check is individually admissible and the
union of all of them is not, which is the only shape in which joint selection
and per-check selection disagree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imas_codex.standard_names.review import cascade_calibration as cc

# Setting this arm replaces joint selection with the per-check rule the module
# exists to replace. It is how the negative control is run: the budget test must
# then fail, and every other test must still pass.
SELECTION_ARM = "CASCADE_SELECTION_ARM"
PER_CHECK_ARM = "per-check"

ADMISSIBLE_SHARE = 0.09
DEFECT_SHARE = 0.20


def _per_check_select_vetoes(items, axis, *, budget=cc.VETO_BUDGET, hard_veto=False):
    """Select every check that is admissible on its own, ignoring their union."""
    accepted_ids = {
        item["id"] for item in items if item["label"] == cc.ACCEPTABLE_LABEL
    }
    defect_ids = {item["id"] for item in items if item["label"] != cc.ACCEPTABLE_LABEL}
    selected: dict[str, float] = {}
    for check_id in cc._check_ids(items):
        cutoff, _stats = cc._per_check_threshold(
            items, check_id, accepted_ids, defect_ids
        )
        if cutoff is not None:
            selected[check_id] = cutoff
    return selected, {}


@pytest.fixture
def selector(monkeypatch):
    if os.environ.get(SELECTION_ARM) == PER_CHECK_ARM:
        monkeypatch.setattr(cc, "select_vetoes", _per_check_select_vetoes)
    return cc.select_vetoes


def _union_false_veto_rate(selected, items) -> float:
    accepted = [item for item in items if item["label"] == cc.ACCEPTABLE_LABEL]
    flagged = cc._flagged_at(items, selected)
    accepted_ids = {item["id"] for item in accepted}
    return len(flagged & accepted_ids) / len(accepted_ids)


def _disjoint_checks_fixture(
    checks: int = 1, accepted_n: int = 100, defect_n: int = 100
):
    """Items where each check flags a disjoint share of the accepted items.

    Every check flags ``ADMISSIBLE_SHARE`` of the accepted items, inside the
    per-check budget of a tenth, and ``DEFECT_SHARE`` of the defects. The shares
    are disjoint, so the union of ``checks`` of them flags their sum.
    """
    items: list[dict] = []
    accepted_share = round(ADMISSIBLE_SHARE * accepted_n)
    defect_share = round(DEFECT_SHARE * defect_n)
    for index in range(accepted_n):
        items.append(
            {
                "id": f"a{index}",
                "label": cc.ACCEPTABLE_LABEL,
                "check_fail": {f"c{n}": 0.0 for n in range(checks)},
            }
        )
    for index in range(defect_n):
        items.append(
            {
                "id": f"d{index}",
                "label": 0,
                "check_fail": {f"c{n}": 0.0 for n in range(checks)},
            }
        )
    for check_index in range(checks):
        key = f"c{check_index}"
        for index in range(
            check_index * accepted_share, (check_index + 1) * accepted_share
        ):
            items[index]["check_fail"][key] = 0.8
        for index in range(
            check_index * defect_share, (check_index + 1) * defect_share
        ):
            items[accepted_n + index]["check_fail"][key] = 0.8
    return items


def test_union_of_admissible_checks_exceeds_the_budget():
    """The fixture must be able to fail, or the test below proves nothing."""
    items = _disjoint_checks_fixture(checks=4)
    per_check, _ = _per_check_select_vetoes(items, "name")
    assert len(per_check) == 4
    assert _union_false_veto_rate(per_check, items) > cc.VETO_BUDGET


def test_joint_selection_keeps_the_union_inside_its_budget(selector):
    """Add checks by the recall they add, but never past the shared budget."""
    items = _disjoint_checks_fixture(checks=4)
    selected, _diagnostics = selector(items, "name")

    assert selected, "joint selection dropped every admissible check"
    assert _union_false_veto_rate(selected, items) <= cc.VETO_BUDGET
    # One check fits inside a tenth of the accepted items; two cannot, because
    # their flagged shares are disjoint.
    assert len(selected) == 1


def test_joint_selection_is_deterministic(selector):
    """The same receipts must always yield the same cutoffs."""
    items = _disjoint_checks_fixture(checks=4)
    first, _ = selector(items, "name")
    second, _ = selector(items, "name")
    assert first == second

    reversed_items = list(reversed(items))
    third, _ = selector(reversed_items, "name")
    assert third == first


def test_a_missing_receipt_names_the_receipt_it_could_not_read():
    path = Path("/nonexistent/cascade-receipt.json")
    with pytest.raises(cc.ReceiptUnavailable) as excinfo:
        cc._read_json(path)
    assert str(path) in str(excinfo.value)
