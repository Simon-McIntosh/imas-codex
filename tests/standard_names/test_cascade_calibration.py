"""Calibration-only rate matching and shared paired resampling."""

import ast
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from imas_codex.discovery.base.evaluation import bootstrap_interval
from imas_codex.standard_names.review import cascade, cascade_calibration as cc


def test_matched_cutoff_and_mapped_score_cross_at_consumer_threshold():
    scores = [0.1, 0.2, 0.3, 0.4]
    fit = cc.matched_cutoff(scores, [0.1, 0.85, 0.7, 0.9])
    assert fit["cutoff"] == 0.3
    assert fit["pre_veto_accepts"] == fit["panel_accepts"] == 2
    cutoff = fit["cutoff"]
    below, at, above = [
        cascade.map_decision_score(s, cutoff)
        for s in (np.nextafter(cutoff, 0), cutoff, np.nextafter(cutoff, 1))
    ]
    assert below < 0.85 == at <= above
    assert [cascade.map_decision_score(s, cutoff) >= 0.85 for s in scores] == [
        False,
        False,
        True,
        True,
    ]


def test_matching_refuses_a_boundary_tie_instead_of_changing_the_target():
    with pytest.raises(ValueError, match="no inclusive cutoff matches 2/3"):
        cc.matched_cutoff([0, 0, 1], [0.9, 0.9, 0.1])


@pytest.mark.parametrize(
    "scores,panel",
    [
        ([], []),
        ([0.2], []),
        ([float("nan")], [1]),
        ([1.1], [1]),
        ([0.2], [float("nan")]),
    ],
)
def test_matching_refuses_unusable_scores(scores, panel):
    with pytest.raises(ValueError):
        cc.matched_cutoff(scores, panel)


def test_shared_bootstrap_accepts_statistics_and_keeps_copy_groups_paired():
    def statistic(y, scores):
        return float(np.mean(scores))

    result = bootstrap_interval(
        [1, 1],
        [1, 0],
        scores_b=[0.2, 0.0],
        copy_groups=["same", "same"],
        statistic=statistic,
        n_resamples=20,
        seed=17,
    )
    assert result["point"] == result["lower"] == result["upper"] == 0.4
    assert result["n_resamples"] == 20
    # The AUC default must still refuse a population with only one class.
    with pytest.raises(ValueError, match="both label classes"):
        bootstrap_interval([1, 1], [1, 0])


def test_undefined_precision_is_not_reported_as_zero():
    def statistic(y, scores):
        return cc._metric(y, scores, "accept_precision")

    missing = bootstrap_interval([1, 0], [0, 0], statistic=statistic, n_resamples=20)
    assert missing["point"] is None and missing["lower"] is None
    assert missing["n_resamples"] == 0
    control = bootstrap_interval([1, 0], [1, 0], statistic=statistic, n_resamples=20)
    assert control["point"] == 1 and control["n_resamples"] == 20


def test_a_missing_receipt_names_the_receipt_it_could_not_read():
    path = Path("/nonexistent/cascade-receipt.json")
    with pytest.raises(cc.ReceiptUnavailable) as excinfo:
        cc._read_json(path)
    assert str(path) in str(excinfo.value)


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "imas_codex/standard_names/review/cascade_calibration.py"
)
_CALLER_ROOTS = ("imas_codex", "tests")


def _definitions(source):
    """Module-level names: functions, classes and assigned constants."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
    return out


def _loaded_names(node):
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _reads_of_calibration_module(tree):
    """Names a parsed file reads from the calibration module."""
    aliases, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            from_module = bool(node.module) and node.module.split(".")[-1] == (
                "cascade_calibration"
            )
            for alias in node.names:
                if alias.name == "cascade_calibration":
                    aliases.add(alias.asname or alias.name)
                elif from_module:
                    imported.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == "cascade_calibration":
                    aliases.add(alias.asname or alias.name.split(".")[0])
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }
    loaded = {
        item.id
        for item in ast.walk(tree)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }
    return attributes | (loaded & imported)


def _parse(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text())


@lru_cache(maxsize=4)
def _external_references(repo=None):
    """Definition name -> the caller files that read it from outside the module."""
    repo = Path(__file__).resolve().parents[2] if repo is None else Path(repo)
    reads = {}
    for root in _CALLER_ROOTS:
        for path in sorted((repo / root).rglob("*.py")):
            text = path.read_text()
            if "cascade_calibration" not in text:
                continue
            if path.resolve() == _MODULE_PATH.resolve():
                continue
            for name in _reads_of_calibration_module(_parse(path)):
                reads.setdefault(name, set()).add(str(path.relative_to(repo)))
    return reads


def _analysis(source, external):
    """Definitions, the reached subset, and the referrers found for each name."""
    defined = _definitions(source)
    edges, referrers = {}, {}
    for name, paths in external.items():
        if name in defined:
            referrers[name] = {f"imported by {path}" for path in paths}
    reached = set(referrers)
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            edges[node.name] = _loaded_names(node) & set(defined)
            for name in edges[node.name]:
                referrers.setdefault(name, set()).add(node.name)
        else:
            # Module-level code names these but is not a caller of them, so it
            # is recorded for the reader and confers no reach.
            for name in _loaded_names(node) & set(defined):
                referrers.setdefault(name, set()).add("module level")
    frontier = list(reached)
    while frontier:
        for name in edges.get(frontier.pop(), ()):
            if name not in reached:
                reached.add(name)
                frontier.append(name)
    return defined, reached, referrers


def calibration_reachability(source=None, external=None):
    """One row per module-level definition: line, kind and what reaches it."""
    source = _MODULE_PATH.read_text() if source is None else source
    external = _external_references() if external is None else external
    defined, reached, referrers = _analysis(source, external)
    rows = []
    for name, node in sorted(defined.items(), key=lambda item: item[1].lineno):
        kind = "constant"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        rows.append(
            {
                "name": name,
                "line": node.lineno,
                "kind": kind,
                "reached": name in reached,
                "from": sorted(referrers.get(name, ())),
            }
        )
    return rows


def test_every_module_definition_is_reached_from_code_or_a_test():
    rows = calibration_reachability()
    assert [row["name"] for row in rows if not row["reached"]] == []
    # The instrument must see the module rather than an empty read of it.
    names = {row["name"] for row in rows}
    assert {
        "ReceiptUnavailable",
        "_read_json",
        "matched_cutoff",
        "METRICS",
        "_metric_counts",
        "_metric",
    } <= names


def test_reachability_flags_a_function_nothing_reaches():
    source = (
        _MODULE_PATH.read_text() + "\n\ndef _refit_cutoffs(items):\n    return items\n"
    )
    rows = calibration_reachability(source=source)
    assert [row["name"] for row in rows if not row["reached"]] == ["_refit_cutoffs"]
