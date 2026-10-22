"""Guards for the check-driven refine fan-out trigger.

The refine fan-out fires on the failure probability of an answered
retrieval check, so its predicate must read check results and nothing
else.  Two properties are locked here, because both are silent when
broken:

- ``should_trigger_fanout`` acquires no input that carries reviewer
  comment text, so no comment can steer the trigger.
- No keyword-list setting is reintroduced, by import or by settings
  load.

A mutation that must turn each guard red is named beside its test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from imas_codex.standard_names.fanout import trigger as trigger_module
from imas_codex.standard_names.fanout.config import FanoutSettings, load_settings
from imas_codex.standard_names.fanout.trigger import (
    should_trigger_fanout,
    should_trigger_fanout_from_checks,
)
from imas_codex.standard_names.review.checks import JevQuestion, ReviewCheck

# Identifiers that belong only to the retired comment/keyword trigger.
RETIRED_TRIGGER_IDENTIFIERS = (
    "reviewer_comments_per_dim",
    "trigger_source",
    "extract_reviewer_excerpt",
    "char_cap",
    "keywords",
    "dims",
)

# Settings fields and pyproject keys the comment path was configured by.
RETIRED_SETTINGS_FIELDS = (
    "refine_trigger_keywords",
    "refine_trigger_comment_dims",
    "refine_trigger_comment_chars",
    "refine_trigger_source",
)

RETIRED_PYPROJECT_KEYS = (
    "refine-trigger-keywords",
    "refine-trigger-comment-dims",
    "refine-trigger-comment-chars",
)

PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _question_check(*, check_id: str, retrieval: bool) -> ReviewCheck:
    return ReviewCheck(
        id=check_id,
        axis="name",
        dimension="semantic",
        evidence_fields=["source_paths"],
        retrieval=retrieval,
        question=JevQuestion(
            kind="choice",
            text="Which source object does this name denote?",
            options=["different", "bound", "uncertain"],
            conforming=["bound"],
        ),
    )


# ---------------------------------------------------------------------
# Guard 1 — the trigger acquires no comment input.
#   Mutation that must turn these red: give should_trigger_fanout a
#   comment-carrying parameter again (e.g. reviewer_comments_per_dim),
#   or reintroduce a comment-reading helper in the trigger module.
# ---------------------------------------------------------------------


def test_should_trigger_fanout_takes_no_comment_input() -> None:
    parameters = set(inspect.signature(should_trigger_fanout).parameters)
    assert parameters == {"checks", "check_results", "check_threshold"}, (
        "should_trigger_fanout must take check inputs only; a parameter "
        f"named otherwise is a comment input in disguise: {sorted(parameters)}"
    )


def test_trigger_module_reads_no_comment_text() -> None:
    source = inspect.getsource(trigger_module)
    hits = [name for name in RETIRED_TRIGGER_IDENTIFIERS if name in source]
    assert not hits, (
        "the fan-out trigger module still carries the retired comment-path "
        f"identifiers {hits}; it must read check results only"
    )


def test_should_trigger_fanout_fires_only_on_a_retrieval_check() -> None:
    relevant = _question_check(check_id="bound_to_wrong_object", retrieval=True)
    irrelevant = _question_check(check_id="grammar_only", retrieval=False)
    results = {
        relevant.id: {"answered": True, "failure_probability": 0.9},
        irrelevant.id: {"answered": True, "failure_probability": 0.99},
    }
    assert should_trigger_fanout(
        checks=[relevant], check_results=results, check_threshold=0.8
    )[0]
    assert not should_trigger_fanout(
        checks=[irrelevant], check_results=results, check_threshold=0.8
    )[0]
    fires, excerpt = should_trigger_fanout(checks=[], check_results={})
    assert fires is False
    assert excerpt == ""


def test_should_trigger_fanout_matches_the_check_predicate() -> None:
    check = _question_check(check_id="bound_to_wrong_object", retrieval=True)
    results = {check.id: {"answered": True, "failure_probability": 0.7}}
    outcomes: set[bool] = set()
    for threshold in (0.5, 0.7, 0.95):
        expected = should_trigger_fanout_from_checks(
            [check], results, threshold=threshold
        )
        fires, excerpt = should_trigger_fanout(
            checks=[check], check_results=results, check_threshold=threshold
        )
        assert fires is expected, (
            "the wrapper must agree with the predicate it delegates to at every "
            f"threshold; it disagreed at {threshold}"
        )
        assert excerpt == ""
        outcomes.add(fires)
    assert outcomes == {True, False}, (
        "the compared thresholds must straddle the measured probability, or the "
        "assertion holds even for a wrapper that always returns one value"
    )

    unanswered = {check.id: {"answered": False, "failure_probability": 0.99}}
    assert (
        should_trigger_fanout_from_checks([check], unanswered, threshold=0.5) is False
    )
    assert (
        should_trigger_fanout(
            checks=[check], check_results=unanswered, check_threshold=0.5
        )[0]
        is False
    )


# ---------------------------------------------------------------------
# Guard 2 — no keyword-list setting returns.
#   Mutation that must turn this red: restore refine_trigger_keywords on
#   FanoutSettings (and its pyproject key).
# ---------------------------------------------------------------------


def test_no_keyword_trigger_setting_by_import_and_settings_load() -> None:
    for field in RETIRED_SETTINGS_FIELDS:
        assert field not in FanoutSettings.model_fields, (
            f"{field} was reintroduced on FanoutSettings; the keyword "
            "trigger is retired"
        )
    loaded = load_settings()
    for field in RETIRED_SETTINGS_FIELDS:
        assert not hasattr(loaded, field), (
            f"load_settings() still exposes {field}; the keyword trigger is retired"
        )
    assert FanoutSettings().refine_trigger_check_threshold == 0.5


def test_pyproject_declares_no_keyword_trigger_key() -> None:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    present = [key for key in RETIRED_PYPROJECT_KEYS if key in text]
    assert not present, (
        f"pyproject.toml still declares the retired keyword settings: {present}"
    )
