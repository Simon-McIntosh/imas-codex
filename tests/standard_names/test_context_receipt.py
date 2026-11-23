"""Tests for the per-stage context receipt and its census.

The receipt is the guard the section rests on, so these tests make both halves
of the control happen. A name carrying every channel must report complete at
every stage (the positive control). Withholding one channel from one stage must
make that stage's receipt name the omission and refuse to report complete, and
the census must count it (the negative control). A stage that expects a channel
and does not receive it is the failure this record exists to make visible, so
the verifying test is the negative one: a receipt that reports complete with a
channel withheld would pass every other test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from iter_codex.standard_names import context_receipt as cr

_TESTS = Path(__file__).resolve().parent


def _built_case_file(**kwargs):
    """Build the same complete case file the case-file tests build."""
    sys.path.insert(0, str(_TESTS))
    from test_case_file import _build

    return _build(**kwargs)


def test_a_name_with_every_channel_reports_complete_at_every_stage():
    """Positive control: the full record satisfies every stage's expectation."""
    record = _built_case_file()

    for stage in cr.STAGES:
        receipt = cr.receipt_for_stage(stage, record)
        assert receipt.complete is True, (stage, receipt.missing)
        assert receipt.missing == [], stage
        assert receipt.expected, stage
        assert all(receipt.channels[channel].present for channel in receipt.expected)


def test_a_withheld_channel_is_named_and_the_context_is_not_complete():
    """Negative control: the stage reports the omission, by name."""
    record = _built_case_file()

    receipt = cr.receipt_for_stage(
        "documentation", record, withhold=["enclosing_structure_text"]
    )

    assert receipt.complete is False
    assert receipt.missing == ["enclosing_structure_text"]
    assert receipt.withheld == ["enclosing_structure_text"]
    absent = receipt.channels["enclosing_structure_text"]
    assert absent.present is False
    assert "withheld" in absent.reason
    # The other channels still arrived, so the omission is specific.
    assert receipt.channels["grammar"].present is True


def test_the_census_counts_the_withheld_channel_and_ranks_it():
    """The negative control reaches the census as a counted omission."""
    record = _built_case_file()

    complete = [cr.receipt_for_stage(stage, record) for stage in cr.STAGES]
    withheld = [
        cr.receipt_for_stage(
            "documentation", record, withhold=["enclosing_structure_text"]
        )
    ]

    clean = cr.summarise(complete)
    assert clean["omissions"] == []
    assert clean["table"]["documentation"]["enclosing_structure_text"]["absent"] == 0

    counted = cr.summarise(complete + withheld)
    assert counted["table"]["documentation"]["enclosing_structure_text"]["absent"] == 1
    assert counted["table"]["documentation"]["enclosing_structure_text"]["present"] == 1
    top = counted["omissions"][0]
    assert (top["stage"], top["channel"], top["names"]) == (
        "documentation",
        "enclosing_structure_text",
        1,
    )


def test_an_absent_channel_absent_in_the_record_is_reported_not_ignored():
    """A channel that never arrived reads absent without any withholding."""
    content = {"target": {"id": "no_sources"}, "producer": {"status": "none"}}

    receipt = cr.receipt_from_content("extraction", "no_sources", content)

    assert receipt.complete is False
    assert "source_text" in receipt.missing
    assert "enclosing_structure_text" in receipt.missing
    assert "unit" in receipt.missing
    assert "coordinates" not in receipt.missing
    assert "coordinates" in cr.CHANNELS
    assert all("coordinates" not in expected for expected in cr.STAGE_CHANNELS.values())


def test_a_stage_rendering_never_reaches_a_model():
    """Rendering a stage's prompt is a template read; a model call would fail."""
    import iter_codex.discovery.base.llm as llm

    def _forbidden(*args, **kwargs):
        raise AssertionError("a receipt must never call a model")

    if hasattr(llm, "acall_llm_structured"):
        original_async = llm.acall_llm_structured
        llm.acall_llm_structured = _forbidden
    else:  # pragma: no cover - the call layer always defines it
        original_async = None
    original_sync = getattr(llm, "call_llm_structured", None)
    if original_sync is not None:
        llm.call_llm_structured = _forbidden
    try:
        record = _built_case_file()
        for stage in cr.STAGES:
            receipt = cr.receipt_for_stage(stage, record)
            assert receipt.prompt_chars > 0
    finally:
        if original_async is not None:
            llm.acall_llm_structured = original_async
        if original_sync is not None:
            llm.call_llm_structured = original_sync


def test_the_prompt_digest_is_stable_and_moves_with_the_text():
    record = _built_case_file()

    first = cr.receipt_for_stage("name_review", record)
    again = cr.receipt_for_stage("name_review", record)
    assert first.prompt_digest == again.prompt_digest

    other = cr.receipt_for_stage("documentation", record)
    assert other.prompt_digest != first.prompt_digest


def test_an_unknown_stage_or_withheld_channel_is_refused():
    record = _built_case_file()

    with pytest.raises(ValueError, match="unknown stage"):
        cr.receipt_from_content("telepathy", "x", {})

    with pytest.raises(ValueError, match="unknown withheld channel"):
        cr.receipt_from_case_file("extraction", record, withhold=["everything"])


def test_the_sink_holds_one_json_line_per_receipt(tmp_path):
    sink = tmp_path / "receipts.jsonl"
    record = _built_case_file()

    receipt = cr.receipt_for_stage("refinement", record)
    cr.record_receipt(receipt, sink=sink)

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    import json

    payload = json.loads(lines[0])
    assert payload["stage"] == "refinement"
    assert payload["schema_id"] == cr.RECEIPT_SCHEMA
    assert payload["complete"] is True


def test_a_parent_stage_expects_the_children_channel():
    """A derived parent's children are a channel, not an empty source."""
    parent_content = {
        "target": {"id": "elongation_of_boundary"},
        "producer": {"status": "derived", "dd_paths": [], "derived": ["x"]},
        "children": {
            "status": "members",
            "statement": "2 child name(s)",
            "members": [
                {"name": "inner_elongation", "description": "a"},
                {"name": "outer_elongation", "description": "b"},
            ],
        },
        "incumbent": {"id": "elongation_of_boundary", "name": "x"},
        "grammar": {"status": "parsed", "segments": {}},
    }

    receipt = cr.receipt_from_content(
        "parent_enrichment", "elongation_of_boundary", parent_content
    )
    assert "children" in receipt.expected
    assert receipt.channels["children"].present is True
    assert "2 child name(s)" in receipt.channels["children"].reason

    non_parent = dict(
        parent_content,
        children={"status": "none", "statement": "no children", "members": []},
    )
    receipt = cr.receipt_from_content("parent_enrichment", "x", non_parent)
    assert receipt.channels["children"].present is True


def test_a_live_stage_context_records_what_the_stage_sent(tmp_path):
    """The production entry point records a receipt from a stage's own context."""
    from iter_codex.standard_names.context import build_compose_context

    context = dict(build_compose_context())
    context.update(
        {
            "item": {
                "id": "elongation_of_flux_surface",
                "name": "elongation_of_flux_surface",
                "kind": "scalar",
                "unit": "1",
                "description": "Elongation of a flux surface.",
                "dd_description": "Elongation of the flux surface.",
                "source_paths": ["dd:equilibrium/time_slice/profiles_1d/elongation"],
                "derived_children": ["inner_elongation"],
            },
            "anc": [{"path": "equilibrium", "text": "Equilibrium IDS."}],
            "sib": ["elongation_of_boundary"],
            "vector_neighbours": [{"id": "squareness_of_flux_surface"}],
            "coordinates": ["1"],
        }
    )
    sink = tmp_path / "stage.jsonl"

    receipt = cr.record_stage_receipt(
        "name_review",
        "elongation_of_flux_surface",
        context,
        prompt_text="the exact prompt",
        sink=sink,
    )

    assert receipt.complete is True, receipt.missing
    assert receipt.prompt_digest == cr.prompt_digest("the exact prompt")
    assert sink.exists()
    assert '"stage": "name_review"' in sink.read_text(encoding="utf-8")


def test_a_live_context_without_source_text_reports_the_omission():
    """A stage that sent no source text is not recorded as complete."""
    context = {
        "item": {"id": "no_text", "name": "no_text"},
        "vector_neighbours": [{"id": "x"}],
    }

    receipt = cr.record_stage_receipt("name_composition", "no_text", context)

    assert receipt.complete is False
    assert "source_text" in receipt.missing


def test_record_receipt_writes_to_the_default_sink_when_none_is_given(
    tmp_path, monkeypatch
):
    """A stage's receipt reaches a sink nobody configured.

    The default sink is what makes a live stage's record exist without the
    pipeline being wired to produce one; a receipt written only when an
    explicit sink or an environment override is supplied is one no stage
    writes in the course of its run.
    """
    monkeypatch.delenv(cr.RECEIPT_SINK_ENV, raising=False)
    target = tmp_path / "default.jsonl"
    monkeypatch.setattr(cr, "DEFAULT_RECEIPT_SINK", target)

    written = cr.record_receipt(
        cr.receipt_from_content(
            "name_review", "elongation_of_flux_surface", {"target": {"id": "x"}}
        )
    )

    assert written == target
    assert target.exists()
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
