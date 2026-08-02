"""Automated DD-defect evidence remains exact, additive, and non-authoritative."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.models import DDGapEvidence

PATH = "equilibrium/time_slice/profiles_1d/electrons/temperature"
REFERENCE_PATH = "equilibrium/time_slice/profiles_2d/electrons/temperature"


def _evidence(path: str = PATH, *, reference_path: str | None = None) -> DDGapEvidence:
    payload = {
        "path": path,
        "kind": "unit_defect",
        "reason": "The declared unit contradicts the independent quantity definition.",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
    }
    if reference_path is not None:
        payload["reference_path"] = reference_path
        payload["reference_value"] = "Pa"
    return DDGapEvidence.model_validate(payload)


def test_compose_sanitizer_keeps_only_exact_claimed_dd_evidence() -> None:
    from imas_codex.standard_names.workers import _sanitize_compose_result_sources

    result = SimpleNamespace(
        candidates=[],
        attachments=[],
        skipped=[],
        vocab_gaps=[],
        dd_gaps=[_evidence(), _evidence("unclaimed/path")],
    )

    _sanitize_compose_result_sources(result, {PATH}, phase="generate_name")

    assert [report.path for report in result.dd_gaps] == [PATH]


def test_compose_sanitizer_resolves_multiple_dispositions_by_fixed_priority() -> None:
    from imas_codex.standard_names.workers import _sanitize_compose_result_sources

    source = SimpleNamespace(source_id=PATH, dd_paths=[PATH])
    result = SimpleNamespace(
        candidates=[source],
        attachments=[source],
        skipped=[PATH],
        vocab_gaps=[source],
        dd_gaps=[_evidence()],
    )

    _sanitize_compose_result_sources(result, {PATH}, phase="generate_name")

    assert result.vocab_gaps == [source]
    assert result.attachments == []
    assert result.skipped == []
    assert result.candidates == []


def test_reference_evidence_must_also_be_in_claimed_batch() -> None:
    from imas_codex.standard_names.workers import _sanitize_dd_gap_evidence

    evidence = [_evidence(reference_path=REFERENCE_PATH)]

    assert _sanitize_dd_gap_evidence(evidence, {PATH}, phase="review_name") == []
    assert (
        _sanitize_dd_gap_evidence(
            evidence,
            {PATH, REFERENCE_PATH},
            phase="review_name",
        )
        == evidence
    )


def test_persisted_evidence_injects_writer_and_dd_version() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        return_value={"reported": 1},
    ) as write:
        reported = _persist_dd_gap_evidence(
            [_evidence()],
            {PATH},
            phase="generate_name",
            reporter="compose",
            observed_dd_version="4.1.1",
        )

    assert reported == 1
    report = write.call_args.args[0][0]
    assert report["path"] == PATH
    assert report["reporter"] == "compose"
    assert report["observed_dd_version"] == "4.1.1"
    assert "status" not in report


def test_evidence_writer_failure_isolated_from_pipeline_result() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        side_effect=RuntimeError("graph unavailable"),
    ):
        assert (
            _persist_dd_gap_evidence(
                [_evidence()],
                {PATH},
                phase="review_docs",
                reporter="review-docs",
            )
            == 0
        )


def test_no_evidence_performs_no_graph_write() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write:
        assert (
            _persist_dd_gap_evidence(
                [],
                {PATH},
                phase="review_name",
                reporter="review-name",
            )
            == 0
        )
    write.assert_not_called()


def test_unit_injection_reports_only_physical_scalar_edge_conflicts() -> None:
    from imas_codex.standard_names.sources.dd import (
        _unit_declaration_conflict_reports,
    )

    rows = [
        {"path": PATH, "unit": "1", "unit_from_rel": "Pa"},
        {
            "path": "equilibrium/same",
            "unit": "W.m^-3",
            "unit_from_rel": "m^-3.W",
        },
        {"path": "equilibrium/missing", "unit": None, "unit_from_rel": "Pa"},
    ]

    reports = _unit_declaration_conflict_reports(rows, "4.1.1")

    assert reports == [
        {
            "path": PATH,
            "kind": "self_contradiction",
            "reason": (
                "The DD node unit property contradicts its authoritative "
                "HAS_UNIT relationship."
            ),
            "observed_dd_version": "4.1.1",
            "observed_value": "1",
            "expected_value": "Pa",
            "evidence_rule": "unit_equals_expected",
            "reporter": "dd-unit-injection",
        }
    ]


def test_unit_injection_writer_failure_does_not_block_extraction() -> None:
    from imas_codex.standard_names.sources.dd import (
        _persist_unit_declaration_conflicts,
    )

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        side_effect=RuntimeError("graph unavailable"),
    ):
        assert (
            _persist_unit_declaration_conflicts(
                [{"path": PATH, "unit": "1", "unit_from_rel": "Pa"}],
                "4.1.1",
            )
            == 0
        )


def test_attachment_evidence_uses_fetched_unit_property_and_relationship() -> None:
    from imas_codex.standard_names.attachment_audit import (
        _attachment_dd_gap_evidence,
    )

    reports = _attachment_dd_gap_evidence(
        [
            {
                "dd_path": PATH,
                "dd_declared_unit": "1",
                "dd_relationship_units": ["Pa"],
                "dd_version": "4.1.1",
            }
        ]
    )

    assert len(reports) == 1
    assert reports[0]["path"] == PATH
    assert reports[0]["observed_value"] == "1"
    assert reports[0]["expected_value"] == "Pa"
    assert reports[0]["reporter"] == "attachment-audit"
    assert reports[0]["observed_dd_version"] == "4.1.1"


def test_attachment_evidence_reports_multiple_unit_edges_deterministically() -> None:
    from imas_codex.standard_names.attachment_audit import (
        _attachment_dd_gap_evidence,
    )

    reports = _attachment_dd_gap_evidence(
        [
            {
                "dd_path": PATH,
                "dd_declared_unit": "1",
                "dd_relationship_units": ["Pa", "eV", "Pa"],
                "dd_version": "4.1.1",
            }
        ]
    )

    assert reports[0]["expected_value"] == "Pa,eV"
    assert reports[0]["evidence_rule"] == "unit_relationship_is_unique"


@pytest.mark.asyncio
async def test_review_quorum_collects_evidence_without_changing_score() -> None:
    from imas_codex.standard_names.workers import _run_rd_quorum_cycles

    scores = {"grammar": 18, "semantic": 18, "convention": 18, "completeness": 18}

    async def call_llm(**_kwargs):
        result = SimpleNamespace(
            scores=SimpleNamespace(score=0.9, model_dump=lambda: scores),
            comments=None,
            reasoning="The name satisfies the rubric independently of DD evidence.",
            dd_gaps=[_evidence()],
        )
        return result, 0.01, 100

    result = await _run_rd_quorum_cycles(
        sn_id="electron_temperature",
        review_axis="names",
        response_model=object,
        user_prompt="review",
        system_prompt="system",
        models=["reviewer"],
        disagreement_threshold=0.2,
        rubric_dims=("grammar", "semantic", "convention", "completeness"),
        lease=None,
        phase="review_name",
        acall_llm_structured=call_llm,
    )

    assert result is not None
    assert result["winning_score"] == pytest.approx(0.9)
    assert result["resolution_method"] == "single_review"
    assert result["dd_gaps"] == [_evidence()]


def test_attachment_reconcile_dry_run_never_writes_evidence() -> None:
    from imas_codex.standard_names.attachment_audit import (
        reconcile_attachment_consistency,
    )

    gc = MagicMock()
    gc.query.return_value = [
        {
            "source_node_id": f"dd:{PATH}",
            "dd_path": PATH,
            "sn_id": "electron_temperature",
            "name_stage": "drafted",
            "origin": "pipeline",
            "dd_unit": "Pa",
            "dd_declared_unit": "1",
            "dd_relationship_units": ["Pa"],
            "sn_unit": "Pa",
            "other_live_names": 0,
        }
    ]

    with patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write:
        result = reconcile_attachment_consistency(gc, dry_run=True)

    assert result.dd_gap_evidence
    write.assert_not_called()


def test_targeted_extraction_dry_run_has_no_graph_writers() -> None:
    from imas_codex.standard_names.sources.dd import extract_specific_paths

    gc = MagicMock()
    gc.query.side_effect = [
        [{"dd_version": "4.1.1", "cocos_version": None, "cocos_params": None}],
        [{"path": PATH, "unit": "1", "unit_from_rel": "Pa"}],
    ]
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.sources.dd._persist_unit_declaration_conflicts"
        ) as persist_conflicts,
        patch(
            "imas_codex.standard_names.sources.dd._apply_unit_overrides",
            side_effect=lambda rows, **_: rows,
        ) as apply_units,
        patch(
            "imas_codex.standard_names.sources.dd._qualify_sources", return_value=[]
        ) as qualify,
    ):
        assert extract_specific_paths([PATH], write_side_effects=False) == []

    persist_conflicts.assert_not_called()
    assert apply_units.call_args.kwargs["write_skipped"] is False
    assert qualify.call_args.kwargs["write_skipped"] is False


@pytest.mark.parametrize("extractor", ["full", "targeted"])
@pytest.mark.parametrize("edge_units", [["Pa", "eV"], ["eV", "Pa"]])
def test_extraction_refuses_multiple_unit_edges_deterministically(
    extractor: str, edge_units: list[str]
) -> None:
    from imas_codex.standard_names.sources.dd import (
        extract_dd_candidates,
        extract_specific_paths,
    )

    gc = MagicMock()
    gc.query.side_effect = [
        [{"dd_version": "4.1.1", "cocos_version": None, "cocos_params": None}],
        [
            {
                "path": PATH,
                "unit": "1",
                "unit_from_rel": edge_units[0],
                "unit_relationships": edge_units,
            }
        ],
    ]
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.sources.dd.report_extract_breakdown",
            side_effect=RuntimeError("not needed"),
        ),
        patch(
            "imas_codex.standard_names.sources.dd._apply_unit_overrides",
            side_effect=lambda rows, **_: rows,
        ) as apply_units,
        patch("imas_codex.standard_names.sources.dd._qualify_sources", return_value=[]),
        patch(
            "imas_codex.standard_names.dd_gaps.write_dd_gaps",
            return_value={"reported": 1},
        ) as write,
    ):
        if extractor == "full":
            batches = extract_dd_candidates(explicit_paths=[PATH])
        else:
            batches = extract_specific_paths([PATH])

    assert batches == []
    report = write.call_args.args[0][0]
    assert report["expected_value"] == "Pa,eV"
    assert report["evidence_rule"] == "unit_relationship_is_unique"
    assert apply_units.call_args.args[0][0]["unit"] is None
    assert "unit_rels[0]" not in gc.query.call_args_list[1].args[0]


@pytest.mark.parametrize("axis", ["name", "docs"])
def test_review_claim_projects_authoritative_dd_bindings(axis: str) -> None:
    from imas_codex.standard_names import graph_ops

    claim_name = f"claim_review_{axis}_batch"
    with (
        patch.object(graph_ops, "_claim_sn_atomic", return_value=[]) as claim,
        patch.object(graph_ops, f"_verify_{axis}_claim_winners", return_value=[]),
    ):
        getattr(graph_ops, claim_name)(batch_size=1)

    projection = claim.call_args.kwargs["extra_return_fields"]
    assert "PRODUCED_NAME" in projection
    assert "source.dd_path" in projection
    assert "source.dd_version" in projection


def _review_item(axis: str) -> dict:
    item = {
        "id": "electron_temperature",
        "name": "electron_temperature",
        "description": "Electron temperature.",
        "documentation": "Electron temperature documentation.",
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": "core_plasma_physics",
        "validation_status": "valid",
        "claim_token": "claim-token",
        "claim_seq": 4,
        # Deliberately stale: evidence authorization must ignore this cache.
        "source_paths": ["dd:stale/scalar/path"],
        "source_bindings": [
            {
                "id": f"dd:{PATH}",
                "source_type": "dd",
                "source_id": PATH,
                "dd_path": PATH,
                "dd_version": "4.1.1",
            }
        ],
    }
    item[f"{axis}_stage"] = "drafted"
    item[f"{axis}_chain_length"] = 0
    return item


@pytest.mark.asyncio
@pytest.mark.parametrize("axis", ["name", "docs"])
@pytest.mark.parametrize("transition", [None, "accepted"])
async def test_review_writer_requires_successful_owned_transition(
    axis: str, transition: str | None
) -> None:
    from imas_codex.standard_names import workers

    dims = (
        {"grammar": 18, "semantic": 18, "convention": 18, "completeness": 18}
        if axis == "name"
        else {
            "description_quality": 18,
            "documentation_quality": 18,
            "completeness": 18,
            "physics_accuracy": 18,
        }
    )
    llm_result = SimpleNamespace(
        scores=SimpleNamespace(score=0.9, model_dump=lambda: dims),
        comments=None,
        reasoning="Evidence is independent of the review score.",
        dd_gaps=[_evidence()],
    )
    mgr = MagicMock(run_id="test-run")
    mgr.reserve.return_value = MagicMock()
    process = (
        workers.process_review_name_batch
        if axis == "name"
        else workers.process_review_docs_batch
    )
    persist_name = (
        "persist_reviewed_name" if axis == "name" else "persist_reviewed_docs"
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new=AsyncMock(return_value=(llm_result, 0.01, 100)),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["reviewer"],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["reviewer"],
            )
        )
        stack.enter_context(
            patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt")
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.context.fetch_review_neighbours",
                return_value={
                    "vector_neighbours": [],
                    "same_base_neighbours": [],
                    "same_path_neighbours": [],
                },
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.context._build_enum_lists", return_value={}
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.context.build_compose_context",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.example_loader.load_review_examples",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                f"imas_codex.standard_names.graph_ops.{persist_name}",
                return_value=transition,
            )
        )
        stack.enter_context(patch("imas_codex.standard_names.graph_ops.write_reviews"))
        stack.enter_context(
            patch("imas_codex.standard_names.graph_ops.update_review_aggregates")
        )
        write = stack.enter_context(
            patch(
                "imas_codex.standard_names.dd_gaps.write_dd_gaps",
                return_value={"reported": 1},
            )
        )
        await process([_review_item(axis)], mgr, asyncio.Event())

    if transition is None:
        write.assert_not_called()
    else:
        report = write.call_args.args[0][0]
        assert report["path"] == PATH
        assert report["observed_dd_version"] == "4.1.1"


def _compose_response(disposition: str) -> object:
    candidate = SimpleNamespace(
        source_id=PATH,
        dd_paths=[PATH],
        description="Electron temperature.",
        kind="scalar",
        reason="generated",
        compose_name=lambda: "electron_temperature",
    )
    attachment = SimpleNamespace(
        source_id=PATH,
        standard_name="electron_temperature",
        reason="existing name",
    )
    vocab_gap = SimpleNamespace(
        source_id=PATH,
        segment="physical_base",
        token="temperature",
        reason="missing token",
    )
    result = SimpleNamespace(
        candidates=[candidate] if disposition == "generated" else [],
        attachments=[attachment] if disposition == "attachment" else [],
        vocab_gaps=[vocab_gap] if disposition == "vocab_gap" else [],
        skipped=[PATH] if disposition == "skip" else [],
        dd_gaps=[_evidence()],
    )

    class LLMResult:
        input_tokens = 50
        output_tokens = 50
        cache_read_tokens = 0
        cache_creation_tokens = 0

        def __iter__(self):
            return iter((result, 0.01, 100))

    return LLMResult()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disposition", ["generated", "attachment", "vocab_gap", "skip"]
)
@pytest.mark.parametrize("claim_won", [False, True])
async def test_compose_evidence_follows_each_committed_disposition(
    disposition: str, claim_won: bool
) -> None:
    from imas_codex.standard_names.workers import process_generate_name_batch

    batch = [
        {
            "id": f"dd:{PATH}",
            "path": PATH,
            "claim_token": "winner",
            "claim_seq": 3,
            "description": "Electron temperature.",
            "physics_domain": "core_plasma_physics",
            "unit": "eV",
            "dd_version": "4.1.1",
            "cocos_version": None,
        }
    ]
    winner_ids = [f"dd:{PATH}"] if claim_won else []
    mgr = MagicMock(run_id="test-run")
    mgr.reserve.return_value = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new=AsyncMock(return_value=_compose_response(disposition)),
            )
        )
        stack.enter_context(
            patch("imas_codex.settings.get_model", return_value="model")
        )
        stack.enter_context(
            patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt")
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.context.build_compose_context",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
                return_value="",
            )
        )
        stack.enter_context(
            patch("imas_codex.standard_names.workers._enrich_batch_items")
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._search_nearby_names",
                return_value=[],
            )
        )
        stack.enter_context(
            patch("imas_codex.standard_names.workers._enrich_ids_context")
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._compute_token_reuse_hits",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._active_editorial_gap_guidance",
                return_value=("", False),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.graph_ops._verify_source_claim_winners",
                return_value=batch,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers.is_well_formed_candidate",
                return_value=(True, None),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers.is_non_nameable_coordinate",
                return_value=(False, None),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._is_attachment_consistent",
                return_value=(True, ""),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._auto_detect_physical_base_gaps",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.example_loader.load_compose_examples",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
                return_value=winner_ids,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._process_attachments_core",
                return_value={
                    "accepted": int(claim_won),
                    "rejected": int(not claim_won),
                    "winner_ids": winner_ids,
                },
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._consume_claimed_vocab_gaps",
                return_value=(
                    [
                        {
                            "source_id": PATH,
                            "segment": "physical_base",
                            "token": "temperature",
                            "reason": "missing token",
                        }
                    ],
                    winner_ids,
                ),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.workers._persist_claimed_skips",
                return_value=winner_ids,
            )
        )
        stack.enter_context(patch("imas_codex.graph.client.GraphClient"))
        write = stack.enter_context(
            patch(
                "imas_codex.standard_names.dd_gaps.write_dd_gaps",
                return_value={"reported": 1},
            )
        )
        await process_generate_name_batch(batch, mgr, asyncio.Event())

    if claim_won:
        assert write.call_args.args[0][0]["path"] == PATH
    else:
        write.assert_not_called()
