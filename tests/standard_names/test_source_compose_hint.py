"""Durable exact-DD-source steering for pooled name composition."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

_PATH = "spectrometer_visible/channel/isotope_ratios/isotope/density_ratio"
_SOURCE_ID = f"dd:{_PATH}"
_HINT = "Name the per-isotope neutral-density odds ratio."
_REASON = "The DD denominator is the sum of every other isotope."


def _decision(
    value: str = "eligible",
    *,
    status: str | None = None,
) -> dict[str, object]:
    return {
        "id": _SOURCE_ID,
        "source_path": _PATH,
        "source_type": "dd",
        "source_status": "extracted",
        "claimed_at": None,
        "attempt_count": 0,
        "compose_hint_status": status,
        "decision": value,
    }


def test_schema_declares_source_hint_lifecycle() -> None:
    schema = yaml.safe_load(
        (
            Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
        ).read_text()
    )
    values = schema["enums"]["ComposeHintStatus"]["permissible_values"]
    assert set(values) == {"open", "consumed", "rejected"}
    source = schema["classes"]["StandardNameSource"]["attributes"]
    assert source["compose_hint_status"]["range"] == "ComposeHintStatus"
    assert source["compose_hint_requested_at"]["range"] == "datetime"
    assert source["compose_hint_consumed_at"]["range"] == "datetime"
    assert "non-authoritative" in source["compose_hint"]["description"]


@pytest.mark.parametrize(
    "path",
    ["", "equilibrium/*/psi", "equilibrium/[ab]/psi", "signals:tcv:x"],
)
def test_source_hint_requires_one_exact_dd_path(path: str) -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    with pytest.raises(ValueError, match="exact DD path"):
        set_source_compose_hint(path, hint=_HINT, reason=_REASON, gc=gc)
    gc.query.assert_not_called()


@pytest.mark.parametrize(
    ("hint", "reason", "message"),
    [
        (" ", _REASON, "non-empty hint"),
        (_HINT, " ", "non-empty reason"),
        ("x" * 4001, _REASON, "4000"),
        (_HINT, "x" * 2001, "2000"),
    ],
)
def test_source_hint_text_is_nonblank_and_bounded(
    hint: str,
    reason: str,
    message: str,
) -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    with pytest.raises(ValueError, match=message):
        set_source_compose_hint(_PATH, hint=hint, reason=reason, gc=MagicMock())


def test_source_hint_dry_run_is_nonmutating_and_exact() -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.return_value = [_decision()]
    result = set_source_compose_hint(
        _PATH,
        hint=_HINT,
        reason=_REASON,
        dry_run=True,
        gc=gc,
    )
    assert result == {
        **_decision(),
        "source_id": _SOURCE_ID,
        "source_path": _PATH,
        "eligible": True,
        "applied": False,
        "replaced": False,
        "dry_run": True,
        "decision": "would_set",
    }
    gc.query.assert_called_once()


def test_source_hint_cas_sets_only_steering_fields() -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.side_effect = [
        [_decision()],
        [{"id": _SOURCE_ID, "source_path": _PATH, "replaced": False}],
    ]
    result = set_source_compose_hint(f"dd:{_PATH}", hint=_HINT, reason=_REASON, gc=gc)
    assert result["applied"] is True
    assert result["decision"] == "set"
    write = gc.query.call_args_list[1]
    cypher = write.args[0]
    assert "sns.status = 'extracted'" in cypher
    assert "sns.claimed_at IS NULL" in cypher
    assert "coalesce(sns.attempt_count, 0) < $attempt_cap" in cypher
    assert "NOT EXISTS" in cypher and "PRODUCED_NAME" in cypher
    assert "sns.compose_hint_status = 'open'" in cypher
    assert "sns.compose_hint_consumed_at = null" in cypher
    set_clause = cypher.split("SET sns.compose_hint", 1)[1]
    for authority in (
        "sns.unit",
        "sns.cocos",
        "sns.dd_version",
        "sns.physics_domain",
        "sns.last_error",
    ):
        assert authority not in set_clause
    assert write.kwargs["source_id"] == _SOURCE_ID


def test_source_hint_replace_reopens_and_clears_consumption_time() -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.side_effect = [
        [_decision(status="open")],
        [{"id": _SOURCE_ID, "source_path": _PATH, "replaced": True}],
    ]
    result = set_source_compose_hint(
        _PATH,
        hint="Use the strict ratio grammar.",
        reason=_REASON,
        replace=True,
        gc=gc,
    )
    assert result["decision"] == "replaced"
    assert result["replaced"] is True
    write = gc.query.call_args_list[1]
    assert write.kwargs["replace"] is True
    assert "sns.compose_hint_consumed_at = null" in write.args[0]


@pytest.mark.parametrize(
    "decision",
    [
        "missing_source",
        "active_claim",
        "source_not_extracted",
        "attempt_cap_reached",
        "live_name_binding",
        "open_hint_exists",
    ],
)
def test_source_hint_refusals_never_write(decision: str) -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.return_value = [_decision(decision)]
    result = set_source_compose_hint(
        _PATH, hint=_HINT, reason=_REASON, replace=False, gc=gc
    )
    assert result["eligible"] is False
    assert result["decision"] == decision
    gc.query.assert_called_once()


def test_source_hint_claim_turnover_loses_cas() -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.side_effect = [
        [_decision()],
        [],
        [_decision("active_claim")],
    ]
    result = set_source_compose_hint(_PATH, hint=_HINT, reason=_REASON, gc=gc)
    assert result["applied"] is False
    assert result["eligible"] is False
    assert result["decision"] == "active_claim"
    assert gc.query.call_count == 3


def test_source_hint_unclassified_cas_loss_is_explicit() -> None:
    from imas_codex.standard_names.graph_ops import set_source_compose_hint

    gc = MagicMock()
    gc.query.side_effect = [[_decision()], [], [_decision()]]
    result = set_source_compose_hint(_PATH, hint=_HINT, reason=_REASON, gc=gc)
    assert result["applied"] is False
    assert result["eligible"] is False
    assert result["decision"] == "compare_and_set_lost"


def test_source_hint_decision_query_has_every_cas_refusal() -> None:
    from imas_codex.standard_names.graph_ops import _source_compose_hint_decision

    gc = MagicMock()
    gc.query.return_value = [_decision()]
    _source_compose_hint_decision(gc, source_id=_SOURCE_ID, replace=False)
    cypher = gc.query.call_args.args[0]
    for marker in (
        "missing_source",
        "not_dd_source",
        "active_claim",
        "source_not_extracted",
        "attempt_cap_reached",
        "live_name_binding",
        "open_hint_exists",
    ):
        assert marker in cypher


def test_source_hint_cli_success_dry_run_replace_and_refusal() -> None:
    from imas_codex.cli.sn import sn

    runner = CliRunner()
    base = ["source-hint", _PATH, "--hint", _HINT, "--reason", _REASON]
    with patch(
        "imas_codex.standard_names.graph_ops.set_source_compose_hint",
        return_value={
            "source_id": _SOURCE_ID,
            "eligible": True,
            "applied": True,
            "decision": "set",
        },
    ):
        applied = runner.invoke(sn, base)
    assert applied.exit_code == 0
    assert f"set: {_SOURCE_ID}" in applied.output

    with patch(
        "imas_codex.standard_names.graph_ops.set_source_compose_hint",
        return_value={
            "source_id": _SOURCE_ID,
            "eligible": True,
            "applied": False,
            "decision": "would_replace",
        },
    ) as setter:
        dry = runner.invoke(sn, [*base, "--replace", "--dry-run"])
    assert dry.exit_code == 0
    assert f"would_replace: {_SOURCE_ID} (dry-run)" in dry.output
    setter.assert_called_once_with(
        _PATH,
        hint=_HINT,
        reason=_REASON,
        replace=True,
        dry_run=True,
    )

    with patch(
        "imas_codex.standard_names.graph_ops.set_source_compose_hint",
        return_value={
            "source_id": _SOURCE_ID,
            "eligible": False,
            "applied": False,
            "decision": "active_claim",
        },
    ):
        refused = runner.invoke(sn, base)
    assert refused.exit_code != 0
    assert "active_claim" in refused.output


def test_claim_readback_returns_both_source_and_name_steering() -> None:
    from imas_codex.standard_names.graph_ops import claim_generate_name_batch

    tx = MagicMock()
    tx.closed = False
    tx.run.side_effect = [
        [
            {
                "_cluster_id": "cluster",
                "_unit": "1",
                "_physics_domain": "spectroscopy",
                "_batch_key": "spectroscopy:1",
            }
        ],
        [
            {
                "id": _SOURCE_ID,
                "source_id": _PATH,
                "source_type": "dd",
                "claim_token": "token",
                "claim_seq": 4,
                "compose_hint": _HINT,
                "compose_hint_reason": _REASON,
                "compose_hint_status": "open",
                "previous_name": "hydrogen_fraction",
                "name_hint": "retain the per-isotope denominator",
                "edit_reason": "the prior name collapsed isotope scope",
                "edit_origin": "human",
            }
        ],
    ]
    session = MagicMock()
    session.begin_transaction.return_value = tx
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False
    gc.session.return_value.__enter__.return_value = session
    gc.session.return_value.__exit__.return_value = False
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.graph_ops._verify_source_claim_winners",
            side_effect=lambda items: items,
        ),
    ):
        rows = claim_generate_name_batch(batch_size=1)

    assert rows[0]["compose_hint"] == _HINT
    assert rows[0]["name_hint"] == "retain the per-isotope denominator"
    readback = tx.run.call_args_list[1].args[0]
    for field in (
        "compose_hint",
        "compose_hint_reason",
        "compose_hint_status",
        "name_hint",
        "edit_reason",
        "edit_origin",
    ):
        assert field in readback
    assert "PRODUCED_NAME" in readback
    assert "hinted.edit_status = 'open'" in readback


def _prompt_context(item: dict[str, object]) -> dict[str, object]:
    return {
        "items": [item],
        "ids_name": "spectrometer_visible",
        "ids_contexts": {},
        "existing_names": [],
        "cluster_context": None,
        "nearby_existing_names": [],
        "reference_exemplars": [],
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }


def test_source_hint_prompt_is_conditional_and_non_authoritative() -> None:
    from imas_codex.llm.prompt_loader import render_prompt

    base = {
        "path": _PATH,
        "description": "Ratio of this isotope density to all other isotopes.",
        "data_type": "FLT_1D",
        "units": "1",
    }
    plain = render_prompt("sn/generate_name_dd", _prompt_context(dict(base)))
    explicit_nulls = render_prompt(
        "sn/generate_name_dd",
        _prompt_context(
            {
                **base,
                "compose_hint": None,
                "compose_hint_reason": None,
                "compose_hint_status": None,
            }
        ),
    )
    assert plain == explicit_nulls
    assert "Operator steering for this exact DD source" not in plain

    steered = render_prompt(
        "sn/generate_name_dd",
        _prompt_context(
            {
                **base,
                "compose_hint": _HINT,
                "compose_hint_reason": _REASON,
                "compose_hint_status": "open",
            }
        ),
    )
    assert _HINT in steered and _REASON in steered
    for authority in (
        "pinned DD snapshot",
        "unit",
        "COCOS convention",
        "physics domain",
        "grammar validation",
    ):
        assert authority in steered


@pytest.mark.asyncio
async def test_pooled_compose_delivers_source_and_name_steering_to_prompt() -> None:
    from imas_codex.llm.prompt_loader import render_prompt as real_render
    from imas_codex.standard_names.workers import compose_batch

    batch = [
        {
            "id": _SOURCE_ID,
            "path": _PATH,
            "claim_token": "token",
            "claim_seq": 4,
            "description": "Ratio of this isotope density to all other isotopes.",
            "physics_domain": "spectroscopy",
            "unit": "1",
            "dd_version": "4.0.0",
            "cocos_version": 11,
            "compose_hint": _HINT,
            "compose_hint_reason": _REASON,
            "compose_hint_status": "open",
            "previous_name": "hydrogen_fraction",
            "name_hint": "keep the isotope-specific numerator",
            "edit_reason": "the earlier candidate erased the array element",
            "edit_origin": "human",
        }
    ]
    captured: dict[str, object] = {}

    def _render(name: str, context: dict[str, object]) -> str:
        if name == "sn/generate_name_system":
            return "system"
        if name == "sn/generate_name_dd":
            captured.update(context)
            return real_render(name, context)
        return "unused"

    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False
    graph.query.return_value = []
    lease = MagicMock()
    manager = MagicMock(run_id="run")
    manager.reserve.return_value = lease
    with (
        patch(
            "imas_codex.standard_names.graph_ops._verify_source_claim_winners",
            side_effect=lambda items, **_kwargs: items,
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context", return_value={}
        ),
        patch("imas_codex.settings.get_model", return_value="test-model"),
        patch(
            "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
            return_value="",
        ),
        patch(
            "imas_codex.standard_names.review.themes.extract_reviewer_themes",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=graph),
        patch("imas_codex.standard_names.workers._enrich_batch_items"),
        patch(
            "imas_codex.standard_names.workers._search_nearby_names", return_value=[]
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_ids_context", return_value=None
        ),
        patch("imas_codex.llm.prompt_loader.render_prompt", side_effect=_render),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new=AsyncMock(side_effect=RuntimeError("stop after prompt capture")),
        ),
        pytest.raises(RuntimeError, match="stop after prompt capture"),
    ):
        await compose_batch(batch, manager, asyncio.Event())

    item = captured["items"][0]  # type: ignore[index]
    assert item["compose_hint"] == _HINT
    assert item["review_feedback"]["name_hint"] == (
        "keep the isotope-specific numerator"
    )
    rendered = real_render("sn/generate_name_dd", captured)
    assert _HINT in rendered
    assert "keep the isotope-specific numerator" in rendered


def test_only_successful_binding_queries_consume_open_hints() -> None:
    from imas_codex.standard_names.graph_ops import (
        _lock_claimed_name_bindings,
        persist_claimed_attachments,
        persist_claimed_source_outcomes,
        persist_generated_name_winners,
        retry_failed_sources,
    )

    generated = inspect.getsource(persist_generated_name_winners)
    attached = inspect.getsource(persist_claimed_attachments)
    assert generated.count("sns.compose_hint_status = 'consumed'") == 2
    assert "reserved_finalize_batch" in generated
    assert "stable_finalize_batch" in generated
    assert attached.count("sns.compose_hint_status = 'consumed'") == 1
    for non_success in (
        _lock_claimed_name_bindings,
        persist_claimed_source_outcomes,
        retry_failed_sources,
    ):
        source = inspect.getsource(non_success)
        assert "compose_hint_status = 'consumed'" not in source
        assert "compose_hint = null" not in source
        assert "compose_hint_reason = null" not in source
    assert "sns.compose_hint = null" not in generated + attached
    assert "sns.compose_hint_reason = null" not in generated + attached
