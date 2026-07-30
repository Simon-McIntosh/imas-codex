"""Vocabulary observations and editorial decisions remain independent."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError


class _Transaction:
    def __init__(
        self,
        rows: list[dict],
        token_rows: list[dict] | None = None,
        applied_count: int | None = None,
    ) -> None:
        self.rows = rows
        self.token_rows = token_rows or []
        self.applied_count = applied_count
        self._closed = False
        self.committed = False
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "graph_token_ids" in query:
            return [dict(row) for row in self.token_rows]
        if "requested_id" in query:
            return [dict(row) for row in self.rows]
        if "RETURN count(vg) AS applied" in query:
            applied = (
                len(params["items"])
                if self.applied_count is None
                else self.applied_count
            )
            return [{"applied": applied}]
        return []

    def commit(self) -> None:
        self.committed = True
        self._closed = True

    def close(self) -> None:
        self._closed = True

    def closed(self) -> bool:
        return self._closed


class _Client:
    def __init__(
        self,
        rows: list[dict],
        token_rows: list[dict] | None = None,
        applied_count: int | None = None,
    ) -> None:
        self.tx = _Transaction(rows, token_rows, applied_count)

    @contextmanager
    def session(self):
        yield SimpleNamespace(begin_transaction=lambda: self.tx)


def _artifact_path() -> Path:
    return Path(__file__).parents[2] / "docs/evidence/sn-vocabulary-adjudication.json"


def _single_batch(
    *,
    decision: str = "fold",
    target: str | None = "heat_flux",
    token: str = "heat_flux",
):
    from imas_codex.standard_names.vocab_adjudication import (
        VocabGapAdjudicationFile,
    )

    return VocabGapAdjudicationFile.model_validate(
        {
            "count": 1,
            "summary": {
                "add": int(decision == "add"),
                "fold": int(decision == "fold"),
                "reject": int(decision == "reject"),
            },
            "decisions": [
                {
                    "segment": "physical_base",
                    "token": token,
                    "decision": decision,
                    "canonical_target": target,
                    "rationale": "Express the quantity with registered grammar.",
                }
            ],
        }
    )


def _existing_row(**overrides) -> dict:
    row = {
        "requested_id": "vocab_gap:physical_base:heat_flux",
        "id": "vocab_gap:physical_base:heat_flux",
        "segment": "physical_base",
        "token": "heat_flux",
        "editorial_disposition": None,
        "editorial_target": None,
        "editorial_reason": None,
        "editorial_actor": None,
        "editorial_grammar_signature": None,
        "editorial_grammar_version": None,
        "editorial_active": None,
    }
    row.update(overrides)
    return row


def _resolution_batch():
    from imas_codex.standard_names.vocab_adjudication import (
        VocabGapAdjudicationFile,
    )

    decisions = [
        {
            "segment": "physical_base",
            "token": f"live_token_{index}",
            "decision": "reject",
            "rationale": "This observed term is not grammar vocabulary.",
        }
        for index in range(140)
    ]
    decisions.extend(
        {
            "segment": "physical_base",
            "token": f"registered_add_{index}",
            "decision": "add",
            "canonical_target": f"registered_add_{index}",
            "rationale": "The grammar now registers this reviewed addition.",
        }
        for index in range(18)
    )
    decisions.append(
        {
            "segment": "position",
            "token": "retired_reject",
            "decision": "reject",
            "rationale": "The grammar no longer carries this rejected term.",
        }
    )
    return VocabGapAdjudicationFile.model_validate({"decisions": decisions})


def _resolution_graph_rows(batch) -> list[dict]:
    rows = []
    for index, decision in enumerate(batch.decisions):
        rows.append(
            {
                "requested_id": decision.gap_id,
                "id": decision.gap_id if index < 140 else None,
                "segment": decision.segment if index < 140 else None,
                "token": decision.token if index < 140 else None,
                "editorial_disposition": None,
                "editorial_target": None,
                "editorial_reason": None,
                "editorial_actor": None,
                "editorial_grammar_signature": None,
                "editorial_grammar_version": None,
                "editorial_active": None,
            }
        )
    return rows


def _resolution_context(
    *,
    extra_tokens: list[str] | None = None,
    aliases=None,
    vocabularies=None,
):
    tokens = [f"registered_add_{index}" for index in range(18)]
    tokens.extend(extra_tokens or [])
    return {
        "vocabulary_sections": [
            {"segment": "physical_base", "tokens": tokens},
            {"segment": "position", "tokens": ["center"]},
        ],
        "grammar": {
            "advisory_aliases": aliases or {},
            "vocabularies": vocabularies or {},
        },
    }


def test_schema_separates_editorial_state_from_observation_state() -> None:
    schema_path = Path(__file__).parents[2] / "imas_codex/schemas/standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text())
    gap = schema["classes"]["VocabGap"]["attributes"]
    evidence = schema["classes"]["VocabGapEvidence"]["attributes"]

    assert gap["editorial_disposition"]["range"] == "VocabGapDisposition"
    assert gap["editorial_grammar_signature"]["description"]
    assert gap["editorial_active"]["range"] == "boolean"
    assert gap["evidence"]["range"] == "VocabGapEvidence"
    assert gap["evidence"]["annotations"]["relationship_type"] == "HAS_EVIDENCE"
    assert evidence["observed_at"]["range"] == "datetime"
    assert evidence["vocab_gap_id"]["required"] is True


def test_reviewed_artifact_validates_all_rows_and_counts() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        load_vocab_gap_adjudications,
    )

    batch = load_vocab_gap_adjudications(_artifact_path())
    assert len(batch.decisions) == 159
    assert batch.disposition_counts() == {"add": 18, "fold": 27, "reject": 114}
    assert len({row.gap_id for row in batch.decisions}) == 159


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rationale", "", "rationale"),
        ("decision", "defer", "Input should be"),
        ("canonical_target", None, "canonical_target"),
    ],
)
def test_invalid_editorial_rows_are_rejected(field: str, value, message: str) -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        VocabGapAdjudicationFile,
    )

    row = {
        "segment": "physical_base",
        "token": "heat_flux",
        "decision": "fold",
        "canonical_target": "heat_flux",
        "rationale": "Use registered grammar.",
    }
    row[field] = value
    with pytest.raises(ValidationError, match=message):
        VocabGapAdjudicationFile.model_validate({"decisions": [row]})


def test_duplicate_gap_identity_is_rejected() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        VocabGapAdjudicationFile,
    )

    row = {
        "segment": "physical_base",
        "token": "heat_flux",
        "decision": "fold",
        "canonical_target": "heat_flux",
        "rationale": "Use registered grammar.",
    }
    with pytest.raises(ValidationError, match="exactly one decision"):
        VocabGapAdjudicationFile.model_validate({"decisions": [row, row]})


def test_missing_graph_node_aborts_before_any_write() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    client = _Client(
        [
            {
                "requested_id": "vocab_gap:physical_base:heat_flux",
                "id": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="missing VocabGap"):
        apply_vocab_gap_adjudications(
            _single_batch(),
            actor="catalog review",
            dry_run=False,
            grammar_signature="abc",
            grammar_version="1.0",
            gc=client,
        )

    assert client.tx.closed() is True
    assert client.tx.committed is False
    assert len(client.tx.calls) == 1


def test_apply_uses_one_transaction_and_dry_run_never_writes() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    dry_client = _Client([_existing_row()])
    result = apply_vocab_gap_adjudications(
        _single_batch(),
        actor="catalog review",
        grammar_signature="abc",
        grammar_version="1.0",
        gc=dry_client,
    )
    assert result["changed"] == 1
    assert result["dry_run"] is True
    assert len(dry_client.tx.calls) == 1
    assert dry_client.tx.committed is False

    write_client = _Client([_existing_row()])
    result = apply_vocab_gap_adjudications(
        _single_batch(),
        actor="catalog review",
        dry_run=False,
        grammar_signature="abc",
        grammar_version="1.0",
        gc=write_client,
    )
    assert result["changed"] == 1
    assert write_client.tx.committed is True
    assert len(write_client.tx.calls) == 2
    write_query, params = write_client.tx.calls[1]
    assert "SET vg.editorial_disposition" in write_query
    assert params["items"][0]["disposition"] == "fold"


def test_apply_aborts_when_an_extant_node_is_not_written() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    client = _Client([_existing_row()], applied_count=0)
    with pytest.raises(ValueError, match="did not apply every extant"):
        apply_vocab_gap_adjudications(
            _single_batch(),
            actor="catalog review",
            dry_run=False,
            grammar_signature="abc",
            grammar_version="1.0",
            gc=client,
        )
    assert client.tx.committed is False
    assert client.tx.closed() is True


def test_second_apply_is_idempotent() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    client = _Client(
        [
            _existing_row(
                editorial_disposition="fold",
                editorial_target="heat_flux",
                editorial_reason="Express the quantity with registered grammar.",
                editorial_actor="catalog review",
                editorial_grammar_signature="abc",
                editorial_grammar_version="1.0",
                editorial_active=True,
            )
        ]
    )
    result = apply_vocab_gap_adjudications(
        _single_batch(),
        actor="catalog review",
        dry_run=False,
        grammar_signature="abc",
        grammar_version="1.0",
        gc=client,
    )

    assert result["changed"] == 0
    assert result["unchanged"] == 1
    assert len(client.tx.calls) == 1
    assert client.tx.committed is True


def test_resolved_history_classifies_complete_batch_and_writes_receipt(
    tmp_path: Path,
) -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    batch = _resolution_batch()
    rows = _resolution_graph_rows(batch)
    token_rows = [
        {"requested_id": row["requested_id"], "graph_token_ids": []}
        for row in rows
        if row["id"] is None
    ]
    receipt_path = tmp_path / "adjudication-receipt.json"
    client = _Client(rows, token_rows)

    result = apply_vocab_gap_adjudications(
        batch,
        actor="catalog review",
        grammar_signature="abc",
        grammar_version="1.0",
        resolve_missing_from_grammar=True,
        grammar_context=_resolution_context(),
        receipt_path=receipt_path,
        gc=client,
    )

    assert result["resolution_counts"] == {
        "applied": 140,
        "satisfied_by_grammar": 18,
        "resolved_reject": 1,
    }
    assert len(result["receipt"]["decisions"]) == 159
    assert result["receipt"]["grammar_signature"] == "abc"
    assert result["receipt"]["actor"] == "catalog review"
    assert result["receipt"]["dry_run"] is True
    assert json.loads(receipt_path.read_text()) == result["receipt"]
    assert len(client.tx.calls) == 2
    assert client.tx.committed is False

    apply_receipt_path = tmp_path / "applied-receipt.json"
    apply_client = _Client(rows, token_rows)
    applied = apply_vocab_gap_adjudications(
        batch,
        actor="catalog review",
        dry_run=False,
        grammar_signature="abc",
        grammar_version="1.0",
        resolve_missing_from_grammar=True,
        grammar_context=_resolution_context(),
        receipt_path=apply_receipt_path,
        gc=apply_client,
    )

    assert applied["resolution_counts"] == result["resolution_counts"]
    assert applied["receipt"]["dry_run"] is False
    assert len(apply_client.tx.calls) == 3
    write_query, params = apply_client.tx.calls[2]
    assert "SET vg.editorial_disposition" in write_query
    assert len(params["items"]) == 140
    assert apply_client.tx.committed is True


def test_resolved_history_requires_an_explicit_receipt_path() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    batch = _single_batch(decision="add")
    client = _Client(
        [{"requested_id": batch.decisions[0].gap_id, "id": None}],
        [{"requested_id": batch.decisions[0].gap_id, "graph_token_ids": []}],
    )
    with pytest.raises(ValueError, match="receipt path"):
        apply_vocab_gap_adjudications(
            batch,
            actor="catalog review",
            grammar_signature="abc",
            grammar_version="1.0",
            resolve_missing_from_grammar=True,
            grammar_context={
                "vocabulary_sections": [
                    {"segment": "physical_base", "tokens": ["heat_flux"]}
                ]
            },
            gc=client,
        )
    assert client.tx.calls == []


@pytest.mark.parametrize(
    ("batch", "context", "token_ids", "message"),
    [
        (
            _single_batch(decision="add", target="heat_flux"),
            _resolution_context(),
            [],
            "add target is not registered",
        ),
        (
            _single_batch(decision="reject", target=None),
            _resolution_context(extra_tokens=["heat_flux"]),
            [],
            "reject token remains registered",
        ),
        (
            _single_batch(decision="reject", target=None),
            _resolution_context(vocabularies={"operators": {"heat_flux": {}}}),
            [],
            "reject token remains registered",
        ),
        (
            _single_batch(decision="reject", target=None),
            _resolution_context(),
            ["vocab_gap:subject:heat_flux"],
            "reject token remains in the graph",
        ),
        (
            _single_batch(decision="fold", target="heat_flux"),
            _resolution_context(extra_tokens=["heat_flux"]),
            [],
            "fold has no exact advisory alias",
        ),
        (
            _single_batch(decision="fold", target="heat_flux"),
            _resolution_context(
                extra_tokens=["heat_flux", "thermal_flux"],
                aliases={
                    "physical_base": {
                        "heat_flux": {
                            "canonical": "thermal_flux",
                            "reason": "Use the grammar-owned spelling.",
                        }
                    }
                },
            ),
            [],
            "fold advisory alias targets",
        ),
    ],
)
def test_resolved_history_fails_closed_on_grammar_or_graph_mismatch(
    batch,
    context,
    token_ids,
    message: str,
    tmp_path: Path,
) -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    gap_id = batch.decisions[0].gap_id
    client = _Client(
        [{"requested_id": gap_id, "id": None}],
        [{"requested_id": gap_id, "graph_token_ids": token_ids}],
    )
    with pytest.raises(ValueError, match=message):
        apply_vocab_gap_adjudications(
            batch,
            actor="catalog review",
            dry_run=False,
            grammar_signature="abc",
            grammar_version="1.0",
            resolve_missing_from_grammar=True,
            grammar_context=context,
            receipt_path=tmp_path / "receipt.json",
            gc=client,
        )
    assert client.tx.committed is False
    assert len(client.tx.calls) == 2


def test_missing_fold_is_satisfied_only_by_matching_segment_alias(
    tmp_path: Path,
) -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    batch = _single_batch(decision="fold", target="thermal_flux")
    gap_id = batch.decisions[0].gap_id
    client = _Client(
        [{"requested_id": gap_id, "id": None}],
        [{"requested_id": gap_id, "graph_token_ids": []}],
    )
    result = apply_vocab_gap_adjudications(
        batch,
        actor="catalog review",
        grammar_signature="abc",
        grammar_version="1.0",
        resolve_missing_from_grammar=True,
        grammar_context=_resolution_context(
            extra_tokens=["thermal_flux"],
            aliases={
                "physical_base": {
                    "heat_flux": {
                        "canonical": "thermal_flux",
                        "reason": "Use the grammar-owned spelling.",
                    }
                }
            },
        ),
        receipt_path=tmp_path / "receipt.json",
        gc=client,
    )
    assert result["resolution_counts"]["satisfied_by_grammar"] == 1


@pytest.mark.parametrize(
    "token",
    [
        "diagnostic",
        "engineering",
        "geometry",
        "normalized",
        "reaction_channel",
        "species",
        "temporal",
        "transport",
    ],
)
def test_missing_reject_ignores_grammar_vocabulary_metadata_keys(
    token: str,
    tmp_path: Path,
) -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        apply_vocab_gap_adjudications,
    )

    batch = _single_batch(decision="reject", target=None, token=token)
    gap_id = batch.decisions[0].gap_id
    client = _Client(
        [{"requested_id": gap_id, "id": None}],
        [{"requested_id": gap_id, "graph_token_ids": []}],
    )
    result = apply_vocab_gap_adjudications(
        batch,
        actor="catalog review",
        grammar_signature="abc",
        grammar_version="1.0",
        resolve_missing_from_grammar=True,
        grammar_context=_resolution_context(
            vocabularies={
                "operators": {"gradient": {}},
                "qualifier_categories": {
                    token: ["metadata category"],
                },
            }
        ),
        receipt_path=tmp_path / "receipt.json",
        gc=client,
    )

    assert result["resolution_counts"]["resolved_reject"] == 1


def test_observation_write_preserves_editorial_and_dedup_state() -> None:
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = []
    gaps = [
        {
            "source_id": "path/a",
            "segment": "physical_base",
            "token": "unregistered_quantity",
            "reason": "source requires this quantity",
            "dedup_decision": "unchecked",
        }
    ]
    with (
        patch.object(graph_ops, "GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.segments.is_valid_segment",
            return_value=True,
        ),
        patch(
            "imas_codex.standard_names.segments.classify_gap",
            return_value=("absent", []),
        ),
    ):
        graph_ops.write_vocab_gaps(gaps)

    merge_call = next(
        call for call in gc.query.call_args_list if "MERGE (vg:VocabGap" in call.args[0]
    )
    query = merge_call.args[0]
    assert "editorial_" not in query
    assert "settled_dedup_decisions" in query
    assert "checked_no_reuse" in merge_call.kwargs["settled_dedup_decisions"]

    evidence_call = next(
        call
        for call in gc.query.call_args_list
        if "CREATE (e:VocabGapEvidence" in call.args[0]
    )
    evidence = evidence_call.kwargs["batch"][0]
    assert evidence["reason"] == "source requires this quantity"
    assert evidence["gap_id"] == "vocab_gap:physical_base:unregistered_quantity"


def test_compose_stamp_never_erases_a_settled_decision() -> None:
    from imas_codex.standard_names.workers import _stamp_dedup_decision

    gaps = [
        {
            "segment": "physical_base",
            "token": "quantity",
            "dedup_decision": "reuse_confirmed",
        }
    ]
    _stamp_dedup_decision(gaps, {})
    assert gaps[0]["dedup_decision"] == "reuse_confirmed"


def test_batch_observations_cannot_downgrade_mechanical_reuse() -> None:
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = []
    gaps = [
        {
            "source_id": "path/a",
            "segment": "physical_base",
            "token": "unregistered_quantity",
            "reason": "mechanically resolved",
            "dedup_decision": "reuse_confirmed",
        },
        {
            "source_id": "path/b",
            "segment": "physical_base",
            "token": "unregistered_quantity",
            "reason": "later unchecked observation",
            "dedup_decision": "unchecked",
        },
    ]
    with (
        patch.object(graph_ops, "GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.segments.is_valid_segment",
            return_value=True,
        ),
        patch(
            "imas_codex.standard_names.segments.classify_gap",
            return_value=("absent", []),
        ),
    ):
        graph_ops.write_vocab_gaps(gaps)

    merge_call = next(
        call for call in gc.query.call_args_list if "MERGE (vg:VocabGap" in call.args[0]
    )
    assert merge_call.kwargs["batch"][0]["dedup_decision"] == "reuse_confirmed"


def test_editorial_lookup_includes_an_observed_empty_token() -> None:
    from imas_codex.standard_names.graph_ops import (
        fetch_vocab_gap_adjudications,
    )

    gc = MagicMock()
    gc.query.return_value = []
    fetch_vocab_gap_adjudications(
        [{"segment": "position", "token": ""}],
        gc=gc,
    )
    assert gc.query.call_args.kwargs["ids"] == ["vocab_gap:position:"]


def test_editorial_guidance_routes_fold_reject_and_add_honestly() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        editorial_retry_guidance,
    )

    text, retry = editorial_retry_guidance(
        [
            {
                "segment": "physical_base",
                "token": "heat_flux",
                "disposition": "fold",
                "target": "heat_flux",
                "reason": "Use registered channel and base tokens.",
            },
            {
                "segment": "qualifier",
                "token": "storage_detail",
                "disposition": "reject",
                "target": None,
                "reason": "This detail belongs in metadata.",
            },
            {
                "segment": "device",
                "token": "detector",
                "disposition": "add",
                "target": "detector",
                "reason": "This is reusable vocabulary.",
            },
        ]
    )

    assert retry is True
    assert "canonical `heat_flux`" in text
    assert "Do not request or add this token" in text
    assert "remains unavailable until the installed grammar registers it" in text


def test_explicit_signature_reset_deactivates_without_erasing_decision() -> None:
    from imas_codex.standard_names.vocab_adjudication import (
        reset_vocab_gap_adjudications,
    )

    gc = MagicMock()
    gc.query.side_effect = [[{"eligible": 4}], [{"reset": 4}]]
    result = reset_vocab_gap_adjudications(
        "old-signature",
        actor="catalog review",
        reason="grammar structure changed",
        dry_run=False,
        current_grammar_signature="new-signature",
        gc=gc,
    )

    assert result["reset"] == 4
    reset_query = gc.query.call_args_list[1].args[0]
    assert "vg.editorial_active = false" in reset_query
    assert "vg.editorial_disposition = null" not in reset_query
    assert "vg.editorial_reason = null" not in reset_query


def test_active_editorial_decisions_project_into_triage() -> None:
    from imas_codex.standard_names.graph_ops import triage_vocab_gaps

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "vocab_gap:physical_base:heat_flux",
            "segment": "physical_base",
            "token": "heat_flux",
            "category": "absent",
            "dedup": "unchecked",
            "editorial_disposition": "fold",
            "last_seen": "2026-07-30",
            "n": 3,
        }
    ]
    result = triage_vocab_gaps(gc, stale_before="2026-07-01", dry_run=True)
    assert result["counts"]["fold"] == 1
    assert result["counts"]["genuine"] == 0


def test_artifact_has_no_implicit_production_path() -> None:
    """The reviewed JSON is operator input, never a module-level data dependency."""
    module_path = (
        Path(__file__).parents[2] / "imas_codex/standard_names/vocab_adjudication.py"
    )
    source = module_path.read_text()
    assert "docs/evidence" not in source
    assert json.loads(_artifact_path().read_text())["count"] == 159
