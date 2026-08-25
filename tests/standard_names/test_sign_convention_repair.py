"""Fail-closed coverage for deterministic sign-convention document repair."""

from __future__ import annotations

import json
from typing import Any

import pytest

from imas_codex.standard_names.audits import (
    _strip_final_sign_convention_paragraph,
    repair_invariant_sign_convention_documents,
)


def _documentation(identity: str) -> str:
    return (
        f"{identity} is a deterministic quantity.\n\n"
        "Its definition and relationships remain unchanged.\n\n"
        f"Sign convention: Positive when {identity} increases."
    )


class _Transaction:
    def __init__(self, state: dict[str, dict[str, Any]]) -> None:
        self.state = state
        self.closed = False
        self.committed = False
        self.postconditions_json: str | None = None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "repair replay postconditions" in cypher:
            return [
                {
                    "requested_id": identity,
                    "documentation": self.state[identity]["documentation"],
                    "transformation_type": self.state[identity]["transformation_type"],
                    "cocos": self.state[identity]["cocos"],
                    "cocos_ids": self.state[identity]["cocos_ids"],
                }
                for identity in params["name_ids"]
            ]
        if "repair replay" in cypher:
            return [{"postconditions_json": self.postconditions_json}]
        if "repair authority" in cypher:
            return [self._authority_row(identity) for identity in params["name_ids"]]
        if "repair lock" in cypher:
            return [{"element_ids": list(params["element_ids"])}]
        if "documentation mutation" in cypher:
            for row in params["rows"]:
                assert (
                    self.state[row["id"]]["documentation"]
                    == row["documentation_before"]
                )
                self.state[row["id"]]["documentation"] = row["documentation_after"]
            return [{"ids": [row["id"] for row in params["rows"]]}]
        if "metadata mutation" in cypher:
            for row in params["rows"]:
                current = self.state[row["id"]]
                current["transformation_type"] = None
                current["cocos"] = None
                current["cocos_ids"] = []
                current["edge_element_id"] = None
            return [{"ids": [row["id"] for row in params["rows"]]}]
        if "repair postconditions" in cypher:
            return [
                {
                    "requested_id": identity,
                    "documentation": self.state[identity]["documentation"],
                    "transformation_type": self.state[identity]["transformation_type"],
                    "cocos": self.state[identity]["cocos"],
                    "cocos_ids": self.state[identity]["cocos_ids"],
                }
                for identity in params["name_ids"]
            ]
        if "durable receipt" in cypher:
            self.postconditions_json = params["postconditions_json"]
            assert set(json.loads(params["removed_suffixes_json"])) == set(
                params["name_ids"]
            )
            return [{"change_id": params["event_id"]}]
        raise AssertionError(f"unexpected query: {cypher}")

    def _authority_row(self, identity: str) -> dict[str, Any]:
        row = self.state[identity]
        edges = (
            [
                {
                    "element_id": row["edge_element_id"],
                    "cocos_id": row["cocos_ids"][0],
                }
            ]
            if row["cocos_ids"]
            else []
        )
        return {
            "requested_id": identity,
            "element_id": f"node:{identity}",
            "documentation": row["documentation"],
            "docs_stage": "accepted",
            "name_stage": "accepted",
            "transformation_type": row["transformation_type"],
            "cocos": row["cocos"],
            "has_claimed_at": False,
            "has_claim_token": False,
            "cocos_edges": edges,
        }

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def rollback(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def begin_transaction(self) -> _Transaction:
        self.transaction.closed = False
        return self.transaction


class _Client:
    def __init__(self, state: dict[str, dict[str, Any]]) -> None:
        self.transaction = _Transaction(state)

    def session(self) -> _Session:
        return _Session(self.transaction)


def _state(*identities: str) -> dict[str, dict[str, Any]]:
    return {
        identity: {
            "documentation": _documentation(identity),
            "transformation_type": "one_like",
            "cocos": 17,
            "cocos_ids": [17],
            "edge_element_id": f"edge:{identity}",
        }
        for identity in identities
    }


def test_strip_preserves_every_character_before_the_forbidden_paragraph() -> None:
    before = _documentation("density")
    kept, separator, removed = _strip_final_sign_convention_paragraph(before)

    assert before == kept + separator + removed
    assert kept == (
        "density is a deterministic quantity.\n\n"
        "Its definition and relationships remain unchanged."
    )
    assert separator == "\n\n"
    assert removed == "Sign convention: Positive when density increases."


@pytest.mark.parametrize(
    "documentation",
    [
        "Sign convention: positive.\n\nA later paragraph.",
        "Definition only, with no convention paragraph.",
        "Sign convention: positive.\n\nSign convention: negative.",
    ],
)
def test_strip_refuses_any_shape_other_than_one_final_paragraph(
    documentation: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly one final sign-convention paragraph",
    ):
        _strip_final_sign_convention_paragraph(documentation)


def test_preview_signs_exact_deltas_and_separates_metadata_clears() -> None:
    state = _state("document_only", "metadata_clear", "already_clean")
    state["already_clean"]["documentation"] = (
        "already_clean is a deterministic quantity.\n\n"
        "Its definition and relationships remain unchanged."
    )
    client = _Client(state)

    preview = repair_invariant_sign_convention_documents(
        ["metadata_clear", "document_only", "already_clean"],
        metadata_clear_ids=["metadata_clear"],
        excluded_regeneration_ids=["magnetic_field"],
        reason="remove unsupported convention prose",
        gc=client,
    )

    assert preview["outcome"] == "would_apply"
    assert preview["would_change"] == 2
    assert preview["counts"]["covered"] == 3
    assert preview["counts"]["already_clean"] == 1
    assert preview["counts"]["documents_only"] == 1
    assert preview["counts"]["metadata_to_clear"] == 1
    assert preview["counts"]["model_spend_usd"] == 0.0
    assert preview["excluded_regeneration_ids"] == ["magnetic_field"]
    assert all(action["prefix_preserved"] for action in preview["manifest"]["actions"])
    assert all(
        action["character_delta"]
        == action["removed_paragraph_chars"] + action["paragraph_separator_chars"]
        for action in preview["manifest"]["actions"]
    )


def test_apply_requires_preview_hash_and_excludes_regeneration_identity() -> None:
    with pytest.raises(ValueError, match="apply requires manifest_sha256"):
        repair_invariant_sign_convention_documents(
            ["document_only"],
            metadata_clear_ids=[],
            excluded_regeneration_ids=["magnetic_field"],
            reason="remove unsupported convention prose",
            apply=True,
            gc=_Client(_state("document_only")),
        )
    with pytest.raises(
        ValueError,
        match="regeneration-required identities cannot be repair targets",
    ):
        repair_invariant_sign_convention_documents(
            ["magnetic_field"],
            metadata_clear_ids=[],
            excluded_regeneration_ids=["magnetic_field"],
            reason="do not strip sensitive physics",
            gc=_Client(_state("magnetic_field")),
        )


def test_apply_is_atomic_audited_and_zero_model_cost() -> None:
    state = _state("document_only", "metadata_clear", "already_clean")
    state["already_clean"]["documentation"] = (
        "already_clean is a deterministic quantity.\n\n"
        "Its definition and relationships remain unchanged."
    )
    client = _Client(state)
    kwargs = {
        "name_ids": ["document_only", "metadata_clear", "already_clean"],
        "metadata_clear_ids": ["metadata_clear"],
        "excluded_regeneration_ids": ["magnetic_field"],
        "reason": "remove unsupported convention prose",
        "gc": client,
    }
    preview = repair_invariant_sign_convention_documents(**kwargs)

    applied = repair_invariant_sign_convention_documents(
        **kwargs,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        run_id="deterministic-repair",
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 2
    assert applied["counts"]["already_clean"] == 1
    assert applied["counts"]["model_spend_usd"] == 0.0
    assert client.transaction.committed
    assert "sign convention" not in state["document_only"]["documentation"].lower()
    assert state["document_only"]["transformation_type"] == "one_like"
    assert state["document_only"]["cocos_ids"] == [17]
    assert state["metadata_clear"]["transformation_type"] is None
    assert state["metadata_clear"]["cocos"] is None
    assert state["metadata_clear"]["cocos_ids"] == []
    assert applied["change_id"].endswith(preview["manifest_sha256"])

    replay = repair_invariant_sign_convention_documents(
        **kwargs,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["model_spend_usd"] == 0.0
