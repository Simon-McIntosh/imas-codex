"""Tests for the sn-edit steering schema fields on StandardName.

Covers the LinkML-generated enums (EditMode, EditOrigin, EditStatus,
EditScope) and the scalar edit-steering slots on StandardName
(edit_mode, name_hint, docs_hint, edit_reason, edit_origin, edit_status,
edit_scope, edit_requested_at) added to support `imas-codex sn edit`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestEditEnums:
    """Verify the four sn-edit enums exist with the expected members."""

    def test_edit_mode_members(self) -> None:
        from imas_codex.graph.models import EditMode

        assert EditMode.hint == "hint"
        assert EditMode.rename == "rename"
        assert EditMode.docs == "docs"
        assert len(EditMode) == 3

    def test_edit_origin_members(self) -> None:
        from imas_codex.graph.models import EditOrigin

        assert EditOrigin.human == "human"
        assert EditOrigin.agent == "agent"
        assert len(EditOrigin) == 2

    def test_edit_status_members(self) -> None:
        from imas_codex.graph.models import EditStatus

        assert EditStatus.open == "open"
        assert EditStatus.applied == "applied"
        assert EditStatus.exhausted == "exhausted"
        assert EditStatus.rejected == "rejected"
        assert len(EditStatus) == 4

    def test_edit_scope_members(self) -> None:
        """`self` collides with jsonasobj2's ExtendedNamespace kwarg — the
        member is named `only_self` instead (see EditScope description in
        the schema for the full rationale)."""
        from imas_codex.graph.models import EditScope

        assert EditScope.only_self == "only_self"
        assert EditScope.family == "family"
        assert EditScope.subtree == "subtree"
        assert len(EditScope) == 3


class TestStandardNameEditFields:
    """Verify StandardName accepts and round-trips the edit-steering slots."""

    def test_fields_present_on_model(self) -> None:
        from imas_codex.graph.models import StandardName

        fields = StandardName.model_fields
        for name in (
            "edit_mode",
            "name_hint",
            "docs_hint",
            "edit_reason",
            "edit_origin",
            "edit_status",
            "edit_scope",
            "edit_requested_at",
        ):
            assert name in fields, f"missing slot: {name}"

    def test_construct_with_edit_fields(self) -> None:
        from imas_codex.graph.models import StandardName

        sn = StandardName(
            id="electron_temperature",
            edit_mode="hint",
            name_hint="prefer a shorter subject qualifier",
            docs_hint=None,
            edit_reason="reviewer keeps reverting to the verbose form",
            edit_origin="agent",
            edit_status="open",
            edit_scope="only_self",
            edit_requested_at="2026-07-03T00:00:00Z",
        )
        assert sn.edit_mode == "hint"
        assert sn.name_hint == "prefer a shorter subject qualifier"
        assert sn.docs_hint is None
        assert sn.edit_reason == "reviewer keeps reverting to the verbose form"
        assert sn.edit_origin == "agent"
        assert sn.edit_status == "open"
        assert sn.edit_scope == "only_self"

    def test_model_dump_round_trips_edit_fields(self) -> None:
        from imas_codex.graph.models import StandardName

        sn = StandardName(
            id="ion_temperature",
            edit_mode="rename",
            name_hint="ion_temperature_core",
            edit_reason="disambiguate from edge measurement",
            edit_origin="human",
            edit_status="open",
            edit_scope="family",
        )
        dumped = sn.model_dump()
        restored = StandardName.model_validate(dumped)
        assert restored.edit_mode == sn.edit_mode
        assert restored.name_hint == sn.name_hint
        assert restored.edit_reason == sn.edit_reason
        assert restored.edit_origin == sn.edit_origin
        assert restored.edit_status == sn.edit_status
        assert restored.edit_scope == sn.edit_scope

    def test_edit_fields_default_to_none(self) -> None:
        """A StandardName with no edit attached leaves all edit fields null."""
        from imas_codex.graph.models import StandardName

        sn = StandardName(id="plasma_current")
        assert sn.edit_mode is None
        assert sn.name_hint is None
        assert sn.docs_hint is None
        assert sn.edit_reason is None
        assert sn.edit_origin is None
        assert sn.edit_status is None
        assert sn.edit_scope is None
        assert sn.edit_requested_at is None

    def test_invalid_edit_mode_rejected(self) -> None:
        from imas_codex.graph.models import StandardName

        with pytest.raises(ValidationError):
            StandardName(id="plasma_current", edit_mode="not_a_real_mode")

    def test_invalid_edit_scope_rejected(self) -> None:
        from imas_codex.graph.models import StandardName

        with pytest.raises(ValidationError):
            StandardName(id="plasma_current", edit_scope="self")
