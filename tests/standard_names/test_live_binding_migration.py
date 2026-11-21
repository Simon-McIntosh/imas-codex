"""Live-binding compare-and-set coverage for source migrations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iter_codex.standard_names.provenance_lifecycle import (
    reset_standard_name_sources,
    retarget_standard_name_sources,
)


def _binding(name: str, stage: str) -> dict[str, object]:
    return {
        "id": name,
        "name_stage": stage,
        "catalog_pr_number": None,
        "origin": "pipeline",
        "other_sources": 0,
    }


def _migration_row(
    bindings: list[dict[str, object]],
    *,
    source_id: str = "dd:example/path",
    status: str = "composed",
    scalar: str | None = "old_name",
    claimed: bool = False,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_exists": True,
        "source_status": status,
        "scalar_binding": scalar,
        "actively_claimed": claimed,
        "current_bindings": [entry["id"] for entry in bindings],
        "binding_state": bindings,
        "manifest_recorded": False,
    }


def _reset_row(
    bindings: list[dict[str, object]],
    *,
    status: str = "composed",
    scalar: str | None = "old_name",
    claimed: bool = False,
) -> dict[str, object]:
    return {
        "source_id": "dd:example/path",
        "source_exists": True,
        "status": status,
        "scalar": scalar,
        "actively_claimed": claimed,
        "binding_state": bindings,
        "event_exists": False,
        "event_reason": None,
    }


def _reset_manifest() -> list[dict[str, object]]:
    return [
        {
            "source_id": "dd:example/path",
            "expected_status": "composed",
            "expected_scalar": "old_name",
            "expected_bindings": ["old_name"],
        }
    ]


def test_real_source_with_retired_provenance_is_pending_for_retarget() -> None:
    source_id = "dd:hard_x_rays/emissivity_profile_1d/half_width_internal"
    live_name = (
        "inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width"
    )
    bindings = [
        _binding(live_name, "drafted"),
        _binding("lower_bound_hard_xray_peak_width", "superseded"),
        _binding("normalized_toroidal_hard_xray_peak_lower_bound_width", "exhausted"),
    ]
    gc = MagicMock()
    gc.query.side_effect = [
        [
            _migration_row(
                bindings,
                source_id=source_id,
                status="attached",
                scalar=live_name,
            )
        ],
        [{"moved": 1}],
    ]

    moved = retarget_standard_name_sources(
        gc,
        live_name,
        "replacement_name",
        source_ids=[source_id],
        expected_current_bindings={source_id: live_name},
        record_change=False,
        enforce_consistency=False,
    )

    assert moved == 1
    assert gc.query.call_count == 2
    preflight_cypher = gc.query.call_args_list[0].args[0]
    assert "name_stage: target.name_stage" in preflight_cypher


def test_source_reset_ignores_and_preserves_retired_bindings() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        _reset_row(
            [
                _binding("old_name", "reviewed"),
                _binding("retired_name", "superseded"),
            ]
        )
    ]

    result = reset_standard_name_sources(
        gc,
        _reset_manifest(),
        manifest_id="source-reset",
        reason="recompose the live binding",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["applied"] == 0
    assert gc.query.call_count == 1


@pytest.mark.parametrize(
    "row",
    [
        _migration_row(
            [
                _binding("old_name", "drafted"),
                _binding("other_live_name", "reviewed"),
            ]
        ),
        _migration_row([]),
        _migration_row(
            [
                _binding("wrong_live_name", "drafted"),
                _binding("old_name", "exhausted"),
            ]
        ),
        _migration_row([_binding("old_name", "drafted")], status="stale"),
        _migration_row([_binding("old_name", "drafted")], claimed=True),
    ],
    ids=[
        "two-live-bindings",
        "zero-live-bindings",
        "wrong-live-binding",
        "stale-source",
        "active-claim",
    ],
)
def test_retarget_refuses_ambiguous_or_ineligible_live_state(
    row: dict[str, object],
) -> None:
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(RuntimeError, match="source migration compare-and-set failed"):
        retarget_standard_name_sources(
            gc,
            "old_name",
            "replacement_name",
            source_ids=["dd:example/path"],
            expected_current_bindings={"dd:example/path": "old_name"},
            record_change=False,
            enforce_consistency=False,
        )

    assert gc.query.call_count == 1


def test_retarget_scalar_mirror_must_still_match_the_live_name() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        _migration_row(
            [
                _binding("old_name", "drafted"),
                _binding("retired_name", "superseded"),
            ],
            scalar="retired_name",
        )
    ]

    with pytest.raises(RuntimeError, match="source migration compare-and-set failed"):
        retarget_standard_name_sources(
            gc,
            "old_name",
            "replacement_name",
            source_ids=["dd:example/path"],
            expected_current_bindings={"dd:example/path": "old_name"},
            record_change=False,
            enforce_consistency=False,
        )

    assert gc.query.call_count == 1
