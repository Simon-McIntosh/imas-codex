"""Signed stale-source lifecycle detachment tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    _load_signed_stale_source_rows,
    detach_signed_stale_source_bindings,
)

AUTHORITY_PATH = (
    Path(__file__).parents[2]
    / "docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json"
)
BLOCKING_SOURCE_IDS = (
    "dd:neutron_diagnostic/detectors/aperture/centre/phi",
    "dd:neutron_diagnostic/detectors/detector/centre/phi",
    "dd:refractometer/channel/frequencies",
)


def _write_authority(tmp_path: Path, authority: dict[str, object]) -> Path:
    path = tmp_path / "stale-source-lifecycle.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path


def _resign_rows(authority: dict[str, object]) -> None:
    rows = authority["rows"]
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    authority["signature"]["digest"] = sha256((canonical + "\n").encode()).hexdigest()


def test_committed_authority_signature_selects_exact_blocking_rows() -> None:
    file_hash, rows_hash, rows = _load_signed_stale_source_rows(
        AUTHORITY_PATH, BLOCKING_SOURCE_IDS
    )

    assert (
        file_hash == "f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad"
    )
    assert (
        rows_hash == "316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198"
    )
    assert [row["source_id"] for row in rows] == sorted(BLOCKING_SOURCE_IDS)
    assert {row["disposition"] for row in rows} == {"detach"}
    assert {tuple(row["live_target_ids"]) for row in rows} == {
        ("frequency_of_diagnostic_antenna",),
        ("toroidal_angle_of_measurement_position",),
    }


def test_tampered_signed_rows_are_rejected(tmp_path: Path) -> None:
    authority = json.loads(AUTHORITY_PATH.read_text())
    authority["rows"][0]["scalar_target"] = "tampered"
    path = _write_authority(tmp_path, authority)

    with pytest.raises(ValueError, match="signature does not match"):
        _load_signed_stale_source_rows(path, [authority["rows"][0]["source_id"]])


def test_signed_non_detach_row_is_not_execution_authority(tmp_path: Path) -> None:
    authority = json.loads(AUTHORITY_PATH.read_text())
    selected = next(
        row for row in authority["rows"] if row["source_id"] == BLOCKING_SOURCE_IDS[0]
    )
    selected["disposition"] = "versioned_migration"
    _resign_rows(authority)
    path = _write_authority(tmp_path, authority)

    with pytest.raises(ValueError, match="lacks exact DD detach authority"):
        _load_signed_stale_source_rows(path, [BLOCKING_SOURCE_IDS[0]])


def test_source_outside_signed_authority_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside signed authority"):
        _load_signed_stale_source_rows(AUTHORITY_PATH, ["dd:not/signed"])


def test_apply_requires_a_preview_manifest_hash() -> None:
    gc = MagicMock()

    with pytest.raises(ValueError, match="requires manifest_sha256"):
        detach_signed_stale_source_bindings(
            gc,
            AUTHORITY_PATH,
            BLOCKING_SOURCE_IDS,
            reason="Detach sources removed from current DD authority.",
            apply=True,
        )

    gc.session.assert_not_called()
