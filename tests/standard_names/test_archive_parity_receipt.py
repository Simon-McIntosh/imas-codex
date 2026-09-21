"""Archive-reconstruction receipts distinguish established expectations.

The refusal remains governed by the reconstruction closure. The receipt names
whether each expected count came from the archive census, the reconstruction
closure, or neither, and still reports the live count read inside the applied
transaction when no expectation was established.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imas_codex.standard_names import signed_manifest
from standard_names.test_archive_reconstruction import (
    _apply,
    _ArchiveGraph,
    _authority,
    _preview,
    _write_authority,
)


def _apply_authority(
    tmp_path: Path,
    *,
    archive_roles: dict[str, dict[str, int]],
    extra_counts: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(archive_roles=archive_roles)
    )
    graph = _ArchiveGraph()
    graph.extra_counts.update(extra_counts or {})
    preview = _preview(graph, path, file_hash, payload_hash)
    assert preview["outcome"] == "would_apply", preview["refusals"]
    return _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])


def test_parity_rows_name_all_three_expected_count_sources(tmp_path: Path) -> None:
    """Census, closure, and unestablished expectations stay distinguishable."""
    receipt = _apply_authority(
        tmp_path,
        archive_roles={"archived_temperature": {"EVIDENCED_BY": 2}},
    )
    parity = receipt["identity_role_parity"]["archived_temperature"]

    assert set(parity) == set(signed_manifest._ARCHIVE_PARITY_ROLES)
    assert parity["EVIDENCED_BY"] == {
        "expected": 2,
        "expected_source": "archive_census",
        "observed": 0,
    }
    assert parity["HAS_UNIT"] == {
        "expected": 1,
        "expected_source": "reconstruction_closure",
        "observed": 1,
    }
    assert parity["HAS_PARENT"] == {
        "expected": None,
        "expected_source": "neither",
        "observed": 0,
    }
    assert all(
        set(row) == {"expected", "expected_source", "observed"}
        for row in parity.values()
    )


def test_applied_receipt_reads_live_count_without_an_expected_source(
    tmp_path: Path,
) -> None:
    """An unestablished expectation does not suppress the transaction's read."""
    receipt = _apply_authority(
        tmp_path,
        archive_roles={},
        extra_counts={("archived_temperature", "EVIDENCED_BY"): 1},
    )

    assert receipt["identity_role_parity"]["archived_temperature"]["EVIDENCED_BY"] == {
        "expected": None,
        "expected_source": "neither",
        "observed": 1,
    }
