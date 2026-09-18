"""The applied receipt records the per-role parity the guard already proves.

The reconstruction transaction builds an expected count per registry role,
compares it with the live count inside the transaction, and raises on any
difference. Before this receipt field existed the comparison was a local
variable: the receipt named which roles came back but never the role set they
were measured against, and a role with no reconstruction route appeared only in
a separate bucket rather than counted against the archive census that holds its
loss.

The harness is the archive reconstruction fixture reused from
``test_archive_reconstruction`` -- a fake transaction graph, an authority
builder and the preview/apply wrappers -- so this module measures the receipt
rather than carrying a second implementation of the transaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names import signed_manifest
from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    SignedManifestConflict,
)
from standard_names.test_archive_reconstruction import (
    _apply,
    _ArchiveGraph,
    _authority,
    _preview,
    _write_authority,
)


def _applied_receipt(
    tmp_path: Path, archive_roles: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Apply one reconstructable authority and return the applied receipt."""
    authority = _authority(archive_roles=archive_roles)
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, authority)
    graph = _ArchiveGraph()
    preview = _preview(graph, path, file_hash, payload_hash)
    assert preview["outcome"] == "would_apply", preview["refusals"]
    return _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])


def test_applied_receipt_counts_expected_and_observed_for_every_parity_role(
    tmp_path: Path,
) -> None:
    """Every parity role is present, its zeros explicit, on an applied restore."""
    receipt = _applied_receipt(tmp_path, {"archived_temperature": {"HAS_UNIT": 1}})

    assert receipt["outcome"] == "applied"
    parity = receipt["identity_role_parity"]["archived_temperature"]

    assert set(parity) == set(signed_manifest._ARCHIVE_PARITY_ROLES)
    assert parity["HAS_UNIT"] == {"expected": 1, "observed": 1}
    assert parity["HAS_PARENT"] == {"expected": 0, "observed": 0}
    for role, counts in parity.items():
        assert set(counts) == {"expected", "observed"}, role


def test_parity_counts_the_loss_of_a_role_with_no_reconstruction_route(
    tmp_path: Path,
) -> None:
    """A role no closure can carry is counted against the archive census."""
    receipt = _applied_receipt(
        tmp_path, {"archived_temperature": {"HAS_UNIT": 1, "EVIDENCED_BY": 2}}
    )

    assert "EVIDENCED_BY" in signed_manifest._ARCHIVE_PARITY_ROLES
    assert "EVIDENCED_BY" not in signed_manifest._ARCHIVE_EDGE_COUNTERPARTS

    parity = receipt["identity_role_parity"]["archived_temperature"]
    assert parity["EVIDENCED_BY"] == {"expected": 2, "observed": 0}

    roles = receipt["identity_roles"]["archived_temperature"]
    assert roles["unreinstatable"] == {"EVIDENCED_BY": 2}


def test_counting_a_role_does_not_admit_it_as_a_reconstruction_route(
    tmp_path: Path,
) -> None:
    """The counted role stays unroutable: an edge of it is still refused."""
    edge = {
        "owner_id": "archived_temperature",
        "relationship_type": "EVIDENCED_BY",
        "direction": "incoming",
        "counterpart_id": "candidate:1",
        "properties": {"source": "archive"},
    }
    authority = _authority(edges=[edge])
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, authority)

    with pytest.raises(
        SignedManifestAuthorityError, match="outside the reconstruction registry"
    ):
        _preview(_ArchiveGraph(), path, file_hash, payload_hash)


def test_parity_receipt_does_not_weaken_the_exact_count_guard(
    tmp_path: Path,
) -> None:
    """A live count that differs from the archive census still refuses."""
    authority = _authority(archive_roles={"archived_temperature": {"HAS_UNIT": 1}})
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, authority)
    graph = _ArchiveGraph()
    preview = _preview(graph, path, file_hash, payload_hash)
    graph.extra_counts[("archived_temperature", "HAS_UNIT")] = 1

    with pytest.raises(SignedManifestConflict, match="counts differ"):
        _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])
