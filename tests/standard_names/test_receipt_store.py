"""Durability contracts for Standard Names operation receipts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.standard_names.receipt_store import (
    ReceiptPersistenceError,
    canonical_receipt_bytes,
    persist_receipt,
)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_default_receipt_is_content_addressed_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    payload = {"z": [3, 2, 1], "a": {"ok": True}}

    stored = persist_receipt(
        "protected-structural-reconciliation",
        payload,
    )

    expected_bytes = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    expected_root = data_home / "imas-codex/receipts/standard-names"
    assert stored.path == (
        expected_root / "protected-structural-reconciliation" / f"{expected_hash}.json"
    )
    assert stored.sha256 == expected_hash
    assert stored.path.read_bytes() == expected_bytes
    assert _mode(expected_root) == 0o700
    assert _mode(stored.path.parent) == 0o700
    assert _mode(stored.path) == 0o600


def test_canonical_receipt_bytes_are_order_independent() -> None:
    first = canonical_receipt_bytes({"a": 1, "b": {"x": 2}})
    second = canonical_receipt_bytes({"b": {"x": 2}, "a": 1})

    assert first == second == b'{"a":1,"b":{"x":2}}\n'


def test_default_receipt_falls_back_to_project_user_data_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    with patch(
        "imas_codex.standard_names.receipt_store.Path.home",
        return_value=tmp_path,
    ):
        stored = persist_receipt("protected-structural-reconciliation", {"ok": True})

    assert stored.path.is_relative_to(
        tmp_path / ".local/share/imas-codex/receipts/standard-names"
    )


def test_identical_receipt_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    payload = {"mode": "dry_run", "counts": {"planned": 2}}
    first = persist_receipt("protected-structural-reconciliation", payload)

    with patch("imas_codex.standard_names.receipt_store.os.link") as link:
        second = persist_receipt("protected-structural-reconciliation", payload)

    assert second == first
    link.assert_not_called()
    assert first.path.read_bytes() == canonical_receipt_bytes(payload)


def test_identical_default_receipt_with_public_mode_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    payload = {"mode": "dry_run"}
    encoded = canonical_receipt_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    output = (
        tmp_path
        / "imas-codex/receipts/standard-names/protected-structural-reconciliation"
        / f"{digest}.json"
    )
    output.parent.mkdir(parents=True)
    output.write_bytes(encoded)
    output.chmod(0o644)

    with pytest.raises(ReceiptPersistenceError, match="mode 0600"):
        persist_receipt("protected-structural-reconciliation", payload)

    assert output.read_bytes() == encoded
    assert _mode(output) == 0o644


def test_identical_explicit_receipt_with_public_mode_fails_closed(
    tmp_path: Path,
) -> None:
    payload = {"mode": "dry_run"}
    encoded = canonical_receipt_bytes(payload)
    backing = tmp_path / "shared.json"
    backing.write_bytes(encoded)
    backing.chmod(0o644)
    output = tmp_path / "receipt.json"
    output.hardlink_to(backing)

    with pytest.raises(ReceiptPersistenceError, match="mode 0600"):
        persist_receipt(
            "protected-structural-reconciliation",
            payload,
            output_path=output,
        )

    assert output.read_bytes() == encoded
    assert _mode(backing) == 0o644
    assert _mode(output) == 0o644


def test_existing_mismatched_content_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    payload = {"mode": "dry_run"}
    encoded = canonical_receipt_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    output = (
        tmp_path
        / "imas-codex/receipts/standard-names/protected-structural-reconciliation"
        / f"{digest}.json"
    )
    output.parent.mkdir(parents=True)
    output.write_text("different\n")

    with pytest.raises(ReceiptPersistenceError, match="different content"):
        persist_receipt("protected-structural-reconciliation", payload)

    assert output.read_text() == "different\n"


def test_file_and_parent_directory_are_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    with patch(
        "imas_codex.standard_names.receipt_store.os.fsync",
        wraps=os.fsync,
    ) as fsync:
        persist_receipt("protected-structural-reconciliation", {"ok": True})

    assert fsync.call_count == 2


def test_atomic_install_failure_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    operation_dir = (
        tmp_path
        / "imas-codex/receipts/standard-names/protected-structural-reconciliation"
    )

    with (
        patch(
            "imas_codex.standard_names.receipt_store.os.link",
            side_effect=OSError("link refused"),
        ),
        pytest.raises(ReceiptPersistenceError, match="link refused"),
    ):
        persist_receipt("protected-structural-reconciliation", {"ok": True})

    assert list(operation_dir.glob(".*.tmp")) == []
    assert list(operation_dir.glob("*.json")) == []


def test_identical_race_winner_is_idempotent_and_temp_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    payload = {"mode": "dry_run"}
    expected = canonical_receipt_bytes(payload)

    def win_race(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        destination_path = Path(destination)
        destination_path.write_bytes(expected)
        destination_path.chmod(0o600)
        raise FileExistsError(errno.EEXIST, "already installed", destination)

    with patch(
        "imas_codex.standard_names.receipt_store.os.link",
        side_effect=win_race,
    ):
        stored = persist_receipt("protected-structural-reconciliation", payload)

    assert stored.path.read_bytes() == expected
    assert list(stored.path.parent.glob(".*.tmp")) == []


def test_mismatched_race_winner_fails_closed_and_temp_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    def win_race(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(destination).write_bytes(b"different\n")
        raise FileExistsError(errno.EEXIST, "already installed", destination)

    with (
        patch(
            "imas_codex.standard_names.receipt_store.os.link",
            side_effect=win_race,
        ),
        pytest.raises(ReceiptPersistenceError, match="different content"),
    ):
        persist_receipt("protected-structural-reconciliation", {"mode": "dry_run"})

    operation_dir = (
        tmp_path
        / "imas-codex/receipts/standard-names/protected-structural-reconciliation"
    )
    assert list(operation_dir.glob(".*.tmp")) == []


def test_explicit_path_cannot_alias_protected_input(tmp_path: Path) -> None:
    protected = tmp_path / "manifest.json"
    protected.write_text("manifest\n")
    alias = tmp_path / "receipt.json"
    alias.hardlink_to(protected)

    with pytest.raises(ReceiptPersistenceError, match="input artifact"):
        persist_receipt(
            "protected-structural-reconciliation",
            {"ok": True},
            output_path=alias,
            protected_inputs=(protected,),
        )

    assert protected.read_text() == "manifest\n"
