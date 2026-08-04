"""Durable private storage for Standard Names operation receipts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OPERATION_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


class ReceiptPersistenceError(RuntimeError):
    """Raised when a receipt cannot be installed with durable semantics."""


@dataclass(frozen=True)
class StoredReceipt:
    """The installed path and content digest of one canonical receipt."""

    path: Path
    sha256: str


def canonical_receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode *payload* as deterministic newline-terminated UTF-8 JSON."""
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReceiptPersistenceError(f"receipt is not canonical JSON: {exc}") from exc
    return f"{encoded}\n".encode()


def _user_data_root() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".local/share"


def _receipt_root() -> Path:
    return _user_data_root() / "imas-codex" / "receipts" / "standard-names"


def _absolute_path(path: str | Path) -> Path:
    """Return an absolute path without following its final directory entry."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _ensure_directory(path: Path, *, enforce_private: bool) -> None:
    existed = path.exists()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if enforce_private or not existed:
        path.chmod(0o700)


def _protected_identities(
    protected_inputs: Sequence[str | Path],
) -> tuple[set[Path], set[tuple[int, int]]]:
    paths: set[Path] = set()
    identities: set[tuple[int, int]] = set()
    for protected in protected_inputs:
        protected_path = _absolute_path(protected)
        paths.add(protected_path)
        try:
            protected_stat = protected_path.stat()
        except OSError as exc:
            raise ReceiptPersistenceError(
                f"cannot verify protected input artifact {protected_path}: {exc}"
            ) from exc
        identities.add((protected_stat.st_dev, protected_stat.st_ino))
    return paths, identities


def _open_existing_receipt(
    path: Path,
    expected: bytes,
    protected_identities: set[tuple[int, int]],
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        path_stat = os.fstat(descriptor)
        if not stat.S_ISREG(path_stat.st_mode):
            raise ReceiptPersistenceError(
                f"receipt destination is not a regular file: {path}"
            )
        if (path_stat.st_dev, path_stat.st_ino) in protected_identities:
            raise ReceiptPersistenceError(
                f"receipt destination aliases an input artifact: {path}"
            )
        if path_stat.st_size != len(expected):
            raise ReceiptPersistenceError(
                f"receipt destination contains different content: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            actual = stream.read()
        if actual != expected:
            raise ReceiptPersistenceError(
                f"receipt destination contains different content: {path}"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_existing(
    path: Path,
    expected: bytes,
    protected_identities: set[tuple[int, int]],
) -> bool:
    try:
        descriptor = _open_existing_receipt(path, expected, protected_identities)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReceiptPersistenceError(
            f"cannot inspect existing receipt {path}: {exc}"
        ) from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return True


def _install_no_clobber(
    path: Path,
    encoded: bytes,
    protected_identities: set[tuple[int, int]],
) -> None:
    temporary_path: Path | None = None
    temporary_descriptor = -1
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        os.fchmod(temporary_descriptor, 0o600)
        with os.fdopen(temporary_descriptor, "wb") as stream:
            temporary_descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if not _verify_existing(path, encoded, protected_identities):
                raise ReceiptPersistenceError(
                    f"receipt destination disappeared during installation: {path}"
                ) from exc
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(path.parent)
    except ReceiptPersistenceError:
        raise
    except OSError as exc:
        raise ReceiptPersistenceError(str(exc)) from exc
    finally:
        if temporary_descriptor >= 0:
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def persist_receipt(
    operation: str,
    payload: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    protected_inputs: Sequence[str | Path] = (),
) -> StoredReceipt:
    """Durably install one canonical receipt without replacing any file.

    Omitted output paths use a content-addressed private user-data store. An
    existing identical file is an idempotent success; any alias or content
    mismatch fails closed.
    """
    if _OPERATION_PATTERN.fullmatch(operation) is None:
        raise ReceiptPersistenceError(f"invalid receipt operation: {operation!r}")

    encoded = canonical_receipt_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    protected_paths, protected_identities = _protected_identities(protected_inputs)

    if output_path is None:
        root = _receipt_root()
        _ensure_directory(root, enforce_private=True)
        operation_dir = root / operation
        _ensure_directory(operation_dir, enforce_private=True)
        path = operation_dir / f"{digest}.json"
    else:
        path = _absolute_path(output_path)
        _ensure_directory(path.parent, enforce_private=False)

    if path in protected_paths:
        raise ReceiptPersistenceError(
            f"receipt destination aliases an input artifact: {path}"
        )
    if _verify_existing(path, encoded, protected_identities):
        return StoredReceipt(path=path, sha256=digest)

    _install_no_clobber(path, encoded, protected_identities)
    return StoredReceipt(path=path, sha256=digest)


__all__ = [
    "ReceiptPersistenceError",
    "StoredReceipt",
    "canonical_receipt_bytes",
    "persist_receipt",
]
