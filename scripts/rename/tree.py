"""Tree traversal helpers shared by the rename commands."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.rename.rules import OLD_NAME_PATTERN, SCHEMA_VALUE_PATTERN, SKIP_DIRS


def tracked_files(tree: Path) -> list[str]:
    """Return the tree's text-bearing file set as repository-relative paths.

    A git checkout is listed through the index plus untracked files that git
    does not ignore, so ignored build outputs (generated models, environments,
    secrets) are never considered while a file the branch has not staged yet
    still moves with the rename.  A plain directory - the tests' temporary
    trees - is walked instead.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(tree),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _walk(tree)
    return sorted({p for p in out.split("\0") if p})


def _walk(tree: Path) -> list[str]:
    found: list[str] = []
    for root, dirs, files in os.walk(tree, topdown=True, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        base = Path(root).relative_to(tree)
        for name in sorted(files):
            found.append((base / name).as_posix())
    return found


def read_text(path: Path) -> str | None:
    """Decode a file as UTF-8 text, or return None for binary content."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def path_digest(tree: Path) -> str:
    """Digest of every file's path and bytes under the tree.

    Used to prove a second application changes nothing without relying on git:
    the measure must be able to fail even where the index is already dirty.
    """
    digest = hashlib.sha256()
    for rel in _walk(tree):
        path = tree / rel
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def kept_identifiers(tree: Path, kept: KeptList) -> dict[str, int]:
    """Stored artifact identities that the kept bytes actually name.

    A signed authority or a receipt carries its identity in a ``schema`` value,
    and the evidence holding it keeps its bytes.  Rewriting that identity in the
    code that reads it makes the stored file unreadable - the schema and the
    signature both stop matching.  Reading the identities out of the kept bytes
    themselves keeps the set honest: a TOML section name such as
    ``tool.kept.sn-enrich`` is not an artifact identity and is not protected,
    while a stored schema is, even when only a historical receipt still names
    it.
    """
    counts: dict[str, int] = {}
    for rel in _walk(tree):
        if kept.covering(rel) is None:
            continue
        text = read_text(tree / rel)
        if text is not None:
            for match in SCHEMA_VALUE_PATTERN.finditer(text):
                value = match.group(1)
                if OLD_NAME_PATTERN.search(value):
                    counts[value] = counts.get(value, 0) + 1
    return counts


def subtree_digest(tree: Path, entry: KeptPath) -> str:
    """Digest of every file under one kept path, paths and bytes together.

    The hit floor in the closing search cannot see an addition that carries no
    old name, and cannot see a reworded line that keeps the count.  This digest
    is the exact form of the claim the kept list makes - these bytes are
    evidence, not a product surface - so the presence of an old name is not the
    only thing it can be surprised by.
    """
    digest = hashlib.sha256()
    for rel in _walk(tree):
        if not entry.covers(rel):
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256((tree / rel).read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class KeptPath:
    """A path whose bytes the rename must not touch, with why."""

    id: str
    path: str
    reason: str
    disposition: str
    surface: str
    include: tuple[str, ...] = ()
    min_hits: int = 0

    def covers(self, rel: str) -> bool:
        root = self.path.rstrip("/")
        if not (rel == root or rel.startswith(root + "/")):
            return False
        if not self.include:
            return True
        return any(
            fnmatch.fnmatch(rel, f"{root}/{pattern}") for pattern in self.include
        )


@dataclass(frozen=True)
class KeptName:
    """A literal that must survive the rename, with why."""

    id: str
    marker: str
    reason: str


@dataclass(frozen=True)
class KeptList:
    kept_paths: tuple[KeptPath, ...]
    kept_names: tuple[KeptName, ...]
    old_names: tuple[str, ...]

    def covering(self, rel: str) -> KeptPath | None:
        for entry in self.kept_paths:
            if entry.covers(rel):
                return entry
        return None


def load_kept_list(path: Path) -> KeptList:
    data = json.loads(path.read_text())
    return KeptList(
        kept_paths=tuple(KeptPath(**entry) for entry in data["kept_paths"]),
        kept_names=tuple(KeptName(**entry) for entry in data["kept_names"]),
        old_names=tuple(data["old_names"]),
    )
