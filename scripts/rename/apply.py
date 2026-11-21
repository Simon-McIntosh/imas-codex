"""Apply the project rename to a tree.

The command is repeatable: a second application finds no old name outside the
kept-name list and no path to move, so it reports zero changed files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.rename.rules import PATH_TOKENS, count_hits, protect, rename_text, restore
from scripts.rename.tree import (
    KeptList,
    kept_identifiers,
    load_kept_list,
    path_digest,
    read_text,
    subtree_digest,
    tracked_files,
)

KEPT_LIST = Path(__file__).with_name("kept_names.json")

# Basename tokens move to their new spelling; the separator decides which.
PATH_RENAMES: dict[str, str] = {"imas_codex": "iter_codex", "imas-codex": "iter-codex"}


@dataclass
class ApplyReport:
    tree: str
    kept_list: str
    old_name_hits_before: int = 0
    old_name_hits_after: int = 0
    hits_by_spelling: dict[str, int] = field(default_factory=dict)
    reckon_key_hits: dict[str, int] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    renamed: list[dict[str, str]] = field(default_factory=list)
    kept_hits: int = 0
    kept_digests: dict[str, str] = field(default_factory=dict)
    kept_identifiers: dict[str, int] = field(default_factory=dict)
    kept_altered: list[str] = field(default_factory=list)
    binary_skipped: int = 0
    dry_run: bool = False
    elapsed_seconds: float = 0.0

    @property
    def changed_file_count(self) -> int:
        return len(self.changed_files)

    @property
    def renamed_count(self) -> int:
        return len(self.renamed)

    @property
    def renamed_occurrences(self) -> int:
        return sum(self.hits_by_spelling.values())

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["changed_file_count"] = self.changed_file_count
        data["renamed_count"] = self.renamed_count
        data["renamed_occurrences"] = self.renamed_occurrences
        return data


def _kept_directories(tree: Path, kept: KeptList) -> set[str]:
    """Directories that hold a file the kept list covers, as relative paths.

    A kept surface is a set of file bytes, and a subtree's directory names are
    part of where those bytes live.  `docs/evidence` now keeps only its data and
    prose, so a directory under it is covered by no include pattern and would
    otherwise be renamed with the code beside it - moving the kept bytes and
    breaking the receipt that says they survived untouched.
    """
    directories: set[str] = set()
    for path in tree.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(tree).as_posix()
        if kept.covering(rel) is None:
            continue
        parent = Path(rel).parent.as_posix()
        while parent not in (".", ""):
            directories.add(parent)
            parent = Path(parent).parent.as_posix()
    return directories


def _rename_pass(tree: Path, kept: KeptList) -> list[dict[str, str]]:
    """Move every file and directory whose basename carries the project name."""
    moved: list[dict[str, str]] = []
    while True:
        held = _kept_directories(tree, kept)
        candidates = [
            path
            for path in tree.rglob("*")
            if any(token in path.name for token in PATH_TOKENS)
            and not any(
                part in {".git", ".venv"} for part in path.relative_to(tree).parts
            )
        ]
        candidates.sort(key=lambda p: len(p.relative_to(tree).parts), reverse=True)
        progressed = False
        for path in candidates:
            rel = path.relative_to(tree).as_posix()
            if rel in held or kept.covering(rel) is not None or not path.exists():
                continue
            new_name = path.name
            for token, replacement in PATH_RENAMES.items():
                new_name = new_name.replace(token, replacement)
            target = path.with_name(new_name)
            moved_report = {"from": rel, "to": target.relative_to(tree).as_posix()}
            if target.exists():
                raise FileExistsError(
                    f"cannot rename {rel}: {moved_report['to']} already exists"
                )
            path.rename(target)
            moved.append(moved_report)
            progressed = True
        if not progressed:
            return moved


def apply_rename(
    tree: Path, *, kept_list: Path = KEPT_LIST, dry_run: bool = False
) -> ApplyReport:
    """Rewrite the tree's tracked text and move the package directory."""
    tree = tree.resolve()
    kept = load_kept_list(kept_list)
    report = ApplyReport(tree=str(tree), kept_list=str(kept_list), dry_run=dry_run)
    started = time.monotonic()
    before = {entry.id: subtree_digest(tree, entry) for entry in kept.kept_paths}
    report.kept_digests = before
    persisted = kept_identifiers(tree, kept)
    report.kept_identifiers = persisted
    identifiers = sorted(persisted)

    for rel in tracked_files(tree):
        text = read_text(tree / rel)
        if text is None:
            report.binary_skipped += 1
            continue
        hits = count_hits(text)
        if kept.covering(rel) is not None:
            report.kept_hits += hits
            report.old_name_hits_before += hits
            report.old_name_hits_after += hits
            continue
        report.old_name_hits_before += hits
        guarded, stand_ins, context_counts = protect(
            text, identifiers, kept.contexts_for(rel)
        )
        replaced, counts = rename_text(guarded)
        replaced = restore(replaced, stand_ins)
        for spelling, found in counts.items():
            report.hits_by_spelling[spelling] = (
                report.hits_by_spelling.get(spelling, 0) + found
            )
        for context_id, found in context_counts.items():
            report.reckon_key_hits[context_id] = (
                report.reckon_key_hits.get(context_id, 0) + found
            )
        if replaced != text:
            report.changed_files.append(rel)
            if not dry_run:
                (tree / rel).write_bytes(replaced.encode("utf-8"))
        report.old_name_hits_after += count_hits(replaced)

    if not dry_run:
        report.renamed = _rename_pass(tree, kept)
        report.kept_altered = [
            entry.id
            for entry in kept.kept_paths
            if subtree_digest(tree, entry) != before[entry.id]
        ]

    report.elapsed_seconds = round(time.monotonic() - started, 3)
    return report


def tree_digest(tree: Path) -> str:
    """Digest of every file under the tree, for the idempotence measure."""
    return path_digest(Path(tree).resolve())
