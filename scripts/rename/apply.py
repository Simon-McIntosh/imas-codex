"""Apply the project rename to a tree.

The command is repeatable: a second application finds no old name outside the
kept-name list and no path to move, so it reports zero changed files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.rename.rules import OLD_NAME_PATTERN, PATH_TOKENS, REPLACEMENTS
from scripts.rename.tree import (
    KeptList,
    load_kept_list,
    path_digest,
    read_text,
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
    changed_files: list[str] = field(default_factory=list)
    renamed: list[dict[str, str]] = field(default_factory=list)
    kept_hits: int = 0
    binary_skipped: int = 0
    dry_run: bool = False
    elapsed_seconds: float = 0.0

    @property
    def changed_file_count(self) -> int:
        return len(self.changed_files)

    @property
    def renamed_count(self) -> int:
        return len(self.renamed)

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["changed_file_count"] = self.changed_file_count
        data["renamed_count"] = self.renamed_count
        return data


def _count_hits(text: str) -> int:
    return len(OLD_NAME_PATTERN.findall(text))


def _replace(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for old, new in REPLACEMENTS:
        found = text.count(old)
        if found:
            counts[old] = found
            text = text.replace(old, new)
    return text, counts


def _rename_pass(tree: Path, kept: KeptList) -> list[dict[str, str]]:
    """Move every file and directory whose basename carries the project name."""
    moved: list[dict[str, str]] = []
    while True:
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
            if kept.covering(rel) is not None or not path.exists():
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

    for rel in tracked_files(tree):
        text = read_text(tree / rel)
        if text is None:
            report.binary_skipped += 1
            continue
        hits = _count_hits(text)
        if kept.covering(rel) is not None:
            report.kept_hits += hits
            report.old_name_hits_before += hits
            report.old_name_hits_after += hits
            continue
        report.old_name_hits_before += hits
        replaced, counts = _replace(text)
        for spelling, found in counts.items():
            report.hits_by_spelling[spelling] = (
                report.hits_by_spelling.get(spelling, 0) + found
            )
        if replaced != text:
            report.changed_files.append(rel)
            if not dry_run:
                (tree / rel).write_bytes(replaced.encode("utf-8"))
        report.old_name_hits_after += _count_hits(replaced)

    if not dry_run:
        report.renamed = _rename_pass(tree, kept)

    report.elapsed_seconds = round(time.monotonic() - started, 3)
    return report


def tree_digest(tree: Path) -> str:
    """Digest of every file under the tree, for the idempotence measure."""
    return path_digest(Path(tree).resolve())
