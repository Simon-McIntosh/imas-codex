"""Closing search and kept-name checks for a renamed tree."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from scripts.rename.apply import KEPT_LIST
from scripts.rename.rules import OLD_NAME_PATTERN
from scripts.rename.tree import load_kept_list, read_text, tracked_files


@dataclass
class Violation:
    kind: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class CheckReport:
    tree: str
    files_scanned: int = 0
    old_name_hits: int = 0
    kept_hits: int = 0
    leftover: dict[str, int] = field(default_factory=dict)
    kept_paths: dict[str, dict] = field(default_factory=dict)
    kept_names: dict[str, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["violations"] = [v.as_dict() for v in self.violations]
        data["ok"] = self.ok
        data["leftover_count"] = len(self.leftover)
        return data


def check_tree(tree: Path, *, kept_list: Path = KEPT_LIST) -> CheckReport:
    """Search a tree for the old name and hold the kept-name list to account.

    Every hit outside a kept path is a leftover.  A kept path whose counted
    hits fall below the floor the list declares has been altered by hand, and a
    kept name that no longer appears in the tree has been renamed by mistake.
    """
    tree = Path(tree).resolve()
    kept = load_kept_list(kept_list)
    report = CheckReport(tree=str(tree))
    for entry in kept.kept_paths:
        report.kept_paths[entry.id] = {
            "path": entry.path,
            "disposition": entry.disposition,
            "hits": 0,
            "files": 0,
        }

    live_texts: dict[str, str] = {}
    for rel in tracked_files(tree):
        text = read_text(tree / rel)
        if text is None:
            continue
        report.files_scanned += 1
        hits = len(OLD_NAME_PATTERN.findall(text))
        entry = kept.covering(rel)
        if hits:
            report.old_name_hits += hits
        if entry is None:
            live_texts[rel] = text
            if hits:
                report.leftover[rel] = hits
            continue
        stats = report.kept_paths[entry.id]
        stats["hits"] += hits
        if hits:
            stats["files"] += 1
        report.kept_hits += hits

    # Instrument control: a search that saw no file at all cannot report an
    # absence, and a tree with no kept hit at all means the classified paths
    # were not read (or the kept list is empty), not that the rename is clean.
    if report.files_scanned == 0:
        report.violations.append(Violation("empty-scan", "no text file was read"))
    if report.kept_hits == 0:
        report.violations.append(
            Violation(
                "kept-search-blind",
                "the closing search found no hit under any kept path; the instrument "
                "has not been shown to see a value known to be present",
            )
        )

    for entry in kept.kept_paths:
        stats = report.kept_paths[entry.id]
        if stats["hits"] < entry.min_hits:
            report.violations.append(
                Violation(
                    "kept-path-altered",
                    f"{entry.path} carries {stats['hits']} old-name hits, "
                    f"below the declared floor of {entry.min_hits}",
                )
            )

    for name in kept.kept_names:
        count = sum(text.count(name.marker) for text in live_texts.values())
        report.kept_names[name.id] = count
        if count == 0:
            report.violations.append(
                Violation(
                    "kept-name-missing",
                    f"{name.marker!r} no longer appears outside the kept paths "
                    f"({name.reason})",
                )
            )

    for rel, hits in sorted(report.leftover.items()):
        report.violations.append(
            Violation("leftover", f"{rel}: {hits} old-name hit(s) in a covered file")
        )
    return report


def format_report(report: CheckReport) -> str:
    lines = [
        f"tree: {report.tree}",
        f"files scanned: {report.files_scanned}",
        f"old-name hits: {report.old_name_hits} (kept {report.kept_hits}, "
        f"leftover {sum(report.leftover.values())})",
        "kept paths:",
    ]
    for entry_id, stats in report.kept_paths.items():
        lines.append(
            f"  {entry_id}: {stats['path']} [{stats['disposition']}] "
            f"files={stats['files']} hits={stats['hits']}"
        )
    lines.append(
        "kept names: " + ", ".join(f"{k}={v}" for k, v in report.kept_names.items())
    )
    if report.violations:
        lines.append("violations:")
        lines.extend(f"  {v.kind}: {v.detail}" for v in report.violations)
    else:
        lines.append("violations: none")
    return "\n".join(lines)


def main(tree: Path, *, kept_list: Path = KEPT_LIST, as_json: bool = False) -> int:
    report = check_tree(tree, kept_list=kept_list)
    print(json.dumps(report.as_dict(), indent=2) if as_json else format_report(report))
    return 0 if report.ok else 1
