"""Account every surface of the consumer inventory in the cutover sequence."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

from scripts.rename.tree import KeptList, load_kept_list

CUTOVER_FILE = Path(__file__).with_name("cutover.json")
REPOSITORY_ROOT = CUTOVER_FILE.parents[2]
DISPOSITIONS = frozenset({"renamed", "kept", "migrated", "not measured"})
ORIGINS = frozenset({"inventory", "brief"})

# A step whose written instruction was corrected after a defect is checked
# structurally as well as by reading it: the block it carries is the part that
# must survive an edit, so removing it fails here rather than in a review.
REQUIRED_STEP_BLOCKS = {
    "C0-freeze": {
        "queued_requests": (
            "listing",
            "operator_instruction",
            "rule",
            "parked_worktree",
        )
    },
    "C7-reckon-project-key": {
        "unchanged": (
            "project_key",
            "mounts_entry",
            "state_directory",
            "stored_reviews",
            "ledger_rows",
            "worktree_root",
            "rule",
        )
    },
    "C9-gate-scripts": {
        "kept_directory": ("path", "disposition", "regenerated_by", "rule")
    },
}
REQUIRED_BLOCK_VALUES = {"kept_directory": {"disposition": "kept"}}
REQUIRED_STEP_DISPOSITIONS = {
    "C7-reckon-project-key": "kept",
    "C9-gate-scripts": "kept",
}
REQUIRED_STEP_TITLE_WORDS = {"C9-gate-scripts": ("regenerate", "evidence")}
# The reckon step is a check, not a move: a text that carries either of these
# has turned it back into a migration of the key or the state directory;
FORBIDDEN_STEP_TEXT = {"C7-reckon-project-key": ("git mv", "docs/state/iter-codex")}


@dataclass
class CutoverReport:
    source: str
    inventory: dict[str, dict] = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        covered: dict[str, list[str]] = {key: [] for key in self.inventory}
        for step in self.steps:
            for surface in step.get("surfaces", []):
                covered.setdefault(surface, []).append(step["id"])
        return {
            "source": self.source,
            "inventory_count": len(self.inventory),
            "step_count": len(self.steps),
            "coverage": covered,
            "unaccounted": [k for k, steps in covered.items() if not steps],
            "violations": self.violations,
            "ok": self.ok,
        }


def check_cutover(path: Path = CUTOVER_FILE) -> CutoverReport:
    data = json.loads(Path(path).read_text())
    report = CutoverReport(source=str(path), steps=data["steps"])
    report.inventory = {row["id"]: row for row in data["inventory"]}

    covered = dict.fromkeys(report.inventory, 0)
    for row in report.inventory.values():
        if row.get("origin") not in ORIGINS:
            report.violations.append(
                f"surface {row['id']!r} has origin {row.get('origin')!r}"
            )
        elif (
            row["origin"] == "inventory"
            and not (row.get("document_label") or "").strip()
        ):
            report.violations.append(
                f"surface {row['id']!r} claims the inventory document but names no label"
            )
    for step in report.steps:
        if not step.get("id"):
            report.violations.append("a cutover step has no id")
        if step.get("disposition") not in DISPOSITIONS:
            report.violations.append(
                f"step {step.get('id')!r} has disposition {step.get('disposition')!r}"
            )
        for key in ("step", "check", "undo"):
            if not (step.get(key) or "").strip():
                report.violations.append(f"step {step.get('id')!r} has no {key}")
        for surface in step.get("surfaces", []):
            if surface not in report.inventory:
                report.violations.append(
                    f"step {step.get('id')!r} names unknown surface {surface!r}"
                )
            else:
                covered[surface] += 1

    for surface, count in covered.items():
        if count == 0:
            report.violations.append(f"surface {surface!r} is not accounted for")

    _check_checkout_directory(data, report)
    _check_tracked_rename(data, report)
    _check_required_blocks(data, report)
    return report


def _check_checkout_directory(data: dict, report: CutoverReport) -> None:
    """The sequence answers, once, whether the checkout directory moves."""
    checkout = data.get("checkout_directory")
    if not isinstance(checkout, dict):
        report.violations.append(
            "the sequence states no checkout_directory: one answer is required, "
            "and 'the renamed checkout' and 'the checkout keeps its name' cannot both stand"
        )
        return
    if not (checkout.get("path") or "").strip():
        report.violations.append("checkout_directory names no path")
    if not (checkout.get("statement") or "").strip():
        report.violations.append("checkout_directory carries no statement")
    if checkout.get("moves_at_cutover") is not False:
        report.violations.append(
            "checkout_directory must record moves_at_cutover: false; consumers resolve "
            "through the directory, so a move is its own step with its own check and undo"
        )
    if not checkout.get("consumers"):
        report.violations.append("checkout_directory names no consumers to re-point")


def _check_tracked_rename(data: dict, report: CutoverReport) -> None:
    """The sequence names the kept split the tracked rename relies on."""
    tracked = data.get("tracked_rename")
    if not isinstance(tracked, dict):
        report.violations.append("the sequence names no tracked_rename block")
        return
    for key in ("command", "kept_list", "evidence_split"):
        if not (tracked.get(key) or "").strip():
            report.violations.append(f"tracked_rename has no {key}")
    scopes = tracked.get("kept_scopes") or []
    if not scopes:
        report.violations.append("tracked_rename names no kept scope")
        return
    kept = _load_kept_list(tracked.get("kept_list") or "", report)
    if kept is None:
        return
    entries = {entry.id: entry for entry in kept.kept_paths}
    for scope in scopes:
        entry = entries.get(scope)
        if entry is None:
            report.violations.append(
                f"tracked_rename names kept scope {scope!r}, which the kept list does not carry"
            )
            continue
        if entry.path.rstrip("/") == "docs/evidence":
            if entry.include:
                report.violations.append(
                    f"kept scope {scope!r} narrows {entry.path} to an include list; the "
                    "evidence default is every file under the path, so a new evidence "
                    "file type needs no change to this list"
                )
            for name in ("probe.py", "probe.sh"):
                if not any(fnmatch.fnmatch(name, pattern) for pattern in entry.exclude):
                    report.violations.append(
                        f"kept scope {scope!r} does not rename {name} under "
                        f"{entry.path}: the live code beside the kept bytes follows "
                        "the package"
                    )
        else:
            if not entry.include:
                report.violations.append(
                    f"kept scope {scope!r} keeps every file under {entry.path}; this "
                    "scope names the data extensions it keeps, because the script "
                    "beside them is live code"
                )
            elif any(fnmatch.fnmatch("probe.py", pattern) for pattern in entry.include):
                report.violations.append(
                    f"kept scope {scope!r} keeps Python under {entry.path}; live code "
                    "is renamed"
                )


def _load_kept_list(reference: str, report: CutoverReport) -> KeptList | None:
    if not reference:
        return None
    path = Path(reference)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    try:
        return load_kept_list(path)
    except (OSError, ValueError, KeyError) as error:
        report.violations.append(f"kept list {reference!r} cannot be read: {error}")
        return None


def _check_required_blocks(data: dict, report: CutoverReport) -> None:
    """Each corrected step still carries the block that correction added."""
    steps = {step.get("id"): step for step in data.get("steps", [])}
    for step_id, blocks in REQUIRED_STEP_BLOCKS.items():
        step = steps.get(step_id)
        if step is None:
            report.violations.append(f"the sequence carries no {step_id} step")
            continue
        for block, keys in blocks.items():
            value = step.get(block)
            if not isinstance(value, dict) or not value:
                report.violations.append(
                    f"step {step_id} carries no {block}: the rule this step is "
                    "corrected for has been removed"
                )
                continue
            for key in keys:
                if not str(value.get(key) or "").strip():
                    report.violations.append(f"step {step_id} {block} has no {key}")
            for key, expected in REQUIRED_BLOCK_VALUES.get(block, {}).items():
                if value.get(key) != expected:
                    report.violations.append(
                        f"step {step_id} {block} records {key}={value.get(key)!r}, "
                        f"expected {expected!r}"
                    )
        disposition = REQUIRED_STEP_DISPOSITIONS.get(step_id)
        if disposition and step.get("disposition") != disposition:
            report.violations.append(
                f"step {step_id} records disposition {step.get('disposition')!r}, "
                f"expected {disposition!r}"
            )
        text = str(step.get("title") or "").lower()
        for word in REQUIRED_STEP_TITLE_WORDS.get(step_id, ()):
            if word not in text:
                report.violations.append(
                    f"step {step_id} no longer states {word!r}: the title and the "
                    "body have drifted apart"
                )
        body = " ".join(
            str(step.get(key) or "") for key in ("title", "step", "check", "undo")
        ).lower()
        for phrase in FORBIDDEN_STEP_TEXT.get(step_id, ()):
            if phrase in body:
                report.violations.append(
                    f"step {step_id} carries {phrase!r}; this step checks that reckon "
                    "keeps its key and state directory, it does not move them"
                )


def format_report(report: CutoverReport) -> str:
    data = report.as_dict()
    lines = [
        f"cutover: {report.source}",
        f"inventory surfaces: {data['inventory_count']}",
        f"cutover steps: {data['step_count']}",
        f"unaccounted: {data['unaccounted'] or 'none'}",
    ]
    for surface, steps in data["coverage"].items():
        lines.append(f"  {surface}: {', '.join(steps) or 'UNACCOUNTED'}")
    lines.append(f"violations: {report.violations or 'none'}")
    return "\n".join(lines)


def main(path: Path = CUTOVER_FILE, *, as_json: bool = False) -> int:
    report = check_cutover(path)
    print(json.dumps(report.as_dict(), indent=2) if as_json else format_report(report))
    return 0 if report.ok else 1
