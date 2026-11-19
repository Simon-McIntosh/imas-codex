"""Account every surface of the consumer inventory in the cutover sequence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CUTOVER_FILE = Path(__file__).with_name("cutover.json")
DISPOSITIONS = frozenset({"renamed", "kept", "migrated", "not measured"})


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
    return report


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
