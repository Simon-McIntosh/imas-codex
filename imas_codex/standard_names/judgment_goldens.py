"""Rebuild independent standard-name judgment fixtures from recorded decisions."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

ROOT = Path(__file__).parents[2]
WEST = ROOT / "docs/evidence/sn-west-catalog-release"
EXPERT = Path(
    "/home/ITER/mcintos/Code/imas-standard-names/docs/archive/independent-review-fixes-dup-paths-classification.html"
)
OUTPUT = ROOT / "docs/evidence/fragments/sn-judgment-evaluation"
SEED = 17041
WEST_AUDITS = (
    "west-name-audit.md",
    "west-name-audit-bremsstrahlung-to-equilibrium-boundary.md",
    "west-name-audit-equilibrium-boundary-to-profiles.md",
    "west-name-audit-equilibrium-profiles-to-magnetics.md",
    "west-name-audit-magnetics-to-spectrometer-visible.md",
    "west-name-audit-spectrometer-visible-to-wall.md",
)
EXPERT_NAME_RULINGS = (
    ("second_local_tangential_coordinate_of_optical_element", 4),
    ("fast_ion_charge_state_absorbed_wave_power", 6),
    ("total_power_due_to_ion_cyclotron_heating", 7),
    ("thermal_electron_absorbed_wave_power", 8),
    ("power_density", 9),
    ("electron_power_density", 9),
    ("thermal_electron_energy", 10),
    ("volume_of_first_wall", 11),
    ("electron_deposited_power", 12),
    ("plasma_current", 13),
    ("thermal_plasma_energy", 13),
    ("total_plasma_energy", 13),
)


def _domain(unit: str, name: str = "") -> str:
    path = unit.split("/", 1)[0]
    if path in {"*", "..."} or path.startswith("{"):
        return "multi"
    return (
        path
        if "/" in unit
        else (name.split("_of_")[-1].split("_")[0] if "_of_" in name else "general")
    )


def _family(name: str) -> str:
    for marker in (
        "coordinate",
        "flux",
        "density",
        "power",
        "temperature",
        "current",
        "velocity",
        "field",
        "energy",
        "pressure",
    ):
        if marker in name:
            return marker
    return name.split("_")[0]


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin physics domains and name-pattern families with a fixed seed."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["physics_domain"], row["pattern_family"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _rank(row["unit"]))
    chosen: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda key: _rank(repr(key)))
    while len(chosen) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop(0))
    return chosen


def _row(
    unit: str,
    label: int,
    source: str,
    evidence: str,
    *,
    name: str = "",
    group: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "label": label,
        "canonical_unit": unit,
        "copy_group": group or unit,
        "label_source": source,
        "source_evidence": evidence,
        "physics_domain": domain or _domain(unit, name),
        "pattern_family": _family(name or unit.rsplit("/", 1)[-1]),
    }


def _west_verdicts() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    heading = re.compile(
        r"^###\s+\d+\.\s+`([^`]+)`\s+—\s+\*\*(correct|INCORRECT)\*\*", re.M
    )
    for filename in WEST_AUDITS:
        text = (WEST / filename).read_text()
        matches = list(heading.finditer(text))
        for index, match in enumerate(matches):
            body = text[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            ]
            path = re.search(r"^- source path:\s*`([^`]+)`", body, re.M)
            if path:
                out.append(
                    {
                        "name": match[1],
                        "path": path[1],
                        "verdict": match[2].lower(),
                        "description_issue": bool(
                            re.search(
                                r"description is wrong|description defect|defect is in the description",
                                body,
                                re.IGNORECASE,
                            )
                        ),
                        "evidence": f"{filename}:row={index + 1}",
                    }
                )
    return out


def _expert_verdicts() -> list[dict[str, str]]:
    text = EXPERT.read_text()
    table = re.search(r"<tbody>(.*?)</tbody>", text, re.S)
    if table is None:
        raise ValueError("expert duplicate-path table missing")
    out = []
    for index, tr in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", table[1], re.S), 1):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(cells) != 3:
            continue
        path = html.unescape(re.sub(r"<[^>]*>", "", cells[0])).strip()
        verdict = html.unescape(re.sub(r"<[^>]*>", "", cells[2])).strip()
        kept = re.search(r"keeps\s+([a-z][a-z0-9_]*)", verdict)
        out.append(
            {
                "path": path,
                "bucket": cells[1].strip(),
                "verdict": verdict,
                "kept": kept[1] if kept else "",
                "evidence": f"independent-review-fixes-dup-paths-classification.html:table-row={index}",
            }
        )
    return out


def _review_coverage(graph: GraphClient, names: list[str]) -> set[str]:
    if not names:
        return set()
    rows = graph.query(
        "UNWIND $names AS name MATCH (sn:StandardName {id:name}) WHERE EXISTS { MATCH (sn)-[:HAS_REVIEW]->(:StandardNameReview) } RETURN DISTINCT sn.id AS name LIMIT 2000",
        names=sorted(set(names))[:2000],
    )
    return {row["name"] for row in rows}


def _finish(
    rows: list[dict[str, Any]],
    graph: GraphClient,
    *,
    name_key: str = "name",
) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    unique = []
    for row in rows:
        key = (row["unit"], row["label"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    covered = _review_coverage(
        graph, [row.get(name_key, row["unit"]) for row in unique]
    )
    groups = sorted({row["copy_group"] for row in unique}, key=_rank)
    split = {
        group: "calibration" if index % 2 == 0 else "validation"
        for index, group in enumerate(groups)
    }
    for row in unique:
        row["split"] = split[row["copy_group"]]
        row["incumbent_review"] = row.get(name_key, row["unit"]) in covered
        row["domain"] = row.pop("physics_domain")
        row["pattern"] = row.pop("pattern_family")
        row.pop("name", None)
    return sorted(
        unique, key=lambda row: (row["unit"], row["label"], row["label_source"])
    )


def _group_name_copies(rows: list[dict[str, Any]]) -> None:
    """Keep spelling copies and each human correction pair on one split."""
    parent: dict[str, str] = {}

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    corrections: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["label_source"].startswith("human_edit_"):
            corrections[row["copy_group"]].append(row["unit"])
    for names in corrections.values():
        if len(names) == 2:
            left, right = sorted(names)
            parent[root(right)] = root(left)
    for row in rows:
        name = row.get("name", row["unit"])
        row["copy_group"] = f"name:{_rank(root(name))[:16]}"


def build_name_acceptable(graph: GraphClient) -> list[dict[str, Any]]:
    """Name labels from deterministic quarantine, human edits, and DD audits."""
    quarantines = graph.query(
        "MATCH (sn:StandardName {validation_status:'quarantined'}) RETURN sn.id AS name, sn.physics_domain AS domain, sn.source_paths AS paths ORDER BY sn.id LIMIT 700"
    )
    edits = graph.query(
        "MATCH (change:StandardNameChange {operation:'human_edit', origin:'human'}) WHERE change.from_name IS NOT NULL AND change.to_name IS NOT NULL RETURN change.id AS id, change.from_name AS old, change.to_name AS new ORDER BY old, new LIMIT 400"
    )
    west = _west_verdicts()
    rows = []
    for item in _sample(
        [
            _row(
                item["name"],
                0,
                "grammar_quarantine",
                "StandardName:validation_status=quarantined",
                name=item["name"],
                group=f"name:{_rank(item['name'])[:16]}",
                domain=item.get("domain")
                or _domain((item.get("paths") or [""])[0], item["name"]),
            )
            for item in quarantines
        ],
        200,
    ):
        rows.append(item)
    edit_pairs = []
    for item in edits:
        if item["old"] == item["new"]:
            continue
        group = f"human_edit:{_rank(item['old'] + '->' + item['new'])[:16]}"
        evidence = f"StandardNameChange:{item['id']}"
        edit_pairs.append(
            (
                _row(
                    item["old"],
                    0,
                    "human_edit_source",
                    evidence,
                    name=item["old"],
                    group=group,
                ),
                _row(
                    item["new"],
                    1,
                    "human_edit_target",
                    evidence,
                    name=item["new"],
                    group=group,
                ),
            )
        )
    used_names: set[str] = {row["unit"] for row in rows}
    selected_edits = _sample(
        [
            {
                "unit": old["unit"],
                "physics_domain": old["physics_domain"],
                "pattern_family": old["pattern_family"],
                "pair": (old, new),
            }
            for old, new in edit_pairs
        ],
        len(edit_pairs),
    )
    for item in selected_edits:
        old, new = item["pair"]
        if old["unit"] in used_names or new["unit"] in used_names:
            continue
        rows.extend((old, new))
        used_names.update((old["unit"], new["unit"]))
        if sum(row["label_source"] == "human_edit_target" for row in rows) == 100:
            break
    west_bad = [
        _row(
            f"{item['path']}::{item['name']}",
            0,
            "west_audit",
            item["evidence"],
            name=item["name"],
            group=f"name:{_rank(item['name'])[:16]}",
            domain=_domain(item["path"]),
        )
        | {"name": item["name"]}
        for item in west
        if item["verdict"] == "incorrect"
    ]
    west_good = [
        _row(
            f"{item['path']}::{item['name']}",
            1,
            "west_audit",
            item["evidence"],
            name=item["name"],
            group=f"name:{_rank(item['name'])[:16]}",
            domain=_domain(item["path"]),
        )
        | {"name": item["name"]}
        for item in west
        if item["verdict"] == "correct"
    ]
    rows.extend(_sample(west_bad, 60))
    rows.extend(_sample(west_good, 140))
    for path, reference in REFERENCE_NAMES.items():
        rows.append(
            _row(
                f"{path}::{reference['name']}",
                1,
                "benchmark_reference",
                "benchmark_reference.py:REFERENCE_NAMES",
                name=reference["name"],
                domain=_domain(path),
            )
            | {"name": reference["name"]}
        )
    for item in _expert_verdicts():
        if item["kept"] and item["bucket"] in {"A", "B"}:
            rows.append(
                _row(
                    f"{item['path']}::{item['kept']}",
                    1,
                    "isn_expert_review",
                    item["evidence"],
                    name=item["kept"],
                    domain=_domain(item["path"]),
                )
                | {"name": item["kept"]}
            )
    for name, comment in EXPERT_NAME_RULINGS:
        rows.append(
            _row(
                f"expert-review-reply:comment={comment}::{name}",
                1,
                "isn_expert_reply",
                f"expert-review-reply.html:comment={comment}",
                name=name,
                group=f"name:{_rank(name)[:16]}",
            )
            | {"name": name}
        )
    _group_name_copies(rows)
    return _finish(rows, graph)


def build_description_object(graph: GraphClient) -> list[dict[str, Any]]:
    """Object correctness for individually audited DD bindings."""
    west = _west_verdicts()
    bad = [item for item in west if item["verdict"] == "incorrect"]
    good = [
        item
        for item in west
        if item["verdict"] == "correct" and not item["description_issue"]
    ]
    rows = []
    for item in _sample(
        [
            dict(
                item,
                unit=f"{item['path']}::{item['name']}",
                physics_domain=_domain(item["path"]),
                pattern_family=_family(item["name"]),
            )
            for item in bad
        ],
        74,
    ):
        row = _row(
            item["unit"],
            0,
            "west_audit",
            item["evidence"],
            name=item["name"],
            group=item["path"],
            domain=item["physics_domain"],
        )
        row["name"] = item["name"]
        rows.append(row)
    for item in _sample(
        [
            dict(
                item,
                unit=f"{item['path']}::{item['name']}",
                physics_domain=_domain(item["path"]),
                pattern_family=_family(item["name"]),
            )
            for item in good
        ],
        26,
    ):
        row = _row(
            item["unit"],
            1,
            "west_audit",
            item["evidence"],
            name=item["name"],
            group=item["path"],
            domain=item["physics_domain"],
        )
        row["name"] = item["name"]
        rows.append(row)
    for path, reference in REFERENCE_NAMES.items():
        row = _row(
            f"{path}::{reference['name']}",
            1,
            "benchmark_reference",
            f"imas_codex/standard_names/benchmark_reference.py:REFERENCE_NAMES:{path}",
            name=reference["name"],
            group=path,
            domain=_domain(path),
        )
        row["name"] = reference["name"]
        rows.append(row)
    for item in _expert_verdicts():
        if item["kept"] and item["bucket"] in {"A", "B"}:
            row = _row(
                f"{item['path']}::{item['kept']}",
                1,
                "isn_expert_review",
                item["evidence"],
                name=item["kept"],
                domain=_domain(item["path"]),
            )
            row["name"] = item["kept"]
            rows.append(row)
    for name, comment in EXPERT_NAME_RULINGS:
        row = _row(
            f"expert-review-reply:comment={comment}::{name}",
            1,
            "isn_expert_reply",
            f"expert-review-reply.html:comment={comment}",
            name=name,
            group=f"name:{_rank(name)[:16]}",
        )
        row["name"] = name
        rows.append(row)
    return _finish(rows, graph)


def build_near_duplicate(graph: GraphClient) -> list[dict[str, Any]]:
    """Distinct-name pairs from explicit split and duplicate-path verdicts."""
    rows = []
    text = (WEST / "west-name-shared-identities.md").read_text()
    for number, name, verdict in re.findall(
        r"^###\s+(\d+)\.\s+`([^`]+)`\s+—\s+\*\*(ONE-QUANTITY|MUST-SPLIT)", text, re.M
    ):
        if verdict != "MUST-SPLIT":
            continue
        group = next(
            (
                g
                for g in json.loads(
                    (WEST / "west-name-shared-identities.json").read_text()
                )["groups"]
                if g["name"] == name
            ),
            None,
        )
        if group is None:
            continue
        bindings = [binding["path"] for binding in group["bindings"]]
        for path in bindings[1:]:
            rows.append(
                _row(
                    f"{name}::{bindings[0]}::{path}",
                    0,
                    "west_shared_identity",
                    f"west-name-shared-identities.md:group={number}",
                    name=name,
                    group=name,
                    domain=_domain(path),
                )
                | {"name": name}
            )
    verdicts = _expert_verdicts()
    exact_paths = sorted(
        {
            verdict["path"]
            for verdict in verdicts
            if verdict["bucket"] in {"A", "B", "C"}
            and re.fullmatch(r"[a-z0-9_]+(?:/[a-z0-9_]+)+", verdict["path"])
        }
    )
    pairs = graph.query(
        "UNWIND $paths AS path MATCH (dd:IMASNode {id:path})-[:HAS_STANDARD_NAME]->(sn:StandardName) WITH path, collect(DISTINCT sn.id) AS names WHERE size(names) = 2 RETURN path, names ORDER BY path LIMIT 100",
        paths=exact_paths,
    )
    by_path = {pair["path"]: pair["names"] for pair in pairs}
    for verdict in verdicts:
        if verdict["bucket"] not in {"A", "B", "C"}:
            continue
        names = by_path.get(verdict["path"])
        if names is None:
            continue
        names = sorted(names)
        rows.append(
            _row(
                f"{names[0]}::{names[1]}::{verdict['path']}",
                0,
                "isn_duplicate_path_review",
                verdict["evidence"],
                name=names[0],
                group="::".join(names),
                domain=_domain(verdict["path"]),
            )
            | {"name": names[0]}
        )
    return _finish(rows, graph)


def write_fixture(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist rows and return a compact composition record for the evidence fragment."""
    encoded = ",\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    path.write_text("[\n" + encoded + "\n]\n")
    return {
        "path": str(path),
        "rows": len(rows),
        "sources": dict(Counter(row["label_source"] for row in rows)),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "incumbent_overlap": sum(row["incumbent_review"] for row in rows),
        "splits": dict(Counter(row["split"] for row in rows)),
    }


def main() -> None:
    with GraphClient() as graph:
        builders = (
            ("golden-name-acceptable.json", build_name_acceptable),
            ("golden-description-object.json", build_description_object),
            ("golden-near-duplicate.json", build_near_duplicate),
        )
        for filename, builder in builders:
            print(json.dumps(write_fixture(OUTPUT / filename, builder(graph))))


if __name__ == "__main__":
    main()
