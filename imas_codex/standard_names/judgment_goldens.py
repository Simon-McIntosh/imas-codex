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


def assign_group_splits(rows: list[dict[str, Any]], group_key: str) -> dict[str, str]:
    """Alternate ranked groups between calibration and validation."""
    groups = sorted({row[group_key] for row in rows}, key=_rank)
    return {
        group: "calibration" if index % 2 == 0 else "validation"
        for index, group in enumerate(groups)
    }


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
    canonical: str | None = None,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "label": label,
        "canonical_unit": canonical or unit,
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
    canonical_labels: dict[str, set[int]] = defaultdict(set)
    for row in unique:
        canonical_labels[row["canonical_unit"]].add(row["label"])
    ambiguous = {unit for unit, labels in canonical_labels.items() if len(labels) > 1}
    if ambiguous:
        unique = [row for row in unique if row["canonical_unit"] not in ambiguous]
    group_labels: dict[str, set[int]] = defaultdict(set)
    for row in unique:
        group_labels[row["copy_group"]].add(row["label"])
    incomplete = {group for group, labels in group_labels.items() if len(labels) < 2}
    unique = [
        row
        for row in unique
        if not (
            row["label_source"].startswith("human_edit_")
            and row["copy_group"] in incomplete
        )
    ]
    covered = _review_coverage(
        graph, [row.get(name_key, row["unit"]) for row in unique]
    )
    split = assign_group_splits(unique, "copy_group")
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
                canonical=item["name"],
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
                    canonical=item["old"],
                ),
                _row(
                    item["new"],
                    1,
                    "human_edit_target",
                    evidence,
                    name=item["new"],
                    group=group,
                    canonical=item["new"],
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
        if sum(row["label_source"] == "human_edit_target" for row in rows) == 200:
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
            canonical=item["name"],
        )
        | {"name": item["name"]}
        for item in west
        if item["verdict"] == "incorrect"
    ]
    rows.extend(_sample(west_bad, 60))
    for path, reference in REFERENCE_NAMES.items():
        rows.append(
            _row(
                f"{path}::{reference['name']}",
                1,
                "benchmark_reference",
                "benchmark_reference.py:REFERENCE_NAMES",
                name=reference["name"],
                domain=_domain(path),
                canonical=reference["name"],
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
                    canonical=item["kept"],
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
                canonical=name,
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
            canonical=item["unit"],
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
            canonical=item["unit"],
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
            canonical=f"{path}::{reference['name']}",
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
                canonical=f"{item['path']}::{item['kept']}",
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
            canonical=name,
        )
        row["name"] = name
        rows.append(row)
    return _finish(rows, graph)


SYNONYM_RULINGS = OUTPUT / "golden-synonym-rulings.json"


def _synonym_ruling_rows() -> list[dict[str, Any]]:
    """One row per ruled name pair; pairs sharing a name share a copy group.

    The label is the ruling's own (1: one quantity, 0: distinct quantities).
    A group is a connected component of the name graph, named by its
    alphabetically first name, so no name sits on both sides of the split.
    """
    rulings = json.loads(SYNONYM_RULINGS.read_text())["rows"]
    pairs = [
        (tuple(sorted((ruling["left"], ruling["right"]))), ruling) for ruling in rulings
    ]
    parent: dict[str, str] = {}

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for (left, right), _ in pairs:
        low, high = sorted((root(left), root(right)))
        parent[high] = low
    members: dict[str, list[str]] = defaultdict(list)
    for name in list(parent):
        members[root(name)].append(name)
    group_of = {name: min(names) for names in members.values() for name in names}
    rows = []
    for (left, right), ruling in pairs:
        unit = f"{left}::{right}"
        rows.append(
            _row(
                unit,
                int(ruling["label"]),
                "coordinator_synonym_ruling",
                f"{SYNONYM_RULINGS.name}:pair={ruling['pair']}",
                name=left,
                group=group_of[left],
                canonical=unit,
            )
            | {"name": left}
        )
    return rows


def build_near_duplicate(graph: GraphClient) -> list[dict[str, Any]]:
    """Name pairs: split and duplicate-path verdicts plus ruled synonym pairs."""
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
                    canonical="::".join(
                        sorted((f"{name}::{bindings[0]}", f"{name}::{path}"))
                    ),
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
                canonical="::".join(
                    sorted(
                        (
                            f"{names[0]}::{verdict['path']}",
                            f"{names[1]}::{verdict['path']}",
                        )
                    )
                ),
            )
            | {"name": names[0]}
        )
    rows.extend(_synonym_ruling_rows())
    return _finish(rows, graph)


DOCUMENTATION_SHEET = OUTPUT / "golden-documentation-acceptable-sheet.json"
DD_DOCUMENTATION_CHARS = 600
LIVE_NAME_ROWS = 6000
_CONVENTIONS: Any = None


def _live_documentation_names(graph: GraphClient) -> list[dict[str, Any]]:
    """Accepted or approved names carrying non-empty documentation.

    ``source_paths`` is the name's DD binding list; a derived name carries
    none and still belongs on the sheet, so the read never requires it.
    """
    return graph.query(
        "MATCH (sn:StandardName) "
        "WHERE sn.name_stage IN ['accepted','approved'] "
        "AND sn.documentation IS NOT NULL AND trim(sn.documentation) <> '' "
        "RETURN sn.id AS name, sn.description AS description, "
        "sn.documentation AS documentation, sn.unit AS unit, "
        "sn.physics_domain AS physics_domain, sn.source_paths AS source_paths "
        "ORDER BY sn.id LIMIT $limit",
        limit=LIVE_NAME_ROWS,
    )


def _unit_symbols(graph: GraphClient, units: list[str]) -> dict[str, str]:
    """The notation symbol behind each name's unit id."""
    wanted = sorted({unit for unit in units if unit})
    if not wanted:
        return {}
    rows = graph.query(
        "UNWIND $units AS unit_id MATCH (unit:Unit {id:unit_id}) "
        "RETURN unit_id AS id, unit.symbol AS symbol LIMIT $limit",
        units=wanted,
        limit=LIVE_NAME_ROWS,
    )
    return {row["id"]: row["symbol"] for row in rows}


def _dd_path_documentation(graph: GraphClient, paths: list[str]) -> dict[str, str]:
    """The DD documentation body behind each IMAS path."""
    wanted = sorted({path for path in paths if path})
    if not wanted:
        return {}
    rows = graph.query(
        "UNWIND $paths AS path MATCH (dd:IMASNode {id:path}) "
        "RETURN path AS path, dd.documentation AS documentation LIMIT $limit",
        paths=wanted,
        limit=LIVE_NAME_ROWS,
    )
    return {row["path"]: row["documentation"] or "" for row in rows}


def _conventions_registry() -> Any:
    """Load the registry entries once; ``select_conventions`` reloads per call."""
    global _CONVENTIONS
    if _CONVENTIONS is None:
        from imas_standard_names.grammar.terms import load_conventions

        _CONVENTIONS = load_conventions()
    return _CONVENTIONS


def _applied_conventions(name: str, registry: Any = None) -> dict[str, Any]:
    """The registry conventions a name selects, or an empty refused record.

    ``select_conventions`` parses the name with the strict public grammar and
    raises for a legacy id or one whose vocabulary this checkout has not
    reached.  That name still belongs on the sheet: it is recorded with empty
    convention lists and ``parse_refused`` true.  The registry is loaded once
    and passed in, because a per-call load re-reads the convention files.
    """
    from imas_standard_names.grammar.parser import ParseError
    from imas_standard_names.grammar.terms import select_conventions

    try:
        applied = select_conventions(name, registry=registry or _conventions_registry())
    except ParseError:
        return {"parse_refused": True, "concepts": [], "notation": [], "templates": []}
    return {
        "parse_refused": False,
        "concepts": [
            {
                "token": entry.token,
                "phrase": entry.phrase,
                "definition": entry.definition,
                "equation": entry.equation,
                "symbols": [
                    {"symbol": entry_symbol.symbol, "meaning": entry_symbol.meaning}
                    for entry_symbol in entry.symbols
                ],
            }
            for entry in applied.concepts
        ],
        "notation": [
            {"token": entry.token, "symbol": entry.symbol} for entry in applied.notation
        ],
        "templates": [
            {
                "base": entry.base,
                "segments": list(entry.segments),
                "opening": entry.documentation,
            }
            for entry in applied.templates
        ],
    }


def build_documentation_sheet(
    graph: GraphClient,
    count: int = 400,
    reserve: int = 200,
    lead_overlap: int = 50,
) -> list[dict[str, Any]]:
    """The blind documentation-acceptability sheet the coordinator labels.

    Draws ``count + reserve`` live documented names with the shared stratified
    sampler, numbers each row's ``draw_rank`` from 1 in draw order, and marks
    ``lead_overlap`` sheet rows for the lead's independent labelling.  No
    incumbent review score, comment or verdict is carried: every ``label`` is
    null and the only provenance is ``label_source = 'coordinator'``.
    """
    candidates = [
        {
            "unit": item["name"],
            "physics_domain": item.get("physics_domain")
            or _domain(
                (item.get("source_paths") or [""])[0].removeprefix("dd:"), item["name"]
            ),
            "pattern_family": _family(item["name"]),
            "item": item,
        }
        for item in _live_documentation_names(graph)
        if item.get("name") and item.get("documentation")
    ]
    drawn = _sample(candidates, count + reserve)
    registry = _conventions_registry()
    symbols = _unit_symbols(graph, [item["item"].get("unit") for item in drawn])
    dd_docs = _dd_path_documentation(
        graph,
        [
            raw.removeprefix("dd:")
            for item in drawn
            for raw in (item["item"].get("source_paths") or [])[:2]
        ],
    )
    rows: list[dict[str, Any]] = []
    for draw_rank, candidate in enumerate(drawn, 1):
        item = candidate["item"]
        documentation = item["documentation"]
        row = _row(
            item["name"],
            None,
            "coordinator",
            hashlib.sha256(documentation.encode()).hexdigest(),
            name=item["name"],
            group=candidate["pattern_family"],
            domain=candidate["physics_domain"],
            canonical=item["name"],
        )
        row["draw_rank"] = draw_rank
        row["description"] = item.get("description") or ""
        row["documentation"] = documentation
        row["unit_symbol"] = symbols.get(item.get("unit")) or item.get("unit") or ""
        row["dd_sources"] = [
            {
                "path": path,
                "documentation": dd_docs.get(path, "")[:DD_DOCUMENTATION_CHARS],
            }
            for path in [
                raw.removeprefix("dd:") for raw in (item.get("source_paths") or [])[:2]
            ]
        ]
        row["conventions"] = _applied_conventions(item["name"], registry)
        rows.append(row)
    finished = _finish(rows, graph)
    for row in finished:
        row.pop("incumbent_review", None)
        row["physics_domain"] = row.pop("domain")
        row["pattern_family"] = row.pop("pattern")
        row["lead_overlap"] = False
    sheet = sorted(
        (row for row in finished if row["draw_rank"] <= count),
        key=lambda row: _rank(row["unit"]),
    )
    if lead_overlap:
        step = len(sheet) / lead_overlap
        for index in range(lead_overlap):
            sheet[min(int(index * step), len(sheet) - 1)]["lead_overlap"] = True
    return sorted(finished, key=lambda row: row["draw_rank"])


def write_sheet(path: Path, rows: list[dict[str, Any]], count: int) -> dict[str, Any]:
    """Persist an unlabelled sheet and return its composition.

    Separate from :func:`write_fixture` because a sheet row carries a null
    label and reading context the labelled fixtures do not: the composition
    reports the sheet/reserve partition rather than label counts, and the
    labelled writer's contract is left untouched.
    """
    encoded = ",\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    path.write_text("[\n" + encoded + "\n]\n")
    return {
        "path": str(path),
        "rows": len(rows),
        "sheet": count,
        "reserve": len(rows) - count,
        "lead_overlap": sum(row["lead_overlap"] for row in rows),
        "domains": dict(sorted(Counter(row["physics_domain"] for row in rows).items())),
        "families": dict(
            sorted(Counter(row["pattern_family"] for row in rows).items())
        ),
        "splits": dict(Counter(row["split"] for row in rows)),
        "parse_refused": sum(row["conventions"]["parse_refused"] for row in rows),
        "largest_row_bytes": max(
            len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
            for row in rows
        ),
        "bytes": path.stat().st_size,
    }


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
        "canonical_units": len({row["canonical_unit"] for row in rows}),
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
        print(
            json.dumps(
                write_sheet(DOCUMENTATION_SHEET, build_documentation_sheet(graph), 400)
            )
        )


if __name__ == "__main__":
    main()
