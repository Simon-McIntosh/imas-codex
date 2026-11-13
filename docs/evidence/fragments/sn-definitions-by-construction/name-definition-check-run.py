"""Measure the name-versus-definition check against held-out labels and the catalog.

Two arms, one pass each:

* the labelled arm splits ``name-definition-labels.json`` once, by the seed
  recorded below, into a development third the check's wording was shaped on
  and a held-out two thirds that were not read until the rule set was frozen;
  it reports recall on the mismatched names and the false-flag rate on the
  names labelled as agreeing;
* the catalog arm runs the same frozen check over every published name in the
  catalog checkout and lists the findings by qualifier family, with the
  disposition each flagged name is given.

Usage: ``uv run --no-sync python name-definition-check-run.py``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

from imas_codex.standard_names.audits import name_definition_agreement_check

REPO = Path(__file__).resolve().parents[4]
LABELS = (
    REPO
    / "docs/evidence/fragments/sn-definitions-by-construction/name-definition-labels.json"
)
SECOND_READING = (
    REPO / "docs/evidence/fragments/sn-name-definition-second-reading/labels.json"
)
CATALOG = Path("/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names")
OUT = Path(__file__).with_name("name-definition-check-results.json")
# The seat's per-name verdicts, recorded beside the run's logs. The results file
# also carries them, so the arms can be recomputed from the repository alone
# once the seat has been paid for; the cache copy is the seat's own output.
JUDGMENT_ANSWERS = Path.home() / (
    ".cache/imas-codex-sdbc-name-definition-check/name-definition-judgment-answers.json"
)

# The split seed and the stratified draw. Committed before the check was
# shaped, so the held-out names were fixed before any rule was written.
SPLIT_SEED = 7714
DEV_SHARE = 1 / 3

# Metric floor the measure is judged against.
RECALL_FLOOR = 0.80
FALSE_FLAG_CEILING = 0.10


def split(labelled: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified development/held-out split by the labelled class."""

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in labelled:
        groups[row["class_under_ruling"]].append(row)
    dev: list[dict] = []
    held: list[dict] = []
    for rows in groups.values():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                (str(SPLIT_SEED) + row["name"]).encode()
            ).hexdigest(),
        )
        cut = max(1, round(len(ordered) * DEV_SHARE))
        if rows[0]["class_under_ruling"] == "agrees":
            cut = round(len(ordered) * DEV_SHARE)
        dev.extend(ordered[:cut])
        held.extend(ordered[cut:])
    return dev, held


def flag(row: dict) -> bool:
    return bool(name_definition_agreement_check(row))


def families(row: dict) -> list[str]:
    return [
        finding.split(":")[1].strip()
        for finding in name_definition_agreement_check(row)
    ]


def arm(rows: list[dict]) -> dict:
    mismatched = [row for row in rows if row["class_under_ruling"] != "agrees"]
    agreeing = [row for row in rows if row["class_under_ruling"] == "agrees"]
    caught = [row for row in mismatched if flag(row)]
    flagged_agreeing = [row for row in agreeing if flag(row)]
    return {
        "names": len(rows),
        "mismatched": len(mismatched),
        "mismatched_found": len(caught),
        "recall": round(len(caught) / len(mismatched), 3) if mismatched else None,
        "agreeing": len(agreeing),
        "agreeing_flagged": len(flagged_agreeing),
        "false_flag_rate": (
            round(len(flagged_agreeing) / len(agreeing), 3) if agreeing else None
        ),
        "mismatched_missed": [
            {"name": row["name"], "pattern": row.get("pattern")}
            for row in mismatched
            if not flag(row)
        ],
        "agreeing_flagged_names": [
            {"name": row["name"], "families": families(row)}
            for row in flagged_agreeing
        ],
    }


def published_names(catalog: Path) -> list[dict]:
    """Every published name, with the definition text the catalog stores."""

    names: list[dict] = []
    for path in sorted(catalog.glob("*.yml")):
        import yaml

        payload = yaml.safe_load(path.read_text())
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            names.append(
                {
                    "id": entry.get("name") or entry.get("id"),
                    "description": entry.get("description"),
                    "documentation": entry.get("documentation"),
                    "unit": entry.get("unit"),
                    "file": path.name,
                }
            )
    return names


def abstract_parent_census(names: list[dict]) -> dict:
    """How many published names a reader would call an umbrella, and how many
    of those state one quantity.

    The catalog does not carry a parent flag in the files, so the census reads
    the definition wording: a name whose definition declares an umbrella is
    counted, and of those, the ones whose definition also fixes a single
    population, component or averaging are counted as stating one quantity.
    """

    umbrella = re.compile(
        r"intentionally unspecified by this (?:parent|quantity)|"
        r"unqualified parent quantity|more specific (?:descendants|instances)",
        re.IGNORECASE,
    )
    fixes_one = re.compile(
        r"flux[-\s]surface[-\s]averaged|volume[-\s]averaged|volume[-\s]integrated|"
        r"of (?:one|a) specified|summed over all",
        re.IGNORECASE,
    )
    declared = []
    for name in names:
        text = f"{name.get('description') or ''} {name.get('documentation') or ''}"
        if umbrella.search(text):
            declared.append(
                {
                    "name": name["id"],
                    "states_one_quantity": bool(fixes_one.search(text)),
                }
            )
    return {
        "umbrella_definitions": len(declared),
        "states_one_quantity": sum(1 for row in declared if row["states_one_quantity"]),
        "states_no_single_quantity": sum(
            1 for row in declared if not row["states_one_quantity"]
        ),
        "names": declared,
    }


JUDGMENT_CHECK = "definition_qualifier_absent_from_name"


def judgment_items(rows: list[dict]) -> list[dict]:
    """One judgment item per labelled name, carrying the evidence it reads."""

    return [
        {
            "id": row["name"],
            "description": row["description"],
            "documentation": row["documentation"],
            "unit": row["unit"],
        }
        for row in rows
    ]


def derived_children(names: set[str]) -> dict[str, list[str]]:
    """The pipeline's own derived-parent state, read from the graph.

    A derived parent abstracts over its accepted ``HAS_PARENT`` children, so
    the parent relation is read here rather than inferred from the definition's
    prose. One bounded query over the named cohort; a name with no children is
    absent from the mapping, which is observed-and-empty rather than unknown.
    """

    from imas_codex.graph.client import GraphClient

    rows = GraphClient().query(
        "MATCH (parent:StandardName)<-[:HAS_PARENT]-(child:StandardName) "
        "WHERE parent.id IN $names "
        "RETURN parent.id AS id, collect(child.id) AS children",
        names=sorted(names),
    )
    return {row["id"]: row["children"] for row in rows}


async def judge(rows: list[dict], *, concurrency: int = 8) -> dict[str, dict]:
    """Ask the taxonomy's judgment question of each name on the review seat.

    The state is the one the review builds: the name with its grammar parse
    (``decomposition``) and segments, the description, the documentation and
    the unit, plus the derived-parent evidence the pipeline already carries.
    The question is the taxonomy's own, rendered by the review's helper, so the
    measurement asks exactly what review asks.
    """

    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model
    from imas_codex.standard_names.review.checks import load_review_checks
    from imas_codex.standard_names.review.jev_review import (
        SERVICE,
        _question_def,
        build_name_state,
        neighbourhood_fields,
    )

    checks = load_review_checks()[0]
    checks = list(checks) if not isinstance(checks, list) else checks
    check = next(c for c in checks if c.id == JUDGMENT_CHECK)
    questions = {f"check:{JUDGMENT_CHECK}": _question_def(check.question)}

    extra = ("derived_children", "parent_description")
    neighbourhood = tuple(neighbourhood_fields(checks)) + extra
    items = judgment_items(rows)
    derived = derived_children({item["id"] for item in items})
    for item in items:
        item["derived_children"] = derived.get(item["id"], [])
    states = [build_name_state(item, neighbourhood=neighbourhood) for item in items]

    results, cost = await decide_batch(
        states,
        questions,
        model=get_model("sn-judge"),
        service=SERVICE,
        concurrency=concurrency,
    )
    answers: dict[str, dict] = {}
    for row, result in zip(rows, results, strict=True):
        answers[row["name"]] = (result[0] and result[0].get(f"check:{JUDGMENT_CHECK}")) or None
    return {"answers": answers, "cost_usd": cost}


def _judgment_flags(answers: dict[str, dict], row: dict) -> bool:
    verdict = (answers.get(row["name"]) or {}).get("choice")
    return verdict not in (None, "agrees")


def judgment_arm(rows: list[dict], answers: dict[str, dict]) -> dict:
    """Score the judgment's verdicts against one arm of the labels."""

    mismatched = [row for row in rows if row["class_under_ruling"] != "agrees"]
    agreeing = [row for row in rows if row["class_under_ruling"] == "agrees"]

    def verdict(row: dict) -> str | None:
        answer = answers.get(row["name"])
        if not answer:
            return None
        return answer.get("choice")

    def flagged(row: dict) -> bool:
        return verdict(row) not in (None, "agrees")

    caught = [row for row in mismatched if flagged(row)]
    flagged_agreeing = [row for row in agreeing if flagged(row)]
    unanswered = [row["name"] for row in rows if verdict(row) is None]
    by_pattern: dict[str, dict] = {}
    for row in mismatched:
        bucket = by_pattern.setdefault(
            row.get("pattern") or "-", {"mismatched": 0, "found": 0}
        )
        bucket["mismatched"] += 1
        if flagged(row):
            bucket["found"] += 1
    return {
        "names": len(rows),
        "answered": len(rows) - len(unanswered),
        "unanswered": unanswered,
        "mismatched": len(mismatched),
        "mismatched_found": len(caught),
        "recall": round(len(caught) / len(mismatched), 3) if mismatched else None,
        "agreeing": len(agreeing),
        "agreeing_flagged": len(flagged_agreeing),
        "false_flag_rate": (
            round(len(flagged_agreeing) / len(agreeing), 3) if agreeing else None
        ),
        "by_pattern": dict(sorted(by_pattern.items())),
        "mismatched_missed": [
            {
                "name": row["name"],
                "pattern": row.get("pattern"),
                "verdict": verdict(row),
            }
            for row in mismatched
            if not flagged(row)
        ],
        "agreeing_flagged_names": [
            {"name": row["name"], "verdict": verdict(row)} for row in flagged_agreeing
        ],
    }


def derived_parent_census(names: list[dict]) -> dict:
    """How many published names are derived parents, from the graph state."""

    try:
        children = derived_children({name["id"] for name in names})
    except Exception as exc:  # noqa: BLE001 - the census is not a gate
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "published_names_with_derived_children": len(children),
        "names": sorted(children),
    }


def disposition(row: dict) -> str:
    """The disposition a flagged name is given, from its qualifier family."""

    found = families(row)
    if "abstract-parent-as-quantity" in found:
        return "hold-from-the-cut"
    if "unstated-normalisation" in found or "equivalent-electron-count" in found:
        return "rewrite-the-definition"
    if "unstated-restriction" in found:
        return "rename"
    return "rewrite-the-definition"


def _union(rows: list[dict], judgment: dict[str, dict]) -> dict:
    """The delivered check: the deterministic pass or the judgment, whichever fires."""

    def flagged(row: dict) -> bool:
        verdict = (judgment.get(row["name"]) or {}).get("choice")
        return bool(name_definition_agreement_check(row)) or verdict not in (
            None,
            "agrees",
        )

    mismatched = [row for row in rows if row["class_under_ruling"] != "agrees"]
    agreeing = [row for row in rows if row["class_under_ruling"] == "agrees"]
    found = sum(1 for row in mismatched if flagged(row))
    flagged_agreeing = sum(1 for row in agreeing if flagged(row))
    return {
        "mismatched": len(mismatched),
        "mismatched_found": found,
        "recall": round(found / len(mismatched), 3),
        "agreeing": len(agreeing),
        "agreeing_flagged": flagged_agreeing,
        "false_flag_rate": round(flagged_agreeing / len(agreeing), 3),
    }


def main() -> None:
    labelled = json.loads(LABELS.read_text())
    rows = labelled["rows"]
    dev, held = split(rows)
    second = json.loads(SECOND_READING.read_text())

    dev_arm = arm(dev)
    held_arm = arm(held)

    # The frozen judgment's verdicts, scored against the same splits. It was
    # shaped on the development third, frozen, then scored on the held-out two
    # thirds -- the second time those rows were scored, after the lexical check
    # whose first scoring is why the judgment exists.
    if JUDGMENT_ANSWERS.exists():
        recorded = json.loads(JUDGMENT_ANSWERS.read_text())["by_name"]
    else:
        recorded = json.loads(OUT.read_text())["judgment_verdicts"]
    judgment = {name: value for name, value in recorded.items()}
    dev_judgment = judgment_arm(dev, judgment)
    held_judgment = judgment_arm(held, judgment)

    # The names the two readers disagree on, scored separately on the first
    # reader's ruling column. The second reading joins through the draw's
    # index list, one item per drawn row in draw order.
    indices = labelled["second_reader"]["subset_indices"]
    disagreeing = []
    for item, index in zip(second, indices, strict=False):
        row = rows[index]
        if item["class"] != row["class_under_ruling"]:
            disagreeing.append(row)

    catalog = published_names(CATALOG)
    flagged = []
    by_family: dict[str, int] = collections.Counter()
    for name in catalog:
        findings = name_definition_agreement_check(name)
        if not findings:
            continue
        for finding in findings:
            by_family[finding.split(":")[1].strip()] += 1
        flagged.append(
            {
                "name": name["id"],
                "file": name["file"],
                "unit": name["unit"],
                "families": families(name),
                "findings": findings,
                "disposition": disposition(name),
            }
        )

    result = {
        "schema": "name-definition-check",
        "check": {
            "id": "definition_states_qualifier_name_lacks",
            "implementation": "imas_codex.standard_names.audits.name_definition_agreement_check",
            "reads": "description, documentation, name",
            "calls_a_model": False,
        },
        "labels": {
            "path": str(LABELS.relative_to(REPO)),
            "produced_by": labelled["reader"],
            "labelled_total": len(rows),
            "mismatched_under_ruling": sum(
                1 for row in rows if row["class_under_ruling"] != "agrees"
            ),
            "second_reading": {
                "path": str(SECOND_READING.relative_to(REPO)),
                "produced_by": labelled["second_reader"]["reader"],
                "status": labelled["second_reader"]["status"],
                "exact_class_agreement_under_ruling": labelled["second_reader"][
                    "exact_class_agreement_under_ruling"
                ],
                "joined": labelled["second_reader"]["joined"],
            },
        },
        "split": {
            "seed": SPLIT_SEED,
            "development_share": DEV_SHARE,
            "development_names": len(dev),
            "held_out_names": len(held),
            "shaped_on": "development third only; held-out rows' labels were not read before the rule set was frozen",
        },
        "development_arm": dev_arm,
        "held_out_arm": held_arm,
        "development_arm_judgment": dev_judgment,
        "held_out_arm_judgment": held_judgment,
        "held_out_arm_union": _union(held, judgment),
        "judgment_verdicts": recorded,
        "readers_disagree": {
            "names_scored": len(disagreeing),
            "flagged": [row["name"] for row in disagreeing if flag(row)],
            "judgment_flagged": [
                row["name"] for row in disagreeing if _judgment_flags(judgment, row)
            ],
            "judgment_verdicts": {
                row["name"]: judgment.get(row["name"], {}).get("choice")
                for row in disagreeing
            },
        },
        "catalog_arm": {
            "catalog_path": str(CATALOG),
            "published_names": len(catalog),
            "flagged": len(flagged),
            "by_family": dict(sorted(by_family.items())),
            "names": flagged,
        },
        "abstract_parent_census": abstract_parent_census(catalog),
        "gate": {
            "recall_floor": RECALL_FLOOR,
            "false_flag_ceiling": FALSE_FLAG_CEILING,
            "deterministic_arm": {
                "recall": held_arm["recall"],
                "false_flag_rate": held_arm["false_flag_rate"],
            },
            "judgment_arm": {
                "recall": held_judgment["recall"],
                "false_flag_rate": held_judgment["false_flag_rate"],
            },
            "delivered_check_union": _union(held, judgment),
            "met": False,
            "met_reason": (
                "the delivered check is the deterministic first pass or the "
                "frozen judgment, whichever fires; neither reaches the 80 % "
                "recall floor on the held-out third"
            ),
        },
        "abstract_parent_state": derived_parent_census(catalog),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("development_arm", "held_out_arm", "gate")}, indent=1))
    print("catalog flagged:", len(flagged), "of", len(catalog), result["catalog_arm"]["by_family"])
    print("abstract parent:", {k: v for k, v in result["abstract_parent_census"].items() if k != "names"})


if __name__ == "__main__":
    main()