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


def main() -> None:
    labelled = json.loads(LABELS.read_text())
    rows = labelled["rows"]
    dev, held = split(rows)
    second = json.loads(SECOND_READING.read_text())

    dev_arm = arm(dev)
    held_arm = arm(held)

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
        "readers_disagree": {
            "names_scored": len(disagreeing),
            "flagged": [row["name"] for row in disagreeing if flag(row)],
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
            "recall": held_arm["recall"],
            "false_flag_rate": held_arm["false_flag_rate"],
            "met": bool(
                held_arm["recall"] is not None
                and held_arm["recall"] >= RECALL_FLOOR
                and held_arm["false_flag_rate"] is not None
                and held_arm["false_flag_rate"] <= FALSE_FLAG_CEILING
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("development_arm", "held_out_arm", "gate")}, indent=1))
    print("catalog flagged:", len(flagged), "of", len(catalog), result["catalog_arm"]["by_family"])
    print("abstract parent:", {k: v for k, v in result["abstract_parent_census"].items() if k != "names"})


if __name__ == "__main__":
    main()