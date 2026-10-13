"""Worklist of live standard names whose description embeddings nearly coincide."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.standard_names.audits import _synonym_matches
from imas_codex.standard_names.grammar_adapter import parse_canonical_name

SEGMENTS = {
    "physical_base": "base",
    "subject": "qualifiers",
    "component": "projection",
    "operator": "operators",
    "locus": "locus",
    "process": "mechanism",
}
VERDICTS = ("distinct quantities", "related; needs ruling", "same quantity")


def _grammar_parts(name: str) -> dict[str, Any]:
    try:
        data = parse_canonical_name(name).ir.model_dump(mode="json")
        return {segment: data[key] for segment, key in SEGMENTS.items()}
    except ValueError:
        return {"spelling": name.split("_")}


def _differences(left: str, right: str) -> dict[str, dict[str, Any]]:
    first, second = _grammar_parts(left), _grammar_parts(right)
    if first.keys() != second.keys():
        return {"spelling": {"left": left, "right": right}}
    changes = {
        segment: {"left": value, "right": second[segment]}
        for segment, value in first.items()
        if value != second[segment]
    }
    return changes or {"spelling": {"left": left, "right": right}}


def _overlap_key(changes: Mapping[str, Mapping[str, Any]]) -> str:
    """Group the same grammar substitution across different quantity names."""
    substitutions = [
        (
            segment,
            sorted(
                (
                    json.dumps(change["left"], sort_keys=True),
                    json.dumps(change["right"], sort_keys=True),
                )
            ),
        )
        for segment, change in sorted(changes.items())
    ]
    return json.dumps(substitutions, separators=(",", ":"))


def _group_overlaps(pairs: list[dict[str, Any]]) -> None:
    """Group one varying grammar axis without chaining unrelated axes."""
    contexts: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        changes = pair["divergent_segments"]
        first = _grammar_parts(pair["left"])
        if "spelling" in changes:
            context = json.dumps((pair["left"], pair["right"]))
        else:
            fixed = {
                segment: value
                for segment, value in first.items()
                if segment not in changes
            }
            context = json.dumps((pair["unit"], sorted(changes), fixed), sort_keys=True)
        pair["_context"] = context
        contexts.setdefault(context, []).append(pair)

    cluster_keys: dict[int, str] = {}
    for context, members in contexts.items():
        remaining = set(range(len(members)))
        while remaining:
            component = {remaining.pop()}
            names = {
                name
                for index in component
                for name in (members[index]["left"], members[index]["right"])
            }
            while True:
                connected = {
                    index
                    for index in remaining
                    if names & {members[index]["left"], members[index]["right"]}
                }
                if not connected:
                    break
                component.update(connected)
                remaining.difference_update(connected)
                names.update(
                    name
                    for index in connected
                    for name in (members[index]["left"], members[index]["right"])
                )
            if len(component) > 1:
                identities = sorted(
                    (members[index]["left"], members[index]["right"])
                    for index in component
                )
                key = f"context:{context}:{json.dumps(identities)}"
                for index in component:
                    cluster_keys[id(members[index])] = key

    groups: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        pair.pop("_context")
        key = cluster_keys.get(id(pair), f"substitution:{pair['overlap']}")
        groups.setdefault(key, []).append(pair)
    for members in groups.values():
        identities = sorted((pair["left"], pair["right"]) for pair in members)
        digest = hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:12]
        for pair in members:
            pair["overlap"] = f"overlap-{digest}"
            pair["overlap_size"] = len(members)


def build_worklist(
    names: Sequence[Mapping[str, Any]],
    *,
    west_names: set[str] | frozenset[str] = frozenset(),
    threshold: float = 0.95,
) -> dict[str, list[dict[str, Any]]]:
    """Build same-unit overlaps using the audit's unrounded cosine comparison."""
    members: list[dict[str, Any]] = []
    for name in names:
        if name.get("unit") is None or name.get("embedding") is None:
            continue
        members.append(
            {
                "name": str(name.get("name") or name["id"]),
                "unit": str(name["unit"]),
                "description": name.get("description") or "",
                "description_embedding": name["embedding"],
            }
        )

    worklist: dict[str, list[dict[str, Any]]] = {"west": [], "other": []}
    members.sort(key=lambda row: row["name"])
    for index, candidate in enumerate(members):
        for existing, cosine in _synonym_matches(
            {"id": candidate["name"], **candidate},
            members[index + 1 :],
            same_unit_only=True,
            threshold=threshold,
            inclusive=True,
        ):
            left, right = candidate["name"], existing["name"]
            changes = _differences(left, right)
            pair = {
                "left": left,
                "right": right,
                "unit": candidate["unit"],
                "cosine": round(cosine, 6),
                "description_left": candidate["description"],
                "description_right": existing["description"],
                "divergent_segments": changes,
                "overlap": _overlap_key(changes),
            }
            scope = "west" if left in west_names or right in west_names else "other"
            worklist[scope].append(pair)
    _group_overlaps(worklist["west"] + worklist["other"])
    return worklist


def _questions() -> dict[str, Any]:
    questions: dict[str, Any] = {
        "quantity_relation": {
            "type": "score",
            "instructions": (
                "Compare the physical quantities named by these two standard names, "
                "using their descriptions and grammar. Score semantic identity, "
                "not spelling similarity. A lexical overlap alone is not identity."
            ),
            "criteria": list(VERDICTS),
        }
    }
    for segment in SEGMENTS:
        questions[f"differs_{segment}"] = {
            "type": "noul",
            "instructions": (
                f"Compare only grammar_segments.left.{segment} with "
                f"grammar_segments.right.{segment}. Answer the probability that "
                "these two segment values differ. Exact equality means no, even "
                "when other segments or the full names differ."
            ),
        }
    return questions


async def judge_worklist(
    pairs: Sequence[dict[str, Any]], *, model: str
) -> tuple[list[dict[str, Any]], float]:
    """Attach Jev's graded verdict and per-segment judgments to each pair."""
    states = [
        {
            "left": {"name": pair["left"], "description": pair["description_left"]},
            "right": {"name": pair["right"], "description": pair["description_right"]},
            "unit": pair["unit"],
            "grammar_segments": {
                "left": _grammar_parts(pair["left"]),
                "right": _grammar_parts(pair["right"]),
            },
            "divergent_segments": pair["divergent_segments"],
        }
        for pair in pairs
    ]
    results, cost = await decide_batch(
        states, _questions(), model=model, service="standard-names"
    )
    judged: list[dict[str, Any]] = []
    for pair, result in zip(pairs, results, strict=True):
        item = dict(pair)
        if result is None:
            item.update(
                verdict=None, same_quantity_probability=None, segment_judgments={}
            )
        else:
            answers, spent = result
            score = answers["quantity_relation"]
            probabilities = score.get("probabilities") or {}
            level = (
                max(
                    range(3),
                    key=lambda index: float(
                        probabilities.get(index, probabilities.get(str(index), 0))
                    ),
                )
                if probabilities
                else round(float(score["score"]))
            )
            item.update(
                verdict=VERDICTS[level],
                same_quantity_probability=float(
                    probabilities.get(2, probabilities.get("2", 0))
                ),
                segment_judgments={
                    segment: float(answers[f"differs_{segment}"]["noul"])
                    for segment in SEGMENTS
                },
                judgment_cost=spent,
            )
        judged.append(item)
    judged.sort(
        key=lambda item: (
            -(VERDICTS.index(item["verdict"]) if item["verdict"] in VERDICTS else -1),
            -float(item["same_quantity_probability"] or 0),
            item["overlap"],
            item["left"],
            item["right"],
        )
    )
    return judged, cost
