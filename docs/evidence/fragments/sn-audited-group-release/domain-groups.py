"""Census, ranking and release proposal for the live physics-domain groups.

The measurement has three parts and one proposal.

*Census*: every live name is counted once, in the row of the stored
``physics_domain`` it carries, and the names that carry none get a row of their
own, so the rows sum to the live names read in the same run. A name's membership
is *settled* when its stored ``physics_domain`` is one of the domains its own
``source_domains`` records.

*Ranking*: groups are ordered by the share of their live names whose membership
is settled — the reading the plan's own opening paragraph already quotes
("equilibrium 200 of 210"). Two further readings are computed over the same
table to show what the order does and does not depend on: the settled share of
the names that record a source domain at all (it removes the names that could
never agree from the denominator), and the settled share against the domains of
the name's *current* DD sources rather than the append-only accumulator.

*Wrong-domain counts*: read from the section-1 audit's own records, joined to a
group by the name's stored domain. No rate is stated for a group: each group
contributes 14 or 15 names to that sample, and the script shows the narrowest
95 % interval such a sample can carry, so the audit's own rule — a rate only
where the interval is narrower than twenty percentage points — admits none.

*Proposal*: the release order, the standing-or-split choice for ``transport``,
and one of three clearing ways for every family whose members sit in more than
one group. The lead rules; this module only puts the evidence beside each
option.

Read-only on the graph. Two reads: ``audit_domains`` for the per-name domain
state, and one bounded read of the live names' DD source paths, which the audit
result does not carry and family detection needs. Nothing is written to the
graph.
"""

from __future__ import annotations

import collections
import hashlib
import html
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iter_codex.standard_names.benchmark_roles import spearman_rho
from iter_codex.standard_names.domain_audit import (
    LIVE_NAME_STAGES,
    DomainAuditResult,
    NameDomainState,
    audit_domains,
)
from iter_codex.standard_names.domain_ranking import select_primary_domain
from iter_codex.standard_names.families import detect_families

FRAGMENTS = Path(__file__).resolve().parent
OUT_JSON = FRAGMENTS / "domain-groups.json"
OUT_HTML = FRAGMENTS / "sagr-domain-groups.html"
AUDIT_JSON = FRAGMENTS / "blind-audit.json"

#: The row the names that carry no stored physics domain are counted in. It is a
#: row of the census, never a group: no release can be cut from it.
NO_DOMAIN = "(no physics domain)"

#: A group's settled share is ranked only at this many live names — the same
#: floor section 1 used to decide which groups its sample had to cover. Below
#: it a share moves by a whole step per name and measures nothing.
SAMPLE_FLOOR = 20

WRONG_DOMAIN_CLASS = "wrong_physics_domain"

#: The live names' DD source paths. The audit result keeps the count of DD
#: sources per name but not the paths, and a family is a set of DD paths, so the
#: mapping is read once here. Row set: one row per live name's source-to-DD-path
#: hop, a few thousand rows over an indexed edge.
_SOURCE_PATHS_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
RETURN dd.id AS dd_path,
       dd.unit AS unit,
       sn.id AS sn_id,
       sn.physics_domain AS physics_domain
ORDER BY sn_id, dd_path
"""


# ---------------------------------------------------------------------------
# Census and ranking
# ---------------------------------------------------------------------------


def group_of(state: NameDomainState) -> str:
    """The census row a name counts in: its stored group, or the no-domain row."""
    return state.physics_domain or NO_DOMAIN


def census(states: tuple[NameDomainState, ...] | list[NameDomainState]) -> dict:
    """Count every name once, in its stored group's row.

    ``settled`` is the done-when's "domain is among its sources' domains": the
    name's stored ``physics_domain`` is a member of the ``source_domains`` the
    name itself records. A name that records no source domain cannot be settled
    and is counted in ``no_sources``, so the rows still sum to the live names.
    """
    rows: dict[str, dict[str, Any]] = {}
    for state in states:
        group = group_of(state)
        row = rows.setdefault(
            group,
            {
                "group": group,
                "live": 0,
                "records_domains": 0,
                "settled": 0,
                "not_among": 0,
                "no_sources": 0,
                "settled_expected": 0,
            },
        )
        row["live"] += 1
        if state.source_domains:
            row["records_domains"] += 1
            if state.physics_domain and state.physics_domain in state.source_domains:
                row["settled"] += 1
            else:
                row["not_among"] += 1
        else:
            row["no_sources"] += 1
        if state.physics_domain and state.physics_domain in state.expected:
            row["settled_expected"] += 1
    return rows


def measures(row: dict[str, Any]) -> dict[str, float | None]:
    """The three settled shares of one census row, None where undefined."""
    live = row["live"]
    recorded = row["records_domains"]
    return {
        "m1": row["settled"] / live if live else None,
        "m2": row["settled"] / recorded if recorded else None,
        "m3": row["settled_expected"] / live if live else None,
    }


def ranked_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The groups, ranked by settled share, ties by size then name.

    The no-domain row is not a group and is never ranked. A group below
    :data:`SAMPLE_FLOOR` names is reported but not ranked.
    """
    table: list[dict[str, Any]] = []
    for group, row in rows.items():
        if group == NO_DOMAIN:
            continue
        entry = {**row, **measures(row)}
        entry["ranked"] = row["live"] >= SAMPLE_FLOOR
        table.append(entry)
    table.sort(
        key=lambda entry: (
            0 if entry["ranked"] else 1,
            -(entry["m1"] or 0.0),
            -entry["live"],
            entry["group"],
        )
    )
    rank = 0
    for entry in table:
        if entry["ranked"]:
            rank += 1
            entry["rank"] = rank
        else:
            entry["rank"] = None
    return table


def _rank_agreement(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank agreement over the ranked groups, ``None`` when undefined.

    :func:`spearman_rho` is the repository's one implementation; it returns 0.0
    for degenerate input, which here would print as a measured agreement of zero
    where nothing was measured, so that sentinel is converted back to ``None``.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return spearman_rho(xs, ys)


def sensitivity(table: list[dict[str, Any]]) -> dict[str, Any]:
    """How the order moves when the measure is redefined three ways."""
    ranked = [entry for entry in table if entry["ranked"]]
    m1 = [entry["m1"] or 0.0 for entry in ranked]
    m2 = [entry["m2"] or 0.0 for entry in ranked]
    m3 = [entry["m3"] or 0.0 for entry in ranked]

    def order_by(key: str) -> list[str]:
        return [
            entry["group"]
            for entry in sorted(
                ranked,
                key=lambda entry: (
                    -(entry[key] or 0.0),
                    -entry["live"],
                    entry["group"],
                ),
            )
        ]

    order_m1 = [entry["group"] for entry in ranked]
    order_m2 = order_by("m2")
    order_m3 = order_by("m3")
    pos2 = {group: i for i, group in enumerate(order_m2, start=1)}
    pos3 = {group: i for i, group in enumerate(order_m3, start=1)}
    moves = [
        {
            "group": entry["group"],
            "rank_m1": entry["rank"],
            "rank_m2": pos2[entry["group"]],
            "rank_m3": pos3[entry["group"]],
            "move_m2": pos2[entry["group"]] - entry["rank"],
            "move_m3": pos3[entry["group"]] - entry["rank"],
        }
        for entry in ranked
    ]
    return {
        "rho_m1_m2": _rank_agreement(m1, m2),
        "rho_m1_m3": _rank_agreement(m1, m3),
        "rho_m2_m3": _rank_agreement(m2, m3),
        "largest_move_m2": max(abs(move["move_m2"]) for move in moves) if moves else 0,
        "largest_move_m3": max(abs(move["move_m3"]) for move in moves) if moves else 0,
        "moves": moves,
        "order_m1": order_m1,
        "order_m2": order_m2,
        "order_m3": order_m3,
    }


# ---------------------------------------------------------------------------
# Section-1 audit records: names it read as wrong-domain
# ---------------------------------------------------------------------------


def _section_one_wilson() -> Any:
    """The Wilson interval section 1 reports, from the sibling blind-audit run.

    ``blind-audit.py`` is a script rather than an importable module (its file
    name is not a valid module name), so it is loaded by path. It carries a
    ``__main__`` guard, so loading it executes only its definitions.
    """
    spec = importlib.util.spec_from_file_location(
        "sn_audited_group_release_blind_audit", FRAGMENTS / "blind-audit.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the sibling blind-audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wilson


#: The 95 % interval section 1's per-group counts are stated with, read from the
#: sibling fragment so both report one interval rather than two copies of the
#: same arithmetic.
WILSON = _section_one_wilson()


def narrowest_group_interval(sampled: int) -> dict[str, float]:
    """The narrowest 95 % interval a group's own count from that sample can carry.

    The best case is no wrong-domain name at all, so the width reported is the
    ceiling of what the group's sample supports.
    """
    low, high = WILSON(0, sampled)
    return {
        "sampled": sampled,
        "width_pp_best_case": (high - low) * 100.0,
        "upper_pp_best_case": high * 100.0,
    }


def wrong_domain_counts(
    audit_path: Path, states: tuple[NameDomainState, ...]
) -> dict[str, Any]:
    """Join the section-1 wrong-domain readings to the census's groups.

    A name is counted in ``any`` when at least one reader assigned the class and
    in ``both`` when both did — the two counts section 1 publishes per class. A
    name the section-1 sample read but the live census no longer carries is
    reported unmapped rather than dropped silently.
    """
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    group_of_name = {state.sn_id: group_of(state) for state in states}
    per_group: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"any": 0, "both": 0, "names": []}
    )
    unmapped: list[str] = []
    for record in payload.get("defective_names") or []:
        a = WRONG_DOMAIN_CLASS in (record.get("A") or [])
        b = WRONG_DOMAIN_CLASS in (record.get("B") or [])
        if not (a or b):
            continue
        group = group_of_name.get(record["name"])
        if group is None:
            unmapped.append(record["name"])
            continue
        per_group[group]["any"] += 1
        if a and b:
            per_group[group]["both"] += 1
        per_group[group]["names"].append(record["name"])
    sampled = {
        group: int(entry.get("sampled") or 0)
        for group, entry in (payload.get("groups") or {}).items()
    }
    intervals = {
        group: narrowest_group_interval(n) for group, n in sampled.items() if n
    }
    planted_rows = (payload.get("planted_detail") or {}).get("rows") or []
    plant_names = {row.get("id") for row in planted_rows if row.get("id")}
    sampled_names = {record["name"] for record in payload.get("defective_names") or []}
    return {
        "source": audit_path.name,
        "generated_at": payload.get("generated_at"),
        "names_read": payload.get("names_read"),
        "per_group": dict(per_group),
        "sampled": sampled,
        "intervals": intervals,
        "unmapped": sorted(unmapped),
        "planted_rows": len(planted_rows),
        "planted_in_sample": sorted(plant_names & sampled_names),
        "sample_total_wrong_domain": sum(entry["any"] for entry in per_group.values()),
        "sample_total_wrong_domain_both": sum(
            entry["both"] for entry in per_group.values()
        ),
        "class_total_from_audit": (
            (payload.get("per_class") or {}).get(WRONG_DOMAIN_CLASS) or {}
        ).get("any"),
    }


# ---------------------------------------------------------------------------
# Families: DD structural sets whose live names sit in more than one group
# ---------------------------------------------------------------------------


def family_evidence(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect DD families over the paths that produced live names.

    ``detect_families`` owns what a family is; this supplies the population —
    every DD path a live name was produced from — and reads each detected
    family's live names back onto their stored groups. A family whose live names
    sit in one group spans nothing and is not reported.
    """
    path_names: dict[str, set[str]] = collections.defaultdict(set)
    path_unit: dict[str, str] = {}
    name_group: dict[str, str] = {}
    for row in source_rows:
        path = row.get("dd_path")
        name = row.get("sn_id")
        if not path or not name:
            continue
        path_names[path].add(name)
        path_unit[path] = row.get("unit") or ""
        name_group[name] = row.get("physics_domain") or NO_DOMAIN

    items = [{"path": path, "unit": path_unit[path]} for path in sorted(path_names)]
    families = detect_families(items)

    spanning: list[dict[str, Any]] = []
    for family in families:
        members = sorted(
            member.dd_path for member in family.members if member.dd_path in path_names
        )
        by_group: dict[str, set[str]] = collections.defaultdict(set)
        for path in members:
            for name in path_names[path]:
                by_group[name_group[name]].add(name)
        named_groups = {
            group: names for group, names in by_group.items() if group != NO_DOMAIN
        }
        if len(named_groups) < 2:
            continue
        suffixes = [path.rsplit("/", 1)[-1] for path in members]
        digest = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()[:8]
        family_id = "|".join([family.parent_path, family.family_type, digest])
        spanning.append(
            {
                "id": family_id,
                "parent_path": family.parent_path,
                "family_type": family.family_type,
                "members": members,
                "suffixes": suffixes,
                "member_groups": {
                    group: sorted(names) for group, names in by_group.items()
                },
                "group_counts": {
                    group: len(names)
                    for group, names in sorted(
                        by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])
                    )
                },
                "named_members": sum(len(names) for names in named_groups.values()),
            }
        )
    spanning.sort(key=lambda entry: (entry["parent_path"], entry["id"]))
    return {
        "detected": len(families),
        "source_paths": len(path_names),
        "spanning": spanning,
        "member_check": family_member_check(spanning, path_names),
    }


def family_member_check(
    families: list[dict[str, Any]], path_names: dict[str, set[str]]
) -> dict[str, Any]:
    """Recount every family's live names from its paths and refuse a repeat.

    A name produced from two paths of one family is one member of that family,
    and the aggregation above enforces that. This recomputes each family's
    population from the recorded paths and compares it with what was recorded,
    so the per-path append that counts such a name once per path shows up as a
    repeated name and as a total above the family's distinct names.
    """
    repeated: list[dict[str, Any]] = []
    miscounted: list[dict[str, Any]] = []
    members = 0
    for family in families:
        seen = {name for path in family["members"] for name in path_names.get(path, ())}
        listed = [name for names in family["member_groups"].values() for name in names]
        members += len(seen)
        duplicates = sorted({name for name in listed if listed.count(name) > 1})
        if duplicates:
            repeated.append({"family": family["id"], "names": duplicates})
        elif len(listed) != len(seen) or set(listed) != seen:
            miscounted.append(
                {
                    "family": family["id"],
                    "listed": len(listed),
                    "distinct": len(seen),
                }
            )
    return {
        "families": len(families),
        "distinct_members": members,
        "repeated": repeated,
        "miscounted": miscounted,
        "passed": not repeated and not miscounted,
    }


def clearing_proposal(
    family: dict[str, Any], order_position: dict[str, int]
) -> dict[str, Any]:
    """One of the three ways the done-when allows a spanning family to be cleared.

    The choices are deterministic and stated so the lead can overrule any one of
    them:

    ``bring_together`` — one group holds at least two thirds of the family's
    named members (and at least two of them), so the family is one quantity and
    the stray members are brought into that group.

    ``cut_together`` — no group has that majority, and the two largest groups sit
    within three places of each other in the proposed order, so the two groups
    are audited and cut as one candidate.

    ``exception`` — neither holds: the family spans groups that are far apart in
    the order, and it is ruled an exception that may span groups.
    """
    counts = family["group_counts"]
    named = [(group, count) for group, count in counts.items() if group != NO_DOMAIN]
    total = sum(count for _, count in named)
    top_group, top_count = named[0]
    second_group, _second_count = named[1] if len(named) > 1 else (None, 0)
    if top_count >= 2 and top_count * 3 >= total * 2:
        return {
            "choice": "bring_together",
            "into": top_group,
            "moves": total - top_count,
            "rule": "one group holds at least two thirds of the named members",
        }
    first = order_position.get(top_group)
    second = order_position.get(second_group) if second_group else None
    if first is not None and second is not None and abs(first - second) <= 2:
        return {
            "choice": "cut_together",
            "groups": [top_group, second_group],
            "rule": "the two largest groups are within three places in the order",
        }
    return {
        "choice": "exception",
        "groups": [group for group, _ in named],
        "rule": "no two-thirds majority and the largest groups are far apart",
    }


# ---------------------------------------------------------------------------
# transport: stand, or split
# ---------------------------------------------------------------------------


def transport_evidence(states: tuple[NameDomainState, ...]) -> dict[str, Any]:
    """What a split of ``transport`` would do, measured from its own sources.

    The split measured is the mechanical one: hand each transport name to the
    highest-ranked domain among its other source domains — the rule that already
    selects a name's stored domain — so the split costs no new taxonomy. A name
    whose only source domain is ``transport`` stays.
    """
    members = [state for state in states if state.physics_domain == "transport"]
    moves: dict[str, list[str]] = collections.defaultdict(list)
    stays: list[str] = []
    for state in members:
        others = sorted(state.source_domains - {"transport"})
        try:
            target = select_primary_domain(others)
        except ValueError:
            target = None
        if target:
            moves[target].append(state.sn_id)
        else:
            stays.append(state.sn_id)
    ordered = sorted(moves.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return {
        "live": len(members),
        "settled": sum(1 for state in members if "transport" in state.source_domains),
        "records_no_source": sum(1 for state in members if not state.source_domains),
        "stays": len(stays),
        "moves": {
            group: {"count": len(names), "names": sorted(names)}
            for group, names in ordered
        },
        "split_total": sum(len(names) for names in moves.values()),
    }


def transport_structure(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What ``transport``'s members draw their sources from, by top-level IDS.

    The group's membership is described in the plan's opening paragraph as an
    operator class fed by profile, source, distribution and summary structures;
    this counts the names that touch each IDS so the standing option can be read
    as what it contains rather than asserted.
    """
    names_by_ids: dict[str, set[str]] = collections.defaultdict(set)
    sources_by_ids: dict[str, int] = collections.Counter()
    for row in source_rows:
        if row.get("physics_domain") != "transport":
            continue
        path = row.get("dd_path") or ""
        top = path.split("/", 1)[0] if path else "(no path)"
        names_by_ids[top].add(row["sn_id"])
        sources_by_ids[top] += 1
    ordered = sorted(names_by_ids, key=lambda top: (-len(names_by_ids[top]), top))
    return [
        {
            "ids": top,
            "names": len(names_by_ids[top]),
            "sources": sources_by_ids[top],
        }
        for top in ordered
    ]


# ---------------------------------------------------------------------------
# The instrument checks
# ---------------------------------------------------------------------------


def _synthetic(name: str, group: str, sources: frozenset[str]) -> NameDomainState:
    return NameDomainState(
        sn_id=name,
        name_stage="accepted",
        physics_domain=group,
        source_domains=sources,
        expected=sources,
        edge_domains=sources,
        stored_choice=None,
        correction=None,
        decided_domain=None,
        undecided_reason=None,
        decision_candidates=(),
        dd_source_count=len(sources),
        is_derived_parent=False,
        unnamed_edge_count=0,
    )


def census_plant_check(
    states: tuple[NameDomainState, ...], table: list[dict[str, Any]]
) -> dict[str, Any]:
    """Plant two names and show the census counting each where it belongs.

    One planted name carries a domain among its sources' domains and one carries
    a domain that is not. The census must raise the target group's settled count
    by exactly one and its not-among count by exactly one, so a join that
    matched nothing could not read as agreement. As a second arm the two plants
    are recounted with their source sets emptied: there the settled count must
    not move at all, which shows the comparison is on the source set and not on
    the presence of a row.
    """
    target = table[0]["group"]
    other = table[1]["group"] if len(table) > 1 else "not_a_group"
    agree = _synthetic(
        "census_probe_settled_member", target, frozenset({target, other})
    )
    disagree = _synthetic("census_probe_unsettled_member", target, frozenset({other}))

    base = census(states)
    planted = census((*states, agree, disagree))
    base_row = base.get(target, {})
    planted_row = planted[target]
    delta_settled = planted_row["settled"] - base_row.get("settled", 0)
    delta_not_among = planted_row["not_among"] - base_row.get("not_among", 0)
    delta_live = planted_row["live"] - base_row.get("live", 0)

    emptied = census(
        (
            _synthetic(agree.sn_id, target, frozenset()),
            _synthetic(disagree.sn_id, target, frozenset()),
        )
    )
    emptied_row = emptied[target]
    ok = (
        delta_settled == 1
        and delta_not_among == 1
        and delta_live == 2
        and emptied_row["settled"] == 0
        and emptied_row["no_sources"] == 2
    )
    return {
        "target_group": target,
        "delta_live": delta_live,
        "delta_settled": delta_settled,
        "delta_not_among": delta_not_among,
        "emptied_sources_settled": emptied_row["settled"],
        "emptied_sources_no_source": emptied_row["no_sources"],
        "passed": ok,
    }


# ---------------------------------------------------------------------------
# The proposal: release order
# ---------------------------------------------------------------------------


def proposed_order(table: list[dict[str, Any]], transport_live: int) -> dict[str, Any]:
    """The order put to the lead, with the reason each group sits where it does.

    M1's order within the ranked groups, with two constraints stated from the
    evidence: ``transport`` last among them (largest group, least settled), and
    every group below the floor after them (its rate cannot be sampled, so it
    must be read in full before section 3's bar can be applied at all).
    """
    ranked = [entry for entry in table if entry["ranked"]]
    order = [entry["group"] for entry in ranked if entry["group"] != "transport"]
    order.append("transport")
    below = [entry for entry in table if not entry["ranked"]]
    below.sort(key=lambda entry: (-entry["live"], entry["group"]))
    order.extend(entry["group"] for entry in below)
    reasons = {
        entry["group"]: (
            f"M1 rank {entry['rank']}: {entry['settled']} of {entry['live']} "
            f"settled ({_pct(entry['m1'])})"
        )
        for entry in ranked
    }
    if "transport" in reasons:
        reasons["transport"] = (
            f"last of the ranked groups ({transport_live} live names): "
            + reasons["transport"]
        )
    else:
        reasons["transport"] = f"last of the ranked groups: {transport_live} live names"
    for entry in below:
        noun = "name" if entry["live"] == 1 else "names"
        reasons[entry["group"]] = (
            f"below the {SAMPLE_FLOOR}-name floor: {entry['live']} live {noun}, "
            f"{entry['settled']} settled; must be read in full before a bar applies"
        )
    return {
        "proposed": order,
        "reasons": reasons,
        "ranked_count": len(ranked),
        "reasons_source": "the measured order put to the lead before the ruling",
        "below_floor": [entry["group"] for entry in below],
    }


#: The rulings recorded on the plan at commit 7b7bf5fbb (2026-11-17), section 2
#: for the order, the transport choice and the families, section 6 for what
#: ``transport`` is divided along instead.
FIRST_GROUP = "equilibrium"
TRANSPORT_PARTS_LABEL = "transport's parts (section 6)"
FAMILY_RULING = "exception for the first cut"
RULING = {
    "date": "2026-11-17",
    "source": "plan sn-audited-group-release section 2 and section 6, commit 7b7bf5fbb",
    "order_statement": (
        "equilibrium is released first and the remaining groups follow the "
        "measured order"
    ),
    "transport": (
        "transport does not stand as one group and is not split by its sources' "
        "domains; it is divided along the data dictionary's own structures, which "
        "returns a vocabulary the domain list does not hold today and is carried by "
        "section 6; its parts are released after the parts of the other groups"
    ),
    "families": (
        "no family clearing moves a stored domain until section 6 has measured "
        "reading a name's group from its sources' data structures over the whole "
        "catalog; until then every family that spans groups is an exception for the "
        "first cut, and the families that touch equilibrium are listed by name"
    ),
    "repair_ceiling_usd": 300,
}


def ruled_order(order: dict[str, Any]) -> dict[str, Any]:
    """Section 2's ruled position of every group, against the measured order.

    The first group is ruled, not measured: ``equilibrium`` is released first and
    every other group follows the order measured here. ``transport`` leaves this
    list as a group — section 2 divides it along data structures under section 6
    and releases its parts last — so the ruled sequence ends with one entry that
    is not a group.
    """
    rest = [
        group for group in order["proposed"] if group not in (FIRST_GROUP, "transport")
    ]
    return {
        "first": FIRST_GROUP,
        "rest": rest,
        "last": TRANSPORT_PARTS_LABEL,
        "sequence": [FIRST_GROUP, *rest, TRANSPORT_PARTS_LABEL],
        "measured_place": {
            group: position for position, group in enumerate(order["proposed"], start=1)
        },
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def _lead_symbol(value: float | None) -> str:
    return "—" if value is None else f"ρ = {value:.2f}"


def render_page(payload: dict[str, Any]) -> str:
    node = "sagr-domain-groups"
    census = payload["census"]
    rows = census["rows"]
    partition = census["partition"]
    table = payload["ranking"]["groups"]
    sens = payload["ranking"]["sensitivity"]
    audit = payload["wrong_domain"]
    families = payload["families"]["spanning"]
    transport = payload["transport"]
    checks = payload["instrument"]
    environment = payload["environment"]

    parts: list[str] = []
    parts.append('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n')
    parts.append('<meta name="docs-project" content="iter-codex">\n')
    parts.append('<meta name="reckon-type" content="evidence-fragment">\n')
    parts.append('<meta name="plan-slug" content="sn-audited-group-release">\n')
    parts.append(f'<meta name="node-id" content="{node}">\n')
    parts.append("<title>The physics-domain groups measured before release</title>\n")
    parts.append("</head>\n<body>\n<main>\n")

    top = table[0]
    ranked_table = [entry for entry in table if entry["ranked"]]
    tail = ranked_table[-1] if ranked_table else table[-1]
    no_domain_row = next(
        (row for row in rows if row["group"] == NO_DOMAIN), {"live": 0}
    )
    headline = (
        f"{partition['live']} live names in {len(table)} physics-domain groups, "
        f"{no_domain_row['live']} of them in no group. "
        f"<strong>{partition['settled']} of {partition['live']} names "
        f"({_pct(partition['settled_share'])}) carry a domain that is one of "
        "their sources'</strong>, and so do "
        f"{_pct(partition['settled_share_recorded'])} of the names that record a "
        "source domain at all. Ranked by that share, "
        f"<code>{_esc(top['group'])}</code> leads "
        f"({top['settled']} of {top['live']}, {_pct(top['m1'])}) and "
        f"<code>{_esc(tail['group'])}</code> trails "
        f"({tail['settled']} of {tail['live']}, {_pct(tail['m1'])}). "
        f"{len(families)} DD families have live names in more than one group, and "
        f"<code>transport</code> holds {transport['live']} names at "
        f"{_pct(transport['settled'] / transport['live'] if transport['live'] else None)} "
        "settled. "
        "Ruled 2026-11-17: <code>equilibrium</code> is released first and the "
        "remaining groups follow the measured order; <code>transport</code> does not "
        "stand and is not split by its sources' domains but divided along the data "
        "dictionary's own structures under section 6, its parts last; and every "
        f"family that spans groups is recorded as an {FAMILY_RULING}, with the proposal "
        "kept beside "
        "it, until section 6 measures reading a name's group from its sources."
    )
    parts.append(
        f'<h1 id="{node}-title">The physics-domain groups measured before release</h1>\n'
    )
    parts.append(f'<p id="{node}-summary">{headline}</p>\n')

    # -- census
    parts.append(
        f'<h2 id="{node}-census-head">Census: every live name in one row</h2>\n'
    )
    parts.append(
        f'<p id="{node}-census-note">Every live name was read in one pass at stage '
        f"{_esc(', '.join(environment['live_stages']))}; a name is counted once, in "
        "the row of the stored domain it carries, and the names that carry none get "
        "a row of their own, so the rows sum to the live names read in the same run. "
        "<em>Settled</em> means the stored domain is a member of the source domains "
        "the name itself records; a name that records none cannot be settled and is "
        "counted in its own column. The last two columns are section 1's "
        f"wrong-domain readings, joined by the name's stored domain.</p>\n"
    )
    census_head = (
        "<tr><th>Group</th><th>Live</th><th>Records source domains</th>"
        "<th>Settled: domain among its sources'</th><th>Not among</th>"
        "<th>No source domain</th><th>Settled against current sources</th>"
        "<th>&#167;1 wrong-domain, any reader</th>"
        "<th>&#167;1 wrong-domain, both readers</th></tr>"
    )
    body: list[str] = []
    for row in rows:
        wrong = audit["per_group"].get(row["group"], {"any": 0, "both": 0})
        body.append(
            "<tr>"
            f"<td>{_esc(row['group'])}</td><td>{row['live']}</td>"
            f"<td>{row['records_domains']}</td><td>{row['settled']}</td>"
            f"<td>{row['not_among']}</td><td>{row['no_sources']}</td>"
            f"<td>{row['settled_expected']}</td>"
            f"<td>{wrong['any']}</td><td>{wrong['both']}</td>"
            "</tr>"
        )
    body.append(
        "<tr><td>total</td>"
        f"<td>{partition['live']}</td><td>{partition['records_domains']}</td>"
        f"<td>{partition['settled']}</td><td>{partition['not_among']}</td>"
        f"<td>{partition['no_sources']}</td><td>{partition['settled_expected']}</td>"
        f"<td>{audit['sample_total_wrong_domain']}</td>"
        f"<td>{audit['sample_total_wrong_domain_both']}</td></tr>"
    )
    parts.append(
        f'<table id="{node}-census-table"><thead>{census_head}</thead><tbody>'
        + "\n".join(body)
        + "</tbody></table>\n"
    )

    # -- ranking
    parts.append(
        f'<h2 id="{node}-ranking-head">Ranking by how well membership is defined</h2>\n'
    )
    parts.append(
        f'<p id="{node}-measure">The measure is <strong>M1: the share of a group\'s '
        "live names whose stored domain is one of the domains their own "
        "<code>source_domains</code> records</strong> — the reading the plan's "
        "opening paragraph already quotes. M2 changes only the denominator: the "
        "same question asked of the names that record a source domain at all. M3 "
        f"changes the source set: the domains of the name's <em>current</em> DD "
        "sources, which the domain audit derives, instead of the append-only "
        "accumulator.</p>\n"
    )
    parts.append(
        f'<p id="{node}-floor">A group of fewer than {SAMPLE_FLOOR} live names is '
        "reported but not ranked: below that floor a share moves by a whole step per "
        "name, which is the floor section 1 used to decide which groups its sample "
        "had to cover. The no-domain row is not a group and is never ranked.</p>\n"
    )
    rank_head = (
        "<tr><th>Rank</th><th>Group</th><th>Live</th><th>Settled</th>"
        "<th>M1 settled / live</th><th>M2 settled / records source domains</th>"
        "<th>M3 settled / live, current sources</th>"
        "<th>Move under M2</th><th>Move under M3</th></tr>"
    )
    moves = {entry["group"]: entry for entry in sens["moves"]}
    body = []
    for entry in table:
        move = moves.get(entry["group"])
        rank = entry["rank"] if entry["rank"] is not None else "—"
        move_m2 = f"{move['move_m2']:+d}" if move else "—"
        move_m3 = f"{move['move_m3']:+d}" if move else "—"
        body.append(
            "<tr>"
            f"<td>{rank}</td><td>{_esc(entry['group'])}</td>"
            f"<td>{entry['live']}</td><td>{entry['settled']}</td>"
            f"<td>{_pct(entry['m1'])}</td><td>{_pct(entry['m2'])}</td>"
            f"<td>{_pct(entry['m3'])}</td><td>{move_m2}</td><td>{move_m3}</td>"
            "</tr>"
        )
    parts.append(
        f'<table id="{node}-ranking-table"><thead>{rank_head}</thead><tbody>'
        + "\n".join(body)
        + "</tbody></table>\n"
    )
    parts.append(
        f'<h2 id="{node}-sensitivity-head">Sensitivity of the order to the measure</h2>\n'
    )
    parts.append(
        f'<p id="{node}-sensitivity">Across the ranked groups the M1 order against '
        f"M2 correlates at Spearman {_lead_symbol(sens['rho_m1_m2'])}, and against "
        f"M3 at {_lead_symbol(sens['rho_m1_m3'])}; M2 against M3 at "
        f"{_lead_symbol(sens['rho_m2_m3'])}. The largest single move is "
        f"{sens['largest_move_m2']} places under M2 and {sens['largest_move_m3']} "
        "under M3. The order M1 gives is what the proposal uses; the move columns "
        "beside each group's numbers show where a ruling that prefers M2 or M3 would "
        "put it instead.</p>\n"
    )
    full_settled = [
        entry
        for entry in table
        if entry["m1"] == 1.0 and audit["per_group"].get(entry["group"], {}).get("any")
    ]
    mismatch_text = ""
    mismatches = payload.get("mismatches") or []
    if mismatches:
        shapes = "; ".join(
            f"<code>{_esc(shape['stored'])}</code> over "
            f"<code>{_esc(shape['source'])}</code> sources, {shape['names']} names"
            for shape in mismatches[:3]
        )
        mismatch_text = (
            "The dominant shapes are " + shapes + ". A name whose sources are "
            f"<code>{_esc(mismatches[0]['source'])}</code> paths but whose stored "
            f"domain is <code>{_esc(mismatches[0]['stored'])}</code> is a flux or a "
            "coefficient named for what it does while its source sits elsewhere — "
            "the disagreement the plan's opening paragraph already describes, "
            "counted here rather than judged. "
        )
    caveat = ""
    if full_settled:
        example = full_settled[0]
        example_wrong = audit["per_group"][example["group"]]["any"]
        caveat = (
            f'<p id="{node}-caveat">What M1 does and does not measure: the schema '
            "says a name's stored primary domain is promoted from the domains of the "
            "sources attached to it, and a catalog following that rule would settle "
            "every name that records a source. The live catalog does not: "
            f"{partition['not_among']} names carry a primary that is none of their "
            "recorded source domains, and the promotion rule would not produce the "
            "largest of those shapes, so the stored primary and the recorded source set "
            "are written by paths that do not reconcile with each other. "
            "A settled name is a name whose stored domain agrees with its recorded "
            "provenance, not a name filed under the right domain, and a group whose "
            "members all draw on one domain settles by construction: "
            f"<code>{_esc(example['group'])}</code> settles {example['settled']} "
            f"of {example['live']} names and section 1 still read {example_wrong} of "
            "the names it sampled there as wrong-domain. "
            + mismatch_text
            + "Settlement is the consistency measure and section 1's reading is the "
            "independent one; the order puts the internally consistent groups first, "
            "and the wrong-domain counts are the caution on all of them.</p>\n"
        )
    parts.append(caveat)

    # -- wrong domain
    interval = audit["intervals"]
    best_case = (
        max(interval.values(), key=lambda entry: entry["sampled"]) if interval else None
    )
    upper_best = best_case["upper_pp_best_case"] if best_case else 0.0
    width_best = best_case["width_pp_best_case"] if best_case else 0.0
    sampled_span = sorted({entry["sampled"] for entry in interval.values()})
    if not interval:
        sampled_text = "no"
    elif len(sampled_span) == 1:
        sampled_text = str(sampled_span[0])
    else:
        sampled_text = " or ".join(str(n) for n in sampled_span)
    parts.append(
        f'<h2 id="{node}-wrong-domain-head">What section 1 read as wrong-domain, '
        "by group</h2>\n"
    )
    parts.append(
        f'<p id="{node}-wrong-domain">The counts are joined from the section-1 '
        f"audit records (<code>{_esc(audit['source'])}</code>, "
        f"{audit['names_read']} names read), matched to a group by the name's stored "
        f"domain. <strong>No group rate is stated, and none can be.</strong> A group "
        f"contributes {sampled_text} names to that sample, and a group with no "
        "wrong-domain name at all would still carry a 95 % interval of 0–"
        f"{upper_best:.1f} %, {width_best:.1f} pp wide: section 1's own rule — a rate "
        "only where the interval is narrower than twenty percentage points — admits "
        "no group. A group's rate is measured when the group is read in full, as "
        "sections 4 and 5 do.</p>\n"
    )
    parts.append(
        f'<p id="{node}-wrong-domain-join">'
        f"The join matches every name it should: {audit['sample_total_wrong_domain']} "
        f"of the {audit['class_total_from_audit']} names section 1 reads as "
        "wrong-domain by at least one reader carry a group row, so the zeros in the "
        "per-group column are read after the join is shown to see names known to be "
        "present. "
        + (
            f"{len(audit['unmapped'])} of them are no longer live in this census; "
            if audit["unmapped"]
            else "No name it read as wrong-domain has dropped out of the live catalog; "
        )
        + f"{len(audit['planted_in_sample'])} of the audit's {audit['planted_rows']} "
        "planted rows appear among the sampled records, so the planted names enter "
        "no count above.</p>\n"
    )

    # -- families
    counts = collections.Counter(family["clearing"]["choice"] for family in families)
    parts.append(
        f'<h2 id="{node}-families-head">Families whose members sit in more than one '
        "group</h2>\n"
    )
    bring = counts.get("bring_together", 0)
    cut = counts.get("cut_together", 0)
    exception = counts.get("exception", 0)
    parts.append(
        f'<p id="{node}-families">{payload["families"]["detected"]} DD families are '
        "detected by <code>detect_families</code> over the "
        f"{payload['families']['source_paths']} DD paths that produced the live "
        f"names; <strong>{len(families)} of them have live names in more than one "
        "group</strong>, and a family whose live names all sit in one group spans "
        "nothing and is not listed. Each is listed with its members' groups, the "
        "clearing proposed here, and the state the lead ruled. "
        "<strong>Ruled 2026-11-17: every one of these families is recorded as an "
        f"{FAMILY_RULING}</strong> — no clearing moves a stored domain until section "
        "6 has measured reading a name's group from its sources' data structures "
        "over the whole catalog — so the proposal below is kept beside the ruling "
        "and is not in the release bar for the first cut. The proposals, by "
        f"construction: {bring} bring the members into one group, {cut} cut with the "
        f"groups they span, and {exception} are exceptions as well as ruled ones.</p>\n"
    )
    family_head = (
        "<tr><th>Family (DD parent, type, members)</th><th>Members' groups</th>"
        "<th>Named members</th><th>Ruled state</th><th>Proposed clearing (not in "
        "the bar)</th></tr>"
    )
    body = []
    for family in families:
        choice = family["clearing"]
        if choice["choice"] == "bring_together":
            verdict = f"bring into {_esc(choice['into'])} ({choice['moves']} move)"
        elif choice["choice"] == "cut_together":
            verdict = "cut together: " + " + ".join(_esc(g) for g in choice["groups"])
        else:
            verdict = "exception (may span groups)"
        suffix_text = _esc(", ".join(family["suffixes"]))
        groups_cell = ", ".join(
            f"{_esc(group)} {count}" for group, count in family["group_counts"].items()
        )
        body.append(
            "<tr>"
            f"<td><code>{_esc(family['parent_path'])}</code><br>"
            f"{_esc(family['family_type'])}: {suffix_text}</td>"
            f"<td>{groups_cell}</td><td>{family['named_members']}</td>"
            f"<td>{_esc(family['ruled'])}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    parts.append(
        f'<table id="{node}-families-table"><thead>{family_head}</thead><tbody>'
        + "\n".join(body)
        + "</tbody></table>\n"
    )
    touching = [family for family in families if FIRST_GROUP in family["group_counts"]]
    parts.append(
        f'<h2 id="{node}-equilibrium-families-head">The families that touch '
        f"{_esc(FIRST_GROUP)}</h2>\n"
    )
    parts.append(
        f'<p id="{node}-equilibrium-families">Section 2\'s ruling lists these by name '
        f"because <code>{_esc(FIRST_GROUP)}</code> is released first: "
        f"<strong>{len(touching)} of the {len(families)} spanning families have a "
        f"member in <code>{_esc(FIRST_GROUP)}</code></strong>, so each is an "
        "exception the first cut carries and each is a place where the first cut's "
        "own group is entangled with a group released later. The right-hand column "
        "names the groups the rest of the family sits in.</p>\n"
    )
    touching_head = (
        "<tr><th>Family (DD parent, type, members in equilibrium)</th>"
        f"<th>Members in {_esc(FIRST_GROUP)}</th><th>Its other groups</th></tr>"
    )
    body = []
    for family in touching:
        suffix_text = _esc(", ".join(family["suffixes"]))
        members = ", ".join(
            f"<code>{_esc(name)}</code>"
            for name in family["member_groups"].get(FIRST_GROUP) or []
        )
        others = ", ".join(
            f"{_esc(group)} {count}"
            for group, count in family["group_counts"].items()
            if group != FIRST_GROUP
        )
        body.append(
            "<tr>"
            f"<td><code>{_esc(family['parent_path'])}</code><br>"
            f"{_esc(family['family_type'])}: {suffix_text}</td>"
            f"<td>{members}</td><td>{others}</td>"
            "</tr>"
        )
    parts.append(
        f'<table id="{node}-equilibrium-families-table"><thead>{touching_head}'
        "</thead><tbody>" + "\n".join(body) + "</tbody></table>\n"
    )

    # -- transport
    parts.append(f'<h2 id="{node}-transport-head">transport: stand, or split</h2>\n')
    lead_moves = ", ".join(
        f"{entry['count']} to {_esc(group)}"
        for group, entry in list(transport["moves"].items())[:3]
    )
    fast = transport["moves"].get("fast_particles", {}).get("count", 0)
    core = transport["moves"].get("core_plasma_physics", {}).get("count", 0)
    parts.append(
        f'<p id="{node}-transport"><code>transport</code> holds '
        f"{transport['live']} live names, "
        f"{_pct(transport['live'] / partition['live'] if partition['live'] else None)} "
        f"of the catalog, and {transport['settled']} of them are settled "
        f"({_pct(transport['settled'] / transport['live'] if transport['live'] else None)}). "
        "The split measured is the mechanical one: hand each name to the "
        "highest-ranked of its other source domains, the rule that already selects a "
        f"name's stored domain. It would move {transport['split_total']} names — "
        f"{lead_moves} — and leave {transport['stays']} in the group. That is also "
        "the split's refutation: it hands "
        f"{fast} names to a group of 17 and {core} to one of 45, so it re-keys the "
        "catalog to where each number was measured rather than to what the quantity "
        "is, and it leaves the receiving groups no better defined than "
        "<code>transport</code>. The alternative is that the group stands as one "
        "release unit, last in the order, its membership read as an operator class "
        "that spans every core IDS. The lead rules between the two.</p>\n"
    )
    structure = transport.get("structure") or []
    if structure:
        top_structure = ", ".join(
            f"<code>{_esc(entry['ids'])}</code> {entry['names']}"
            for entry in structure[:6]
        )
        parts.append(
            f'<p id="{node}-transport-structure">What the standing option contains: '
            f"the {transport['live']} members draw their sources through "
            f"{len(structure)} top-level IDSs, led by {top_structure} names. That is "
            "a domain of what the quantity <em>does</em> — a flux, a source, a "
            "coefficient — not of where it is measured, which is why their sources span "
            f"{len(structure)} IDSs and why the mechanical split sends them there "
            "too. A split by IDS structure would therefore define a "
            "new domain vocabulary rather than divide the existing groups, and is "
            "outside what a split among them can do.</p>\n"
        )
    parts.append(
        f'<p id="{node}-transport-choice"><strong>Ruled 2026-11-17: '
        "<code>transport</code> does not stand as one group, and it is not split by "
        "the domains of its sources.</strong> Section 2 refuses that split for the "
        "reason measured here — it would send "
        f"{fast} names to a 17-name group and {core} to one of 45 — and divides the "
        "group along the data dictionary's own structures instead, which returns "
        "group names the domain vocabulary does not hold today and is carried by the "
        "plan's section 6. Its parts are released after the other groups. No stored "
        "domain moves until section 6's parts are ruled. The table below stays as "
        "the measurement behind the refusal, not as a proposal.</p>\n"
    )

    transport_head = "<tr><th>Destination</th><th>Names</th><th>Rule</th></tr>"
    body = [
        f"<tr><td>stays in <code>transport</code></td><td>{transport['stays']}</td>"
        "<td>only <code>transport</code> among its sources, or no recorded source "
        "domain</td></tr>"
    ]
    for group, entry in transport["moves"].items():
        body.append(
            f"<tr><td>moves to {_esc(group)}</td><td>{entry['count']}</td>"
            "<td>highest-ranked of its other source domains</td></tr>"
        )
    body.append(
        f"<tr><td>every member</td><td>{transport['live']}</td>"
        "<td>held by the split</td></tr>"
    )
    parts.append(
        f'<table id="{node}-transport-table"><thead>{transport_head}</thead><tbody>'
        + "\n".join(body)
        + "</tbody></table>\n"
    )

    # -- order
    order = payload["order"]
    ruled = payload["ruling"]["sequence"]
    parts.append(f'<h2 id="{node}-order-head">The ruled release order</h2>\n')
    parts.append(
        f'<p id="{node}-order"><strong>Ruled 2026-11-17: <code>{_esc(ruled["first"])}'
        "</code> is released first and the remaining groups follow the measured "
        "order.</strong> The measured order is M1 within the ranked groups, with a "
        f"group below the {SAMPLE_FLOOR}-name floor after them because the bar of "
        "section 3 is fixed from audited rates and such a group must be read in full "
        "before any bar can apply. The ruling moves "
        f"<code>{_esc(ruled['first'])}</code> from measured place "
        f"{ruled['measured_place'][ruled['first']]} to the first release, leaves every "
        "other group where the measurement puts it, and takes <code>transport</code> "
        "out of the group list: section 2 divides it along the data dictionary's own "
        f"structures under section 6, so its parts are released last as "
        f"{_esc(ruled['last'])}. Neither the first group nor the last line is a "
        "measured position; each is the ruling.</p>\n"
    )
    order_head = (
        "<tr><th>Release position</th><th>Group</th><th>Measured place</th>"
        "<th>Why here</th></tr>"
    )
    body = []
    for position, group in enumerate(ruled["sequence"], start=1):
        if group == ruled["first"]:
            why = "ruled first on 2026-11-17"
            measured = ruled["measured_place"][group]
        elif group == ruled["last"]:
            why = (
                "not a group after section 2's ruling: transport is divided along "
                "data dictionary structures under section 6 and its parts are "
                "released after the groups"
            )
            measured = "—"
        else:
            why = order["reasons"].get(group, "")
            measured = ruled["measured_place"].get(group, "—")
        body.append(
            f"<tr><td>{position}</td><td>{_esc(group)}</td><td>{measured}</td>"
            f"<td>{why}</td></tr>"
        )
    parts.append(
        f'<table id="{node}-order-table"><thead>{order_head}</thead><tbody>'
        + "\n".join(body)
        + "</tbody></table>\n"
    )

    # -- instrument checks
    plant = checks["census_plant"]
    parts.append(f'<h2 id="{node}-instrument-head">The instrument checks</h2>\n')
    parts.append(
        f'<p id="{node}-plant">Two names were planted in the census in memory, both '
        f"in <code>{_esc(plant['target_group'])}</code>: "
        "<code>census_probe_settled_member</code> with a domain among its sources' "
        "domains, and <code>census_probe_unsettled_member</code> with a domain that "
        f"is not. The group's live count rose by {plant['delta_live']}, its settled "
        f"count by {plant['delta_settled']} and its not-among count by "
        f"{plant['delta_not_among']}. With the two plants' source sets emptied the "
        f"settled count fell to {plant['emptied_sources_settled']} and both names "
        f"fell into the no-source column ({plant['emptied_sources_no_source']}), so "
        "the comparison reads the source set rather than the presence of a row, and "
        "a join that matched nothing could not read as agreement. "
        f"Passed: {plant['passed']}. The census rows sum to {checks['census_sum']} = "
        f"the {partition['live']} live names read in the same run.</p>\n"
    )
    parts.append(
        f'<h2 id="{node}-reproduce-head">Reproduce</h2>\n'
        f'<pre id="{node}-reproduce">UV_NO_SYNC=1 '
        "UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv "
        "PYTHONPATH=$PWD uv run --no-sync python "
        "docs/evidence/fragments/sn-audited-group-release/domain-groups.py</pre>\n"
        f'<p id="{node}-checks-head">The run writes its files only after every '
        "instrument arm passes; a failed arm leaves the committed evidence "
        "untouched and exits non-zero.</p>\n"
    )
    parts.append(
        f'<p id="{node}-environment">Read at {_esc(environment["generated_at"])} on '
        f"{_esc(environment['host'])} ({_esc(environment['python'])}) on measured "
        f"base revision <code>{_esc(environment['measured_base'])}</code>.</p>\n"
    )
    parts.append("</main></body></html>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _measured_base() -> str:
    """The commit this measurement stands on, not the commit that carries it.

    The page names the base its run was taken from rather than the commit that
    contains it, so a run never needs a follow-up commit to restamp itself and
    the evidence does not describe the commit it is stored in. The base is the
    merge base of the worktree with the branch it was cut from.
    """
    for reference in ("origin/main", "main"):
        try:
            base = subprocess.run(
                ["git", "merge-base", "HEAD", reference],
                cwd=FRAGMENTS,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if base:
            return base
    return "unknown"


def mismatch_shapes(
    states: tuple[NameDomainState, ...], limit: int = 6
) -> list[dict[str, Any]]:
    """The most common stored-domain-over-source-domain shapes among unsettled names.

    A name whose stored domain is not among its sources' contributes one shape
    per source domain it records, so the table names the classifier disagreement
    behind the unsettled count rather than leaving it as a number.
    """
    shapes: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for state in states:
        if not state.physics_domain or not state.source_domains:
            continue
        if state.physics_domain in state.source_domains:
            continue
        for source in state.source_domains:
            shapes[(state.physics_domain, source)].append(state.sn_id)
    ordered = sorted(shapes.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [
        {
            "stored": stored,
            "source": source,
            "names": len(names),
            "examples": sorted(names)[:3],
        }
        for (stored, source), names in ordered[:limit]
    ]


def build(gc: Any) -> dict[str, Any]:
    """Read the graph twice and compute every section of the evidence."""
    result: DomainAuditResult = audit_domains(gc)
    states = result.states
    source_rows = [
        dict(row)
        for row in gc.query(_SOURCE_PATHS_QUERY, live=sorted(LIVE_NAME_STAGES)) or []
    ]
    rows = census(states)
    table = ranked_rows(rows)
    families = family_evidence(source_rows)
    transport = transport_evidence(states)
    transport["structure"] = transport_structure(source_rows)
    mismatches = mismatch_shapes(states)
    order = proposed_order(table, transport["live"])
    order_position = {
        group: position for position, group in enumerate(order["proposed"], start=1)
    }
    for family in families["spanning"]:
        family["clearing"] = clearing_proposal(family, order_position)
        family["ruled"] = FAMILY_RULING
    touching = [
        family
        for family in families["spanning"]
        if FIRST_GROUP in family["group_counts"]
    ]
    families["touching_first_group"] = {
        "group": FIRST_GROUP,
        "count": len(touching),
        "ids": [family["id"] for family in touching],
    }
    audit = wrong_domain_counts(AUDIT_JSON, states)
    census_rows = sorted(rows.values(), key=lambda row: (-row["live"], row["group"]))
    partition = {
        "live": sum(row["live"] for row in census_rows),
        "records_domains": sum(row["records_domains"] for row in census_rows),
        "settled": sum(row["settled"] for row in census_rows),
        "not_among": sum(row["not_among"] for row in census_rows),
        "no_sources": sum(row["no_sources"] for row in census_rows),
        "settled_expected": sum(row["settled_expected"] for row in census_rows),
    }
    partition["settled_share"] = (
        partition["settled"] / partition["live"] if partition["live"] else 0.0
    )
    partition["settled_share_recorded"] = (
        partition["settled"] / partition["records_domains"]
        if partition["records_domains"]
        else 0.0
    )
    plant = census_plant_check(states, table)
    return {
        "census": {"rows": census_rows, "partition": partition},
        "ranking": {"groups": table, "sensitivity": sensitivity(table)},
        "wrong_domain": audit,
        "families": families,
        "transport": transport,
        "order": order,
        "ruling": {**RULING, "sequence": ruled_order(order)},
        "mismatches": mismatches,
        "instrument": {
            "census_plant": plant,
            "family_members": families["member_check"],
            "census_sum": partition["live"],
            "audit_live_names": result.live_names,
            "sum_matches_read": partition["live"] == result.live_names,
            "planted_in_sample": audit["planted_in_sample"],
        },
        "environment": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": platform.node(),
            "python": platform.python_version(),
            "measured_base": _measured_base(),
            "live_stages": sorted(LIVE_NAME_STAGES),
        },
    }


def instrument_checks(payload: dict[str, Any]) -> list[tuple[str, bool]]:
    """Every way the run could be wrong, one arm each so every one can fail."""
    instrument = payload["instrument"]
    plant = instrument["census_plant"]
    members = instrument["family_members"]
    return [
        (
            "census rows sum to the live names read",
            instrument["sum_matches_read"],
        ),
        ("planted settled member counted once", plant["delta_settled"] == 1),
        ("planted unsettled member counted once", plant["delta_not_among"] == 1),
        (
            "emptied source sets never read as settled",
            plant["emptied_sources_settled"] == 0,
        ),
        (
            "no planted name in the audit sample",
            not instrument["planted_in_sample"],
        ),
        (
            "every family lists each of its members once",
            members["passed"] and not members["repeated"],
        ),
        (
            "every family's listed members are its paths' distinct names",
            not members["miscounted"],
        ),
    ]


def main() -> int:
    """Measure, check, and only then write: a failed arm leaves the files alone."""
    from iter_codex.graph.client import GraphClient

    with GraphClient() as gc:
        payload = build(gc)

    checks = instrument_checks(payload)
    for description, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {description}")
    print(
        f"live_names={payload['census']['partition']['live']} "
        f"groups={len(payload['ranking']['groups'])} "
        f"settled={payload['census']['partition']['settled']} "
        f"spanning_families={len(payload['families']['spanning'])} "
        f"transport_live={payload['transport']['live']}"
    )
    if not all(ok for _, ok in checks):
        print(
            f"not written: {OUT_JSON.name} and {OUT_HTML.name} keep the last run that passed"
        )
        return 1
    OUT_JSON.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    OUT_HTML.write_text(render_page(payload) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name} and {OUT_HTML.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
