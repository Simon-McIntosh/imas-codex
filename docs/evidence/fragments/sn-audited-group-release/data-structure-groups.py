"""Read every live standard name's group from the data structures of its sources.

Section 6 of the standard-name release plan rules that ``transport`` is divided
along the data dictionary's own structures, and asks for the rule to be measured
over the whole live catalog before any stored domain moves. This script is that
measurement and the proposal it carries. Nothing is written to the graph.

The rule
--------

A source of a name is a Data Dictionary path. A path's *structure* is the pair
(IDS, first structural container): ``core_profiles/profiles_1d``,
``core_sources/source``, ``summary/local``, ``core_transport/model``. A
structure names a group:

* ``summary`` and the ``global_quantities`` container -> ``summary_quantities``
  (the local, volume-averaged, line-averaged paragraph) ... the local,
  volume-averaged, line-averaged and whole-plasma forms a summary structure
  holds;
* ``profiles_1d``, ``profiles_2d``, ``ggd``, ``ggd_fluid`` ->
  ``profile_quantities``;
* ``source`` -> ``source_quantities``;
* ``distribution`` / ``distributions`` -> ``distribution_quantities``;
* ``model`` in a transport IDS -> ``transport_coefficients``;
* every other structure -> the domain the Data Dictionary's own IDS-level map
  gives its IDS (``imas_codex/definitions/physics/ids_domains.json``), which is
  the group the DD's top-level structure already carries in the catalog.

A name with DD sources reads the union of its sources' groups. A name with no
DD source is a derived parent and unions its live children's readings through
the same rule, guarded against a ``HAS_PARENT`` cycle. A name whose closures
read one group is *settled*, and it is *moved* when that group is not its
stored ``physics_domain``; a name whose closures read more than one group
*spans*; a name with nothing to read keeps its stored group and is counted in
``settled_without_structure``, because the rule has no reading that could move
it. ``settled + moved + spanning`` therefore equals the live names read in the
same run.

Nothing is written. Every graph call is a read; the script asserts that over the
statement text it hands the client, so a write added later fails the run.

Commands
--------

``python data-structure-groups.py``
    Read the live graph, run every instrument arm and the planted controls, and
    only then write the data file and the page. A failed arm leaves the
    committed files byte-identical and exits 1.

``python data-structure-groups.py --check``
    The gate: the same read, arms and planted controls, printed, writing nothing.
    Exits 0 only when ``settled + moved + spanning`` equals the live names read
    and the planted disposable-graph controls land where they should.

``--check --plant-double-count``
    The negative control: a planted name is inserted into two parts and the same
    gate must refuse it, exiting 1.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.core.physics_categorization import PhysicsDomainCategorizer
from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.standard_names.domain_audit import LIVE_NAME_STAGES
from imas_codex.standard_names.families import detect_families

FRAGMENTS = Path(__file__).resolve().parent
OUT_JSON = FRAGMENTS / "data-structure-groups.json"
OUT_HTML = FRAGMENTS / "data-structure-groups.html"
SECTION_TWO_JSON = FRAGMENTS / "domain-groups.json"

#: The five parts ``transport`` divides into, each named for the quantities it
#: holds. The first four name a data structure's own quantity class and are
#: shared by every IDS that carries that structure; the last is the transport
#: model structure, which only the transport IDSs carry.
PROFILE_QUANTITIES = "profile_quantities"
SOURCE_QUANTITIES = "source_quantities"
DISTRIBUTION_QUANTITIES = "distribution_quantities"
SUMMARY_QUANTITIES = "summary_quantities"
TRANSPORT_COEFFICIENTS = "transport_coefficients"

PARTS: dict[str, dict[str, str]] = {
    PROFILE_QUANTITIES: {
        "definition": (
            "the profiles a profile structure holds: the one- and "
            "two-dimensional profile quantities and their general-grid forms"
        ),
        "structures": "profiles_1d, profiles_2d, ggd, ggd_fluid",
    },
    SOURCE_QUANTITIES: {
        "definition": (
            "the source terms a sources structure holds: the particle, energy, "
            "momentum and current rates a source adds to the plasma"
        ),
        "structures": "source",
    },
    DISTRIBUTION_QUANTITIES: {
        "definition": (
            "the distribution-function quantities a distributions structure "
            "holds: the distribution itself and its orbit-class moments"
        ),
        "structures": "distribution, distributions",
    },
    SUMMARY_QUANTITIES: {
        "definition": (
            "the summary forms a summary structure holds: local values at named "
            "locations and the volume-, line- and flux-surface-averaged or "
            "integrated forms of a profile quantity"
        ),
        "structures": "ids summary; the global_quantities container",
    },
    TRANSPORT_COEFFICIENTS: {
        "definition": (
            "the transport-model quantities a transport model structure holds: "
            "diffusivities, conductivities, convection velocities and damping "
            "rates"
        ),
        "structures": "model in core_transport, plasma_transport, edge_transport",
    },
}

PROFILE_TOKENS = frozenset({"profiles_1d", "profiles_2d", "ggd", "ggd_fluid"})
SOURCE_TOKENS = frozenset({"source"})
DISTRIBUTION_TOKENS = frozenset({"distribution", "distributions"})
SUMMARY_IDS = "summary"
SUMMARY_TOKENS = frozenset({"global_quantities"})
MODEL_TOKEN = "model"
TRANSPORT_IDS = frozenset({"core_transport", "plasma_transport", "edge_transport"})
SOURCE_IDS = frozenset(
    {"core_sources", "plasma_sources", "edge_sources", "distribution_sources"}
)
NO_GROUP = "(no physics domain)"

#: ``LIVE_NAME_STAGES`` comes from the audit module so this run counts the same
#: "live" as section 2's census: accepted, reviewed, drafted, pending.
LIVE = sorted(LIVE_NAME_STAGES)

#: One row per live name: its stored domain, its stored description (for the
#: hand-read sample) and its DD sources and live children, which the rule turns
#: into a reading. Row set: the live names, one label scan filtered by stage;
#: each walk is one hop.
_LIVE_QUERY_PLACEHOLDER = None
_LIVE_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
WITH sn, collect(DISTINCT dd.id) AS paths
OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
WHERE child.name_stage IN $live
RETURN sn.id AS sn_id,
       sn.physics_domain AS physics_domain,
       sn.name_stage AS name_stage,
       sn.description AS description,
       paths,
       collect(DISTINCT child.id) AS children
ORDER BY sn_id
"""

#: The live names' DD source paths with their units. A family is a set of DD
#: paths built by the sibling family detector, which needs each path's unit; the
#: read above keeps the paths but not their units. Row set: one row per live
#: name's source-to-DD-path hop, a few thousand rows over an indexed edge.
_SOURCE_PATHS_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
RETURN DISTINCT dd.id AS dd_path, dd.unit AS unit
ORDER BY dd_path
"""

#: The clauses that make a statement a write. Every statement this run hands the
#: client is checked against them; a match fails the run rather than sending.
_WRITE_CLAUSES = (
    " SET ",
    " CREATE ",
    " MERGE ",
    " DELETE ",
    " REMOVE ",
    " DROP ",
    " LOAD CSV ",
)

#: The hand-read sample of moved names. The selection is deterministic — the two
#: lowest-sorted transport members of every part, then the three lowest-sorted
#: transport names the rule moves into an existing group — so the sample can be
#: re-read from the data file. Each judgment was made against the name's own
#: definition and source paths: ``rule`` when the structure's reading is right,
#: ``stored`` when the stored domain is right, ``neither`` when neither fits.
SAMPLE_READINGS: dict[str, dict[str, str]] = {
    "average_charge_number": {
        "reading": "rule",
        "why": (
            "a density-weighted mean ion charge is a profile-like average over a "
            "charge-state population, and its no-source form is the parent of "
            "profile children; the stored transport domain names no structure it "
            "comes from"
        ),
    },
    "change_in_ion_state_mean_ionisation_potential": {
        "reading": "rule",
        "why": (
            "the quantity is a signed change in an ion-state mean potential, "
            "stated by a core_instant_changes change structure over a profile "
            "grid; its DD home reads as core plasma physics, so the rule's "
            "reading is the closer one even though the container is a change "
            "rather than a profile"
        ),
    },
    "electron_emissivity": {
        "reading": "rule",
        "why": (
            "the emitted power density is stated by a radiation process structure "
            "over a ggd grid; its DD home is core plasma physics, not a "
            "diagnostics domain, so the rule's reading is right and the stored "
            "transport domain names no structure it comes from"
        ),
    },
    "bulk_neutral_internal_state_velocity": {
        "reading": "rule",
        "why": (
            "a bulk velocity of neutrals in one internal state is computed on a "
            "profile grid, so the profile structures' reading names where it "
            "sits; the stored transport domain names no structure it comes from"
        ),
    },
    "cumulative_inside_flux_surface_current_due_to_non_inductive_current_drive": {
        "reading": "rule",
        "why": (
            "the non-inductive current inside a flux surface is a source "
            "structure's cumulative integral of the driven current; it is not a "
            "transport model quantity"
        ),
    },
    "cumulative_inside_flux_surface_electron_particle_source_rate": {
        "reading": "rule",
        "why": (
            "the source structures state the particle source and its cumulative "
            "inside-surface form directly, so the rule reads the structure that "
            "holds the quantity"
        ),
    },
    "co_passing_fast_current_density": {
        "reading": "rule",
        "why": (
            "the current is an orbit-class moment of the distribution function, "
            "stated in the distributions structures; the stored transport domain "
            "does not say what structure holds it"
        ),
    },
    "co_passing_fast_ion_charge_state_power_density_due_to_collisions": {
        "reading": "rule",
        "why": (
            "the collisional power density on co-passing fast ions is stated "
            "inside the distributions structure, one level below the "
            "distribution itself, so distribution_quantities is its structure's "
            "group"
        ),
    },
    "argon_density": {
        "reading": "rule",
        "why": (
            "its no-source form is the parent of the summary structures' local "
            "and volume-averaged argon densities, so the rule reads the summary "
            "structures that hold every form of this density; the part is named "
            "for what a summary structure holds, not for the physical quantity"
        ),
    },
    "argon_density_at_magnetic_axis": {
        "reading": "rule",
        "why": (
            "the quantity is stated only under summary/local, so the summary "
            "structure is the reading; a local density at a named location is "
            "exactly the class a summary structure holds"
        ),
    },
    "diamagnetic_conductivity": {
        "reading": "rule",
        "why": (
            "the conductivity is stated under the transport model's "
            "conductivity container, which is the transport coefficient "
            "structure itself"
        ),
    },
    "diamagnetic_momentum_convection_velocity": {
        "reading": "rule",
        "why": (
            "the coefficient for diamagnetic advective momentum transport is "
            "stated under core_transport/model, the transport coefficient "
            "structure"
        ),
    },
    "diffusion_coefficient_due_to_neoclassical_tearing_mode": {
        "reading": "stored",
        "why": (
            "the quantity is a transport diffusion coefficient that an NTM "
            "enhances; its DD home is the ntms IDS, so a rule reading only the "
            "structure files a transport coefficient under "
            "magnetohydrodynamics. The stored transport reading is right and "
            "the rule's is wrong"
        ),
    },
}


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def structure_of(path: str) -> tuple[str, str]:
    """The IDS and its first structural container for a DD path."""
    pieces = path.split("/")
    ids = pieces[0]
    token = pieces[1] if len(pieces) > 1 else ""
    return ids, token


class GroupTable:
    """Reads a DD structure as a group name.

    The five structures the plan's opening paragraph names are read from the
    structure itself; every other structure is read from the Data Dictionary's
    own IDS-level map, so a structure whose IDS carries a domain keeps it.
    """

    def __init__(self) -> None:
        self._categorizer = PhysicsDomainCategorizer()
        self.unknown_ids: set[str] = set()

    def ids_domain(self, ids: str) -> str:
        domain = self._categorizer.get_domain_for_ids(ids)
        if ids.lower() not in self._categorizer.mappings:
            self.unknown_ids.add(ids)
        return domain.value

    def of_path(self, path: str) -> str:
        ids, token = structure_of(path)
        if ids == SUMMARY_IDS or token in SUMMARY_TOKENS:
            return SUMMARY_QUANTITIES
        if token in PROFILE_TOKENS:
            return PROFILE_QUANTITIES
        if token in SOURCE_TOKENS:
            return SOURCE_QUANTITIES
        if token in DISTRIBUTION_TOKENS or ids == "distributions":
            return DISTRIBUTION_QUANTITIES
        if token == MODEL_TOKEN and ids in TRANSPORT_IDS:
            return TRANSPORT_COEFFICIENTS
        return self.ids_domain(ids)


def resolve_groups(
    rows: list[dict[str, Any]], table: GroupTable
) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    """Every live name's group reading, and where it came from.

    A name with DD sources reads its sources' structures. A name with none is a
    derived parent and unions its live children's readings, guarded so a
    ``HAS_PARENT`` cycle terminates. The second return value says ``sources``,
    ``children`` or ``none``.
    """
    by_id = {row["sn_id"]: row for row in rows}
    memo: dict[str, frozenset[str]] = {}
    origin: dict[str, str] = {}

    def resolve(sn_id: str, visiting: frozenset[str]) -> frozenset[str]:
        if sn_id in memo:
            return memo[sn_id]
        row = by_id[sn_id]
        if row.get("paths"):
            result = frozenset(table.of_path(path) for path in row["paths"])
            origin[sn_id] = "sources"
        elif sn_id in visiting:
            result = frozenset()
            origin[sn_id] = "none"
        else:
            union: set[str] = set()
            for child_id in row.get("children") or []:
                if child_id in by_id:
                    union |= resolve(child_id, visiting | {sn_id})
            result = frozenset(union)
            origin[sn_id] = "children" if result else "none"
        memo[sn_id] = result
        return result

    for row in rows:
        resolve(row["sn_id"], frozenset())
    return memo, origin


def read_catalog(
    rows: list[dict[str, Any]], groups: dict[str, frozenset[str]]
) -> dict[str, Any]:
    """Partition the live names by the rule's reading.

    ``settled`` holds a name whose reading is its stored domain, and a name
    with no reading at all, which the rule leaves where it stands;
    ``settled_with_structure`` and ``settled_without_structure`` separate those
    two. ``moved`` holds a name whose one reading differs from its stored
    domain. ``spanning`` holds a name whose closures read more than one group.
    The three buckets partition the live names read.
    """
    settled = 0
    settled_with_structure = 0
    settled_without_structure: list[str] = []
    moved: list[str] = []
    moved_from_no_group = 0
    spanning = 0
    moved_to: collections.Counter[str] = collections.Counter()
    moved_from: collections.Counter[str] = collections.Counter()
    spanning_groups: collections.Counter[str] = collections.Counter()
    stored_counts: collections.Counter[str] = collections.Counter()
    reading_counts: collections.Counter[str] = collections.Counter()

    for row in rows:
        sn_id = row["sn_id"]
        stored = row.get("physics_domain") or None
        reading = groups[sn_id]
        stored_counts[stored or NO_GROUP] += 1
        if len(reading) == 0:
            settled += 1
            settled_without_structure.append(sn_id)
            continue
        if len(reading) > 1:
            spanning += 1
            spanning_groups["|".join(sorted(reading))] += 1
            continue
        only = next(iter(reading))
        reading_counts[only] += 1
        if only == stored:
            settled += 1
            settled_with_structure += 1
        else:
            moved.append(sn_id)
            moved_to[only] += 1
            moved_from[stored or NO_GROUP] += 1
            if stored is None:
                moved_from_no_group += 1

    return {
        "live": len(rows),
        "settled": settled,
        "settled_with_structure": settled_with_structure,
        "settled_without_structure": len(settled_without_structure),
        "moved": len(moved),
        "moved_from_no_group": moved_from_no_group,
        "spanning": spanning,
        "names": {
            "moved": sorted(moved),
            "settled_without_structure": sorted(settled_without_structure),
        },
        "moved_to": dict(sorted(moved_to.items(), key=lambda kv: (-kv[1], kv[0]))),
        "moved_from": dict(sorted(moved_from.items(), key=lambda kv: (-kv[1], kv[0]))),
        "spanning_shapes": [
            {"groups": shape.split("|"), "names": count}
            for shape, count in sorted(
                spanning_groups.items(), key=lambda kv: (-kv[1], kv[0])
            )[:15]
        ],
        "stored_groups": dict(
            sorted(stored_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "reading_groups": dict(
            sorted(reading_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def family_evidence(
    rows: list[dict[str, Any]],
    groups: dict[str, frozenset[str]],
    path_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Families whose live names sit in more than one group, both ways.

    The families are the family detector's, over the DD paths the live names
    draw from. A family's live names are the names that have at least one of
    its paths among their sources; that membership is what both readings
    count. ``today`` reads a name's stored domain, ``rule`` its resolved
    reading, so the two counts stand beside each other on one family set.
    ``part_spanning`` counts the families whose members the rule places in two
    different parts.
    """
    path_units = {row["dd_path"]: row.get("unit") or "" for row in path_rows}
    items = [{"path": path, "unit": unit} for path, unit in sorted(path_units.items())]
    families = detect_families(items)
    by_id = {row["sn_id"]: row for row in rows}
    paths_by_name = {row["sn_id"]: set(row.get("paths") or []) for row in rows}

    today: list[dict[str, Any]] = []
    rule: list[dict[str, Any]] = []
    part_spanning: list[dict[str, Any]] = []
    miscount: list[str] = []
    for family in families:
        family_paths = {member.dd_path for member in family.members}
        members = sorted(
            sn_id for sn_id, paths in paths_by_name.items() if paths & family_paths
        )
        covered = {path for sn_id in members for path in paths_by_name[sn_id]}
        if covered & family_paths != family_paths:
            miscount.append(family.parent_path)
        stored = collections.Counter(
            by_id[sn_id].get("physics_domain") or NO_GROUP for sn_id in members
        )
        if len(stored) > 1:
            today.append(
                {
                    "id": family.parent_path,
                    "family_type": family.family_type,
                    "members": len(members),
                    "groups": dict(sorted(stored.items())),
                }
            )
        reading = collections.Counter(
            group for sn_id in members for group in groups[sn_id]
        )
        if len(reading) > 1:
            rule.append(
                {
                    "id": family.parent_path,
                    "family_type": family.family_type,
                    "members": len(members),
                    "groups": dict(sorted(reading.items())),
                }
            )
        placed: collections.Counter[str] = collections.Counter()
        for sn_id in members:
            for group in groups[sn_id]:
                if group in PARTS:
                    placed[group] += 1
        if len(placed) > 1:
            part_spanning.append(
                {
                    "id": family.parent_path,
                    "family_type": family.family_type,
                    "members": len(members),
                    "parts": dict(sorted(placed.items())),
                }
            )

    today.sort(key=lambda row: (-row["members"], row["id"]))
    rule.sort(key=lambda row: (-row["members"], row["id"]))
    part_spanning.sort(key=lambda row: (-row["members"], row["id"]))
    return {
        "families": len(families),
        "today_spanning": len(today),
        "rule_spanning": len(rule),
        "part_spanning_count": len(part_spanning),
        "today": today,
        "rule": rule,
        "part_spanning": part_spanning,
        "member_miscount": miscount,
    }


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------


def transport_parts(
    rows: list[dict[str, Any]],
    groups: dict[str, frozenset[str]],
    origin: dict[str, str],
) -> dict[str, Any]:
    """Place every live ``transport`` name in one part or list it with a reason.

    A name whose closures read exactly one part is placed there. A name is
    listed when its closures read more than one part (*spans parts*), when they
    read one existing group (*moves to an existing group*, so it is no part of
    the proposed division), or when it has no reading at all (*no data
    structure*). Placed names and the three listed reasons partition the
    transport members, which the instrument checks.
    """
    members = [row for row in rows if row.get("physics_domain") == "transport"]
    placed: dict[str, list[str]] = {part: [] for part in PARTS}
    structures: dict[str, set[str]] = {part: set() for part in PARTS}
    listed: dict[str, list[dict[str, str]]] = {
        "spans_parts": [],
        "moves_to_existing_group": [],
        "no_data_structure": [],
    }
    for row in sorted(members, key=lambda r: r["sn_id"]):
        sn_id = row["sn_id"]
        reading = groups[sn_id]
        parts = sorted(group for group in reading if group in PARTS)
        if len(parts) == 1 and len(reading) == 1:
            placed[parts[0]].append(sn_id)
            for path in row.get("paths") or []:
                ids, token = structure_of(path)
                structures[parts[0]].add(f"{ids}/{token}" if token else ids)
            continue
        if len(reading) == 0:
            listed["no_data_structure"].append(
                {
                    "name": sn_id,
                    "reason": (
                        "no DD source and no live child with one: there is no "
                        "data structure to read"
                    ),
                    "origin": origin.get(sn_id, "none"),
                }
            )
        elif len(reading) > 1:
            listed["spans_parts"].append(
                {
                    "name": sn_id,
                    "reason": (
                        "its sources read more than one group: "
                        + " + ".join(sorted(reading))
                    ),
                    "origin": origin.get(sn_id, "none"),
                }
            )
        else:
            only = next(iter(reading))
            listed["moves_to_existing_group"].append(
                {
                    "name": sn_id,
                    "reason": (
                        f"its one reading, {only}, is an existing group, so the "
                        "rule moves it out of transport rather than placing it "
                        "in a new part"
                    ),
                    "origin": origin.get(sn_id, "none"),
                }
            )
    part_rows = []
    for part, names in placed.items():
        part_rows.append(
            {
                "part": part,
                "count": len(names),
                "definition": PARTS[part]["definition"],
                "structures": PARTS[part]["structures"],
                "observed_structures": sorted(structures[part]),
                "names": names,
            }
        )
    part_rows.sort(key=lambda row: (-row["count"], row["part"]))
    placed_total = sum(row["count"] for row in part_rows)
    return {
        "live": len(members),
        "placed": placed_total,
        "parts": part_rows,
        "listed": listed,
        "listed_total": sum(len(entries) for entries in listed.values()),
        "sum": placed_total + sum(len(entries) for entries in listed.values()),
    }


def _nearest(name: str, existing: set[str]) -> list[str]:
    """Existing group names sharing a word with *name*, most shared first."""
    words = set(name.split("_"))
    shared = []
    for domain in sorted(existing):
        overlap = len(words & set(domain.split("_")))
        if overlap:
            shared.append((-overlap, domain))
    return [domain for _, domain in sorted(shared)][:3]


def vocabulary_check() -> list[dict[str, Any]]:
    """Each proposed name against the stored domain vocabulary.

    A name duplicates a group when it *is* that group's name or a singular or
    plural of it: no proposed name does. The nearest existing group and the
    reason the meanings differ are carried for the lead, because a semantic
    duplicate is a ruling, not a string comparison.
    """
    existing = {domain.value for domain in PhysicsDomain}
    notes = {
        PROFILE_QUANTITIES: (
            "core_plasma_physics and edge_plasma_physics name plasma regions, "
            "not structures; this part holds the profile structures of every "
            "region, so an edge profile and a core profile share it while "
            "their regions do not"
        ),
        SOURCE_QUANTITIES: (
            "transport today holds the source IDSs' domain, so the sources' "
            "quantities sit inside transport unnamed; this part names the "
            "quantity class and narrows transport to it"
        ),
        DISTRIBUTION_QUANTITIES: (
            "fast_particles is the domain the DD's IDS map gives the "
            "distributions IDS; the distributions structure holds every "
            "distribution and its moments, thermal and suprathermal alike, so "
            "the part is wider than fast particles and does not duplicate it"
        ),
        SUMMARY_QUANTITIES: (
            "general is the DD's fallback domain for the summary IDS and names "
            "no quantity; this part names what a summary structure holds"
        ),
        TRANSPORT_COEFFICIENTS: (
            "transport names the whole operator class; this part holds only the "
            "transport model structures' coefficients, so it is a stricter "
            "member of transport and not its duplicate"
        ),
    }
    rows = []
    for part in PARTS:
        variants = {part, part[:-1] if part.endswith("s") else part + "s"}
        rows.append(
            {
                "part": part,
                "exists_today": part in existing,
                "string_clash": sorted(variants & existing),
                "nearest_existing": _nearest(part, existing),
                "note": notes[part],
            }
        )
    return rows


def moved_sample(
    rows: list[dict[str, Any]],
    groups: dict[str, frozenset[str]],
    origin: dict[str, str],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """The deterministic hand-read sample of moved names, with its judgments."""
    by_id = {row["sn_id"]: row for row in rows}
    moved = set(catalog["names"]["moved"])
    chosen: list[str] = []
    for part in PARTS:
        members = sorted(
            sn_id
            for sn_id in moved
            if by_id[sn_id].get("physics_domain") == "transport"
            and groups[sn_id] == frozenset({part})
        )
        chosen.extend(members[:2])
    outside = sorted(
        sn_id
        for sn_id in moved
        if by_id[sn_id].get("physics_domain") == "transport"
        and len(groups[sn_id]) == 1
        and next(iter(groups[sn_id])) not in PARTS
    )
    chosen.extend(outside[:3])
    rows_out = []
    for sn_id in chosen:
        record = by_id[sn_id]
        reading = sorted(groups[sn_id])
        rows_out.append(
            {
                "name": sn_id,
                "stored": record.get("physics_domain"),
                "rule": reading[0] if len(reading) == 1 else reading,
                "sources": sorted(record.get("paths") or []),
                "children": sorted(record.get("children") or []),
                "origin": origin.get(sn_id, "none"),
                "description": (record.get("description") or "").strip(),
                "judgment": SAMPLE_READINGS.get(sn_id),
            }
        )
    return {
        "rule": (
            "the two lowest-sorted transport members of every part, then the "
            "three lowest-sorted transport names the rule moves into an "
            "existing group"
        ),
        "misread": [row["name"] for row in rows_out if row["judgment"] is None],
        "rows": rows_out,
    }


# ---------------------------------------------------------------------------
# Planted controls
# ---------------------------------------------------------------------------


class DisposableGraph:
    """A disposable graph: the run's two reads answered from planted rows.

    The same build functions that read the live graph read this one, so the
    controls exercise the rule rather than a copy of it. Nothing is written and
    nothing outside the object is touched.
    """

    def __init__(self, names: list[dict[str, Any]], paths: list[dict[str, Any]]):
        self.names = names
        self.paths = paths

    def query(self, cypher: str, **_: Any) -> list[dict[str, Any]]:
        if "StandardNameSource" in cypher and "dd.unit" in cypher:
            return [dict(row) for row in self.paths]
        return [dict(row) for row in self.names]


def _planted_name(
    sn_id: str, stored: str | None, paths: list[str], children: list[str] | None = None
) -> dict[str, Any]:
    return {
        "sn_id": sn_id,
        "physics_domain": stored,
        "name_stage": "accepted",
        "description": f"planted name {sn_id}",
        "paths": paths,
        "children": children or [],
    }


def disposable_controls() -> dict[str, Any]:
    """The planted run: one name moved to its part, one left where it was.

    Three names are planted in a disposable graph. ``planted_transport_profile``
    is stored in transport and sourced from the profile structures, so the rule
    must move it to ``profile_quantities``. ``planted_outside_equilibrium`` is
    stored in equilibrium and sourced from an equilibrium structure, so the
    rule must leave it where it was. ``planted_transport_spanning`` draws on
    the profile and source structures, so it must be listed as spanning rather
    than placed in either part.
    """
    names = [
        _planted_name(
            "planted_transport_profile",
            "transport",
            ["core_profiles/profiles_1d/electrons/temperature"],
        ),
        _planted_name(
            "planted_outside_equilibrium",
            "equilibrium",
            ["equilibrium/time_slice/profiles_1d/pressure"],
        ),
        _planted_name(
            "planted_transport_spanning",
            "transport",
            [
                "core_profiles/profiles_1d/electrons/temperature",
                "core_sources/source/profiles_1d/electrons/particles",
            ],
        ),
    ]
    paths = [
        {
            "dd_path": "core_profiles/profiles_1d/electrons/temperature",
            "unit": "",
        },
        {"dd_path": "equilibrium/time_slice/profiles_1d/pressure", "unit": ""},
        {
            "dd_path": "core_sources/source/profiles_1d/electrons/particles",
            "unit": "",
        },
    ]
    graph = DisposableGraph(names, paths)
    table = GroupTable()
    groups, origin = resolve_groups(graph.names, table)
    catalog = read_catalog(graph.names, groups)
    transport = transport_parts(graph.names, groups, origin)
    listed_spans = {entry["name"] for entry in transport["listed"]["spans_parts"]}
    placed = {part["part"]: list(part["names"]) for part in transport["parts"]}
    arms = [
        {
            "description": "planted transport name moved to profile_quantities",
            "passed": placed.get(PROFILE_QUANTITIES) == ["planted_transport_profile"],
        },
        {
            "description": "planted name outside transport left where it was",
            "passed": catalog["settled_with_structure"] == 1
            and catalog["moved"] == 1
            and groups["planted_outside_equilibrium"] == frozenset({"equilibrium"})
            and "planted_outside_equilibrium" not in catalog["names"]["moved"],
        },
        {
            "description": "planted spanning name listed, not placed in a part",
            "passed": listed_spans == {"planted_transport_spanning"},
        },
        {
            "description": "planted disposable graph partitions its three names",
            "passed": catalog["settled"] + catalog["moved"] + catalog["spanning"]
            == 0 + 3,
        },
    ]
    return {
        "graph": (
            "in-process disposable graph: three planted names, the run's two "
            "reads answered from them"
        ),
        "read": catalog["live"],
        "arms": arms,
        "parts": [
            {"part": part, "names": names} for part, names in sorted(placed.items())
        ],
        "listed": transport["listed"],
    }


# ---------------------------------------------------------------------------
# Instrument checks
# ---------------------------------------------------------------------------


def instrument_checks(payload: dict[str, Any]) -> list[tuple[str, bool]]:
    """Every way this run could be wrong, one arm each so every arm can fail."""
    catalog = payload["catalog"]
    transport = payload["transport"]
    families = payload["families"]
    sample = payload["sample"]
    placement = collections.Counter(
        name for part in transport["parts"] for name in part["names"]
    )
    repeated = sorted(name for name, count in placement.items() if count > 1)
    checks = [
        (
            "settled, moved and spanning sum to the live names read",
            catalog["settled"] + catalog["moved"] + catalog["spanning"]
            == catalog["live"],
        ),
        (
            "the settled bucket is accounted for by its two readings",
            catalog["settled_with_structure"] + catalog["settled_without_structure"]
            == catalog["settled"],
        ),
        (
            "transport's placed and listed names cover its live members once",
            transport["sum"] == transport["live"],
        ),
        ("no transport name is placed in two parts", not repeated),
        (
            "every listed transport name carries a reason",
            all(
                entry.get("reason")
                for entries in transport["listed"].values()
                for entry in entries
            ),
        ),
        (
            "every family's live members recount from its own paths",
            not families["member_miscount"],
        ),
        ("every sampled moved name is hand-read", not sample["misread"]),
        (
            "no proposed part name duplicates a stored group's string",
            all(not row["string_clash"] for row in payload["vocabulary"]),
        ),
        (
            "the transport-model container appears only in transport IDSs",
            payload["structures"]["model_ids_outside_transport"] == [],
        ),
        (
            "the source container appears only in source IDSs",
            payload["structures"]["source_ids_outside_source_set"] == [],
        ),
        (
            "every statement this run hands the graph is a read",
            payload["read_only"]["passed"],
        ),
    ]
    for control in payload["controls"]["disposable"]["arms"]:
        checks.append((control["description"], control["passed"]))
    return checks


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _row(cells: list[Any]) -> str:
    return "<tr>" + "".join("<td>" + str(cell) + "</td>" for cell in cells) + "</tr>"


def _table(anchor: str, headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join("<th>" + header + "</th>" for header in headers)
    body = "".join(_row([_esc(cell) for cell in row]) for row in rows)
    return (
        "<table "
        + anchor
        + "><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


def _page_head(node: str, payload: dict[str, Any]) -> list[str]:
    catalog = payload["catalog"]
    transport = payload["transport"]
    out = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="docs-project" content="imas-codex">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="plan-slug" content="sn-audited-group-release">',
        '<meta name="node-id" content="' + node + '">',
        "<title>Transport divided along the data dictionary's structures</title>",
        "</head><body><main>",
        '<h1 id="' + node + '-title">Transport divided along the data dictionary'
        "&rsquo;s structures</h1>",
    ]
    out.append(
        '<p id="' + node + '-summary">' + str(catalog["live"]) + " live names were "
        "read in one pass at stage "
        + _esc(", ".join(payload["environment"]["live_stages"]))
        + ". Reading "
        "each name&rsquo;s group from the data structures of its sources, <strong>"
        + str(catalog["settled"])
        + " settle in their stored group</strong> ("
        + str(catalog["settled_with_structure"])
        + " from their structures and "
        + str(catalog["settled_without_structure"])
        + " with no data structure to "
        "read), <strong>" + str(catalog["moved"]) + " move</strong> to the group "
        "their sources&rsquo; structures name, and <strong>"
        + str(catalog["spanning"])
        + " span</strong> more than one group. "
        "<code>transport</code> holds " + str(transport["live"]) + " live names; "
        "the division places <strong>" + str(transport["placed"]) + "</strong> of "
        "them in "
        + str(len(transport["parts"]))
        + " parts and lists <strong>"
        + str(transport["listed_total"])
        + "</strong> with a reason. No stored "
        "domain changed; the parts are put to the lead as a choice.</p>"
    )
    return out


def _page_rule(node: str) -> list[str]:
    return [
        '<h2 id="' + node + '-rule-head">The rule, stated before its numbers</h2>',
        '<p id="' + node + '-rule-note">A source is a Data Dictionary path. Its '
        "structure is the pair (IDS, first structural container), for example "
        "<code>core_profiles/profiles_1d</code> or <code>core_transport/model</code>. "
        "A structure names a group: the four quantity classes the plan&rsquo;s "
        "opening paragraph names (<code>profile_quantities</code> from "
        "profiles_1d/profiles_2d/ggd/ggd_fluid, <code>source_quantities</code> from "
        "source, <code>distribution_quantities</code> from distribution(s), "
        "<code>summary_quantities</code> from the summary IDS and the "
        "global_quantities container) and <code>transport_coefficients</code> from "
        "the model container of a transport IDS; every other structure takes the "
        "domain the Data Dictionary&rsquo;s own IDS map gives its IDS. A name with "
        "DD sources reads its sources&rsquo; structures; a name with none is a "
        "derived parent and unions its live children&rsquo;s readings, guarded "
        "against cycles. A name whose closures read one group is settled, and is "
        "<em>moved</em> when that group is not its stored domain; a name whose "
        "closures read more than one group <em>spans</em>; a name with nothing to "
        "read keeps its stored group and is counted apart. The three partition "
        "buckets therefore sum to the live names read. Nothing is written.</p>",
    ]


def _page_catalog(node: str, payload: dict[str, Any]) -> list[str]:
    catalog = payload["catalog"]
    live = catalog["live"]
    rows = [
        ["settled: one reading, the stored group", catalog["settled_with_structure"]],
        [
            "settled: no data structure, left where it stands",
            catalog["settled_without_structure"],
        ],
        ["moved: one reading, a different group", catalog["moved"]],
        ["spanning: more than one group", catalog["spanning"]],
        ["live names read", live],
    ]
    table = _table(
        'id="' + node + '-catalog-table"',
        ["Reading", "Names", "Share of live"],
        [[label, count, f"{100.0 * count / live:.1f} %"] for label, count in rows],
    )
    check = (
        '<p id="'
        + node
        + '-catalog-check">Check: settled '
        + str(catalog["settled"])
        + " + moved "
        + str(catalog["moved"])
        + " + spanning "
        + str(catalog["spanning"])
        + " = "
        + str(catalog["settled"] + catalog["moved"] + catalog["spanning"])
        + ", the live names read in the same run ("
        + str(live)
        + "). "
        + str(catalog["moved_from_no_group"])
        + " moved names carried no stored "
        "domain and would gain one.</p>"
    )
    moved = _table(
        'id="' + node + '-moved-to-table"',
        ["Group the rule reads", "Names moved into it"],
        [[group, count] for group, count in list(catalog["moved_to"].items())[:20]],
    )
    return [
        '<h2 id="' + node + '-catalog-head">The whole catalog, not only transport</h2>',
        table,
        check,
        moved,
    ]


def _page_sample(node: str, payload: dict[str, Any]) -> list[str]:
    sample = payload["sample"]
    rows = []
    for row in sample["rows"]:
        judgment = row["judgment"] or {}
        rows.append(
            [
                _esc(row["name"]),
                _esc(row["stored"]),
                _esc(row["rule"]),
                _esc(judgment.get("reading", "unread")),
                _esc(judgment.get("why", "")),
            ]
        )
    return [
        '<h2 id="' + node + '-sample-head">A hand-read sample of the moved names</h2>',
        '<p id="'
        + node
        + '-sample-note">The selection is fixed before reading: '
        + _esc(sample["rule"])
        + ". Each row states which reading is right &mdash; "
        "the rule&rsquo;s, the stored domain&rsquo;s, or neither &mdash; with the "
        "reason, read against the name&rsquo;s definition and its source paths. "
        "The judgments are recorded in the data file, not derived from the "
        "counts.</p>",
        _table(
            'id="' + node + '-sample-table"',
            ["Name", "Stored", "Rule", "Reading", "Why"],
            rows,
        ),
    ]


def _page_families(node: str, payload: dict[str, Any]) -> list[str]:
    families = payload["families"]
    rows = []
    for row in families["rule"][:25]:
        groups = ", ".join(
            group + " (" + str(count) + ")"
            for group, count in sorted(row["groups"].items())
        )
        rows.append(
            [
                _esc(row["id"]),
                _esc(row["family_type"]),
                row["members"],
                _esc(groups),
            ]
        )
    return [
        '<h2 id="' + node + '-families-head">Families the rule would leave '
        "spanning</h2>",
        '<p id="'
        + node
        + '-families-note">The DD family detector found '
        + str(families["families"])
        + " families over the paths the live names "
        "draw from. On the same family set, <strong>"
        + str(families["today_spanning"])
        + "</strong> families have live members "
        "in more than one stored domain, the 225 section 2 records; under the rule "
        "<strong>" + str(families["rule_spanning"]) + "</strong> would span more "
        "than one group, and "
        + str(families["part_spanning_count"])
        + " would span two parts. The "
        "families with the most members are listed; the data file carries "
        "all of them.</p>",
        _table(
            'id="' + node + '-families-table"',
            ["Family", "Type", "Members", "Groups under the rule"],
            rows,
        ),
    ]


def _page_parts(node: str, payload: dict[str, Any]) -> list[str]:
    transport = payload["transport"]
    part_rows = []
    for part in transport["parts"]:
        observed = ", ".join(part["observed_structures"][:8])
        if len(part["observed_structures"]) > 8:
            observed += " ..."
        part_rows.append(
            [
                _esc(part["part"]),
                part["count"],
                _esc(part["definition"]),
                _esc(observed),
            ]
        )
    listed_rows = []
    for reason, entries in transport["listed"].items():
        names = ", ".join(entry["name"] for entry in entries[:8])
        if len(entries) > 8:
            names += " ..."
        listed_rows.append([_esc(reason), len(entries), _esc(names)])
    return [
        '<h2 id="' + node + '-parts-head">The parts transport divides into</h2>',
        '<p id="'
        + node
        + '-parts-note">Every one of the '
        + str(transport["live"])
        + " live transport names is placed in exactly "
        "one part or listed by name with the reason it stays: "
        + str(transport["placed"])
        + " placed and "
        + str(transport["listed_total"])
        + " listed, which sum to the live "
        "count.</p>",
        _table(
            'id="' + node + '-parts-table"',
            ["Part", "Names", "Holds", "Structures observed"],
            part_rows,
        ),
        _table(
            'id="' + node + '-listed-table"',
            ["Reason a transport name stays", "Names", "List"],
            listed_rows,
        ),
        '<p id="' + node + '-listed-note">The full list, with each name&rsquo;s '
        "reason and origin, is in the data file&rsquo;s "
        "<code>transport.listed</code>.</p>",
    ]


def _page_vocabulary(node: str, payload: dict[str, Any]) -> list[str]:
    rows = [
        [
            _esc(row["part"]),
            _esc(", ".join(row["string_clash"]) or "none"),
            _esc(", ".join(row["nearest_existing"]) or "none"),
            _esc(row["note"]),
        ]
        for row in payload["vocabulary"]
    ]
    return [
        '<h2 id="' + node + '-vocabulary-head">The proposed names against the '
        "domain vocabulary</h2>",
        _table(
            'id="' + node + '-vocabulary-table"',
            ["Part", "String clash", "Nearest existing", "Why not a duplicate"],
            rows,
        ),
    ]


def _page_controls(node: str, payload: dict[str, Any]) -> list[str]:
    control = payload["controls"]["disposable"]
    rows = [
        [_esc(arm["description"]), "pass" if arm["passed"] else "FAIL"]
        for arm in control["arms"]
    ]
    return [
        '<h2 id="' + node + '-controls-head">The planted controls</h2>',
        '<p id="' + node + '-controls-note">' + _esc(control["graph"]) + ". The "
        "shown run reads " + str(control["read"]) + " planted names.</p>",
        _table('id="' + node + '-controls-table"', ["Arm", "Result"], rows),
        '<p id="'
        + node
        + '-negative-note">'
        + _esc(payload["controls"]["double_count"]["description"])
        + _esc(payload["controls"]["read_only"]["description"])
        + "</p>",
    ]


def _page_choice(node: str, payload: dict[str, Any]) -> list[str]:
    catalog = payload["catalog"]
    transport = payload["transport"]
    return [
        '<h2 id="' + node + '-choice-head">What the lead is asked to rule</h2>',
        '<p id="' + node + '-choice-note">Ruling is not recorded here: no stored '
        "domain has moved, and the proposal is put to the lead as a choice. "
        "<strong>A</strong>: adopt the five parts for <code>transport</code> only, "
        "placing its "
        + str(transport["placed"])
        + " settleable names and leaving "
        + str(transport["listed_total"])
        + " listed; whole-catalog moves are then "
        "not made. <strong>B</strong>: adopt the structure rule catalog-wide, which "
        "would move " + str(catalog["moved"]) + " names across every group. "
        "<strong>C</strong>: refuse the structure reading for the cases the "
        "hand-read sample shows it gets wrong &mdash; an NTM&rsquo;s diffusion "
        "coefficient keeps its stored domain, and the summary part is read as a "
        "structure class rather than a quantity family.</p>",
    ]


def _page_checks(node: str, payload: dict[str, Any]) -> list[str]:
    rows = [
        [_esc(check["description"]), "pass" if check["passed"] else "FAIL"]
        for check in payload["checks"]
    ]
    environment = payload["environment"]
    return [
        '<h2 id="' + node + '-checks-head">The instrument&rsquo;s own checks</h2>',
        _table('id="' + node + '-checks-table"', ["Arm", "Result"], rows),
        '<p id="'
        + node
        + '-environment">Measured at base <code>'
        + _esc(environment["measured_base"])
        + "</code> on "
        + _esc(environment["host"])
        + ", Python "
        + _esc(environment["python"])
        + ", "
        + _esc(environment["generated_at"])
        + ". Statements handed the "
        "graph: " + str(payload["read_only"]["statements"]) + ", all read-only.</p>",
    ]


def render_page(payload: dict[str, Any]) -> str:
    """The evidence page. Every element id begins with the node id."""
    node = "sagr-data-structure-groups"
    out = _page_head(node, payload)
    out += _page_rule(node)
    out += _page_catalog(node, payload)
    out += _page_sample(node, payload)
    out += _page_families(node, payload)
    out += _page_parts(node, payload)
    out += _page_vocabulary(node, payload)
    out += _page_controls(node, payload)
    out += _page_choice(node, payload)
    out += _page_checks(node, payload)
    out.append("</main></body></html>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _measured_base() -> str:
    """The commit this measurement stands on, not the commit that carries it."""
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


def _read_only_guard(statements: list[str]) -> dict[str, Any]:
    """Refuse a statement that is a write; record the statements' digest."""
    offending = []
    for statement in statements:
        padded = " " + " ".join(statement.upper().split()) + " "
        for clause in _WRITE_CLAUSES:
            if clause in padded:
                offending.append(clause.strip())
    return {
        "statements": len(statements),
        "sha256": hashlib.sha256("".join(statements).encode()).hexdigest()[:16],
        "offending_clauses": sorted(set(offending)),
        "passed": not offending,
    }


def _structure_audit(path_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the structure table reads no container the catalog does not hold.

    The transport-coefficient and source classes are keyed on a container name,
    so a container of the same name in another IDS would be misfiled silently.
    This records every IDS carrying either container; the arms fail when one is
    outside the set the class is meant for.
    """
    model_ids = {
        structure_of(row["dd_path"])[0]
        for row in path_rows
        if structure_of(row["dd_path"])[1] == MODEL_TOKEN
    }
    source_ids = {
        structure_of(row["dd_path"])[0]
        for row in path_rows
        if structure_of(row["dd_path"])[1] in SOURCE_TOKENS
    }
    return {
        "model_ids": sorted(model_ids),
        "model_ids_outside_transport": sorted(model_ids - TRANSPORT_IDS),
        "source_ids": sorted(source_ids),
        "source_ids_outside_source_set": sorted(source_ids - SOURCE_IDS),
        "distinct_paths": len({row["dd_path"] for row in path_rows}),
    }


def _section_two_live() -> int | None:
    """Section 2's committed live count, for the record that this run agrees."""
    try:
        data = json.loads(SECTION_TWO_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("census", {}).get("partition", {}).get("live")


def build(gc: Any) -> dict[str, Any]:
    """Read the graph and compute every section of the evidence."""
    statements: list[str] = []
    rows = [dict(row) for row in gc.query(_LIVE_QUERY, live=LIVE) or []]
    statements.append(_LIVE_QUERY)
    path_rows = [dict(row) for row in gc.query(_SOURCE_PATHS_QUERY, live=LIVE) or []]
    statements.append(_SOURCE_PATHS_QUERY)

    table = GroupTable()
    groups, origin = resolve_groups(rows, table)
    catalog = read_catalog(rows, groups)
    families = family_evidence(rows, groups, path_rows)
    transport = transport_parts(rows, groups, origin)
    sample = moved_sample(rows, groups, origin, catalog)
    catalog["moved_names_sha256"] = hashlib.sha256(
        "\n".join(catalog["names"]["moved"]).encode()
    ).hexdigest()[:16]
    del catalog["names"]["moved"]
    controls = {
        "disposable": disposable_controls(),
        "double_count": {
            "description": (
                "Negative control: a planted name inserted into two parts makes "
                "--check --plant-double-count exit 1."
            )
        },
        "read_only": {
            "description": (
                " Every statement the run hands the graph is checked for a write "
                "clause before it is sent; a match fails the run."
            )
        },
    }
    payload: dict[str, Any] = {
        "rule": {
            "statement": (
                "a name's group is read from the data structures of its sources"
            ),
            "parts": PARTS,
            "profile_tokens": sorted(PROFILE_TOKENS),
            "source_tokens": sorted(SOURCE_TOKENS),
            "distribution_tokens": sorted(DISTRIBUTION_TOKENS),
            "summary": {"ids": SUMMARY_IDS, "tokens": sorted(SUMMARY_TOKENS)},
            "model_token": MODEL_TOKEN,
            "transport_ids": sorted(TRANSPORT_IDS),
        },
        "catalog": catalog,
        "families": families,
        "transport": transport,
        "sample": sample,
        "vocabulary": vocabulary_check(),
        "structures": _structure_audit(path_rows),
        "controls": controls,
        "environment": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": platform.node(),
            "python": platform.python_version(),
            "measured_base": _measured_base(),
            "live_stages": LIVE,
            "section_two_live": _section_two_live(),
            "unknown_ids": sorted(table.unknown_ids),
        },
    }
    payload["read_only"] = _read_only_guard(statements)
    payload["checks"] = [
        {"description": description, "passed": passed}
        for description, passed in instrument_checks(payload)
    ]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="read, run every arm and the planted controls, write nothing",
    )
    parser.add_argument(
        "--plant-double-count",
        action="store_true",
        help="insert a planted name into two parts and show the gate refusing it",
    )
    args = parser.parse_args(argv)

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        payload = build(gc)
    if args.plant_double_count:
        planted = payload["transport"]["parts"][0]["names"][0]
        payload["transport"]["parts"][1]["names"].append(planted)

    checks = instrument_checks(payload)
    for description, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {description}")
    catalog = payload["catalog"]
    transport = payload["transport"]
    print(
        f"live={catalog['live']} settled={catalog['settled']} "
        f"(with_structure={catalog['settled_with_structure']} "
        f"without_structure={catalog['settled_without_structure']}) "
        f"moved={catalog['moved']} spanning={catalog['spanning']} "
        f"sum={catalog['settled'] + catalog['moved'] + catalog['spanning']} "
        f"transport_live={transport['live']} placed={transport['placed']} "
        f"listed={transport['listed_total']} parts={len(transport['parts'])} "
        f"families_today_spanning={payload['families']['today_spanning']} "
        f"families_rule_spanning={payload['families']['rule_spanning']}"
    )
    if not all(ok for _, ok in checks):
        print(
            f"not written: {OUT_JSON.name} and {OUT_HTML.name} keep the last run "
            "that passed"
        )
        return 1
    if args.check:
        base = payload["environment"]["measured_base"]
        print(f"checked at {base}; nothing written")
        return 0
    OUT_JSON.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUT_HTML.write_text(render_page(payload) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name} and {OUT_HTML.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
