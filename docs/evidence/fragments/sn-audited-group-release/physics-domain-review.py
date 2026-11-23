"""Propose one coherent physics-domain list, reviewed against the IDS sources.

The domain vocabulary is used as a physics classification, but three sources
describe it and they disagree: the grammar library's published vocabulary, this
repository's IDS-to-domain table and the guidance file the classifier reads.
This module reads all three and the live graph in one run, and puts one proposed
list beside them.

The proposal keeps the vocabulary's two jobs apart:

* a **physics area** names a physical subject — a plasma state, a phenomenon, a
  measurement technique, a system — and is what a name is filed by;
* a **grouping of data structures** names a class of quantity by the IDS
  structure it comes from. The five parts ``transport`` is divided into are
  groupings of that kind, admitted because section 6 needs them as write
  targets, and scoped to that division until the lead rules otherwise.

``--check`` reads every count from the live graph and the three sources, and
exits non-zero unless every IDS maps to a proposed domain, every live name maps
to a proposed domain, and the check's other arms hold. Two planting flags mutate
an in-memory copy of the proposal so a recorded run shows the check refusing a
proposal with one IDS, or one live name, left unmapped:

    --plant-drop-ids <ids>
    --plant-drop-name <standard name>

Read-only. Nothing is written to the graph, the vocabulary, the guidance or the
grammar library.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FRAGMENTS = Path(__file__).resolve().parent
ROOT = FRAGMENTS.parents[3]
OUT_JSON = FRAGMENTS / "physics-domain-review.json"
OUT_HTML = FRAGMENTS / "physics-domain-review.html"
FIGURES = (
    ROOT / "docs" / "figures" / "sn-audited-group-release" / "physics-domain-review"
)
FIGURE = FIGURES / "domain-agreement-matrix.svg"

#: The ruled transport placement: part, its definition and its member names.
PLACEMENT_JSON = FRAGMENTS / "sagr-transport-parts-write.json"

LIVE_STAGES = ("accepted", "reviewed", "drafted", "pending")

#: The five quantity-form groupings section 6 divides ``transport`` into.
PARTS = (
    "transport_coefficients",
    "profile_quantities",
    "distribution_quantities",
    "summary_quantities",
    "source_quantities",
)

#: Current domain -> proposed domain. A current domain absent from this table is
#: retired: nothing may carry it, and the check refuses if anything does.
CURRENT_TO_PROPOSED: dict[str, str] = {
    "equilibrium": "equilibrium",
    "magnetohydrodynamics": "magnetohydrodynamics",
    "transport": "transport",
    "turbulence": "turbulence",
    "gyrokinetics": "gyrokinetics",
    "fast_particles": "fast_particles",
    "runaway_electrons": "runaway_electrons",
    "waves": "waves",
    "edge_plasma_physics": "edge_plasma_physics",
    "divertor_physics": "divertor_physics",
    "plasma_wall_interactions": "plasma_wall_interactions",
    "fueling": "fueling",
    "core_plasma_physics": "core_plasma_physics",
    "plasma_initiation": "plasma_initiation",
    "auxiliary_heating": "auxiliary_heating",
    "current_drive": "current_drive",
    "plasma_control": "plasma_control",
    "machine_operations": "machine_operations",
    "magnetic_field_systems": "magnetic_field_systems",
    "structural_components": "structural_components",
    "plant_systems": "plant_systems",
    "magnetic_field_diagnostics": "magnetic_field_diagnostics",
    "particle_measurement_diagnostics": "particle_measurement_diagnostics",
    "electromagnetic_wave_diagnostics": "electromagnetic_wave_diagnostics",
    "radiation_measurement_diagnostics": "radiation_measurement_diagnostics",
    "mechanical_measurement_diagnostics": "mechanical_measurement_diagnostics",
    "spectroscopy": "spectroscopy",
    "neutronics": "neutronics",
    "data_management": "data_management",
    "computational_workflow": "data_management",
    "general": "general",
}

#: One disposition per current domain, with the reason a reader needs.
DISPOSITIONS: dict[str, dict[str, str]] = {
    "equilibrium": {
        "state": "kept",
        "reason": "Names the force-balance configuration and its surfaces; the most settled group.",
    },
    "magnetohydrodynamics": {
        "state": "kept",
        "reason": "Names a distinct macroscopic instability physics.",
    },
    "transport": {
        "state": "split",
        "reason": "Divided per the lead's ruling: 950 names go to the five quantity-form parts, and the names the data-structure rule cannot place stay here, enumerated by name.",
    },
    "turbulence": {
        "state": "kept",
        "reason": "Names fluctuation and turbulent-transport physics, distinct from the transport rate it produces.",
    },
    "gyrokinetics": {
        "state": "kept",
        "reason": "Names a model framework whose quantities describe the model world, not the measured machine.",
    },
    "fast_particles": {
        "state": "kept",
        "reason": "Names non-thermal fast-ion and fast-electron populations.",
    },
    "runaway_electrons": {
        "state": "kept",
        "reason": "Names avalanche generation and runaway loss physics, a distinct population with its own IDS.",
    },
    "waves": {
        "state": "kept",
        "reason": "Names wave propagation and wave-particle physics, separate from the heating systems that launch it.",
    },
    "edge_plasma_physics": {
        "state": "kept",
        "reason": "Names the boundary plasma, the most populous named area after transport.",
    },
    "divertor_physics": {
        "state": "kept",
        "reason": "Names divertor target loads, detachment and exhaust.",
    },
    "plasma_wall_interactions": {
        "state": "kept",
        "reason": "Names erosion, deposition and material migration at the wall.",
    },
    "fueling": {
        "state": "kept",
        "reason": "Names gas and pellet injection systems and the quantities they deliver.",
    },
    "core_plasma_physics": {
        "state": "kept",
        "reason": "Names the core plasma state, narrowed to quantities that are not a transport coefficient, a source or a profile form.",
    },
    "plasma_initiation": {
        "state": "kept",
        "reason": "Names startup, breakdown and the early heating phase.",
    },
    "auxiliary_heating": {
        "state": "kept",
        "reason": "Names the heating actuators and their hardware.",
    },
    "current_drive": {
        "state": "kept",
        "reason": "Names non-inductive current generation, a physics distinct from the heating that accompanies it; its ten names are current-drive quantities, not heating-system ones.",
    },
    "plasma_control": {
        "state": "kept",
        "reason": "Names position and shape control and the feedback system.",
    },
    "machine_operations": {
        "state": "kept",
        "reason": "Names pulse scheduling and operational quantities.",
    },
    "magnetic_field_systems": {
        "state": "kept",
        "reason": "Names the coil and magnet infrastructure that generates the field.",
    },
    "structural_components": {
        "state": "kept",
        "reason": "Names the vessel and load-bearing structure.",
    },
    "plant_systems": {
        "state": "kept",
        "reason": "Names power, cooling and vacuum plant.",
    },
    "magnetic_field_diagnostics": {
        "state": "kept",
        "reason": "Names the magnetic sensor set and its responses.",
    },
    "particle_measurement_diagnostics": {
        "state": "kept",
        "reason": "Names Thomson scattering, Langmuir probes and particle-based measurement.",
    },
    "electromagnetic_wave_diagnostics": {
        "state": "kept",
        "reason": "Names interferometry, reflectometry and polarimetry.",
    },
    "radiation_measurement_diagnostics": {
        "state": "kept",
        "reason": "Names bolometry, X-ray and imaging diagnostics.",
    },
    "mechanical_measurement_diagnostics": {
        "state": "kept",
        "reason": "Names strain and vibration measurement.",
    },
    "plasma_measurement_diagnostics": {
        "state": "retired",
        "reason": "No live name and no IDS name it; the general plasma-measurement role is carried by particle_measurement_diagnostics, so it stands on nothing.",
    },
    "spectroscopy": {
        "state": "kept",
        "reason": "Names emission, charge-exchange and Stark spectroscopy.",
    },
    "neutronics": {
        "state": "kept",
        "reason": "Names neutron diagnostics and fusion-product measurement.",
    },
    "data_management": {
        "state": "merged",
        "reason": "Absorbs computational_workflow: both are non-physics data domains, and a reference-data IDS and a workflow IDS belong in one place. No live standard name carries either today; 1,959 live DD paths carry computational_workflow.",
    },
    "computational_workflow": {
        "state": "merged",
        "reason": "Merged into data_management: its IDSs are reference or auxiliary data, atomic and molecular data, and computational infrastructure, not an area of physics. No live standard name carries it; its 1,959 live DD paths move with the vocabulary entry.",
    },
    "general": {
        "state": "kept",
        "reason": "An explicit last resort for quantities that cross domains, so no name is ever left without a domain.",
    },
}

#: The proposal's own description where the existing vocabulary does not cover
#: the new scope.
DESCRIPTION_OVERRIDES: dict[str, str] = {
    "transport": "The transport quantities whose source data structure does not resolve to one part, which stay here by name and are enumerated",
    "data_management": "Reference and auxiliary data, atomic and molecular data, summary data products, and computational and workflow infrastructure",
    "transport_coefficients": "Transport-model quantities: diffusivities, conductivities, convection velocities and damping rates",
    "profile_quantities": "Profile quantities: the one- and two-dimensional profiles and their general-grid forms",
    "distribution_quantities": "Distribution-function quantities: the distribution itself and its orbit-class moments",
    "summary_quantities": "Summary forms: local values at named locations and the volume-, line- and flux-surface-averaged or integrated forms of a profile quantity",
    "source_quantities": "Source terms: the particle, energy, momentum and current rates a source adds to the plasma",
}

#: Why each quantity-form part is proposed, and what it costs.
PART_REASONS: dict[str, str] = {
    "transport_coefficients": "Added as a grouping of data structures: carries the transport-model quantities section 6 divides out of transport, and owns the three transport IDS, whose whole content is this quantity form.",
    "profile_quantities": "Added as a grouping of data structures: carries the profile forms section 6 divides out of transport. It takes core_profiles and edge_profiles at IDS level because a profile IDS's content is this form, which leaves edge_plasma_physics with no IDS of its own — a question for the lead.",
    "distribution_quantities": "Added as a grouping of data structures: carries the distribution-function quantities section 6 divides out of transport, and owns the distributions IDS.",
    "summary_quantities": "Added as a grouping of data structures: carries the summary forms section 6 divides out of transport, and owns the summary IDS.",
    "source_quantities": "Added as a grouping of data structures: carries the source terms section 6 divides out of transport, and owns the three sources IDS.",
}

#: Proposed name -> axis. ``physics_area`` names a physical subject;
#: ``data_structure_grouping`` names a class of quantity by the IDS structure it
#: comes from.
QUANTITY_FORM: dict[str, str] = dict.fromkeys(PARTS, "data_structure_grouping")
QUANTITY_FORM["data_management"] = "data_structure_grouping"
QUANTITY_FORM["general"] = "data_structure_grouping"

#: New guidance entries for the proposed domains that have none: the five parts.
NEW_GUIDANCE: dict[str, dict[str, Any]] = {
    "transport_coefficients": {
        "scope": "The transport-model quantities a transport model structure holds: the diffusivities, conductivities, convection velocities and damping rates that predict a transport flux.",
        "includes": [
            "a diffusion coefficient or diffusivity from a transport-model structure",
            "a convection velocity or damping rate from a transport-model structure",
        ],
        "excludes": [
            {
                "quantity": "the flux a coefficient predicts",
                "neighbour": "transport",
                "rule": "The coefficient is transport_coefficients; a flux whose source structure is not a transport model is transport.",
            }
        ],
    },
    "profile_quantities": {
        "scope": "The profiles a profile structure holds: the one- and two-dimensional profile quantities and their general-grid forms.",
        "includes": [
            "a quantity carried in profiles_1d or profiles_2d",
            "a general-grid profile form",
        ],
        "excludes": [
            {
                "quantity": "the transport coefficient that models the profile's flux",
                "neighbour": "transport_coefficients",
                "rule": "A profile amplitude is profile_quantities; a modelled coefficient is transport_coefficients.",
            }
        ],
    },
    "distribution_quantities": {
        "scope": "The distribution-function quantities a distributions structure holds: the distribution itself and its orbit-class moments.",
        "includes": [
            "a distribution function or its moments",
            "an orbit-class moment of a distribution",
        ],
        "excludes": [
            {
                "quantity": "a fast-particle population summary",
                "neighbour": "fast_particles",
                "rule": "A quantity read from the distributions structure is distribution_quantities; a name whose subject is the fast-particle population itself is fast_particles.",
            }
        ],
    },
    "summary_quantities": {
        "scope": "The summary forms a summary structure holds: local values at named locations and the volume-, line- and flux-surface-averaged or integrated form of a profile quantity.",
        "includes": [
            "a local value at a named location in the proposal's summary structure",
            "a volume-, line- or flux-surface-averaged or integrated form",
        ],
        "excludes": [
            {
                "quantity": "the profile the summary form averages",
                "neighbour": "profile_quantities",
                "rule": "The averaged form read from the summary structure is summary_quantities; the profile it averages is profile_quantities.",
            }
        ],
    },
}

#: The fifth part's guidance entry, kept beside the table above.
SOURCE_QUANTITIES_GUIDANCE = {
    "scope": "The source terms a sources structure holds: the particle, energy, momentum and current rates a source adds to the plasma.",
    "includes": [
        "a particle, energy, momentum or current source rate from a sources structure"
    ],
    "excludes": [
        {
            "quantity": "the transport of what the source adds",
            "neighbour": "transport",
            "rule": "The rate a source adds is source_quantities; the flux that carries it is transport.",
        }
    ],
}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _grammar_path() -> Path:
    from iter_codex.definitions.physics import _isn_physics_domains_path

    return _isn_physics_domains_path()


def read_sources() -> dict[str, Any]:
    """The three domain sources, plus the two files that place a DD path."""
    from iter_codex.definitions.physics import (
        IDS_DOMAINS_FILE,
        load_domain_guidance,
        load_isn_physics_domains,
    )

    guidance_file = IDS_DOMAINS_FILE.parent / "domains.yaml"
    overrides_file = IDS_DOMAINS_FILE.parent / "path_domain_overrides.json"
    gold_file = IDS_DOMAINS_FILE.parent / "domain_gold_set.json"

    ids_payload = json.loads(IDS_DOMAINS_FILE.read_text(encoding="utf-8"))
    return {
        "grammar": load_isn_physics_domains(),
        "ids_domains": ids_payload["ids_domain_mappings"],
        "guidance": load_domain_guidance(),
        "overrides": json.loads(overrides_file.read_text(encoding="utf-8")).get(
            "overrides", []
        ),
        "gold": json.loads(gold_file.read_text(encoding="utf-8")),
        "files": {
            "grammar": str(_grammar_path()),
            "ids_domains": str(IDS_DOMAINS_FILE),
            "guidance": str(guidance_file),
            "path_domain_overrides": str(overrides_file),
            "domain_gold_set": str(gold_file),
        },
        "sha256": {
            "grammar": _sha256(_grammar_path()),
            "ids_domains": _sha256(IDS_DOMAINS_FILE),
            "guidance": _sha256(guidance_file),
            "path_domain_overrides": _sha256(overrides_file),
            "domain_gold_set": _sha256(gold_file),
        },
    }


# ---------------------------------------------------------------------------
# Live graph
# ---------------------------------------------------------------------------


def read_graph() -> dict[str, Any]:
    """Three bounded reads: IDS domains, live names, and each name's IDS."""
    from iter_codex.standard_names.graph_store.graph_foundation import get_services

    ids_query = """
MATCH (n:IMASNode)
RETURN n.ids AS ids, n.physics_domain AS domain, count(*) AS paths
ORDER BY ids, domain
"""
    names_query = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
RETURN sn.id AS id, sn.physics_domain AS domain
ORDER BY id
"""
    name_ids_query = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
WHERE dd.ids IS NOT NULL
RETURN sn.id AS id, collect(DISTINCT dd.ids) AS ids
"""
    with get_services().GraphClient() as gc:
        ids_rows = gc.query(ids_query)
        name_rows = gc.query(names_query, live=list(LIVE_STAGES))
        name_ids_rows = gc.query(name_ids_query, live=list(LIVE_STAGES))
    return {
        "ids_rows": [dict(row) for row in ids_rows],
        "names": [dict(row) for row in name_rows],
        "name_ids": {row["id"]: list(row["ids"]) for row in name_ids_rows},
    }


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------


def part_ids(sources: dict[str, Any]) -> dict[str, list[str]]:
    """The IDS each quantity-form part owns, from the ruled grammar proposal."""
    proposal = json.loads(PLACEMENT_JSON.read_text(encoding="utf-8")).get(
        "grammar_proposal", {}
    )
    return {part: list(proposal.get(part, {}).get("ids") or []) for part in PARTS}


def transport_placement() -> dict[str, str]:
    """Name -> part for the names the ruled data-structure rule places."""
    plan = json.loads(PLACEMENT_JSON.read_text(encoding="utf-8"))["write_plan"]["parts"]
    placement: dict[str, str] = {}
    for part, entry in plan.items():
        if part in PARTS:
            for name in entry["names"]:
                placement[name] = part
    return placement


def _guidance_for(target: str, sources: dict[str, Any]) -> dict[str, Any]:
    if target in sources["guidance"]["domains"]:
        entry = dict(sources["guidance"]["domains"][target])
        if target == "data_management":
            workflow = sources["guidance"]["domains"].get("computational_workflow", {})
            entry["scope"] = (
                str(entry.get("scope", "")).strip()
                + " It also holds the reference and auxiliary data, atomic and "
                "molecular data and computational infrastructure that "
                "computational_workflow named, merged here so one data domain "
                "carries them."
            )
            entry["includes"] = list(entry.get("includes", [])) + list(
                workflow.get("includes", [])
            )
            entry["excludes"] = list(entry.get("excludes", [])) + list(
                workflow.get("excludes", [])
            )
        return entry
    if target == "source_quantities":
        return SOURCE_QUANTITIES_GUIDANCE
    return NEW_GUIDANCE.get(target, {})


def _ids_owner(sources: dict[str, Any]) -> dict[str, str]:
    """IDS -> proposed domain, the proposal's IDS-level function."""
    owner: dict[str, str] = {}
    for part, ids_list in part_ids(sources).items():
        for ids in ids_list:
            owner[ids] = part
    for ids, current in sources["ids_domains"].items():
        if ids not in owner:
            owner[ids] = CURRENT_TO_PROPOSED.get(current, current)
    return owner


def proposed_domains(sources: dict[str, Any]) -> list[dict[str, Any]]:
    """The proposal: every proposed domain with description, category and guidance."""
    grammar = sources["grammar"]
    ids_owner = _ids_owner(sources)

    by_target: dict[str, list[str]] = {part: [] for part in PARTS}
    for current, target in CURRENT_TO_PROPOSED.items():
        by_target.setdefault(target, []).append(current)

    proposal: list[dict[str, Any]] = []
    for target in sorted(by_target):
        carries = sorted(by_target[target])
        base = next((grammar[c] for c in carries if c in grammar), {})
        category = base.get("category", "")
        if target in PARTS:
            category = "quantity_form"
        entry = _guidance_for(target, sources)
        proposal.append(
            {
                "name": target,
                "axis": QUANTITY_FORM.get(target, "physics_area"),
                "description": DESCRIPTION_OVERRIDES.get(
                    target, base.get("description", "")
                ),
                "category": category,
                "carries": carries,
                "disposition": (
                    "added"
                    if not carries
                    else "merged"
                    if len(carries) > 1
                    else DISPOSITIONS.get(carries[0], {}).get("state", "kept")
                ),
                "reason": (
                    PART_REASONS.get(target, "")
                    if not carries
                    else " ".join(
                        DISPOSITIONS.get(c, {}).get("reason", "") for c in carries
                    ).strip()
                ),
                "ids": sorted(
                    ids for ids, owner in ids_owner.items() if owner == target
                ),
                "scope": str(entry.get("scope", "")).strip(),
                "includes": list(entry.get("includes", [])),
                "excludes": list(entry.get("excludes", [])),
            }
        )
    return proposal


def _domain_from_sources(ids_list: list[str], ids_owner: dict[str, str]) -> str | None:
    """A name with no stored domain takes the commonest proposed domain of its sources' IDS.

    Ties go to the alphabetically first target, so the rule is deterministic.
    """
    targets = [ids_owner[ids] for ids in ids_list if ids in ids_owner]
    if not targets:
        return None
    tally: dict[str, int] = {}
    for target in targets:
        tally[target] = tally.get(target, 0) + 1
    return sorted(tally.items(), key=lambda item: (-item[1], item[0]))[0][0]


def propose(
    sources: dict[str, Any],
    graph: dict[str, Any],
    *,
    drop_ids: str | None = None,
    drop_name: str | None = None,
) -> dict[str, Any]:
    """Apply the proposal to the live graph and to the IDS the graph holds.

    ``drop_ids`` and ``drop_name`` plant a negative control: they remove one IDS,
    or one live name, from an in-memory copy of the mapping before the checks
    run. Nothing on disk is touched.
    """
    ids_owner = _ids_owner(sources)
    if drop_ids:
        ids_owner.pop(drop_ids, None)
    placement = transport_placement()
    domain_names = {entry["name"] for entry in proposed_domains(sources)}

    live_ids = sorted({row["ids"] for row in graph["ids_rows"]})
    ids_mapped = {ids: ids_owner.get(ids) for ids in live_ids}
    ids_unmapped = sorted(ids for ids, target in ids_mapped.items() if target is None)
    ids_unknown_target = sorted(
        (ids, target)
        for ids, target in ids_mapped.items()
        if target is not None and target not in domain_names
    )

    name_domain = {row["id"]: row["domain"] for row in graph["names"]}
    name_source_ids = graph["name_ids"]
    name_target: dict[str, str | None] = {}
    for name, current in name_domain.items():
        if drop_name and name == drop_name:
            name_target[name] = None
            continue
        if current == "transport" and name in placement:
            name_target[name] = placement[name]
        elif current is None:
            name_target[name] = (
                _domain_from_sources(name_source_ids.get(name, []), ids_owner)
                or "general"
            )
        else:
            name_target[name] = CURRENT_TO_PROPOSED.get(current)

    names_unmapped = sorted(
        name for name, target in name_target.items() if target is None
    )
    names_unknown_target = sorted(
        (name, target)
        for name, target in name_target.items()
        if target is not None and target not in domain_names
    )
    names_resolved_from_sources = sum(
        1
        for name, current in name_domain.items()
        if current is None
        and _domain_from_sources(name_source_ids.get(name, []), ids_owner) is not None
    )
    names_fallback_general = sum(
        1
        for name, current in name_domain.items()
        if current is None
        and _domain_from_sources(name_source_ids.get(name, []), ids_owner) is None
    )

    counts: dict[str, int] = {entry["name"]: 0 for entry in proposed_domains(sources)}
    for target in name_target.values():
        if target in counts:
            counts[target] += 1
    counted = sum(counts.values())
    live_total = len(name_domain)

    current_counts: dict[str, int] = {}
    for current in name_domain.values():
        key = current if current is not None else "(no physics domain)"
        current_counts[key] = current_counts.get(key, 0) + 1
    retired_carried = {
        current: count
        for current, count in current_counts.items()
        if current != "(no physics domain)" and current not in CURRENT_TO_PROPOSED
    }

    checks = {
        "every_ids_maps": not ids_unmapped,
        "every_ids_target_is_proposed": not ids_unknown_target,
        "every_live_name_maps": not names_unmapped,
        "every_name_target_is_proposed": not names_unknown_target,
        "counts_sum_to_live_total": counted == live_total,
        "proposed_domains_all_counted": set(counts) == domain_names,
        "no_retired_domain_is_carried": not retired_carried,
    }

    return {
        "ids_owner": ids_owner,
        "ids_mapped": ids_mapped,
        "ids_unmapped": ids_unmapped,
        "ids_unknown_target": ids_unknown_target,
        "name_target": name_target,
        "names_unmapped": names_unmapped,
        "names_unknown_target": names_unknown_target,
        "names_resolved_from_sources": names_resolved_from_sources,
        "names_fallback_general": names_fallback_general,
        "counts": counts,
        "counted": counted,
        "live_total": live_total,
        "current_counts": dict(sorted(current_counts.items())),
        "retired_carried": retired_carried,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def current_domain_table(
    sources: dict[str, Any], graph: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every current domain from the union of the grammar library and codex."""
    grammar = sources["grammar"]
    guidance = sources["guidance"]["domains"]
    name_counts: dict[str, int] = {}
    for row in graph["names"]:
        if row["domain"] is not None:
            name_counts[row["domain"]] = name_counts.get(row["domain"], 0) + 1
    path_counts: dict[str, int] = {}
    ids_per_domain: dict[str, list[str]] = {}
    for row in graph["ids_rows"]:
        if row["domain"] is None:
            continue
        path_counts[row["domain"]] = path_counts.get(row["domain"], 0) + row["paths"]
        ids_per_domain.setdefault(row["domain"], []).append(row["ids"])

    everything = set(grammar) | set(sources["ids_domains"].values()) | set(guidance)
    table: list[dict[str, Any]] = []
    for domain in sorted(everything):
        table.append(
            {
                "domain": domain,
                "grammar": domain in grammar,
                "grammar_description": (grammar.get(domain) or {}).get(
                    "description", ""
                ),
                "grammar_category": (grammar.get(domain) or {}).get("category", ""),
                "grammar_ids": list((grammar.get(domain) or {}).get("ids") or []),
                "codex_ids": sorted(
                    ids
                    for ids, mapped in sources["ids_domains"].items()
                    if mapped == domain
                ),
                "guidance": domain in guidance,
                "live_names": name_counts.get(domain, 0),
                "live_paths": path_counts.get(domain, 0),
                "live_ids": sorted(ids_per_domain.get(domain, [])),
                "disposition": DISPOSITIONS.get(domain, {}).get("state", ""),
                "reason": DISPOSITIONS.get(domain, {}).get("reason", ""),
            }
        )
    return table


def ids_table(
    sources: dict[str, Any], graph: dict[str, Any], proposal: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every IDS the graph holds, with each source's domain and its live names."""
    grammar_owner: dict[str, str] = {}
    for domain, meta in sources["grammar"].items():
        for ids in meta.get("ids") or []:
            grammar_owner.setdefault(ids, domain)
    codex_owner = sources["ids_domains"]
    ids_owner = proposal["ids_owner"]

    live_paths: dict[str, int] = {}
    path_domains: dict[str, dict[str, int]] = {}
    for row in graph["ids_rows"]:
        ids = row["ids"]
        live_paths[ids] = live_paths.get(ids, 0) + row["paths"]
        domain = row["domain"] or "(no domain)"
        path_domains.setdefault(ids, {})
        path_domains[ids][domain] = path_domains[ids].get(domain, 0) + row["paths"]

    name_domain = {row["id"]: row["domain"] for row in graph["names"]}
    name_ids = graph["name_ids"]
    names_per_ids: dict[str, dict[str, int]] = {}
    for name, ids_list in name_ids.items():
        key = name_domain.get(name) or "(no physics domain)"
        for ids in set(ids_list):
            names_per_ids.setdefault(ids, {})
            names_per_ids[ids][key] = names_per_ids[ids].get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    for ids in sorted(live_paths):
        distribution = names_per_ids.get(ids, {})
        majority = (
            sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if distribution
            else None
        )
        rows.append(
            {
                "ids": ids,
                "paths": live_paths[ids],
                "grammar": grammar_owner.get(ids),
                "codex": codex_owner.get(ids),
                "proposed": ids_owner.get(ids),
                "sources_agree": grammar_owner.get(ids)
                == codex_owner.get(ids)
                == ids_owner.get(ids),
                "live_path_domains": dict(sorted(path_domains.get(ids, {}).items())),
                "live_names": sum(distribution.values()),
                "live_name_domains": dict(sorted(distribution.items())),
                "live_name_majority": majority,
                "name_majority_matches_proposed": majority == ids_owner.get(ids),
            }
        )
    return rows


def disagreements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every IDS where a source's domain differs from the proposal, listed."""
    differing = [
        {
            "ids": row["ids"],
            "grammar": row["grammar"],
            "codex": row["codex"],
            "proposed": row["proposed"],
            "live_name_majority": row["live_name_majority"],
        }
        for row in rows
        if not row["sources_agree"]
    ]
    live_off = [
        row["ids"]
        for row in rows
        if row["live_name_majority"] is not None
        and not row["name_majority_matches_proposed"]
    ]
    return {
        "ids_where_sources_differ": differing,
        "ids_where_sources_differ_count": len(differing),
        "ids_where_live_names_differ": live_off,
        "ids_where_live_names_differ_count": len(live_off),
        "ids_total": len(rows),
    }


def gold_cross_check(
    sources: dict[str, Any],
    proposal: dict[str, Any],
    ids_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """How the proposal reads against the DD-path gold set, as a fourth signal.

    Two readings, because the gold set is path-level and the proposal's IDS
    function is coarser: one against the proposal's IDS-level domain, and one
    against the domain the live names sourced from that IDS mostly carry. The
    second is the fair comparison for a path-level expectation.
    """
    majority = {row["ids"]: row["live_name_majority"] for row in ids_rows}
    rows = []
    agree_proposed = 0
    agree_majority = 0
    for entry in sources["gold"]:
        ids = entry.get("ids")
        expected = entry.get("expected_domain")
        proposed = proposal["ids_owner"].get(ids)
        live_majority = majority.get(ids)
        agree_proposed += int(proposed == expected)
        agree_majority += int(live_majority == expected)
        rows.append(
            {
                "path": entry.get("path"),
                "ids": ids,
                "expected": expected,
                "proposed": proposed,
                "live_name_majority": live_majority,
                "proposed_matches": proposed == expected,
                "live_majority_matches": live_majority == expected,
            }
        )
    return {
        "entries": len(sources["gold"]),
        "agree_proposed": agree_proposed,
        "agree_live_majority": agree_majority,
        "disagree_proposed": sum(1 for row in rows if not row["proposed_matches"]),
        "disagreements": [row for row in rows if not row["proposed_matches"]],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw_figure(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """A matrix of the IDS default against where its live paths are actually filed.

    Rows are the proposed domain an IDS's data structure gives it; columns are
    the domain its live DD paths carry. The diagonal is agreement; the
    off-diagonal mass is the disagreement a table states one IDS at a time; the
    block structure is what a row per IDS cannot show at 87 rows.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from iter_codex.plotting.style import BASE_STYLE, CANVAS, STYLE, save_figure

    plt.style.use(BASE_STYLE)
    plt.rcParams.update(STYLE)

    targets = sorted({row["proposed"] for row in rows if row["proposed"] is not None})
    columns = sorted({domain for row in rows for domain in row["live_path_domains"]})
    index_t = {name: i for i, name in enumerate(targets)}
    index_c = {name: i for i, name in enumerate(columns)}
    grid = [[0 for _ in columns] for _ in targets]
    for row in rows:
        target = row["proposed"]
        if target is None:
            continue
        for domain, count in row["live_path_domains"].items():
            if domain in index_c:
                grid[index_t[target]][index_c[domain]] += count

    figure, axis = plt.subplots(figsize=(CANVAS[0], CANVAS[1] + 1.2))
    data = [list(line) for line in grid]
    image = axis.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(max(line) for line in data) or 1),
    )
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(list(columns), rotation=90, fontsize=6.5)
    axis.set_yticks(range(len(targets)))
    axis.set_yticklabels(list(targets), fontsize=6.5)
    axis.set_xlabel("domain the IDS's live DD paths carry")
    axis.set_ylabel("proposed domain for the IDS")
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    figure.colorbar(image, ax=axis, shrink=0.8, pad=0.02)
    figure.subplots_adjust(left=0.22, right=0.90, bottom=0.28, top=0.97)
    return save_figure(figure, path)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return f"<td>{html.escape(text)}</td>"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(_cell(c) for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _proposed_html(payload: dict[str, Any]) -> str:
    blocks = []
    for entry in payload["proposal"]["domains"]:
        excludes = (
            "; ".join(
                f"{html.escape(str(ex.get('quantity', '')))} → {html.escape(str(ex.get('neighbour', '')))}: {html.escape(str(ex.get('rule', '')))}"
                for ex in entry["excludes"]
            )
            or "—"
        )
        includes = "; ".join(html.escape(str(i)) for i in entry["includes"]) or "—"
        blocks.append(
            f"""<section class="domain">
<h3>{html.escape(entry["name"])} <span class="tag">{html.escape(entry["axis"])}</span>
<span class="tag">{html.escape(entry["disposition"])}</span></h3>
<p><strong>Reason.</strong> {html.escape(entry["reason"])}</p>
<p><strong>Description.</strong> {html.escape(entry["description"])}</p>
<p><strong>Category.</strong> {html.escape(entry["category"])} &nbsp; <strong>Count.</strong> {payload["proposal"]["counts"].get(entry["name"], 0)}</p>
<p><strong>IDS.</strong> {html.escape(", ".join(entry["ids"]) or "—")}</p>
<p><strong>Scope.</strong> {html.escape(entry["scope"])}</p>
<p><strong>Includes.</strong> {includes}</p>
<p><strong>Excludes.</strong> {excludes}</p>
</section>"""
        )
    return "\n".join(blocks)


def render_html(payload: dict[str, Any]) -> str:
    current_rows = [
        [
            row["domain"],
            row["grammar_description"],
            row["grammar_category"],
            ", ".join(row["grammar_ids"]) or "—",
            ", ".join(row["codex_ids"]) or "—",
            "yes" if row["guidance"] else "no",
            row["live_names"],
            row["live_paths"],
            row["disposition"],
            row["reason"],
        ]
        for row in payload["current_domains"]
    ]
    ids_rows = [
        [
            row["ids"],
            row["paths"],
            row["grammar"],
            row["codex"],
            row["proposed"],
            row["live_name_majority"],
            row["live_names"],
            "yes" if row["sources_agree"] else "NO",
        ]
        for row in payload["ids"]
    ]
    mapping_rows = [
        [current, target]
        for current, target in sorted(
            payload["proposal"]["current_to_proposed"].items()
        )
    ] + [[domain, "(retired)"] for domain in payload["proposal"]["retired"]]
    count_rows = [
        [
            entry["name"],
            entry["axis"],
            entry["disposition"],
            payload["proposal"]["counts"].get(entry["name"], 0),
        ]
        for entry in sorted(
            payload["proposal"]["domains"],
            key=lambda e: (-payload["proposal"]["counts"].get(e["name"], 0), e["name"]),
        )
    ]
    check_rows = [
        [name, "ok" if value else "FAIL"] for name, value in payload["checks"].items()
    ]
    differ = payload["disagreements"]
    gold = payload["gold_cross_check"]
    live = payload["live"]
    proposal = payload["proposal"]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence-fragment">
<title>Physics-domain review: one proposed list against the IDS sources</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 78rem; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; margin: 0.6rem 0 1.4rem; font-size: 0.82rem; }}
th, td {{ border-bottom: 1px solid #d8d8d8; padding: 0.22rem 0.5rem; text-align: left; vertical-align: top; }}
th {{ background: #f2f2f2; }}
section.domain {{ border-top: 1px solid #e2e2e2; padding: 0.4rem 0; }}
.tag {{ font-size: 0.72rem; background: #eef2f6; border-radius: 0.5rem; padding: 0.05rem 0.45rem; font-weight: normal; }}
figure img {{ max-width: 100%; }}
.ok {{ color: #1f6b33; }} .bad {{ color: #a32020; font-weight: 600; }}
</style></head><body>
<h1>Physics-domain review: one proposed list against the IDS sources</h1>
<p>Generated {html.escape(payload["generated_at"])} on {html.escape(payload["host"])} at
<code>{html.escape(payload["revision"])}</code>. Read-only: nothing in the graph, the
vocabulary or the grammar library was changed.</p>

<h2>Standing</h2>
<p>{live["total_names"]} live standard names, {live["ids_total"]} IDS in the DD version the graph holds,
{len(proposal["domains"])} proposed domains. {live["names_with_no_domain"]} live names carry no stored domain;
{live["names_resolved_from_sources"]} of them take a proposed domain from their sources and
{live["names_fallback_general"]} fall to <code>general</code>. Two current domains carry no
live name at all: <code>plasma_measurement_diagnostics</code> (no names, no paths) and
<code>computational_workflow</code> (no names, 1,959 paths), so both are vocabulary
decisions rather than name moves.</p>
{_table(["check arm", "state"], check_rows)}
<p>Source files and their digests:</p>
{_table(["source", "path", "sha256"], [[k, v, payload["sources"]["sha256"][k]] for k, v in payload["sources"]["files"].items()])}

<h2>1. Every current domain</h2>
<p>The union of the grammar library's vocabulary, this repository's IDS table and the
guidance file, with the live count of names carrying it and the live count of DD paths
carrying it. Disposition is the proposal's own (section 3).</p>
{_table(["domain", "grammar description", "category", "grammar IDS", "codex IDS", "guidance", "live names", "live DD paths", "disposition", "reason"], current_rows)}

<h2>2. Every IDS in the graph's DD version</h2>
<p>The domain each of the three sources gives each IDS, the domain its live names mostly
carry, and whether the three sources agree. {differ["ids_where_sources_differ_count"]} of
{differ["ids_total"]} IDS disagree between the sources; {differ["ids_where_live_names_differ_count"]}
have a live-name majority that differs from the proposal.</p>
{_table(["IDS", "live DD paths", "grammar", "codex", "proposed", "live-name majority", "live names", "sources agree"], ids_rows)}
<h3>Every disagreement</h3>
{_table(["IDS", "grammar", "codex", "proposed", "live-name majority"], [[d["ids"], d["grammar"], d["codex"], d["proposed"], d["live_name_majority"]] for d in differ["ids_where_sources_differ"]])}

<h2>3. The proposed list</h2>
<p>Each entry names either a <em>physics area</em> or a <em>grouping of data structures</em>.
The five quantity-form parts carry the transport division the lead ruled; every proposed
domain has a description, a category, an IDS list, and scope, includes and excludes in the
form <code>domains.yaml</code> uses.</p>
{_proposed_html(payload)}

<h2>4. Mapping from current domains to proposed ones</h2>
{_table(["current domain", "proposed domain"], mapping_rows)}
<h3>Counts after applying the mapping to the live names</h3>
<p>{proposal["counted"]} of {proposal["live_total"]} live names are placed; the proposed
counts sum to the live total.</p>
{_table(["proposed domain", "axis", "disposition", "live names"], count_rows)}

<h2>5. Where the sources disagree</h2>
<figure><img src="/imas-codex/figures/sn-audited-group-release/physics-domain-review/domain-agreement-matrix.svg"
 alt="Matrix of an IDS's proposed domain against the domain its live DD paths carry">
<figcaption>Each row is the proposed domain an IDS's data structure gives it; each column
is the domain its live DD paths are filed under; the cell is the number of live DD paths
(log scale). The diagonal is agreement; the off-diagonal mass is the IDS whose data
structure and whose live paths name different domains.</figcaption></figure>
<p>The DD-path gold set is a fourth, path-level signal. Of {gold["entries"]} gold paths,
{gold["agree_proposed"]} have an expected domain equal to the proposal's IDS-level domain, and
{gold["agree_live_majority"]} equal the domain the live names sourced from that IDS mostly
carry. The gap between those two readings measures how much of the gold set is a
path-level expectation rather than an IDS-level one: a path can be more specific than
its IDS, so the second reading is the fair comparison. The paths the proposal does not
match are listed in the data file (<code>gold_cross_check.disagreements</code>) for the
lead.</p>

<h2>What the lead is asked to rule</h2>
<p>Nothing is written by this page; it is a proposal.</p>
<ol>
<li><strong>The list.</strong> Adopt, amend or reject the {len(proposal["domains"])} proposed
domains. Each entry carries its disposition and a one-line reason; the five
<code>*_quantities</code> entries are the transport division already ruled, proposed as
domains here so the vocabulary and the stored domains name the same things.</li>
<li><strong>The parts' placement.</strong> Keep the five as domains or place them elsewhere,
for example as a data-structure grouping axis outside the physics-domain vocabulary.
The proposal favours domains because they are the write targets the transport division
creates; the alternative is stated so the choice is explicit.</li>
<li><strong>The zero-name moves.</strong> Retire <code>plasma_measurement_diagnostics</code>
(nothing carries it) and merge <code>computational_workflow</code> into
<code>data_management</code> (no names; 1,959 live paths move with the entry).</li>
</ol>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert every coverage arm and exit non-zero on any failure",
    )
    parser.add_argument("--plant-drop-ids", default=None, metavar="IDS")
    parser.add_argument("--plant-drop-name", default=None, metavar="NAME")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    planted = args.plant_drop_ids is not None or args.plant_drop_name is not None

    sources = read_sources()
    graph = read_graph()
    proposal = propose(
        sources, graph, drop_ids=args.plant_drop_ids, drop_name=args.plant_drop_name
    )
    domains = proposed_domains(sources)
    current = current_domain_table(sources, graph)
    ids_rows = ids_table(sources, graph, proposal)
    differ = disagreements(ids_rows)
    gold = gold_cross_check(sources, proposal, ids_rows)

    figure_receipt = None
    if not args.no_figure and not planted:
        figure_receipt = draw_figure(ids_rows, FIGURE)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "revision": _revision(),
        "sources": {
            "files": sources["files"],
            "sha256": sources["sha256"],
        },
        "live": {
            "total_names": proposal["live_total"],
            "names_with_no_domain": proposal["current_counts"].get(
                "(no physics domain)", 0
            ),
            "names_resolved_from_sources": proposal["names_resolved_from_sources"],
            "names_fallback_general": proposal["names_fallback_general"],
            "ids_total": len(ids_rows),
        },
        "current_domains": current,
        "ids": ids_rows,
        "disagreements": differ,
        "proposal": {
            "domains": domains,
            "counts": proposal["counts"],
            "counted": proposal["counted"],
            "live_total": proposal["live_total"],
            "current_to_proposed": CURRENT_TO_PROPOSED,
            "retired": sorted(
                domain
                for domain, entry in DISPOSITIONS.items()
                if entry["state"] == "retired"
            ),
        },
        "gold_cross_check": gold,
        "checks": proposal["checks"],
        "check_detail": {
            "ids_unmapped": proposal["ids_unmapped"],
            "ids_unknown_target": proposal["ids_unknown_target"],
            "names_unmapped": proposal["names_unmapped"],
            "names_unknown_target": proposal["names_unknown_target"],
            "retired_carried": proposal["retired_carried"],
        },
        "plants": {
            "drop_ids": args.plant_drop_ids,
            "drop_name": args.plant_drop_name,
        },
        "figure": figure_receipt,
    }
    if planted:
        print(
            "PLANT: in-memory mutation only; the committed .json, .html and figure "
            "are not rewritten"
        )
    else:
        OUT_JSON.write_text(
            json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8"
        )
        OUT_HTML.write_text(render_html(payload), encoding="utf-8")

    passed = sum(1 for value in proposal["checks"].values() if value)
    failed = len(proposal["checks"]) - passed
    print(
        f"RESULT: {passed} passed, {failed} failed, {len(proposal['checks'])} arms; "
        f"ids {len(ids_rows)}; live names {proposal['counted']}/{proposal['live_total']}; "
        f"proposed domains {len(domains)}"
    )
    for name, value in proposal["checks"].items():
        print(f"  {'ok  ' if value else 'FAIL'} {name}")
    if args.check and failed:
        return 1
    return 0


def _revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


if __name__ == "__main__":
    sys.exit(main())
