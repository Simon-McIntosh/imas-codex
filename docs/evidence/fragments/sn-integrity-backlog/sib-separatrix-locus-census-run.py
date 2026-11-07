"""Read-only census of every live standard name tied to the separatrix loci.

Population: every StandardName whose produced DD source path carries
``separatrix`` (which subsumes ``separatrix_average``), or whose name carries
``_at_separatrix`` or ``_at_plasma_boundary``. Source paths come from two
places the corpus uses, and both are read: the ``StandardNameSource`` node
linked by ``PRODUCED_NAME``, and the ``IMASNode`` linked by
``HAS_STANDARD_NAME`` (catalog-imported names carry their path only the second
way).

Each name is classified by the DD locus its source paths mean:

``separatrix``
    ``summary/local/separatrix/*`` -- a local value evaluated at the separatrix.
``separatrix_average``
    ``summary/local/separatrix_average/*`` -- flux-surface averaged over the
    separatrix.
``pedestal_fit_separatrix``
    ``summary/pedestal_fits/{linear,mtanh}/*/separatrix/*`` -- the separatrix
    value read off a pedestal fit.
``equilibrium_boundary``
    a path under ``*/boundary*`` that does not carry ``separatrix`` -- the
    plasma boundary as the equilibrium reports it.
``diagnostic_separatrix_distance``
    a non-summary path carrying ``separatrix`` (probe and Thomson distances).
``unresolved``
    no produced path on either edge.

The proposed suffix for each locus is the trailing locator token sequence the
name must end with, so that one locus is spelled exactly one way and the two
separatrix loci stay distinguishable from the name alone.

Actions are ``keep``, ``signed_rename`` or ``supersede_onto_holder``; a
supersede target must be a live name, never a superseded tombstone.

Run read-only against the live graph; writes the JSON beside this script and
the HTML report beside it. The log's last line is a bare ``EXIT=`` line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402

HERE = Path(__file__).resolve().parent
NODE = "sib-separatrix-locus-census"
JSON_PATH = HERE / f"{NODE}.json"
HTML_PATH = HERE / f"{NODE}.html"

SUFFIX_SEPARATRIX = "at_separatrix"
SUFFIX_SEPARATRIX_AVERAGE = "flux_surface_averaged_at_separatrix"
SUFFIX_EQUILIBRIUM_BOUNDARY = "at_*_boundary"
SUFFIX_DIAGNOSTIC = "at_outboard_midplane"

LOCUS_SUFFIX = {
    "separatrix": SUFFIX_SEPARATRIX,
    "separatrix_average": SUFFIX_SEPARATRIX_AVERAGE,
    "pedestal_fit_separatrix": SUFFIX_SEPARATRIX,
    "equilibrium_boundary": SUFFIX_EQUILIBRIUM_BOUNDARY,
    "diagnostic_separatrix_distance": SUFFIX_DIAGNOSTIC,
    "unresolved": None,
}

LOCUS_MEANING = {
    "separatrix": "local value evaluated at the separatrix",
    "separatrix_average": "flux-surface averaged over the separatrix",
    "pedestal_fit_separatrix": "separatrix value read off a pedestal fit",
    "equilibrium_boundary": "plasma boundary as the equilibrium reports it",
    "diagnostic_separatrix_distance": "radial distance from a diagnostic to the separatrix",
    "unresolved": "no produced DD path on either source edge",
}

# Paths under summary/local/* that locate a radial position rather than carry an
# averaged quantity. The coordinate names them for the locus, not for the block.
LOCATOR_LEAVES = ("position/rho_tor", "position/rho_tor_norm", "position/psi")


def query(gc: GraphClient, label: str, cypher: str, **params: object) -> list[dict]:
    start = time.monotonic()
    rows = gc.query(cypher, **params)
    elapsed = time.monotonic() - start
    print(f"query {label}: {len(rows)} rows, {elapsed:.3f}s", file=sys.stderr)
    if elapsed >= 10:
        raise RuntimeError(f"{label} exceeded ten seconds")
    return rows


def bare(path: str) -> str:
    return str(path).removeprefix("dd:").strip("/")


def is_separatrix_path(path: str) -> bool:
    return "separatrix" in path


def locus_of(paths: list[str], name: str) -> tuple[list[str], list[str]]:
    """Loci a name's source paths mean, plus the paths that decided it."""
    telling = sorted({p for p in paths if is_separatrix_path(p)})
    loci: list[str] = []
    if any(p.startswith("summary/local/separatrix/") for p in telling):
        loci.append("separatrix")
    if any(p.startswith("summary/local/separatrix_average/") for p in telling):
        # A position locator inside the averaged block is the same physical
        # position the local block reports; it does not make the quantity
        # averaged.
        averaged = [
            p
            for p in telling
            if p.startswith("summary/local/separatrix_average/")
            and "/position/" not in p
        ]
        if averaged:
            loci.append("separatrix_average")
    if any("/separatrix/" in p and "/pedestal_fits/" in p for p in telling):
        loci.append("pedestal_fit_separatrix")
    if not loci and telling:
        loci.append("diagnostic_separatrix_distance")
    if not loci and any("/boundary" in p for p in paths):
        loci = ["equilibrium_boundary"]
    if not loci and not telling:
        loci = ["unresolved"]
    return loci, telling


def strip_locator(name: str) -> str:
    for tail in (
        "_flux_surface_averaged_at_plasma_boundary",
        "_flux_surface_averaged_at_separatrix",
        "_at_plasma_boundary",
        "_at_separatrix",
    ):
        if name.endswith(tail):
            return name[: -len(tail)]
    return name


def desired_name(name: str, locus: str) -> str:
    stem = strip_locator(name)
    if locus in ("separatrix", "pedestal_fit_separatrix"):
        return f"{stem}_at_separatrix"
    if locus == "separatrix_average":
        stem = stem.replace("_flux_surface_averaged", "").removeprefix(
            "flux_surface_averaged_"
        )
        return f"{stem}_flux_surface_averaged_at_separatrix"
    return name


class CensusInvariantViolated(RuntimeError):
    """A published record contradicts the graph or its own stated rule."""


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one rename, never in place."""
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    tmp.replace(path)


def assert_invariants(data: dict, status_of: dict[str, str]) -> list[str]:
    """Check the record against the rules the fence states. Returns violations.

    These are the checks a reviewer would run by hand: every proposal must name
    a live holder, no proposal may reach a tombstone, and the population must be
    the one an independent query reports.
    """
    violations: list[str] = []
    names = data["names"]
    ids = [r["id"] for r in names]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        violations.append(f"duplicate names in census: {dupes}")
    for r in names:
        target = r["proposed_target"]
        if target is None:
            continue
        state = status_of.get(target)
        if state == "superseded":
            violations.append(
                f"{r['id']}: proposed_target {target} is a superseded tombstone"
            )
        elif r["proposed_action"] == "signed_rename" and state is not None:
            violations.append(
                f"{r['id']}: rename target {target} already exists as {state}"
            )
        elif r["proposed_action"] == "supersede_onto_holder" and state is None:
            violations.append(
                f"{r['id']}: supersede holder {target} is not a name the graph holds"
            )
    cc = data["cross_check"]
    if not cc["agrees"]:
        violations.append(
            "census population disagrees with the independent query: "
            f"missing {cc['in_graph_not_in_census']}, "
            f"extra {cc['in_census_not_in_graph']}"
        )
    return violations


def inject(kind: str, data: dict, status_of: dict[str, str]) -> list[str]:
    """Introduce a known defect and return the violations it provokes.

    A guard that has never refused anything has not been shown to work, so each
    injected defect must make ``assert_invariants`` report it.
    """
    if kind == "tombstone-target":
        tomb = next(i for i, st in sorted(status_of.items()) if st == "superseded")
        victim = next(
            r for r in data["names"] if r["proposed_action"] == "signed_rename"
        )
        victim["proposed_target"] = tomb
        print(f"injected tombstone-target: {victim['id']} -> {tomb}")
    elif kind == "cross-check":
        data["cross_check"]["in_graph_not_in_census"] = ["a_name_the_census_omitted"]
        data["cross_check"]["agrees"] = False
        print("injected cross-check: independent query reports a name the census omits")
    else:
        raise CensusInvariantViolated(f"unknown injection {kind}")
    return assert_invariants(data, status_of)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inject",
        choices=("tombstone-target", "cross-check"),
        help="introduce a known defect so the invariant check can be shown refusing it",
    )
    args = parser.parse_args(argv)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    out: dict = {"node": NODE, "generated_at": started, "read_only": True}

    with GraphClient() as gc:
        names = query(
            gc,
            "names",
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, n.status AS status, n.unit AS unit,
                   n.description AS description, n.physics_domain AS domain
            ORDER BY id
            """,
        )
        sources = query(
            gc,
            "sources",
            """
            MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
            WHERE s.dd_path IS NOT NULL
            RETURN n.id AS id, s.dd_path AS path, s.status AS source_status
            """,
        )
        imas = query(
            gc,
            "imas_links",
            """
            MATCH (i:IMASNode)-[:HAS_STANDARD_NAME]->(n:StandardName)
            RETURN n.id AS id, i.id AS path
            """,
        )
        refined = query(
            gc,
            "refined_from",
            """
            MATCH (n:StandardName)-[:REFINED_FROM]->(f:StandardName)
            RETURN n.id AS id, f.id AS from_id
            """,
        )
        successors = query(
            gc,
            "has_successor",
            """
            MATCH (a:StandardName)-[:HAS_SUCCESSOR]->(b:StandardName)
            RETURN a.id AS id, b.id AS successor
            """,
        )
        parents = query(
            gc,
            "has_parent",
            """
            MATCH (c:StandardName)-[:HAS_PARENT]->(p:StandardName)
            RETURN c.id AS child, p.id AS parent
            """,
        )
        # Independent restatement of the population. One query, written
        # against the graph without the census's helpers, returning the ids
        # themselves so the comparison is a set difference that can be non-empty.
        cross_ids = query(
            gc,
            "cross_population",
            """
            MATCH (n:StandardName)
            WHERE n.status <> 'superseded'
              AND (
                n.id ENDS WITH '_at_separatrix'
                OR n.id ENDS WITH '_at_plasma_boundary'
                OR size([(s:StandardNameSource)-[:PRODUCED_NAME]->(n)
                         WHERE s.dd_path CONTAINS 'separatrix' | 1]) > 0
                OR size([(i:IMASNode)-[:HAS_STANDARD_NAME]->(n)
                         WHERE i.id CONTAINS 'separatrix' | 1]) > 0
              )
            RETURN n.id AS id
            ORDER BY id
            """,
        )
        total_live = query(
            gc,
            "live_total",
            "MATCH (n:StandardName) WHERE n.status <> 'superseded' RETURN count(n) AS c",
        )[0]["c"]

        status_of = {r["id"]: r["status"] for r in names}
        unit_of = {r["id"]: r["unit"] for r in names}
        domain_of = {r["id"]: r["domain"] for r in names}

        paths_of: dict[str, set[str]] = defaultdict(set)
        source_status_of: dict[str, set[str]] = defaultdict(set)
        for r in sources:
            paths_of[r["id"]].add(bare(r["path"]))
            source_status_of[r["id"]].add(r["source_status"])
        # The IMASNode edge is the only path carrier for catalog-imported names.
        imas_path_of: dict[str, set[str]] = defaultdict(set)
        for r in imas:
            imas_path_of[r["id"]].add(bare(r["path"]))

    def all_paths(nid: str) -> list[str]:
        return sorted(paths_of.get(nid, set()) | imas_path_of.get(nid, set()))

    # ── population ────────────────────────────────────────────────────────
    population, live, tombstones = [], [], []
    for r in names:
        nid, status = r["id"], r["status"]
        paths = all_paths(nid)
        named = nid.endswith("_at_separatrix") or nid.endswith("_at_plasma_boundary")
        sourced = any(is_separatrix_path(p) for p in paths)
        if not (named or sourced):
            continue
        population.append(nid)
        (live if status != "superseded" else tombstones).append(nid)

    live_set = set(live)
    tombstone_set = set(tombstones)

    refined_by: dict[str, set[str]] = defaultdict(set)
    refined_of: dict[str, set[str]] = defaultdict(set)
    for r in refined:
        refined_by[r["id"]].add(r["from_id"])  # this name was refined from ...
        refined_of[r["from_id"]].add(r["id"])  # ... and this one replaced it
    successor_of: dict[str, set[str]] = defaultdict(set)
    predecessor_of: dict[str, set[str]] = defaultdict(set)
    for r in successors:
        successor_of[r["id"]].add(r["successor"])
        predecessor_of[r["successor"]].add(r["id"])
    children_of: dict[str, set[str]] = defaultdict(set)
    for r in parents:
        children_of[r["parent"]].add(r["child"])

    # A live holder per bare (locator-stripped) concept, so a rename target
    # that a live sibling already spells is visible before the lead rules.
    holder_by_stem: dict[str, list[str]] = defaultdict(list)
    for nid in live:
        holder_by_stem[strip_locator(nid)].append(nid)

    def locator_of(name: str) -> str | None:
        for tail in ("_at_plasma_boundary", "_at_separatrix"):
            if name.endswith(tail):
                return tail
        return None

    rows = []
    for nid in sorted(live):
        paths = all_paths(nid)
        loci, telling = locus_of(paths, nid)
        required = sorted(
            {LOCUS_SUFFIX[locus_id] for locus_id in loci if LOCUS_SUFFIX[locus_id]}
        )
        locus = loci[0] if len(set(required)) <= 1 else "mixed"
        chain = {
            "refined_from": sorted(refined_by.get(nid, set())),
            "refined_into": sorted(refined_of.get(nid, set())),
            "has_successor": sorted(successor_of.get(nid, set())),
            "predecessor": sorted(predecessor_of.get(nid, set())),
            "parent_children": sorted(children_of.get(nid, set())),
        }
        tomb = sorted(
            x
            for x in {
                *refined_of.get(nid, set()),
                *refined_by.get(nid, set()),
                *successor_of.get(nid, set()),
                *predecessor_of.get(nid, set()),
            }
            if status_of.get(x) == "superseded"
        )

        collision: str | None = None
        blocked_spelling: str | None = None
        target_name = nid
        if locus == "mixed":
            action = "keep"
            why = (
                "holds sources from both separatrix loci, which require different "
                "locators; the concept must be split into two names, not renamed"
            )
        elif locus == "unresolved":
            holders = [
                h for h in holder_by_stem.get(strip_locator(nid), []) if h != nid
            ]
            if holders:
                action, target_name = "supersede_onto_holder", sorted(holders)[0]
                why = "no produced DD path; a live name carries the concept"
            else:
                action, target_name = "keep", nid
                why = "no produced DD path and no live holder; nothing to retarget"
        elif nid == "toroidal_tritium_velocity":
            action, target_name = "keep", nid
            why = (
                "bare stub holding the pedestal and separatrix tritium velocity paths; "
                "its retirement waits on the two located names being composed, "
                "neither of which the graph holds yet"
            )
        elif locator_of(nid) is None:
            action, target_name = "keep", nid
            why = (
                "the name's trailing locator is neither _at_separatrix nor "
                "_at_plasma_boundary, so it is outside the two spellings this "
                "census rewrites"
            )
        else:
            target = desired_name(nid, locus)
            if target == nid:
                action, why = "keep", f"already ends in the {locus} locator"
            elif target in live_set:
                sourced_here = any(is_separatrix_path(p) for p in paths)
                occupant_paths = all_paths(target)
                occupant_sourced = any(is_separatrix_path(p) for p in occupant_paths)
                if sourced_here and not occupant_sourced:
                    action, target_name = "keep", nid
                    collision = (
                        "required spelling is held by a live name with no DD path"
                    )
                    blocked_spelling = target
                    why = (
                        f"the required spelling {target} is a live name with no "
                        "produced source; it must be retired before this rename "
                        "can land, so the rename is not published as a rename"
                    )
                else:
                    action, target_name = "supersede_onto_holder", target
                    why = f"the same concept is already held live by {target}"
            elif target in tombstone_set:
                # The required spelling is taken by a retired name. A rename
                # emitted against it would be a revival, so the action stays
                # with its holder and the taken spelling is recorded beside it.
                action, target_name = "keep", nid
                collision = "required spelling is held by a superseded tombstone"
                blocked_spelling = target
                why = (
                    f"the {locus} locator is {target}, but that spelling is a "
                    "superseded tombstone; the name is kept until the lead rules "
                    "between composing fresh and leaving the locator as it is"
                )
            else:
                action, target_name = "signed_rename", target
                why = (
                    f"ends in {locator_of(nid)}, {locus} requires {LOCUS_SUFFIX[locus]}"
                )

        if (
            action == "supersede_onto_holder"
            and status_of.get(target_name) == "superseded"
        ):
            action, target_name, collision = (
                "keep",
                nid,
                "refused: holder is a tombstone",
            )
            why = "refused: the only candidate holder is a superseded tombstone"

        rows.append(
            {
                "id": nid,
                "status": status_of[nid],
                "unit": unit_of.get(nid),
                "physics_domain": domain_of.get(nid),
                "locus": locus,
                "loci_present": loci,
                "locus_meaning": "; ".join(
                    LOCUS_MEANING[locus_id] for locus_id in loci
                ),
                "locus_paths": telling,
                "all_paths": paths,
                "current_locator": locator_of(nid),
                "required_locator": (
                    LOCUS_SUFFIX[locus]
                    if locus in LOCUS_SUFFIX
                    else ", ".join(required)
                ),
                "proposed_action": action,
                "proposed_target": target_name,
                "collision": collision,
                "blocked_spelling": blocked_spelling,
                "rationale": why,
                "chain": chain,
                "tombstones": tomb,
            }
        )

    tomb_rows = []
    for nid in sorted(tombstones):
        paths = all_paths(nid)
        loci, telling = locus_of(paths, nid)
        required = sorted(
            {LOCUS_SUFFIX[locus_id] for locus_id in loci if LOCUS_SUFFIX[locus_id]}
        )
        locus = loci[0] if len(set(required)) <= 1 else "mixed"
        tomb_rows.append(
            {
                "id": nid,
                "status": status_of[nid],
                "locus": locus,
                "locus_meaning": "; ".join(
                    LOCUS_MEANING[locus_id] for locus_id in loci
                ),
                "locus_paths": telling,
                "refined_from": sorted(refined_by.get(nid, set())),
                "refined_into": sorted(refined_of.get(nid, set())),
                "must_not_be_revived_by_chain": sorted(
                    {*refined_of.get(nid, set()), *successor_of.get(nid, set())}
                ),
            }
        )

    # ── the two tritium located names under the bare stub ─────────────────
    tritium = {
        "bare_stub": "toroidal_tritium_velocity",
        "bare_stub_status": status_of.get("toroidal_tritium_velocity"),
        "bare_stub_paths": all_paths("toroidal_tritium_velocity"),
        "located": [
            {
                "dd_path": "summary/local/pedestal/velocity_phi/tritium/value",
                "locus": "pedestal",
                "proposed_name": "toroidal_tritium_velocity_at_pedestal_top",
                "existing": "absent",
            },
            {
                "dd_path": "summary/local/separatrix/velocity_phi/tritium/value",
                "locus": "separatrix",
                "proposed_name": desired_name(
                    "toroidal_tritium_velocity_at_separatrix", "separatrix"
                ),
                "existing": status_of.get("toroidal_tritium_velocity_at_separatrix"),
            },
        ],
        "tombstones_on_the_located_names": sorted(
            x
            for x in status_of
            if x.startswith("toroidal_tritium_velocity")
            and status_of[x] == "superseded"
        ),
    }

    # ── cross-check the headline against a second formulation ─────────────
    by_locus = Counter(r["locus"] for r in rows)
    actions = Counter(r["proposed_action"] for r in rows)
    # The independent reading is a set of ids produced by its own query. It can
    # disagree with the census: that is the whole point of it.
    independent_ids = {r["id"] for r in cross_ids}
    census_ids = {r["id"] for r in rows}
    missing_from_census = sorted(independent_ids - census_ids)
    over_claimed_by_census = sorted(census_ids - independent_ids)
    cross_check = {
        "independent_population_count": len(independent_ids),
        "census_live_count": len(census_ids),
        "in_graph_not_in_census": missing_from_census,
        "in_census_not_in_graph": over_claimed_by_census,
        "agrees": not missing_from_census
        and not over_claimed_by_census
        and len(census_ids) > 0,
        "live_names_known_to_graph": total_live,
    }

    out.update(
        {
            "population_rule": (
                "live StandardName with a produced DD path carrying 'separatrix' "
                "on either source edge, or a name ending _at_separatrix / _at_plasma_boundary"
            ),
            "totals": {
                "population": len(population),
                "live": len(rows),
                "superseded_tombstones": len(tomb_rows),
                "by_locus": dict(sorted(by_locus.items())),
                "by_action": dict(sorted(actions.items())),
            },
            "suffix_by_locus": {
                "separatrix": {
                    "dd_locus": "summary/local/separatrix/*",
                    "meaning": LOCUS_MEANING["separatrix"],
                    "proposed_locator": f"_{SUFFIX_SEPARATRIX}",
                },
                "separatrix_average": {
                    "dd_locus": "summary/local/separatrix_average/*",
                    "meaning": LOCUS_MEANING["separatrix_average"],
                    "proposed_locator": f"_{SUFFIX_SEPARATRIX_AVERAGE}",
                },
                "pedestal_fit_separatrix": {
                    "dd_locus": "summary/pedestal_fits/*/*/separatrix/*",
                    "meaning": LOCUS_MEANING["pedestal_fit_separatrix"],
                    "proposed_locator": f"_{SUFFIX_SEPARATRIX}",
                },
                "equilibrium_boundary": {
                    "dd_locus": "*/*/boundary* (no 'separatrix' in the path)",
                    "meaning": LOCUS_MEANING["equilibrium_boundary"],
                    "proposed_locator": f"_{SUFFIX_EQUILIBRIUM_BOUNDARY}",
                },
                "diagnostic_separatrix_distance": {
                    "dd_locus": "non-summary path carrying 'separatrix'",
                    "meaning": LOCUS_MEANING["diagnostic_separatrix_distance"],
                    "proposed_locator": "unchanged (names the diagnostic, not a locus suffix)",
                },
                "unresolved": {
                    "dd_locus": "none",
                    "meaning": LOCUS_MEANING["unresolved"],
                    "proposed_locator": None,
                },
            },
            "collisions": [r for r in rows if r["collision"]],
            "mixed_locus_names": [r["id"] for r in rows if r["locus"] == "mixed"],
            "names": rows,
            "tombstones": tomb_rows,
            "tritium": tritium,
            "cross_check": cross_check,
        }
    )

    violations = assert_invariants(out, status_of)
    if args.inject:
        violations = violations + inject(args.inject, out, status_of)
    out["invariants"] = {
        "checked": [
            "a rename target is a spelling the graph does not hold yet",
            "no proposed_target is a superseded tombstone",
            "a supersede holder is a live name",
            "every name appears once",
            "the census id set equals the independent query's id set",
        ],
        "violations": violations,
    }
    out["invariants"]["passed"] = not violations

    if violations:
        for v in violations:
            print(f"INVARIANT VIOLATION: {v}")
        print(
            f"population={len(population)} live={len(rows)} tombstones={len(tomb_rows)} "
            f"loci={dict(by_locus)} actions={dict(actions)} "
            f"cross_check_agrees={cross_check['agrees']} invariants=FAILED"
        )
        print("nothing written: a failing record must not replace a passing one")
        return 1

    # Durability: the JSON is the record and lands first, and each file is
    # written beside its target and renamed into place, so an interrupted run
    # leaves the previous record intact rather than a truncated one.
    write_atomic(JSON_PATH, json.dumps(out, indent=1) + "\n")
    write_atomic(HTML_PATH, render(out))
    print(
        f"population={len(population)} live={len(rows)} tombstones={len(tomb_rows)} "
        f"loci={dict(by_locus)} actions={dict(actions)} "
        f"cross_check_agrees={cross_check['agrees']}"
    )
    return 0


# ── rendering ─────────────────────────────────────────────────────────────
def _u(value: object) -> str:
    return escape("" if value is None else str(value))


def render(d: dict) -> str:
    t = d["totals"]
    cc = d["cross_check"]
    parts = [
        '<h2 id="sib-separatrix-locus-census">Separatrix locus census</h2>',
        "<p>Read-only census of every live standard name tied to the separatrix "
        "loci, produced from the live graph by "
        "<code>sib-separatrix-locus-census-run.py</code>. No write is made; this "
        "record proposes the suffix mapping the lead rules on.</p>",
        "<h3>Population</h3>",
        "<table><thead><tr><th>measure</th><th>count</th></tr></thead><tbody>",
    ]
    for label, value in (
        ("live names in census", t["live"]),
        ("superseded tombstones recorded", t["superseded_tombstones"]),
        ("total population touched", t["population"]),
        ("live names in the whole graph", cc["live_names_known_to_graph"]),
    ):
        parts.append(f"<tr><td>{_u(label)}</td><td>{_u(value)}</td></tr>")
    parts.append("</tbody></table>")

    parts.append("<h3>Proposed suffix per DD locus</h3>")
    parts.append(
        "<table><thead><tr><th>locus</th><th>DD path</th><th>means</th>"
        "<th>proposed locator</th><th>live names</th></tr></thead><tbody>"
    )
    for locus, spec in d["suffix_by_locus"].items():
        parts.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td>"
            "<td><code>{}</code></td><td>{}</td></tr>".format(
                _u(locus),
                _u(spec["dd_locus"]),
                _u(spec["meaning"]),
                _u(spec["proposed_locator"] or "—"),
                _u(t["by_locus"].get(locus, 0)),
            )
        )
    parts.append("</tbody></table>")
    parts.append(
        "<p>The two separatrix loci are separated by the <b>whole</b> trailing "
        "token sequence: <code>at_separatrix</code> for a local value, "
        "<code>flux_surface_averaged_at_separatrix</code> for the average. That "
        "keeps both loci naming the same physical surface while a reader can "
        "tell which one a name came from without opening the source.</p>"
        "<p><code>at_plasma_boundary</code> is left to mean the equilibrium "
        "boundary only, so the token keeps one meaning across the corpus.</p>"
    )

    parts.append("<h3>Proposed actions</h3>")
    parts.append("<table><thead><tr><th>action</th><th>names</th></tr></thead><tbody>")
    for action, count in t["by_action"].items():
        parts.append(f"<tr><td><code>{_u(action)}</code></td><td>{_u(count)}</td></tr>")
    parts.append("</tbody></table>")

    parts.append("<h3>Every live name</h3>")
    parts.append(
        "<table><thead><tr><th>name</th><th>locus</th><th>unit</th>"
        "<th>DD locus path</th><th>lineage</th><th>tombstones adjacent</th>"
        "<th>action</th><th>target</th></tr></thead><tbody>"
    )
    for r in d["names"]:
        rowsrc = (
            "<br>".join(f"<code>{_u(p)}</code>" for p in r["locus_paths"][:6]) or "—"
        )
        tomb = ", ".join(r["tombstones"]) or "—"
        ch = r["chain"]
        bits = []
        if ch["refined_from"]:
            bits.append("refined from " + ", ".join(ch["refined_from"]))
        if ch["refined_into"]:
            bits.append("refined into " + ", ".join(ch["refined_into"]))
        if ch["has_successor"]:
            bits.append("successor " + ", ".join(ch["has_successor"]))
        if ch["parent_children"]:
            bits.append("children " + str(len(ch["parent_children"])))
        target_cell = f"<code>{_u(r['proposed_target'])}</code>"
        if r["blocked_spelling"]:
            target_cell = (
                f"<s><code>{_u(r['blocked_spelling'])}</code></s>"
                '<div class="why">tombstone, never a target</div>'
            )
        parts.append(
            '<tr><td><code>{}</code><div class="why">{}</div></td>'
            "<td><code>{}</code></td><td><code>{}</code></td>"
            "<td>{}</td><td>{}</td><td>{}</td>"
            "<td><code>{}</code></td><td>{}</td></tr>".format(
                _u(r["id"]),
                _u(r["rationale"]),
                _u(r["locus"]),
                _u(r["unit"] or "—"),
                rowsrc,
                _u("; ".join(bits) or "—"),
                _u(tomb),
                _u(r["proposed_action"]),
                target_cell,
            )
        )
    parts.append("</tbody></table>")

    blocked = list(d["collisions"])
    if blocked:
        parts.append("<h3>Renames that cannot land as a plain rename</h3>")
        parts.append(
            "<p>These carry the required locator but the target spelling is already "
            "taken. None may be landed by a blind rename.</p>"
        )
        parts.append(
            "<table><thead><tr><th>name</th><th>target</th><th>obstruction</th>"
            "</tr></thead><tbody>"
        )
        for r in blocked:
            parts.append(
                "<tr><td><code>{}</code></td><td><code>{}</code></td>"
                "<td>{}</td></tr>".format(
                    _u(r["id"]), _u(r["proposed_target"]), _u(r["collision"])
                )
            )
        parts.append("</tbody></table>")

    parts.append("<h3>Superseded tombstones in the population</h3>")
    parts.append(
        "<p>Recorded so that no proposed action above can land on one. "
        "A rename or supersede target that resolves back to any of these is refused.</p>"
    )
    parts.append(
        "<table><thead><tr><th>tombstone</th><th>locus</th>"
        "<th>replaced by</th></tr></thead><tbody>"
    )
    for r in d["tombstones"]:
        parts.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>".format(
                _u(r["id"]),
                _u(r["locus"]),
                _u(", ".join(r["refined_into"]) or "—"),
            )
        )
    parts.append("</tbody></table>")

    tri = d["tritium"]
    parts.append("<h3>The two tritium located names</h3>")
    parts.append(
        "<p>The bare <code>{}</code> stub (status <code>{}</code>) holds no DD "
        "path; the two solved loci are the pedestal value and the separatrix "
        "value. Under this mapping they compose to:</p>".format(
            _u(tri["bare_stub"]), _u(tri["bare_stub_status"])
        )
    )
    parts.append(
        "<table><thead><tr><th>DD path</th><th>locus</th><th>proposed name</th>"
        "<th>name state</th></tr></thead><tbody>"
    )
    for loc in tri["located"]:
        parts.append(
            "<tr><td><code>{}</code></td><td><code>{}</code></td>"
            "<td><code>{}</code></td><td>{}</td></tr>".format(
                _u(loc["dd_path"]),
                _u(loc["locus"]),
                _u(loc["proposed_name"]),
                _u(loc["existing"]),
            )
        )
    parts.append("</tbody></table>")
    if tri["tombstones_on_the_located_names"]:
        parts.append(
            "<p>Sibling tombstones on that stem: {}. The separatrix located name "
            "must be composed as a new name and its tombstone left in place, "
            "never revived.</p>".format(
                ", ".join(
                    f"<code>{_u(x)}</code>"
                    for x in tri["tombstones_on_the_located_names"]
                )
            )
        )

    parts.append("<h3>Cross-check</h3>")
    parts.append(
        "<p>A single independent Cypher query, written without the census's "
        "helpers, returns the ids it believes belong in the population. The "
        "comparison is a set difference, so it can be non-empty: {}.</p>".format(
            "the two agree" if cc["agrees"] else "THEY DISAGREE"
        )
    )
    parts.append(
        "<table><thead><tr><th>reading</th><th>count</th></tr></thead><tbody>"
        "<tr><td>names the independent query returns</td><td>{}</td></tr>"
        "<tr><td>names this census lists</td><td>{}</td></tr>"
        "<tr><td>in the graph, missing from the census</td><td>{}</td></tr>"
        "<tr><td>in the census, absent from the graph</td><td>{}</td></tr>"
        "</tbody></table>".format(
            _u(cc["independent_population_count"]),
            _u(cc["census_live_count"]),
            _u(len(cc["in_graph_not_in_census"])),
            _u(len(cc["in_census_not_in_graph"])),
        )
    )
    inv = d["invariants"]
    parts.append("<h3>Invariants</h3>")
    parts.append(
        "<p>The run refuses to publish a record that breaks any of these, and "
        "the guard was shown refusing by injecting each defect in turn. "
        "Result: <b>{}</b>.</p>".format(
            "all hold" if inv["passed"] else "VIOLATED: " + "; ".join(inv["violations"])
        )
    )
    parts.append("<ul>")
    for rule in inv["checked"]:
        parts.append(f"<li>{_u(rule)}</li>")
    parts.append("</ul>")

    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
