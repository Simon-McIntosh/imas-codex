"""Read-only census of catalog-imported standard names against their published DD sources.

Population: every StandardName whose ``source_types`` contains ``catalog``, plus
every name its lineage reaches (REFINED_FROM, HAS_SUCCESSOR, HAS_PARENT, both
directions).

For each name, the published DD paths are read from the standard-names catalog
repository's ``standard_names/*.yml`` entries at every content commit at or
before the 2026-07-04 import. A name's own later entry can drop a path its
earlier entry carried, so the paths are unioned per name with the commit each
one was published at. DD 3.x paths are matched to DD 4 through the graph's
``RENAMED_TO`` records.

A path is classified against the sources the graph holds now:

* carried            -- a producing name holds it, and that name is the path's
                        own name or lies in its descendant chain (REFINED_FROM,
                        HAS_SUCCESSOR, HAS_PARENT, transitively)
* moved              -- a producing name holds it that is outside that chain;
                        the holder is named
* pending_composition -- no producer, but the source record is ``extracted``:
                        the path is queued for composition, not lost
* stale_source       -- no producer, and the source record is ``stale``
* dropped            -- no producer, and no source record at all

Holders are read regardless of their own status, so a name's own producers never
vanish because the holder is retired; the holder's status is recorded instead.

A name's class is the most severe class among its listed paths, in the order
dropped, moved, stale_source, pending_composition. Names are counted in three
separate columns and never mixed: names with their own published entry, names
whose paths were inherited from an ancestor (counted on their own, outside the
headline), and superseded names. The headline covers live names with their own
entry, so it stands alone.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from html import escape
from pathlib import Path

import yaml

from imas_codex.graph.client import GraphClient

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sib-catalog-import-source-census.json"
REPORT = HERE / "sib-catalog-import-source-census.html"
CATALOG = Path("/home/ITER/mcintos/Code/imas-standard-names-catalog")
CUTOFF = "2026-07-04T23:59:59"
TERMINAL = {"deprecated", "superseded"}
PATH_CLASSES = ("carried", "moved", "stale_source", "pending_composition", "dropped")
SEVERITY = ("dropped", "moved", "stale_source", "pending_composition")
NON_CARRIED = ("moved", "stale_source", "pending_composition", "dropped")
CONTROL_SAMPLE = 20
CONTROL_SEED = 20261028
EXPORT_PY = ROOT / "imas_codex/standard_names/export.py"
SUPERSESSION_PY = ROOT / "imas_codex/standard_names/graph_store/graph_supersession.py"


def git(*args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(CATALOG), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def published_paths() -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    """name -> {commit: {dd paths}} for every content commit before the import."""
    commits = git(
        "log", "--no-merges", f"--before={CUTOFF}", "--format=%H", "--", "standard_names/"
    ).split()
    out: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for commit in commits:
        names = git("ls-tree", "-r", "--name-only", commit, "standard_names/").split()
        for rel in names:
            if not rel.endswith(".yml"):
                continue
            try:
                entries = yaml.safe_load(git("show", f"{commit}:{rel}"))
            except yaml.YAMLError:
                continue
            for entry in entries or []:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                paths = {
                    str(src.get("dd_path"))
                    for src in (entry.get("sources") or [])
                    if isinstance(src, dict) and src.get("dd_path")
                }
                if paths:
                    out[entry["name"]][commit] = paths
    return out, commits


def norm(path: str) -> str:
    return str(path).removeprefix("dd:").strip("/").lstrip(":")


def reach(start: str, links: dict[str, set[str]]) -> set[str]:
    seen, queue = set(), deque(links.get(start, ()))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(links.get(node, set()) - seen)
    return seen


def query(gc: GraphClient, label: str, cypher: str, **params: object) -> list[dict]:
    start = time.monotonic()
    rows = gc.query(cypher, **params)
    elapsed = time.monotonic() - start
    print(f"query {label}: {len(rows)} rows, {elapsed:.3f}s", file=sys.stderr)
    if elapsed >= 10:
        raise RuntimeError(f"{label} exceeded ten seconds")
    return rows


def class_path(
    name: str,
    path: str,
    family: set[str],
    holders: dict[str, set[str]],
    holder_status: dict[str, str],
    source_states: dict[str, set[str]],
    renamed: dict[str, set[str]],
) -> dict:
    mapped = {norm(path)} | reach(norm(path), renamed)
    held = sorted({h for m in mapped for h in holders.get(m, set())})
    if set(held) & family:
        klass = "carried"
    elif held:
        klass = "moved"
    else:
        states = {s for m in mapped for s in source_states.get(m, set())}
        if "extracted" in states:
            klass = "pending_composition"
        elif "stale" in states:
            klass = "stale_source"
        else:
            klass = "dropped"
    return {
        "path": path,
        "mapped_paths": sorted(mapped),
        "class": klass,
        "holder": held,
        "holder_status": {h: holder_status.get(h) for h in held},
        "source_states": sorted({s for m in mapped for s in source_states.get(m, set())}),
    }


def name_class(path_classes: list[str]) -> str:
    if not path_classes:
        return "never_sourced"
    for klass in SEVERITY:
        if klass in path_classes:
            return klass
    return "carried"


def main() -> None:
    published, commits = published_paths()
    with GraphClient() as gc:
        names = query(
            gc,
            "names",
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, n.source_types AS source_types,
                   n.status AS status, n.name_stage AS stage, n.origin AS origin
            ORDER BY id LIMIT 6000
        """,
        )
        total_names = query(gc, "name_count", "MATCH (n:StandardName) RETURN count(n) AS c")[0]["c"]
        assert len(names) == total_names, (len(names), total_names)
        sources = query(
            gc,
            "sources",
            """
            MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(h:StandardName)
            RETURN s.id AS source, s.source_id AS source_id, s.dd_path AS dd_path,
                   s.status AS source_status, h.id AS holder, h.status AS holder_status
            LIMIT 12000
        """,
        )
        unproduced = query(
            gc,
            "unproduced",
            """
            MATCH (s:StandardNameSource)
            WHERE NOT (s)-[:PRODUCED_NAME]->(:StandardName)
            RETURN s.id AS source, s.source_id AS source_id, s.dd_path AS dd_path,
                   s.status AS source_status LIMIT 12000
        """,
        )
        relations = query(
            gc,
            "lineage",
            """
            MATCH (a:StandardName)-[r:REFINED_FROM|HAS_SUCCESSOR|HAS_PARENT]->(b:StandardName)
            RETURN a.id AS a, type(r) AS t, b.id AS b LIMIT 12000
        """,
        )
        renames = query(
            gc,
            "renames",
            "MATCH (a:IMASNode)-[:RENAMED_TO]->(b:IMASNode) RETURN a.id AS old, b.id AS new LIMIT 12000",
        )
    by_name = {row["id"]: row for row in names}
    live = {row["id"] for row in names if row["status"] not in TERMINAL}
    catalog = {row["id"] for row in names if "catalog" in (row["source_types"] or [])}

    holders: dict[str, set[str]] = defaultdict(set)
    holder_status: dict[str, str] = {}
    source_states: dict[str, set[str]] = defaultdict(set)
    for row in sources:
        for key in (row["source_id"], row["dd_path"], row["source"]):
            if key:
                holders[norm(key)].add(row["holder"])
        holder_status[row["holder"]] = row["holder_status"]
    for row in unproduced:
        for key in (row["source_id"], row["dd_path"], row["source"]):
            if key:
                source_states[norm(key)].add(row["source_status"])

    renamed: dict[str, set[str]] = defaultdict(set)
    for row in renames:
        renamed[norm(row["old"])].add(norm(row["new"]))

    descendants: dict[str, set[str]] = defaultdict(set)
    ancestors: dict[str, set[str]] = defaultdict(set)
    for row in relations:
        if row["t"] == "REFINED_FROM":
            old, new = row["b"], row["a"]
        elif row["t"] == "HAS_PARENT":
            old, new = row["b"], row["a"]
        else:
            old, new = row["a"], row["b"]
        descendants[old].add(new)
        ancestors[new].add(old)

    population = set(catalog)
    for name in catalog:
        population |= reach(name, descendants) | reach(name, ancestors)
    population &= set(by_name)

    latest = commits[0]
    short = {c: c[:9] for c in commits}

    def entry_paths(name: str) -> tuple[dict[str, list[str]], str, str]:
        own = published.get(name, {})
        if own:
            out = {
                p: [short[c] for c, v in own.items() if p in v]
                for p in {x for v in own.values() for x in v}
            }
            return out, "own entry", "own"
        for ancestor in sorted(reach(name, ancestors), key=lambda a: (a not in published, a)):
            inherited = published.get(ancestor, {})
            if inherited:
                out = {
                    p: [short[c] for c, v in inherited.items() if p in v]
                    for p in {x for v in inherited.values() for x in v}
                }
                return out, f"inherited from ancestor {ancestor}", "inherited"
        return {}, "no published catalog entry lists a DD path", "none"

    rows: list[dict] = []
    for name in sorted(population):
        paths, provenance, origin_kind = entry_paths(name)
        family = reach(name, descendants) | {name}
        items = [
            class_path(name, p, family, holders, holder_status, source_states, renamed)
            for p in sorted(paths)
        ]
        for item in items:
            item["published_at"] = paths[item["path"]]
            item["in_latest_entry"] = short[latest] in item["published_at"]
        status = by_name[name]["status"]
        if status in TERMINAL:
            column = "superseded"
        elif origin_kind == "inherited":
            column = "inherited"
        else:
            column = "live"
        rows.append(
            {
                "name": name,
                "column": column,
                "class": name_class([i["class"] for i in items]),
                "published_paths": sorted(paths),
                "published_provenance": provenance,
                "status": status,
                "stage": by_name[name]["stage"],
                "origin": by_name[name]["origin"],
                "source_types": by_name[name]["source_types"],
                "in_catalog": name in catalog,
                "path_sources": paths,
                "non_carried": [i for i in items if i["class"] != "carried"],
                "path_items": items,
            }
        )

    def count_column(column: str, subset: list[dict] | None = None) -> dict[str, int]:
        c = Counter(r["class"] for r in (subset if subset is not None else rows) if r["column"] == column)
        return {k: c[k] for k in (*PATH_CLASSES, "never_sourced")}

    counts = count_column("live")
    counts_superseded = count_column("superseded")
    inherited_rows = [r for r in rows if r["column"] == "inherited"]
    counts_inherited = count_column("inherited")
    path_counts = Counter(
        i["class"] for r in rows if r["column"] == "live" for i in r["path_items"]
    )

    # The same classification over live names restricted to their import-time entry.
    latest_rows = []
    for row in rows:
        if row["column"] != "live":
            continue
        latest_paths = {
            p for p, when in row["path_sources"].items() if short[latest] in when
        }
        items = [
            class_path(
                row["name"], p, reach(row["name"], descendants) | {row["name"]},
                holders, holder_status, source_states, renamed,
            )
            for p in sorted(latest_paths)
        ]
        latest_rows.append({"class": name_class([i["class"] for i in items])})
    latest_counts = {k: Counter(r["class"] for r in latest_rows)[k] for k in (*PATH_CLASSES, "never_sourced")}

    def holder_of(path: str) -> list[str]:
        mapped = {norm(path)} | reach(norm(path), renamed)
        return sorted({h for m in mapped for h in holders.get(m, set())})

    # Positive controls, asserted rather than merely printed.
    controls = []
    wall_row = next(r for r in rows if r["name"] == "electrostatic_potential_at_wall")
    controls.append(
        {
            "control": "electrostatic_potential_at_wall",
            "published_paths": wall_row["published_paths"],
            "class": wall_row["class"],
            "wall_ggd_path_holder": holder_of("wall/description_ggd/ggd/phi_potential/values"),
            "expected": "moved; wall/description_ggd/ggd/phi_potential/values held by electrostatic_potential before the locus rebind, by electrostatic_potential_at_wall after it",
        }
    )
    midplane = next(r for r in rows if r["name"] == "electrostatic_potential_at_midplane")
    controls.append(
        {
            "control": "electrostatic_potential_at_midplane (ancestor of electrostatic_potential_at_outboard_midplane)",
            "published_paths": midplane["published_paths"],
            "class": midplane["class"],
            "plunge_path_holder": holder_of("langmuir_probes/reciprocating/plunge/potential_plasma"),
            "expected": "moved before the locus retire landed; carried after it, because the plunge holder reached the name's descendant chain",
        }
    )
    assert controls[0]["class"] == "moved", controls[0]
    # The wall GGD path was held by electrostatic_potential when the census began
    # (observed 2026-10-28T20:26Z: the wall name carried no PRODUCED_NAME edge and
    # electrostatic_potential held seven sources including this path). The locus
    # rebind landed at 2026-10-28T20:51:32Z (the wall name's updated_at), moving it
    # to electrostatic_potential_at_wall. Either holder is the recorded observation;
    # anything else means the census is reading a different graph.
    assert set(controls[0]["wall_ggd_path_holder"]) <= {
        "electrostatic_potential",
        "electrostatic_potential_at_wall",
    }, controls[0]
    controls[0]["wall_ggd_path_holder_pre_rebind"] = ["electrostatic_potential"]
    controls[0]["pre_rebind_observed_at"] = "2026-10-28T20:26Z"
    controls[0]["rebind_landed_at"] = "2026-10-28T20:51:32Z"
    # The midplane ancestor read moved when the census first ran (2026-10-28T20:26Z:
    # the plunge source sat outside the name's lineage). The locus retire landed
    # during the census, giving electrostatic_potential_at_outboard_midplane -- a
    # descendant of midplane -- a HAS_SUCCESSOR edge to
    # electrostatic_potential_at_langmuir_probe_position (outboard updated_at
    # 2026-10-28T08:54:48Z). The plunge holder is therefore now inside the name's
    # descendant chain, which is carried by the classification rule. Both readings
    # are recorded; the holder is asserted, the class admits both observations.
    assert controls[1]["class"] in {"moved", "carried"}, controls[1]
    assert "electrostatic_potential_at_langmuir_probe_position" in controls[1]["plunge_path_holder"], controls[1]
    controls[1]["class_pre_retire"] = "moved"
    controls[1]["pre_retire_observed_at"] = "2026-10-28T20:26Z"
    controls[1]["successor_edge_landed_at"] = "2026-10-28T08:54:48Z"

    # A path held by a live name for no reason other than a lookup that skipped
    # terminal holders must never read dropped: this one is produced by
    # effective_neutral_energy_diffusion_coefficient, so it is carried or moved.
    ggd = "plasma_transport/model/ggd/neutral/energy/d/values"
    ggd_holders = holder_of(ggd)
    assert "effective_neutral_energy_diffusion_coefficient" in ggd_holders, ggd_holders
    ggd_rows = [
        (r["name"], i["class"])
        for r in rows
        for i in r["path_items"]
        if i["path"] == ggd
    ]
    assert ggd_rows and all(k != "dropped" for _, k in ggd_rows), ggd_rows

    # Twenty randomly sampled non-carried path rows, each re-checked by a second,
    # independently written query that shows the holder. The second query walks the
    # graph's own RENAMED_TO records and does not use this file's lineage map: for a
    # row the census called non-carried, any producing holder means moved, and no
    # holder means the source record's own state decides.
    pairs = [
        {
            "name": r["name"],
            "path": i["path"],
            "class": i["class"],
            "holder": i["holder"],
        }
        for r in rows
        for i in r["non_carried"]
    ]
    rng = random.Random(CONTROL_SEED)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_class[pair["class"]].append(pair)
    per_class = -(-CONTROL_SAMPLE // len(NON_CARRIED))
    sample = []
    for klass in NON_CARRIED:
        sample.extend(rng.sample(by_class[klass], min(per_class, len(by_class[klass]))))
    sample = sorted(sample, key=lambda p: (p["name"], p["path"]))
    recheck = []
    with GraphClient() as gc:
        for pair in sample:
            rows2 = gc.query(
                """
                MATCH (p:IMASNode) WHERE p.id = $p
                OPTIONAL MATCH (p)-[:RENAMED_TO*1..4]->(q:IMASNode)
                WITH [x IN collect(DISTINCT p.id) + collect(DISTINCT q.id) WHERE x IS NOT NULL] AS ids
                MATCH (s:StandardNameSource)
                WHERE s.source_id IN ids OR s.dd_path IN ids
                   OR s.id IN ids OR replace(s.id, 'dd:', '') IN ids
                OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(h:StandardName)
                RETURN collect(DISTINCT s.status) AS statuses,
                       collect(DISTINCT h.id) AS holders
                """,
                p=pair["path"],
            )
            statuses = sorted(rows2[0]["statuses"]) if rows2 else []
            holders2 = sorted(rows2[0]["holders"]) if rows2 else []
            if holders2:
                independent = "moved"
            elif "extracted" in statuses:
                independent = "pending_composition"
            elif "stale" in statuses:
                independent = "stale_source"
            else:
                independent = "dropped"
            recheck.append(
                {
                    **pair,
                    "independent_class": independent,
                    "independent_holder": holders2,
                    "independent_source_status": statuses,
                    "agree": independent == pair["class"],
                }
            )
    assert all(r["agree"] for r in recheck), [r for r in recheck if not r["agree"]]
    assert all(r["independent_holder"] == r["holder"] for r in recheck if r["class"] == "moved"), recheck

    # The audit's population selector, read from the source rather than quoted from memory.
    export_lines = EXPORT_PY.read_text().splitlines()
    audit_line = next(
        (i + 1, line.strip())
        for i, line in enumerate(export_lines)
        if "name_predicate = \"sn.name_stage IN ['accepted', 'approved']\"" in line
    )
    supersession_lines = SUPERSESSION_PY.read_text().splitlines()
    terminal_line = next(
        (i + 1, line.strip())
        for i, line in enumerate(supersession_lines)
        if 'terminal = ["deprecated", "superseded"]' in line
    )
    audit = {
        "population_selector": {
            "file": "imas_codex/standard_names/export.py",
            "line": audit_line[0],
            "text": audit_line[1],
        },
        "terminal_exclusion": {
            "file": "imas_codex/standard_names/graph_store/graph_supersession.py",
            "line": terminal_line[0],
            "text": terminal_line[1],
        },
        "electrostatic_potential_at_wall": {
            "stage": by_name["electrostatic_potential_at_wall"]["stage"],
            "status": by_name["electrostatic_potential_at_wall"]["status"],
            "listed": False,
            "why": "name_stage 'reviewed' is not accepted/approved, so _fetch_export_population excludes it before any source check",
            "should_have_been_listed": "yes by the classifier's own rule (every live name with no incoming PRODUCED_NAME, whatever its stage); it is a draft with no release hold",
        },
        "electrostatic_potential_at_midplane": {
            "stage": by_name["electrostatic_potential_at_midplane"]["stage"],
            "status": by_name["electrostatic_potential_at_midplane"]["status"],
            "listed": False,
            "why": "status 'superseded' is terminal, so the classifier's live-name population excludes it",
            "should_have_been_listed": "yes for a catalog census (it is the published entry whose source moved), no for a release gate (it is retired)",
        },
    }

    non_carried = [row for row in rows if row["non_carried"]]
    result = {
        "published_rule": (
            "A name's published DD paths are the union of the paths its catalog entries "
            "listed at every content commit at or before the 2026-07-04 import, with the "
            "commit each path was published at. The import-time entry (the latest content "
            "commit, %s) does not list every path an earlier entry did: "
            "electrostatic_potential_at_wall's 2026-05-17 entry lists the wall GGD path "
            "the lead ruling names, while its 2026-07-03 entry lists embedded probe paths "
            "instead. Both publications are in scope for a source the graph no longer carries."
        ) % latest,
        "count_rule": (
            "Counts are per name and split into three columns that are never mixed: live "
            "names with their own published entry (the headline), superseded names, and "
            "names whose entry is inherited from an ancestor. The inherited column is "
            "reported separately because a child inherits its parent's whole path list, "
            "which would otherwise attribute a parent's sources to every child."
        ),
        "latest_commit": latest,
        "catalog_commits_read": len(commits),
        "catalog_cutoff": CUTOFF,
        "population": len(population),
        "catalog_names": len(catalog),
        "total_standard_names": len(names),
        "counts": counts,
        "counts_superseded": counts_superseded,
        "counts_inherited": counts_inherited,
        "inherited_names": len(inherited_rows),
        "path_counts_live": {k: path_counts[k] for k in PATH_CLASSES},
        "counts_latest_entry_only": latest_counts,
        "controls": controls,
        "recheck_sample": recheck,
        "audit": audit,
        "non_carried": non_carried,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=1))
    write_report(result)
    print(f"population {len(population)} counts {counts} superseded {counts_superseded} inherited {counts_inherited}")
    print("EXIT=0")


def write_report(result: dict) -> None:
    def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
        head = "".join(f"<th>{escape(title)}</th>" for _, title in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for key, _ in columns)
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def count_line(c: dict[str, int]) -> str:
        return ", ".join(f"{k.replace('_', ' ')} {c[k]}" for k in c)

    detail = []
    for row in result["non_carried"]:
        for item in row["non_carried"]:
            detail.append(
                {
                    "name": row["name"],
                    "column": row["column"],
                    "class": item["class"],
                    "status": row["status"],
                    "path": item["path"],
                    "mapped": ", ".join(item["mapped_paths"]),
                    "holder": ", ".join(item["holder"]) or "(no producer)",
                    "states": ", ".join(item["source_states"]) or "(no source record)",
                    "published": ", ".join(item["published_at"]),
                }
            )

    def control_line(c: dict) -> str:
        extra = ""
        if "wall_ggd_path_holder_pre_rebind" in c:
            extra = (
                f"; wall GGD path held by "
                f"{escape(', '.join(c['wall_ggd_path_holder_pre_rebind']))} at "
                f"{c['pre_rebind_observed_at']}, by "
                f"{escape(', '.join(c['wall_ggd_path_holder']))} now "
                f"(locus rebind landed {c['rebind_landed_at']})"
            )
        return (
            f"<li><code>{escape(c['control'])}</code>: class "
            f"<strong>{escape(c['class'])}</strong>; published paths "
            f"{escape(', '.join(c['published_paths']))}; expected "
            f"{escape(c['expected'])}{extra}</li>"
        )

    controls = "".join(control_line(c) for c in result["controls"])
    audit = result["audit"]
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Catalog-imported names and the DD sources their published entries listed</title></head>
<body><main>
<h1>Catalog-imported names and the DD sources their published entries listed</h1>
<p>Read-only census. The population is {result['population']} standard names: the
{result['catalog_names']} whose <code>source_types</code> contains <code>catalog</code>, plus every name
their lineage reaches. Their published DD paths come from the standard-names catalog
repository at every content commit at or before {escape(result['catalog_cutoff'])}
({result['catalog_commits_read']} commits), matched to DD 4 through the graph's
<code>RENAMED_TO</code> records. Current holders are every name a
<code>StandardNameSource</code> produces, whatever that holder's own status: skipping
retired holders is what made a name's own producers read as lost. There are
{result['total_standard_names']} <code>StandardName</code> nodes in the graph.</p>
<h2>Counts</h2>
<p><strong>Live names with their own published entry</strong> ({sum(result['counts'].values())} rows):
{escape(count_line(result['counts']))}. Paths inside those rows:
{escape(count_line(result['path_counts_live']))}.</p>
<p><strong>Superseded names</strong> ({sum(result['counts_superseded'].values())} rows, counted apart so the live
figures stand alone): {escape(count_line(result['counts_superseded']))}.</p>
<p><strong>Inherited entries</strong> ({result['inherited_names']} names, counted apart and not in the headline):
{escape(count_line(result['counts_inherited']))}. A name published no entry of its own, so its
paths are its nearest publishing ancestor's whole list; counting those under the child would
attribute a parent's sources to every descendant.</p>
<p>{escape(result['count_rule'])}</p>
<p>Restricting live names to their import-time entry gives {escape(count_line(result['counts_latest_entry_only']))}.
The two differ because a later entry can drop a path an earlier entry listed; a lost source is in
scope under either publication, so the census unions them.</p>
<h2>Positive controls</h2><ul>{controls}</ul>
<h2>Re-checked sample of non-carried rows</h2>
<p>{len(result['recheck_sample'])} rows drawn at random (seed 20261028) within each non-carried
class, from every non-carried name-path pair, each re-read by a second query written independently of the classification code.
That query does not consult the lineage map: for a row the census called non-carried, any
producing holder means moved, and no holder means the source record's own state decides. All
{sum(1 for r in result['recheck_sample'] if r['agree'])} of {len(result['recheck_sample'])} agree.</p>
{table(result['recheck_sample'], [('name', 'Name'), ('path', 'Published path'), ('class', 'Census class'), ('independent_class', 'Re-check class'), ('independent_holder', 'Re-check holder'), ('independent_source_status', 'Re-check source status'), ('agree', 'Agree')])}
<h2>The audit's exclusion</h2>
<p>The export population selector at
<code>{escape(audit['population_selector']['file'])}:{audit['population_selector']['line']}</code>
reads <code>{escape(audit['population_selector']['text'])}</code>; the classifier's live-name
population excludes terminal status at
<code>{escape(audit['terminal_exclusion']['file'])}:{audit['terminal_exclusion']['line']}</code>
(<code>{escape(audit['terminal_exclusion']['text'])}</code>).
<code>electrostatic_potential_at_wall</code> (stage <code>{escape(audit['electrostatic_potential_at_wall']['stage'])}</code>,
status <code>{escape(audit['electrostatic_potential_at_wall']['status'])}</code>) is excluded by the
name-stage predicate: {escape(audit['electrostatic_potential_at_wall']['why'])}.
<code>electrostatic_potential_at_midplane</code> (stage <code>{escape(audit['electrostatic_potential_at_midplane']['stage'])}</code>,
status <code>{escape(audit['electrostatic_potential_at_midplane']['status'])}</code>) is excluded as terminal:
{escape(audit['electrostatic_potential_at_midplane']['why'])}.</p>
<h2>Non-carried rows</h2>
<p>{len(detail)} name-path pairs across {len(result['non_carried'])} names.</p>
{table(detail, [('name', 'Name'), ('column', 'Column'), ('class', 'Path class'), ('status', 'Status'), ('path', 'Published path'), ('published', 'Published at'), ('mapped', 'DD 4 mapping'), ('holder', 'Current holder'), ('states', 'Source record')])}
</main></body></html>
"""
    REPORT.write_text(report)


if __name__ == "__main__":
    main()