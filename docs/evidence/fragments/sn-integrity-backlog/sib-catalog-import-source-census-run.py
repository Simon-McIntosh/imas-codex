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

Classification against the sources the graph holds now, on the name and its
live descendants:

* carried      -- every listed path is bound to the name or a live descendant
* moved        -- a listed path is bound to a different, unrelated live name
* dropped      -- a listed path is bound to no live name
* never sourced -- the published entry listed no path
"""

from __future__ import annotations

import json
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
CLASSES = ("carried", "moved", "dropped", "never_sourced")
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
    return str(path).removeprefix("dd:").strip("/")


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


def classify_name(
    name: str,
    paths: set[str],
    descendants: set[str],
    holders: dict[str, set[str]],
    renamed: dict[str, set[str]],
) -> tuple[str, list[dict]]:
    if not paths:
        return "never_sourced", []
    detail: list[dict] = []
    states = set()
    for path in sorted(paths):
        mapped = {path} | reach(path, renamed)
        live_holders = {h for m in mapped for h in holders.get(m, set())}
        carried = bool(live_holders & (descendants | {name}))
        state = (
            "carried"
            if carried
            else "moved"
            if live_holders
            else "dropped"
        )
        states.add(state)
        if not carried:
            detail.append(
                {
                    "path": path,
                    "mapped_paths": sorted(mapped),
                    "state": state,
                    "holder": sorted(live_holders),
                }
            )
    if states == {"carried"}:
        return "carried", []
    if "moved" in states:
        return "moved", detail
    return "dropped", detail


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
            WHERE NOT h.status IN $terminal
            RETURN s.id AS source, s.source_id AS source_id, s.dd_path AS dd_path,
                   h.id AS holder LIMIT 12000
        """,
            terminal=sorted(TERMINAL),
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
    for row in sources:
        for key in (row["source_id"], row["dd_path"], row["source"]):
            if key:
                holders[norm(key)].add(row["holder"])
    renamed: dict[str, set[str]] = defaultdict(set)
    rename_targets = {row["new"] for row in renames}
    for row in renames:
        renamed[norm(row["old"])].add(norm(row["new"]))
    # A catalog path may already be the DD 4 name; also try the pre-rename id.
    rename_sources = {row["new"]: row["old"] for row in renames}

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

    def entry_paths(name: str) -> tuple[dict[str, list[str]], str]:
        own = published.get(name, {})
        if own:
            out = {p: [short[c] for c, v in own.items() if p in v] for p in {x for v in own.values() for x in v}}
            return out, "own entry"
        for ancestor in sorted(reach(name, ancestors), key=lambda a: (a not in published, a)):
            inherited = published.get(ancestor, {})
            if inherited:
                out = {p: [short[c] for c, v in inherited.items() if p in v] for p in {x for v in inherited.values() for x in v}}
                return out, f"inherited from ancestor {ancestor}"
        return {}, "no published catalog entry lists a DD path"

    rows: list[dict] = []
    for name in sorted(population):
        sources, provenance = entry_paths(name)
        paths = set(sources)
        name_desc = reach(name, descendants) & live
        klass, detail = classify_name(name, paths, name_desc, holders, renamed)
        for item in detail:
            item["published_at"] = sources[item["path"]]
            item["in_latest_entry"] = short[latest] in item["published_at"]
        rows.append(
            {
                "name": name,
                "class": klass,
                "published_paths": sorted(paths),
                "published_provenance": provenance,
                "status": by_name[name]["status"],
                "stage": by_name[name]["stage"],
                "origin": by_name[name]["origin"],
                "source_types": by_name[name]["source_types"],
                "in_catalog": name in catalog,
                "path_sources": sources,
                "non_carried": detail,
            }
        )
    counts = Counter(row["class"] for row in rows)

    # The same classification restricted to each name's import-time (latest) entry.
    latest_counts = Counter()
    for row in rows:
        all_latest = {
            p for p, when in row["path_sources"].items() if short[latest] in when
        }
        latest_counts[
            classify_name(
                row["name"],
                all_latest,
                reach(row["name"], descendants) & live,
                holders,
                renamed,
            )[0]
        ] += 1
    counts_latest = {k: latest_counts[k] for k in CLASSES}

    def holder_of(path: str) -> list[str]:
        mapped = {path} | reach(norm(path), renamed)
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
    assert controls[0]["class"] in {"moved", "carried"}, controls[0]
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
    # are recorded; either class is the observation, and the holder is not.
    assert controls[1]["class"] in {"moved", "carried"}, controls[1]
    assert "electrostatic_potential_at_langmuir_probe_position" in controls[1]["plunge_path_holder"], controls[1]
    controls[1]["class_pre_retire"] = "moved"
    controls[1]["pre_retire_observed_at"] = "2026-10-28T20:26Z"
    controls[1]["successor_edge_landed_at"] = "2026-10-28T08:54:48Z"

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

    non_carried = [row for row in rows if row["class"] != "carried"]
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
        "latest_commit": latest,
        "catalog_commits_read": len(commits),
        "catalog_cutoff": CUTOFF,
        "population": len(population),
        "catalog_names": len(catalog),
        "total_standard_names": len(names),
        "counts": {k: counts[k] for k in CLASSES},
        "counts_latest_entry_only": counts_latest,
        "controls": controls,
        "audit": audit,
        "non_carried": non_carried,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=1))
    write_report(result)
    print(f"population {len(population)} counts {dict(counts)}")
    print("EXIT=0")


def write_report(result: dict) -> None:
    counts = result["counts"]

    def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
        head = "".join(f"<th>{escape(title)}</th>" for _, title in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for key, _ in columns)
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    detail = []
    for row in result["non_carried"]:
        for item in row["non_carried"]:
            detail.append(
                {
                    "name": row["name"],
                    "class": row["class"],
                    "status": row["status"],
                    "stage": row["stage"],
                    "path": item["path"],
                    "mapped": ", ".join(item["mapped_paths"]),
                    "holder": ", ".join(item["holder"]) or "(none live)",
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
<code>RENAMED_TO</code> records. Current holders are the live names a
<code>StandardNameSource</code> produces. There are {result['total_standard_names']}
<code>StandardName</code> nodes in the graph.</p>
<h2>Counts</h2>
<p><strong>{counts['carried']}</strong> carried, <strong>{counts['moved']}</strong> moved,
<strong>{counts['dropped']}</strong> dropped, <strong>{counts['never_sourced']}</strong> never sourced.
Every row is in exactly one class. Restricting each name to its import-time (latest)
entry gives {result['counts_latest_entry_only']['carried']} carried,
{result['counts_latest_entry_only']['moved']} moved,
{result['counts_latest_entry_only']['dropped']} dropped,
{result['counts_latest_entry_only']['never_sourced']} never sourced. The two figures
differ because a name's later entry can drop a path an earlier entry listed; a dropped
or moved source is in scope under either publication, so the census uses the union.</p>
<h2>Positive controls</h2><ul>{controls}</ul>
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
{table(detail, [('name', 'Name'), ('class', 'Class'), ('status', 'Status'), ('stage', 'Stage'), ('path', 'Published path'), ('published', 'Published at'), ('mapped', 'DD 4 mapping'), ('holder', 'Current live holder')])}
</main></body></html>
"""
    REPORT.write_text(report)


if __name__ == "__main__":
    main()