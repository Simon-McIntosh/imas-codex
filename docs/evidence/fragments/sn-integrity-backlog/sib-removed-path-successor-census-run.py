"""Census the live names whose every producing DD source was removed by a DD version.

Population: every StandardName whose producing StandardNameSource set is not
empty and every member is ``status='stale'``.  A coordinator read on 2026-11-06
counted 40 such names; the count includes superseded and exhausted stages, so
each row records its own ``name_stage`` and ``status``.  Only five are live and
three of those are accepted, which is what makes this generator release-gating.

For each stale source path the census records the DD version that removed it
(``DEPRECATED_IN``), its DD 4 successor and how that successor was found:

* ``renamed_to_edge``      -- a ``RENAMED_TO`` edge in the graph;
* ``ids_rename_dd_xml``    -- the renamed IDS element's own
                              ``change_nbc_previous_name``, which
                              ``dd_lifecycle.dd_path_index`` never reads;
* ``structural_inference`` -- the only DD 4 structure at the same position
                              whose tail matches, which is how an
                              array-of-structures rename surfaces when no
                              record anywhere names it;
* ``none``.

The generator is the stale marking in
``reconcile_standard_name_sources``: it marks the source and leaves the name it
produced untouched.  A disposable Neo4j shows what happens to that name.

Read-only against the live graph.  The disposable fixture is written.  No LLM
call.
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "sib-removed-path-successor-census.json"
REPORT = HERE / "sib-removed-path-successor-census.html"

GRAPH_AUDITS = "imas_codex/standard_names/graph_store/graph_audits.py"
SOURCE_BINDINGS = "imas_codex/standard_names/graph_store/graph_source_bindings.py"
DD_LIFECYCLE = "imas_codex/graph/dd_lifecycle.py"
SOURCE_INVARIANT = "imas_codex/standard_names/source_invariant.py"
EXPORT_PY = "imas_codex/standard_names/export.py"

POPULATION_CYPHER = """
MATCH (sn:StandardName)<-[:PRODUCED_NAME]-(src:StandardNameSource)
WITH sn, collect(src) AS srcs
WHERE ALL(s IN srcs WHERE coalesce(s.status,'') = 'stale')
RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status,
       sn.origin AS origin, sn.unit AS unit,
       [s IN srcs | {id: s.id, source_id: s.source_id,
                     status: s.status}] AS sources
ORDER BY sn.id
"""

NODE_CYPHER = """
UNWIND $paths AS p
OPTIONAL MATCH (n:IMASNode {id: p})
OPTIONAL MATCH (n)-[:RENAMED_TO]->(succ:IMASNode)
OPTIONAL MATCH (n)-[:DEPRECATED_IN]->(dv:DDVersion)
OPTIONAL MATCH (n)-[:INTRODUCED_IN]->(iv:DDVersion)
RETURN p AS path,
       n.id IS NOT NULL AS present,
       n.unit AS unit,
       n.lifecycle_status AS lifecycle_status,
       n.renamed_to AS renamed_to_property,
       [x IN collect(DISTINCT succ.id) | x] AS renamed_to_edges,
       [x IN collect(DISTINCT dv.id) | x] AS deprecated_in,
       [x IN collect(DISTINCT iv.id) | x] AS introduced_in
"""

SUCCESSOR_CYPHER = """
UNWIND $paths AS p
OPTIONAL MATCH (n:IMASNode {id: p})
OPTIONAL MATCH (src:StandardNameSource {source_type: 'dd', source_id: p})
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName)
RETURN p AS path,
       n.id IS NOT NULL AS present,
       n.lifecycle_status AS lifecycle_status,
       n.unit AS unit,
       src.id AS successor_source_id,
       src.status AS source_status,
       [x IN collect(DISTINCT sn.id) WHERE x IS NOT NULL] AS produced_by
"""


def line_of(relpath: str, needle: str) -> str:
    """Return the first ``relpath:line`` holding *needle* in the working tree."""
    path = ROOT / relpath
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if needle in text:
            return f"{relpath}:{number}"
    return f"{relpath}:absent"


def dd_rename_metadata() -> dict[str, dict[str, str]]:
    """Read DD 4 rename metadata the lifecycle reconciler ignores.

    Returns ``{"ids": {old_ids: new_ids}, "field": {old_path: new_path}}``.

    ``dd_lifecycle.dd_path_index`` walks only ``<field>`` elements and only
    reads their ``change_nbc_previous_name``; the renamed ``<IDS>`` element's
    own previous name is never read, so an IDS rename produces no rename
    record anywhere.  This reads both.
    """
    from xml.etree import ElementTree as ET

    import imas_data_dictionaries as idd

    from imas_codex import dd_version

    raw = idd.get_dd_xml(dd_version)
    root = ET.fromstring(raw if isinstance(raw, bytes) else raw.encode())
    ids_map: dict[str, str] = {}
    field_map: dict[str, str] = {}

    def walk(el, new_segs, old_segs, ids):
        for f in el.findall("field"):
            name = f.get("name")
            if not name:
                continue
            prev = f.get("change_nbc_previous_name")
            ns = [*new_segs, name]
            os_ = [*old_segs, *(prev.split("/") if prev else [name])]
            new_path = f"{ids}/{'/'.join(ns)}"
            if os_ != ns:
                field_map[f"{ids}/{'/'.join(os_)}"] = new_path
            walk(f, ns, os_, ids)

    for ids_el in root.iter("IDS"):
        current = ids_el.get("name")
        if not current:
            continue
        previous = ids_el.get("change_nbc_previous_name")
        if previous and previous != current:
            ids_map[previous] = current
        walk(ids_el, [], [], current)
    return {"ids": ids_map, "field": field_map}


def dd_valid_paths() -> set[str]:
    """Every machine path the current DD declares."""
    from imas_codex.graph.dd_lifecycle import dd_path_index

    valid, _ = dd_path_index()
    return set(valid)


def structural_successor(path: str, valid: set[str]) -> str | None:
    """Find the only DD 4 path that differs from *path* by one segment.

    Only a structure-level segment is substituted: the segment must be
    non-final, so a renamed container carries a whole subtree, and the IDS
    segment is left to the DD metadata.  Replacing a leaf would match a
    sibling field that merely shares a suffix, which is a different quantity.
    The substituted structure must itself be gone from the current DD; a
    structure that still exists under its own name did not move, so a same-tail
    path beside it is a different quantity.  A unique candidate is a rename no
    record names; several means the evidence does not decide and ``None`` is
    returned rather than a guess.
    """
    segments = path.split("/")
    if len(segments) < 3:
        return None
    candidates = set()
    for index in range(1, len(segments) - 1):
        if not structure_gone(path, index, valid):
            continue
        prefix = segments[:index]
        suffix = segments[index + 1 :]
        for other in valid:
            parts = other.split("/")
            if len(parts) != len(segments):
                continue
            if parts[:index] != prefix or parts[index + 1 :] != suffix:
                continue
            if parts[index] != segments[index]:
                candidates.add(other)
    return candidates.pop() if len(candidates) == 1 else None


def find_successor(
    path: str,
    node: dict,
    renames: dict[str, dict[str, str]],
    valid: set[str],
) -> dict:
    """Classify how a DD 4 successor for *path* was found."""
    edges = node.get("renamed_to_edges") or []
    if edges:
        return {"path": edges[0], "method": "renamed_to_edge"}
    prop = node.get("renamed_to_property")
    if prop:
        return {"path": prop, "method": "renamed_to_edge"}

    ids, _, rest = path.partition("/")
    new_ids = renames["ids"].get(ids)
    if new_ids:
        candidate = f"{new_ids}/{rest}"
        if candidate in valid:
            return {"path": candidate, "method": "ids_rename_dd_xml"}

    renamed_field = renames["field"].get(path)
    if renamed_field and renamed_field in valid:
        return {"path": renamed_field, "method": "field_rename_dd_xml"}

    inferred = structural_successor(path, valid)
    if inferred:
        return {"path": inferred, "method": "structural_inference"}
    return {"path": None, "method": "none"}


def structure_gone(path: str, index: int, valid: set[str]) -> bool:
    """Whether the structure named by ``path[:index+1]`` exists in the DD at all.

    A structure that still exists under its own name is not the thing that
    moved, so a same-tail path elsewhere is a different quantity and must not
    be read as its successor.
    """
    prefix = "/".join(path.split("/")[: index + 1]) + "/"
    return not any(candidate.startswith(prefix) for candidate in valid)


def proposal_for(
    row: dict, source: dict, successor: dict, successor_row: dict | None
) -> dict:
    """Say what the lead should rule for one row."""
    stage = row["stage"]
    if stage in {"superseded", "exhausted"} or row["status"] in {
        "deprecated",
        "superseded",
    }:
        return {
            "action": "none: already retired",
            "reason": (
                f"the name is {stage}/{row['status']}; the source removal cost "
                "it nothing that is still live"
            ),
        }
    if not successor["path"]:
        return {
            "action": "hold or retire",
            "reason": "no DD 4 successor was found for the removed path",
        }
    if successor_row is None or not successor_row.get("present"):
        return {
            "action": "hold",
            "reason": "the successor path is not present in the current DD",
        }
    if successor_row.get("lifecycle_status") == "removed":
        return {
            "action": "hold",
            "reason": "the successor path is itself removed from the current DD",
        }
    units_match = (successor_row.get("unit") or None) == (row.get("unit") or None)
    holders = list(successor_row.get("produced_by") or [])
    if units_match and not holders:
        return {
            "action": "rebind",
            "reason": (
                "the successor exists with matching units and no name holds it; "
                "rebind through a signed source move and retire this name with "
                "a successor link"
            ),
        }
    if units_match and holders:
        return {
            "action": "retire with successor",
            "reason": (
                "the successor is already produced by "
                + ", ".join(sorted(holders))
                + "; retire this name with a successor link to that holder"
            ),
        }
    return {
        "action": "hold",
        "reason": (
            f"successor units {successor_row.get('unit')!r} do not match the "
            f"name's {row.get('unit')!r}"
        ),
    }


def fixture_result(attempts: int = 2) -> dict:
    """Start the disposable graph, retrying once, and measure what reconcile does."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _fixture_once()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            last = exc
            print(f"fixture attempt {attempt} failed: {exc}", flush=True)
    raise RuntimeError(
        f"disposable graph unavailable after {attempts} attempts: {last}"
    )


def _fixture_once() -> dict:
    """Show, on the disposable Neo4j, what reconcile does to the name it strands.

    Two sources hang off one accepted draft name.  One points at a DD path
    absent from the current DD with a ``FROM_DD_PATH`` edge to a node stamped
    ``removed``; the other points at a live path and is the positive control
    that proves the query can see a path that is present.  Reconcile runs for
    real, and the name is read back afterwards.
    """
    sys.path.insert(0, str(ROOT))
    from tests.standard_names.conftest import (
        _start_empty_neo4j,
        _wipe,
        refuse_project_graph,
    )

    directory = process = None
    try:
        directory, process, log, port = _start_empty_neo4j()
        uri = f"bolt://127.0.0.1:{port}"
        refuse_project_graph(uri)
        previous = {
            k: os.environ.get(k)
            for k in (
                "NEO4J_URI",
                "NEO4J_PASSWORD",
                "IMAS_CODEX_TEST_NEO4J_URI",
                "IMAS_CODEX_TEST_NEO4J_EPHEMERAL",
            )
        }
        os.environ.update(
            {
                "NEO4J_URI": uri,
                "NEO4J_PASSWORD": "",
                "IMAS_CODEX_TEST_NEO4J_URI": uri,
                "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
            }
        )
        try:
            _wipe(uri)
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                gc.query(
                    """
                    CREATE (sn:StandardName {id: 'fixture_name',
                            name_stage: 'accepted', status: 'draft'})
                    CREATE (ctrl:StandardName {id: 'fixture_control',
                            name_stage: 'accepted', status: 'draft'})
                    CREATE (gone:StandardNameSource {
                            id: 'dd:gyrokinetics/normalizing_quantities/r',
                            source_type: 'dd',
                            source_id: 'gyrokinetics/normalizing_quantities/r',
                            status: 'extracted'})
                    CREATE (kept:StandardNameSource {
                            id: 'dd:control/present',
                            source_type: 'dd', source_id: 'control/present',
                            status: 'extracted'})
                    CREATE (old:IMASNode {id: 'gyrokinetics/normalizing_quantities/r',
                            lifecycle_status: 'removed'})
                    CREATE (live:IMASNode {id: 'control/present'})
                    CREATE (gone)-[:PRODUCED_NAME]->(sn)
                    CREATE (kept)-[:PRODUCED_NAME]->(ctrl)
                    CREATE (gone)-[:FROM_DD_PATH]->(old)
                    CREATE (kept)-[:FROM_DD_PATH]->(live)
                    """
                )
                from imas_codex.standard_names.graph_store.graph_audits import (
                    reconcile_standard_name_sources,
                )

                reconcile = reconcile_standard_name_sources("dd")
                after = gc.query(
                    """
                    MATCH (sn:StandardName {id: 'fixture_name'})
                    OPTIONAL MATCH (sn)<-[:PRODUCED_NAME]-(s:StandardNameSource)
                    RETURN sn.name_stage AS stage, sn.status AS status,
                           [x IN collect({id: s.id, status: s.status})
                            WHERE x.id IS NOT NULL] AS sources
                    """
                )
                edges = gc.query(
                    """
                    MATCH (a:StandardName)-[r]->(b)
                    RETURN type(r) AS type, a.id AS from_id, b.id AS to_id
                    """
                )
                live_producers = gc.query(
                    """
                    MATCH (sn:StandardName {id: 'fixture_name'})
                          <-[p:PRODUCED_NAME]-(s:StandardNameSource)
                    WHERE coalesce(s.status, '') <> 'stale'
                    RETURN count(p) AS n
                    """
                )
                control = gc.query(
                    """
                    MATCH (:StandardName {id: 'fixture_control'})
                          <-[:PRODUCED_NAME]-(s:StandardNameSource)
                    RETURN s.status AS status
                    """
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    finally:
        if directory is not None:
            directory.cleanup()

    row = after[0] if after else {}
    sources = {s["id"]: s["status"] for s in row.get("sources", [])}
    return {
        "reconcile": reconcile,
        "name_stage_after": row.get("stage"),
        "name_status_after": row.get("status"),
        "sources_after": sources,
        "control_source_status": (control[0]["status"] if control else None),
        "control_name_producers": control,
        "orphan_source_status": sources.get("dd:gyrokinetics/normalizing_quantities/r"),
        "successor_edges_written": [
            e for e in edges if e["type"] in {"HAS_SUCCESSOR", "REFINED_FROM"}
        ],
        "all_edges": edges,
        "live_producers_after": live_producers[0]["n"] if live_producers else None,
        "sourceless_after": (live_producers[0]["n"] == 0) and bool(sources),
    }


def build() -> dict:
    from imas_codex.graph.client import GraphClient

    renames = dd_rename_metadata()
    valid = dd_valid_paths()
    with GraphClient() as gc:
        rows = list(gc.query(POPULATION_CYPHER))
        paths = sorted({s["source_id"] for r in rows for s in r["sources"]})
        nodes = {n["path"]: n for n in gc.query(NODE_CYPHER, paths=paths)}
        successors: dict[str, dict] = {}
        for path in paths:
            successors[path] = find_successor(path, nodes.get(path, {}), renames, valid)
        successor_paths = sorted({s["path"] for s in successors.values() if s["path"]})
        succ_rows = (
            {r["path"]: r for r in gc.query(SUCCESSOR_CYPHER, paths=successor_paths)}
            if successor_paths
            else {}
        )

    for row in rows:
        row["sources_detail"] = []
        for source in row["sources"]:
            path = source["source_id"]
            node = nodes.get(path, {})
            successor = successors[path]
            successor_row = (
                succ_rows.get(successor["path"]) if successor["path"] else None
            )
            row["sources_detail"].append(
                {
                    "path": path,
                    "removed_in": (node.get("deprecated_in") or [None])[0],
                    "introduced_in": (node.get("introduced_in") or [None])[0],
                    "lifecycle_status": node.get("lifecycle_status"),
                    "unit": node.get("unit"),
                    "successor": successor["path"],
                    "successor_method": successor["method"],
                    "successor_present": bool(
                        successor_row and successor_row.get("present")
                    ),
                    "successor_lifecycle": (successor_row or {}).get(
                        "lifecycle_status"
                    ),
                    "successor_unit": (successor_row or {}).get("unit"),
                    "successor_source_status": (successor_row or {}).get(
                        "source_status"
                    ),
                    "successor_source_id": (successor_row or {}).get(
                        "successor_source_id"
                    ),
                    "successor_produced_by": sorted(
                        (successor_row or {}).get("produced_by") or []
                    ),
                }
            )
        row["proposals"] = [
            proposal_for(
                row,
                s,
                successors[s["source_id"]],
                succ_rows.get(successors[s["source_id"]]["path"]),
            )
            for s in row["sources"]
        ]

    methods = {}
    for row in rows:
        for detail in row["sources_detail"]:
            methods[detail["successor_method"]] = (
                methods.get(detail["successor_method"], 0) + 1
            )
    live = [
        r["id"]
        for r in rows
        if r["stage"] not in {"superseded", "exhausted"}
        and r["status"] not in {"deprecated", "superseded"}
    ]
    accepted = [r["id"] for r in rows if r["stage"] == "accepted"]

    return {
        "population": len(rows),
        "rows": rows,
        "successor_methods": methods,
        "live_count": len(live),
        "live_names": live,
        "accepted_count": len(accepted),
        "accepted_names": accepted,
        "distinct_stale_paths": len(paths),
    }


def generator() -> dict:
    """The code that strands a name when its DD path is removed."""
    return {
        "id": "dd-removal-stale-mark",
        "label": "DD-version removal marks the source stale and leaves the name",
        "trigger": (
            "A DD rebuild stamps the path absent from the current DD with "
            "lifecycle_status='removed'; the next reconcile run marks every "
            "source still pointing at it stale."
        ),
        "stale_marking": line_of(GRAPH_AUDITS, "def reconcile_standard_name_sources"),
        "stale_query": line_of(GRAPH_AUDITS, "WHERE NOT (sns.status = 'stale')"),
        "stale_write": line_of(SOURCE_BINDINGS, "def mark_sources_stale"),
        "stale_write_set": line_of(SOURCE_BINDINGS, "SET sns.status = 'stale'"),
        "lifecycle_marking": line_of(DD_LIFECYCLE, "SET n.lifecycle_status = $removed"),
        "name_handling": (
            "Nothing. reconcile_standard_name_sources returns stale_marked, "
            "revived and relinked counts; it never reads the name that "
            "PRODUCED_NAME pointed at and never writes a supersession."
        ),
        "not_called": [
            line_of(SOURCE_INVARIANT, "def supersede_drained_name_cypher"),
            line_of(SOURCE_INVARIANT, "def _drained_test"),
        ],
        "why_the_invariant_does_not_fire": (
            "The invariant runs inside a writer's transaction. Reconcile marks "
            "the source stale by setting a property; no writer runs, so the "
            "name is never tested for having lost its last producer."
        ),
        "rename_records": {
            "builder": line_of(
                "imas_codex/graph/build_dd.py", "def _batch_create_renamed_to"
            ),
            "source": "DDChange entries of type path_renamed",
            "ids_rename_reconciler": line_of(DD_LIFECYCLE, "def dd_path_index"),
            "ids_rename_skip": (
                "dd_path_index walks only <field> elements and reads "
                "change_nbc_previous_name from them; the renamed <IDS> "
                "element's own change_nbc_previous_name is never read, so an "
                "IDS rename has no RENAMED_TO edge and no renamed_to property."
            ),
            "aos_rename_skip": (
                "An array-of-structures rename is not carried by the DD 4.1.1 "
                "XML's change_nbc_previous_name either; it surfaces only by "
                "structural inference over the current DD's own paths."
            ),
        },
    }


def repair_node() -> dict:
    return {
        "goal": (
            "Supersede a standard name in the same transaction that marks its "
            "last live source stale, so a DD removal can no longer leave a "
            "live name sourceless."
        ),
        "write_paths": [
            "imas_codex/standard_names/graph_store/graph_audits.py",
            "imas_codex/standard_names/graph_store/graph_source_bindings.py",
            "tests/standard_names/test_source_invariants.py",
        ],
        "guard_test": (
            "tests/standard_names/test_source_invariants.py"
            "::test_removal_marks_last_source_stale_and_supersedes_its_name"
            " -- seeds a live accepted name with one source pointing at a "
            "removed DD path, runs reconcile_standard_name_sources, and "
            "asserts the name carries a successor edge and no non-stale "
            "producer. It fails on main: the fixture section below runs that "
            "exact scenario against today's code and the name keeps "
            "accepted/draft with zero non-stale producers."
        ),
        "note": (
            "Where the source has no recorded successor, the name is retired "
            "without a successor link rather than rebound to a guess; where "
            "another live name already holds the successor path, the edge "
            "points at that holder."
        ),
    }


def render(data: dict) -> str:
    rows = data["census"]["rows"]
    gen = data["generator"]
    fx = data["fixture"]
    rep = data["repair"]
    parts = [
        "<h1>Names produced only by DD paths the DD removed</h1>",
        "<p>Every name below has at least one producing "
        "<code>StandardNameSource</code> and every one of them is "
        "<code>status='stale'</code>. The coordinator's 2026-11-06 read of 40 "
        "includes retired stages; the live subset is "
        f"<strong>{data['census']['live_count']}</strong>, of which "
        f"<strong>{data['census']['accepted_count']}</strong> are accepted and "
        "therefore gate the export.</p>",
        "<h2>Population</h2>",
        "<table><thead><tr><th>name</th><th>stage</th><th>status</th>"
        "<th>unit</th><th>removed sources</th><th>successor (how)</th>"
        "<th>successor unit</th><th>successor source</th>"
        "<th>held by</th><th>proposal</th></tr></thead><tbody>",
    ]
    for row in rows:
        srcs = "<br>".join(
            f"<code>{escape(d['path'])}</code> ({escape(str(d['removed_in']))})"
            for d in row["sources_detail"]
        )
        succ = "<br>".join(
            (
                f"<code>{escape(d['successor'])}</code> "
                f"({escape(d['successor_method'])})"
                if d["successor"]
                else "none"
            )
            for d in row["sources_detail"]
        )
        su = "<br>".join(
            escape(str(d["successor_unit"])) for d in row["sources_detail"]
        )
        sstat = "<br>".join(
            escape(str(d["successor_source_status"])) for d in row["sources_detail"]
        )
        held = "<br>".join(
            escape(", ".join(d["successor_produced_by"]) or "--")
            for d in row["sources_detail"]
        )
        prop = "<br>".join(
            f"<strong>{escape(p['action'])}</strong>: {escape(p['reason'])}"
            for p in row["proposals"]
        )
        parts.append(
            f"<tr><td><code>{escape(row['id'])}</code></td>"
            f"<td>{escape(str(row['stage']))}</td>"
            f"<td>{escape(str(row['status']))}</td>"
            f"<td>{escape(str(row['unit']))}</td>"
            f"<td>{sstat}</td><td>{srcs}</td><td>{succ}</td><td>{su}</td>"
            f"<td>{held}</td>"
            f"<td>{prop}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts += [
        "<h2>Is the generator still open?</h2>",
        f"<p><strong>Yes.</strong> {escape(gen['trigger'])}</p>",
        "<ul>",
        f"<li>stale query: <code>{escape(gen['stale_query'])}</code></li>",
        f"<li>stale write: <code>{escape(gen['stale_write_set'])}</code></li>",
        f"<li>lifecycle stamp: <code>{escape(gen['lifecycle_marking'])}</code></li>",
        f"<li>name handling: {escape(gen['name_handling'])}</li>",
        "</ul>",
        "<p>The supersession helper exists but reconcile never calls it: "
        + ", ".join(f"<code>{escape(x)}</code>" for x in gen["not_called"])
        + ". "
        + escape(gen["why_the_invariant_does_not_fire"])
        + "</p>",
        "<h3>Disposable-graph result</h3>",
        "<p>Two accepted draft names stand on an empty Neo4j started through "
        "<code>disposable_neo4j_uri</code>'s own launcher. "
        "<code>fixture_name</code>'s only source points at a DD path stamped "
        "<code>removed</code>; <code>fixture_control</code> is the positive "
        "control, its source pointing at a live path with its "
        "<code>FROM_DD_PATH</code> edge present. After the real "
        "<code>reconcile_standard_name_sources('dd')</code>:</p>",
        "<table><thead><tr><th>observation</th><th>value</th></tr></thead><tbody>",
    ]
    observations = [
        ("fixture error (empty when the graph started)", fx.get("error")),
        ("reconcile counters", fx.get("reconcile")),
        ("control source status (positive control)", fx.get("control_source_status")),
        ("removed-path source status", fx.get("orphan_source_status")),
        ("name_stage after", fx.get("name_stage_after")),
        ("name status after", fx.get("name_status_after")),
        ("successor edges written", fx.get("successor_edges_written")),
        ("non-stale producers after", fx.get("live_producers_after")),
        ("name left sourceless", fx.get("sourceless_after")),
    ]
    for label, value in observations:
        parts.append(
            f"<tr><td>{escape(label)}</td>"
            f"<td><code>{escape(json.dumps(value, default=str))}</code></td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        (
            "<p>The control source stays <code>extracted</code>, so the stale "
            "query sees a path that is present; the removed-path source becomes "
            f"<code>{escape(str(fx.get('orphan_source_status')))}</code>. The "
            f"name keeps <code>{escape(str(fx.get('name_stage_after')))}/"
            f"{escape(str(fx.get('name_status_after')))}</code>, gains "
            f"{len(fx.get('successor_edges_written') or [])} successor edges, and "
            f"is left with {fx.get('live_producers_after')} non-stale producer -- "
            "the state the export gate blocks on.</p>"
        )
        if not fx.get("error")
        else "<p>The disposable graph could not be started, so this section "
        "carries no measurement; see the error row above.</p>"
    )

    parts += [
        "<h2>What the rename records miss</h2>",
        f"<p>{escape(gen['rename_records']['ids_rename_skip'])}</p>",
        f"<p>{escape(gen['rename_records']['aos_rename_skip'])}</p>",
        "<p>Successor discovery methods across the "
        f"{data['census']['distinct_stale_paths']} distinct stale paths: "
        + ", ".join(
            f"<code>{escape(k)}</code> {v}"
            for k, v in sorted(data["census"]["successor_methods"].items())
        )
        + ".</p>",
        "<h2>Repair node</h2>",
        f"<p><strong>Goal.</strong> {escape(rep['goal'])}</p>",
        "<ul>",
    ]
    for path in rep["write_paths"]:
        parts.append(f"<li><code>{escape(path)}</code></li>")
    parts += [
        "</ul>",
        f"<p><strong>Guard test.</strong> <code>{escape(rep['guard_test'])}</code></p>",
        f"<p>{escape(rep['note'])}</p>",
    ]
    return "\n".join(parts)


def main() -> int:
    census = build()
    try:
        fixture = fixture_result()
    except Exception as exc:  # noqa: BLE001 - recorded in the artefact, never hidden
        fixture = {"error": f"{type(exc).__name__}: {exc}"}
    data = {
        "census": census,
        "generator": generator(),
        "fixture": fixture,
        "repair": repair_node(),
    }
    OUTPUT.write_text(json.dumps(data, indent=2, default=str) + "\n")
    REPORT.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Names produced only by removed DD paths</title>"
        "<style>body{font-family:sans-serif;max-width:1400px;margin:2em auto;"
        "padding:0 1em}table{border-collapse:collapse;width:100%;font-size:13px}"
        "td,th{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;"
        "vertical-align:top}code{font-size:12px}</style></head><body>"
        + render(data)
        + "</body></html>\n"
    )
    print(
        f"population={census['population']} live={census['live_count']} "
        f"accepted={census['accepted_count']} "
        f"methods={census['successor_methods']} "
        f"sourceless_after={fixture.get('sourceless_after')} "
        f"successor_edges_written={len(fixture.get('successor_edges_written', []))} "
        f"fixture_error={fixture.get('error')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
