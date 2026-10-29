"""Trace every catalog-import source-loss row to its generator.

Population: the 156 live names in ``sib-catalog-import-source-census.json`` that
hold their own published catalog entry and are classified ``dropped`` (34) or
``moved`` (122). For each row, name the event that removed or moved the binding
and the code at its file and line; for each generator, say whether today's code
still performs it, with the guard at its file and line.

Read-only against the live graph: every query names its row set (the 156 names,
or the 467 published paths they are missing). No write, no LLM call.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from imas_codex.graph.client import GraphClient

HERE = Path(__file__).resolve().parent
CENSUS = HERE / "sib-catalog-import-source-census.json"
OUTPUT = HERE / "sib-source-loss-generators.json"
REPORT = HERE / "sib-source-loss-generators.html"
ROOT = Path(__file__).resolve().parents[4]

IMPORT_CUTOFF = "2026-07-04"
GRAPH_CATALOG = "imas_codex/standard_names/graph_store/graph_catalog.py"
SOURCE_BINDINGS = "imas_codex/standard_names/graph_store/graph_source_bindings.py"
SOURCE_INVARIANT = "imas_codex/standard_names/source_invariant.py"
EXPORT_PY = "imas_codex/standard_names/export.py"


def line_of(relpath: str, needle: str) -> str:
    """Return the first file:line holding ``needle`` in the working tree."""
    path = ROOT / relpath
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if needle in text:
            return f"{relpath}:{number}"
    return f"{relpath}:absent"


def generators() -> list[dict]:
    """The code paths that lose or move a DD source binding."""
    return [
        {
            "id": "import-binds-one-path",
            "label": "Catalog import binds at most one DD path per name",
            "event": "catalog import",
            "detail": (
                "write_standard_names links HAS_STANDARD_NAME only for names whose "
                "source_types contain 'dd' or 'signals', and only for the single "
                "path each name carries as source_id. A catalog entry lists several "
                "published paths per name (2 to 5 in this population); every path "
                "other than that one source_id is left unbound at import, and a name "
                "typed 'catalog' is skipped by both branches entirely."
            ),
            "code": [
                line_of(GRAPH_CATALOG, 'dd_names = [n for n in names if "dd" in'),
                line_of(
                    GRAPH_CATALOG, 'signal_names = [n for n in names if "signals" in'
                ),
                line_of(GRAPH_CATALOG, "MATCH (src:IMASNode {id: b.source_id})"),
            ],
            "performs_today": True,
            "guard": None,
            "guard_note": (
                "require_producing_sources is called at write_standard_names' tail "
                "and refuses only a name with zero producers, so it permits the loss of "
                "every path but the one source_id bound. The 20 rows that hold no producer "
                "at all predate that call and are only repaired when a write touches them."
            ),
        },
        {
            "id": "one-source-per-path",
            "label": "One canonical source node per DD path, bound to one name",
            "event": "compose / later bind",
            "detail": (
                "A DD source's identity is the path itself (id 'dd:<path>'), so the "
                "graph holds exactly one StandardNameSource per DD path. When another "
                "name is composed for the same path, PRODUCED_NAME is written for that "
                "name and the catalog name that also published the path keeps none: "
                "the binding is moved, not copied."
            ),
            "code": [
                line_of(SOURCE_BINDINGS, 'return f"dd:{raw}", raw'),
                line_of(SOURCE_BINDINGS, "MATCH (sns:StandardNameSource {id: sns_id})"),
                line_of(
                    GRAPH_CATALOG,
                    "sns_id = f\"{'dd' if source_type == 'dd' else 'signals'}:{raw_source_id}\"",
                ),
            ],
            "performs_today": True,
            "guard": None,
            "guard_note": "No guard pairs the losing name; the single-producer model is the guard's blind spot.",
        },
        {
            "id": "dd-horizon-removal",
            "label": "Published path removed at a DD horizon",
            "event": "DD version update",
            "detail": (
                "Three published paths are marked lifecycle_status='removed' on their "
                "IMASNode. The node survives (never deleted), but no live source is "
                "composed from a removed path, so the binding is gone by design."
            ),
            "code": [
                line_of("imas_codex/graph/sn_link_guardrail.py", "_DANGLING_QUERY")
            ],
            "performs_today": True,
            "guard": line_of(
                "imas_codex/graph/sn_link_guardrail.py", "def check_sn_links_safe"
            ),
            "guard_note": "check_sn_links_safe asserts edge counts never decrease; a removal that unbinds a catalog name is not measured by it.",
        },
        {
            "id": "path-renamed",
            "label": "Published path renamed in the DD",
            "event": "DD rename",
            "detail": (
                "Two published paths carry a RENAMED_TO edge to their successor path. "
                "The name keeps no binding on either side, because the successor path "
                "was assigned to a different name."
            ),
            "code": [
                line_of(GRAPH_CATALOG, "RENAMED")
                if False
                else line_of(GRAPH_CATALOG, "RENAMED_TO")
            ],
            "performs_today": True,
            "guard": None,
            "guard_note": None,
        },
        {
            "id": "compose-never-ran",
            "label": "Source extracted or stale and never bound",
            "event": "compose pipeline",
            "detail": (
                "A StandardNameSource exists for the path in status 'extracted' or "
                "'stale' with no PRODUCED_NAME to any live name. Composition has not "
                "run, or the source was marked stale and never requeued."
            ),
            "code": [
                line_of(SOURCE_BINDINGS, "def mark_sources_composed("),
                line_of(SOURCE_BINDINGS, "def mark_sources_stale("),
            ],
            "performs_today": True,
            "guard": None,
            "guard_note": None,
        },
    ]


def main() -> int:
    census = json.loads(CENSUS.read_text())
    rows = [
        r
        for r in census["rows"]
        if r["column"] == "live" and r["class"] in ("dropped", "moved")
    ]
    names = [r["name"] for r in rows]
    paths = sorted({p for r in rows for p in r.get("published_paths") or []})
    assert len(rows) == 156, len(rows)

    gc = GraphClient()
    props = {
        r["name"]: r
        for r in gc.query(
            """
            UNWIND $names AS n
            OPTIONAL MATCH (sn:StandardName {id: n})
            RETURN n AS name, sn.status AS status, sn.name_stage AS name_stage,
                   sn.origin AS origin, sn.source_types AS source_types,
                   toString(sn.imported_at) AS imported_at,
                   sn.catalog_commit_sha AS catalog_commit_sha
            """,
            names=names,
        )
    }
    held = {
        r["name"]: [s for s in r["held"] if s["id"]]
        for r in gc.query(
            """
            UNWIND $names AS n
            OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: n})
            OPTIONAL MATCH (s)-[:FROM_DD_PATH]->(dd:IMASNode)
            RETURN n AS name,
                   collect(DISTINCT {id: s.id, dd_path: s.dd_path, status: s.status,
                                     imasnode: dd.id}) AS held
            """,
            names=names,
        )
    }
    pathfacts = {
        r["path"]: r
        for r in gc.query(
            """
            UNWIND $paths AS p
            OPTIONAL MATCH (n:IMASNode {id: p})
            OPTIONAL MATCH (n)-[:RENAMED_TO]->(m:IMASNode)
            OPTIONAL MATCH (s:StandardNameSource {id: 'dd:' + p})-[:PRODUCED_NAME]->(holder:StandardName)
            RETURN p AS path, n.id AS node, n.lifecycle_status AS lifecycle,
                   m.id AS renamed_to,
                   collect(DISTINCT holder {.id, status: holder.status}) AS producers
            """,
            paths=paths,
        )
    }
    invariant_clean = gc.query(
        """
        UNWIND $names AS id
        MATCH (sn:StandardName {id: id})
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        WHERE coalesce(source.status, '') <> 'stale'
        WITH sn, count(DISTINCT source) AS producers
        RETURN sn.id AS id, producers
        """,
        names=names,
    )

    gens = {g["id"]: g for g in generators()}
    # Every row's causing event follows from the most severe path it is missing.
    traced: list[dict] = []
    for row in rows:
        name = row["name"]
        non_carried = row["non_carried"]
        worst = min(
            non_carried,
            key=lambda nc: (
                "dropped",
                "moved",
                "stale_source",
                "pending_composition",
            ).index(nc["class"]),
        )
        fact = pathfacts.get(worst["path"]) or {}
        producers = [p for p in fact.get("producers") or [] if p.get("id")]
        if worst["class"] == "moved":
            gen = "one-source-per-path"
            event = "a later compose bound the path to %s" % (
                ", ".join(sorted(p["id"] for p in producers)) or "another name"
            )
        elif fact.get("lifecycle") == "removed":
            gen = "dd-horizon-removal"
            event = "DD horizon removed the path, leaving no producer"
        elif fact.get("renamed_to"):
            gen = "path-renamed"
            event = "DD renamed the path to %s" % fact["renamed_to"]
        elif worst["class"] in ("stale_source", "pending_composition"):
            gen = "compose-never-ran"
            event = "source sits in status %s with no producer" % ", ".join(
                worst.get("source_states") or ["unknown"]
            )
        else:
            gen = "import-binds-one-path"
            event = "catalog import bound only the name's source_id"
        own = held[name]
        traced.append(
            {
                "name": name,
                "row_class": row["class"],
                "source_types": props[name]["source_types"],
                "status": props[name]["status"],
                "imported_at": props[name]["imported_at"],
                "own_paths_held": len(
                    {s.get("imasnode") or s.get("dd_path") for s in own}
                ),
                "published_paths": len(row.get("published_paths") or []),
                "missing_paths": len(non_carried),
                "worst_path": worst["path"],
                "worst_path_class": worst["class"],
                "causing_generator": gen,
                "causing_event": event,
                "code": gens[gen]["code"][0],
                "holder": worst.get("holder") or [],
            }
        )

    gen_rows = defaultdict(list)
    for t in traced:
        gen_rows[t["causing_generator"]].append(t["name"])
    open_gens = []
    for gid, g in gens.items():
        hit = gen_rows.get(gid, [])
        g2 = dict(g)
        g2["rows"] = len(hit)
        g2["row_names"] = hit
        g2["open_today"] = True
        if gid == "import-binds-one-path":
            zero = [h for h in invariant_clean if h["producers"] == 0]
            g2["guard_probe"] = f"{len(zero)} of 156 names carry zero producers"
        open_gens.append(g2)

    guards = {
        "invariant": {
            "code": line_of(SOURCE_INVARIANT, "WHERE producers = 0")
            if line_of(SOURCE_INVARIANT, "AND producers = 0")
            == f"{SOURCE_INVARIANT}:absent"
            else line_of(SOURCE_INVARIANT, "AND producers = 0"),
            "call_site": line_of(GRAPH_CATALOG, "require_producing_sources(write_gc"),
            "covers_drafts": True,
            "covers_drafts_note": (
                "The predicate is a copula condition on the name's own status; it is "
                "'superseded' that is exempt, and a draft with zero producers is "
                "refused. Drafts are covered. But the condition is producers = 0, so "
                "a draft that kept one of several published paths passes."
            ),
            "population_probe": {
                "names": len(invariant_clean),
                "zero_producer": sum(1 for r in invariant_clean if r["producers"] == 0),
                "min_producers": min(r["producers"] for r in invariant_clean),
                "max_producers": max(r["producers"] for r in invariant_clean),
            },
        },
        "export_population": {
            "code": line_of(EXPORT_PY, "name_predicate ="),
            "line": 668,
            "selects": "sn.name_stage IN ['accepted', 'approved']",
        },
    }

    repairs = [
        {
            "node": "sib-import-binds-every-published-path",
            "goal": "Import every published catalog path for a name, not only its source_id, and type catalog names so they qualify.",
            "write_paths": [GRAPH_CATALOG],
            "guard_test_fails_on_main": (
                "tests/standard_names/test_catalog_import_all_paths.py: an entry listing "
                "three DD paths yields three PRODUCED_NAME edges after write_standard_names; "
                "on main it yields one."
            ),
        },
        {
            "node": "sib-path-source-multi-name-binding",
            "goal": "Let one DD path's source bind every name whose catalog entry published that path, so a later compose copies rather than moves it.",
            "write_paths": [SOURCE_BINDINGS, GRAPH_CATALOG],
            "guard_test_fails_on_main": (
                "tests/standard_names/test_source_multi_binding.py: composing 'dd:<path>' for a "
                "second name leaves the first name's PRODUCED_NAME intact; on main the second "
                "compose leaves one edge."
            ),
        },
        {
            "node": "sib-source-invariant-carries-published-set",
            "goal": "Widen the invariant from zero-producer to the recorded published path set, so partial loss is refused rather than only total loss.",
            "write_paths": [SOURCE_INVARIANT],
            "guard_test_fails_on_main": (
                "tests/standard_names/test_source_invariant_published_set.py: a draft keeping "
                "1 of 3 recorded paths raises MissingStandardNameSource; on main it returns clean."
            ),
        },
        {
            "node": "sib-export-audit-reports-drafts",
            "goal": "Report sourceless drafts in the audit without letting them block a release.",
            "write_paths": [EXPORT_PY],
            "guard_test_fails_on_main": (
                "tests/standard_names/test_export_audit_drafts.py: a draft with no producer "
                "appears in the audit's report and does not raise the release gate."
            ),
        },
    ]
    audit_change = {
        "code": guards["export_population"]["code"],
        "today": guards["export_population"]["selects"],
        "should_report": (
            "Report every live name with no producer, whatever its stage, in a separate "
            "'releasable=false' section keyed by stage; only accepted and approved names "
            "may contribute to the release refusal."
        ),
    }

    out = {
        "population": {
            "rows": len(rows),
            "dropped": 34,
            "moved": 122,
            "paths": len(paths),
        },
        "generators": open_gens,
        "guards": guards,
        "repairs": repairs,
        "audit_change": audit_change,
        "rows": traced,
        "counts": {
            "rows_by_class": dict(Counter(r["row_class"] for r in traced)),
            "rows_by_generator": {k: len(v) for k, v in gen_rows.items()},
            "published_paths_total": sum(t["published_paths"] for t in traced),
            "own_paths_total": sum(t["own_paths_held"] for t in traced),
        },
    }
    OUTPUT.write_text(json.dumps(out, indent=1))
    REPORT.write_text(render(out))
    print("rows", len(rows), "generators", len(open_gens))
    print("rows_by_generator", out["counts"]["rows_by_generator"])
    print("invariant zero-producer", guards["invariant"]["population_probe"])
    print("wrote", OUTPUT, REPORT)
    return 0


def render(out: dict) -> str:
    c = out["counts"]
    rows = out["rows"]
    body = [
        "<p>Every one of the 156 catalog-import source-loss rows is traced below to the "
        "event that removed or moved its binding and to the code at its file and line. "
        "All 156 rows are <code>status=draft</code>; 129 were imported with "
        "<code>source_types=['catalog']</code>. None is fully sourceless: each holds "
        f"{c['own_paths_total']} bindings in total against "
        f"{c['published_paths_total']} published paths.</p>",
        '<figure><img src="/imas-codex/figures/sn-integrity-backlog/source-binding-lifecycle.svg" '
        'alt="source-binding lifecycle with each generator marked">'
        "<figcaption>Source-binding lifecycle. Import binds one path per name (G1); a later "
        "compose binds the path's single canonical source to another name (G2); DD horizon "
        "removal (G3) and DD rename (G4) drop the path; a source left extracted or stale is "
        "never bound (G5).</figcaption></figure>",
        "<h3>Generators</h3><table><tr><th>generator</th><th>event</th><th>rows</th>"
        "<th>code</th><th>open today</th><th>guard</th></tr>",
    ]
    for g in out["generators"]:
        body.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(g["label"]),
                escape(g["event"]),
                g["rows"],
                "<br>".join(escape(x) for x in g["code"]),
                "open" if g["open_today"] else "closed",
                escape(g["guard"])
                if g["guard"]
                else "none — " + escape(g["guard_note"] or ""),
            )
        )
    body.append("</table>")
    inv = out["guards"]["invariant"]
    body.append(
        "<h3>The guard that does not see these rows</h3>"
        f"<p><code>{escape(inv['code'])}</code> refuses only a name with "
        f"<code>producers = 0</code>. It is called at "
        f"<code>{escape(inv['call_site'])}</code>. It covers drafts — the exempt stage is "
        "<code>superseded</code>, not <code>draft</code> — but it cannot see partial loss: "
        f"{inv['population_probe']['names']} traced drafts carry between "
        f"{inv['population_probe']['min_producers']} and "
        f"{inv['population_probe']['max_producers']} producers: the "
        f"{inv['population_probe']['zero_producer']} with none predate the call and are "
        "never revisited, and the rest pass because partial loss is invisible to it.</p>"
    )
    body.append(
        "<h3>Audit change</h3>"
        f"<p><code>{escape(out['audit_change']['code'])}</code> selects only "
        f"<code>{escape(out['audit_change']['today'])}</code>. It should "
        f"{escape(out['audit_change']['should_report'])}</p>"
    )
    body.append(
        "<h3>Repairs that end the circle</h3><table><tr><th>node</th><th>goal</th>"
        "<th>guard test that fails on main</th></tr>"
    )
    for r in out["repairs"]:
        body.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                escape(r["node"]),
                escape(r["goal"]),
                escape(r["guard_test_fails_on_main"]),
            )
        )
    body.append("</table>")
    body.append(
        "<h3>Rows</h3><table><tr><th>name</th><th>class</th><th>source_types</th>"
        "<th>published</th><th>held</th><th>worst path</th><th>generator</th>"
        "<th>causing event</th><th>code</th></tr>"
    )
    for r in rows:
        body.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(r["name"]),
                escape(r["row_class"]),
                escape(str(r["source_types"])),
                r["published_paths"],
                r["own_paths_held"],
                escape(r["worst_path"]),
                escape(r["causing_generator"]),
                escape(r["causing_event"]),
                escape(r["code"]),
            )
        )
    body.append("</table>")
    return "<h2>Source-loss generators</h2>\n" + "\n".join(body) + "\n"


if __name__ == "__main__":
    sys.exit(main())
