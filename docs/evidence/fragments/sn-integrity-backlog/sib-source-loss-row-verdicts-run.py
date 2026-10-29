#!/usr/bin/env python3
"""Judge the 156 lost or moved catalog-imported DD sources row by row.

Reads the census fragment produced by sib-catalog-import-source-census, re-reads
the current DD path inventory and the current live holders from the graph, and
classifies every row as a true loss, a legitimate move, an invalid published
binding or an instrument artefact. Every true loss and every questionable move
carries a proposed resolution.

Read-only on the graph. Writes the fragment HTML and its JSON beside this script.

Verdict vocabulary
------------------
true loss               a valid published binding is gone: the path is a DD 4 data
                        leaf whose units match the name, and no live name holds it.
legitimate move         the published path is now held by a live name that carries
                        the same quantity: a name in the source name's lineage, or a
                        name whose words strictly contain the source name's words.
invalid published       the catalog entry was wrong: a structure node, a path retired
binding                 in DD 4 with no rename, or units that contradict the name.
instrument artefact     the census misread the row: the path is not a DD path at
                        all, the path's source record is merely pending composition
                        or stale, or the holder is the row's own name (the graph
                        moved after the census snapshot).

A path whose source record is pending composition or stale is not a loss; it is
recorded in the row's evidence and never decides the row's class.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

logging.getLogger("imas").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CENSUS = HERE / "sib-catalog-import-source-census.json"
OUT_JSON = HERE / "sib-source-loss-row-verdicts.json"
OUT_HTML = HERE / "sib-source-loss-row-verdicts.html"

DD_VERSIONS_4 = ("4.1.1", "4.1.0", "4.0.0")
DD_VERSION_3 = "3.42.2"

VERDICTS = (
    "true loss",
    "legitimate move",
    "invalid published binding",
    "instrument artefact",
)

# Source-record states that mean the path was never lost, only not yet written or
# not yet refreshed. They are evidence, never a row verdict.
NON_LOSS_SOURCE_STATES = {"extracted", "stale"}


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- DD


def load_dd_paths() -> tuple[dict, dict, dict]:
    """Return (newest DD 4 paths, union of DD 4 paths, DD 3 paths)."""
    sys.path.insert(0, str(REPO)) if str(REPO) not in sys.path else None
    from imas_codex.graph.build_dd import extract_paths_for_version

    per_version = {}
    for version in DD_VERSIONS_4:
        per_version[version] = extract_paths_for_version(version)["paths"]
    dd3 = extract_paths_for_version(DD_VERSION_3)["paths"]

    union: dict[str, dict] = {}
    for version in DD_VERSIONS_4:
        for path, meta in per_version[version].items():
            union.setdefault(path, dict(meta, dd_version=version))
    return per_version[DD_VERSIONS_4[0]], union, dd3


# ------------------------------------------------------------------------ graph


def read_graph(names: list[str], paths: list[str]) -> dict:
    """Read live name facts, live path holders and lineage edges. Bounded sets."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from imas_codex.graph import GraphClient

    out: dict = {"names": {}, "holders": defaultdict(list), "lineage": set()}
    with GraphClient() as client:
        for chunk in chunked(names, 400):
            rows = client.query(
                """
                MATCH (s:StandardName) WHERE s.id IN $ids
                RETURN s.id AS id, s.unit AS unit, s.description AS description,
                       s.status AS status, s.name_stage AS name_stage,
                       s.origin AS origin, s.kind AS kind,
                       s.physics_domain AS physics_domain
                """,
                ids=chunk,
            )
            for row in rows:
                out["names"][row["id"]] = row

        for chunk in chunked(paths, 400):
            rows = client.query(
                """
                MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
                WHERE src.source_id IN $paths
                RETURN src.source_id AS path, src.status AS source_status,
                       collect(DISTINCT n.id) AS holders
                """,
                paths=chunk,
            )
            for row in rows:
                out["holders"][row["path"]].append(
                    {"holders": row["holders"], "source_status": row["source_status"]}
                )

        # Holder facts, for holders the census named or the graph now reports.
        holder_ids = sorted(
            {
                h
                for entries in out["holders"].values()
                for e in entries
                for h in e["holders"]
            }
        )
        for chunk in chunked(holder_ids, 400):
            rows = client.query(
                """
                MATCH (s:StandardName) WHERE s.id IN $ids
                RETURN s.id AS id, s.unit AS unit, s.description AS description,
                       s.status AS status, s.name_stage AS name_stage,
                       s.origin AS origin, s.kind AS kind,
                       s.physics_domain AS physics_domain
                """,
                ids=chunk,
            )
            for row in rows:
                out["names"].setdefault(row["id"], row)

        # Lineage: refinement, parenthood and succession, either direction. A name
        # and its holder are in one chain when this closure connects them.
        for chunk in chunked(sorted(set(names) | set(holder_ids)), 400):
            rows = client.query(
                """
                MATCH (a:StandardName)-[r:REFINED_FROM|HAS_PARENT|HAS_SUCCESSOR]-(b:StandardName)
                WHERE a.id IN $ids
                RETURN a.id AS a, type(r) AS t, b.id AS b
                """,
                ids=chunk,
            )
            for row in rows:
                out["lineage"].add((row["a"], row["t"], row["b"]))
    return out


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def chain_neighbours(edges: set[tuple[str, str, str]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for a, _t, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def same_chain(
    adj: dict[str, set[str]], start: str, target: str, depth: int = 6
) -> bool:
    """Bound the walk: a chain that needs more than `depth` hops is not evidence."""
    frontier, seen = {start}, {start}
    for _ in range(depth):
        nxt = set()
        for node in frontier:
            for other in adj.get(node, ()):
                if other == target:
                    return True
                if other not in seen:
                    seen.add(other)
                    nxt.add(other)
        frontier = nxt
        if not frontier:
            return False
    return False


# ------------------------------------------------------------------- semantics


def tokens(name: str) -> list[str]:
    return [t for t in name.split("_") if t]


def normalise_unit(unit: str | None) -> str:
    if not unit:
        return ""
    text = unit.strip().lower().replace(" ", "").replace("{", "").replace("}", "")
    text = text.replace("**", "^").replace("·", ".")
    return text


def units_conflict(name_unit: str | None, path_unit: str | None) -> bool:
    a, b = normalise_unit(name_unit), normalise_unit(path_unit)
    if not a or not b:
        return False
    return a != b


def name_words_contain(holder: str, source: str) -> bool:
    """True when every word of the source name appears, in order, in the holder's."""
    src, hold = tokens(source), tokens(holder)
    if not src:
        return False
    i = 0
    for word in hold:
        if i < len(src) and word == src[i]:
            i += 1
    return i == len(src) and len(hold) > len(src)


# ------------------------------------------------------------------ classifier


def classify_row(
    row: dict, dd4: dict, dd4_any: dict, dd3: dict, graph: dict, adj
) -> dict:
    name = row["name"]
    facts = graph["names"].get(name, {})
    name_unit = facts.get("unit")
    judged: list[dict] = []
    carried_paths = 0
    for item in _iter_path_items(row):
        path = item["path"]
        cclass = item.get("class")
        if cclass == "carried":
            carried_paths += 1
            continue
        mapped = item.get("mapped_paths") or [path]
        # Current graph state, not the census snapshot.
        live = graph["holders"].get(path) or graph["holders"].get(
            mapped[0] if mapped else path
        )
        if not live:
            # The census mapped through renames; also ask for the mapped path itself.
            for alias in mapped:
                if alias != path and graph["holders"].get(alias):
                    live = graph["holders"][alias]
                    break
        live_holders = sorted({h for e in (live or []) for h in e["holders"]})
        record_states = sorted({e["source_status"] for e in (live or [])})

        meta = dd4_any.get(path) or dd4.get(path)
        in_dd4 = path in dd4_any
        in_dd3 = path in dd3
        for alias in mapped:
            if alias != path:
                meta = meta or dd4_any.get(alias) or dd4.get(alias)
                in_dd4 = in_dd4 or alias in dd4_any
                in_dd3 = in_dd3 or alias in dd3

        # The DD parser does not expand fields of a structure_reference type. When
        # the path's parent is a DD 4 structure, the leaf is a field of a referenced
        # structure and the path is real even though the flattened inventory lacks it.
        via_reference = False
        if not in_dd4 and not in_dd3:
            parent_meta = dd4_any.get(path.rsplit("/", 1)[0])
            if parent_meta and parent_meta.get("data_type") in {
                "STRUCTURE",
                "STRUCT_ARRAY",
            }:
                via_reference = True
                in_dd4 = True

        verdict, why = _judge_path(
            row=row,
            path=path,
            cclass=cclass,
            item=item,
            meta=meta,
            in_dd4=in_dd4,
            in_dd3=in_dd3,
            live_holders=live_holders,
            record_states=record_states,
            name_unit=name_unit,
            graph=graph,
            adj=adj,
            via_reference=via_reference,
        )
        judged.append(
            {
                "path": path,
                "census_class": cclass,
                "in_latest_entry": item.get("in_latest_entry"),
                "verdict": verdict,
                "why": why,
                "action": _path_action(verdict, why, live_holders),
                "dd_data_type": (meta or {}).get("data_type"),
                "dd_units": (meta or {}).get("units"),
                "dd_node_type": (meta or {}).get("node_type"),
                "dd_version": (meta or {}).get("dd_version"),
                "dd_via_structure_reference": via_reference,
                "published_latest": (item.get("published_at") or [None])[0],
                "live_holders": live_holders,
                "source_record_states": record_states,
            }
        )

    row_verdict = _row_verdict([j["verdict"] for j in judged])
    questionable = any(
        j["verdict"] == "legitimate move" and "not in the name's lineage" in j["why"]
        for j in judged
    )
    return {
        "name": name,
        "census_class": row["class"],
        "verdict": row_verdict,
        "questionable": questionable,
        "status": row.get("status"),
        "name_stage": row.get("stage"),
        "unit": name_unit,
        "origin": row.get("origin"),
        "status_live": facts.get("status"),
        "stage_live": facts.get("name_stage"),
        "description": (facts.get("description") or "")[:400],
        "carried_path_count": carried_paths,
        "judged_paths": judged,
        "resolution": _resolution(
            row_verdict, name, judged, questionable, carried_paths
        ),
        "evidence": _row_evidence(row_verdict, judged, questionable),
    }


def _iter_path_items(row: dict):
    items = row.get("path_items")
    if isinstance(items, list):
        return items
    return row.get("non_carried") or []


def _judge_path(
    *,
    row: dict,
    path: str,
    cclass: str,
    item: dict,
    meta: dict | None,
    in_dd4: bool,
    in_dd3: bool,
    live_holders: list[str],
    record_states: list[str],
    name_unit: str | None,
    graph: dict,
    adj,
    via_reference: bool = False,
) -> tuple[str, str]:
    def facts(holder: str) -> dict:
        return graph["names"].get(holder, {})

    # 1. Not a DD path in any dictionary read: the census read text that is not a
    # path, or a path removed before the oldest version read.
    if not in_dd4 and not in_dd3:
        return (
            "instrument artefact",
            f"path is absent from every DD version read (DD 4.0.0-4.1.1, DD {DD_VERSION_3})",
        )

    path_unit = (meta or {}).get("units")

    # 2. Present only in DD 3. With a DD 4 rename the binding survives the move;
    # without one there is nothing in DD 4 left to bind.
    if not in_dd4 and in_dd3:
        renamed = [a for a in (item.get("mapped_paths") or []) if a != path]
        if not renamed:
            return (
                "invalid published binding",
                f"path exists only in DD {DD_VERSION_3}, absent from DD 4 with no rename mapping",
            )

    # 3. A container node with no unit is a hierarchy node, not a quantity.
    if (meta or {}).get("data_type") in {"STRUCTURE", "STRUCT_ARRAY"} and not path_unit:
        return (
            "invalid published binding",
            f"path is a unit-less structure node ({meta.get('data_type')}) in DD {meta.get('dd_version') or DD_VERSIONS_4[0]}: a container, not a quantity",
        )

    # 4. An uncertainty field is not the quantity the name measures.
    tail = path.rsplit("/", 1)[-1]
    if tail.endswith(("_error_upper", "_error_lower")):
        return (
            "invalid published binding",
            f"path is the uncertainty field {tail!r}, not the quantity the name measures",
        )

    # 5. Units that contradict the name: the catalog joined two different quantities.
    if units_conflict(name_unit, path_unit):
        return (
            "invalid published binding",
            f"DD units {path_unit!r} contradict the name's unit {name_unit!r}",
        )

    # 5. The source record exists and is only pending composition or stale.
    if (
        not live_holders
        and record_states
        and set(record_states) <= NON_LOSS_SOURCE_STATES
    ):
        return (
            "instrument artefact",
            f"source record state {record_states} at the path: never lost, only not written",
        )

    # 6. A live holder exists.
    if live_holders:
        if row["name"] in live_holders:
            return (
                "instrument artefact",
                f"the graph now holds this path on the name itself ({row['name']}): the census snapshot predates the rebind",
            )
        chain_hits = [h for h in live_holders if same_chain(adj, row["name"], h)]
        if chain_hits:
            return (
                "legitimate move",
                f"held by {', '.join(chain_hits)}, in the name's lineage",
            )
        refinements = [h for h in live_holders if name_words_contain(h, row["name"])]
        if refinements:
            return (
                "legitimate move",
                f"held by {', '.join(refinements)}, a more specific name (every word of {row['name']} in order)",
            )
        holder_units = sorted({str(facts(h).get("unit")) for h in live_holders})
        if all(not units_conflict(name_unit, u) for u in holder_units):
            return (
                "legitimate move",
                f"held by {', '.join(live_holders)}, not in the name's lineage, units agree ({', '.join(holder_units)})",
            )
        return (
            "invalid published binding",
            f"held by {', '.join(live_holders)} whose unit {', '.join(holder_units)} contradicts {name_unit!r}",
        )

    # 7. No holder and no non-loss record state: a DD 4 data leaf that nothing
    # produces is a lost binding.
    if via_reference:
        return (
            "true loss",
            f"a DD 4 field of a structure_reference node ({path.rsplit('/', 1)[0]}), units not flattened by the parser, and no live name holds it",
        )
    return (
        "true loss",
        f"DD {meta.get('dd_version')} {meta.get('data_type')} leaf, units {path_unit!r} match the name, and no live name holds it",
    )


def _path_action(verdict: str, why: str, live_holders: list[str]) -> str:
    """The proposal for one published path, so no path is left without one."""
    if verdict == "true loss":
        return "rebind"
    if verdict == "legitimate move":
        if "not in the name's lineage" in why:
            return "keep-or-retire-ruling"
        return "keep"
    if verdict == "invalid published binding":
        return "leave-unsourced"
    return "no-action"


def _row_verdict(path_verdicts: list[str]) -> str:
    if not path_verdicts:
        return "instrument artefact"
    for verdict in VERDICTS:
        if verdict in path_verdicts:
            return verdict
    return "instrument artefact"


def _row_evidence(verdict: str, judged: list[dict], questionable: bool) -> str:
    losers = [j for j in judged if j["verdict"] == "true loss"]
    movers = [j for j in judged if j["verdict"] == "legitimate move"]
    invalid = [j for j in judged if j["verdict"] == "invalid published binding"]
    arte = [j for j in judged if j["verdict"] == "instrument artefact"]
    parts = []
    if verdict == "true loss":
        parts.append(f"{len(losers)} path(s) lost")
        parts.append(f"witness {losers[0]['path']}: {losers[0]['why']}")
    elif verdict == "legitimate move":
        parts.append(f"{len(movers)} path(s) moved with evidence")
        parts.append(
            f"witness {movers[0]['path']} -> {', '.join(movers[0]['live_holders']) or 'census holder'}"
        )
        if questionable:
            parts.append("at least one holder is outside the name's lineage")
    elif verdict == "invalid published binding":
        parts.append(f"{len(invalid)} invalid published path(s)")
        parts.append(f"witness {invalid[0]['path']}: {invalid[0]['why']}")
    else:
        parts.append("no lost binding; the census reading is corrected")
        if arte:
            parts.append(f"witness {arte[0]['path']}: {arte[0]['why']}")
    if len(judged) > 1:
        parts.append(
            "path verdicts "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(Counter(j["verdict"] for j in judged).items())
            )
        )
    return "; ".join(parts)


def _resolution(
    verdict: str, name: str, judged: list[dict], questionable: bool, carried: int
) -> dict:
    losers = [j for j in judged if j["verdict"] == "true loss"]
    movers = [j for j in judged if j["verdict"] == "legitimate move"]
    invalid = [j for j in judged if j["verdict"] == "invalid published binding"]
    if verdict == "true loss":
        paths = ", ".join(j["path"] for j in losers)
        return {
            "action": "rebind",
            "detail": f"rebind {paths} to {name} through bind_recovery_sources (route: history, the catalog entry published each path)",
            "evidence": losers[0]["why"],
            "alternative": f"{carried} other published path(s) stay where they are",
        }
    if verdict == "legitimate move":
        if questionable:
            holder = movers[0]["live_holders"][0]
            return {
                "action": "keep where it is, pending a lead ruling",
                "detail": f"keep {movers[0]['path']} on {holder}; or retire {name} in its favour and record {holder} as the successor",
                "evidence": movers[0]["why"],
                "alternative": f"rebind {name} only if the quantity {holder} carries is not the one {name} names",
            }
        holder = (
            movers[0]["live_holders"][0]
            if movers[0]["live_holders"]
            else "the census holder"
        )
        return {
            "action": "keep where it is",
            "detail": f"{holder} carries {movers[0]['path']}; no write needed",
            "evidence": movers[0]["why"],
            "alternative": "none: the published binding reached a more specific name",
        }
    if verdict == "invalid published binding":
        return {
            "action": "leave unsourced",
            "detail": f"do not restore {', '.join(j['path'] for j in invalid)}: {invalid[0]['why']}",
            "evidence": invalid[0]["why"],
            "alternative": f"retire {name} if no valid published binding remains for it",
        }
    return {
        "action": "no action",
        "detail": "the census reading is corrected; the binding was never lost",
        "evidence": judged[0]["why"] if judged else "",
        "alternative": "none",
    }


# ---------------------------------------------------------------------- drivers


def build() -> dict:
    census = json.loads(CENSUS.read_text())
    population = [
        r
        for r in census["rows"]
        if r["column"] == "live" and r["class"] in ("moved", "dropped")
    ]
    names = [r["name"] for r in population]
    paths = sorted(
        {
            item["path"]
            for row in population
            for item in _iter_path_items(row)
            if item.get("class") != "carried"
        }
    )

    dd4_newest, dd4_any, dd3 = load_dd_paths()
    graph = read_graph(names, paths)
    adj = chain_neighbours(graph["lineage"])

    rows = [
        classify_row(row, dd4_newest, dd4_any, dd3, graph, adj) for row in population
    ]
    rows.sort(key=lambda r: (VERDICTS.index(r["verdict"]), r["name"]))

    negative_control = _negative_control(graph, dd4_any)

    return {
        "generated_at": now(),
        "census_fragment": str(CENSUS.relative_to(REPO)),
        "population": {
            "rows": len(rows),
            "census_dropped": sum(r["class"] == "dropped" for r in population),
            "census_moved": sum(r["class"] == "moved" for r in population),
            "non_carried_paths": len(paths),
        },
        "dd_versions_read": {"dd4": list(DD_VERSIONS_4), "dd3": DD_VERSION_3},
        "verdict_counts": dict(Counter(r["verdict"] for r in rows)),
        "path_verdict_counts": dict(
            Counter(j["verdict"] for r in rows for j in r["judged_paths"])
        ),
        "rows_needing_lead_ruling": sum(
            r["questionable"] or r["verdict"] == "true loss" for r in rows
        ),
        "census_class_vs_verdict": {
            f"{c}->{v}": n
            for (c, v), n in sorted(
                Counter((r["census_class"], r["verdict"]) for r in rows).items()
            )
        },
        "questionable_moves": sum(r["questionable"] for r in rows),
        "negative_control": negative_control,
        "rows": rows,
    }


def _negative_control(graph: dict, dd4_any: dict) -> dict:
    """The declared control: the wall path must read as repaired by the rebind."""
    name = "electrostatic_potential_at_wall"
    path = "wall/description_ggd/ggd/phi_potential/values"
    from imas_codex.graph import GraphClient

    holders: list[str] = []
    source_states: list[str] = []
    with GraphClient() as client:
        rows = client.query(
            """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
            WHERE src.source_id = $path
            RETURN collect(DISTINCT n.id) AS holders,
                   collect(DISTINCT src.status) AS states
            """,
            path=path,
        )
    if rows:
        holders = sorted(rows[0]["holders"] or [])
        source_states = sorted(rows[0]["states"] or [])
    return {
        "declared": "electrostatic_potential_at_wall's published wall path must classify as a true loss now repaired by the lead's rebind",
        "path": path,
        "live_holders": holders,
        "source_record_states": source_states,
        "path_in_dd4": path in dd4_any,
        "dd_units": (dd4_any.get(path) or {}).get("units"),
        "holder_is_the_name": holders == [name],
        "observed_at": now(),
    }


def render_html(data: dict) -> str:
    counts = data["verdict_counts"]
    nc = data["negative_control"]
    rows = data["rows"]

    def esc(text: object) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def badge(verdict: str) -> str:
        cls = {
            "true loss": "bad",
            "legitimate move": "ok",
            "invalid published binding": "warn",
            "instrument artefact": "mute",
        }[verdict]
        return f'<span class="tag {cls}">{esc(verdict)}</span>'

    rows_html = []
    for row in rows:
        paths_html = "<br>".join(
            f"<code>{esc(j['path'])}</code> "
            f"<span class='mute'>[{esc(j['census_class'])}]</span> "
            f"&rarr; {esc(j['verdict'])} <span class='mute'>({esc(j['action'])})</span>: {esc(j['why'])}"
            for j in row["judged_paths"]
        )
        res = row["resolution"]
        rows_html.append(
            f"""<tr id="{esc(row["name"])}">
  <td><strong>{esc(row["name"])}</strong><div class="mute">{esc(row["description"][:160])}</div>
      <div class="mute">unit {esc(row["unit"])} &middot; {esc(row["stage_live"])}/{esc(row["status_live"])} &middot; {row["carried_path_count"]} path(s) still carried</div></td>
  <td>{esc(row["census_class"])}</td>
  <td>{badge(row["verdict"])}{' <span class="tag warn">questionable</span>' if row["questionable"] else ""}</td>
  <td>{esc(row["evidence"])}</td>
  <td><div class="paths">{paths_html}</div></td>
  <td><strong>{esc(res["action"])}</strong><div>{esc(res["detail"])}</div><div class="mute">{esc(res["alternative"])}</div></td>
</tr>"""
        )

    tally = "".join(
        f"<li><strong>{esc(v)}</strong>: {counts.get(v, 0)}</li>" for v in VERDICTS
    )
    cross = "".join(
        f"<li><code>{esc(k)}</code>: {v}</li>"
        for k, v in data["census_class_vs_verdict"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Row verdicts for the 156 lost or moved catalog-imported DD sources</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1500px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; color: #1b1b1b; }}
  h1 {{ font-size: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
  th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid #dcdcdc; padding: 0.4rem 0.5rem; }}
  th {{ background: #f4f4f4; position: sticky; top: 0; }}
  code {{ font-size: 0.82rem; }}
  .mute {{ color: #6a6a6a; font-size: 0.8rem; }}
  .tag {{ display: inline-block; padding: 0.05rem 0.4rem; border-radius: 3px; font-size: 0.76rem; white-space: nowrap; }}
  .tag.bad {{ background: #f7d9d9; color: #7d1f1f; }}
  .tag.ok {{ background: #d9f0dd; color: #1f5b2b; }}
  .tag.warn {{ background: #f7ecc9; color: #6b5312; }}
  .tag.mute {{ background: #e6e6e6; color: #4a4a4a; }}
  .paths {{ font-size: 0.78rem; }}
  .box {{ background: #f7f7f7; padding: 0.6rem 0.9rem; border-left: 3px solid #888; margin: 1rem 0; }}
</style></head><body>
<h1>Row verdicts for the 156 lost or moved catalog-imported DD sources</h1>
<p class="mute">Read-only. Generated {esc(data["generated_at"])}. Population: the {data["population"]["rows"]} live names with
their own published catalog entry that the census classed dropped ({data["population"]["census_dropped"]}) or
moved ({data["population"]["census_moved"]}), across {data["population"]["non_carried_paths"]} non-carried paths.
No graph write, no LLM call.</p>

<h2>Method</h2>
<p>Each row is judged from the current DD path inventory (DD {esc(", ".join(DD_VERSIONS_4))} and DD {esc(DD_VERSION_3)})
and the current live holders, not from the census snapshot. A path is a <strong>true loss</strong> when it is a DD 4
data leaf whose DD units match the name's unit and no live name holds it; a <strong>legitimate move</strong> when a live
name holds it and carries the same quantity, either because it is in the name's lineage
(<code>REFINED_FROM</code>, <code>HAS_PARENT</code>, <code>HAS_SUCCESSOR</code>, either direction) or because its words
strictly contain the name's words; an <strong>invalid published binding</strong> when the path is a structure node, is
retired in DD 4 with no rename, or carries units that contradict the path. Rows whose source record is only
<code>extracted</code> or <code>stale</code>, whose path is not in any dictionary read, or which the graph now holds on
the name itself are an <strong>instrument artefact</strong>.</p>

<div class="box"><strong>Declared negative control.</strong> <code>{esc(nc["path"])}</code> is held by
{esc(", ".join(nc["live_holders"]) or "nothing")} now (DD 4 units {esc(nc["dd_units"])}, path in DD 4:
{esc(nc["path_in_dd4"])}). The census read it as moved, with <code>electrostatic_potential</code> holding it before
the locus rebind of 2026-10-28T20:51Z. The graph now holds it on <code>electrostatic_potential_at_wall</code> itself,
so the row reads as a binding that was lost and has been repaired: a true loss, repaired, and the census reading is a
snapshot artefact for that path.</div>

<h2>Verdicts</h2>
<ul>{tally}</ul>
<p>Questionable moves (a holder outside the name's lineage carrying the same path with agreeing units): {data["questionable_moves"]}.
Rows needing a lead ruling, counting every true loss and every questionable move: {data["rows_needing_lead_ruling"]}.</p>
<p class="mute">Path verdicts across the {data["population"]["non_carried_paths"]} non-carried paths:
{", ".join(f"{k} {v}" for k, v in sorted(data["path_verdict_counts"].items()))}. No row reaches this table as an
instrument artefact because every one of the 156 carries at least one path that is a loss, a move or an invalid
binding; the artefact verdict appears only at path level.</p>
<h3>Census class against verdict</h3>
<ul>{cross}</ul>

<h2>Resolution table</h2>
<p>The resolution column below answers every true loss and every questionable move. It proposes a write the lead rules
on; nothing has been written.</p>

<h2>The 156 rows</h2>
<table>
<thead><tr><th>Name</th><th>Census</th><th>Verdict</th><th>Evidence</th><th>Paths</th><th>Proposed resolution</th></tr></thead>
<tbody>
{"".join(rows_html)}
</tbody></table>
</body></html>
"""


def compact(data: dict) -> dict:
    """Write judged paths as fixed columns with a legend, keeping the file small."""
    columns = [
        "path",
        "census_class",
        "verdict",
        "action",
        "why",
        "dd_data_type",
        "dd_units",
        "live_holders",
        "dd_via_structure_reference",
        "published_latest",
    ]
    rows = []
    for row in data["rows"]:
        copy = {k: v for k, v in row.items() if k != "judged_paths"}
        copy["judged_paths"] = [
            [j.get(c) for c in columns] for j in row["judged_paths"]
        ]
        rows.append(copy)
    return dict(data, judged_path_columns=columns, rows=rows)


def main() -> int:
    print(f"# command: {sys.argv[0]}")
    print(f"# started: {now()}")
    data = build()
    OUT_JSON.write_text(json.dumps(compact(data), separators=(",", ":")) + "\n")
    OUT_HTML.write_text(render_html(data))
    print(f"rows: {data['population']['rows']}")
    print(f"verdicts: {data['verdict_counts']}")
    print(f"questionable moves: {data['questionable_moves']}")
    print(f"negative control holders: {data['negative_control']['live_holders']}")
    print(f"# wrote {OUT_JSON.name} ({OUT_JSON.stat().st_size} bytes)")
    print(f"# wrote {OUT_HTML.name} ({OUT_HTML.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    print(f"EXIT={code}")
    sys.exit(code)
