#!/usr/bin/env python3
"""Judge the 156 lost or moved catalog-imported DD sources row by row.

Reads the census fragment produced by sib-catalog-import-source-census, re-reads
the current DD path inventory, the StandardNameSource records and the live holders
from the graph, and classifies every row and every published path, with the evidence
each verdict rests on.

Read-only on the graph. Writes the fragment HTML and its JSON beside this script.

Verdict vocabulary
------------------
true loss               a published binding is gone and the path's own DD description
                        denotes the quantity the name names: no live holder, units
                        that match, and every content token of the name present in
                        the path's description. Only this verdict proposes a rebind.
unverified loss         no live holder and matching units, but the path's description
                        and its own segments do not carry the name's words, or the
                        path is not in the DD versions read. A rebind is NOT
                        proposed: the description comparison is quoted and the lead
                        rules on it.
legitimate move         the published path is held by a live name in the source
                        name's lineage, or by a more specific name on the same path.
questionable move       the path is held by a live name outside the name's lineage,
                        on the same path with agreeing units. The row carries the
                        one-sentence question the lead must answer.
invalid published       the catalog entry was wrong: units that contradict the name,
binding                 a unit-less container, an uncertainty field, a path retired
                        from DD 4 with no rename, or a description that names a
                        different subject.
pending composition     a DD StandardNameSource record exists for the path with
                        status 'extracted' and no PRODUCED_NAME edge: extraction
                        seeded it, compose has not run, and no published binding was
                        available to lose. Its resolution is to let compose run,
                        never a rebind.
excluded by design      the path has no StandardNameSource record at all and its
                        IMASNode.node_category is outside SN_SOURCE_CATEGORIES, so
                        extraction never admits it. The rule: a '/data' leaf under a
                        STRUCTURE parent classifies 'structural' (node_classifier.py
                        Rule 7, imas_codex/core/node_classifier.py:489), GGD
                        representation nodes classify 'representation', and
                        SN_SOURCE_CATEGORIES is quantity|geometry|coordinate
                        (imas_codex/core/node_categories.py:30).
stale source            a StandardNameSource exists with status 'stale': refreshed,
                        not restored.
instrument artefact     the path is not in the DD versions read and is not a DD node
                        in the graph either, so the census recorded a path no
                        dictionary has.

Path lists inherited from an ancestor's catalog entry are suspect by default and are
never proposed for rebind; they are recorded as unverified losses for the lead.
"""

from __future__ import annotations

import json
import logging
import random
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

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES  # noqa: E402

DD_VERSIONS_4 = ("4.1.1", "4.1.0", "4.0.0")
DD_VERSION_3 = "3.42.2"

SAMPLE_SEED = 20261029
SAMPLE_SIZE = 15

# The extraction gate, read from the code rather than restated from memory. The
# category set decides what may seed a StandardNameSource at all.
EXTRACTION_RULE = {
    "rule": "/data leaf under a STRUCTURE parent classifies 'structural', which the source gate refuses",
    "classified_at": "imas_codex/core/node_classifier.py:489 (Rule 7)",
    "gate": "imas_codex/standard_names/sources/dd.py:678",
    "categories_at": "imas_codex/core/node_categories.py:30",
}

PATH_VERDICTS = (
    "true loss",
    "unverified loss",
    "invalid published binding",
    "questionable move",
    "legitimate move",
    "pending composition",
    "excluded by design",
    "stale source",
    "instrument artefact",
)

RELATIONAL = {
    "of",
    "at",
    "due",
    "to",
    "in",
    "for",
    "per",
    "from",
    "between",
    "with",
    "and",
    "the",
}


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def cut(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- DD


def load_dd_paths() -> tuple[dict, dict, dict]:
    """Return (newest DD 4 paths, union of DD 4 paths, DD 3 paths)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from imas_codex.graph.build_dd import extract_paths_for_version

    per_version = {
        version: extract_paths_for_version(version)["paths"]
        for version in DD_VERSIONS_4
    }
    dd3 = extract_paths_for_version(DD_VERSION_3)["paths"]

    union: dict[str, dict] = {}
    for version in DD_VERSIONS_4:
        for path, meta in per_version[version].items():
            union.setdefault(path, dict(meta, dd_version=version))
    return per_version[DD_VERSIONS_4[0]], union, dd3


# ------------------------------------------------------------------------ graph


def chunked(items: list[str], size: int = 400):
    for i in range(0, len(items), size):
        yield items[i : i + size]


NAME_FACTS = """
MATCH (s:StandardName) WHERE s.id IN $ids
RETURN s.id AS id, s.unit AS unit, s.description AS description,
       s.status AS status, s.name_stage AS name_stage,
       s.origin AS origin, s.kind AS kind, s.physics_domain AS physics_domain
"""


def read_graph(names: list[str], paths: list[str]) -> dict:
    """Read name facts, every source record per path, live holders and lineage."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from imas_codex.graph import GraphClient

    out: dict = {
        "names": {},
        "sources": defaultdict(list),
        "holders": defaultdict(set),
        "lineage": set(),
        "nodes": {},
    }
    with GraphClient() as client:
        for chunk in chunked(names):
            for row in client.query(NAME_FACTS, ids=chunk):
                out["names"][row["id"]] = row

        for chunk in chunked(paths):
            rows = client.query(
                """
                MATCH (n:IMASNode) WHERE n.id IN $paths
                RETURN n.id AS path, n.node_category AS category, n.data_type AS data_type
                """,
                paths=chunk,
            )
            for row in rows:
                out["nodes"][row["path"]] = {
                    "node_category": row["category"],
                    "data_type": row["data_type"],
                }

        for chunk in chunked(paths):
            # Every source record for the path, whether or not a name holds it. A
            # record with no producer is pending composition, not a lost binding.
            rows = client.query(
                """
                MATCH (src:StandardNameSource)
                WHERE src.source_id IN $paths
                OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(n:StandardName)
                RETURN src.source_id AS path, src.source_type AS source_type,
                       src.status AS status, collect(DISTINCT n.id) AS holders
                """,
                paths=chunk,
            )
            for row in rows:
                records = out["sources"][row["path"]]
                records.append(
                    {
                        "source_type": row["source_type"],
                        "status": row["status"],
                        "holders": sorted(h for h in (row["holders"] or []) if h),
                    }
                )
                for holder in row["holders"] or []:
                    if holder:
                        out["holders"][row["path"]].add(holder)

        holder_ids = sorted({h for hs in out["holders"].values() for h in hs})
        for chunk in chunked(holder_ids):
            for row in client.query(NAME_FACTS, ids=chunk):
                out["names"].setdefault(row["id"], row)

        members = sorted(set(names) | set(holder_ids))
        for chunk in chunked(members):
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


def chain_adjacency(edges: set[tuple[str, str, str]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for a, _t, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def same_chain(
    adj: dict[str, set[str]], start: str, target: str, depth: int = 6
) -> bool:
    """Bound the walk: a chain needing more than `depth` hops is not evidence."""
    frontier, seen = {start}, {start}
    for _ in range(depth):
        nxt: set[str] = set()
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


def content_tokens(name: str) -> list[str]:
    """The words that state what the quantity is, dropping relational glue."""
    return [t for t in tokens(name) if t not in RELATIONAL]


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def doc_tokens(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text or "")
    return {singular(w) for w in cleaned.split()}


def token_present(token: str, doc: set[str]) -> bool:
    stem = singular(token.lower())
    if stem in doc:
        return True
    # A longer word that starts with the token still denotes it (density/densities,
    # current/currents, potential/potentials, coil/coils).
    return any(word.startswith(stem) and len(word) <= len(stem) + 3 for word in doc)


def description_match(
    name: str, path_doc: str, path: str = ""
) -> tuple[list[str], list[str]]:
    """Return (matched, missing) name content tokens against the path's own words.

    The path's DD description states the quantity; its segments state the subject
    the description inherits from its parent ('Temperature, given on various grid
    subsets' under `ggd/electrons/` is an electron temperature). Both are quoted in
    the evidence.
    """
    doc = doc_tokens(path_doc) | {singular(seg) for seg in path.split("/") if seg}
    matched = [t for t in content_tokens(name) if token_present(t, doc)]
    missing = [t for t in content_tokens(name) if t not in matched]
    return matched, missing


def normalise_unit(unit: str | None) -> str:
    if not unit:
        return ""
    text = unit.strip().lower().replace(" ", "").replace("{", "").replace("}", "")
    return text.replace("**", "^").replace("·", ".")


def units_conflict(name_unit: str | None, path_unit: str | None) -> bool:
    a, b = normalise_unit(name_unit), normalise_unit(path_unit)
    return bool(a and b and a != b)


def name_words_contain(holder: str, source: str) -> bool:
    """True when every word of the source name appears, in order, in the holder's."""
    src, hold = tokens(source), tokens(holder)
    i = 0
    for word in hold:
        if i < len(src) and word == src[i]:
            i += 1
    return bool(src) and i == len(src) and len(hold) > len(src)


# ------------------------------------------------------------------ classifier


def source_state(records: list[dict]) -> tuple[list[str], bool, list[str]]:
    """Return (record states, a DD record has no producer, live holders).

    Only a DD extraction record can be waiting on compose; a manual or derived
    source with no producer says nothing about whether the DD path was extracted.
    """
    states = sorted({r["status"] for r in records if r.get("status")})
    unproduced = any(not r["holders"] and r.get("source_type") == "dd" for r in records)
    holders = sorted({h for r in records for h in r["holders"]})
    return states, unproduced, holders


def classify_row(
    row: dict, dd4: dict, dd4_any: dict, dd3: dict, graph: dict, adj
) -> dict:
    name = row["name"]
    facts = graph["names"].get(name, {})
    name_unit = facts.get("unit")
    name_desc = facts.get("description") or ""
    inherited = row.get("published_provenance") != "own entry"

    judged: list[dict] = []
    carried = 0
    for item in iter_path_items(row):
        path = item["path"]
        if item.get("class") == "carried":
            carried += 1
            continue

        mapped = item.get("mapped_paths") or [path]
        meta = dd4_any.get(path) or dd4.get(path)
        in_dd4 = path in dd4_any
        in_dd3 = path in dd3
        for alias in mapped:
            if alias != path:
                meta = meta or dd4_any.get(alias) or dd4.get(alias)
                in_dd4 = in_dd4 or alias in dd4_any
                in_dd3 = in_dd3 or alias in dd3

        # The DD parser does not expand structure_reference fields.
        via_reference = False
        if not in_dd4 and not in_dd3:
            parent_meta = dd4_any.get(path.rsplit("/", 1)[0])
            if parent_meta and parent_meta.get("data_type") in {
                "STRUCTURE",
                "STRUCT_ARRAY",
            }:
                via_reference = True
                in_dd4 = True

        # A renamed path keeps its sources and holders under the mapped path, so a
        # published path is judged over its whole alias set.
        lookup = list(dict.fromkeys([path, *mapped]))
        records = [rec for p in lookup for rec in (graph["sources"].get(p) or [])]
        states, unproduced, live_holders = source_state(records)
        path_doc = (meta or {}).get("documentation") or ""
        node_category = (graph["nodes"].get(path) or {}).get("node_category")

        matched, missing = description_match(name, path_doc, path)

        verdict, why = judge_path(
            row=row,
            path=path,
            item=item,
            meta=meta,
            in_dd4=in_dd4,
            in_dd3=in_dd3,
            via_reference=via_reference,
            states=states,
            unproduced=unproduced,
            live_holders=live_holders,
            name_unit=name_unit,
            matched=matched,
            missing=missing,
            inherited=inherited,
            node_category=node_category,
            graph=graph,
            adj=adj,
        )
        judged.append(
            {
                "path": path,
                "census_class": item.get("class"),
                "verdict": verdict,
                "action": path_action(verdict),
                "why": cut(why, 110),
                # Both descriptions are quoted only where a rebind or a
                # description ruling rests on them.
                "path_description": (
                    cut(path_doc, 300)
                    if verdict in {"true loss", "unverified loss"}
                    else ""
                ),
                "matched_tokens": matched
                if verdict in {"true loss", "unverified loss"}
                else [],
                "missing_tokens": missing,
                "dd_data_type": (meta or {}).get("data_type"),
                "dd_units": (meta or {}).get("units"),
                "dd_via_structure_reference": via_reference,
                "node_category": (
                    node_category
                    if verdict in {"excluded by design", "pending composition"}
                    else ""
                ),
                "live_holders": live_holders,
                "source_record_states": states,
                "source_without_producer": unproduced,
                "published_latest": (item.get("published_at") or [None])[0],
                "inherit_from_ancestor": inherited,
            }
        )

    row_verdict = row_severity([j["verdict"] for j in judged])
    questionable = [j for j in judged if j["verdict"] == "questionable move"]
    resolution = {
        key: cut(value, 100)
        for key, value in row_resolution(
            row_verdict, name, name_desc, judged, questionable, carried
        ).items()
    }
    return {
        "name": name,
        "census_class": row["class"],
        "verdict": row_verdict,
        "questionable": bool(questionable),
        "lead_question": lead_question(name, questionable) if questionable else "",
        "status": row.get("status"),
        "name_stage": row.get("stage"),
        "unit": name_unit,
        "origin": row.get("origin"),
        "status_live": facts.get("status"),
        "stage_live": facts.get("name_stage"),
        "description": cut(name_desc, 112),
        "carried_path_count": carried,
        "judged_paths": judged,
        "resolution": resolution,
        "evidence": cut(row_evidence(row_verdict, judged), 120),
    }


def iter_path_items(row: dict):
    items = row.get("path_items")
    return items if isinstance(items, list) else (row.get("non_carried") or [])


def judge_path(
    *,
    row: dict,
    path: str,
    item: dict,
    meta: dict | None,
    in_dd4: bool,
    in_dd3: bool,
    via_reference: bool,
    states: list[str],
    unproduced: bool,
    live_holders: list[str],
    name_unit: str | None,
    matched: list[str],
    missing: list[str],
    inherited: bool,
    node_category: str | None,
    graph: dict,
    adj,
) -> tuple[str, str]:
    path_unit = (meta or {}).get("units")

    # 1. No source record at all, and the extractor never admits this node.
    #    The gate takes SN_SOURCE_CATEGORIES only (quantity|geometry|coordinate),
    #    so a '/data' leaf under a STRUCTURE parent, which classifies
    #    'structural', can never seed a source in the first place.
    if (
        not states
        and not live_holders
        and node_category is not None
        and node_category not in SN_SOURCE_CATEGORIES
    ):
        return (
            "excluded by design",
            f"no source record; node_category {node_category!r}",
        )

    # 2. A record exists but compose has not produced a name from it.
    if unproduced and not live_holders and "extracted" in states:
        return (
            "pending composition",
            f"source record {states}, no PRODUCED_NAME edge: compose has not run",
        )

    # 3. A stale record is refreshed, not restored.
    if states and set(states) <= {"stale"} and not live_holders:
        return (
            "stale source",
            f"source record {states}: refreshed, not restored",
        )

    if not in_dd4 and not in_dd3:
        if node_category is not None:
            # The graph carries DD nodes from versions this instrument did not read,
            # so absence here is an instrument limit, not a finding about the path.
            return (
                "unverified loss",
                "a DD node in the graph but not in the versions read "
                f"({DD_VERSION_3}, 4.0.0-4.1.1): units and description unavailable",
            )
        return (
            "instrument artefact",
            f"not in DD 4.0.0-4.1.1 or DD {DD_VERSION_3}",
        )

    # 4. Retired from DD 4 with no rename.
    if not in_dd4 and in_dd3:
        renamed = [a for a in (item.get("mapped_paths") or []) if a != path]
        if not renamed:
            return (
                "invalid published binding",
                f"only in DD {DD_VERSION_3}, absent from DD 4 with no rename",
            )

    # 5. A unit-less container is a hierarchy node, not a quantity.
    if (meta or {}).get("data_type") in {"STRUCTURE", "STRUCT_ARRAY"} and not path_unit:
        return (
            "invalid published binding",
            f"unit-less {meta.get('data_type')} container, not a quantity",
        )

    # 6. An uncertainty field is not the quantity the name measures.
    tail = path.rsplit("/", 1)[-1]
    if tail.endswith(("_error_upper", "_error_lower")):
        return (
            "invalid published binding",
            f"the uncertainty field {tail!r}, not the quantity",
        )

    # 7. Units that contradict the name.
    if units_conflict(name_unit, path_unit):
        return (
            "invalid published binding",
            f"DD units {path_unit!r} contradict the name's {name_unit!r}",
        )

    # 8. A live holder carries it.
    if live_holders:
        if row["name"] in live_holders:
            return (
                "instrument artefact",
                "the name itself holds this path today: the census predates its rebind",
            )
        chain_hits = [h for h in live_holders if same_chain(adj, row["name"], h)]
        if chain_hits:
            return (
                "legitimate move",
                f"held by {', '.join(chain_hits)}, in the lineage",
            )
        refinements = [h for h in live_holders if name_words_contain(h, row["name"])]
        if refinements:
            return (
                "legitimate move",
                f"held by {', '.join(refinements)}, a more specific name",
            )
        return (
            "questionable move",
            f"held by {', '.join(live_holders)}, outside the lineage, units agree",
        )

    # 9. No holder: a loss only if the path's own description denotes the name.
    full_match = not missing
    if inherited:
        return (
            "unverified loss",
            f"inherited path list (ancestor's entry); description shares {matched or 'no'} name words"
            + (f", missing {', '.join(missing)}" if missing else ""),
        )
    if full_match:
        return (
            "true loss",
            f"no holder; units {path_unit!r} match; description carries every name word ({', '.join(matched)})",
        )
    return (
        "unverified loss",
        f"no holder; units {path_unit!r} match; description does not name {', '.join(missing)}",
    )


def path_action(verdict: str) -> str:
    return {
        "true loss": "rebind",
        "unverified loss": "ruling-on-description",
        "questionable move": "keep-or-retire-ruling",
        "legitimate move": "keep",
        "invalid published binding": "leave-unsourced",
        "pending composition": "let-compose-run",
        "excluded by design": "no-action",
        "stale source": "leave-to-refresh",
        "instrument artefact": "no-action",
    }[verdict]


def row_severity(path_verdicts: list[str]) -> str:
    if not path_verdicts:
        return "instrument artefact"
    for verdict in PATH_VERDICTS:
        if verdict in path_verdicts:
            return verdict
    return "instrument artefact"


def lead_question(name: str, questionable: list[dict]) -> str:
    """One sentence the lead must answer for a row with a questionable move."""
    first = questionable[0]
    holder = first["live_holders"][0] if first["live_holders"] else "another live name"
    return (
        f"Does {holder} carry the quantity {name} names — it binds {first['path']} with agreeing "
        f"units but sits outside {name}'s lineage — so that {name} can be retired in its favour, "
        f"or should {name}'s own binding be restored?"
    )


def row_evidence(verdict: str, judged: list[dict]) -> str:
    counts = Counter(j["verdict"] for j in judged)
    witness = next((j for j in judged if j["verdict"] == verdict), judged[0])
    parts = [
        f"{counts[verdict]} path(s) {verdict}",
        f"witness {witness['path']}: {witness['why']}",
    ]
    if len(judged) > 1:
        parts.append(
            "all path verdicts "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
    return "; ".join(parts)


def row_resolution(
    verdict: str,
    name: str,
    name_desc: str,
    judged: list[dict],
    questionable: list[dict],
    carried: int,
) -> dict:
    same = [j for j in judged if j["verdict"] == verdict]
    first = same[0] if same else judged[0]
    if verdict == "true loss":
        return {
            "action": "rebind",
            "detail": f"rebind {', '.join(j['path'] for j in same)} to {name}",
            "evidence": f"name: {name_desc[:200]} | path: {first['path_description'][:200]}",
            "alternative": f"{carried} other published path(s) stay where they are",
        }
    if verdict == "unverified loss":
        return {
            "action": "send to the lead with both descriptions",
            "detail": f"do not rebind {', '.join(j['path'] for j in same)}: the path's description does not carry the name's words ({', '.join(first['missing_tokens'])})",
            "evidence": f"name: {name_desc[:200]} | path: {first['path_description'][:200]}",
            "alternative": "rebind only if the lead confirms both descriptions denote one quantity",
        }
    if verdict == "invalid published binding":
        return {
            "action": "leave unsourced",
            "detail": f"do not restore {', '.join(j['path'] for j in same)}: {first['why']}",
            "evidence": first["why"],
            "alternative": f"retire {name} if no valid published binding remains for it",
        }
    if verdict == "pending composition":
        return {
            "action": "let compose run",
            "detail": f"leave {', '.join(j['path'] for j in same)} to compose: the record is {first['source_record_states']} with no producer",
            "evidence": first["why"],
            "alternative": "none: no rebind is needed and none would be correct",
        }
    if verdict == "excluded by design":
        return {
            "action": "no action",
            "detail": f"{', '.join(j['path'] for j in same)} is outside the source gate: no rebind is possible",
            "evidence": f"{EXTRACTION_RULE['rule']} ({EXTRACTION_RULE['classified_at']})",
            "alternative": "none: naming it would need a change to the extraction gate, not a rebind",
        }
    if verdict == "stale source":
        return {
            "action": "leave to the refresh",
            "detail": f"{', '.join(j['path'] for j in same)}: the source refresh rewrites the record",
            "evidence": first["why"],
            "alternative": "none",
        }
    if verdict == "questionable move":
        return {
            "action": "keep where it is, pending a lead ruling",
            "detail": lead_question(name, questionable),
            "evidence": f"holder {', '.join(first['live_holders'])} binds {first['path']} with agreeing units, outside the name's lineage",
            "alternative": "retire the name in the holder's favour, or restore its own binding",
        }
    if verdict == "legitimate move":
        holder = ", ".join(first["live_holders"]) or "the census holder"
        return {
            "action": "keep where it is",
            "detail": f"{holder} carries {first['path']}; no write needed",
            "evidence": first["why"],
            "alternative": "none: the published binding reached a more specific name",
        }
    return {
        "action": "no action",
        "detail": "the census reading is corrected; the binding was never lost",
        "evidence": first["why"],
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
            p
            for row in population
            for item in iter_path_items(row)
            if item.get("class") != "carried"
            for p in [item["path"], *(item.get("mapped_paths") or [])]
        }
    )

    dd4_newest, dd4_any, dd3 = load_dd_paths()
    graph = read_graph(names, paths)
    adj = chain_adjacency(graph["lineage"])

    rows = [classify_row(r, dd4_newest, dd4_any, dd3, graph, adj) for r in population]
    rows.sort(key=lambda r: (PATH_VERDICTS.index(r["verdict"]), r["name"]))

    return {
        "generated_at": now(),
        "census_fragment": str(CENSUS.relative_to(REPO)),
        "population": {
            "rows": len(rows),
            "census_dropped": sum(r["class"] == "dropped" for r in population),
            "census_moved": sum(r["class"] == "moved" for r in population),
            "non_carried_paths": len(
                {
                    item["path"]
                    for row in population
                    for item in iter_path_items(row)
                    if item.get("class") != "carried"
                }
            ),
            "paths_judged_including_renames": len(paths),
            "rows_with_pending_composition": sum(
                any(j["verdict"] == "pending composition" for j in r["judged_paths"])
                for r in rows
            ),
            "rows_with_excluded_paths": sum(
                any(j["verdict"] == "excluded by design" for j in r["judged_paths"])
                for r in rows
            ),
            "rows_with_stale_sources": sum(
                any(j["verdict"] == "stale source" for j in r["judged_paths"])
                for r in rows
            ),
        },
        "dd_versions_read": {"dd4": list(DD_VERSIONS_4), "dd3": DD_VERSION_3},
        "extraction_rule": EXTRACTION_RULE,
        "verdict_counts": dict(Counter(r["verdict"] for r in rows)),
        "path_verdict_counts": dict(
            Counter(j["verdict"] for r in rows for j in r["judged_paths"])
        ),
        "path_action_counts": dict(
            Counter(j["action"] for r in rows for j in r["judged_paths"])
        ),
        "census_class_vs_verdict": {
            f"{c}:{v}": n
            for (c, v), n in sorted(
                Counter((r["census_class"], r["verdict"]) for r in rows).items()
            )
        },
        "rows_needing_lead_ruling": sum(
            r["verdict"] in {"true loss", "unverified loss", "questionable move"}
            for r in rows
        ),
        "rebind_sample": rebind_sample(rows),
        "negative_control": negative_control(dd4_any),
        "rows": rows,
    }


def rebind_sample(rows: list[dict]) -> list[dict]:
    """A random sample of the rebind proposals, both descriptions side by side."""
    proposals = [
        {
            "name": row["name"],
            "path": j["path"],
            "name_description": row["description"],
            "path_description": j["path_description"],
            "matched_tokens": j["matched_tokens"],
            "units": row["unit"],
            "dd_units": j["dd_units"],
        }
        for row in rows
        for j in row["judged_paths"]
        if j["verdict"] == "true loss"
    ]
    ordered = sorted(proposals, key=lambda p: (p["name"], p["path"]))
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(ordered, min(SAMPLE_SIZE, len(ordered)))
    return sample


def negative_control(dd4_any: dict) -> dict:
    """The declared control: the wall path must read as repaired by the rebind."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from imas_codex.graph import GraphClient

    name = "electrostatic_potential_at_wall"
    path = "wall/description_ggd/ggd/phi_potential/values"
    holders: list[str] = []
    states: list[str] = []
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
        states = sorted(rows[0]["states"] or [])
    return {
        "declared": "electrostatic_potential_at_wall's published wall path must classify as a true loss now repaired by the lead's rebind",
        "path": path,
        "live_holders": holders,
        "source_record_states": states,
        "path_in_dd4": path in dd4_any,
        "dd_units": (dd4_any.get(path) or {}).get("units"),
        "holder_is_the_name": holders == [name],
        "observed_at": now(),
    }


# ---------------------------------------------------------------------- render


def compact(data: dict) -> dict:
    """Write judged paths as fixed columns with a legend, keeping the file small."""
    columns = [
        "path",
        "census_class",
        "verdict",
        "action",
        "why",
        "dd_units",
        "live_holders",
        "source_record_states",
        "matched_tokens",
        "missing_tokens",
        "node_category",
        "inherit_from_ancestor",
        "path_description",
    ]
    rows = []
    for row in data["rows"]:
        copy = {k: v for k, v in row.items() if k != "judged_paths"}
        # The row-level evidence already carries the one-line summary; the
        # resolution's own evidence is quoted in the HTML.
        copy["resolution"] = {
            k: v for k, v in copy["resolution"].items() if k != "evidence"
        }
        copy["judged_paths"] = [
            [j.get(c) or None for c in columns] for j in row["judged_paths"]
        ]
        rows.append(copy)
    return dict(data, judged_path_columns=columns, rows=rows)


def render_html(data: dict) -> str:
    counts = data["verdict_counts"]
    nc = data["negative_control"]
    rule = data["extraction_rule"]

    def esc(text: object) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def badge(verdict: str) -> str:
        cls = {
            "true loss": "bad",
            "unverified loss": "warn",
            "questionable move": "warn",
            "legitimate move": "ok",
            "invalid published binding": "warn",
            "pending composition": "info",
            "excluded by design": "mute",
            "stale source": "info",
            "instrument artefact": "mute",
        }[verdict]
        return f'<span class="tag {cls}">{esc(verdict)}</span>'

    rows_html = []
    for row in data["rows"]:
        paths_html = "<br>".join(
            f"<code>{esc(j['path'])}</code> <span class='mute'>[{esc(j['census_class'])}]</span> "
            f"&rarr; {esc(j['verdict'])} <span class='mute'>({esc(j['action'])})</span>: {esc(cut(j['why'], 120))}"
            for j in row["judged_paths"]
        )
        res = row["resolution"]
        question = (
            f'<div class="q"><em>Lead question:</em> {esc(row["lead_question"])}</div>'
            if row["lead_question"]
            else ""
        )
        rows_html.append(
            f"""<tr id="{esc(row["name"])}">
  <td><strong>{esc(row["name"])}</strong><div class="mute">{esc(cut(row["description"], 120))}</div>
      <div class="mute">unit {esc(row["unit"])} &middot; {esc(row["stage_live"])}/{esc(row["status_live"])} &middot; {row["carried_path_count"]} carried</div></td>
  <td>{esc(row["census_class"])}</td>
  <td>{badge(row["verdict"])}</td>
  <td>{esc(row["evidence"])}</td>
  <td><div class="paths">{paths_html}</div></td>
  <td><strong>{esc(res["action"])}</strong><div>{esc(res["detail"])}</div>{question}<div class="mute">{esc(res["alternative"])}</div></td>
</tr>"""
        )

    tally = "".join(
        f"<li><strong>{esc(v)}</strong>: {counts.get(v, 0)}</li>" for v in PATH_VERDICTS
    )
    path_tally = "".join(
        f"<li><code>{esc(k)}</code>: {v}</li>"
        for k, v in sorted(data["path_verdict_counts"].items())
    )
    cross = "".join(
        f"<li><code>{esc(k)}</code>: {v}</li>"
        for k, v in data["census_class_vs_verdict"].items()
    )
    sample_rows = "".join(
        f"""<tr><td><strong>{esc(s["name"])}</strong><div class="mute">unit {esc(s["units"])}</div></td>
<td class="desc">{esc(s["name_description"])}</td>
<td><code>{esc(s["path"])}</code><div class="mute">DD units {esc(s["dd_units"])}</div></td>
<td class="desc">{esc(s["path_description"])}</td>
<td>{esc(", ".join(s["matched_tokens"]))}</td></tr>"""
        for s in data["rebind_sample"]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Row verdicts for the 156 lost or moved catalog-imported DD sources</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1500px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; color: #1b1b1b; }}
  h1 {{ font-size: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; }}
  th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid #dcdcdc; padding: 0.4rem 0.5rem; }}
  th {{ background: #f4f4f4; }}
  code {{ font-size: 0.8rem; }}
  .mute {{ color: #6a6a6a; font-size: 0.78rem; }}
  .tag {{ display: inline-block; padding: 0.05rem 0.4rem; border-radius: 3px; font-size: 0.74rem; white-space: nowrap; }}
  .tag.bad {{ background: #f7d9d9; color: #7d1f1f; }}
  .tag.ok {{ background: #d9f0dd; color: #1f5b2b; }}
  .tag.warn {{ background: #f7ecc9; color: #6b5312; }}
  .tag.info {{ background: #dde8f7; color: #1f3d6b; }}
  .tag.mute {{ background: #e6e6e6; color: #4a4a4a; }}
  .paths {{ font-size: 0.76rem; }}
  .desc {{ font-size: 0.76rem; max-width: 320px; }}
  .q {{ background: #fbf6e3; padding: 0.25rem 0.45rem; margin-top: 0.3rem; font-size: 0.8rem; }}
  .box {{ background: #f7f7f7; padding: 0.6rem 0.9rem; border-left: 3px solid #888; margin: 1rem 0; }}
</style></head><body>
<h1>Row verdicts for the 156 lost or moved catalog-imported DD sources</h1>
<p class="mute">Read-only. Generated {esc(data["generated_at"])}. Population: the {data["population"]["rows"]} live names
with their own published catalog entry that the census classed dropped ({data["population"]["census_dropped"]}) or
moved ({data["population"]["census_moved"]}), across {data["population"]["non_carried_paths"]} published non-carried paths
({data["population"]["paths_judged_including_renames"]} counting DD renames).
No graph write, no LLM call.</p>

<h2>How a row is judged</h2>
<p>Each published path is judged on four independent facts: whether a <code>StandardNameSource</code> record exists
at all, what state that record is in, who currently produces the path, and whether the path's own DD description
denotes the quantity the name names.</p>
<ul>
<li><strong>excluded by design</strong> — no source record at all, and the path is one the extractor never admits.
A <code>/data</code> leaf under a STRUCTURE parent classifies <code>structural</code> ({esc(rule["classified_at"])}),
and GGD representation nodes classify <code>representation</code>; the source gate
({esc(rule["gate"])}) admits only <code>quantity | geometry | coordinate</code>
({esc(rule["categories_at"])}). Every excluded path carries its own node category as evidence.</li>
<li><strong>pending composition</strong> — a source record exists with state <code>extracted</code> and no
<code>PRODUCED_NAME</code> edge. Extraction seeded it and compose has not run, so no published binding was lost;
the resolution is to let compose run, never to rebind the catalog name.</li>
<li><strong>true loss</strong> — no live holder, units that match, and the path's description carries every content
word of the name. Only this verdict proposes a rebind.</li>
<li><strong>unverified loss</strong> — no live holder and matching units, but the path's description does not carry
the name's words, or the path list came from an ancestor's entry. No rebind is proposed; the lead reads both
descriptions.</li>
<li><strong>questionable move</strong> — a live name outside the lineage holds the same path with agreeing units; the
row carries the one-sentence question the lead must answer.</li>
</ul>

<div class="box"><strong>Declared negative control.</strong> <code>{esc(nc["path"])}</code> is held by
{esc(", ".join(nc["live_holders"]) or "nothing")} (source state {esc(", ".join(nc["source_record_states"]))};
DD 4 units {esc(nc["dd_units"])}; in DD 4: {esc(nc["path_in_dd4"])}), so the lead's rebind is in place and the row
does not read as a loss.</div>

<h2>Recount</h2>
<ul>{tally}</ul>
<p class="mute">Path-level verdicts over the {data["population"]["non_carried_paths"]} non-carried paths:</p>
<ul>{path_tally}</ul>
<h3>Census class against row verdict</h3>
<ul>{cross}</ul>
<p>Rows that need a lead ruling, counting true losses, unverified losses and questionable moves:
{data["rows_needing_lead_ruling"]}.</p>
<p class="mute">A row's verdict is its most severe path verdict, so a row also holding a moved or kept path shows
that one. Paths awaiting compose appear in {data["population"]["rows_with_pending_composition"]} rows, paths the
source gate never admits in {data["population"]["rows_with_excluded_paths"]} rows, and stale source records in
{data["population"]["rows_with_stale_sources"]} rows.</p>

<h2>Rebind proposals sampled for review</h2>
<p>{len(data["rebind_sample"])} of the rebind proposals drawn at random (seed {SAMPLE_SEED};
{SAMPLE_SIZE} requested, {data["path_verdict_counts"].get("true loss", 0)} available), name description beside the
path's own DD description. A rebind is proposed only where the path's description denotes the name's quantity; every
path whose description does not carry the name's words is sent to the lead as an unverified loss instead.</p>
<table><thead><tr><th>Name</th><th>Name description</th><th>Path</th><th>Path description</th><th>Matched words</th></tr></thead>
<tbody>{sample_rows}</tbody></table>

<h2>The 156 rows</h2>
<table>
<thead><tr><th>Name</th><th>Census</th><th>Verdict</th><th>Evidence</th><th>Paths</th><th>Proposed resolution</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>

<h2>What this does not settle</h2>
<p>The population is the 156 live names carrying their own published catalog entry. Superseded and inherited
census rows are outside it. A rebind is a proposal: the lead rules on every row above that reaches a loss verdict or
carries a questionable move, and nothing is written to the graph by this instrument.</p>
</body></html>
"""


def main() -> int:
    try:
        data = build()
        OUT_HTML.write_text(render_html(data), encoding="utf-8")
        OUT_JSON.write_text(
            json.dumps(compact(data), separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"rows {data['population']['rows']}")
        print(
            "row verdicts "
            + ", ".join(f"{k}={v}" for k, v in data["verdict_counts"].items())
        )
        print(
            "path verdicts "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(data["path_verdict_counts"].items())
            )
        )
        print(
            "path actions "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(data["path_action_counts"].items())
            )
        )
        print(f"rows needing a lead ruling {data['rows_needing_lead_ruling']}")
        print(f"rebind sample {len(data['rebind_sample'])}")
        nc = data["negative_control"]
        print(
            f"negative control holders {nc['live_holders']} states {nc['source_record_states']} "
            f"in_dd4 {nc['path_in_dd4']} units {nc['dd_units']} is_name {nc['holder_is_the_name']}"
        )
        print(
            f"html bytes {OUT_HTML.stat().st_size} json bytes {OUT_JSON.stat().st_size}"
        )
        print("EXIT=0")
        return 0
    except Exception:
        traceback.print_exc()
        print("EXIT=1")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
