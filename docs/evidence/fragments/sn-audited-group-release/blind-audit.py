"""Read a stratified sample of the live catalog against its Data Dictionary sources.

This is the blind audit of the live standard names. It extends the owners the
pipeline already has rather than building a second harness beside them:

* the draw is :func:`imas_codex.standard_names.benchmark_roles._stratified_sample`,
  the seeded round-robin sampler the benchmark and golden builders use;
* the fixed class list and the per-row reasoning live in this file, and they
  were committed before any name was read;
* the DD leaf text is read the way the name review reads it
  (``IMASNode.description`` / ``documentation`` / ``unit``), and the enclosing
  structures come from ``worker_extraction_and_enrichment._ANCESTOR_CONTEXT_QUERY``,
  the one ancestor read the docs and name grounding share;
* each verdict is stored in the record shape
  :func:`imas_codex.standard_names.review.pipeline._build_review_record` builds,
  tagged ``resolution_role='audit'`` so no accepting, refining or exporting path
  can read it;
* the statistics are the two-seat form of
  :mod:`imas_codex.standard_names.review.panel_audit`'s agreement report.

The two readers are seats of the paid panel. No production seat and no
selectable review profile composes, refines, generates docs or reviews with
either model, and the two are different families:

* ``openrouter/google/gemini-3.8-flash`` — Google. It is a priced endpoint; no
  ``[tool.imas-codex.<seat>].model`` and no ``sn-review`` seat or profile names
  it.
* ``openrouter/moonshotai/kimi-k3.1-pro`` — Moonshot. Same test: no production
  seat and no review profile names either model.

Both appear in the offline ``sn-benchmark`` candidate slate, which evaluates
models outside the pipeline rather than composing or reviewing with them.
:func:`assert_readers_not_pipeline` derives the production set from
``pyproject.toml`` (seat ``model`` values and every ``sn-review`` ``models``
list, profiles included) and refuses a reader that appears in it.

Neither reader is told a name's stage, score, review history, or that any name
is planted. A prompt carries only the audited fields: the name, its stored
description, documentation and unit, its assigned physics domain, the DD text
of its sources, the text stated by those sources' enclosing structures, and the
ids of other live names produced by the same sources. :func:`assert_blind`
checks that no forbidden field reached the rendered prompt.

Usage::

    uv run --no-sync python blind-audit.py --dry-run
    uv run --no-sync python blind-audit.py --run
    uv run --no-sync python blind-audit.py --report
    uv run --no-sync python blind-audit.py --persist

The run appends one record per call to ``<state>/records.jsonl`` so a run cut
short keeps what it has paid for, and writes the committed summary to
``blind-audit.json`` and ``sagr-blind-audit.html``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

FRAGMENTS = Path(__file__).resolve().parent
OUT_JSON = FRAGMENTS / "blind-audit.json"
OUT_HTML = FRAGMENTS / "sagr-blind-audit.html"
STATE = Path(
    os.environ.get("BLIND_AUDIT_STATE")
    or (Path.home() / ".cache/imas-codex-live-rewrite/blind-audit")
)

SEED = 20261117
SAMPLE_SIZE = 340
PLANTS_PER_SIDE = 20
MIN_GROUP = 20
SPEND_CEILING = 30.0
CONCURRENCY = 10
MAX_TOKENS = 4000
TIMEOUT_S = 240

# Fixed before any name was read; do not change after a reading was taken.
DEFECT_CLASSES: tuple[tuple[str, str], ...] = (
    (
        "name_contradicts_source",
        "The name states a quantity, locus, component or relation that the Data "
        "Dictionary source text contradicts.",
    ),
    (
        "name_omits_source",
        "The source text states a qualifier of the quantity — a counting basis, "
        "a species basis, a weighting or average, a normalisation, a coordinate "
        "convention, a component or a locus — and the name leaves it out.",
    ),
    (
        "definition_contradicts_source",
        "The stored definition (description or documentation) contradicts the "
        "Data Dictionary source text.",
    ),
    (
        "name_definition_disagree",
        "The stored name and the stored definition describe different "
        "quantities or different scopes, even if each is internally consistent.",
    ),
    (
        "unit_wrong",
        "The stored unit does not fit the quantity the name states: it is "
        "dimensionally wrong for that kind of quantity, or it is the wrong unit "
        "for the stated kind of quantity.",
    ),
    (
        "wrong_physics_domain",
        "The assigned physics domain is not the domain of the quantity the "
        "source describes.",
    ),
    (
        "duplicate_of_live_name",
        "The name denotes the same quantity as another live standard name "
        "listed for it, so the catalog carries one quantity twice.",
    ),
)
CLASS_KEYS = tuple(key for key, _ in DEFECT_CLASSES)

READERS: tuple[dict[str, str], ...] = (
    {"seat": "A", "model": "openrouter/google/gemini-3.8-flash", "family": "google"},
    {"seat": "B", "model": "openrouter/moonshotai/kimi-k3.1-pro", "family": "moonshot"},
)

FORBIDDEN_PROMPT_FIELDS = (
    "name_stage",
    "review_mean_score",
    "reviewer_score",
    "reviewer_model",
    "review_count",
)

LIVE = (
    "coalesce(sn.name_stage,'') <> 'superseded' AND "
    "coalesce(sn.status,'') <> 'superseded'"
)


class Judgment(BaseModel):
    """One reader's verdict on one name."""

    defect_classes: list[str] = Field(
        default_factory=list,
        description=(
            "Every fixed defect class that applies to this name, by its exact "
            "key; an empty list means no defect."
        ),
    )
    rationale: str = Field(
        description=(
            "The stored or source wording that decides the verdict, at most 60 "
            "words. For the empty list, the one quantity the source, name and "
            "rationale structures share."
        )
    )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def graph_client() -> Any:
    from imas_codex.graph.client import GraphClient

    return GraphClient()


def load_live_names(gc: Any) -> list[dict[str, Any]]:
    """Every live name with the fields the audit reads."""
    from imas_codex.standard_names.source_reads import source_paths_projection

    rows = gc.query(
        f"""
        MATCH (sn:StandardName)
        WHERE {LIVE}
        RETURN sn.id AS id,
               coalesce(sn.description,'') AS description,
               coalesce(sn.documentation,'') AS documentation,
               coalesce(sn.unit,'') AS unit,
               coalesce(sn.physics_domain,'') AS physics_domain,
               {source_paths_projection()} AS sources
        ORDER BY id
        """,
    )
    for row in rows:
        row["source_paths"] = list(row.get("sources") or [])
        row.pop("sources", None)
    return list(rows)


def domain_groups(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(r.get("physics_domain") or "") for r in rows)
    return {g: c for g, c in sorted(counts.items()) if g and c >= MIN_GROUP}


def draw_sample(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """The seeded stratified draw over every physics-domain group of >= 20 names."""
    from imas_codex.standard_names.benchmark_roles import _stratified_sample

    groups = domain_groups(rows)
    eligible = [r for r in rows if str(r.get("physics_domain") or "") in groups]
    drawn = _stratified_sample(eligible, SAMPLE_SIZE, SEED, key="physics_domain")
    counts = Counter(str(r.get("physics_domain") or "") for r in rows)
    return drawn, {
        "frame": len(eligible),
        "groups": groups,
        "excluded_groups": {
            g: c for g, c in sorted(counts.items()) if not g or c < MIN_GROUP
        },
        "sample": len(drawn),
        "seed": SEED,
        "sampler": "benchmark_roles._stratified_sample(key='physics_domain')",
    }


def rank(*parts: str) -> str:
    return hashlib.sha256(":".join((str(SEED), *parts)).encode()).hexdigest()


def load_plants(
    live: dict[str, set[str]], in_sample: set[str]
) -> tuple[list[dict[str, Any]], dict]:
    """Twenty names known defective and twenty known sound, from the west audit.

    The west catalog release carries a physicist's row-by-row ruling on 341
    names with the source path each ruling was made against. A planted name
    must still be live, still produced by the audited source, and outside the
    drawn sample. Sound plants come from rows ruled ``correct``; defective
    plants from rows ruled ``INCORRECT``.
    """
    from imas_codex.standard_names.judgment_goldens import _west_verdicts

    verdicts = _west_verdicts()
    chosen: dict[str, list[dict[str, Any]]] = {}
    for verdict in ("incorrect", "correct"):
        pool = [
            row
            for row in verdicts
            if row["verdict"] == verdict
            and row["name"] in live
            and row["path"] in live[row["name"]]
            and row["name"] not in in_sample
        ]
        pool.sort(key=lambda row: rank("plant", row["name"]))
        seen: set[str] = set()
        unique = []
        for row in pool:
            if row["name"] in seen:
                continue
            seen.add(row["name"])
            unique.append(row)
        chosen[verdict] = unique[:PLANTS_PER_SIDE]
    plants: list[dict[str, Any]] = []
    for verdict, known in (("incorrect", "defective"), ("correct", "sound")):
        for row in chosen[verdict]:
            plants.append(
                {
                    "id": row["name"],
                    "known": known,
                    "citation": row["evidence"],
                    "audited_source_path": row["path"],
                    "auditor_ruled_description_wrong": bool(
                        row.get("description_issue")
                    ),
                }
            )
    return plants, {
        "defective_pool": len(
            {
                row["name"]
                for row in verdicts
                if row["verdict"] == "incorrect"
                and row["name"] in live
                and row["path"] in live[row["name"]]
                and row["name"] not in in_sample
            }
        ),
        "sound_pool": len(
            {
                row["name"]
                for row in verdicts
                if row["verdict"] == "correct"
                and row["name"] in live
                and row["path"] in live[row["name"]]
                and row["name"] not in in_sample
            }
        ),
        "plants": len(plants),
        "label_source": (
            "docs/evidence/sn-west-catalog-release/west-name-audit*.md, read "
            "row by row by judgment_goldens._west_verdicts"
        ),
    }


def load_dd_context(gc: Any, names: list[dict[str, Any]]) -> None:
    """Attach the DD leaf and enclosing-structure text to each name row."""
    from imas_codex.standard_names.source_paths import split_source_paths
    from imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment import (
        _ANCESTOR_CONTEXT_QUERY,
    )

    all_paths: set[str] = set()
    primaries: dict[str, str] = {}
    for row in names:
        dd = split_source_paths(row.get("source_paths") or []).get("dd", [])
        row["dd_paths"] = dd
        all_paths.update(dd)
        if dd:
            primaries[row["id"]] = dd[0]

    docs: dict[str, dict[str, Any]] = {}
    ordered = sorted(all_paths)
    for index in range(0, len(ordered), 500):
        rows = gc.query(
            """
            UNWIND $paths AS pid
            MATCH (n:IMASNode {id: pid})
            RETURN n.id AS id,
                   coalesce(n.unit,'') AS unit,
                   coalesce(n.description,'') AS description,
                   coalesce(n.documentation,'') AS documentation
            """,
            paths=ordered[index : index + 500],
        )
        for row in rows or []:
            docs[str(row["id"])] = {
                "unit": str(row.get("unit") or ""),
                "description": str(row.get("description") or ""),
                "documentation": str(row.get("documentation") or ""),
            }

    for row in names:
        row["dd_docs"] = [
            {"path": path} | docs[path] for path in row["dd_paths"] if path in docs
        ]
        row["ancestors"] = []
        row["other_live_names"] = []
        primary = primaries.get(row["id"])
        if not primary:
            continue
        try:
            ancestor_rows = list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=primary))
        except Exception:  # a missing chain is absent context, not a failure
            ancestor_rows = []
        lineage: list[dict[str, str]] = []
        seen: set[str] = set()
        for node in ancestor_rows:
            text = (node.get("description") or "").strip() or (
                node.get("documentation") or ""
            ).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            lineage.append({"path": str(node.get("path") or ""), "text": text})
            if len(lineage) >= 4:
                break
        row["ancestors"] = lineage


def attach_duplicate_candidates(
    names: list[dict[str, Any]],
    names_by_source: dict[str, list[str]],
) -> None:
    """Ids of other live names produced by the same DD sources, at most ten."""
    for row in names:
        seen: list[str] = []
        for path in row.get("dd_paths") or []:
            for other in names_by_source.get(path, []):
                if other != row["id"] and other not in seen:
                    seen.append(other)
        row["other_live_names"] = sorted(seen)[:10]


def names_by_dd_source(live_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    from imas_codex.standard_names.source_paths import split_source_paths

    out: dict[str, list[str]] = defaultdict(list)
    for row in live_rows:
        for path in split_source_paths(row.get("source_paths") or []).get("dd", []):
            out[path].append(row["id"])
    return out


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def system_prompt() -> str:
    classes = "\n".join(f"- `{key}`: {text}" for key, text in DEFECT_CLASSES)
    return (
        "You are an independent physics auditor of the standard names of a "
        "fusion data dictionary. A standard name identifies exactly one "
        "quantity. You are given one name record at a time and must decide "
        "which of these fixed defect classes apply to it:\n\n"
        f"{classes}\n\n"
        "Rules:\n"
        "- Report every class that applies. A name may carry more than one.\n"
        "- Report an empty list when no class applies. Do not invent a defect, "
        "and do not flag a name for style, brevity, missing prose, or for "
        "information no part of the record states.\n"
        "- A qualifier stated by a source's enclosing structure is stated by the "
        "source. Report a missing qualifier only when neither the source leaf "
        "nor any enclosing structure states it.\n"
        "- Report a duplicate only against a name in the list of other live "
        "names given for the record.\n"
        "- If the record carries no Data Dictionary text, judge only the classes "
        "that need no source: name_definition_disagree, unit_wrong, "
        "wrong_physics_domain, duplicate_of_live_name.\n"
        "- Quote the stored or source wording that decides the verdict. At most "
        "60 words.\n"
        "- Return the JSON object the schema asks for; no other text."
    )


def clip(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text or "(none recorded)"
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def render_user_prompt(name: dict[str, Any]) -> str:
    lines = [
        f"Standard name: {name['id']}",
        f"Definition (stored description): {clip(name.get('description'), 1500)}",
        f"Documentation (stored): {clip(name.get('documentation'), 3000)}",
        f"Stored unit: {name.get('unit') or '(none recorded)'}",
        f"Assigned physics domain: {name.get('physics_domain') or '(none recorded)'}",
    ]
    docs = name.get("dd_docs") or []
    if docs:
        lines.append("Data Dictionary source text:")
        for index, doc in enumerate(docs, 1):
            lines.append(f"  Source {index} path: {doc['path']}")
            lines.append(f"    unit: {doc.get('unit') or '(none stated)'}")
            lines.append(f"    description: {clip(doc.get('description'), 1200)}")
            lines.append(f"    documentation: {clip(doc.get('documentation'), 1200)}")
    else:
        lines.append("Data Dictionary source text: (none recorded)")
    ancestors = name.get("ancestors") or []
    if ancestors:
        lines.append("Enclosing structures of the first source, nearest first:")
        for node in ancestors:
            lines.append(f"  {node['path']}: {clip(node['text'], 500)}")
    others = name.get("other_live_names") or []
    if others:
        lines.append(
            "Other live standard names produced by the same source(s): "
            + ", ".join(others)
        )
    lines.append(
        "Return every fixed defect class that applies, or an empty list, with "
        "the wording that decides it."
    )
    return "\n".join(lines)


def assert_blind(prompt: str, record: dict[str, Any]) -> None:
    """Refuse a prompt that carries a stage, score or earlier verdict."""
    lowered = prompt.lower()
    for field in FORBIDDEN_PROMPT_FIELDS:
        if field in lowered:
            raise AssertionError(f"forbidden field name in prompt: {field}")
    for field in ("review_mean_score", "reviewer_score", "name_stage"):
        value = record.get(field)
        if value is not None and str(value) in prompt:
            raise AssertionError(f"forbidden field value in prompt: {value}")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


async def judge_one(
    reader: dict[str, str], name: dict[str, Any], spend: dict[str, float]
) -> dict[str, Any]:
    from imas_codex.discovery.base.llm import acall_llm_structured

    if spend["total"] > SPEND_CEILING:
        raise RuntimeError(f"spend ceiling reached at {spend['total']:.2f} USD")
    system = system_prompt()
    user = render_user_prompt(name)
    assert_blind(system + "\n" + user, name)
    started = time.time()
    try:
        result = await acall_llm_structured(
            model=reader["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=Judgment,
            service="standard-names",
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT_S,
            max_retries=2,
        )
    except Exception as exc:  # noqa: BLE001 - logged, never invented
        return {
            "name": name["id"],
            "seat": reader["seat"],
            "model": reader["model"],
            "defect_classes": [],
            "rationale": "",
            "error": f"{type(exc).__name__}: {exc}"[:400],
            "cost": None,
            "tokens": 0,
            "seconds": round(time.time() - started, 1),
        }
    classes: list[str] = []
    for key in result.parsed.defect_classes:
        key = str(key).strip()
        if key in CLASS_KEYS and key not in classes:
            classes.append(key)
    spend["total"] += float(result.cost)
    return {
        "name": name["id"],
        "seat": reader["seat"],
        "model": reader["model"],
        "defect_classes": classes,
        "rationale": (result.parsed.rationale or "").strip(),
        "cost": float(result.cost),
        "tokens": int(result.tokens),
        "input_tokens": int(result.input_tokens),
        "output_tokens": int(result.output_tokens),
        "cache_read_tokens": int(result.cache_read_tokens),
        "seconds": round(time.time() - started, 1),
    }


def progress_path() -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / "records.jsonl"


def load_progress() -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    path = progress_path()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                continue
            done[(record["seat"], record["name"])] = record
    return done


async def run_reading(
    items: list[dict[str, Any]], reviewers: tuple[dict[str, str], ...]
) -> list[dict[str, Any]]:
    done = load_progress()
    path = progress_path()
    gate = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    spend = {"total": 0.0}
    todo = [
        (reader, item)
        for reader in reviewers
        for item in items
        if (reader["seat"], item["id"]) not in done
    ]
    print(
        f"reading {len(todo)} name-seat pairs; {len(done)} already on disk",
        flush=True,
    )

    async def one(reader: dict[str, str], item: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            record = await judge_one(reader, item, spend)
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    results: list[dict[str, Any]] = list(done.values())
    tasks = [asyncio.create_task(one(reader, item)) for reader, item in todo]
    for finished in asyncio.as_completed(tasks):
        record = await finished
        results.append(record)
        if len(results) % 40 == 0:
            print(f"  {len(results)} readings, {spend['total']:.3f} USD", flush=True)
    print(f"readings done: {len(results)}; spend {spend['total']:.3f} USD", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    pof = p * (1 - p) / total
    half = z * math.sqrt(pof + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa(both: int, only_a: int, only_b: int, neither: int) -> float | None:
    """Cohen's kappa on the two readers' defective-or-not labels.

    Delegates to the repository's one implementation,
    :func:`imas_codex.discovery.base.evaluation.cohen_kappa`, rebuilding the
    label sequences the counts came from so both callers cannot drift.
    """
    from imas_codex.discovery.base.evaluation import cohen_kappa

    if both + only_a + only_b + neither == 0:
        return None
    labels_a = [1] * (both + only_a) + [0] * (only_b + neither)
    labels_b = [1] * both + [0] * only_a + [1] * only_b + [0] * neither
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(cohen_kappa(labels_a, labels_b))
    return None if math.isnan(value) else value


def agreement_counts(
    readings: dict[tuple[str, str], dict[str, Any]],
    names: list[str],
    key: str | None = None,
) -> tuple[int, int, int, int]:
    both = only_a = only_b = neither = 0
    for name in names:
        a_classes = readings.get(("A", name), {}).get("defect_classes", [])
        b_classes = readings.get(("B", name), {}).get("defect_classes", [])
        if key is None:
            a = bool(a_classes)
            b = bool(b_classes)
        else:
            a = key in a_classes
            b = key in b_classes
        if a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1
    return both, only_a, only_b, neither


def round4(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _share(count: int, n: int) -> dict[str, Any]:
    low, high = wilson(count, n)
    return {
        "names": count,
        "rate": round(count / n, 4) if n else None,
        "ci": [round(low, 4), round(high, 4)],
    }


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def summarize(
    *,
    readings_raw: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    plants: list[dict[str, Any]],
    draw: dict[str, Any],
    prompts: dict[str, str],
) -> dict[str, Any]:
    readings = {(r["seat"], r["name"]): r for r in readings_raw if not r.get("error")}
    errors = [r for r in readings_raw if r.get("error")]
    sampled_ids = [row["id"] for row in sampled]
    plant_ids = [p["id"] for p in plants]
    complete = [
        name
        for name in sampled_ids
        if ("A", name) in readings and ("B", name) in readings
    ]
    through_plants = [
        name
        for name in plant_ids
        if ("A", name) in readings and ("B", name) in readings
    ]
    n = len(complete)

    per_reader: dict[str, dict[str, Any]] = {}
    for seat in ("A", "B"):
        defective = sum(
            1 for name in complete if readings[(seat, name)].get("defect_classes")
        )
        low, high = wilson(defective, n)
        per_reader[seat] = {
            "read": sum(1 for name in sampled_ids if (seat, name) in readings),
            "defective": defective,
            "rate": round(defective / n, 4) if n else None,
            "ci": [round(low, 4), round(high, 4)],
            "spend": round(
                sum(float(r["cost"] or 0.0) for r in readings_raw if r["seat"] == seat),
                4,
            ),
            "tokens": sum(
                r.get("tokens", 0) for r in readings_raw if r["seat"] == seat
            ),
        }

    both, only_a, only_b, neither = agreement_counts(readings, complete)
    at_least_one = both + only_a + only_b
    low, high = wilson(at_least_one, n)

    per_class: dict[str, dict[str, Any]] = {}
    for key in CLASS_KEYS:
        c_both, c_a, c_b, c_n = agreement_counts(readings, complete, key)
        total = c_both + c_a + c_b
        c_low, c_high = wilson(total, n)
        per_class[key] = {
            "both": c_both,
            "only_A": c_a,
            "only_B": c_b,
            "one_reader": c_a + c_b,
            "any": total,
            "agreement": round((c_both + c_n) / n, 4) if n else None,
            "agreement_ci": [
                round(wilson(c_both + c_n, n)[0], 4),
                round(wilson(c_both + c_n, n)[1], 4),
            ],
            "kappa": round4(kappa(c_both, c_a, c_b, c_n)),
            "rate_any": round(total / n, 4) if n else None,
            "ci_any": [round(c_low, 4), round(c_high, 4)],
        }

    groups: dict[str, dict[str, Any]] = {}
    by_group = Counter(str(row.get("physics_domain") or "") for row in sampled)
    for group, _ in sorted(by_group.items()):
        ids = [
            row["id"]
            for row in sampled
            if str(row.get("physics_domain") or "") == group
        ]
        ids = [name for name in ids if name in complete]
        groups[group] = {
            "sampled": len(ids),
            "frame": draw["groups"].get(group),
            "defective_A": sum(
                1 for name in ids if readings[("A", name)].get("defect_classes")
            ),
            "defective_B": sum(
                1 for name in ids if readings[("B", name)].get("defect_classes")
            ),
            "both": sum(
                1
                for name in ids
                if readings[("A", name)].get("defect_classes")
                and readings[("B", name)].get("defect_classes")
            ),
        }

    plant_rows = [
        {
            **plant,
            "A": plant_reading(readings, "A", plant["id"]),
            "B": plant_reading(readings, "B", plant["id"]),
        }
        for plant in plants
    ]
    defective_plants = [p for p in plant_rows if p["known"] == "defective"]
    sound_plants = [p for p in plant_rows if p["known"] == "sound"]
    planted = {
        "defective": {
            "count": len(defective_plants),
            "recall_A": rate(
                sum(1 for p in defective_plants if p["A"]["defective"]),
                len(defective_plants),
            ),
            "recall_B": rate(
                sum(1 for p in defective_plants if p["B"]["defective"]),
                len(defective_plants),
            ),
        },
        "sound": {
            "count": len(sound_plants),
            "false_alarm_A": rate(
                sum(1 for p in sound_plants if p["A"]["defective"]),
                len(sound_plants),
            ),
            "false_alarm_B": rate(
                sum(1 for p in sound_plants if p["B"]["defective"]),
                len(sound_plants),
            ),
        },
        "rows": plant_rows,
    }

    disagreements = []
    for name in complete:
        a = readings[("A", name)]
        b = readings[("B", name)]
        if bool(a.get("defect_classes")) == bool(b.get("defect_classes")):
            continue
        disagreements.append(
            {
                "name": name,
                "A": {
                    "classes": a.get("defect_classes", []),
                    "rationale": a.get("rationale", ""),
                },
                "B": {
                    "classes": b.get("defect_classes", []),
                    "rationale": b.get("rationale", ""),
                },
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "class_list_fixed": [{"key": k, "definition": d} for k, d in DEFECT_CLASSES],
        "readers": [dict(r) for r in READERS],
        "draw": draw,
        "names_read": n,
        "requested": len(sampled_ids),
        "missing_pairs": [
            {
                "name": name,
                "missing": [
                    seat for seat in ("A", "B") if (seat, name) not in readings
                ],
            }
            for name in sampled_ids
            if name not in complete
        ],
        "plant_pairs": len(through_plants),
        "plant_missing_pairs": [
            {
                "name": name,
                "missing": [
                    seat for seat in ("A", "B") if (seat, name) not in readings
                ],
            }
            for name in plant_ids
            if name not in through_plants
        ],
        "sampled": len(sampled),
        "planted": len(plants),
        "per_reader": per_reader,
        "defective_any": {
            "both": both,
            "only_A": only_a,
            "only_B": only_b,
            "neither": neither,
            "at_least_one": at_least_one,
            "at_least_one_rate": round(at_least_one / n, 4) if n else None,
            "at_least_one_ci": [round(low, 4), round(high, 4)],
            "rows": {
                "both": _share(both, n),
                "exactly_one": _share(only_a + only_b, n),
                "neither": _share(neither, n),
            },
            "observed_agreement": round((both + neither) / n, 4) if n else None,
            "observed_agreement_ci": [
                round(wilson(both + neither, n)[0], 4),
                round(wilson(both + neither, n)[1], 4),
            ],
            "kappa": round4(kappa(both, only_a, only_b, neither)),
        },
        "per_class": per_class,
        "defective_names": [
            {
                "name": name,
                "A": readings[("A", name)].get("defect_classes", []),
                "B": readings[("B", name)].get("defect_classes", []),
            }
            for name in complete
            if readings[("A", name)].get("defect_classes")
            or readings[("B", name)].get("defect_classes")
        ],
        "groups": groups,
        "planted_detail": planted,
        "disagreements": disagreements,
        "errors": errors[:20],
        "error_count": len(errors),
        "spend": {
            "total": round(sum(float(r["cost"] or 0.0) for r in readings_raw), 4),
            "by_seat": {seat: per_reader[seat]["spend"] for seat in ("A", "B")},
            "unknown_cost_calls": sum(1 for r in readings_raw if r.get("cost") is None),
            "ceiling": SPEND_CEILING,
        },
        "prompts": prompts,
        "environment": {
            "seed": SEED,
            "sample_size_requested": SAMPLE_SIZE,
            "sample_size_drawn": len(sampled),
            "concurrency": CONCURRENCY,
            "python": sys.version.split()[0],
            "cwd": str(Path.cwd()),
        },
    }


def plant_reading(
    readings: dict[tuple[str, str], dict[str, Any]], seat: str, name: str
) -> dict[str, Any]:
    row = readings.get((seat, name), {})
    return {
        "defective": bool(row.get("defect_classes")),
        "classes": row.get("defect_classes", []),
        "rationale": row.get("rationale", ""),
        "model": row.get("model"),
    }


def rate(k: int, n: int) -> float | None:
    return round(k / n, 2) if n else None


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.{digits}f}\u2009%"
    return str(value)


def render_page(summary: dict[str, Any]) -> str:
    import html as _html

    def esc(text: Any) -> str:
        return _html.escape(str(text))

    rows: list[str] = []
    defective = summary["defective_any"]
    per = summary["per_reader"]
    rows.append(
        '<h1 id="sagr-blind-audit-title">The live catalog read blind by two '
        "independent readers</h1>"
    )
    rows.append(
        f'<p id="sagr-blind-audit-summary">{summary["sampled"]} live names '
        f"drawn by seed {summary['draw']['seed']} from the "
        f"{summary['draw']['frame']} names in the {len(summary['draw']['groups'])} "
        f"physics-domain groups of at least {MIN_GROUP} names, plus "
        f"{summary['planted']} planted names, were read by "
        f"{esc(READERS[0]['model'])} and {esc(READERS[1]['model'])}. Each reader "
        f"returned every fixed class that applies, or none. "
        f"<strong>{defective['at_least_one']} of the "
        f"{summary['names_read']} sampled names with both readings "
        f"({fmt(defective['at_least_one_rate'])}, 95\u2009% CI "
        f"{fmt(defective['at_least_one_ci'][0])}–"
        f"{fmt(defective['at_least_one_ci'][1])}) were read defective by at "
        f"least one reader</strong>; reader A read "
        f"{per['A']['defective']} defective ({fmt(per['A']['rate'])}) and reader "
        f"B read {per['B']['defective']} ({fmt(per['B']['rate'])}). The two "
        f"agree on defective-or-not for {fmt(defective['observed_agreement'])} "
        f"of names (Cohen's \u03ba {defective['kappa']}). Spend was "
        f"<strong>USD {summary['spend']['total']:.2f}</strong> of the USD "
        f"{SPEND_CEILING:.0f} ceiling"
        + (
            f", excluding {summary['spend']['unknown_cost_calls']} failed "
            "call(s) whose billed cost the call layer does not return"
            if summary["spend"].get("unknown_cost_calls")
            else ""
        )
        + ". The planted names are reported in their own section and enter no "
        "sample share above.</p>"
    )
    if summary["missing_pairs"]:
        missing = "; ".join(
            f"<code>{esc(row['name'])}</code> (no reader {'/'.join(row['missing'])})"
            for row in summary["missing_pairs"]
        )
        rows.append(
            f'<p id="sagr-blind-audit-missing">{summary["names_read"]} of '
            f"{summary['requested']} sampled names carry both readings; "
            f"{missing} returned no content at the token cap after two attempts "
            "and is excluded from every paired statistic.</p>"
        )

    rows.append(
        '<h2 id="sagr-blind-audit-defective">Defective in any class</h2>'
        '<table id="sagr-blind-audit-defective-table"><thead><tr><th>Reading'
        "</th><th>Names</th><th>Share</th><th>95&#8201;% CI</th></tr></thead><tbody>"
    )
    for seat, label in (("A", "reader A"), ("B", "reader B")):
        item = per[seat]
        rows.append(
            f"<tr><td>{label}: defective</td><td>{item['defective']}</td>"
            f"<td>{fmt(item['rate'])}</td>"
            f"<td>{fmt(item['ci'][0])}–{fmt(item['ci'][1])}</td></tr>"
        )
    for label, key in (
        ("both readers", "both"),
        ("exactly one reader", "exactly_one"),
        ("neither reader", "neither"),
    ):
        rows.append(
            f"<tr><td>{label}</td><td>{defective['rows'][key]['names']}</td>"
            f"<td>{fmt(defective['rows'][key]['rate'])}</td>"
            f"<td>{fmt(defective['rows'][key]['ci'][0])}–{fmt(defective['rows'][key]['ci'][1])}</td></tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        '<p id="sagr-blind-audit-any-note">Defective in any class means the '
        "reader returned at least one class. A name with two defects is counted "
        "once here and once in each class it carries.</p>"
    )

    rows.append(
        '<h2 id="sagr-blind-audit-classes">Per class</h2>'
        '<table id="sagr-blind-audit-classes-table"><thead><tr><th>Class'
        "</th><th>Both</th><th>Only A</th><th>Only B</th><th>Any</th><th>Share "
        "of names</th><th>95&#8201;% CI</th><th>Agreement (95&#8201;% CI)</th><th>&kappa;</th>"
        "</tr></thead><tbody>"
    )
    for key in CLASS_KEYS:
        item = summary["per_class"][key]
        rows.append(
            f"<tr><td><code>{esc(key)}</code></td><td>{item['both']}</td>"
            f"<td>{item['only_A']}</td><td>{item['only_B']}</td>"
            f"<td>{item['any']}</td><td>{fmt(item['rate_any'])}</td>"
            f"<td>{fmt(item['ci_any'][0])}–{fmt(item['ci_any'][1])}</td>"
            f"<td>{fmt(item['agreement'])} ({fmt(item['agreement_ci'][0])}–{fmt(item['agreement_ci'][1])})</td><td>{item['kappa']}</td></tr>"
        )
    rows.append("</table>")

    rows.append(
        '<h2 id="sagr-blind-audit-class-definitions">The fixed class list</h2>'
        '<table id="sagr-blind-audit-class-definitions-table"><thead><tr>'
        "<th>Key</th><th>Definition</th></tr></thead><tbody>"
    )
    for item in summary["class_list_fixed"]:
        rows.append(
            f"<tr><td><code>{esc(item['key'])}</code></td>"
            f"<td>{esc(item['definition'])}</td></tr>"
        )
    rows.append(
        "</tbody></table><p>The list was fixed in the committed script before "
        "any name was read.</p>"
    )

    rows.append(
        '<h2 id="sagr-blind-audit-groups">Per group: counts only</h2>'
        '<p id="sagr-blind-audit-groups-note">Each group contributes about '
        f"{summary['sampled'] // max(1, len(summary['groups']))} names to a "
        f"sample of {summary['sampled']}. At that size a group's own defect "
        "rate carries a 95&#8201;% interval wider than twenty percentage "
        "points, so no group rate is stated; a group's rate is measured when "
        "the group is read in full. The counts are the reader A defective "
        "count, the reader B defective count, and the count both call "
        'defective.</p><table id="sagr-blind-audit-groups-table"><thead><tr>'
        "<th>Group</th><th>Live names</th><th>Sampled</th><th>A defective</th>"
        "<th>B defective</th><th>Both</th></tr></thead><tbody>"
    )
    for group, item in summary["groups"].items():
        rows.append(
            f"<tr><td>{esc(group)}</td><td>{item['frame']}</td>"
            f"<td>{item['sampled']}</td><td>{item['defective_A']}</td>"
            f"<td>{item['defective_B']}</td><td>{item['both']}</td></tr>"
        )
    rows.append("</tbody></table>")

    planted = summary["planted_detail"]
    rows.append(
        '<h2 id="sagr-blind-audit-planted">Planted names</h2>'
        f'<p id="sagr-blind-audit-planted-note">'
        f"{planted['defective']['count']} names a physicist's committed west "
        f"audit rules INCORRECT and {planted['sound']['count']} it rules "
        "correct were planted unmarked among the sample. Each citation is the "
        "audit row the label comes from. Recall on the defective plants: "
        f"A {fmt(planted['defective']['recall_A'])}, B "
        f"{fmt(planted['defective']['recall_B'])}. False-alarm rate on the "
        f"sound plants: A {fmt(planted['sound']['false_alarm_A'])}, B "
        f"{fmt(planted['sound']['false_alarm_B'])}. These rows are excluded "
        f"from every sample share above; {summary['plant_pairs']} of "
        f"{summary['planted']} planted names carry both readings.</p>"
        '<table id="sagr-blind-audit-planted-table"><thead><tr><th>Name</th>'
        "<th>Known</th><th>Citation</th><th>A</th><th>B</th></tr></thead><tbody>"
    )
    for row in planted["rows"]:
        rows.append(
            f"<tr><td><code>{esc(row['id'])}</code></td><td>{row['known']}</td>"
            f"<td><code>{esc(row['citation'])}</code></td>"
            f"<td>{'defective: ' + ', '.join(row['A']['classes']) if row['A']['defective'] else 'no defect'}</td>"
            f"<td>{'defective: ' + ', '.join(row['B']['classes']) if row['B']['defective'] else 'no defect'}</td></tr>"
        )
    rows.append("</tbody></table>")

    disagreements = summary["disagreements"]
    rows.append(
        '<h2 id="sagr-blind-audit-disagreements">Names the readers '
        "disagree on</h2>"
        f'<p id="sagr-blind-audit-disagreements-note">'
        f"{len(disagreements)} names carry a verdict the other reader does not. "
        "Both readings are listed for the lead"
        + ("; the first thirty are shown." if len(disagreements) > 30 else ".")
        + '</p><table id="sagr-blind-audit-disagreements-table"><thead><tr>'
        "<th>Name</th><th>Reader A</th><th>Reader B</th></tr></thead><tbody>"
    )
    for row in disagreements[:30]:
        rows.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(_reading_text(row['A']))}</td>"
            f"<td>{esc(_reading_text(row['B']))}</td></tr>"
        )
    rows.append("</tbody></table>")

    prompts = summary["prompts"]
    rows.append(
        '<h2 id="sagr-blind-audit-blindness">What a reader saw</h2>'
        '<p id="sagr-blind-audit-blindness-note">A prompt is built from a '
        "fixed allow-list: name, stored description, documentation and unit, "
        "assigned physics domain, DD text of the sources, the text of their "
        "enclosing structures, and other live names produced by the same "
        "sources. No stage, score, review history or planted marker is read. "
        "The committed script asserts that before every call.</p>"
        '<h3>System prompt</h3><pre id="sagr-blind-audit-system-prompt">'
        f"{esc(prompts['system'])}</pre>"
        '<h3>One full user prompt</h3><pre id="sagr-blind-audit-user-prompt">'
        f"{esc(prompts['user'])}</pre>"
    )

    rows.append(
        '<h2 id="sagr-blind-audit-spend">Spend and calls</h2>'
        '<table id="sagr-blind-audit-spend-table"><thead><tr><th>Seat</th>'
        "<th>Model</th><th>Readings</th><th>USD</th><th>Tokens</th></tr>"
        "</thead><tbody>"
    )
    for seat in ("A", "B"):
        item = per[seat]
        model = next(r["model"] for r in READERS if r["seat"] == seat)
        rows.append(
            f"<tr><td>{seat}</td><td><code>{esc(model)}</code></td>"
            f"<td>{item['read']}</td><td>{item['spend']:.4f}</td>"
            f"<td>{item['tokens']}</td></tr>"
        )
    rows.append(
        f"</tbody></table><p>Total USD {summary['spend']['total']:.4f}. "
        f"Unanswered calls: {summary['error_count']}.</p>"
        "</main></body></html>"
    )

    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="docs-project" content="imas-codex">\n'
        '<meta name="reckon-type" content="evidence-fragment">\n'
        '<meta name="plan-slug" content="sn-audited-group-release">\n'
        '<meta name="node-id" content="sagr-blind-audit">\n'
        "<title>The live catalog read blind by two independent readers</title>\n"
        "</head>\n<body>\n<main>\n" + "\n".join(rows)
    )


def _reading_text(reading: dict[str, Any]) -> str:
    classes = ", ".join(reading.get("classes") or []) or "no defect"
    rationale = reading.get("rationale") or ""
    return f"{classes}. {rationale}".strip()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_items() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    from imas_codex.standard_names.source_paths import split_source_paths

    gc = graph_client()
    try:
        live = load_live_names(gc)
        by_id = {row["id"]: row for row in live}
        sampled, draw = draw_sample(live)
        sampled_ids = {row["id"] for row in sampled}
        live_by_source = names_by_dd_source(live)
        live_sources = {
            row["id"]: set(
                split_source_paths(row.get("source_paths") or []).get("dd", [])
            )
            for row in live
        }
        plants, plant_meta = load_plants(live_sources, sampled_ids)
        draw["plants"] = plant_meta
        items = sampled + [
            {
                "id": plant["id"],
                "description": by_id[plant["id"]]["description"],
                "documentation": by_id[plant["id"]]["documentation"],
                "unit": by_id[plant["id"]]["unit"],
                "physics_domain": by_id[plant["id"]]["physics_domain"],
                "source_paths": by_id[plant["id"]]["source_paths"],
            }
            for plant in plants
        ]
        load_dd_context(gc, items)
        attach_duplicate_candidates(items, live_by_source)
    finally:
        gc.close()
    random.Random(SEED).shuffle(items)
    return items, draw, plants


def pipeline_models() -> set[str]:
    """Every model a production seat or a selectable review profile names.

    Reads the same configuration surface the pipeline does, through the
    accessors that own it (:mod:`imas_codex.settings`), rather than walking
    ``pyproject.toml`` for keys: a seat added with a different key name or a
    profile added later is then still in the set.
    """
    from imas_codex import settings

    found = {
        settings.get_model(section)
        for section in settings.MODEL_SECTIONS
        if section != "embedding"
    }
    found.update(settings.get_sn_review_names_models())
    found.update(settings.get_sn_review_docs_models())
    for profile in settings._VALID_REVIEWER_PROFILES:
        found.update(settings.get_sn_review_profile_models(profile))
    return found


def assert_readers_not_pipeline(reviewers: tuple[dict[str, str], ...]) -> None:
    """Refuse a reader the pipeline composes or reviews with."""
    used = pipeline_models()
    for reviewer in reviewers:
        if reviewer["model"] in used:
            raise AssertionError(
                f"reader {reviewer['seat']} uses a pipeline model: {reviewer['model']}"
            )


def _self_test() -> int:
    """Prove the sample statistics cannot count a planted name.

    Two sampled names and two plants are enough: if plant rows reach any
    sample share, ``names_read`` or a per-reader or per-class count moves.
    The reader guard is checked in the same run so its positive control sits
    beside the code it protects.
    """
    sampled = [
        {"id": "s1", "physics_domain": "equilibrium"},
        {"id": "s2", "physics_domain": "equilibrium"},
    ]
    plants = [
        {
            "id": "p1",
            "known": "defective",
            "citation": "synthetic:1",
            "audited_source_path": "a/b",
            "auditor_ruled_description_wrong": False,
        },
        {
            "id": "p2",
            "known": "sound",
            "citation": "synthetic:2",
            "audited_source_path": "c/d",
            "auditor_ruled_description_wrong": False,
        },
    ]

    def reading(name: str, seat: str, classes: list[str], cost: float | None) -> dict:
        return {
            "name": name,
            "seat": seat,
            "model": "test/model",
            "defect_classes": classes,
            "rationale": "",
            "cost": cost,
            "tokens": 1,
        }

    readings = [
        reading("s1", "A", ["unit_wrong"], 0.01),
        reading("s1", "B", ["unit_wrong"], 0.01),
        reading("s2", "A", [], 0.01),
        reading("s2", "B", [], 0.01),
        reading("p1", "A", ["unit_wrong"], 0.01),
        reading("p1", "B", ["unit_wrong"], 0.01),
        reading("p2", "A", ["unit_wrong"], 0.01),
        reading("p2", "B", [], 0.01),
        {**reading("s1", "A", [], None), "error": "ValueError: billed"},
    ]
    summary = summarize(
        readings_raw=readings,
        sampled=sampled,
        plants=plants,
        draw={"seed": 0, "frame": 2, "groups": {"equilibrium": 2}, "excluded": {}},
        prompts={"system": "", "user": ""},
    )
    checks = {
        "sample pairs exclude the planted names": summary["names_read"] == 2,
        "sample defective count excludes the plants": (
            summary["defective_any"]["at_least_one"] == 1
        ),
        "per-reader count excludes the plants": (
            summary["per_reader"]["A"]["defective"] == 1
            and summary["per_reader"]["B"]["defective"] == 1
        ),
        "per-class count excludes the plants": (
            summary["per_class"]["unit_wrong"]["any"] == 1
        ),
        "sample shares are over the sample": (
            summary["defective_any"]["at_least_one_rate"] == 0.5
        ),
        "plant recall is measured on the plants": (
            summary["planted_detail"]["defective"]["recall_A"] == 1.0
            and summary["planted_detail"]["sound"]["false_alarm_A"] == 1.0
        ),
        "a failed call's cost is unknown, not free": (
            summary["spend"]["unknown_cost_calls"] == 1
        ),
        "the delivered readers pass the pipeline-model guard": True,
    }
    try:
        assert_readers_not_pipeline(READERS)
    except AssertionError as exc:  # pragma: no cover - reported, not raised
        checks["the delivered readers pass the pipeline-model guard"] = False
        print(f"guard refused a delivered reader: {exc}")
    refused = False
    try:
        assert_readers_not_pipeline(
            ({"seat": "A", "model": "openrouter/openai/gpt-6-luna"},)
        )
    except AssertionError:
        refused = True
    checks["the guard refuses a configured pipeline model"] = refused
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.check:
        return _self_test()

    assert_readers_not_pipeline(READERS)
    items, draw, plants = build_items()
    if args.limit:
        items = items[: args.limit]
    prompts = {
        "system": system_prompt(),
        "user": render_user_prompt(items[0]),
    }
    if args.dry_run:
        print(json.dumps(draw, indent=1))
        print(f"{len(items)} items, first {len(items[0]['dd_docs'])} DD docs")
        print("--- SYSTEM ---")
        print(prompts["system"])
        print("--- USER ---")
        print(prompts["user"])
        return 0

    if args.report or args.run:
        if args.run:
            asyncio.run(run_reading(items, READERS))
        readings = list(load_progress().values())
        summary = summarize(
            readings_raw=readings,
            sampled=[
                row for row in items if row["id"] not in {p["id"] for p in plants}
            ],
            plants=plants,
            draw=draw,
            prompts=prompts,
        )
        OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n")
        OUT_HTML.write_text(render_page(summary) + "\n")
        print(
            f"wrote {OUT_JSON.name} and {OUT_HTML.name}; "
            f"spend {summary['spend']['total']:.2f} USD"
        )
        return 0

    if args.persist:
        return persist(readings=list(load_progress().values()))

    parser.print_help()
    return 2


def stored_snapshot(ids: list[str]) -> dict[str, str]:
    """A digest of every stored field an audit must leave unchanged."""
    gc = graph_client()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName) WHERE sn.id IN $ids
            RETURN sn.id AS id,
                   coalesce(sn.description,'') AS description,
                   coalesce(sn.documentation,'') AS documentation,
                   coalesce(sn.unit,'') AS unit,
                   coalesce(sn.physics_domain,'') AS physics_domain,
                   coalesce(sn.name_stage,'') AS name_stage,
                   coalesce(sn.status,'') AS status
            """,
            ids=ids,
        )
    finally:
        gc.close()
    return {
        str(row["id"]): hashlib.sha256(
            json.dumps(
                [
                    row.get("description"),
                    row.get("documentation"),
                    row.get("unit"),
                    row.get("physics_domain"),
                    row.get("name_stage"),
                    row.get("status"),
                ]
            ).encode()
        ).hexdigest()
        for row in rows
    }


def persist(readings: list[dict[str, Any]]) -> int:
    """Write each reading as a non-deciding review record."""
    from imas_codex.standard_names.graph_store import graph_catalog
    from imas_codex.standard_names.review.pipeline import _build_review_record

    records = []
    reviewed_at = datetime.now(UTC).isoformat()
    group = f"blind-audit:{SEED}"
    for reading in readings:
        if reading.get("error"):
            continue
        judgment = {
            "defect_classes": {
                "answer": reading["defect_classes"],
                "probs": None,
                "confidence": None,
            }
        }
        item = {
            "id": reading["name"],
            "description": "",
            "unit": "",
        }
        record = _build_review_record(
            item,
            model=reading["model"],
            is_canonical=False,
            reviewed_at=reviewed_at,
            comments=json.dumps(
                {
                    "reader_seat": reading["seat"],
                    "defect_classes": reading["defect_classes"],
                    "rationale": reading["rationale"],
                    "audit": "blind-audit",
                }
            ),
            cost_usd=reading["cost"],
            tokens_in=reading.get("input_tokens"),
            tokens_out=reading.get("output_tokens"),
            judgment=judgment,
            judgment_model=reading["model"],
        )
        if not record:
            continue
        record.update(
            id=f"{reading['name']}:blind-audit:{reading['seat']}",
            standard_name_id=reading["name"],
            review_axis="name",
            cycle_index=-1,
            review_group_id=group,
            resolution_role="audit",
            resolution_method="blind_audit",
            llm_tokens_cached_read=reading.get("cache_read_tokens", 0),
            llm_tokens_cached_write=0,
        )
        records.append(record)
    ids = sorted({reading["name"] for reading in readings if not reading.get("error")})
    before = stored_snapshot(ids)
    written = graph_catalog.write_reviews(records, skip_cost=True)
    after = stored_snapshot(ids)
    changed = sorted(
        name for name, digest in before.items() if after.get(name) != digest
    )
    verification = {
        "records_written": int(written),
        "names_snapshotted": len(ids),
        "stored_text_changed": changed,
        "method": (
            "sha256 of description, documentation, unit, physics_domain, "
            "name_stage and status for every audited name, before and after "
            "the audit-record write"
        ),
    }
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "persist-verification.json").write_text(
        json.dumps(verification, indent=1) + "\n"
    )
    print(json.dumps(verification, indent=1))
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
