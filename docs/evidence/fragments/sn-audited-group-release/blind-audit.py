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

# The frontier third reader. Chosen from the project's priced seat catalog
# (tests/config/data/model-catalog-2026-11-15.json, USD per million tokens) by
# the rule in "Third reader" below: a family that is neither reader's, no model
# a production seat or selectable review profile names, the largest capability
# tier the catalog prices, and the larger measured context window of the two
# such candidates.
THIRD_SEAT = "C"
THIRD_MODEL = "openrouter/openai/gpt-6-astra"
THIRD_FAMILY = "openai"
THIRD_REASONING_EFFORT = "high"
THIRD_CONTROL_SIZE = 60
THIRD_SEED = 20261117
THIRD_SPEND_CEILING = 20.0
THIRD_PROJECTION_AFTER = 30
THIRD_STATE = STATE / "third-reader"
THIRD_JSON = FRAGMENTS / "frontier-third-reader.json"
THIRD_HTML = FRAGMENTS / "sagr-frontier-third-reader.html"
PRIOR_GROUP = f"blind-audit:{SEED}"
# Pre-declared before the third reading is taken: the bar at which this reader
# may decide a group audit on its own.
READER_RECALL_BAR = 0.90
READER_FALSE_ALARM_BAR = 0.10

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
    from imas_codex.standard_names.worker_tasks.worker_common import (
        _ANCESTOR_CONTEXT_QUERY,
        ancestor_statements,
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
        row["ancestors"] = ancestor_statements(ancestor_rows)


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
    reader: dict[str, str],
    name: dict[str, Any],
    spend: dict[str, float],
    *,
    ceiling: float = SPEND_CEILING,
) -> dict[str, Any]:
    from imas_codex.discovery.base.llm import acall_llm_structured

    if spend["total"] > ceiling:
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
            reasoning_effort=reader.get("reasoning_effort"),
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


def progress_path(state_dir: Path | None = None) -> Path:
    base = Path(state_dir) if state_dir else STATE
    base.mkdir(parents=True, exist_ok=True)
    return base / "records.jsonl"


def load_progress(
    state_dir: Path | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    path = progress_path(state_dir)
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
    items: list[dict[str, Any]],
    reviewers: tuple[dict[str, str], ...],
    *,
    state_dir: Path | None = None,
    ceiling: float = SPEND_CEILING,
    projection_after: int = 0,
) -> list[dict[str, Any]]:
    """Read every (seat, name) pair, appending each record to ``records.jsonl``.

    ``projection_after`` bounds an unpriced run: once that many new calls have
    returned a cost, the spend is projected to all remaining pairs and the run
    stops before the ceiling is reached rather than after it.
    """
    done = load_progress(state_dir)
    path = progress_path(state_dir)
    gate = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    spend: dict[str, Any] = {"total": 0.0}
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
            record = await judge_one(reader, item, spend, ceiling=ceiling)
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    results: list[dict[str, Any]] = list(done.values())
    tasks = [asyncio.create_task(one(reader, item)) for reader, item in todo]
    finished_new = 0
    for finished in asyncio.as_completed(tasks):
        record = await finished
        results.append(record)
        finished_new += 1
        if projection_after and finished_new == projection_after:
            costed = [r for r in results if r.get("cost") is not None]
            spent = sum(float(r["cost"]) for r in costed)
            projected = spent / len(costed) * len(todo) if costed else None
            projection = {
                "calls_priced": len(costed),
                "spend_usd": round(spent, 4),
                "projected_usd": None if projected is None else round(projected, 4),
                "ceiling": ceiling,
                "pairs": len(todo),
            }
            base = Path(state_dir) if state_dir else STATE
            (base / "spend-projection.json").write_text(
                json.dumps(projection, indent=1) + "\n"
            )
            print(f"spend projection: {projection}", flush=True)
            if projected is not None and projected > ceiling:
                spend["stop_reason"] = (
                    f"projected USD {projected:.2f} over the USD {ceiling:.2f} "
                    f"ceiling after {len(costed)} priced calls"
                )
                print(f"stopping: {spend['stop_reason']}", flush=True)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                break
        if finished_new % 40 == 0:
            print(
                f"  {finished_new} new readings, {spend['total']:.3f} USD", flush=True
            )
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
# Third reader
# --------------------------------------------------------------------------- #


def third_reader() -> dict[str, str]:
    """The frontier seat: model, family and reasoning effort."""
    return {
        "seat": THIRD_SEAT,
        "model": THIRD_MODEL,
        "family": THIRD_FAMILY,
        "reasoning_effort": THIRD_REASONING_EFFORT,
    }


def model_selection() -> dict[str, Any]:
    """Why this model qualifies, from the catalog and the audited listing.

    The catalog prices two frontier candidates of eligible families at the top
    capability tier of their line: ``gpt-6-astra`` (USD 10 / 50 per million
    tokens) and ``claude-fable-5.2`` (USD 8 / 40). The OpenRouter listing the
    catalog was read from measures their context windows at 1,050,000 and
    1,000,000 tokens; the larger-window candidate is chosen.
    """
    return {
        "catalog": "tests/config/data/model-catalog-2026-11-15.json",
        "context_source": (
            "docs/evidence/fragments/sn-definitions-by-construction/"
            "openrouter-models-20261114T123028Z.json.gz"
        ),
        "candidates": [
            {
                "model": "openrouter/openai/gpt-6-astra",
                "family": "openai",
                "prompt_usd_per_mtok": 10.0,
                "completion_usd_per_mtok": 50.0,
                "context_window": 1_050_000,
                "pipeline_model": False,
            },
            {
                "model": "openrouter/anthropic/claude-fable-5.2",
                "family": "anthropic",
                "prompt_usd_per_mtok": 8.0,
                "completion_usd_per_mtok": 40.0,
                "context_window": 1_000_000,
                "pipeline_model": False,
            },
        ],
        "chosen": THIRD_MODEL,
        "reason": (
            "Neither reader's family (Google, Moonshot) and no production seat "
            "or selectable review profile names it; of the two frontier "
            "candidates the catalog prices, it carries the larger measured "
            "context window (1,050,000 against 1,000,000 tokens). "
            "claude-opus-4.1 is priced higher (USD 15 / 75) but is a 2025-08 "
            "release the catalog's own release list supersedes within its "
            "line cap, and its 200,000-token window is the smallest of the "
            "priced frontier entries."
        ),
        "reasoning_effort": THIRD_REASONING_EFFORT,
    }


def prior_audit_readings() -> dict[str, dict[str, dict[str, Any]]]:
    """The two audit readers' readings, from the rows they left in the graph.

    The audit's progress file is not kept, so the persisted audit review rows
    are the one record of which names were read and what each returned. A name
    absent from this mapping was not in the audit; a seat absent from a name's
    mapping returned no content for it.
    """
    gc = graph_client()
    try:
        rows = gc.query(
            """
            MATCH (r:StandardNameReview)
            WHERE r.review_group_id = $group AND r.resolution_role = 'audit'
            RETURN r.standard_name_id AS name, r.comments AS comments
            """,
            group=PRIOR_GROUP,
        )
    finally:
        gc.close()
    readings: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        try:
            payload = json.loads(row.get("comments") or "{}")
        except json.JSONDecodeError:
            continue
        seat = str(payload.get("reader_seat") or "")
        if seat not in ("A", "B"):
            continue
        readings.setdefault(str(row["name"]), {})[seat] = {
            "classes": [
                key for key in payload.get("defect_classes") or [] if key in CLASS_KEYS
            ],
            "rationale": str(payload.get("rationale") or ""),
        }
    return readings


def prior_plant_rows() -> list[dict[str, Any]]:
    """The 40 planted names and their independent west-audit verdicts."""
    summary = json.loads(OUT_JSON.read_text())
    return list(summary["planted_detail"]["rows"])


def third_selection(
    readings: dict[str, dict[str, dict[str, Any]]],
    plants: list[dict[str, Any]],
) -> dict[str, Any]:
    """Planted names, every flagged sample name, and a seeded control.

    Refuses to proceed when the rows it read are not the audited set: the
    counts must match the committed audit record, otherwise the reading is
    over a different catalog and its rates mean nothing.
    """
    plant_ids = {row["id"] for row in plants}
    sample = sorted(name for name in readings if name not in plant_ids)
    complete = [
        name for name in sample if "A" in readings[name] and "B" in readings[name]
    ]
    flagged = [
        name
        for name in complete
        if readings[name]["A"]["classes"] or readings[name]["B"]["classes"]
    ]
    flagged_both = [
        name
        for name in flagged
        if readings[name]["A"]["classes"] and readings[name]["B"]["classes"]
    ]
    sound = [
        name
        for name in complete
        if not readings[name]["A"]["classes"] and not readings[name]["B"]["classes"]
    ]
    control = sorted(random.Random(THIRD_SEED).sample(sound, THIRD_CONTROL_SIZE))
    incomplete = [name for name in sample if name not in set(complete)]

    prior = json.loads(OUT_JSON.read_text())
    expected = {
        "plants": prior["planted"],
        "sample": prior["sampled"],
        "complete": prior["names_read"],
        "flagged": prior["defective_any"]["at_least_one"],
        "sound": prior["defective_any"]["neither"],
    }
    observed = {
        "plants": len(plants),
        "sample": len(sample),
        "complete": len(complete),
        "flagged": len(flagged),
        "sound": len(sound),
    }
    mismatched = {
        key: (observed[key], expected[key])
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatched:
        raise RuntimeError(
            "the persisted audit rows are not the audited set: "
            f"{mismatched} (observed, expected)"
        )
    if len(flagged) + len(sound) + len(incomplete) != len(sample):
        raise RuntimeError("sample partitions do not add up")
    return {
        "seed": THIRD_SEED,
        "control_size": THIRD_CONTROL_SIZE,
        "plants": plants,
        "sample": sample,
        "complete": complete,
        "flagged": flagged,
        "flagged_both": flagged_both,
        "flagged_one": [n for n in flagged if n not in set(flagged_both)],
        "sound": sound,
        "control": control,
        "incomplete": incomplete,
        "counts": {
            "audited_names": len(readings),
            "sample": len(sample),
            "plants": len(plants),
            "complete": len(complete),
            "flagged": len(flagged),
            "flagged_both": len(flagged_both),
            "flagged_one": len(flagged) - len(flagged_both),
            "sound": len(sound),
            "control": len(control),
            "incomplete": len(incomplete),
        },
    }


def load_third_items(
    ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], list[str]]:
    """The stored record and DD context of every selected name.

    Reads by id rather than from the live set: a name superseded since the
    audit is still the name the first two readers saw, so its stored text is
    read and its stage is recorded separately, never in the prompt.
    """
    from imas_codex.standard_names.source_reads import source_paths_projection

    gc = graph_client()
    try:
        live = load_live_names(gc)
        live_by_source = names_by_dd_source(live)
        rows = gc.query(
            f"""
            MATCH (sn:StandardName) WHERE sn.id IN $ids
            RETURN sn.id AS id,
                   coalesce(sn.description,'') AS description,
                   coalesce(sn.documentation,'') AS documentation,
                   coalesce(sn.unit,'') AS unit,
                   coalesce(sn.physics_domain,'') AS physics_domain,
                   coalesce(sn.name_stage,'') AS name_stage,
                   coalesce(sn.status,'') AS status,
                   {source_paths_projection()} AS sources
            """,
            ids=sorted(ids),
        )
        stage: dict[str, dict[str, str]] = {}
        for row in rows:
            row["source_paths"] = list(row.get("sources") or [])
            row.pop("sources", None)
            stage[str(row["id"])] = {
                "name_stage": str(row.pop("name_stage") or ""),
                "status": str(row.pop("status") or ""),
            }
        missing = sorted(set(ids) - set(stage))
        load_dd_context(gc, rows)
        attach_duplicate_candidates(rows, live_by_source)
    finally:
        gc.close()
    return rows, stage, missing


def third_items(selection: dict[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    ids = (
        [row["id"] for row in selection["plants"]]
        + list(selection["flagged"])
        + list(selection["control"])
    )
    rows, stage, missing = load_third_items(ids)
    if missing:
        raise RuntimeError(f"{len(missing)} selected names are absent from the graph")
    random.Random(THIRD_SEED).shuffle(rows)
    return rows, stage


def bootstrap_defect_rate(
    flagged_hits: list[int],
    control_hits: list[int],
    n_flagged: int,
    n_sound: int,
    draws: int = 4000,
) -> tuple[float, float]:
    """Percentile interval for the two-stratum estimate.

    Resamples the flagged names read and the control names read with
    replacement, and scales the control's proportion to the unread remainder of
    the both-sound stratum.
    """
    rng = random.Random(THIRD_SEED + 1)
    n_control = len(control_hits)
    total = n_flagged + n_sound
    values: list[float] = []
    for _ in range(draws):
        flagged = sum(
            flagged_hits[rng.randrange(len(flagged_hits))]
            for _ in range(len(flagged_hits))
        )
        control = sum(control_hits[rng.randrange(n_control)] for _ in range(n_control))
        p_flagged = flagged / len(flagged_hits)
        p_control = control / n_control
        values.append((p_flagged * n_flagged + p_control * n_sound) / total)
    values.sort()
    return values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def summarize_third(
    *,
    readings_raw: list[dict[str, Any]],
    selection: dict[str, Any],
    readings: dict[str, dict[str, dict[str, Any]]],
    prompts: dict[str, str],
    revision: str,
    guard: dict[str, Any],
    stage: dict[str, dict[str, str]],
) -> dict[str, Any]:
    third = {
        (r["name"]): r
        for r in readings_raw
        if not r.get("error") and r["seat"] == THIRD_SEAT
    }
    errors = [r for r in readings_raw if r.get("error")]

    def classes(name: str) -> list[str]:
        return list(third.get(name, {}).get("defect_classes") or [])

    def defective(name: str) -> bool:
        return bool(classes(name))

    def reason(name: str) -> str:
        return str(third.get(name, {}).get("rationale") or "")

    def prior_classes(name: str, seat: str) -> list[str]:
        return list(readings.get(name, {}).get(seat, {}).get("classes") or [])

    plant_rows = []
    for plant in selection["plants"]:
        name = plant["id"]
        plant_rows.append(
            {
                "name": name,
                "known": plant["known"],
                "citation": plant.get("citation", ""),
                "audited_source_path": plant.get("audited_source_path", ""),
                "A": {
                    "classes": prior_classes(name, "A"),
                    "rationale": readings.get(name, {})
                    .get("A", {})
                    .get("rationale", ""),
                },
                "B": {
                    "classes": prior_classes(name, "B"),
                    "rationale": readings.get(name, {})
                    .get("B", {})
                    .get("rationale", ""),
                },
                "C": {"classes": classes(name), "rationale": reason(name)},
                "read": name in third,
            }
        )
    defective_plants = [row for row in plant_rows if row["known"] == "defective"]
    sound_plants = [row for row in plant_rows if row["known"] == "sound"]
    recall_c = sum(1 for row in defective_plants if row["C"]["classes"])
    false_alarm_c = sum(1 for row in sound_plants if row["C"]["classes"])
    recall_a = sum(1 for row in defective_plants if row["A"]["classes"])
    recall_b = sum(1 for row in defective_plants if row["B"]["classes"])
    alarm_a = sum(1 for row in sound_plants if row["A"]["classes"])
    alarm_b = sum(1 for row in sound_plants if row["B"]["classes"])
    wrong = [
        row
        for row in plant_rows
        if (row["known"] == "defective" and not row["C"]["classes"])
        or (row["known"] == "sound" and row["C"]["classes"])
    ]

    def share(count: int, n: int) -> dict[str, Any]:
        low, high = wilson(count, n)
        return {
            "names": count,
            "total": n,
            "rate": round(count / n, 4) if n else None,
            "ci": [round(low, 4), round(high, 4)],
        }

    flagged_rows = []
    for name in selection["flagged"]:
        if name not in third:
            continue
        earlier = (
            "both" if prior_classes(name, "A") and prior_classes(name, "B") else "one"
        )
        flagged_rows.append(
            {
                "name": name,
                "earlier": earlier,
                "A": prior_classes(name, "A"),
                "B": prior_classes(name, "B"),
                "C": classes(name),
                "C_rationale": reason(name),
            }
        )
    by_class: dict[str, dict[str, int]] = {
        key: {"both": 0, "one": 0, "any": 0} for key in CLASS_KEYS
    }
    for row in flagged_rows:
        if not row["C"]:
            continue
        for key in row["C"]:
            by_class[key][row["earlier"]] += 1
            by_class[key]["any"] += 1
    confirmed = [row for row in flagged_rows if row["C"]]
    flagged_summary = {
        "read": len(flagged_rows),
        "confirmed": len(confirmed),
        "confirmed_both": sum(1 for row in confirmed if row["earlier"] == "both"),
        "confirmed_one": sum(1 for row in confirmed if row["earlier"] == "one"),
        "flagged_both": selection["counts"]["flagged_both"],
        "flagged_one": selection["counts"]["flagged_one"],
        "by_class": by_class,
        "rows": flagged_rows,
    }

    control_rows = []
    for name in selection["control"]:
        if name not in third:
            continue
        control_rows.append(
            {
                "name": name,
                "A": prior_classes(name, "A"),
                "B": prior_classes(name, "B"),
                "C": classes(name),
                "C_rationale": reason(name),
            }
        )
    control_by_class = {
        key: sum(1 for row in control_rows if key in row["C"]) for key in CLASS_KEYS
    }
    control_summary = {
        "read": len(control_rows),
        "defective": sum(1 for row in control_rows if row["C"]),
        "by_class": control_by_class,
        "rows": control_rows,
    }

    n_flagged = selection["counts"]["flagged"]
    n_sound = selection["counts"]["sound"]
    n_control = len(control_rows)
    control_hits = [1 if row["C"] else 0 for row in control_rows]
    flagged_hits = [1 if row["C"] else 0 for row in flagged_rows]
    p_flagged = sum(flagged_hits) / len(flagged_hits) if flagged_hits else 0.0
    p_control = sum(control_hits) / n_control if n_control else 0.0
    point = (p_flagged * n_flagged + p_control * n_sound) / (n_flagged + n_sound)
    if control_hits and flagged_hits:
        low, high = bootstrap_defect_rate(
            flagged_hits, control_hits, n_flagged, n_sound
        )
    else:
        low, high = 0.0, 1.0
    reestimate = {
        "sample_names": n_flagged + n_sound,
        "flagged_stratum": {
            "names": n_flagged,
            "read": len(flagged_rows),
            "confirmed": sum(flagged_hits),
            "rate": round(p_flagged, 4),
        },
        "sound_stratum": {
            "names": n_sound,
            "control_read": n_control,
            "control_defective": sum(control_hits),
            "control_rate": round(p_control, 4),
        },
        "point": round(point, 4),
        "ci": [round(low, 4), round(high, 4)],
        "implied_defective_names": round(
            p_flagged * n_flagged + p_control * n_sound, 1
        ),
        "prior": {
            "both_readers": 0.248,
            "both_readers_ci": [0.205, 0.296],
            "either_reader": 0.513,
            "either_reader_ci": [0.460, 0.566],
        },
        "assumptions": [
            "The third reader's verdict is taken as the defect criterion; the "
            "planted names measure how far that criterion is from the "
            "physicist's, and the two rates are stated together.",
            "Planted names are outside both strata and enter no count.",
            "The 174 names at least one earlier reader flagged are read in "
            "full, so that stratum is measured, not extrapolated.",
            "The 60 control names are a simple random draw from the 165 names "
            "both earlier readers read sound; the 105 unread names of that "
            "stratum are assumed to carry the same defect rate.",
            "The one sampled name with no reader B reading is in neither "
            "stratum; the estimate covers 339 of the 340 sampled names.",
            "A rate against a named criterion is not a repeat-measurement "
            "estimate: this rate cannot be quoted beside the two earlier "
            "figures as though all three measured the same thing.",
        ],
    }

    disagreements = []
    for name in sorted(set(selection["flagged"]) | set(selection["control"])):
        if name not in third:
            continue
        a = bool(prior_classes(name, "A"))
        b = bool(prior_classes(name, "B"))
        c = defective(name)
        if c != a and c != b:
            disagreements.append(
                {
                    "name": name,
                    "set": (
                        "flagged" if name in set(selection["flagged"]) else "control"
                    ),
                    "A": {
                        "classes": prior_classes(name, "A"),
                        "rationale": readings.get(name, {})
                        .get("A", {})
                        .get("rationale", ""),
                    },
                    "B": {
                        "classes": prior_classes(name, "B"),
                        "rationale": readings.get(name, {})
                        .get("B", {})
                        .get("rationale", ""),
                    },
                    "C": {"classes": classes(name), "rationale": reason(name)},
                }
            )

    recall_share = share(recall_c, len(defective_plants))
    alarm_share = share(false_alarm_c, len(sound_plants))
    sufficient = (
        recall_share["rate"] is not None
        and recall_share["rate"] >= READER_RECALL_BAR
        and alarm_share["rate"] is not None
        and alarm_share["rate"] <= READER_FALSE_ALARM_BAR
    )
    verdict = {
        "sufficient_to_decide": sufficient,
        "bar": {
            "recall_at_least": READER_RECALL_BAR,
            "false_alarm_at_most": READER_FALSE_ALARM_BAR,
        },
        "sentence": (
            f"{'Yes' if sufficient else 'No'}: on the 40 planted names "
            f"{THIRD_MODEL} recalls {recall_c} of {len(defective_plants)} "
            f"known-defective ({fmt(recall_share['rate'])}, 95 % CI "
            f"{fmt(recall_share['ci'][0])}–{fmt(recall_share['ci'][1])}) and "
            f"calls {false_alarm_c} of {len(sound_plants)} known-sound names "
            f"defective ({fmt(alarm_share['rate'])}, 95 % CI "
            f"{fmt(alarm_share['ci'][0])}–{fmt(alarm_share['ci'][1])}), "
            f"{'meeting' if sufficient else 'missing'} the pre-declared bar of "
            f"{fmt(READER_RECALL_BAR)} recall with at most "
            f"{fmt(READER_FALSE_ALARM_BAR)} false alarms; on the sample it "
            f"confirms {len(confirmed)} of {len(flagged_rows)} earlier-flagged "
            f"names and reads {control_summary['defective']} of "
            f"{control_summary['read']} earlier-sound names defective, which "
            f"places the catalog's defective share at "
            f"{fmt(reestimate['point'])} (95 % CI "
            f"{fmt(reestimate['ci'][0])}–{fmt(reestimate['ci'][1])})."
        ),
    }

    costs = [float(r.get("cost") or 0.0) for r in third.values()]
    spend = {
        "usd": round(sum(costs), 4),
        "ceiling": THIRD_SPEND_CEILING,
        "calls_priced": sum(1 for r in third.values() if r.get("cost") is not None),
        "unknown_cost_calls": len(errors),
        "tokens": sum(int(r.get("tokens", 0)) for r in third.values()),
    }
    not_live = [
        name
        for name in sorted(stage)
        if stage[name]["name_stage"] == "superseded"
        or stage[name]["status"] == "superseded"
    ]
    return {
        "node": "sagr-frontier-third-reader",
        "generated_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "reader": third_reader(),
        "model_selection": model_selection(),
        "guard": guard,
        "class_list_fixed": [
            {"key": key, "definition": text} for key, text in DEFECT_CLASSES
        ],
        "selection": {
            "seed": selection["seed"],
            "control_size": selection["control_size"],
            "counts": selection["counts"],
            "planted_ids": [row["id"] for row in selection["plants"]],
            "flagged_ids": list(selection["flagged"]),
            "control_ids": list(selection["control"]),
            "incomplete_ids": list(selection["incomplete"]),
        },
        "planted": {
            "recall_C": recall_share,
            "false_alarm_C": alarm_share,
            "recall_A": share(recall_a, len(defective_plants)),
            "recall_B": share(recall_b, len(defective_plants)),
            "false_alarm_A": share(alarm_a, len(sound_plants)),
            "false_alarm_B": share(alarm_b, len(sound_plants)),
            "rows": plant_rows,
            "wrong": wrong,
        },
        "flagged": flagged_summary,
        "control": control_summary,
        "reestimate": reestimate,
        "verdict": verdict,
        "disagreement_rule": (
            "a sampled name whose third-reader defective-or-not verdict "
            "differs from both earlier readers' verdicts"
        ),
        "disagreements": disagreements,
        "spend": spend,
        "errors": errors[:20],
        "error_count": len(errors),
        "missing_readings": sorted(
            set(
                [row["id"] for row in selection["plants"]]
                + list(selection["flagged"])
                + list(selection["control"])
            )
            - set(third)
        ),
        "names_no_longer_live": not_live,
        "prompts": prompts,
        "environment": {
            "python": sys.version.split()[0],
            "cwd": str(Path.cwd()),
            "concurrency": CONCURRENCY,
            "max_tokens": MAX_TOKENS,
            "timeout_s": TIMEOUT_S,
        },
    }


def render_third_page(summary: dict[str, Any]) -> str:
    import html as _html

    def esc(text: Any) -> str:
        return _html.escape(str(text))

    planted = summary["planted"]
    flagged = summary["flagged"]
    control = summary["control"]
    estimate = summary["reestimate"]
    verdict = summary["verdict"]
    counts = summary["selection"]["counts"]
    rows: list[str] = []
    rows.append(
        '<h1 id="sagr-frontier-third-reader-title">The audit re-read by one '
        "frontier reader</h1>"
    )
    rows.append(
        f'<p id="sagr-frontier-third-reader-summary">'
        f"{counts['plants']} planted names, {counts['flagged']} sampled names "
        f"either earlier reader flagged, and {counts['control']} sampled names "
        f"both earlier readers read sound were shuffled together and read blind "
        f"by <strong>{esc(summary['reader']['model'])}</strong> "
        f"({esc(summary['reader']['family'])}, reasoning effort "
        f"{esc(summary['reader']['reasoning_effort'])}) at revision "
        f"<code>{esc(summary['revision'][:9])}</code>. On the planted names it "
        f"recalls {esc(planted['recall_C']['names'])} of "
        f"{esc(planted['recall_C']['total'])} known-defective "
        f"({fmt(planted['recall_C']['rate'])}) and flags "
        f"{esc(planted['false_alarm_C']['names'])} of "
        f"{esc(planted['false_alarm_C']['total'])} known-sound "
        f"({fmt(planted['false_alarm_C']['rate'])}). It confirms "
        f"{flagged['confirmed']} of {flagged['read']} earlier-flagged names as "
        f"defective and reads {control['defective']} of {control['read']} "
        f"earlier-sound control names as defective, which places the catalog's "
        f"defective share at <strong>{fmt(estimate['point'])} (95 % CI "
        f"{fmt(estimate['ci'][0])}–{fmt(estimate['ci'][1])})</strong>. Spend "
        f"was USD {summary['spend']['usd']:.2f} of the USD "
        f"{summary['spend']['ceiling']:.0f} ceiling.</p>"
    )

    rows.append(
        '<h2 id="sagr-frontier-third-reader-model">The reader and why it '
        "qualifies</h2>"
        f"<p>{esc(summary['model_selection']['reason'])}</p>"
        '<table id="sagr-frontier-third-reader-model-table"><thead><tr><th>'
        "Model</th><th>Family</th><th>USD / Mtok in–out</th><th>Context "
        "window</th><th>Named by a pipeline seat or review profile</th></tr>"
        "</thead><tbody>"
    )
    for candidate in summary["model_selection"]["candidates"]:
        rows.append(
            f"<tr><td><code>{esc(candidate['model'])}</code></td>"
            f"<td>{esc(candidate['family'])}</td>"
            f"<td>{candidate['prompt_usd_per_mtok']:g}–"
            f"{candidate['completion_usd_per_mtok']:g}</td>"
            f"<td>{candidate['context_window']:,}</td>"
            f"<td>{esc(candidate['pipeline_model'])}</td></tr>"
        )
    rows.append(
        "</tbody></table>"
        '<p id="sagr-frontier-third-reader-guard">The pipeline-model guard '
        f"passed for all three readers: {esc(summary['guard']['passed'])}; it "
        f"refused a configured pipeline model "
        f"({esc(summary['guard']['refusal_control'])}). The pipeline set read "
        f"from the seat configuration is {len(summary['guard']['pipeline_models'])} "
        "models.</p>"
    )

    rows.append(
        '<h2 id="sagr-frontier-third-reader-planted">Planted names, beside the '
        "first two readers</h2>"
        '<table id="sagr-frontier-third-reader-planted-table"><thead><tr>'
        "<th>Reading</th><th>Known-defective recalled</th><th>95&#8201;% CI"
        "</th><th>Known-sound flagged</th><th>95&#8201;% CI</th></tr>"
        "</thead><tbody>"
    )
    for seat, key in (("A", "A"), ("B", "B"), ("C", "C")):
        if seat == "C":
            rec, alarm = planted["recall_C"], planted["false_alarm_C"]
        else:
            rec, alarm = planted[f"recall_{key}"], planted[f"false_alarm_{key}"]
        rows.append(
            f"<tr><td>reader {seat}</td>"
            f"<td>{rec['names']} of {rec['total']} ({fmt(rec['rate'])})</td>"
            f"<td>{fmt(rec['ci'][0])}–{fmt(rec['ci'][1])}</td>"
            f"<td>{alarm['names']} of {alarm['total']} ({fmt(alarm['rate'])})</td>"
            f"<td>{fmt(alarm['ci'][0])}–{fmt(alarm['ci'][1])}</td></tr>"
        )
    rows.append("</tbody></table>")
    wrong = planted["wrong"]
    rows.append(
        f'<p id="sagr-frontier-third-reader-wrong-note">{len(wrong)} planted '
        "names the third reader gets wrong, with its stated reason, so the "
        "lead can judge whether the planted verdict or the reader is at "
        "fault.</p>"
        '<table id="sagr-frontier-third-reader-wrong-table"><thead><tr>'
        "<th>Name</th><th>Known</th><th>Third reader</th><th>Its reason</th>"
        "<th>Earlier readers</th><th>Citation</th></tr></thead><tbody>"
    )
    for row in wrong:
        earlier = (
            f"A: {', '.join(row['A']['classes']) or 'no defect'}; "
            f"B: {', '.join(row['B']['classes']) or 'no defect'}"
        )
        rows.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(row['known'])}</td>"
            f"<td>{esc(', '.join(row['C']['classes']) or 'no defect')}</td>"
            f"<td>{esc(row['C']['rationale'])}</td>"
            f"<td>{esc(earlier)}</td>"
            f"<td><code>{esc(row['citation'])}</code></td></tr>"
        )
    rows.append("</tbody></table>")

    rows.append(
        '<h2 id="sagr-frontier-third-reader-flagged">The 174 earlier-flagged '
        "names</h2>"
        f'<p id="sagr-frontier-third-reader-flagged-note">'
        f"{flagged['confirmed']} of {flagged['read']} are confirmed defective: "
        f"{flagged['confirmed_both']} of {flagged['flagged_both']} that both "
        f"earlier readers flagged and {flagged['confirmed_one']} of "
        f"{flagged['flagged_one']} that one flagged. A name counts in every "
        "class the third reader returned.</p>"
        '<table id="sagr-frontier-third-reader-flagged-table"><thead><tr>'
        "<th>Class</th><th>Both earlier readers flagged</th><th>One earlier "
        "reader flagged</th><th>Confirmed names</th></tr></thead><tbody>"
    )
    for key, item in flagged["by_class"].items():
        rows.append(
            f"<tr><td><code>{esc(key)}</code></td><td>{item['both']}</td>"
            f"<td>{item['one']}</td><td>{item['any']}</td></tr>"
        )
    rows.append("</tbody></table>")

    rows.append(
        '<h2 id="sagr-frontier-third-reader-control">The 60 control '
        "names</h2>"
        f'<p id="sagr-frontier-third-reader-control-note">'
        f"{control['defective']} of {control['read']} sampled names both "
        "earlier readers read sound are read defective by the third reader "
        "(the earlier readers' misses). Counts are names per class.</p>"
        '<table id="sagr-frontier-third-reader-control-table"><thead><tr>'
        "<th>Class</th><th>Control names read defective</th></tr></thead><tbody>"
    )
    for key, count in control["by_class"].items():
        rows.append(f"<tr><td><code>{esc(key)}</code></td><td>{count}</td></tr>")
    rows.append("</tbody></table>")

    rows.append(
        '<h2 id="sagr-frontier-third-reader-estimate">The catalog\'s defective '
        "share re-estimated</h2>"
        f'<p id="sagr-frontier-third-reader-estimate-note">Combining the '
        f"measured {fmt(estimate['flagged_stratum']['rate'])} "
        f"confirmation rate on the "
        f"{estimate['flagged_stratum']['names']} earlier-flagged names with the "
        f"{fmt(estimate['sound_stratum']['control_rate'])} defect rate the "
        f"control finds among the {estimate['sound_stratum']['names']} "
        f"names both earlier readers read sound gives "
        f"<strong>{fmt(estimate['point'])}</strong> (95&#8201;% CI "
        f"{fmt(estimate['ci'][0])}–{fmt(estimate['ci'][1])}) over the "
        f"{estimate['sample_names']} sampled names. Beside it: "
        f"{fmt(estimate['prior']['both_readers'])} defective by both earlier "
        f"readers and {fmt(estimate['prior']['either_reader'])} by either "
        f"(95&#8201;% CI {fmt(estimate['prior']['either_reader_ci'][0])}–"
        f"{fmt(estimate['prior']['either_reader_ci'][1])}).</p>"
        '<table id="sagr-frontier-third-reader-estimate-table"><thead><tr>'
        "<th>Figure</th><th>Names</th><th>Share</th><th>95&#8201;% CI</th></tr>"
        "</thead><tbody>"
        f"<tr><td>third reader, combined estimate</td>"
        f"<td>{estimate['sample_names']}</td><td>{fmt(estimate['point'])}</td>"
        f"<td>{fmt(estimate['ci'][0])}–{fmt(estimate['ci'][1])}</td></tr>"
        f"<tr><td>both earlier readers</td>"
        f"<td>{estimate['sample_names']}</td>"
        f"<td>{fmt(estimate['prior']['both_readers'])}</td>"
        f"<td>{fmt(estimate['prior']['both_readers_ci'][0])}–"
        f"{fmt(estimate['prior']['both_readers_ci'][1])}</td></tr>"
        f"<tr><td>either earlier reader</td>"
        f"<td>{estimate['sample_names']}</td>"
        f"<td>{fmt(estimate['prior']['either_reader'])}</td>"
        f"<td>{fmt(estimate['prior']['either_reader_ci'][0])}–"
        f"{fmt(estimate['prior']['either_reader_ci'][1])}</td></tr>"
        '</tbody></table><ul id="sagr-frontier-third-reader-assumptions">'
    )
    for assumption in estimate["assumptions"]:
        rows.append(f"<li>{esc(assumption)}</li>")
    rows.append("</ul>")

    rows.append(
        '<h2 id="sagr-frontier-third-reader-verdict">Can this reader decide a '
        "group audit?</h2>"
        f'<p id="sagr-frontier-third-reader-verdict-sentence">'
        f"{esc(verdict['sentence'])}</p>"
        f"<p>The bar was declared before the reading: recall at least "
        f"{fmt(verdict['bar']['recall_at_least'])} on the known-defective "
        f"plants and at most {fmt(verdict['bar']['false_alarm_at_most'])} false "
        "alarms on the known-sound plants, each with its own interval.</p>"
    )

    disagreements = summary["disagreements"]
    rows.append(
        '<h2 id="sagr-frontier-third-reader-disagreements">Where it disagrees '
        "with both earlier readers</h2>"
        f'<p id="sagr-frontier-third-reader-disagreements-note">'
        f"{len(disagreements)} sampled names carry a third-reader "
        "defective-or-not verdict that differs from each earlier reader's"
        + (
            "; fewer than twenty such names exist, so all of them are listed."
            if len(disagreements) < 20
            else "; the first twenty are shown."
        )
        + '</p><table id="sagr-frontier-third-reader-disagreements-table">'
        "<thead><tr><th>Name</th><th>Set</th><th>Reader A</th><th>Reader B</th>"
        "<th>Third reader</th></tr></thead><tbody>"
    )
    for row in disagreements[:20]:
        rows.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(row['set'])}</td>"
            f"<td>{esc(_reading_text(row['A']))}</td>"
            f"<td>{esc(_reading_text(row['B']))}</td>"
            f"<td>{esc(_reading_text(row['C']))}</td></tr>"
        )
    rows.append("</tbody></table>")

    rows.append(
        '<h2 id="sagr-frontier-third-reader-blindness">What the reader '
        "saw</h2>"
        '<p id="sagr-frontier-third-reader-blindness-note">The prompt is the '
        "one the first two readers saw, built from the same allow-list: name, "
        "stored description, documentation and unit, assigned physics domain, "
        "DD text of the sources, the text of their enclosing structures, and "
        "other live names produced by the same sources. The three sets were "
        "shuffled together and no stage, score, earlier verdict or set marker "
        "reaches the prompt; the committed script asserts that before every "
        "call.</p>"
        '<h3>System prompt</h3><pre id="sagr-frontier-third-reader-system-prompt">'
        f"{esc(summary['prompts']['system'])}</pre>"
        '<h3>One full user prompt</h3><pre id="sagr-frontier-third-reader-user-prompt">'
        f"{esc(summary['prompts']['user'])}</pre>"
    )

    rows.append(
        '<h2 id="sagr-frontier-third-reader-spend">Spend and record</h2>'
        f'<p id="sagr-frontier-third-reader-spend-note">USD '
        f"{summary['spend']['usd']:.4f} over "
        f"{summary['spend']['calls_priced']} priced calls and "
        f"{summary['spend']['tokens']} tokens, of a USD "
        f"{summary['spend']['ceiling']:.0f} ceiling"
        + (
            f"; {summary['spend']['unknown_cost_calls']} failed call(s) whose "
            "billed cost the call layer does not return are not counted as free"
            if summary["spend"]["unknown_cost_calls"]
            else ""
        )
        + f". Unanswered calls: {summary['error_count']}. Names selected but no "
        f"longer live: {len(summary['names_no_longer_live'])}"
        + (
            " ("
            + ", ".join(
                f"<code>{esc(name)}</code>" for name in summary["names_no_longer_live"]
            )
            + ")"
            if summary["names_no_longer_live"]
            else ""
        )
        + ". The readings are written as non-deciding audit review rows "
        "(<code>resolution_role='audit'</code>), the shape the first two "
        "readers' rows use.</p>"
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="docs-project" content="imas-codex">\n'
        '<meta name="reckon-type" content="evidence-fragment">\n'
        '<meta name="plan-slug" content="sn-audited-group-release">\n'
        '<meta name="node-id" content="sagr-frontier-third-reader">\n'
        "<title>The audit re-read by one frontier reader</title>\n"
        "</head>\n<body>\n<main>\n" + "\n".join(rows) + "\n</main>\n</body>\n</html>\n"
    )


def run_third(report_only: bool, limit: int = 0) -> int:
    """Read the three sets with the frontier seat and write the two records."""
    import subprocess

    assert_readers_not_pipeline(READERS + (third_reader(),))
    guard = {
        "pipeline_models": sorted(pipeline_models()),
        "readers": [dict(r) for r in READERS + (third_reader(),)],
        "passed": True,
        "refusal_control": False,
    }
    try:
        assert_readers_not_pipeline(
            ({"seat": "X", "model": "openrouter/openai/gpt-6-luna"},)
        )
    except AssertionError:
        guard["refusal_control"] = True
    if not guard["refusal_control"]:
        raise RuntimeError("the pipeline-model guard did not refuse a pipeline model")

    readings = prior_audit_readings()
    plants = prior_plant_rows()
    selection = third_selection(readings, plants)
    print(
        "selected: " + ", ".join(f"{k}={v}" for k, v in selection["counts"].items()),
        flush=True,
    )
    items, stage = third_items(selection)
    prompts = {"system": system_prompt(), "user": render_user_prompt(items[0])}
    if not report_only:
        asyncio.run(
            run_reading(
                items[:limit] if limit else items,
                (third_reader(),),
                state_dir=THIRD_STATE,
                ceiling=THIRD_SPEND_CEILING,
                projection_after=THIRD_PROJECTION_AFTER,
            )
        )
    readings_raw = list(load_progress(THIRD_STATE).values())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    summary = summarize_third(
        readings_raw=readings_raw,
        selection=selection,
        readings=readings,
        prompts=prompts,
        revision=revision,
        guard=guard,
        stage=stage,
    )
    THIRD_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    THIRD_HTML.write_text(render_third_page(summary))
    print(
        f"wrote {THIRD_JSON.name} and {THIRD_HTML.name}; "
        f"{summary['flagged']['read']} flagged and "
        f"{summary['control']['read']} control readings; spend "
        f"{summary['spend']['usd']:.2f} USD",
        flush=True,
    )
    return 0 if not summary["missing_readings"] else 1


def third_self_test() -> int:
    """Fuses the third reader's arithmetic: the checks below must fail on a
    planted defect in any of them."""
    readings = {
        "def1": {
            "A": {"classes": ["name_omits_source"], "rationale": ""},
            "B": {"classes": [], "rationale": ""},
        },
        "def2": {
            "A": {"classes": ["unit_wrong"], "rationale": ""},
            "B": {"classes": ["unit_wrong"], "rationale": ""},
        },
        "sound1": {
            "A": {"classes": [], "rationale": ""},
            "B": {"classes": [], "rationale": ""},
        },
    }
    raw = [
        {
            "name": "def1",
            "seat": "C",
            "model": "t",
            "defect_classes": [],
            "rationale": "fine",
            "cost": 0.01,
            "tokens": 1,
        },
        {
            "name": "def2",
            "seat": "C",
            "model": "t",
            "defect_classes": ["unit_wrong"],
            "rationale": "wrong unit",
            "cost": 0.01,
            "tokens": 1,
        },
        {
            "name": "sound1",
            "seat": "C",
            "model": "t",
            "defect_classes": ["name_omits_source"],
            "rationale": "miss",
            "cost": 0.01,
            "tokens": 1,
        },
    ]
    selection = {
        "seed": THIRD_SEED,
        "control_size": 1,
        "plants": [],
        "sample": sorted(readings),
        "complete": sorted(readings),
        "flagged": ["def1", "def2"],
        "flagged_both": ["def2"],
        "flagged_one": ["def1"],
        "sound": ["sound1"],
        "control": ["sound1"],
        "incomplete": [],
        "counts": {
            "audited_names": 3,
            "sample": 3,
            "plants": 0,
            "complete": 3,
            "flagged": 2,
            "flagged_both": 1,
            "flagged_one": 1,
            "sound": 1,
            "control": 1,
            "incomplete": 0,
        },
    }
    summary = summarize_third(
        readings_raw=raw,
        selection=selection,
        readings=readings,
        prompts={"system": "", "user": ""},
        revision="0" * 40,
        guard={"passed": True, "refusal_control": True, "pipeline_models": []},
        stage={name: {"name_stage": "", "status": ""} for name in readings},
    )
    checks = {
        "plant recall is measured on the planted names": (
            summary["planted"]["recall_C"]["total"] == 0
        ),
        "a flagged name the third reader passes is not confirmed": (
            summary["flagged"]["confirmed"] == 1
            and summary["flagged"]["confirmed_one"] == 0
            and summary["flagged"]["confirmed_both"] == 1
        ),
        "class counts follow the confirmed names only": (
            summary["flagged"]["by_class"]["unit_wrong"]["any"] == 1
            and summary["flagged"]["by_class"]["name_omits_source"]["any"] == 0
        ),
        "a control name the third reader flags is counted": (
            summary["control"]["defective"] == 1
            and summary["control"]["by_class"]["name_omits_source"] == 1
        ),
        "a disagreement with both earlier readers is captured": (
            len([d for d in summary["disagreements"] if d["name"] == "sound1"]) == 1
            and len([d for d in summary["disagreements"] if d["name"] == "def1"]) == 0
        ),
        "the combined estimate uses the measured strata": (
            summary["reestimate"]["flagged_stratum"]["rate"] == 0.5
            and summary["reestimate"]["sound_stratum"]["control_rate"] == 1.0
            and summary["reestimate"]["point"] == round(2 / 3, 4)
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 1 if failed else 0


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
    parser.add_argument("--third", action="store_true")
    parser.add_argument("--third-report", action="store_true")
    parser.add_argument("--third-persist", action="store_true")
    parser.add_argument("--third-check", action="store_true")
    parser.add_argument("--third-limit", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.check:
        return _self_test() or third_self_test()
    if args.third_check:
        return third_self_test()
    if args.third or args.third_report:
        return run_third(
            report_only=args.third_report and not args.third,
            limit=args.third_limit,
        )
    if args.third_persist:
        return persist(
            list(load_progress(THIRD_STATE).values()),
            state_dir=THIRD_STATE,
            reader_round="frontier-third-reader",
        )

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


def persist(
    readings: list[dict[str, Any]],
    *,
    state_dir: Path | None = None,
    reader_round: str | None = None,
) -> int:
    """Write each reading as a non-deciding review record.

    ``reader_round`` tags the row with the audit round it belongs to; the
    resolution role stays ``audit`` so no accepting, refining or exporting path
    can read it.
    """
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
                | ({"reader_round": reader_round} if reader_round else {})
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
    base = Path(state_dir) if state_dir else STATE
    base.mkdir(parents=True, exist_ok=True)
    (base / "persist-verification.json").write_text(
        json.dumps(verification, indent=1) + "\n"
    )
    print(json.dumps(verification, indent=1))
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
