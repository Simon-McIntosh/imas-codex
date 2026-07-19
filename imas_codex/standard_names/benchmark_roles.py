"""Per-role benchmark modes for the standard-name pipeline seats.

The compose benchmark (:mod:`benchmark`) measures one capability: drafting a
name from a DD path.  Each pipeline seat exercises a *different* capability, so
a compose result must not be extrapolated to the refine, review-breaker, docs,
or classifier seats.  This module benchmarks each seat against its own
production prompt, its own real fixtures, and the seat's production
reasoning-effort setting, so re-benchmarking on the next model generation is a
command (``sn bench --role <role>``) rather than a one-off script.

Roles
-----
``refine``          Refine-from-critique: replay real reviewer-critique →
                    refinement cases from graph history; a held-out judge
                    scores defect-resolution and collateral change.
``breaker-names``   Review-breaker independence on the names axis: re-score a
                    stratified sample already scored by the blind pair
                    (qwen + minimax) and measure rank-correlation independence
                    and verdict-flip quality against the final outcome.
``breaker-docs``    Same design over docs reviews, with seeded normative-policy
                    violations to measure policy-defect recall.
``docs``            Docs generation: generate documentation for a fixed
                    stratified sample under the production prompt; score with
                    the production rubric; grep-audit for banned prose.
``classifier``      Domain classifier: run the domain gold set (exact-match).

All corpus loaders are graph read-only.  All scoring arithmetic lives in pure
helper functions so it is unit-testable without a live graph or LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Re-exported for back-compat: the banned-prose vocabulary now lives in the
# neutral ``prose_policy`` module so production selection logic need not import
# a benchmark module.  The docs-seat benchmark below still audits against it.
from imas_codex.standard_names.prose_policy import (
    BANNED_PROSE_PATTERNS as BANNED_PROSE_PATTERNS,  # noqa: F401
    banned_prose_findings,
)

logger = logging.getLogger(__name__)

# The blind reviewer pair whose independence a breaker candidate is measured
# against (see [sn-review.names.profiles.default]).  A breaker that merely
# mirrors the pair adds cost without information.
BLIND_PAIR = ("openrouter/qwen/qwen3.7-max", "openrouter/minimax/minimax-m3")

# Acceptance threshold for a normalised (0-1) review score — mirrors the
# production triage threshold default.
ACCEPT_THRESHOLD = 0.75


def _bench_progress(msg: str) -> None:
    """Append a flushed, timestamped progress line for long bench runs.

    Console output from a bench is block-buffered through the process wrapper,
    so a multi-hour run is otherwise a black box (cannot tell slow from hung).
    This writes an immediately-flushed line to the file named by
    ``IMAS_CODEX_BENCH_PROGRESS`` (no-op when unset), so ``tail -f`` on that
    file is a reliable live progress + liveness signal independent of logging
    level or stdout buffering.
    """
    import os
    from datetime import UTC, datetime

    path = os.environ.get("IMAS_CODEX_BENCH_PROGRESS")
    line = f"{datetime.now(tz=UTC).strftime('%H:%M:%S')} {msg}"
    logger.info("bench-progress: %s", msg)
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Report types
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RoleModelResult:
    """One model's measured row for a role benchmark."""

    model: str
    n: int = 0
    cost: float = 0.0
    # Role-specific measured metrics, e.g. defect_resolution, collateral_change,
    # independence_rho, verdict_flip_quality, rubric_score, banned_prose_rate,
    # accuracy.  Kept as a free dict so each role names its own axes.
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def cost_per_item(self) -> float:
        return round(self.cost / self.n, 6) if self.n else 0.0


@dataclass
class RoleBenchReport:
    """Measured table for a single role, across candidate models."""

    role: str
    results: list[RoleModelResult]
    incumbent: str | None = None
    judge_model: str | None = None
    sample_ids: list[str] = field(default_factory=list)
    seed: int = 0
    axis: str | None = None
    timestamp: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "role": self.role,
                "results": [asdict(r) for r in self.results],
                "incumbent": self.incumbent,
                "judge_model": self.judge_model,
                "sample_ids": self.sample_ids,
                "seed": self.seed,
                "axis": self.axis,
                "timestamp": self.timestamp,
                "provenance": self.provenance,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> RoleBenchReport:
        d = json.loads(text)
        return cls(
            role=d["role"],
            results=[RoleModelResult(**r) for r in d["results"]],
            incumbent=d.get("incumbent"),
            judge_model=d.get("judge_model"),
            sample_ids=d.get("sample_ids", []),
            seed=d.get("seed", 0),
            axis=d.get("axis"),
            timestamp=d.get("timestamp", ""),
            provenance=d.get("provenance", {}),
        )

    def save_atomic(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(p)


# ═══════════════════════════════════════════════════════════════════════
# Pure scoring helpers (unit-testable, no I/O)
# ═══════════════════════════════════════════════════════════════════════


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Returns 0.0 for degenerate input (length < 2 or a constant vector), which
    is the correct "no measurable correlation" sentinel for these samples.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank, 1-based
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 4)


def verdict_flip_quality(
    candidate_scores: list[float],
    pair_mean_scores: list[float],
    final_outcomes: list[bool],
    threshold: float = ACCEPT_THRESHOLD,
) -> tuple[float, int]:
    """Fraction of pair-disagreements where the candidate matched the outcome.

    A "flip" is an item where the candidate's accept/reject verdict differs
    from the blind pair's mean verdict.  Quality = of those flips, the fraction
    where the candidate's verdict agreed with the final accepted/rejected
    ``final_outcome``.  A high value means the breaker adds *correct*
    information when it overrides the pair.

    Returns ``(quality, n_flips)``.  Quality is 0.0 when there are no flips.
    """
    flips = 0
    correct = 0
    for cs, ps, outcome in zip(
        candidate_scores, pair_mean_scores, final_outcomes, strict=False
    ):
        cand_accept = cs >= threshold
        pair_accept = ps >= threshold
        if cand_accept == pair_accept:
            continue
        flips += 1
        if cand_accept == outcome:
            correct += 1
    if flips == 0:
        return 0.0, 0
    return round(correct / flips, 4), flips


def _classify_refine_failure(exc: BaseException) -> str:
    """Bucket a refine compose failure so an all-zero row explains itself.

    ``kind_enum``  — the model set the entry ``kind`` to an out-of-vocabulary
    value (a schema/prompt gap, not name quality).  Matches both the historical
    after-validator message ("kind must be one of ...") and the current
    ``Literal`` schema form ("kind\\n  Input should be 'scalar', ... 'metadata'").
    ``grammar_token`` — the model used an unregistered grammar token.
    ``other`` — anything else (timeouts, provider errors, empty responses).
    """
    msg = str(exc).lower()
    if "kind must be one of" in msg or (
        "input should be" in msg and "'scalar'" in msg and "'metadata'" in msg
    ):
        return "kind_enum"
    if "not a registered" in msg:
        return "grammar_token"
    return "other"


def exact_match_accuracy(
    predicted: dict[str, str], expected: dict[str, str]
) -> tuple[float, int, int]:
    """Exact-match accuracy over keys present in *expected*.

    Returns ``(accuracy, n_correct, n_total)``.  Keys missing from *predicted*
    count as incorrect.
    """
    total = len(expected)
    if total == 0:
        return 0.0, 0, 0
    correct = sum(1 for k, v in expected.items() if predicted.get(k) == v)
    return round(correct / total, 4), correct, total


# ═══════════════════════════════════════════════════════════════════════
# Corpus loaders (graph read-only)
# ═══════════════════════════════════════════════════════════════════════


def _stratified_sample(
    rows: list[dict], sample: int, seed: int, key: str = "physics_domain"
) -> list[dict]:
    """Deterministic stratified sample of *rows* balanced across *key*.

    Draws round-robin across the value buckets of ``key`` so every physics
    domain is represented before any is doubled up.
    """
    rng = random.Random(seed)
    buckets: dict[Any, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r.get(key), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    ordered_keys = sorted(buckets, key=lambda k: (k is None, str(k)))
    out: list[dict] = []
    while len(out) < sample and any(buckets[k] for k in ordered_keys):
        for k in ordered_keys:
            if buckets[k]:
                out.append(buckets[k].pop())
                if len(out) >= sample:
                    break
    return out


def load_refine_corpus(
    sample: int, seed: int, axis: str = "names", gc: Any = None
) -> list[dict]:
    """Load real refine-from-critique cases from REFINED_FROM history.

    Each case is a name that was refined: its immediate REFINED_FROM ancestor
    carries the reviewer critique (per-dimension comments) and the below-par
    score that triggered the refinement.  That ancestor's context + critique is
    the refine input; the bench re-runs each candidate model on it.

    Returns case dicts: sn_id, path, description, unit, data_type,
    physics_domain, prior_name, prior_score, critique (per-dim comment dict).
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName)-[:REFINED_FROM]->(a:StandardName)
        WHERE a.reviewer_comments_per_dim_name IS NOT NULL
          AND a.reviewer_score_name IS NOT NULL
        WITH sn, a
        ORDER BY a.chain_length ASC
        WITH sn, collect(a)[0] AS anc
        RETURN
          sn.id                                   AS sn_id,
          anc.id                                  AS prior_name,
          anc.reviewer_score_name                 AS prior_score,
          anc.reviewer_comments_per_dim_name      AS critique,
          coalesce(sn.source_paths, anc.source_paths) AS source_paths,
          coalesce(sn.description, anc.description)    AS description,
          coalesce(sn.unit, anc.unit)                 AS unit,
          coalesce(sn.kind, anc.kind)                 AS data_type,
          coalesce(sn.physics_domain, anc.physics_domain) AS physics_domain
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(cypher)
    finally:
        if owns:
            gc.close()

    cases: list[dict] = []
    for r in rows:
        crit = r.get("critique")
        if isinstance(crit, str):
            try:
                crit = json.loads(crit)
            except (ValueError, TypeError):
                crit = {"overall": crit}
        paths = r.get("source_paths") or []
        cases.append(
            {
                "sn_id": r["sn_id"],
                "prior_name": r.get("prior_name"),
                "prior_score": float(r.get("prior_score") or 0.0),
                "critique": crit or {},
                "path": paths[0] if paths else "",
                "source_paths": list(paths),
                "description": r.get("description") or "",
                "unit": r.get("unit") or "",
                "data_type": r.get("data_type") or "",
                "physics_domain": r.get("physics_domain") or "",
            }
        )
    return _stratified_sample(cases, sample, seed)


def load_breaker_corpus(
    sample: int, seed: int, axis: str = "names", gc: Any = None
) -> list[dict]:
    """Load names/docs already scored by BOTH blind-pair reviewers.

    Groups StandardNameReview nodes by ``standard_name_id`` on the requested
    axis and keeps only items where both qwen and minimax scored, so a
    candidate breaker can be measured for rank-correlation independence against
    the pair.  ``final_accepted`` is the eventual outcome (name_stage) used for
    verdict-flip quality.

    Returns item dicts: sn_id, name, description, unit, data_type,
    physics_domain, source_paths, pair_scores {qwen, minimax}, final_accepted.
    """
    from imas_codex.graph.client import GraphClient

    review_axis = "names" if axis == "names" else "docs"
    cypher = """
        MATCH (r:StandardNameReview {review_axis: $axis})
        WHERE r.reviewer_model IN $pair AND r.score IS NOT NULL
        WITH r.standard_name_id AS sid, r.reviewer_model AS m,
             avg(toFloat(r.score)) AS s
        WITH sid, collect({m: m, s: s}) AS scores
        WHERE size(scores) = 2
        MATCH (sn:StandardName {id: sid})
        RETURN
          sid                       AS sn_id,
          scores                    AS scores,
          sn.name_stage             AS name_stage,
          sn.description            AS description,
          sn.unit                   AS unit,
          sn.kind                   AS data_type,
          sn.physics_domain         AS physics_domain,
          sn.source_paths           AS source_paths,
          sn.documentation          AS documentation
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(cypher, axis=review_axis, pair=list(BLIND_PAIR))
    finally:
        if owns:
            gc.close()

    items: list[dict] = []
    for r in rows:
        pair = {s["m"]: float(s["s"]) for s in r["scores"]}
        qwen = pair.get(BLIND_PAIR[0])
        minimax = pair.get(BLIND_PAIR[1])
        if qwen is None or minimax is None:
            continue
        srcs = list(r.get("source_paths") or [])
        items.append(
            {
                "sn_id": r["sn_id"],
                "name": r["sn_id"],
                # score_with_reviewer resolves the display name via _resolve_name,
                # which reads standard_name/id; the reviewer echoes it back, so
                # these keys are what re-aligns reviews to corpus items.
                "standard_name": r["sn_id"],
                "id": r["sn_id"],
                "source_id": srcs[0] if srcs else "",
                "description": r.get("description") or "",
                "unit": r.get("unit") or "",
                "data_type": r.get("data_type") or "",
                "physics_domain": r.get("physics_domain") or "",
                "source_paths": list(r.get("source_paths") or []),
                "documentation": r.get("documentation") or "",
                "pair_scores": {"qwen": qwen, "minimax": minimax},
                "final_accepted": (r.get("name_stage") == "accepted"),
            }
        )
    return _stratified_sample(items, sample, seed)


def load_docs_sample(sample: int, seed: int, gc: Any = None) -> list[dict]:
    """Load a stratified sample of accepted names for docs regeneration.

    Returns candidate dicts shaped for :func:`generate_docs_for_candidates`
    (name, unit, kind, physics_domain, description, source_paths).
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName {name_stage: 'accepted'})
        WHERE sn.description IS NOT NULL
        RETURN
          sn.id             AS name,
          sn.description    AS description,
          sn.unit           AS unit,
          sn.kind           AS kind,
          sn.physics_domain AS physics_domain,
          sn.source_paths   AS source_paths
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(cypher)
    finally:
        if owns:
            gc.close()

    cands = [
        {
            "name": r["name"],
            # generate_docs_for_candidates / score_with_reviewer resolve the
            # name via _resolve_name (reads standard_name/id).
            "standard_name": r["name"],
            "id": r["name"],
            "source_id": (list(r.get("source_paths") or []) or [""])[0],
            "description": r.get("description") or "",
            "unit": r.get("unit") or "",
            "kind": r.get("kind") or "scalar",
            "physics_domain": r.get("physics_domain") or "",
            "source_paths": list(r.get("source_paths") or []),
        }
        for r in rows
    ]
    return _stratified_sample(cands, sample, seed)


GOLD_SET_PATH = (
    Path(__file__).resolve().parent.parent
    / "definitions"
    / "physics"
    / "domain_gold_set.json"
)


def load_classifier_gold() -> list[dict]:
    """Load the domain classification gold set (list of path/expected_domain)."""
    return json.loads(GOLD_SET_PATH.read_text())


# ═══════════════════════════════════════════════════════════════════════
# Held-out refine judge
# ═══════════════════════════════════════════════════════════════════════

_REFINE_JUDGE_SYSTEM = """\
You are a strict, held-out judge evaluating whether a REFINED IMAS standard
name resolved the defects a reviewer flagged on the previous attempt, WITHOUT
making needless collateral changes.

You are given: the DD path context, the previous (rejected) name, the reviewer
critique per dimension, and the refined name + its description. Judge only what
the critique named. Score two axes on 0.0-1.0:

- defect_resolution: fraction of the critique's concrete defects that the
  refined name genuinely fixes (1.0 = all named defects addressed; 0.0 = none).
- collateral_change: degree of change UNRELATED to the critique — renaming or
  rewording parts the reviewer did not flag (0.0 = surgical, only flagged parts
  changed; 1.0 = wholesale rewrite ignoring what was actually wrong).

Lower collateral_change is better. Reward surgical fixes, penalise both
under-fixing and scattershot rewrites.
"""

_REFINE_JUDGE_USER = """\
## DD path
{path}
Description: {description}
Unit: {unit}   Data type: {data_type}   Physics domain: {physics_domain}

## Previous (rejected) name — score {prior_score:.2f}
{prior_name}

## Reviewer critique (per dimension)
{critique}

## Refined name
{refined_name}
Refined description: {refined_description}

Return your two scores and a one-sentence justification.
"""


def _format_critique(critique: dict) -> str:
    if not critique:
        return "(no per-dimension critique recorded)"
    return "\n".join(f"- {k}: {v}" for k, v in critique.items() if v)


# ═══════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════


def build_refine_prompt_context(
    case: dict,
    compose_context: dict,
    rules: list,
    scored_examples: list | None = None,
) -> dict:
    """Build the refine prompt context for one case, mirroring production.

    Merges ``compose_context`` (from ``build_compose_context``) so the refine
    system prompt's grammar-reference include renders the closed-vocabulary
    token map.  Omitting it leaves the grammar block empty, and the model
    invents unregistered tokens — the bench would then measure a grammar-
    failure artifact rather than refine quality (the same empty-grammar-block
    gap ``process_refine_name_batch`` guards against in production).

    ``scored_examples`` are the domain-scoped calibration examples production
    loads via ``load_compose_examples``.  The ``_compose_scored_examples.md``
    include renders each with its ``kind=<scalar|vector|metadata>`` tag — the
    only place the refine prompt shows a valid entry kind.  Without them the
    model has no in-prompt signal for the free-string ``kind`` field and sets
    it to a semantic guess (e.g. ``standard_name``), failing RefinedName
    validation on every case, so the examples must be fed like production.
    """
    chain_history = [
        {
            "name": case["prior_name"],
            "reviewer_score": case["prior_score"],
            "reviewer_comments_per_dim": case["critique"],
        }
    ]
    return {
        **compose_context,
        "item": {
            "path": case["path"],
            "ids_name": (case["path"].split("/")[0] if case["path"] else ""),
            "description": case["description"],
            "unit": case["unit"],
            "data_type": case["data_type"],
            "physics_domain": case["physics_domain"],
        },
        "chain_history": chain_history,
        "chain_length": len(chain_history),
        "hybrid_neighbours": [],
        "fanout_evidence": "",
        "composition_rules": rules,
        "compose_scored_examples": scored_examples or [],
    }


async def run_refine_bench(
    models: list[str],
    corpus: list[dict],
    judge_model: str,
    incumbent: str | None = None,
    reasoning_effort: str | None = None,
) -> RoleBenchReport:
    """Benchmark the refine seat: defect-resolution + collateral change."""
    from collections import Counter

    from pydantic import BaseModel, Field

    from imas_codex.discovery.base.llm import acall_llm_structured, ensure_model_prefix
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import load_prompt_config, render_prompt
    from imas_codex.standard_names.benchmark import _resolve_name
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.example_loader import load_compose_examples
    from imas_codex.standard_names.models import RefinedName

    # The refine system prompt's grammar-reference include renders the
    # closed-vocabulary token map from the compose context (vocabulary_sections,
    # closed_vocab_full).  Production merges this into the refine prompt context;
    # without it the model rewrites names with no vocabulary reference and
    # invents unregistered tokens, so the bench must merge it too or it
    # under-feeds every candidate and measures a grammar-failure artifact
    # instead of refine quality.  Cache-backed, so this is a dict lookup.
    compose_context = build_compose_context()

    class _RefineJudgement(BaseModel):
        defect_resolution: float = Field(ge=0.0, le=1.0)
        collateral_change: float = Field(ge=0.0, le=1.0)
        justification: str = ""

    try:
        rules = load_prompt_config("sn_composition_rules").get("composition_rules", [])
    except Exception:
        rules = []

    # Domain-scoped scored examples, loaded once per physics domain and reused
    # across candidates — mirrors production's per-item load_compose_examples.
    # These carry the only in-prompt signal for the valid entry ``kind``.
    examples_by_domain: dict[str, list] = {}
    with GraphClient() as _gc:
        for domain in {c.get("physics_domain") or "" for c in corpus}:
            try:
                examples_by_domain[domain] = load_compose_examples(
                    _gc, physics_domains=[domain], axis="name"
                )
            except Exception:
                logger.debug("refine bench: scored-example load failed for %r", domain)
                examples_by_domain[domain] = []

    results: list[RoleModelResult] = []

    _bench_progress(f"refine: start {len(models)} model(s) x {len(corpus)} cases")
    # Cases run concurrently (bounded); each case's refine→judge stay sequential
    # (the judge needs the refined name). A serial per-case loop let a single
    # hung provider call block the whole model for minutes; max_retries=2 also
    # fails a hung call fast instead of the default 5×timeout amplification.
    _refine_sem = asyncio.Semaphore(6)
    _judge = ensure_model_prefix(judge_model)
    for _mi, model in enumerate(models, 1):
        m = ensure_model_prefix(model)
        res = RoleModelResult(model=model)
        dr_vals: list[float] = []
        cc_vals: list[float] = []
        fail_reasons: Counter[str] = Counter()
        _bench_progress(f"refine: [{_mi}/{len(models)}] {model.split('/')[-1]}…")

        async def _refine_one(
            case: dict, _m: str = m
        ) -> tuple[str, float, float, float]:
            prompt_context = build_refine_prompt_context(
                case,
                compose_context,
                rules,
                examples_by_domain.get(case.get("physics_domain") or "", []),
            )
            async with _refine_sem:
                try:
                    user_prompt = render_prompt("sn/refine_name_user", prompt_context)
                    try:
                        system_prompt = render_prompt(
                            "sn/refine_name_system", prompt_context
                        )
                    except Exception:
                        system_prompt = None
                    messages = (
                        [{"role": "system", "content": system_prompt}]
                        if system_prompt
                        else []
                    ) + [{"role": "user", "content": user_prompt}]
                    refined, cost, _ = await acall_llm_structured(
                        model=_m,
                        messages=messages,
                        response_model=RefinedName,
                        service="standard-names",
                        reasoning_effort=reasoning_effort,
                        max_retries=2,
                    )
                except Exception as exc:
                    logger.warning("refine %s failed on %s: %s", _m, case["sn_id"], exc)
                    return (_classify_refine_failure(exc), 0.0, 0.0, 0.0)

                refined_name = (
                    _resolve_name(
                        {
                            "segments": refined.segments.model_dump()
                            if hasattr(refined.segments, "model_dump")
                            else refined.segments
                        }
                    )
                    or case["sn_id"]
                )
                try:
                    juser = _REFINE_JUDGE_USER.format(
                        path=case["path"],
                        description=case["description"],
                        unit=case["unit"],
                        data_type=case["data_type"],
                        physics_domain=case["physics_domain"],
                        prior_score=case["prior_score"],
                        prior_name=case["prior_name"],
                        critique=_format_critique(case["critique"]),
                        refined_name=refined_name,
                        refined_description=refined.description or "",
                    )
                    judgement, jcost, _ = await acall_llm_structured(
                        model=_judge,
                        messages=[
                            {"role": "system", "content": _REFINE_JUDGE_SYSTEM},
                            {"role": "user", "content": juser},
                        ],
                        response_model=_RefineJudgement,
                        service="standard-names",
                        max_retries=2,
                    )
                except Exception as exc:
                    logger.warning("refine judge failed on %s: %s", case["sn_id"], exc)
                    return ("judge_error", 0.0, 0.0, cost)
                return (
                    "ok",
                    float(judgement.defect_resolution),
                    float(judgement.collateral_change),
                    cost + jcost,
                )

        for outcome, _dr, _cc, _cost in await asyncio.gather(
            *(_refine_one(c) for c in corpus)
        ):
            res.cost += _cost
            if outcome == "ok":
                dr_vals.append(_dr)
                cc_vals.append(_cc)
                res.n += 1
            else:
                fail_reasons[outcome] += 1

        if dr_vals:
            res.metrics["defect_resolution"] = round(sum(dr_vals) / len(dr_vals), 4)
            res.metrics["collateral_change"] = round(sum(cc_vals) / len(cc_vals), 4)
        # A model that judged zero cases produced no signal — record why loudly
        # rather than emitting a hollow all-zeros row that renders as a valid
        # table.  The failure breakdown (kind-enum vs grammar-token vs judge)
        # tells the reader whether it is a candidate defect or a bench gap.
        if res.n == 0:
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items()))
            res.error = f"0/{len(corpus)} cases judged; failures: {breakdown or 'none'}"
        res.metrics["valid_refine_rate"] = (
            round(res.n / len(corpus), 4) if corpus else 0.0
        )
        results.append(res)

    if results and all(r.n == 0 for r in results):
        detail = "; ".join(f"{r.model}: {r.error}" for r in results)
        raise RuntimeError(
            "refine bench produced no judged cases for any model — refusing to "
            f"save a hollow report ({detail})"
        )

    return RoleBenchReport(
        role="refine",
        results=results,
        incumbent=incumbent,
        judge_model=judge_model,
        sample_ids=[c["sn_id"] for c in corpus],
        axis="names",
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={"corpus": "REFINED_FROM history", "n_cases": len(corpus)},
    )


async def run_breaker_bench(
    models: list[str],
    corpus: list[dict],
    axis: str = "names",
    incumbent: str | None = None,
    reasoning_effort: str | None = None,
) -> RoleBenchReport:
    """Benchmark a review-breaker seat: independence + verdict-flip quality."""
    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.standard_names.benchmark import score_with_reviewer

    target = "names" if axis == "names" else "docs"
    qwen = [c["pair_scores"]["qwen"] for c in corpus]
    minimax = [c["pair_scores"]["minimax"] for c in corpus]
    pair_mean = [(a + b) / 2.0 for a, b in zip(qwen, minimax, strict=False)]
    outcomes = [bool(c["final_accepted"]) for c in corpus]

    results: list[RoleModelResult] = []
    _bench_progress(f"breaker-{axis}: start {len(models)} model(s) x {len(corpus)}")
    for _bi, model in enumerate(models, 1):
        res = RoleModelResult(model=model)
        _bench_progress(
            f"breaker-{axis}: [{_bi}/{len(models)}] {model.split('/')[-1]}…"
        )
        try:
            reviews, cost = await score_with_reviewer(
                corpus,
                reviewer_model=ensure_model_prefix(model),
                target=target,
                reasoning_effort=reasoning_effort,
            )
            res.cost = cost
        except Exception as exc:
            logger.warning("breaker %s failed: %s", model, exc)
            res.error = str(exc)[:200]
            results.append(res)
            continue

        by_name = {r["name"]: float(r.get("score") or 0.0) for r in reviews}
        cand_scores: list[float] = []
        aligned_qwen: list[float] = []
        aligned_minimax: list[float] = []
        aligned_pair_mean: list[float] = []
        aligned_outcomes: list[bool] = []
        for c, q, mm, pm, out in zip(
            corpus, qwen, minimax, pair_mean, outcomes, strict=False
        ):
            if c["name"] not in by_name:
                continue
            cand_scores.append(by_name[c["name"]])
            aligned_qwen.append(q)
            aligned_minimax.append(mm)
            aligned_pair_mean.append(pm)
            aligned_outcomes.append(out)

        res.n = len(cand_scores)
        if res.n >= 2:
            rho_q = spearman_rho(cand_scores, aligned_qwen)
            rho_m = spearman_rho(cand_scores, aligned_minimax)
            vfq, n_flips = verdict_flip_quality(
                cand_scores, aligned_pair_mean, aligned_outcomes
            )
            res.metrics["independence_rho"] = round((rho_q + rho_m) / 2.0, 4)
            res.metrics["rho_qwen"] = rho_q
            res.metrics["rho_minimax"] = rho_m
            res.metrics["verdict_flip_quality"] = vfq
            res.metrics["n_flips"] = float(n_flips)
        results.append(res)

    return RoleBenchReport(
        role=f"breaker-{axis}",
        results=results,
        incumbent=incumbent,
        sample_ids=[c["sn_id"] for c in corpus],
        axis=axis,
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={"blind_pair": list(BLIND_PAIR), "n_items": len(corpus)},
    )


# ═══════════════════════════════════════════════════════════════════════
# Review-discrimination bench (blind-pair reviewer seat)
# ═══════════════════════════════════════════════════════════════════════
#
# The breaker bench above measures a THIRD reviewer's independence from an
# existing blind pair.  It cannot measure the blind-pair reviewers themselves,
# and it correlates against a pair that may be retired.  The discrimination
# bench measures a candidate reviewer directly: score a labelled corpus of
# GOOD (accepted, in-catalog) items against BAD twins carrying a seeded defect
# a competent reviewer MUST catch, and measure the score separation, the
# fraction of defects caught, and the fraction of good items not falsely
# rejected.  This is the capability a blind-pair replacement must have.

# Deterministic defect seeds.  Each is an unambiguous quality failure the
# production rubric penalises; a reviewer that cannot separate these from the
# accepted original is not fit for the seat.
_DISCRIM_DEFECTS = ("banned_prose", "vacuous", "unit_contradiction")

_VACUOUS_DOC = (
    "This quantity represents an important physical property of the plasma "
    "that is relevant to the analysis and can be used in various calculations "
    "as appropriate for the scenario under consideration."
)


def _seed_bad_documentation(good_doc: str, defect: str, unit: str) -> str:
    """Return a defective twin of *good_doc* carrying exactly one seeded flaw."""
    good_doc = good_doc or ""
    if defect == "banned_prose":
        # Procedural padding + typical-values estimator prose — the exact
        # normative-policy class the docs campaign exists to remove.
        return (
            good_doc + "\n\nTypical values range from about 0.1 to 10 depending on the "
            "device and scenario; to compute it, first obtain the relevant "
            "profiles, then integrate over the volume and normalise as needed."
        )
    if defect == "vacuous":
        return _VACUOUS_DOC
    if defect == "unit_contradiction":
        stated = unit or "SI units"
        return (
            good_doc
            + f"\n\nAlthough the registered unit is {stated}, this quantity is "
            "in fact dimensionless and carries arbitrary units."
        )
    return good_doc


def _seed_bad_name(good_name: str, defect: str, foreign_name: str) -> str:
    """Return a defective NAME twin for the names axis.

    The names-review rubric scores the NAME (grammar / semantic / convention /
    completeness) against its description — so a genuinely-bad item must corrupt
    the NAME, not the description (which is left unchanged). Corrupting the
    description instead leaves the accepted name intact and the reviewer scores
    it high regardless, producing no discrimination.
    """
    toks = [t for t in good_name.split("_") if t]
    if defect == "vacuous":
        # Strip to a bare, underspecified base — drops every qualifier, so the
        # name no longer captures the specific quantity its description names
        # (completeness / semantic failure).
        return "_".join(toks[-2:]) if len(toks) >= 2 else good_name
    if defect == "unit_contradiction":
        # Reverse the token order — a non-canonical, convention-violating name.
        return "_".join(reversed(toks)) if len(toks) > 1 else good_name
    # banned_prose → SEMANTIC MISMATCH: an unrelated accepted name paired with
    # this item's real description (the name describes a different quantity).
    return foreign_name or good_name


def load_discrimination_corpus(
    sample: int, seed: int, axis: str = "docs", gc: Any = None
) -> list[dict]:
    """Build a labelled good/bad reviewer-discrimination corpus.

    Draws a stratified sample of accepted, documented names (the GOOD set,
    ``label=1``) and, for each, one defective twin (``label=0``) carrying one
    deterministically-assigned seeded defect.  The reviewer under test should
    score GOOD high and BAD low; the gap is the discrimination signal.

    ``axis='docs'`` corrupts the documentation text; ``axis='names'`` corrupts
    the description (with a foreign-description swap standing in for a semantic
    mismatch).  Graph read-only.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName {name_stage: 'accepted'})
        WHERE sn.description IS NOT NULL AND sn.documentation IS NOT NULL
              AND size(sn.documentation) > 40
        RETURN
          sn.id             AS name,
          sn.description    AS description,
          sn.documentation  AS documentation,
          sn.unit           AS unit,
          sn.kind           AS kind,
          sn.physics_domain AS physics_domain,
          sn.source_paths   AS source_paths
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(cypher)
    finally:
        if owns:
            gc.close()

    goods = _stratified_sample([dict(r) for r in rows], sample, seed)
    rng = random.Random(seed)
    all_names = [r["name"] for r in rows]

    corpus: list[dict] = []
    for i, r in enumerate(goods):
        srcs = list(r.get("source_paths") or [])
        base = {
            "name": r["name"],
            "standard_name": r["name"],
            "id": r["name"],
            "source_id": srcs[0] if srcs else "",
            "unit": r.get("unit") or "",
            "kind": r.get("kind") or "scalar",
            "data_type": r.get("kind") or "scalar",
            "physics_domain": r.get("physics_domain") or "",
            "source_paths": srcs,
        }
        good_desc = r.get("description") or ""
        good_doc = r.get("documentation") or ""
        # GOOD item — the accepted, in-catalog text.
        corpus.append(
            {
                **base,
                "label": 1,
                "defect": "",
                "description": good_desc,
                "documentation": good_doc,
            }
        )
        # BAD twin — one seeded defect, cycled deterministically across items.
        # Good and bad twins share a standard name, so they are scored in
        # SEPARATE reviewer passes (see run_review_discrimination_bench) to keep
        # the review→item alignment unambiguous.
        defect = _DISCRIM_DEFECTS[i % len(_DISCRIM_DEFECTS)]
        if axis == "docs":
            corpus.append(
                {
                    **base,
                    "label": 0,
                    "defect": defect,
                    "description": good_desc,
                    "documentation": _seed_bad_documentation(
                        good_doc, defect, base["unit"]
                    ),
                }
            )
        else:
            # Corrupt the NAME (keep the real description) so the names rubric
            # can discriminate. The bad twin carries a distinct standard name.
            foreign_name = rng.choice(all_names) if all_names else ""
            bad_name = _seed_bad_name(r["name"], defect, foreign_name)
            corpus.append(
                {
                    **base,
                    "name": bad_name,
                    "standard_name": bad_name,
                    "id": bad_name,
                    "label": 0,
                    "defect": defect,
                    "description": good_desc,
                    "documentation": good_doc,
                }
            )
    return corpus


def discrimination_metrics(
    good_scores: list[float],
    bad_scores: list[float],
    threshold: float = ACCEPT_THRESHOLD,
) -> dict[str, float]:
    """Compute reviewer-discrimination metrics from labelled scores.

    ``separation`` — mean(good) − mean(bad); the core signal (higher is better).
    ``bad_recall`` — fraction of BAD items scored below *threshold* (caught).
    ``good_pass`` — fraction of GOOD items scored at/above *threshold* (kept).
    ``auc`` — probability a random good outranks a random bad (rank AUC), the
    threshold-free separation measure; 0.5 = no discrimination, 1.0 = perfect.
    """
    metrics: dict[str, float] = {}
    if good_scores:
        metrics["good_mean"] = round(sum(good_scores) / len(good_scores), 4)
        metrics["good_pass"] = round(
            sum(1 for s in good_scores if s >= threshold) / len(good_scores), 4
        )
    if bad_scores:
        metrics["bad_mean"] = round(sum(bad_scores) / len(bad_scores), 4)
        metrics["bad_recall"] = round(
            sum(1 for s in bad_scores if s < threshold) / len(bad_scores), 4
        )
    if good_scores and bad_scores:
        metrics["separation"] = round(metrics["good_mean"] - metrics["bad_mean"], 4)
        wins = 0.0
        for g in good_scores:
            for b in bad_scores:
                wins += 1.0 if g > b else 0.5 if g == b else 0.0
        metrics["auc"] = round(wins / (len(good_scores) * len(bad_scores)), 4)
    return metrics


async def run_review_discrimination_bench(
    models: list[str],
    corpus: list[dict],
    axis: str = "docs",
    incumbent: str | None = None,
    reasoning_effort: str | None = None,
) -> RoleBenchReport:
    """Benchmark the blind-pair reviewer seat: good/bad discrimination."""
    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.standard_names.benchmark import score_with_reviewer

    target = "names" if axis == "names" else "docs"
    goods = [c for c in corpus if c["label"] == 1]
    bads = [c for c in corpus if c["label"] == 0]

    async def _score(items: list[dict], model: str) -> tuple[list[float], float]:
        # Good and bad twins share a standard name, so each label is scored in
        # its own pass — within a pass every item is a distinct accepted name,
        # so reviews align to items unambiguously.
        reviews, cost = await score_with_reviewer(
            items,
            reviewer_model=ensure_model_prefix(model),
            target=target,
            reasoning_effort=reasoning_effort,
        )
        return [float(r.get("score") or 0.0) for r in reviews], cost

    results: list[RoleModelResult] = []
    total = len(models)
    _bench_progress(
        f"review-{axis}: start {total} model(s) x {len(goods)}+{len(bads)} items"
    )
    spend = 0.0
    for i, model in enumerate(models, 1):
        res = RoleModelResult(model=model)
        tag = model.split("/")[-1]
        try:
            _bench_progress(f"review-{axis}: [{i}/{total}] {tag} scoring goods…")
            good_scores, gcost = await _score(goods, model)
            _bench_progress(f"review-{axis}: [{i}/{total}] {tag} scoring bads…")
            bad_scores, bcost = await _score(bads, model)
            res.cost = gcost + bcost
            spend += res.cost
        except Exception as exc:
            logger.warning("discrimination %s failed: %s", model, exc)
            res.error = str(exc)[:200]
            _bench_progress(f"review-{axis}: [{i}/{total}] {tag} FAILED: {exc}")
            results.append(res)
            continue

        res.n = len(good_scores) + len(bad_scores)
        res.metrics.update(discrimination_metrics(good_scores, bad_scores))
        res.metrics["n_good"] = float(len(good_scores))
        res.metrics["n_bad"] = float(len(bad_scores))
        if res.n == 0:
            res.error = f"0/{len(corpus)} items scored (no reviews aligned to corpus)"
        _bench_progress(
            f"review-{axis}: [{i}/{total}] {tag} done "
            f"auc={res.metrics.get('auc')} sep={res.metrics.get('separation')} "
            f"(run spend ${spend:.2f})"
        )
        results.append(res)

    if results and all(r.n == 0 for r in results):
        detail = "; ".join(f"{r.model}: {r.error}" for r in results)
        raise RuntimeError(
            "discrimination bench scored no items for any model — refusing to "
            f"save a hollow report ({detail})"
        )

    return RoleBenchReport(
        role=f"review-{axis}",
        results=results,
        incumbent=incumbent,
        sample_ids=[c["id"] for c in corpus],
        axis=axis,
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={
            "corpus": "labelled accepted (good) + seeded-defect twins (bad)",
            "defects": list(_DISCRIM_DEFECTS),
            "n_items": len(corpus),
        },
    )


async def run_concurrency_bench(
    models: list[str],
    concurrency: int = 16,
    reasoning_effort: str | None = None,
    gc: Any = None,
) -> RoleBenchReport:
    """Benchmark reviewer-seat provider resilience under concurrent load.

    Fires ``concurrency`` production docs-review calls at once per model and
    tallies success vs upstream rate-limiting (HTTP 429) vs empty response.
    This is the axis the review pools actually stress (up to 64 replicas): a
    high-quality reviewer whose OpenRouter provider 429s under load is unusable
    as a blind-pair seat regardless of its discrimination score.
    """
    import asyncio

    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.standard_names.benchmark import score_with_reviewer

    probe = load_discrimination_corpus(1, seed=0, axis="docs", gc=gc)
    if not probe:
        raise RuntimeError("concurrency bench: no accepted item to probe with")
    item = next((c for c in probe if c["label"] == 1), probe[0])

    def _classify(exc: Exception) -> str:
        m = str(exc).lower()
        if "429" in m or "rate" in m or "temporarily rate-limited" in m:
            return "rate_limited"
        if "empty response" in m:
            return "empty"
        if "503" in m or "502" in m or "overload" in m:
            return "unavailable"
        return "error"

    results: list[RoleModelResult] = []
    for model in models:
        res = RoleModelResult(model=model)
        prefixed = ensure_model_prefix(model)

        async def _one(_m: str = prefixed) -> tuple[str, float]:
            try:
                reviews, cost = await score_with_reviewer(
                    [dict(item)],
                    reviewer_model=_m,
                    target="docs",
                    reasoning_effort=reasoning_effort,
                )
                ok = bool(reviews) and reviews[0].get("score") is not None
                return ("ok" if ok else "empty", cost)
            except Exception as exc:  # noqa: BLE001 — outcome tally, not a raise
                return (_classify(exc), 0.0)

        outcomes = await asyncio.gather(*[_one() for _ in range(concurrency)])
        tally: dict[str, int] = {}
        for kind, cost in outcomes:
            tally[kind] = tally.get(kind, 0) + 1
            res.cost += cost
        res.n = concurrency
        ok = tally.get("ok", 0)
        res.metrics["success_rate"] = round(ok / concurrency, 4)
        res.metrics["rate_limited_rate"] = round(
            tally.get("rate_limited", 0) / concurrency, 4
        )
        res.metrics["empty_rate"] = round(tally.get("empty", 0) / concurrency, 4)
        res.metrics["ok"] = float(ok)
        if ok == 0:
            res.error = f"0/{concurrency} succeeded; outcomes={tally}"
        results.append(res)

    return RoleBenchReport(
        role="concurrency",
        results=results,
        sample_ids=[item["id"]],
        axis="docs",
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={"concurrency": concurrency, "probe_item": item["id"]},
    )


async def run_docs_bench(
    models: list[str],
    sample: list[dict],
    judge_model: str,
    incumbent: str | None = None,
    reasoning_effort: str | None = None,
) -> RoleBenchReport:
    """Benchmark the docs-generation seat: rubric score + banned-prose rate."""
    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.standard_names.benchmark import (
        generate_docs_for_candidates,
        score_with_reviewer,
    )
    from imas_codex.standard_names.context import build_compose_context

    context = build_compose_context()
    results: list[RoleModelResult] = []
    _bench_progress(f"docs: start {len(models)} model(s) x {len(sample)} names")
    for _di, model in enumerate(models, 1):
        res = RoleModelResult(model=model)
        _bench_progress(f"docs: [{_di}/{len(models)}] {model.split('/')[-1]}…")
        cands = [dict(c) for c in sample]
        try:
            cands, gcost, _ = await generate_docs_for_candidates(
                cands, ensure_model_prefix(model), context
            )
            res.cost += gcost
        except Exception as exc:
            logger.warning("docs gen %s failed: %s", model, exc)
            res.error = str(exc)[:200]
            results.append(res)
            continue

        # Banned-prose grep audit over generated documentation.
        total_findings = 0
        docs_with_findings = 0
        for c in cands:
            findings = banned_prose_findings(
                (c.get("documentation") or "")
                + "\n"
                + (c.get("docs_description") or "")
            )
            hit = sum(findings.values())
            total_findings += hit
            if hit:
                docs_with_findings += 1

        # Production docs rubric scoring (held-out judge).
        try:
            reviews, rcost = await score_with_reviewer(
                cands, reviewer_model=ensure_model_prefix(judge_model), target="docs"
            )
            res.cost += rcost
            scores = [float(r.get("score") or 0.0) for r in reviews]
        except Exception as exc:
            logger.warning("docs rubric scoring failed: %s", exc)
            scores = []

        res.n = len(cands)
        if scores:
            res.metrics["rubric_score"] = round(sum(scores) / len(scores), 4)
        res.metrics["banned_prose_rate"] = (
            round(docs_with_findings / res.n, 4) if res.n else 0.0
        )
        res.metrics["banned_prose_findings"] = float(total_findings)
        results.append(res)

    return RoleBenchReport(
        role="docs",
        results=results,
        incumbent=incumbent,
        judge_model=judge_model,
        sample_ids=[c.get("name", "") for c in sample],
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={"n_docs": len(sample)},
    )


async def run_classifier_bench(
    models: list[str], gold: list[dict], incumbent: str | None = None, gc: Any = None
) -> RoleBenchReport:
    """Benchmark the domain classifier seat: gold-set exact-match accuracy."""
    from imas_codex.core.physics_domain import PhysicsDomain
    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.dd_domain_classifier import (
        DEFAULT_BATCH_SIZE,
        _classify_batch,
        batch_by_subtree,
        gather_classification_context,
    )

    valid_domains = {d.value for d in PhysicsDomain}
    expected = {g["path"]: g["expected_domain"] for g in gold}
    path_ids = [g["path"] for g in gold]

    owns = gc is None
    gc = gc or GraphClient()
    try:
        contexts = gather_classification_context(gc, path_ids)
    finally:
        if owns:
            gc.close()
    ctx_by_id = {c["id"]: c for c in contexts}
    enriched = [{"id": pid, **ctx_by_id.get(pid, {})} for pid in path_ids]

    results: list[RoleModelResult] = []
    _cls_sem = asyncio.Semaphore(6)
    _bench_progress(f"classifier: start {len(models)} model(s) x {len(path_ids)} paths")
    for _ci, model in enumerate(models, 1):
        res = RoleModelResult(model=model)
        predicted: dict[str, str] = {}
        batches = batch_by_subtree(enriched, batch_size=DEFAULT_BATCH_SIZE)
        _bench_progress(f"classifier: [{_ci}/{len(models)}] {model.split('/')[-1]}…")

        async def _classify_one(
            batch: list[dict], _m: str = ensure_model_prefix(model)
        ) -> tuple[list[dict], float]:
            async with _cls_sem:
                try:
                    return await _classify_batch(
                        batch,
                        model=_m,
                        service="standard-names",
                        valid_domains=valid_domains,
                    )
                except Exception as exc:
                    logger.warning("classifier %s batch failed: %s", _m, exc)
                    return [], 0.0

        for batch_results, cost in await asyncio.gather(
            *(_classify_one(b) for b in batches)
        ):
            res.cost += cost
            for r in batch_results:
                predicted[r["id"]] = r["physics_domain"]
        acc, correct, total = exact_match_accuracy(predicted, expected)
        res.n = total
        res.metrics["accuracy"] = acc
        res.metrics["correct"] = float(correct)
        results.append(res)

    return RoleBenchReport(
        role="classifier",
        results=results,
        incumbent=incumbent,
        sample_ids=path_ids,
        timestamp=datetime.now(tz=UTC).isoformat(),
        provenance={"gold_set": "domain_gold_set.json", "n_paths": len(gold)},
    )


# ═══════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════


def render_role_report(report: RoleBenchReport) -> None:
    """Print a measured comparison table for a role benchmark."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    metric_keys: list[str] = []
    for r in report.results:
        for k in r.metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    table = Table(title=f"Role benchmark — {report.role}")
    table.add_column("model")
    table.add_column("n", justify="right")
    for k in metric_keys:
        table.add_column(k, justify="right")
    table.add_column("cost", justify="right")
    table.add_column("cost/item", justify="right")

    for r in report.results:
        marker = (
            " (incumbent)" if report.incumbent and report.incumbent in r.model else ""
        )
        row = [r.model.split("/")[-1] + marker, str(r.n)]
        for k in metric_keys:
            v = r.metrics.get(k)
            row.append(f"{v:.4f}" if isinstance(v, float) else "—")
        row.append(f"${r.cost:.4f}")
        row.append(f"${r.cost_per_item:.4f}")
        table.add_row(*row)

    console.print(table)
    if report.judge_model:
        console.print(f"  judge: {report.judge_model}")
    if report.incumbent:
        console.print(f"  incumbent: {report.incumbent}")
