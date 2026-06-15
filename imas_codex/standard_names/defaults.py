"""Centralized defaults for the standard-names refine pipeline (Phase 8.1).

All tunable thresholds and timing knobs live here. Override at the CLI or in
``pyproject.toml`` under ``[tool.imas-codex.sn]`` rather than editing this
file.
"""

# Reviewer-score thresholds
DEFAULT_MIN_SCORE: float = 0.85
"""Minimum reviewer score (0-1) for a name or docs revision to be marked accepted."""

# ── Semantic similarity gate ──────────────────────────────────────────
# Cosine similarity between embed(name_as_words) and embed(description).
# Names below CRITICAL are auto-failed (synthetic low score → refine).
# Names below WARNING get an advisory issue injected into review context.

SEMANTIC_SIM_CRITICAL: float = 0.55
"""Below this similarity, the name is semantically ambiguous. Review is
skipped and a synthetic low score (0.30) routes the name to refine."""

SEMANTIC_SIM_WARNING: float = 0.65
"""Below this similarity, a warning is injected into the reviewer context.
The reviewer still scores the name but is alerted to potential ambiguity."""

SEMANTIC_SIM_SYNTHETIC_SCORE: float = 0.30
"""Score assigned to names that fail the critical semantic similarity gate.
Low enough to guarantee routing to the refine_name pool."""

# Refine rotation cap
DEFAULT_REFINE_ROTATIONS: int = 2
"""Maximum REFINED_FROM (or DOCS_REVISION_OF) chain depth before exhaustion.

Tightened 3→2 (2026-06-15): exhaust non-converging names one attempt sooner.
Acceptance still overrides the cap (a passing score wins on the final
rotation), and the escalation model (opus-4.8) fires on that final attempt,
so a name that can be fixed still is — this only trims the tail that wasn't
going to pass. Note: the refine chain was never the review-cost driver
(0 names ever exceeded depth 3); per-name review inflation came from
re-review across runs, not refinement."""

# Escalation model — used on the final refine attempt before exhaustion.
# MUST be a different vendor from both compose (local deepseek-v4-flash)
# and refine (gpt-5.5): escalation exists to break a failure loop with an
# independent perspective, not to hand the item back to a model family
# that already failed it (user ruling 2026-06-11; previously defaulted to
# the compose model, which was backwards). opus-4.8 matches the quorum-
# breaker precedent; it fires only at chain cap so the cost is amortised.
DEFAULT_ESCALATION_MODEL: str = "openrouter/anthropic/claude-opus-4.8"
"""Model used on the final refine attempt (chain_length == cap-1)."""

# Orphan sweep timing
DEFAULT_ORPHAN_SWEEP_INTERVAL_S: int = 30
"""How often the orphan sweep coroutine runs (seconds)."""

DEFAULT_ORPHAN_SWEEP_TIMEOUT_S: int = 1800
"""How long a claim may sit before the orphan sweep clears it (seconds).

Bumped from 600 → 1800 s (2026-05-18) after the refine_name silent-bug
root cause analysis: 500+ refine claims produced only 5 REFINED_FROM
edges because long LLM calls (cross-vendor RD-quorum review + fan-out
expansion + write-queue backpressure) frequently exceeded the prior
10-minute window. The orphan sweep then reverted the 'refining' claim
to 'reviewed' BEFORE persist_refined_name could run, and the WHERE
clause in that persist silently failed (see graph_ops.py:7754).

30 minutes covers RD-quorum (3 LLM cycles) + fan-out evidence build
+ persist round-trip even under write-queue backpressure. Genuinely
orphaned claims (worker crashes) still recover, just on a longer
cadence."""

# ── Backlog throttle caps ─────────────────────────────────────────────
# Upstream generators pause when downstream review queues exceed these
# counts, preventing unbounded backlog growth and wasted budget.

REVIEW_NAME_BACKLOG_CAP: int = 200
"""Max review_name pending items before generate_name / refine_name pause."""

REVIEW_DOCS_BACKLOG_CAP: int = 200
"""Max review_docs pending items before generate_docs / refine_docs pause."""

# ── Deterministic-parent placeholder ──────────────────────────────────
# ``seed_parent_sources`` writes a structurally-derived parent SN
# (e.g. ``magnetic_field``) into the graph as a placeholder before
# ``GENERATE_DOCS`` produces an LLM-quality description. The placeholder
# description carries this exact marker so:
#   - it is obvious in the graph that the description is pending,
#   - ``sn export`` can refuse to emit any node whose description still
#     equals the placeholder (independent quality gate, defends against
#     ``reviewer_score_docs`` being absent or stale),
#   - tests + dashboards have one canonical string to detect "deterministic
#     parent that never had GENERATE_DOCS complete".
DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER: str = (
    "(deterministic parent — description pending LLM enrichment)"
)
"""Sentinel description written by ``seed_parent_sources``. Any export
that still sees this string for a parent SN signals that the docs
pipeline never completed for that name."""
