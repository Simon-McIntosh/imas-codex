"""Centralized defaults for the standard-names refine pipeline (Phase 8.1).

All tunable thresholds and timing knobs live here. Override at the CLI or in
``pyproject.toml`` under ``[tool.imas-codex.sn]`` rather than editing this
file.
"""

# Reviewer-score thresholds
DEFAULT_MIN_SCORE: float = 0.80
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
DEFAULT_REFINE_ROTATIONS: int = 3
"""Maximum REFINED_FROM (or DOCS_REVISION_OF) chain depth before exhaustion."""

# Escalation model — used on the final refine attempt before exhaustion
DEFAULT_ESCALATION_MODEL: str = "openrouter/anthropic/claude-opus-4.6"
"""Higher-capability model used on the final refine attempt (chain_length == cap-1)."""

# Orphan sweep timing
DEFAULT_ORPHAN_SWEEP_INTERVAL_S: int = 30
"""How often the orphan sweep coroutine runs (seconds)."""

DEFAULT_ORPHAN_SWEEP_TIMEOUT_S: int = 600
"""How long a claim may sit before the orphan sweep clears it (seconds).

RD-quorum reviews run 3 LLM cycles per item (primary, secondary,
optional escalator). With sequential batch processing, items later
in the batch can easily exceed 5 minutes. 600 s gives enough headroom
while still recovering genuinely orphaned claims within 10 minutes."""

# ── Backlog throttle caps ─────────────────────────────────────────────
# Upstream generators pause when downstream review queues exceed these
# counts, preventing unbounded backlog growth and wasted budget.

REVIEW_NAME_BACKLOG_CAP: int = 200
"""Max review_name pending items before generate_name / refine_name pause."""

REVIEW_DOCS_BACKLOG_CAP: int = 200
"""Max review_docs pending items before generate_docs / refine_docs pause."""
