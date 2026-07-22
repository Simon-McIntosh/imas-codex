"""Async workers for the standard-name build pipeline.

Five-phase generate pipeline:

    EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST

- **extract**: queries graph for DD paths or facility signals, builds batches
- **compose**: LLM-generates standard names from extraction batches
- **validate**: validates names against grammar via round-trip + fields check
- **consolidate**: cross-batch dedup, conflict detection, coverage accounting
- **persist**: writes consolidated names to graph with provenance

Workers follow the ``dd_workers.py`` pattern: each is an async function
with signature ``async def worker(state, **_kwargs)`` that updates stats,
marks phases done, and respects ``state.should_stop()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re as _re
from collections.abc import Callable, Sequence
from functools import cache as _cache
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)
from imas_codex.standard_names.provenance import detect_value_provenance
from imas_codex.standard_names.source_paths import (
    encode_source_path,
    strip_dd_prefix,
)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease, BudgetManager
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import StandardNameBuildState

logger = logging.getLogger(__name__)


def normalize_spelling(name: str) -> str:
    """Deterministic British→American spelling normalization.

    Splits the underscore-delimited name into tokens and converts each
    via ``breame.spelling.get_american_spelling``.  A small domain-specific
    supplement catches physics terms missing from breame's dictionary.
    """
    from breame.spelling import get_american_spelling

    tokens = name.split("_")
    return "_".join(_DOMAIN_UK_US.get(t, get_american_spelling(t)) for t in tokens)


def normalize_prose_spelling(text: str) -> str:
    """British→American spelling normalization for prose text.

    Unlike ``normalize_spelling`` which operates on underscore-delimited
    name tokens, this function handles natural English prose — splitting
    on word boundaries and preserving punctuation, whitespace, and
    markdown formatting.
    """
    if not text:
        return text
    from breame.spelling import get_american_spelling

    def _replace(match: _re.Match) -> str:
        word = match.group(0)
        # Preserve case: if the word is capitalized, capitalize the replacement
        us = _DOMAIN_UK_US.get(word.lower(), get_american_spelling(word.lower()))
        if us == word.lower():
            return word  # no change
        if word[0].isupper():
            return us[0].upper() + us[1:]
        return us

    # Match word-like sequences (letters only, preserving everything else)
    return _re.sub(r"[A-Za-z]+", _replace, text)


# Spelled-out Greek letters become their Unicode symbols in description prose
# (the ISN description-field convention: φ, θ, ρ; frames as (R, φ, Z)).
# Lowercase only — a capitalized "Phi" may be the flux Φ or a sentence-start
# angle, so it is left for review rather than guessed.
_GREEK_WORD_TO_SYMBOL: dict[str, str] = {
    "phi": "φ",
    "theta": "θ",
    "rho": "ρ",
}
_GREEK_WORD_RE = _re.compile(r"\b(phi|theta|rho)\b")

# Backslash-escaped Greek commands: \phi \theta \rho \pi (and capitals) →
# Unicode symbol. Includes \pi because it is the common LaTeX form that leaks
# into descriptions. The trailing (?![a-zA-Z]) stops \phinx-style over-matching.
_GREEK_CMD_TO_SYMBOL: dict[str, str] = {
    "phi": "φ",
    "theta": "θ",
    "rho": "ρ",
    "pi": "π",
    "Phi": "Φ",
    "Theta": "Θ",
    "Rho": "Ρ",
    "Pi": "Π",
}
_GREEK_CMD_RE = _re.compile(r"\\(phi|theta|rho|pi|Phi|Theta|Rho|Pi)(?![a-zA-Z])")
# A backslash stranded in front of an already-Unicode Greek symbol — the
# half-converted corruption (\φ) produced by the older word-only normalizer.
_GREEK_STRANDED_BACKSLASH_RE = _re.compile(r"\\(?=[φθρπΦΘΡΠ])")
# A LaTeX command carrying a single braced argument: \mathbf{k} → k.
_LATEX_CMD_ARG_RE = _re.compile(r"\\[a-zA-Z]+\s*\{([^{}]*)\}")
# A bare LaTeX command with no argument: \left \right \cdot \, → removed.
_LATEX_CMD_BARE_RE = _re.compile(r"\\[a-zA-Z]+")
# Braced sub/superscripts: ^{1+} → 1+ ; _{i} → _i (keep the underscore).
_SUPERSCRIPT_BRACE_RE = _re.compile(r"\^\{([^{}]*)\}")
_SUBSCRIPT_BRACE_RE = _re.compile(r"_\{([^{}]*)\}")
# Math delimiters: $$…$$ and $…$ — drop the delimiter, keep inner content.
_MATH_DELIM_RE = _re.compile(r"\$\$?")


def normalize_description_notation(text: str) -> str:
    """Normalize description prose to plain Unicode text.

    Descriptions are plain text (the ISN convention); LaTeX and math markup
    belong in the ``documentation`` field. Any LaTeX that leaks into a
    description is stripped to a readable plain form:

    - ``$…$`` / ``$$…$$`` delimiters removed (inner content kept);
    - ``\\phi \\theta \\rho \\pi`` → ``φ θ ρ π`` (plus capitals), and a
      stranded backslash in front of an already-Unicode symbol (``\\φ``) → φ;
    - spelled-out ``phi/theta/rho`` prose words → ``φ/θ/ρ`` (word-bounded, so
      DD tokens like ``phi_tor`` / ``b_field_phi`` are untouched);
    - ``\\mathbf{k}`` → ``k`` and other backslash commands dropped;
    - braced sub/superscripts (``^{1+}`` / ``_{i}``) flattened.

    Idempotent — a description already in plain form is returned unchanged.
    Applies to the description field only; ``documentation`` keeps LaTeX.
    """
    if not text:
        return text
    # 1. Backslash Greek commands and stranded backslashes → Unicode symbols.
    text = _GREEK_CMD_RE.sub(lambda m: _GREEK_CMD_TO_SYMBOL[m.group(1)], text)
    text = _GREEK_STRANDED_BACKSLASH_RE.sub("", text)
    # 2. Spelled-out Greek prose words → Unicode (word-bounded; DD tokens safe).
    text = _GREEK_WORD_RE.sub(lambda m: _GREEK_WORD_TO_SYMBOL[m.group(1)], text)
    # 3. LaTeX commands: keep the argument of \cmd{arg}, drop bare commands.
    text = _LATEX_CMD_ARG_RE.sub(r"\1", text)
    text = _LATEX_CMD_BARE_RE.sub("", text)
    # 4. Flatten braced sub/superscripts, then drop any residual braces.
    text = _SUPERSCRIPT_BRACE_RE.sub(r"\1", text)
    text = _SUBSCRIPT_BRACE_RE.sub(r"_\1", text)
    text = text.replace("{", "").replace("}", "")
    # 5. Drop math delimiters (inner content already kept).
    text = _MATH_DELIM_RE.sub("", text)
    # 6. Drop any residual backslash — plain text never carries one.
    text = text.replace("\\", "")
    # 7. Collapse runs of spaces/tabs left behind by the removals above.
    text = _re.sub(r"[ \t]{2,}", " ", text)
    return text


def normalize_description_text(text: str) -> str:
    """Full description-prose normalization: spelling, then Greek symbols."""
    return normalize_description_notation(normalize_prose_spelling(text))


# Physics-domain UK→US supplements not in breame's dictionary.
_DOMAIN_UK_US: dict[str, str] = {
    "linearised": "linearized",
    "linearise": "linearize",
    "discretised": "discretized",
    "discretise": "discretize",
    "symmetrised": "symmetrized",
    "symmetrise": "symmetrize",
    "parameterised": "parameterized",
    "parameterise": "parameterize",
    "centreline": "centerline",
    "centre": "center",
    "grey": "gray",
    "colour": "color",
    "favour": "favor",
    "favourite": "favorite",
    "honour": "honor",
    "labelled": "labeled",
    "labelling": "labeling",
    "travelled": "traveled",
    "travelling": "traveling",
    "catalogue": "catalog",
    "programme": "program",
    "analogue": "analog",
    "dialogue": "dialog",
}

_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
    "geometric_base",
    "object",
)


def _dedup_adjacent_tokens(name: str, log: logging.Logger | None = None) -> str:
    """Remove adjacent duplicate tokens introduced by grammar round-trip.

    The ISN grammar (≤ 0.7.0rc27) has a known bug where ``parse →
    compose`` doubles certain tokens — e.g.
    ``magnetic_field_probe`` → ``magnetic_magnetic_field_probe``.

    This function collapses any ``tok_tok`` pair into a single ``tok``
    while preserving legitimate compounds via
    :data:`~imas_codex.standard_names.audits._COMPOUND_SUBJECT_PAIRS`.
    """
    from imas_codex.standard_names.audits import _COMPOUND_SUBJECT_PAIRS

    tokens = name.split("_")
    result: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok == result[-1] and f"{tok}_{tok}" not in _COMPOUND_SUBJECT_PAIRS:
            if log:
                log.debug(
                    "Dedup adjacent token %r in %r",
                    f"{tok}_{tok}",
                    name,
                )
            continue
        result.append(tok)
    return "_".join(result)


# ---------------------------------------------------------------------------
# Shape-parameter surface injection
# ---------------------------------------------------------------------------
#
# Dimensionless plasma shape descriptors are only meaningful *of* a surface.
# A bare ``triangularity`` conflates the boundary scalar
# (``equilibrium/.../boundary/triangularity``) with the flux-surface radial
# profile (``.../profiles_1d/triangularity``) and is not itself a measurable
# quantity. When the composer omits the surface locus we inject it
# deterministically from the source DD path so the leaf is always
# surface-explicit (``triangularity_of_plasma_boundary`` /
# ``triangularity_of_flux_surface``) — never bare. This also de-conflates the
# boundary and profile siblings into distinct names instead of one ambiguous
# parent. Derived structural parents are reached later by stripping
# qualifiers/projections, never by stripping a base's defining surface — so no
# bare shape-parameter parent is ever admitted (see ``parents.py``).
_SHAPE_PARAMETER_BASES = frozenset({"triangularity", "elongation", "squareness"})


def _shape_parameter_surface(dd_path: str | None) -> str:
    """Surface a shape parameter from *dd_path* is defined *of*.

    - ``profiles_1d`` / ``flux_surface`` paths → ``flux_surface`` (radial profile)
    - all other paths (``boundary``, ``boundary_separatrix``, pulse-schedule
      control targets, …) → ``plasma_boundary`` (the LCFS)

    The default is the plasma boundary: "the triangularity" without further
    qualification conventionally means the boundary/separatrix value.
    """
    p = (dd_path or "").lower()
    if "/profiles_1d/" in p or "/flux_surface/" in p:
        return "flux_surface"
    return "plasma_boundary"


def _inject_shape_parameter_surface(
    candidate: Any, dd_path: str | None, log: logging.Logger | None = None
) -> bool:
    """Force a surface locus onto a bare shape-parameter candidate.

    Mutates ``candidate.segments`` in place so ``compose_name()`` produces the
    surface-explicit leaf. Deterministic and idempotent: fires only when the
    base is a shape parameter (quantity kind) and the composer left the locus
    empty; a composer-supplied locus is always preserved. Returns ``True`` when
    an injection was applied.
    """
    seg = getattr(candidate, "segments", None)
    if seg is None:
        return False
    if seg.base_kind != "quantity" or seg.base_token not in _SHAPE_PARAMETER_BASES:
        return False
    if seg.locus_token:  # composer already named a surface — trust it
        return False
    surface = _shape_parameter_surface(dd_path)
    seg.locus_token = surface
    seg.locus_relation = "of"
    seg.locus_type = "geometry"
    if log:
        log.debug(
            "shape-param surface injection: base=%s src=%s -> _of_%s",
            seg.base_token,
            dd_path,
            surface,
        )
    return True


# ---------------------------------------------------------------------------
# Pre-validation gate (W4b): reject malformed LLM output before MERGE
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE = _re.compile(r"[a-z][a-z0-9_]*")


def is_well_formed_candidate(raw_name: str) -> tuple[bool, str | None]:
    """Check whether *raw_name* is a plausible standard-name candidate.

    Returns ``(True, None)`` if the name passes all checks, or
    ``(False, reason)`` if it should be rejected **before** MERGE.

    Rejection criteria:
    - Empty or whitespace-only
    - Length > 100 characters (likely LLM monologue)
    - Contains newline, tab, ``{``, ``}``, ``\\``, or triple-dot
    - Not snake_case-shaped after stripping
    """
    if not raw_name or not raw_name.strip():
        return False, "empty_or_whitespace"
    name = raw_name.strip()
    if len(name) > 100:
        return False, "too_long"
    if any(ch in name for ch in ("\n", "\t", "{", "}", "\\")):
        return False, "illegal_chars"
    if "..." in name:
        return False, "triple_dot"
    if not _SNAKE_CASE_RE.fullmatch(name):
        return False, "not_snake_case"
    return True, None


def provenance_canonical_names(
    candidates: list, log: logging.Logger | None = None
) -> dict[str, str]:
    """Map each base quantity to ONE canonical name for its estimator facets.

    Per the locked ``name-multiplicity-rule`` (collapse provenance), the
    measured / reconstructed / reference estimates of one physical quantity must
    share ONE standard name — but the per-facet composer is non-deterministic
    (e.g. ``pressure/measured`` -> ``plasma_pressure`` while
    ``pressure/reconstructed`` -> ``total_plasma_pressure``). Estimator facets of
    a quantity strip to the same ``base_path`` (and share a compose batch via
    ``batch_key``), so we group every value-provenance candidate by base and
    pick one canonical name; the caller overrides each facet's ``name_id`` (the
    MERGE identity) with it, collapsing the group onto a single StandardName.

    Canonical pick: most-frequently-composed name, ties broken by SHORTEST then
    lexicographic. Shortest implements the "physics, not instrument" /
    no-redundant-qualifier preference deterministically — it drops
    ``_of_flux_loop`` (vs ``poloidal_magnetic_flux``) and ``total_plasma_pressure``
    (vs ``plasma_pressure``). Returns ``{base_path: canonical_name}`` only for
    bases whose facets disagree (a no-op map otherwise).
    """
    from collections import Counter

    groups: dict[str, list[str]] = {}
    for c in candidates:
        term, base = detect_value_provenance(getattr(c, "source_id", "") or "")
        if not term:
            continue
        try:
            nm = normalize_spelling(c.compose_name())
        except Exception:
            continue
        groups.setdefault(base, []).append(nm)

    canonical: dict[str, str] = {}
    for base, member_names in groups.items():
        distinct = set(member_names)
        if len(distinct) <= 1:
            continue  # already agree
        counts = Counter(member_names)
        winner = min(distinct, key=lambda n: (-counts[n], len(n), n))
        canonical[base] = winner
        if log:
            log.info(
                "provenance canon: base=%s -> %s (collapsed %d variants: %s)",
                base,
                winner,
                len(distinct),
                sorted(distinct),
            )
    return canonical


# Bare coordinate/infrastructure tokens that are NOT physics observables.
# A standard name composed from one of these is doomed: it fails the
# semantic-similarity gate (a reader cannot tell what is measured), then burns
# review + every refine rotation before exhausting.  Catch them at compose time
# and route the source to ``skipped`` instead of composing a doomed bare name.
_NON_NAMEABLE_BARE_TOKENS: frozenset[str] = frozenset(
    {
        "time",  # independent signal coordinate / timestamp
        "time_stamp",
        "timestamp",
        "delay",  # signal-chain latency
        "latency",
        "dead_time",
        "index",  # array bookkeeping
        "count",
        "counter",
        "version",  # pure metadata
    }
)


def is_non_nameable_coordinate(raw_name: str) -> tuple[bool, str | None]:
    """Detect bare coordinate / infrastructure tokens that get no standard name.

    Returns ``(True, reason)`` when *raw_name* is a bare non-nameable token
    (a time coordinate / timestamp, signal-chain timing, an array counter, or
    pure metadata) that should be routed to ``skipped`` rather than composed.
    Returns ``(False, None)`` otherwise.

    This is intentionally conservative: it only fires on *bare* tokens (no
    qualifier/locus context). A qualified name like ``confinement_time`` or
    ``major_radius_of_magnetic_axis`` is a real quantity and passes through.
    """
    if not raw_name:
        return False, None
    name = raw_name.strip()
    if name in _NON_NAMEABLE_BARE_TOKENS:
        return True, f"non_nameable_coordinate:{name}"
    return False, None


def _coerce_llm_metric(value: Any) -> int:
    """Best-effort int coercion for LLM telemetry fields."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_llm_telemetry(
    llm_out: Any,
    *,
    parsed_obj: Any = None,
    tokens_fallback: Any = None,
) -> tuple[int, int, int, int]:
    """Normalize split token/cache telemetry across LLM return shapes."""

    def _from_attrs(obj: Any) -> tuple[int, int, int, int]:
        return (
            _coerce_llm_metric(getattr(obj, "input_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "output_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "cache_read_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "cache_creation_tokens", 0)),
        )

    metrics = _from_attrs(llm_out)
    if any(metrics):
        return metrics

    if parsed_obj is not None:
        metrics = _from_attrs(parsed_obj)
        if any(metrics):
            return metrics

    if isinstance(tokens_fallback, dict):
        return (
            _coerce_llm_metric(tokens_fallback.get("input_tokens", 0)),
            _coerce_llm_metric(tokens_fallback.get("output_tokens", 0)),
            _coerce_llm_metric(tokens_fallback.get("cache_read_tokens", 0)),
            _coerce_llm_metric(
                tokens_fallback.get(
                    "cache_creation_tokens",
                    tokens_fallback.get("cache_write_tokens", 0),
                )
            ),
        )

    if isinstance(tokens_fallback, tuple | list):
        values = list(tokens_fallback)
        return (
            _coerce_llm_metric(values[0] if len(values) > 0 else 0),
            _coerce_llm_metric(values[1] if len(values) > 1 else 0),
            _coerce_llm_metric(values[2] if len(values) > 2 else 0),
            _coerce_llm_metric(values[3] if len(values) > 3 else 0),
        )

    return (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Auto-VocabGap detection for physical_base (W29)
# ---------------------------------------------------------------------------


@_cache
def _load_known_physical_bases() -> frozenset[str]:
    """Return the set of registered physical_base tokens from ISN.

    Reads the ``grammar/vocabularies/physical_bases.yml`` vocabulary file
    from the installed ``imas_standard_names`` package.  Falls back to an
    empty set if the package or file is unavailable.
    """
    try:
        from importlib.resources import files

        import yaml

        text = (
            files("imas_standard_names")
            .joinpath("grammar/vocabularies/physical_bases.yml")
            .read_text()
        )
        data = yaml.safe_load(text)
        bases = data.get("bases", {})
        if isinstance(bases, dict):
            return frozenset(bases.keys())
        return frozenset()
    except Exception:
        return frozenset()


def _auto_detect_physical_base_gaps(
    candidates: list,  # StandardNameCandidate instances or dicts
    known_bases: frozenset[str] | None = None,
) -> list[dict]:
    """Parse each candidate name and surface novel ``physical_base`` tokens.

    Returns a list of VocabGap-compatible dicts for bases not in the ISN
    registry.  These are auto-tracked so the LLM does not need to emit
    explicit ``vocab_gap`` exits for ``physical_base``.

    Args:
        candidates: Parsed ``StandardNameCandidate`` objects or dicts.
        known_bases: Pre-loaded set of registered physical_base tokens.
            Defaults to ``_load_known_physical_bases()`` if not provided.
            The flat registry under-reports the grammar's lexical-compound
            bases (``internal_inductance``, ``major_radius`` …); on the default
            path a base absent from the flat set is re-checked against the ISN
            parser and emitted as a gap only when the parser also rejects it.
            An explicit ``known_bases`` is treated as authoritative and the
            parser escape hatch is NOT applied.
    """
    from imas_codex.standard_names.segments import is_known_physical_base

    consult_parser = known_bases is None
    if known_bases is None:
        known_bases = _load_known_physical_bases()

    gaps: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (source_id, base) dedup

    for c in candidates:
        try:
            # Handle both model instances and dicts
            if hasattr(c, "base_token"):
                base = c.base_token
                source_id = c.source_id
                base_kind = getattr(c, "base_kind", "quantity")
            elif isinstance(c, dict):
                base = c.get("id", "")  # dict candidates use 'id' for name
                source_id = c.get("source_id", "")
                # For dicts, parse the name to extract base
                try:
                    from imas_standard_names.grammar import parse_standard_name

                    parsed = parse_standard_name(base)
                    base = parsed.physical_base
                    base_kind = "quantity"
                except Exception:
                    continue
            else:
                continue

            if base_kind == "quantity" and base and base not in known_bases:
                # The grammar resolves lexical-compound bases that never appear
                # in the flat registry — those are valid, not gaps.  Skip them
                # on the default path; an explicit known_bases stays strict.
                if consult_parser and is_known_physical_base(base):
                    continue
                key = (source_id, base)
                if key not in seen:
                    seen.add(key)
                    gaps.append(
                        {
                            "source_id": source_id,
                            "segment": "physical_base",
                            "token": base,
                            "reason": (
                                f"Novel physical_base proposed by compose: {base!r}"
                            ),
                        }
                    )
        except Exception:
            continue  # parse failures already handled elsewhere

    return gaps


def _compute_token_reuse_hits(vocab_gaps: list) -> dict[tuple[str, str], Any]:
    """Flag candidate-new tokens that duplicate a registered same-segment token.

    For each ``vocab_gap`` the compose LLM emitted this attempt, embeds the
    proposed token and compares it against its segment's registered vocabulary
    (free local embeddings, batched by segment so each segment's vocab is
    embedded once).  Returns a map ``(segment, token) -> TokenNeighbour`` for
    every proposal scoring at/above ``get_sn_dedup_threshold()``.

    Best-effort: never raises (the helper itself is defensive); the local
    compose model + local embeddings add no OpenRouter spend.  Returns ``{}``
    when there are no gaps or no near-synonym registered token.
    """
    if not vocab_gaps:
        return {}

    from imas_codex.settings import get_sn_dedup_threshold
    from imas_codex.standard_names.vocab_semantic_dedup import (
        nearest_registered_tokens,
    )

    threshold = get_sn_dedup_threshold()
    # threshold > 1.0 can never be met — short-circuit (OFF arm of the A/B).
    if threshold > 1.0:
        return {}

    # Unique (token, segment) pairs; the segment vocab is embedded once each.
    items = sorted({(vg.token, vg.segment) for vg in vocab_gaps})
    hits = nearest_registered_tokens(items, threshold=threshold)
    # Re-key by (segment, token) for stamp/lookup ergonomics.
    return {(hit.segment, hit.proposed_token): hit for hit in hits.values()}


def _build_token_reuse_retry_reason(hits: dict[tuple[str, str], Any]) -> str:
    """Render a retry directive naming each flagged candidate-new token.

    The agent re-composes after seeing this: it either reuses the registered
    token (dedup achieved) or re-emits the same gap (confirming the quantity is
    genuinely distinct on that segment's axis, recorded as distinct-confirmed).
    """
    parts = []
    for hit in hits.values():
        parts.append(
            f"Segment `{hit.segment}`: your proposed new token "
            f"`{hit.proposed_token}` is {hit.similarity:.2f} similar to "
            f"registered `{hit.nearest_token}`. Reuse the registered token "
            f"unless this quantity is genuinely DISTINCT on the {hit.segment} "
            f"axis; if distinct, keep it and it will be recorded as a "
            f"new-vocab request."
        )
    return "Vocabulary reuse check — re-compose to reuse-or-confirm:\n" + "\n".join(
        parts
    )


def _stamp_dedup_decision(
    gap_dicts: list[dict],
    hits: dict[tuple[str, str], Any],
) -> None:
    """Stamp the compose-time token-reuse adjudication onto gap dicts in place.

    A gap whose ``(segment, token)`` was flagged and SURVIVED the chained retry
    (the agent re-emitted it after being shown the near-token) is
    ``distinct_confirmed`` and carries the nearest token + score.  Gaps that
    were never flagged are ``unchecked``.  (Gaps the agent dropped after the
    prompt — "reused" — are no longer in ``result.vocab_gaps`` and so are never
    written.)
    """
    for g in gap_dicts:
        hit = hits.get((g.get("segment"), g.get("token")))
        if hit is not None:
            g["nearest_token"] = hit.nearest_token
            g["nearest_similarity"] = hit.similarity
            g["dedup_decision"] = "distinct_confirmed"
        else:
            g["dedup_decision"] = "unchecked"


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Extract candidate quantities from graph entities into batches.

    For DD source: queries IMASNode paths, groups by cluster/IDS/prefix.
    Skips sources already linked via HAS_STANDARD_NAME unless --force.
    Stores ExtractionBatch objects in ``state.extracted``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_extract_worker")
    wlog.info("Starting extraction (source=%s)", state.source)

    def _on_status(text: str) -> None:
        state.extract_stats.status_text = text
        if state.loop_extract_stats is not None:
            state.loop_extract_stats.status_text = text
            # Stream sub-stage events so the user sees a timeline rather than
            # a single status string that can appear stuck during long
            # downstream phases (compose, enrich, review).
            try:
                state.loop_extract_stats.stream_queue.add(
                    [
                        {
                            "primary_text": "extract",
                            "primary_text_style": "dim cyan",
                            "description": text,
                        }
                    ]
                )
            except Exception:
                pass

    def _run() -> list:
        from imas_codex.standard_names.graph_ops import (
            get_existing_standard_names,
            get_named_source_ids,
        )
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        _on_status("loading existing names…")
        existing = get_existing_standard_names()

        # Regen mode: when --min-score F is passed, re-queue sources whose
        # linked StandardName has reviewer_score < min_score. The subsequent
        # feedback-injection block (always-on) attaches reviewer_comments /
        # tier / scores to each item so compose regenerates with critique
        # in-prompt.
        if state.is_regen_mode() and state.source == "dd":
            from imas_codex.standard_names.graph_ops import (
                fetch_low_score_sources,
            )
            from imas_codex.standard_names.sources.dd import extract_specific_paths

            _on_status(f"loading sources below reviewer_score={state.min_score}…")
            regen_sources = fetch_low_score_sources(
                min_score=state.min_score,
                domain=state.domain_filter,
                ids=state.ids_filter,
                limit=state.limit,
                source_type="dd",
            )
            if not regen_sources:
                wlog.info(
                    "Regen mode (--min-score %s): no low-score sources found "
                    "(domain=%s, ids=%s, limit=%s)",
                    state.min_score,
                    state.domain_filter,
                    state.ids_filter,
                    state.limit,
                )
                return []
            wlog.info(
                "Regen mode: %d sources below reviewer_score=%s "
                "(domain=%s, ids=%s, limit=%s)",
                len(regen_sources),
                state.min_score,
                state.domain_filter,
                state.ids_filter,
                state.limit,
            )
            paths = [r["source_id"] for r in regen_sources]
            batches = extract_specific_paths(
                paths=paths,
                existing_names=existing,
                on_status=_on_status,
            )
            return batches

        # Source-level skip for resumability
        named_ids: set[str] = set()
        if not state.force:
            named_ids = get_named_source_ids()
            if named_ids:
                wlog.info("Skipping %d already-named sources", len(named_ids))

        if state.source == "dd":
            from imas_codex.standard_names.batching import (
                get_generate_batch_config,
            )

            batch_cfg = get_generate_batch_config()
            batches = extract_dd_candidates(
                ids_filter=state.ids_filter,
                domain_filter=state.domain_filter,
                limit=state.limit,
                existing_names=existing,
                on_status=_on_status,
                from_model=state.from_model,
                force=state.force,
                name_only=state.name_only,
                name_only_batch_size=state.name_only_batch_size,
                max_batch_size=batch_cfg["batch_size"],
                max_tokens=batch_cfg["max_tokens"],
            )
        else:
            wlog.error("Unknown source: %s", state.source)
            return []

        # Filter out already-named sources from batches
        if named_ids and not state.force:
            for batch in batches:
                batch.items = [
                    item
                    for item in batch.items
                    if item.get("path", item.get("signal_id")) not in named_ids
                ]
            # Remove empty batches
            batches = [b for b in batches if b.items]

        return batches

    batches = await asyncio.to_thread(_run)

    # Inject previous name context for --force regeneration
    if state.force:
        # --paths mode gets rich metadata (full docs, links, linked DD paths)
        use_rich = False

        def _get_mapping():
            from imas_codex.standard_names.graph_ops import get_source_name_mapping

            return get_source_name_mapping(rich=use_rich)

        source_names = await asyncio.to_thread(_get_mapping)
        injected = 0
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path and path in source_names:
                    item["previous_name"] = source_names[path]
                    injected += 1
        if injected:
            wlog.info("Injected previous_name context for %d items", injected)

    # Inject prior reviewer feedback for targeted regeneration (always on).
    # The compose prompt's {% if item.review_feedback %} block surfaces the
    # previous reviewer critique so the LLM can directly address it in the
    # new name. No-op when no prior feedback exists.
    def _get_feedback():
        from imas_codex.standard_names.graph_ops import (
            fetch_review_feedback_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_review_feedback_for_sources(ids)

    feedback_map = await asyncio.to_thread(_get_feedback)
    fb_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in feedback_map:
                item["review_feedback"] = feedback_map[path]
                fb_injected += 1
    if fb_injected:
        wlog.info(
            "Injected review_feedback for %d items",
            fb_injected,
        )

    # Inject full reviewer history from Review nodes (Phase 3).
    # Complements the single-review review_feedback above with the
    # complete chain of prior reviews — latest full + older themes.
    def _get_reviewer_history():
        from imas_codex.standard_names.graph_ops import (
            fetch_reviewer_history_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_reviewer_history_for_sources(ids)

    history_map = await asyncio.to_thread(_get_reviewer_history)
    hist_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in history_map:
                item["reviewer_history"] = history_map[path]
                hist_injected += 1
    if hist_injected:
        wlog.info(
            "Injected reviewer_history for %d items",
            hist_injected,
        )

    # Write StandardNameSource nodes for crash-resilient tracking
    if not state.dry_run and batches:
        from imas_codex.settings import get_dd_version
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        _dd_ver = get_dd_version()
        sources = []
        source_type = "dd" if state.source == "dd" else "signals"
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if not path:
                    continue
                # Per-source dicts carry no version (the re-seed contract); the
                # current version is the batch default, pinning only new sources
                # while re-seeds keep their immutable stored pin.
                sources.append(
                    {
                        "id": f"{source_type}:{path}",
                        "source_type": source_type,
                        "source_id": path,
                        "dd_path": path if source_type == "dd" else None,
                        "batch_key": batch.group_key,
                        "status": "extracted",
                        "description": item.get("description")
                        or item.get("documentation")
                        or "",
                    }
                )

        if sources:
            written = await asyncio.to_thread(
                merge_standard_name_sources,
                sources,
                force=state.force,
                default_dd_version=_dd_ver,
            )
            wlog.info("Wrote %d StandardNameSource nodes to graph", written)

    total_items = sum(len(b.items) for b in batches)
    state.extracted = batches
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.total = total_items
        state.loop_extract_stats.processed = total_items
        state.loop_extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d items",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_stats.status_text = ""
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} paths in {len(batches)} batches",
            }
        ]
    )
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.freeze_rate()
        state.loop_extract_stats.status_text = ""
        state.loop_extract_stats.stream_queue.add(
            [
                {
                    "primary_text": "extract",
                    "description": f"{total_items} paths in {len(batches)} batches",
                }
            ]
        )


# =============================================================================
# COMPOSE phase
# =============================================================================


def _search_nearby_names(query: str, k: int = 5) -> list[dict]:
    """Search for existing standard names near *query* for collision avoidance.

    Wraps :func:`imas_codex.standard_names.search.search_standard_names_vector`
    with graceful fallback — never raises, returns ``[]`` if graph or
    embeddings are unavailable.
    """
    try:
        from imas_codex.standard_names.search import search_standard_names_vector

        return search_standard_names_vector(query, k=k)
    except Exception:
        return []


# =============================================================================
# DD context enrichment — fetch rich graph data before composing
# =============================================================================

_DD_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:IMASNode)
WHERE sibling.id <> $path
  AND NOT (sibling.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
WITH n, ident, u, parent,
     collect(DISTINCT {
         path: sibling.id,
         description: sibling.description,
         data_type: sibling.data_type
     })[0..8] AS sibling_fields
RETURN n.coordinate1_same_as AS coordinate1,
       n.coordinate2_same_as AS coordinate2,
       n.coordinate3_same_as AS coordinate3,
       n.timebasepath AS timebase,
       n.cocos_transformation_type AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       n.lifecycle_status AS lifecycle_status,
       n.data_type AS data_type,
       n.description AS node_description,
       n.documentation AS node_documentation,
       ident.name AS identifier_schema_name,
       ident.documentation AS identifier_schema_doc,
       ident.options AS identifier_options,
       u.id AS unit_from_rel,
       parent.id AS parent_path,
       parent.description AS parent_description,
       n.enrichment_source AS enrichment_source,
       sibling_fields
"""

# Ancestor lineage context — walk HAS_PARENT up to the IDS root and collect
# every ancestor that carries a non-empty description/documentation. A DD
# *value* leaf is often a terse template stub (e.g. a per-species
# ``.../velocity_phi/<species>/value`` reads only "Deuterium (D)."), while the
# physically-meaningful text — the quantity AND its evaluation locus (e.g.
# "Ion toroidal rotation velocity … at the pedestal top") — lives on a parent
# quantity node. Surfacing the ancestor lineage lets both name and docs
# composition ground on the real physics + locus instead of the stub. Rich
# LLM-enriched descriptions are preferred over terse DD documentation; nearest
# ancestor first.
_ANCESTOR_CONTEXT_QUERY = """
MATCH path = (n:IMASNode {id: $path})-[:HAS_PARENT*1..8]->(a:IMASNode)
WHERE coalesce(trim(a.description), '') <> ''
   OR coalesce(trim(a.documentation), '') <> ''
RETURN length(path) AS depth,
       a.id AS path,
       a.description AS description,
       a.documentation AS documentation,
       a.enrichment_source AS enrichment_source
ORDER BY depth ASC
"""

_CROSS_IDS_QUERY = """
MATCH (n:IMASNode {id: $path})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH c ORDER BY c.scope ASC LIMIT 3
MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
WHERE member.id <> $path
  AND NOT (member.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
RETURN c.label AS cluster_label,
       c.description AS cluster_description,
       c.scope AS cluster_scope,
       collect(DISTINCT member.id)[0..5] AS member_paths
"""

_VERSION_HISTORY_QUERY = """
MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n:IMASNode {id: $path})
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
RETURN vc.id AS change_id, vc.change_type AS change_type
"""

_ERROR_FIELDS_QUERY = """
MATCH (d:IMASNode {id: $path})-[:HAS_ERROR]->(e:IMASNode)
RETURN e.id AS error_path
"""

# Per-member DD documentation grounding: a compose candidate's name can span
# N sibling leaves (family_siblings, cross-IDS equivalents). Fetch the terse
# DD-XML documentation for those member paths so the name is grounded on every
# leaf it covers, not just the primary. Terse doc only — DISTINCT from the rich
# enriched description the primary already renders.
_MEMBER_DOC_QUERY = """
UNWIND $paths AS p
MATCH (n:IMASNode {id: p})
RETURN n.id AS path, n.documentation AS documentation
"""

# Bound the per-item member-doc map so a large family/cluster cannot bloat the
# compose prompt; the primary path's full grounding is always rendered.
_MAX_MEMBER_DOCS = 8

# Truncate each terse member clause so the grounding stays compact.
_MEMBER_DOC_MAX_CHARS = 240

_IDS_CONTEXT_QUERY = """
MATCH (ids:IDS {id: $ids_name})
OPTIONAL MATCH (child:IMASNode)-[:IN_IDS]->(ids)
WHERE child.id STARTS WITH $ids_prefix
  AND child.data_type IN ['STRUCTURE', 'STRUCT_ARRAY', 'FLT_1D']
  AND size([x IN split(child.id, '/') WHERE true]) = 2
WITH ids,
     collect(DISTINCT {
         name: child.id, description: child.description, data_type: child.data_type
     })[0..10] AS top_sections
RETURN ids.description AS ids_description,
       ids.documentation AS ids_documentation,
       top_sections
"""


def _enrich_ids_context(ids_name: str) -> dict | None:
    """Fetch IDS-level context for batch header.

    Returns dict with ids_description, ids_documentation, top_sections,
    or None if IDS not found.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                _IDS_CONTEXT_QUERY,
                ids_name=ids_name,
                ids_prefix=f"{ids_name}/",
            )
        )
        if not rows:
            return None
        row = rows[0]
        sections = row.get("top_sections") or []
        valid = [s for s in sections if s.get("name")]
        return {
            "ids_description": row.get("ids_description") or "",
            "ids_documentation": row.get("ids_documentation") or "",
            "top_sections": valid,
        }


def _hybrid_search_neighbours_batch(
    gc: Any,
    items: list[tuple[str, str | None, str | None]],
    *,
    max_results: int = 15,
    search_k: int = 10,
) -> list[list[dict]]:
    """Batch-embed then search neighbours for multiple (path, desc, domain) items.

    Replaces the per-item N+1 embed pattern: collects all unique
    description-query texts, embeds them in **one** remote round-trip
    via :func:`embed_query_texts`, then fans out ``hybrid_dd_search``
    calls with pre-computed embeddings.

    Path-text queries (``"equilibrium/time_slice/..."``-style) don't
    need embedding — ``hybrid_dd_search`` detects them and uses
    text-only mode automatically.

    Args:
        gc: Active graph client.
        items: List of ``(path, description, physics_domain)`` tuples.
        max_results: Cap per-item neighbour count.
        search_k: ``k`` passed to ``hybrid_dd_search``.

    Returns:
        List of neighbour-lists in the same order as *items*.
    """
    from imas_codex.embeddings.description import embed_query_texts
    from imas_codex.graph.dd_search import hybrid_dd_search

    if not items:
        return []

    # ── 1. Collect unique description queries needing embedding ─────
    text_set: dict[str, None] = {}  # ordered-set via dict
    item_desc_queries: list[str] = []  # per-item desc query text
    for _path, description, _domain in items:
        desc_query = (description or "")[:200].strip()
        item_desc_queries.append(desc_query)
        if desc_query:
            text_set.setdefault(desc_query, None)

    unique_texts = list(text_set)

    # ── 2. Single batch embed call ──────────────────────────────────
    embed_cache: dict[str, list[float]] = {}
    if unique_texts:
        try:
            embeddings = embed_query_texts(unique_texts)
            # Map by index — handle length mismatch gracefully (server may
            # deduplicate or batch differently)
            for i, emb in enumerate(embeddings):
                if i < len(unique_texts):
                    embed_cache[unique_texts[i]] = emb
        except Exception:
            logger.warning(
                "Batch query embedding failed; falling back to text-only search",
                exc_info=True,
            )

    # ── 3. Per-item hybrid search (with pre-computed embeddings) ────
    all_results: list[list[dict]] = []
    # Collect all hit paths across items for a single batch SN resolution
    per_item_hits: list[dict[str, Any]] = []  # path → SearchHit per item

    for idx, (path, _desc, physics_domain) in enumerate(items):
        item_hits: dict[str, Any] = {}  # path → SearchHit (dedup)

        # Query 1: description-based
        desc_query = item_desc_queries[idx]
        if desc_query:
            pre_emb = embed_cache.get(desc_query)
            try:
                hits = hybrid_dd_search(
                    gc,
                    desc_query,
                    node_category="quantity",
                    physics_domain=physics_domain,
                    k=search_k,
                    embedding=pre_emb,
                )
                for h in hits:
                    if h.path != path:
                        item_hits[h.path] = h
            except Exception:
                logger.debug(
                    "Hybrid search (description) failed for %s",
                    path,
                    exc_info=True,
                )

        # Query 2: path-text (no embedding needed — text-only inside)
        try:
            hits = hybrid_dd_search(
                gc,
                path,
                node_category="quantity",
                k=search_k,
            )
            for h in hits:
                if h.path != path and h.path not in item_hits:
                    item_hits[h.path] = h
        except Exception:
            logger.debug("Hybrid search (path) failed for %s", path, exc_info=True)

        per_item_hits.append(item_hits)

    # ── 4. Single batch HAS_STANDARD_NAME resolution ───────────────
    # Gather all unique hit paths across all items
    all_hit_paths: set[str] = set()
    for item_hits in per_item_hits:
        all_hit_paths.update(item_hits)

    sn_map: dict[str, str | None] = {}
    if all_hit_paths:
        try:
            rows = gc.query(
                """
                UNWIND $paths AS pid
                MATCH (n:IMASNode {id: pid})
                OPTIONAL MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                RETURN n.id AS path, sn.id AS sn_id
                """,
                paths=list(all_hit_paths),
            )
            for r in rows or []:
                sn_map[r["path"]] = r.get("sn_id")
        except Exception:
            logger.debug("HAS_STANDARD_NAME batch pre-resolution failed", exc_info=True)

    # ── 5. Build per-item result dicts ──────────────────────────────
    for idx, (_path, _desc, _domain) in enumerate(items):
        item_hits = per_item_hits[idx]

        if not item_hits:
            all_results.append([])
            continue

        sorted_hits = sorted(item_hits.values(), key=lambda h: h.score, reverse=True)[
            :max_results
        ]

        neighbours: list[dict] = []
        for h in sorted_hits:
            sn_id = sn_map.get(h.path)
            tag = f"name:{sn_id}" if sn_id else f"dd:{h.path}"
            # Rich-first: prefer the LLM-enriched description over terse source
            # documentation (matches the DD-enrichment coalesce precedence).
            doc = (h.description or h.documentation or "")[:300]
            neighbours.append(
                {
                    "tag": tag,
                    "path": h.path,
                    "ids": h.ids_name,
                    "unit": h.units or "",
                    "physics_domain": h.physics_domain or "",
                    "doc_short": doc,
                    "cocos_label": h.cocos_transformation_type or "",
                    "lifecycle": h.lifecycle_status or "",
                    "node_type": h.node_type or "",
                    "score": float(h.score) if h.score is not None else None,
                }
            )
        all_results.append(neighbours)

    return all_results


def _hybrid_search_neighbours(
    gc: Any,
    path: str,
    description: str | None = None,
    physics_domain: str | None = None,
    max_results: int = 15,
    search_k: int = 10,
) -> list[dict]:
    """Single-item shim around :func:`_hybrid_search_neighbours_batch`.

    .. deprecated::
        Prefer :func:`_hybrid_search_neighbours_batch` for multi-item
        workloads.  This shim exists for any un-migrated callers.
    """
    results = _hybrid_search_neighbours_batch(
        gc,
        [(path, description, physics_domain)],
        max_results=max_results,
        search_k=search_k,
    )
    return results[0] if results else []


# Cap for graph-relationship neighbour injection (per path).
_RELATED_MAX_RESULTS = 5


# Compose retry: on grammar/validation failure, retry with expanded context.
# Values resolved from settings accessors; module-level constants kept for
# backwards compatibility with any direct importers.
def _retry_attempts() -> int:
    from imas_codex.settings import get_sn_retry_attempts

    return get_sn_retry_attempts()


def _retry_k_expansion() -> int:
    from imas_codex.settings import get_sn_retry_k_expansion

    return get_sn_retry_k_expansion()


def _mark_vocab_gap_sources(
    batch: list[dict],
    error_detail: str,
    source_kind: str,
) -> None:
    """Mark sources after a vocab-gap validation failure.

    When batch_size == 1 the offending source is known, so it is marked
    ``vocab_gap_compose`` (terminal — won't be reclaimed).  When
    batch_size > 1 we cannot identify which source triggered the failure,
    so all sources are marked ``failed`` with a retry-friendly reason —
    they will be reclaimed on the next ``sn run`` and retried in a
    different (possibly smaller) batch.
    """
    try:
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import mark_source_skipped

        single_source = len(batch) == 1
        reason = "vocab_gap_compose" if single_source else "vocab_gap_batch"

        with GraphClient() as gc:
            for item in batch:
                sid = item.get("source_id") or item.get("id", "")
                if not sid:
                    continue
                if single_source:
                    mark_source_skipped(
                        gc,
                        sid,
                        reason=reason,
                        detail=error_detail[:300],
                        source_type=source_kind,
                    )
                else:
                    # Reset to extracted so the source is reclaimed next run
                    sns_id = f"{source_kind}:{sid}"
                    gc.query(
                        """
                        MATCH (sns:StandardNameSource {id: $sns_id})
                        SET sns.status = 'extracted',
                            sns.claimed_at = null,
                            sns.claim_token = null,
                            sns.skip_reason = $reason,
                            sns.skip_reason_detail = $detail
                        """,
                        sns_id=sns_id,
                        reason=reason,
                        detail=error_detail[:300],
                    )
    except Exception as exc:
        logger.debug("Failed to mark sources after vocab gap: %s", exc)


def _related_path_neighbours(
    gc: Any,
    path: str,
    *,
    max_results: int = _RELATED_MAX_RESULTS,
) -> list[dict]:
    """Fetch explicit graph-relationship neighbours for a DD path.

    Calls :func:`related_dd_search` to discover paths related via
    cluster membership, shared coordinates, matching units, identifier
    schemas, or COCOS transformation type.  Returns a compact list of
    dicts suitable for Jinja template injection.
    """
    from imas_codex.graph.dd_search import related_dd_search

    try:
        result = related_dd_search(
            gc,
            path,
            relationship_types="all",
            max_results=max_results,
        )
    except Exception:
        logger.debug("related_dd_search failed for %s", path, exc_info=True)
        return []

    if not result.hits:
        return []

    neighbours: list[dict] = []
    for hit in result.hits:
        entry: dict[str, Any] = {
            "path": hit.path,
            "ids": hit.ids,
            "relationship_type": hit.relationship_type,
            "via": hit.via,
        }
        if hit.doc:
            entry["doc"] = hit.doc
        if hit.physics_domain:
            entry["physics_domain"] = hit.physics_domain
        neighbours.append(entry)

    return neighbours


def _enrich_batch_items(items: list[dict]) -> None:
    """Enrich batch items with rich DD context from the graph.

    Fetches coordinate specs, COCOS info, identifier schemas, sibling
    fields, cross-IDS cluster siblings, hybrid-search neighbours, and
    version history for each item. Modifies items in-place.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for item in items:
            path = item.get("path")
            if not path:
                continue

            # Value-provenance collapse: a measured/reconstructed/reference facet
            # is the SAME quantity as its base — ground on the base quantity (rich
            # description) so the estimator facets compose ONE name, and flag the
            # item so the prompt directs the composer to name the base quantity and
            # NOT encode the estimator (collapse provenance, de-conflate physics).
            # The estimator itself is recorded on StandardNameSource.provenance.
            prov_term, base_path = detect_value_provenance(path)
            grounding_path = path
            if prov_term:
                item["value_provenance"] = prov_term
                item["provenance_base_path"] = base_path
                grounding_path = base_path

            rows = list(gc.query(_DD_CONTEXT_QUERY, path=grounding_path))
            if not rows:
                continue

            row = rows[0]

            # Grounding text: prefer the rich LLM-enriched node description over
            # the terse source documentation (rich-first, terse-fallback — the
            # same precedence the DD-enrichment pipeline uses). Deterministic
            # parent sources carry a placeholder StandardNameSource.description
            # ("… pending LLM enrichment") that SHADOWS the rich node
            # description which already exists on the IMASNode; without this
            # backfill the composer never sees the physical qualifiers the
            # source states (e.g. "coolant", "neutron", "maximum") and may drop
            # them. Backfill the item's description/documentation so the compose
            # prompt's grounding lines render the rich text for every item.
            node_desc = (row.get("node_description") or "").strip()
            node_doc = (row.get("node_documentation") or "").strip()
            rich = node_desc or node_doc
            if rich:
                cur = (item.get("description") or "").strip()
                # For a provenance facet the grounding was redirected to the base
                # quantity, so its terse facet description ("Measured value") must
                # be REPLACED — not just backfilled — with the base description.
                if (
                    not cur
                    or cur == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
                    or prov_term
                ):
                    item["description"] = rich
                # Surface the terse, XML-backed DD documentation as a DISTINCT
                # grounding line — NOT a copy of the rich description. Keeping
                # them separate lets the compose template's
                # ``documentation != description`` guard fire, so the model sees
                # BOTH the rich enriched description AND the authoritative DD
                # clause. Fall back to node_desc only when no terse doc exists
                # (avoids an empty line). Previously this copied node_desc into
                # documentation, making the two equal and silently suppressing
                # the source-documentation line entirely.
                item["documentation"] = node_doc or node_desc

            # Ancestor lineage context: a terse value leaf ("Deuterium (D).")
            # hides the meaningful physics + evaluation locus that lives on a
            # parent quantity node ("Ion toroidal rotation velocity … at the
            # pedestal top"). Surface the ancestor lineage so the composer sees
            # the parent description IN ADDITION to the leaf — this is what lets
            # a name resolve the correct locus (e.g. pedestal_top, not the bare
            # DD path segment pedestal). Prefer rich LLM-enriched descriptions;
            # fall back to terse DD documentation; nearest ancestor first; cap
            # the block so a deep path cannot bloat the prompt.
            if not item.get("ancestor_context"):
                anc_rows = list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=grounding_path))
                lineage: list[dict[str, str]] = []
                seen_text: set[str] = set()
                for a in anc_rows:
                    a_desc = (a.get("description") or "").strip()
                    a_doc = (a.get("documentation") or "").strip()
                    text = a_desc or a_doc
                    if not text or text in seen_text:
                        continue
                    seen_text.add(text)
                    lineage.append({"path": a.get("path") or "", "text": text})
                    if len(lineage) >= 4:
                        break
                if lineage:
                    item["ancestor_context"] = lineage

            # Propagate the authoritative DD unit from HAS_UNIT into
            # the batch item so compose_batch's unit-safety skip and the
            # downstream persist see the real DD unit (Pa, m, m^-2.W, …)
            # rather than None.  StandardNameSource does not (yet) carry a
            # ``unit`` property in the schema, so we re-derive it from the
            # IMASNode at enrich time.  When HAS_UNIT is absent AND the
            # data type is numeric, set unit to "1" (ISN dimensionless
            # convention) — safety factor q, beta, mode numbers, etc.
            if not item.get("unit"):
                unit_from_rel = row.get("unit_from_rel")
                if unit_from_rel:
                    item["unit"] = unit_from_rel
                else:
                    # Dimensionless: numeric DD paths with no HAS_UNIT
                    # relationship are genuinely dimensionless (q, beta,
                    # mode numbers, efficiencies, fractions, etc.).
                    data_type = row.get("data_type", "")
                    _NUMERIC_PREFIXES = (
                        "FLT_",
                        "INT_",
                        "CPX_",
                    )
                    if data_type and any(
                        data_type.startswith(p) for p in _NUMERIC_PREFIXES
                    ):
                        item["unit"] = "1"

            # Apply unit overrides AFTER re-injecting the DD unit.
            # The override engine corrects upstream DD defects (e.g.,
            # unit vectors tagged 'm' instead of dimensionless '1').
            # Without this, the BUG 9 re-injection above would bypass
            # overrides that were applied during extraction.
            if item.get("unit") and path:
                from imas_codex.standard_names.unit_overrides import resolve_unit

                resolved, meta = resolve_unit(path, item["unit"])
                if meta and meta.get("rule") == "override":
                    item["unit"] = resolved
                elif meta and meta.get("rule") == "skip":
                    item["unit"] = None  # will be caught by compose safety

            # Coordinate context
            coords = []
            for key in ("coordinate1", "coordinate2", "coordinate3"):
                val = row.get(key)
                if val:
                    coords.append(val)
            if coords:
                item["coordinate_paths"] = coords

            timebase = row.get("timebase")
            if timebase:
                item["timebase"] = timebase

            # COCOS
            cocos_label = row.get("cocos_label")
            cocos_expr = row.get("cocos_expression")
            if cocos_label:
                item["cocos_label"] = cocos_label
            if cocos_expr:
                item["cocos_expression"] = cocos_expr

            # Lifecycle
            lifecycle = row.get("lifecycle_status")
            if lifecycle and lifecycle != "active":
                item["lifecycle_status"] = lifecycle

            # Parent description (complements parent_path from extraction)
            parent_desc = row.get("parent_description")
            if parent_desc and not item.get("parent_description"):
                item["parent_description"] = parent_desc

            # Identifier schema
            ident_name = row.get("identifier_schema_name")
            if ident_name:
                item["identifier_schema"] = ident_name
                ident_doc = row.get("identifier_schema_doc")
                if ident_doc:
                    item["identifier_schema_doc"] = ident_doc

                # Parse identifier enum values from JSON-encoded options
                raw_options = row.get("identifier_options")
                if raw_options:
                    try:
                        parsed = (
                            json.loads(raw_options)
                            if isinstance(raw_options, str)
                            else raw_options
                        )
                        if isinstance(parsed, list):
                            item["identifier_values"] = [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                    "description": opt.get("description", ""),
                                }
                                for opt in parsed
                                if opt.get("name")
                            ]
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Sibling fields (same parent, different leaf paths)
            siblings = row.get("sibling_fields") or []
            if siblings and isinstance(siblings, list):
                valid = [s for s in siblings if s.get("path")]
                if valid:
                    item["sibling_fields"] = valid

            # Cross-IDS cluster siblings
            cross_rows = list(gc.query(_CROSS_IDS_QUERY, path=path))
            if cross_rows:
                clusters = []
                cross_ids_paths = []
                for cr in cross_rows:
                    label = cr.get("cluster_label")
                    members = cr.get("member_paths") or []
                    if label and members:
                        clusters.append(
                            {
                                "label": label,
                                "description": cr.get("cluster_description") or "",
                                "scope": cr.get("cluster_scope") or "",
                                "members": members,
                            }
                        )
                        cross_ids_paths.extend(members)
                if clusters:
                    item["clusters"] = clusters
                if cross_ids_paths:
                    # Deduplicate
                    seen = set()
                    unique = []
                    for p in cross_ids_paths:
                        if p not in seen:
                            seen.add(p)
                            unique.append(p)
                    item["cross_ids_paths"] = unique[:8]

            # Version history (COCOS/sign changes are most important)
            version_rows = list(gc.query(_VERSION_HISTORY_QUERY, path=path))
            if version_rows:
                valid_changes = []
                for vr in version_rows:
                    change_id = vr.get("change_id") or ""
                    change_type = vr.get("change_type") or ""
                    # Version is encoded in the ID: path:change_type:version
                    parts = change_id.rsplit(":", 1)
                    version = parts[-1] if len(parts) >= 2 else ""
                    if version and change_type:
                        valid_changes.append(
                            {"version": version, "change_type": change_type}
                        )
                if valid_changes:
                    item["version_history"] = valid_changes

            # Hybrid-search neighbours are batched outside the per-item loop
            # (see below). Skip per-item call here.

            # Graph-relationship neighbours (cluster, coordinate, unit,
            # identifier, COCOS — explicit graph edges, not vector search).
            related = _related_path_neighbours(gc, path)
            if related:
                item["related_neighbours"] = related

            # Error companion fields (uncertainty: _error_upper/lower/index)
            error_rows = list(gc.query(_ERROR_FIELDS_QUERY, path=path))
            if error_rows:
                error_fields = [
                    ef["error_path"] for ef in error_rows if ef.get("error_path")
                ]
                if error_fields:
                    item["error_fields"] = error_fields

        # ── Batch hybrid-search neighbours (single embed round-trip) ───
        batch_tuples: list[tuple[str, str | None, str | None]] = []
        batch_indices: list[int] = []  # index into items
        for idx, item in enumerate(items):
            path = item.get("path")
            if not path:
                continue
            batch_tuples.append(
                (path, item.get("description"), item.get("physics_domain"))
            )
            batch_indices.append(idx)

        if batch_tuples:
            hybrid_results = _hybrid_search_neighbours_batch(gc, batch_tuples)
            for bi, hr in zip(batch_indices, hybrid_results, strict=True):
                if hr:
                    items[bi]["hybrid_neighbours"] = hr

        # ── Batch per-member DD documentation (grounding on covered leaves) ──
        # The per-item loop above grounds only the primary ``path``. A name can
        # span N sibling leaves (family_siblings, cross-IDS equivalents) that
        # render today as bare path strings; without their DD clauses the name
        # is grounded on one leaf. Collect every member path across the batch,
        # fetch terse DD documentation in a SINGLE UNWIND, then attach the
        # per-item subset as ``dd_paths_docs`` {path: terse_doc}. Terse and
        # bounded — distinct from the rich primary description.
        member_sets: list[set[str]] = []
        all_members: set[str] = set()
        for item in items:
            primary = item.get("path")
            members: set[str] = set()
            for field in ("family_siblings", "cross_ids_paths"):
                vals = item.get(field) or []
                if isinstance(vals, list):
                    members.update(v for v in vals if v and v != primary)
            bounded = set(sorted(members)[:_MAX_MEMBER_DOCS])
            member_sets.append(bounded)
            all_members.update(bounded)

        if all_members:
            doc_rows = list(gc.query(_MEMBER_DOC_QUERY, paths=sorted(all_members)))
            member_docs: dict[str, str] = {}
            for r in doc_rows:
                p = r.get("path")
                doc = (r.get("documentation") or "").strip()
                if p and doc:
                    if len(doc) > _MEMBER_DOC_MAX_CHARS:
                        doc = doc[:_MEMBER_DOC_MAX_CHARS].rstrip() + "…"
                    member_docs[p] = doc
            for item, members in zip(items, member_sets, strict=True):
                docs = {p: member_docs[p] for p in members if p in member_docs}
                if docs:
                    item["dd_paths_docs"] = docs


_AXIS_LEAVES = frozenset({"x", "y", "z", "r", "phi"})

# Words that mark a locus token as a concrete hardware device (as opposed to a
# spatial feature such as ``magnetic_axis`` / ``plasma_boundary``). Only when
# the name's locus is one of these do we apply the zero-overlap rejection — a
# spatial-feature locus need not appear in the source path.
_HARDWARE_LOCUS_WORDS = frozenset(
    {
        "coil",
        "probe",
        "sensor",
        "gauge",
        "camera",
        "detector",
        "launcher",
        "mirror",
        "antenna",
        "loop",
        "bolometer",
        "injector",
        "strap",
        "electrode",
        "magnet",
        "valve",
        "thermocouple",
        "interferometer",
        "polarimeter",
        "reflectometer",
        "spectrometer",
        "waveguide",
        "calorimeter",
        "manometer",
        "cryopump",
    }
)


def _vector_fields_conflict(path_a: str, path_b: str) -> bool:
    """True when two paths are the SAME axis leaf of DIFFERENT vector fields
    of ONE DD device node.

    e.g. ``.../camera/direction/z`` vs ``.../camera/up/z`` — same leaf ``z``,
    differing vector-field parent (``direction`` vs ``up``), common device
    grandparent (``.../camera``). Derived from path structure, not a hardcoded
    device path.
    """
    sa = path_a.split("/")
    sb = path_b.split("/")
    if len(sa) < 3 or len(sb) < 3:
        return False
    if sa[-1] != sb[-1] or sa[-1] not in _AXIS_LEAVES:
        return False  # not the same axis leaf
    if sa[-2] == sb[-2]:
        return False  # same vector field — legitimate sibling components
    return sa[:-2] == sb[:-2]  # common device node up to the vector-field segment


def _content_tokens(text: str) -> set[str]:
    """Lower-cased content tokens with a simple plural stem (drop trailing s)."""
    raw = _re.split(r"[^a-z0-9]+", text.lower())
    return {(t[:-1] if len(t) > 3 and t.endswith("s") else t) for t in raw if t}


def _name_locus_token(sn_name: str) -> str | None:
    """Extract the trailing ``_of_<locus>`` / ``_at_<locus>`` locus token, or None."""
    of_i = sn_name.rfind("_of_")
    at_i = sn_name.rfind("_at_")
    i = max(of_i, at_i)
    if i < 0:
        return None
    locus = sn_name[i + 4 :]
    # A process attribution follows the locus — trim it off.
    due = locus.find("_due_to_")
    if due >= 0:
        locus = locus[:due]
    return locus or None


def _is_attachment_consistent(
    source_id: str, sn_name: str, existing_sources: Sequence[str] = ()
) -> tuple[bool, str]:
    """Reject attachments where the DD path tense disagrees with the SN tense.

    E.g. ``change_in_electron_density`` may not be attached to
    ``core_profiles/.../density`` (a base quantity, not a change). Symmetric:
    a base-quantity SN may not absorb an ``instant_changes`` path.

    ``existing_sources`` are the DD paths already carried by ``sn_name`` (bare,
    ``dd:``-prefix stripped) — used by the distinct-vector guard to reject a
    second vector field of one device node collapsing onto one scalar name.
    """
    change_prefixes = (
        "change_in_",
        "tendency_of_",
        "rate_of_",
        "rate_of_change_of_",
        "time_derivative_of_",
    )
    change_path_tokens = (
        "instant_changes",
        "/change/",
        "_delta",
        "tendency_",
        "/d_dt/",
        "_dt",
        "/derivatives_1d/",
    )
    sn_is_change = any(sn_name.startswith(p) for p in change_prefixes)
    path_is_change = any(t in source_id for t in change_path_tokens)
    if sn_is_change and not path_is_change:
        return False, (
            f"tense mismatch: SN '{sn_name}' is a change/rate but path "
            f"'{source_id}' is a base quantity"
        )
    if path_is_change and not sn_is_change:
        return False, (
            f"tense mismatch: path '{source_id}' is a change/rate but SN "
            f"'{sn_name}' is a base quantity"
        )

    # State-resolution consistency (R1/R4 rotation finding): the DD resolves
    # species into ionisation/atomic states via ``…/state/…`` sub-structures.
    # A state-resolved path describes ONE state — it must not source a
    # species-level name (the species name is the structural parent, not a
    # synonym), and a species-level path must not source a state name.
    state_subjects = ("ion_state", "ion_charge_state", "neutral_state")
    padded = f"_{sn_name}_"
    sn_is_state = any(f"_{t}_" in padded for t in state_subjects)
    path_is_state = "/state/" in source_id or source_id.endswith("/state")
    if path_is_state and not sn_is_state:
        return False, (
            f"state-resolution mismatch: path '{source_id}' is state-resolved "
            f"but SN '{sn_name}' is species-level"
        )
    if sn_is_state and not path_is_state:
        return False, (
            f"state-resolution mismatch: SN '{sn_name}' is state-resolved "
            f"but path '{source_id}' is species-level"
        )

    # Shape-parameter surface consistency: a shape descriptor
    # (triangularity/elongation/squareness) is defined OF a surface, and the
    # boundary scalar and the flux-surface profile are DISTINCT quantities.
    # A profiles_1d (flux-surface) source must not attach to a
    # ``…_of_plasma_boundary`` name (and vice versa) — that re-merges the
    # de-conflated siblings the compose-time injection separated, and triggers
    # a supersede/refine cascade. A boundary_separatrix source attaching to a
    # ``…_of_plasma_boundary`` name is consistent (the separatrix IS the
    # boundary), since both derive ``plasma_boundary``. Mirrors
    # :func:`_shape_parameter_surface`.
    if any(b in sn_name for b in _SHAPE_PARAMETER_BASES) and any(
        f"_of_{s}" in sn_name for s in ("plasma_boundary", "flux_surface")
    ):
        expected_surface = _shape_parameter_surface(source_id)
        if f"_of_{expected_surface}" not in sn_name:
            return False, (
                f"shape-parameter surface mismatch: path '{source_id}' is "
                f"of_{expected_surface} but SN '{sn_name}' carries a "
                f"different surface"
            )

    # Distinct-vector guard: the target name must not collapse two DIFFERENT
    # vector fields of one DD device node onto one scalar name (e.g. a camera's
    # line-of-sight ``direction`` and its image-up ``up`` vector both landing on
    # ``z_direction_unit_vector``). Derived from path structure.
    for other in existing_sources:
        if other and other != source_id and _vector_fields_conflict(source_id, other):
            return False, (
                f"distinct-vector conflict: path '{source_id}' and already-"
                f"sourced '{other}' are the same axis leaf of DIFFERENT vector "
                f"fields of one device node — SN '{sn_name}' must not carry both"
            )

    # Locus <-> source device compatibility: when the name carries a concrete
    # hardware locus (``_of_<device>``/``_at_<device>``), the source path must
    # share at least one content token with that device. Conservative — only
    # rejects on ZERO overlap when the locus is a recognised hardware token.
    locus = _name_locus_token(sn_name)
    if locus:
        locus_tokens = _content_tokens(locus)
        if locus_tokens & _HARDWARE_LOCUS_WORDS:
            path_tokens = _content_tokens(source_id)
            if not (locus_tokens & path_tokens):
                return False, (
                    f"locus/source device mismatch: SN '{sn_name}' has hardware "
                    f"locus '{locus}' but path '{source_id}' shares no device "
                    f"token with it"
                )

    return True, ""


def _process_attachments_core(
    attachments: list,
    wlog: logging.LoggerAdapter | logging.Logger,
) -> dict[str, int]:
    """Stateless attachment-processing helper used by both compose paths.

    Filters by tense consistency, writes ``HAS_STANDARD_NAME`` edges and
    appends ``source_paths``, and returns ``{"accepted", "rejected"}``
    counts.  Caller is responsible for any state-side stats updates.
    """
    from imas_codex.graph.client import GraphClient

    # Accumulate each target name's accepted sources within this batch so two
    # conflicting attachments — e.g. two different vector fields of one DD
    # device node — cannot both land on one scalar name. (The graph layer is
    # only touched once something is accepted; the seed starts empty.)
    existing_by_sn: dict[str, list[str]] = {}
    rejected: list[tuple[str, str, str]] = []
    accepted: list = []
    for a in attachments:
        existing = existing_by_sn.setdefault(a.standard_name, [])
        ok, reason = _is_attachment_consistent(
            a.source_id, a.standard_name, existing_sources=existing
        )
        if ok:
            accepted.append(a)
            existing.append(a.source_id)
        else:
            rejected.append((a.source_id, a.standard_name, reason))

    for src, sn, why in rejected:
        wlog.warning("Rejected attachment %s → %s: %s", src, sn, why)

    counts = {"accepted": 0, "rejected": len(rejected)}
    if not accepted:
        return counts

    batch = [
        {"source_id": a.source_id, "standard_name": a.standard_name} for a in accepted
    ]

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.standard_name})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                WITH sn, 'dd:' + b.source_id AS uri
                WHERE NOT uri IN coalesce(sn.source_paths, [])
                SET sn.source_paths = coalesce(sn.source_paths, []) + uri
                """,
                batch=batch,
            )

        for a in accepted:
            wlog.info("Attached %s → %s (%s)", a.source_id, a.standard_name, a.reason)

        counts["accepted"] = len(accepted)
    except Exception:
        wlog.warning("Failed to process attachments", exc_info=True)

    return counts


def _process_attachments(
    attachments: list, state: StandardNameBuildState, wlog: logging.LoggerAdapter
) -> None:
    """Attach DD paths to existing standard names without regeneration.

    Linear-path wrapper around :func:`_process_attachments_core` that
    folds counts into the build state's stats dictionary.
    """
    counts = _process_attachments_core(attachments, wlog)
    if counts["accepted"]:
        prev = state.stats.get("attachments", 0)
        state.stats["attachments"] = prev + counts["accepted"]
    if counts["rejected"]:
        state.stats["attachments_rejected"] = (
            state.stats.get("attachments_rejected", 0) + counts["rejected"]
        )


# ---------------------------------------------------------------------------
# StandardNameSource status updaters (Phase 5: incremental tracking)
# ---------------------------------------------------------------------------


def _update_sources_after_compose(
    candidates: list[dict], source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'composed' after successful batch composition.

    Error-sibling candidates (minted deterministically, model=
    ``deterministic:dd_error_modifier``) are excluded — their source
    IMASNodes are not extracted as StandardNameSource, so linking them
    here would produce false-positive "linking gap" warnings.  Their
    provenance is carried on ``StandardName.source_paths``.
    """
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for c in candidates:
        if c.get("model") == "deterministic:dd_error_modifier":
            continue  # error siblings have no StandardNameSource
        source_id = c.get("source_id")
        sn_id = c.get("id")
        if source_id and sn_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{source_id}",
                    "sn_id": sn_id,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'composed',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Compose-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to composed", linked)
    except Exception:
        wlog.warning("Failed to update StandardNameSource status", exc_info=True)


def _update_sources_after_attach(
    attachments: list, source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'attached' status."""
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for a in attachments:
        if a.source_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{a.source_id}",
                    "sn_id": a.standard_name,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'attached',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Attach-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to attached", linked)
    except Exception:
        wlog.warning(
            "Failed to update StandardNameSource attachment status", exc_info=True
        )


def _update_sources_after_vocab_gap(
    vocab_gaps: list[dict],
    source: str,
    wlog: logging.LoggerAdapter,
    *,
    max_attempts: int = 3,
) -> None:
    """Persist the compose-time vocab-gap outcome onto each source.

    A source is retired to ``vocab_gap`` ONLY when it is blocked by a
    genuinely-absent token in a closed grammar segment — the sole actionable
    signal for an ISN vocabulary addition. The blocking ``segment:token`` list
    is recorded in ``skip_reason_detail`` so the source explains itself without
    a join back through the DD path.

    A gap the composer mis-reported — a token that decomposes into existing
    tokens, sits in the wrong slot, is ambiguous, or is a false positive — is
    not a vocabulary deficiency: the source is still nameable and the composer
    erred, so it is kept retryable under the attempt-count cap rather than
    stranded at ``vocab_gap``. Open/pseudo-segment gaps (grammar_ambiguity) are
    ignored entirely.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.segments import (
        is_actionable_gap,
        is_open_segment,
    )

    source_type = "dd" if source == "dd" else "signals"
    gaps_by_source: dict[str, list[dict]] = {}
    skipped_pseudo = 0
    for vg in vocab_gaps:
        if is_open_segment(vg.get("segment")):
            skipped_pseudo += 1
            continue
        sid = vg.get("source_id")
        if sid:
            gaps_by_source.setdefault(f"{source_type}:{sid}", []).append(vg)

    if skipped_pseudo:
        wlog.debug(
            "Skipped vocab_gap update for %d pseudo-segment gaps", skipped_pseudo
        )
    if not gaps_by_source:
        return

    retire_rows: list[dict] = []
    retry_rows: list[dict] = []
    for sns_id, gaps in gaps_by_source.items():
        detail = "; ".join(f"{g.get('segment')}:{g.get('token')}" for g in gaps)[:300]
        row = {"sns_id": sns_id, "detail": detail}
        if any(is_actionable_gap(g.get("segment"), g.get("token")) for g in gaps):
            retire_rows.append(row)
        else:
            retry_rows.append(row)

    try:
        with GraphClient() as gc:
            if retire_rows:
                gc.query(
                    """
                    UNWIND $rows AS row
                    MATCH (sns:StandardNameSource {id: row.sns_id})
                    SET sns.status = 'vocab_gap',
                        sns.skip_reason = 'vocab_gap',
                        sns.skip_reason_detail = row.detail,
                        sns.claimed_at = null,
                        sns.claim_token = null
                    """,
                    rows=retire_rows,
                )
            if retry_rows:
                # Composer mis-report, not a vocabulary gap — increment the
                # attempt count and return the source to the compose pool;
                # a deterministic failure still terminates at 'failed' once
                # the cap is hit, so it is bounded, never stranded.
                gc.query(
                    """
                    UNWIND $rows AS row
                    MATCH (sns:StandardNameSource {id: row.sns_id})
                    SET sns.attempt_count = coalesce(sns.attempt_count, 0) + 1,
                        sns.skip_reason = 'vocab_gap_nonactionable',
                        sns.skip_reason_detail = row.detail,
                        sns.claimed_at = null,
                        sns.claim_token = null,
                        sns.status = CASE
                            WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                            THEN 'failed' ELSE 'extracted' END,
                        sns.failed_at = CASE
                            WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                            THEN datetime() ELSE sns.failed_at END
                    """,
                    rows=retry_rows,
                    max_attempts=max_attempts,
                )
        wlog.debug(
            "vocab_gap update: %d retired, %d kept retryable",
            len(retire_rows),
            len(retry_rows),
        )
    except Exception:
        wlog.warning("Failed to persist vocab_gap source outcome", exc_info=True)


def _update_sources_after_skip(
    skipped_ids: list[str],
    source: str,
    wlog: logging.LoggerAdapter | logging.Logger,
) -> None:
    """Clear claims and mark StandardNameSource nodes ``status='skipped'``.

    The compose LLM may decide a source is not a physics quantity and emit
    its ``source_id`` in ``result.skipped``.  Without this transition the
    source remains ``claimed`` forever and gets re-processed every run.
    """
    from imas_codex.graph.client import GraphClient

    if not skipped_ids:
        return

    source_type = "dd" if source == "dd" else "signals"
    sns_ids = [f"{source_type}:{sid}" for sid in skipped_ids if sid]
    if not sns_ids:
        return

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS sns_id
                MATCH (sns:StandardNameSource {id: sns_id})
                SET sns.status        = 'skipped',
                    sns.claim_token   = null,
                    sns.claimed_at    = null,
                    sns.skipped_at    = datetime()
                """,
                ids=sns_ids,
            )
        wlog.debug("Marked %d StandardNameSource nodes as skipped", len(sns_ids))
    except Exception:
        wlog.warning("Failed to update StandardNameSource skip status", exc_info=True)


def _search_reference_exemplars(
    items: list[dict],
    domains: list[str],
    *,
    k: int = 5,
) -> list[dict]:
    """Synthesise a query from batch items and return reference SN exemplars.

    Builds a query string from up to three item descriptions, calls
    :func:`search_standard_names_with_documentation`, and excludes any SN whose
    ``id`` matches an item already present in the batch (when items carry an
    ``existing_name``).  *domains* filters via embedding-quality semantics
    only (the search itself filters by ``validation_status='valid'``).

    Returns an empty list if no descriptions are available or the search
    backend is unavailable.
    """
    from imas_codex.standard_names.search import (
        search_standard_names_with_documentation,
    )

    desc_snippets = [
        item.get("description", "") for item in items[:3] if item.get("description")
    ]
    if not desc_snippets:
        return []
    synth_query = "; ".join(desc_snippets)

    # Real SN ids of items already present in the batch (cluster-aware
    # nearby retrieval injects these); avoid feeding the LLM duplicates.
    exclude_ids = sorted(
        {item.get("existing_name") for item in items if item.get("existing_name")}
    )

    try:
        return search_standard_names_with_documentation(
            synth_query, k=k, exclude_ids=exclude_ids or None
        )
    except Exception:
        logger.debug("Reference exemplar search failed", exc_info=True)
        return []


# Opus model for L7 borderline revision pass
_L7_REVISION_MODEL = DEFAULT_ESCALATION_MODEL
_L7_MIN_REMAINING_BUDGET = 0.50  # Skip L7 if remaining budget < this


async def _grammar_retry(
    original_name: str,
    parse_error: str,
    model: str,
    acall_fn,
    *,
    reasoning_effort: str | None = None,
) -> tuple[str | None, float, int, int]:
    """L6: Single grammar-failure re-prompt.

    Asks the LLM to revise a name that failed grammar round-trip,
    providing the parse error and a grammar cheat-sheet fragment.

    Returns ``(revised_name | None, cost_usd, tokens_in, tokens_out)``.
    """
    from pydantic import BaseModel, Field

    class GrammarRetryResponse(BaseModel):
        revised_name: str = Field(
            description="The revised standard name that passes grammar parsing"
        )
        explanation: str = Field(description="Brief explanation of the fix")

    retry_prompt = (
        f"The standard name `{original_name}` failed grammar parsing with error:\n"
        f"  {parse_error}\n\n"
        "Revise ONLY the name to pass the grammar round-trip. Rules:\n"
        "- Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]\n"
        "- ALL segments are CLOSED including physical_base (80 registered tokens).\n"
        "  Only use tokens from the registries. If no token fits, emit a vocab_gap.\n"
        "- Closed-vocabulary tokens (e.g. toroidal, parallel, thermal,\n"
        "  e_cross_b_drift, normalized, fast_ion, volume_averaged) MUST be\n"
        "  placed in their closed segment slot. Never absorb them into\n"
        "  physical_base.\n"
        "- No abbreviations, no provenance verbs, no unit suffixes\n"
        "- Return the MINIMAL fix — keep the name as close to the original as possible.\n"
    )

    try:
        llm_out = await acall_fn(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            response_model=GrammarRetryResponse,
            service="standard-names",
            reasoning_effort=reasoning_effort,
        )
        result, _cost, _tokens = llm_out
        return (
            result.revised_name if result else None,
            float(_cost or 0.0),
            getattr(llm_out, "input_tokens", 0) or 0,
            getattr(llm_out, "output_tokens", 0) or 0,
        )
    except Exception:
        return None, 0.0, 0, 0


async def _self_refine_candidate(
    name: str,
    description: str,
    segments: dict[str, Any] | None,
    dd_context: dict[str, Any] | None,
    model: str,
    acall_fn,
    *,
    reasoning_effort: str | None = None,
) -> tuple[str, str]:
    """Free local self-improvement pass over a freshly-composed candidate.

    Runs AFTER compose + grammar normalization and BEFORE persist (and so
    before the paid review quorum). The *local* compose model critiques its
    own ``name`` + ``description`` against deterministic grammar diagnostics
    and the compose rubric and may emit an improved candidate. The contract
    is **improve-or-no-op**: it never rejects.

    Safety: the suggested name is re-validated via the grammar round-trip
    (``parse_standard_name`` → ``compose_standard_name``). If the rewrite
    fails grammar parsing, or loses canonical order (round-trips to a
    different string), or fails the well-formedness gate, the **original**
    ``(name, description)`` is returned unchanged. The local model is free
    (``get_model("sn-compose")`` served on the local GPU), so this charges
    ``$0``; callers do not account any cost.

    Returns ``(name, description)`` — improved when the rewrite is safe and
    the model elected to change it, otherwise the originals unchanged.
    """
    from pydantic import BaseModel, Field

    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    class SelfRefineSegments(BaseModel):
        base_token: str = Field(description="Registered physical_base/geometry token")
        base_kind: str = Field(default="quantity", description="quantity | geometry")
        projection_axis: str | None = None
        # projection_shape is derived from base_kind (see GrammarSegments) — not
        # an LLM input.
        qualifiers: list[str] = Field(default_factory=list)
        locus_token: str | None = None
        locus_relation: str | None = None
        locus_type: str | None = None
        process_token: str | None = None
        operator_token: str | None = None
        operator_kind: str | None = None

    class SelfRefineResponse(BaseModel):
        changed: bool = Field(
            description="True only if name or description was improved"
        )
        name: str = Field(description="Improved (or unchanged) standard name")
        description: str = Field(
            default="", description="Improved (or unchanged) ≤120 char description"
        )
        segments: SelfRefineSegments | None = Field(
            default=None,
            description="IR grammar segments of the improved name",
        )

    # Build the deterministic diagnostics shown to the model. The name was
    # already normalized to canonical form by the caller, so the headline is
    # "round-trip OK"; we surface a couple of cheap structural observations.
    diagnostics: list[str] = []
    try:
        from imas_standard_names.grammar import (
            compose_standard_name,
            parse_standard_name,
        )

        _parsed = parse_standard_name(name)
        _canon = compose_standard_name(_parsed)
        if _canon == name:
            diagnostics.append(
                "Round-trip: OK — name parses and re-composes to canonical form."
            )
        else:
            diagnostics.append(
                f"Round-trip: canonical form is `{_canon}` (composed `{name}`)."
            )
    except Exception:
        diagnostics.append(
            "Round-trip: WARNING — name did not parse cleanly; prefer the closest "
            "grammatical alternative."
        )

    try:
        context = await asyncio.to_thread(build_compose_context)
    except Exception:
        context = {}

    system_prompt = render_prompt("sn/self_refine_system", context)
    user_prompt = render_prompt(
        "sn/self_refine_user",
        {
            **context,
            "name": name,
            "description": description or "",
            "segments": segments or {},
            "dd_context": dd_context or {},
            "diagnostics": diagnostics,
        },
    )

    try:
        llm_out = await acall_fn(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=SelfRefineResponse,
            service="standard-names",
            reasoning_effort=reasoning_effort,
        )
        result, _cost, _tokens = llm_out
    except Exception:
        logger.debug("Self-refine LLM call failed for %r — keeping original", name)
        return name, description

    if result is None or not result.changed:
        return name, description

    new_name = normalize_spelling((result.name or "").strip())
    if not new_name or new_name == name:
        # No-op rename; still adopt a tightened description if offered.
        new_desc = normalize_description_text((result.description or "").strip())
        return name, (new_desc or description)

    # --- Re-validate the suggested name before adopting it. ---
    _well_formed, _ = is_well_formed_candidate(new_name)
    if not _well_formed:
        logger.debug(
            "Self-refine suggestion %r not well-formed — keeping original %r",
            new_name,
            name,
        )
        return name, description

    try:
        from imas_standard_names.grammar import (
            compose_standard_name,
            parse_standard_name,
        )

        _parsed = parse_standard_name(new_name)
        _canon = compose_standard_name(_parsed)
        if _canon != new_name:
            # The "improvement" lost canonical order — discard it.
            logger.debug(
                "Self-refine suggestion %r lost canonical order (→ %r) — "
                "keeping original %r",
                new_name,
                _canon,
                name,
            )
            return name, description
    except Exception:
        logger.debug(
            "Self-refine suggestion %r failed grammar round-trip — keeping %r",
            new_name,
            name,
        )
        return name, description

    new_desc = normalize_description_text((result.description or "").strip())
    logger.info("Self-refine improved %r → %r", name, new_name)
    return new_name, (new_desc or description)


async def _opus_revise_candidate(
    candidate: dict,
    domain_vocabulary: str,
    reviewer_themes: list[str],
    acall_fn,
) -> tuple[str | None, float, int, int]:
    """L7: Revision pass for candidates using Opus model.

    Returns ``(revised_name_or_None, cost_usd, tokens_in, tokens_out)``.
    The cost is always returned so callers can account for it even when
    the revision is discarded.
    """
    from pydantic import BaseModel, Field

    class OpusRevisionResponse(BaseModel):
        revised_name: str = Field(description="Improved standard name")
        explanation: str = Field(description="Why this revision is better")

    name = candidate.get("id", "")
    reason = candidate.get("reason", "")
    description = candidate.get("description", "")

    prompt_parts = [
        "A standard name candidate requires revision:",
        f"  Name: `{name}`",
        f"  Description: {description}",
        f"  Reason: {reason}",
        "",
        "Revise the name to be more precise, following ISN grammar rules.",
        "Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]",
        "",
    ]

    if domain_vocabulary:
        prompt_parts.append("Domain vocabulary (prefer these terms):")
        # Include first 10 lines of vocabulary
        for line in domain_vocabulary.split("\n")[:10]:
            prompt_parts.append(f"  {line}")
        prompt_parts.append("")

    if reviewer_themes:
        prompt_parts.append("Reviewer feedback themes to address:")
        for theme in reviewer_themes[:5]:
            prompt_parts.append(f"  - {theme}")
        prompt_parts.append("")

    prompt_parts.append("Return a revised name only if you can do CLEARLY better.")

    try:
        llm_out = await acall_fn(
            model=_L7_REVISION_MODEL,
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            response_model=OpusRevisionResponse,
            service="standard-names",
        )
        result, _cost, _tokens = llm_out
        _ti = getattr(llm_out, "input_tokens", 0) or 0
        _to = getattr(llm_out, "output_tokens", 0) or 0
        if result:
            return result.revised_name, float(_cost or 0.0), _ti, _to
        return None, float(_cost or 0.0), _ti, _to
    except Exception:
        return None, 0.0, 0, 0


async def compose_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split for
    prompt caching.  Runs batches concurrently with semaphore.
    Results stored in ``state.composed``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_compose_worker")

    total_items = sum(len(b.items) for b in state.extracted)

    if state.dry_run:
        wlog.info("Dry run — skipping composition for %d items", total_items)
        state.compose_stats.total = total_items
        state.compose_stats.processed = total_items
        state.stats["compose_skipped"] = True
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    if not state.extracted:
        wlog.info("No batches to compose — skipping")
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    model = state.compose_model or get_model("sn-compose")
    context = build_compose_context()

    # Enrich batch items with rich DD context (coordinate specs, COCOS, siblings,
    # cross-IDS paths, version history)
    def _enrich_all_batches():
        for batch in state.extracted:
            _enrich_batch_items(batch.items)

    await asyncio.to_thread(_enrich_all_batches)
    wlog.info("Enriched batch items with DD context")

    # Propagate COCOS metadata to state (for downstream phases)
    if state.extracted:
        first_batch = state.extracted[0]
        state.dd_version = first_batch.dd_version
        state.cocos_version = first_batch.cocos_version
        state.cocos_params = first_batch.cocos_params

    # Pre-fetch IDS-level context for each unique IDS across batches
    ids_context_cache: dict[str, dict | None] = {}

    def _prefetch_ids_context():
        for batch in state.extracted:
            # Derive IDS names from item paths (first segment)
            ids_names = {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
            for ids_name in ids_names:
                if ids_name not in ids_context_cache:
                    ids_context_cache[ids_name] = _enrich_ids_context(ids_name)

    await asyncio.to_thread(_prefetch_ids_context)
    wlog.info("Fetched IDS context for %d IDS(s)", len(ids_context_cache))

    # Render system prompt once (cached via prompt caching)
    # Inject COCOS context for system prompt (all batches share one DD version)
    if state.extracted:
        first_batch = state.extracted[0]
        if first_batch.cocos_version:
            context["cocos_version"] = first_batch.cocos_version
            context["dd_version"] = first_batch.dd_version
            if first_batch.cocos_params:
                context["cocos_sigma_bp"] = first_batch.cocos_params.get("sigma_bp")
                context["cocos_e_bp"] = first_batch.cocos_params.get("e_bp")
                context["cocos_sigma_r_phi_z"] = first_batch.cocos_params.get(
                    "sigma_r_phi_z"
                )
                context["cocos_sigma_rho_theta_phi"] = first_batch.cocos_params.get(
                    "sigma_rho_theta_phi"
                )

    # --- L1: Domain-vocabulary pre-seeding ---
    # Inject validated domain vocabulary into system prompt context
    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab = ""
    if state.domain_filter:
        domain_vocab = await asyncio.to_thread(
            build_domain_vocabulary_preseed, state.domain_filter
        )
        if domain_vocab:
            wlog.info(
                "L1: Injected domain vocabulary preseed for %s", state.domain_filter
            )
    context["domain_vocabulary"] = domain_vocab

    # --- L4: Reviewer-theme extraction ---
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    if state.domain_filter:
        reviewer_themes = await asyncio.to_thread(
            extract_reviewer_themes, state.domain_filter
        )
        if reviewer_themes:
            wlog.info(
                "L4: Extracted %d reviewer themes for %s",
                len(reviewer_themes),
                state.domain_filter,
            )
    context["reviewer_themes"] = reviewer_themes

    # --- K3: Scored-example injection ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    # Derive physics_domains from domain_filter and batch items
    batch_domains: list[str] = []
    if state.domain_filter:
        batch_domains = [state.domain_filter]
    else:
        # Collect unique domains from all batch items
        _domains = {
            item.get("physics_domain")
            for batch in state.extracted
            for item in batch.items
            if item.get("physics_domain")
        }
        batch_domains = sorted(_domains)

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(gc, physics_domains=batch_domains, axis="name")

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    if compose_scored_examples:
        wlog.info(
            "K3: Loaded %d scored examples for compose (domains=%s)",
            len(compose_scored_examples),
            batch_domains or "all",
        )
    context["compose_scored_examples"] = compose_scored_examples

    # --- NC rules injection ---
    # Load naming-consistency rules from YAML so the system prompt
    # {% include "_nc_rules.md" %} block renders the full rule set.
    from imas_codex.llm.prompt_loader import load_prompt_config

    try:
        _rules_cfg = load_prompt_config("sn_composition_rules")
        context["composition_rules"] = _rules_cfg.get("composition_rules", [])
    except Exception:
        wlog.debug("NC rules YAML not found — skipping injection")
        context["composition_rules"] = []

    system_prompt = render_prompt("sn/generate_name_system", context)

    wlog.info(
        "Composing standard names for %d items in %d batches (model=%s)",
        total_items,
        len(state.extracted),
        model,
    )
    state.compose_stats.total = total_items

    from imas_codex.settings import get_compose_concurrency

    async def _compose_batch_body(
        batch: ExtractionBatch, lease: BudgetLease | None
    ) -> list[dict]:
        # Search for nearby existing names (per-item for better relevance)
        _nearby_seen: set[str] = set()
        nearby: list[dict] = []
        _PER_ITEM_K = 5
        _NEARBY_CAP = 30
        for item in batch.items:
            hint = item.get("description") or item.get("path", "")
            item_results = _search_nearby_names(hint, k=_PER_ITEM_K)
            for nr in item_results:
                nid = nr.get("id", "")
                if nid and nid not in _nearby_seen:
                    _nearby_seen.add(nid)
                    nearby.append(nr)
                    if len(nearby) >= _NEARBY_CAP:
                        break
            if len(nearby) >= _NEARBY_CAP:
                break

        # IDS-level context — collect for each IDS present in batch
        ids_names = sorted(
            {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
        )
        ids_contexts = []
        for iname in ids_names:
            info = ids_context_cache.get(iname)
            if info:
                ids_contexts.append({"ids_name": iname, **info})

        # Pre-render COCOS guidance for items
        if batch.cocos_params:
            from imas_codex.standard_names.context import render_cocos_guidance

            for item in batch.items:
                cocos_label = item.get("cocos_label")
                if cocos_label:
                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, batch.cocos_params
                    )

        # --- Rate-quantity detection ---
        # When DD documentation indicates a rate/time-derivative, inject a
        # hard constraint so the LLM uses tendency_of_/change_in_ prefix
        # and writes a consistent description (prevents instant_change_*
        # names and name/description verb drift).
        import re as _re

        _RATE_DOC_PATTERNS = _re.compile(
            r"\b(instantaneous change|signed change|rate of change"
            r"|time derivative|per unit time|instant change|d/dt"
            r"|tendency of|time-rate)\b",
            _re.IGNORECASE,
        )
        for item in batch.items:
            haystack = " ".join(
                str(item.get(k, "") or "") for k in ("description", "documentation")
            )
            if _RATE_DOC_PATTERNS.search(haystack):
                item["rate_hint"] = True

        # --- L2: Reference SN few-shot retrieval ---
        # Synthesize query from first 3 path descriptions
        reference_exemplars: list[dict] = []
        try:
            from imas_codex.standard_names.search import (
                search_standard_names_with_documentation,
            )

            desc_snippets = [
                item.get("description", "")
                for item in batch.items[:3]
                if item.get("description")
            ]
            if desc_snippets:
                synth_query = "; ".join(desc_snippets)
                # Exclude names already in this batch's candidate IDs
                batch_ids = [
                    item.get("path", "").replace("/", "_") for item in batch.items
                ]
                reference_exemplars = await asyncio.to_thread(
                    search_standard_names_with_documentation,
                    synth_query,
                    k=5,
                    exclude_ids=batch_ids,
                )
        except Exception:
            wlog.debug("L2: Reference exemplar search failed", exc_info=True)

        user_context = {
            "items": batch.items,
            "ids_name": batch.group_key,
            "ids_contexts": ids_contexts,
            "existing_names": sorted(batch.existing_names)[:200],
            "cluster_context": batch.context,
            "nearby_existing_names": nearby,
            "reference_exemplars": reference_exemplars,
            "cocos_version": batch.cocos_version,
            "dd_version": batch.dd_version,
        }
        # Name-only batches (Workstream 2a) render a leaner user prompt
        # that trades per-item cluster siblings / COCOS blocks / sibling
        # fields for a "identify natural sub-groups, then name" directive.
        # System prompt and per-candidate L6/L7 logic are unchanged so
        # prompt caching and grammar safety stay intact.
        prompt_template = (
            "sn/generate_name_dd_names"
            if batch.mode == "names"
            else "sn/generate_name_dd"
        )
        user_prompt = render_prompt(prompt_template, {**context, **user_context})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # --- Delta H: bounded retry loop for failed compositions ---
        # Accumulate full LLMResult fields across retries so the batch's
        # per-candidate cost/token attribution is accurate.  Compose is
        # unified with review/enrich here: every LLM call site extracts
        # the same cost/token fields from LLMResult and writes them to
        # the graph for single-source cost observability.
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0
        _max_retries = _retry_attempts()
        # Token-reuse hits from the FINAL attempt: candidate-new tokens the
        # agent re-emitted after being shown a near-synonym registered token.
        # Carried to the post-loop VocabGap stamp as distinct_confirmed.
        _token_reuse_hits: dict[tuple[str, str], Any] = {}
        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                )
            except (ValueError, Exception) as compose_exc:
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Compose: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        len(batch),
                        _exc_str[:200],
                    )
                    _mark_vocab_gap_sources(batch.items, _exc_str, "dd")
                    return []
                raise
            result, cost, tokens = llm_out
            _total_compose_cost += cost
            _total_tokens_in += getattr(llm_out, "input_tokens", 0) or 0
            _total_tokens_out += getattr(llm_out, "output_tokens", 0) or 0
            _total_cache_read += getattr(llm_out, "cache_read_tokens", 0) or 0
            _total_cache_creation += getattr(llm_out, "cache_creation_tokens", 0) or 0

            # Charge actual LLM cost to budget lease via typed event.
            # charge_event uses soft-charge semantics: the LLM has
            # already been paid for, so spend is always recorded.
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=getattr(llm_out, "cache_read_tokens", 0) or 0,
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=tuple(
                        c.compose_name() for c in (result.candidates if result else [])
                    ),
                    batch_id=batch.group_key,
                    phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                    service="standard-names",
                )
                _charge = lease.charge_event(cost, _event)
                if state.loop_stats is not None:
                    state.loop_stats.processed += max(1, len(result.candidates))
                    state.loop_stats.cost += cost
                    # Stream one item per candidate so the user sees names
                    # rotate through with grammar composition.
                    _items: list[dict[str, Any]] = []
                    for _cand in result.candidates[:50]:
                        _segs = []
                        if _cand.base_token:
                            _segs.append(f"base={_cand.base_token}")
                        for _q in _cand.qualifiers:
                            _segs.append(f"qualifier={_q}")
                        if _cand.projection_axis:
                            _segs.append(f"projection={_cand.projection_axis}")
                        if _cand.process_token:
                            _segs.append(f"process={_cand.process_token}")
                        if _cand.operator_token:
                            _segs.append(f"operator={_cand.operator_token}")
                        _desc = (
                            "  ".join(_segs) if _segs else ((_cand.reason or "")[:80])
                        )
                        try:
                            _composed = _cand.compose_name()
                        except Exception:
                            _composed = f"[IR error] base={_cand.base_token}"
                        _items.append(
                            {
                                "primary_text": _composed,
                                "primary_text_style": "white",
                                "description": _desc,
                            }
                        )
                    if not _items:
                        _items.append(
                            {
                                "primary_text": (
                                    f"sn={_event.sn_ids[0]}"
                                    if _event.sn_ids
                                    else f"batch={_event.batch_id}"
                                ),
                                "primary_text_style": "white",
                                "description": batch.group_key,
                            }
                        )
                    state.loop_stats.stream_queue.add(_items)
                if _charge.overspend > 0:
                    wlog.warning(
                        "Compose batch %s overspent reservation by $%.4f "
                        "(batch cost $%.4f); budget tracking will report overrun",
                        batch.group_key,
                        _charge.overspend,
                        cost,
                    )

            # Quick grammar round-trip check on all candidates
            _grammar_failures: list[str] = []
            try:
                from imas_standard_names.grammar import parse_standard_name

                for c in result.candidates:
                    try:
                        _name = c.compose_name()
                        parse_standard_name(_name)
                    except Exception:
                        try:
                            _grammar_failures.append(c.compose_name())
                        except Exception:
                            _grammar_failures.append(c.source_id)
            except ImportError:
                pass  # ISN not installed — skip check

            # Component-token reuse check (free local embeddings): flag any
            # candidate-new token semantically near a registered same-segment
            # token so the agent gets a chained chance to reuse-or-confirm.
            _token_reuse_hits = await asyncio.to_thread(
                _compute_token_reuse_hits, result.vocab_gaps
            )

            if (not _grammar_failures and not _token_reuse_hits) or (
                _compose_attempt >= _max_retries
            ):
                break

            # Re-enrich items with expanded hybrid search for retry
            wlog.info(
                "Composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits — re-composing with expanded DD context",
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
                len(_token_reuse_hits),
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    batch_tuples = [
                        (
                            item.get("path"),
                            item.get("description"),
                            item.get("physics_domain"),
                        )
                        for item in batch.items
                        if item.get("path")
                    ]
                    if batch_tuples:
                        hybrid_results = _hybrid_search_neighbours_batch(
                            gc,
                            batch_tuples,
                            search_k=_retry_k_expansion(),
                        )
                        item_idx = 0
                        for item in batch.items:
                            if not item.get("path"):
                                continue
                            if hybrid_results[item_idx]:
                                item["hybrid_neighbours"] = hybrid_results[item_idx]
                            item_idx += 1

            # Only re-enrich when grammar failed; a token-reuse-only retry needs
            # no expanded neighbour context (the agent just reuses-or-confirms).
            if _grammar_failures:
                await asyncio.to_thread(_re_enrich_expanded)

            _reason_parts = []
            if _grammar_failures:
                _reason_parts.append(
                    f"Previous attempt failed: grammar round-trip failed for "
                    f"{', '.join(_grammar_failures[:3])}. Consider expanded "
                    f"neighbour context and produce a different name."
                )
            if _token_reuse_hits:
                _reason_parts.append(_build_token_reuse_retry_reason(_token_reuse_hits))
            _retry_reason = "\n\n".join(_reason_parts)
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt(prompt_template, retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        state.compose_stats.cost += _total_compose_cost
        state.compose_stats.processed += len(batch.items)
        state.compose_stats.record_batch(len(batch.items))

        candidates = []
        # Deterministic provenance collapse (see pooled path): one canonical
        # name per base quantity so estimator facets never drift apart.
        _prov_canon = provenance_canonical_names(result.candidates, log=wlog)
        for c in result.candidates:
            # Find the matching source item to get authoritative fields
            source_item = next(
                (item for item in batch.items if item.get("path") == c.source_id),
                None,
            )
            # Inject unit from DD (authoritative, not LLM output).
            #
            # User invariant (AGENTS.md "Unit safety"): the LLM never
            # decides units; the DD source must provide one.
            # - "1": dimensionless (ISN convention) — valid
            # - "-": DD dimensionless marker — normalize to "1"
            # - "mixed": heterogeneous dimensions — skip
            # - None/empty: enrichment couldn't resolve — skip
            raw_unit = source_item.get("unit") if source_item else None
            if raw_unit == "-":
                raw_unit = "1"  # normalize DD marker to ISN convention
            if raw_unit in ("mixed", None, ""):
                _skip_reason = (
                    "dd_unit_mixed_non_standard"
                    if raw_unit == "mixed"
                    else "dd_unit_unresolvable"
                )
                wlog.warning(
                    "compose: %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                if c.source_id and not state.dry_run:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        _prefix = "dd" if state.source == "dd" else "signals"
                        with GraphClient() as _gc:
                            mark_source_skipped(
                                _gc,
                                c.source_id,
                                reason=_skip_reason,
                                detail=str(raw_unit),
                                source_type=_prefix,
                            )
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                state.compose_stats.errors += 1
                continue
            unit = raw_unit

            # Inject physics_domain from DD (authoritative, like unit).
            # Post-refactor: scalar primary + source_domains list.
            raw_domain = source_item.get("physics_domain") if source_item else None
            physics_domain = raw_domain or None
            source_domains = [raw_domain] if raw_domain else []

            # Inject COCOS metadata from DD (authoritative, like unit)
            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _inject_shape_parameter_surface(c, c.source_id, wlog)

            # Normalize name via grammar round-trip BEFORE persist
            # to avoid duplicate nodes if validate would rename
            name_id = normalize_spelling(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = detect_value_provenance(c.source_id or "")
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]

            # W4b: Pre-validation gate — reject malformed LLM output
            # before it reaches MERGE.
            _well_formed, _reject_reason = is_well_formed_candidate(name_id)
            if not _well_formed:
                state.stats["compose_pre_validation_rejects"] = (
                    state.stats.get("compose_pre_validation_rejects", 0) + 1
                )
                wlog.warning(
                    "Pre-validation reject: %r (%s)",
                    name_id[:80],
                    _reject_reason,
                )
                # Mark source as failed (if identifiable)
                if c.source_id and not state.dry_run:
                    try:
                        from imas_codex.graph.client import GraphClient

                        _prefix = "dd" if state.source == "dd" else "signals"
                        with GraphClient() as gc:
                            gc.query(
                                "MATCH (sns:StandardNameSource {id: $id}) "
                                "SET sns.status = 'failed', "
                                "    sns.claimed_at = null, "
                                "    sns.claim_token = null",
                                id=f"{_prefix}:{c.source_id}",
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            # Non-nameable coordinate/infrastructure gate: a bare timestamp,
            # latency, counter, or metadata token is not a physics observable.
            # Composing it produces a doomed bare name that fails the semantic
            # gate and burns review + every refine rotation before exhausting.
            # Route the source to ``skipped`` (terminal) so it never re-claims.
            _non_nameable, _skip_reason = is_non_nameable_coordinate(name_id)
            if _non_nameable:
                state.stats["compose_non_nameable_skips"] = (
                    state.stats.get("compose_non_nameable_skips", 0) + 1
                )
                wlog.warning(
                    "Non-nameable skip: %r (%s)",
                    name_id[:80],
                    _skip_reason,
                )
                if c.source_id and not state.dry_run:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        _src_type = "dd" if state.source == "dd" else "signals"
                        with GraphClient() as gc:
                            mark_source_skipped(
                                gc,
                                c.source_id,
                                reason=_skip_reason or "non_nameable_coordinate",
                                detail=f"bare non-nameable token: {name_id}",
                                source_type=_src_type,
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s skipped", c.source_id)
                continue

            # Deterministic source<->name consistency gate (tense + state
            # resolution). The LLM sometimes emits a species-level sibling
            # for a state-resolved source (R4 rotation: thermal_neutral_density
            # sourced from .../neutral/state/density_thermal) -- drop the
            # pairing so the source retries rather than persisting a
            # mis-attributed name. Same predicate the auto-attach path uses.
            if c.source_id:
                _paths = [
                    c.source_id,
                    *(p for p in (c.dd_paths or []) if p != c.source_id),
                ]
                _ok, _why = True, ""
                for _p in _paths:
                    _ok, _why = _is_attachment_consistent(
                        _p, name_id, existing_sources=[q for q in _paths if q != _p]
                    )
                    if not _ok:
                        break
                if not _ok:
                    state.stats["compose_consistency_rejects"] = (
                        state.stats.get("compose_consistency_rejects", 0) + 1
                    )
                    wlog.warning(
                        "Source-name consistency reject: %r (%s)",
                        name_id[:80],
                        _why,
                    )
                    continue

            grammar_failed = False
            try:
                from imas_standard_names.grammar import (
                    compose_standard_name,
                    parse_standard_name,
                )

                try:
                    parsed = parse_standard_name(name_id)
                except Exception as _order_exc:
                    # ISN ≥rc35 strict-order rejections carry the unique
                    # canonical spelling — adopt it deterministically instead
                    # of spending an L6 LLM retry. Anything else re-raises to
                    # the L6 path below.
                    _canonical = getattr(_order_exc, "canonical_form", None)
                    if not _canonical:
                        raise
                    wlog.debug(
                        "Canonical-order normalization: %r → %r",
                        name_id,
                        _canonical,
                    )
                    name_id = _canonical
                    parsed = parse_standard_name(name_id)
                normalized = compose_standard_name(parsed)
                if normalized != name_id:
                    wlog.debug(
                        "Pre-persist normalization: %r → %r", name_id, normalized
                    )
                    name_id = normalized

                # Fix grammar library bug where parse→compose doubles
                # adjacent tokens (e.g. magnetic_field_probe →
                # magnetic_magnetic_field_probe).  Safe because no
                # legitimate standard name has adjacent duplicate tokens.
                name_id = _dedup_adjacent_tokens(name_id, wlog)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug("Grammar parse failed for %r — attempting L6 retry", name_id)

                # --- L6: Grammar-failure re-prompt (single retry) ---
                state.grammar_retries += 1
                try:
                    retry_name, _l6_cost, _l6_ti, _l6_to = await _grammar_retry(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if lease and _l6_cost > 0:
                        _l6_event = LLMCostEvent(
                            model=model,
                            tokens_in=_l6_ti,
                            tokens_out=_l6_to,
                            sn_ids=(name_id,),
                            batch_id=f"{batch.group_key}-grammar-retry",
                            phase=(
                                getattr(state, "budget_phase_tag", "")
                                or "generate_name"
                            ),
                            service="standard-names",
                        )
                        lease.charge_event(_l6_cost, _l6_event)
                        if state.loop_stats is not None:
                            state.loop_stats.cost += _l6_cost
                            state.loop_stats.stream_queue.add(
                                [
                                    {
                                        "primary_text": f"sn={name_id}",
                                        "primary_text_style": "white",
                                        "description": f"{batch.group_key}-L6",
                                    }
                                ]
                            )
                    if retry_name and retry_name != name_id:
                        # Verify the retry result actually parses
                        parsed = parse_standard_name(retry_name)
                        normalized = compose_standard_name(parsed)
                        name_id = _dedup_adjacent_tokens(normalized, wlog)
                        grammar_failed = False
                        state.grammar_retries_succeeded += 1
                        wlog.info(
                            "L6: Grammar retry succeeded: %r → %r",
                            c.compose_name(),
                            name_id,
                        )
                except Exception:
                    wlog.debug("L6: Grammar retry also failed for %r", name_id)

            candidates.append(
                {
                    "id": name_id,
                    "source_types": ["dd"] if state.source == "dd" else ["signals"],
                    "source_id": c.source_id,
                    "kind": c.kind,
                    "source_paths": [
                        encode_source_path(
                            "dd" if state.source == "dd" else "signals", p
                        )
                        for p in (c.dd_paths or [])
                    ],
                    "reason": c.reason,
                    "unit": unit,
                    "physics_domain": physics_domain,
                    "source_domains": source_domains,
                    "cocos_transformation_type": cocos_type,
                    "cocos": batch.cocos_version,
                    "dd_version": batch.dd_version,
                    # L6: track grammar retry exhaustion
                    **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
                }
            )

            # --- B9: Mint error siblings deterministically ---
            # If the parent has HAS_ERROR edges, mint uncertainty
            # modifier siblings without LLM calls.
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                from imas_codex.standard_names.error_siblings import (
                    mint_error_siblings,
                )

                siblings = mint_error_siblings(
                    name_id,
                    error_node_ids=source_item["error_node_ids"],
                    unit=unit,
                    physics_domain=physics_domain,
                    cocos_type=cocos_type,
                    cocos_version=batch.cocos_version,
                    dd_version=batch.dd_version,
                )
                if siblings:
                    for s in siblings:
                        s["_from_error_sibling"] = True
                    candidates.extend(siblings)
                    state.error_siblings_minted = getattr(
                        state, "error_siblings_minted", 0
                    ) + len(siblings)
                    wlog.debug(
                        "B9: Minted %d error siblings for parent %r",
                        len(siblings),
                        name_id,
                    )

        # Collect vocab gaps and persist immediately
        if result.vocab_gaps:
            gap_dicts = []
            for vg in result.vocab_gaps:
                gap_dict = {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                state.stats.setdefault("vocab_gaps", []).append(gap_dict)
                gap_dicts.append(gap_dict)

            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _stamp_dedup_decision(gap_dicts, _token_reuse_hits)

            # Persist to graph immediately so gaps survive cost-limit interruption
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            source_type = "dd" if state.source == "dd" else "signals"
            await asyncio.to_thread(write_vocab_gaps, gap_dicts, source_type)
            wlog.debug("Persisted %d vocab gaps to graph", len(gap_dicts))

            if not state.dry_run:
                _update_sources_after_vocab_gap(gap_dicts, state.source, wlog)

        # Auto-detect novel physical_base tokens in composed candidates (W29).
        # These are surfaced as VocabGap nodes for ISN review without
        # requiring the LLM to emit explicit vocab_gap exits.
        if candidates:
            auto_gaps = _auto_detect_physical_base_gaps(candidates)
            if auto_gaps:
                # Dedupe against LLM-emitted gaps (by source_id+segment+token)
                existing_keys = {
                    (g["source_id"], g["segment"], g["token"])
                    for g in state.stats.get("vocab_gaps", [])
                }
                novel = [
                    g
                    for g in auto_gaps
                    if (g["source_id"], g["segment"], g["token"]) not in existing_keys
                ]
                if novel:
                    from imas_codex.standard_names.graph_ops import write_vocab_gaps

                    source_type = "dd" if state.source == "dd" else "signals"
                    await asyncio.to_thread(
                        write_vocab_gaps,
                        novel,
                        source_type,
                        skip_segment_filter=True,
                    )
                    state.stats.setdefault("vocab_gaps", []).extend(novel)
                    wlog.info("Auto-detected %d novel physical_base gaps", len(novel))
        if result.attachments:
            _process_attachments(result.attachments, state, wlog)
            if not state.dry_run:
                _update_sources_after_attach(result.attachments, state.source, wlog)

        # --- GRAPH-STATE-MACHINE: persist immediately per batch ---
        # This ensures completed batches survive cost-limit cancellation.
        if candidates:
            from datetime import UTC, datetime

            from imas_codex.standard_names.graph_ops import persist_generated_name_batch

            # Pro-rata per-candidate cost/token attribution.  One LLM
            # batch call produces N candidates; divide cost and tokens
            # so SUM(sn.llm_cost) aggregates back to the batch total and
            # callers can extract spend directly from the graph instead
            # of scraping logs.  Error-sibling candidates minted
            # deterministically below are excluded from the denominator
            # (they carry no LLM cost).
            _llm_candidates = [
                c for c in candidates if not c.get("_from_error_sibling")
            ]
            _n = max(len(_llm_candidates), 1)
            _pro_rata_cost = _total_compose_cost / _n
            _pro_rata_in = _total_tokens_in // _n
            _pro_rata_out = _total_tokens_out // _n
            _pro_rata_cache_r = _total_cache_read // _n
            _pro_rata_cache_w = _total_cache_creation // _n
            _llm_at = datetime.now(UTC).isoformat()
            for c in candidates:
                if c.get("_from_error_sibling"):
                    continue
                c["llm_cost"] = _pro_rata_cost
                c["llm_model"] = model
                c["llm_service"] = "standard-names"
                c["llm_at"] = _llm_at
                c["llm_tokens_in"] = _pro_rata_in
                c["llm_tokens_out"] = _pro_rata_out
                c["llm_tokens_cached_read"] = _pro_rata_cache_r
                c["llm_tokens_cached_write"] = _pro_rata_cache_w

            # Tag regen-path candidates so persist increments regen_count
            if state.regen:
                for c in candidates:
                    c["regen_increment"] = True

            written = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch.dd_version,
                cocos_version=batch.cocos_version,
                run_id=getattr(state.budget_manager, "run_id", None)
                if state.budget_manager
                else None,
            )
            wlog.debug("Persisted %d names from batch %s", written, batch.group_key)

        # Update StandardNameSource nodes to composed status
        if candidates and not state.dry_run:
            _update_sources_after_compose(candidates, state.source, wlog)

        wlog.info(
            "Batch %s: %d composed, %d attached, %d skipped (cost=$%.4f)",
            batch.group_key,
            len(result.candidates),
            len(result.attachments),
            len(result.skipped),
            cost,
        )
        # Stream batch completion to progress display
        attach_label = (
            f"+{len(result.attachments)} attached  " if result.attachments else ""
        )
        state.compose_stats.stream_queue.add(
            [
                {
                    "primary_text": batch.group_key,
                    "description": (
                        f"{len(result.candidates)} names  {attach_label}${cost:.3f}"
                    ),
                }
            ]
        )
        return candidates

    # --- Polling-based work distribution (42-polling-workers) ---
    # N independent workers poll an asyncio.Queue for batches. Each worker
    # reserves budget before processing; on failure the batch is re-enqueued
    # and the worker sleeps before retrying — no silent drops.  This replaces
    # the previous asyncio.gather fan-out + Semaphore pattern.
    batch_queue: asyncio.Queue = asyncio.Queue()
    for _q_batch in state.extracted:
        batch_queue.put_nowait(_q_batch)

    composed: list[dict] = []
    errors = 0
    _MAX_BUDGET_RETRIES = 5

    async def _compose_polling_worker(worker_id: int) -> None:
        nonlocal errors
        budget_retries = 0

        while not state.should_stop():
            try:
                batch = batch_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Budget gate — reserve before doing any LLM work.
            # Per-item cost calibrated to W36 graph data: compose mean
            # $0.13/item, p95 $0.47/item.  Reserve at $0.20/item (mean+50%
            # headroom) to cover the typical extension and avoid draining
            # the global pool via in-flight overshoot, which would starve
            # downstream review phases.
            lease = None
            if state.budget_manager:
                estimated = len(batch.items) * 0.20
                phase_tag = getattr(state, "budget_phase_tag", "") or "generate_name"
                lease = state.budget_manager.reserve(estimated, phase=phase_tag)
                if lease is None:
                    budget_retries += 1
                    if (
                        budget_retries > _MAX_BUDGET_RETRIES
                        or state.budget_manager.exhausted()
                    ):
                        wlog.info(
                            "Worker %d: budget exhausted — stopping",
                            worker_id,
                        )
                        break
                    # Re-enqueue batch and wait for other workers to release
                    batch_queue.put_nowait(batch)
                    wlog.debug(
                        "Worker %d: budget reserve failed for %s, "
                        "re-enqueuing (retry %d/%d)",
                        worker_id,
                        batch.group_key,
                        budget_retries,
                        _MAX_BUDGET_RETRIES,
                    )
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    continue

            try:
                candidates = await _compose_batch_body(batch, lease)
                if candidates:
                    composed.extend(candidates)
                budget_retries = 0  # Reset on success
            except Exception as exc:
                errors += 1
                wlog.warning(
                    "Worker %d: batch %s failed: %s",
                    worker_id,
                    batch.group_key,
                    exc,
                )
            finally:
                if lease:
                    lease.release_unused()

    n_compose_workers = get_compose_concurrency()
    await asyncio.gather(
        *[_compose_polling_worker(i) for i in range(n_compose_workers)]
    )

    state.composed = composed
    state.compose_stats.errors = errors

    attached = state.stats.get("attachments", 0)
    wlog.info(
        "Composition complete: %d composed, %d attached, %d errors (cost=$%.4f)",
        len(composed),
        attached,
        errors,
        state.compose_stats.cost,
    )

    # --- Batch-size telemetry (Workstream 2a) ---
    # Report the distribution of items per batch and the name-only mode
    # indicator so rotation summaries can compare name-only vs default
    # throughput without bespoke log scraping.
    if state.extracted:
        sizes = [len(b.items) for b in state.extracted]
        total_items_in_batches = sum(sizes)
        name_only_batches = sum(
            1 for b in state.extracted if getattr(b, "mode", "default") == "names"
        )
        singleton_count = sum(1 for s in sizes if s == 1)
        wlog.info(
            "Batch telemetry: %d batches (%d name_only), total_items=%d, "
            "mean=%.2f, min=%d, max=%d, singletons=%d (%.1f%%), cost_per_batch=$%.4f",
            len(sizes),
            name_only_batches,
            total_items_in_batches,
            total_items_in_batches / len(sizes) if sizes else 0.0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
            singleton_count,
            100.0 * singleton_count / len(sizes) if sizes else 0.0,
            state.compose_stats.cost / len(sizes) if sizes else 0.0,
        )
        state.stats["compose_batches"] = len(sizes)
        state.stats["compose_batches_name_only"] = name_only_batches
        state.stats["compose_batch_mean_size"] = (
            total_items_in_batches / len(sizes) if sizes else 0.0
        )
        state.stats["compose_batch_singleton_pct"] = (
            100.0 * singleton_count / len(sizes) if sizes else 0.0
        )

    state.stats["generate_name_count"] = len(composed)
    state.stats["compose_errors"] = errors
    state.stats["compose_cost"] = state.compose_stats.cost
    state.stats["compose_model"] = model
    state.stats["grammar_retries"] = state.grammar_retries
    state.stats["grammar_retries_succeeded"] = state.grammar_retries_succeeded
    state.stats["opus_revisions_attempted"] = state.opus_revisions_attempted
    state.stats["opus_revisions_accepted"] = state.opus_revisions_accepted

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


# =============================================================================
# VALIDATE phase
# =============================================================================


def _validate_via_isn(
    entry: dict,
) -> tuple[list[str], dict]:
    """Construct ISN Pydantic model and collect ALL validation issues.

    Returns:
        (issues: list[str], layer_summary: dict)

    Compose is always name-only (ADR-1): validation uses ISN's name-only
    model. This function is purely an annotator — it never rejects entries.
    Parseability is checked upstream by the grammar round-trip in
    validate_worker. This function attaches quality annotations.
    """
    from pydantic import ValidationError

    issues: list[str] = []
    summary = {
        "pydantic": {"passed": True, "error_count": 0},
        "semantic": {"issue_count": 0, "skipped": False},
        "description": {"issue_count": 0},
    }

    # Map codex dict keys to ISN model fields
    from imas_codex.standard_names.kind_derivation import to_isn_kind

    # Name-only ISN variants forbid dd_paths and physics_domain — those
    # fields only exist on the full (non-name-only) model. Compose is
    # always name-only (ADR-1), so we omit them here. DD-path provenance
    # is retained on the codex graph node via `source_paths`.
    isn_dict: dict[str, Any] = {
        "name": entry.get("id", ""),
        "kind": to_isn_kind(entry.get("kind", "scalar")),
    }
    # ISN metadata kind forbids unit field entirely
    if isn_dict["kind"] != "metadata":
        # User invariant: by the time we build the ISN dict, the unit
        # must come from DD (B1a above filters invalid units before this
        # path is reachable). Defensively assert so a logic regression
        # surfaces in tests rather than silently pollute the catalog
        # with `unit="1"`.
        unit = entry.get("unit")
        assert unit, (
            f"ISN dict missing unit for {entry.get('id')!r}; "
            "B1a (dd_unit_unresolvable skip) invariant violation"
        )
        isn_dict["unit"] = unit

    # Layer 1: Pydantic model construction (fires 18 validators)
    model = None
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict, name_only=True)
    except ValidationError as e:
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = len(e.errors())
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            issues.append(f"[pydantic:{field}] {err['msg']}")
    except Exception as e:
        # Non-validation errors (import issues, etc.) — don't crash
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = 1
        issues.append(f"[pydantic:unknown] {e}")

    # Layer 2: Semantic checks (only if model constructed)
    if model is not None:
        try:
            from imas_standard_names.validation.semantic import run_semantic_checks

            sem_issues = run_semantic_checks({isn_dict["name"]: model})
            summary["semantic"]["issue_count"] = len(sem_issues)
            issues.extend(f"[semantic] {i}" for i in sem_issues)
        except Exception as e:
            summary["semantic"]["skipped"] = True
            issues.append(f"[semantic] check failed: {e}")
    else:
        summary["semantic"]["skipped"] = True

    # Layer 2b: Structural checks (cross-entry consistency, normally run
    # at catalog-load time by ISN's YamlStore). Single-entry invocation
    # surfaces the per-name structural defects (e.g. mode-with-no-base,
    # forbidden-token use) at compose/validate time rather than waiting
    # for catalog assembly. Empty result is the expected pass case.
    summary["structural"] = {"issue_count": 0, "skipped": False}
    if model is not None:
        try:
            from imas_standard_names.validation.structural import (
                run_structural_checks,
            )

            struct_issues = run_structural_checks({isn_dict["name"]: model})
            summary["structural"]["issue_count"] = len(struct_issues)
            issues.extend(f"[structural] {i}" for i in struct_issues)
        except Exception as e:
            summary["structural"]["skipped"] = True
            issues.append(f"[structural] check failed: {e}")
    else:
        summary["structural"]["skipped"] = True

    # Layer 2c: Codex-side canonical-token + preposition checks. Runs even
    # when the ISN model fails to construct, because the violation is
    # determined from the candidate name itself rather than the model.
    from imas_codex.standard_names.audits import canonical_locus_check

    canon_issues = canonical_locus_check({"id": isn_dict.get("name", "")})
    summary["canonical"] = {"issue_count": len(canon_issues)}
    issues.extend(f"[canonical] {i}" for i in canon_issues)

    # Layer 3: Description quality
    try:
        from imas_standard_names.validation.description import validate_description

        desc_issues = validate_description(isn_dict)
        summary["description"]["issue_count"] = len(desc_issues)
        issues.extend(f"[description] {i['message']}" for i in desc_issues)
    except Exception as e:
        issues.append(f"[description] check failed: {e}")

    return issues, summary


def _is_quarantined(issues: list[str], layer_summary: dict) -> bool:
    """Determine whether validation issues are critical (quarantine the name).

    Critical failures that make a name unusable for publication:
    - Grammar round-trip failure (``parse_error:`` prefix)
    - Pydantic validation failure (layer 1 did not pass)
    - Empty or missing description (no ``id`` or empty string)
    - Invalid kind value
    - L3 critical audit failures (latex_def_check, synonym_check, multi_subject_check)
    - L6 grammar retry exhausted
    - ISN semantic/structural ERROR-level issues (ISN's own catalog publish gate
      hard-fails on these, so a name carrying one can never be published)

    Non-critical issues (semantic WARNING/INFO hints, description quality hints,
    non-critical audits) do NOT trigger quarantine — they are advisory.
    """
    # Grammar round-trip failures are always critical
    if any(i.startswith("parse_error:") for i in issues):
        return True

    # ISN semantic/structural ERROR-level issues are catalog-blocking: ISN's
    # publish-time validator (imas_standard_names, run by the ISNC gate) hard-
    # fails on them, so a name carrying one can never be published. Mirror that
    # authority here — an ISN-invalid name (e.g. a bare geometric quantity such
    # as 'coordinate' that lacks the required object/geometry/position
    # qualifier) must not reach validation_status='valid' and slip into the
    # export set. ISN emits an explicit 'ERROR -' severity marker for hard
    # failures; WARNING/INFO-level semantic hints stay advisory.
    for i in issues:
        if (i.startswith("[semantic]") or i.startswith("[structural]")) and (
            " ERROR - " in i
        ):
            return True

    # Grammar ambiguity is also critical — the name can't be reliably parsed
    if any("grammar:ambiguity" in i for i in issues):
        return True

    # Pydantic validation failure (model construction failed)
    pydantic = layer_summary.get("pydantic", {})
    if not pydantic.get("passed", True):
        return True

    # L6: Grammar retry exhausted
    if any("audit:grammar_retry_exhausted" in i for i in issues):
        return True

    # L3: Critical audit failures
    from imas_codex.standard_names.audits import has_critical_audit_failure

    if has_critical_audit_failure(issues):
        return True

    return False


def validate_name_candidate(entry: dict[str, Any]) -> tuple[list[str], dict, str]:
    """Run the full name-admission gate on a single candidate and classify it.

    This is the ONE gate every newly-minted StandardName passes before it can
    reach ``accepted`` — the same checks a pipeline-generated candidate clears:
    grammar round-trip, the ISN Pydantic/semantic/structural/canonical/
    description layers (:func:`_validate_via_isn`), and the L3 post-generation
    audits (:func:`run_audits`).  :func:`_is_quarantined` then decides whether
    the accumulated issues are critical.

    Both the pool ``generate_name`` path (inline C1 audit) and the legacy
    ``validate_worker`` classify names this way; the ``sn edit`` rename path
    calls this so an operator-supplied replacement name rides *exactly* the
    same gate rather than being stamped ``valid`` on a privileged path.

    Returns ``(issues, layer_summary, validation_status)`` where
    ``validation_status`` is ``"valid"`` or ``"quarantined"``.
    """
    name = entry.get("id", "")
    # A derived family parent is a deliberately partial name (a grammar peel
    # that drops the segment its children carry — species, projection axis, …).
    # The full-name round-trip below is the wrong gate for it: dropping that
    # segment is the whole point, so a legitimately partial parent would be
    # mis-flagged as an unparseable standalone name. Such parents are validated
    # STRUCTURALLY (children exist + the peel generalises them) instead — while
    # a structurally-broken parent (orphan / inconsistent peel) still
    # quarantines, so the missed-gate signal is preserved.
    is_derived_parent = entry.get("origin") == "derived"
    derived_children = entry.get("children") or []
    try:
        from imas_standard_names.grammar import (
            StandardName,
            compose_standard_name,
            parse_standard_name,
        )

        from imas_codex.standard_names.audits import derived_parent_structural_check

        # Grammar round-trip validates parsability.
        parsed = parse_standard_name(name)
        compose_standard_name(parsed)

        # Fields-consistency check (best-effort — never rejects here).
        fields_dict = {}
        for fk in _GRAMMAR_FIELDS:
            val = entry.get(fk)
            if val:
                fields_dict[fk] = val
        if fields_dict:
            try:
                sn_fields = _convert_fields_to_grammar(fields_dict)
                if sn_fields:
                    sn = StandardName(**sn_fields)
                    compose_standard_name(sn)
            except Exception:
                pass

        # ISN three-layer validation (annotate).
        issues, layer_summary = _validate_via_isn(entry)

        # L3: post-generation audits.
        try:
            from imas_codex.standard_names.audits import run_audits

            source_path = None
            source_paths = entry.get("source_paths") or []
            if source_paths:
                source_path = strip_dd_prefix(source_paths[0])
            audit_issues = run_audits(
                candidate=entry,
                existing_sns_in_domain=None,
                source_path=source_path,
                source_cocos_type=entry.get("cocos_transformation_type"),
            )
            if audit_issues:
                issues.extend(audit_issues)
        except Exception:
            logger.debug("validate_name_candidate: audits failed for %r", name)

        if entry.get("_grammar_retry_exhausted"):
            issues.append("audit:grammar_retry_exhausted")

        # A derived parent that DOES round-trip still owes its structural
        # contract — an orphan parent that happens to parse generalises nothing.
        if is_derived_parent:
            issues.extend(derived_parent_structural_check(name, derived_children))

        status = "quarantined" if _is_quarantined(issues, layer_summary) else "valid"
        return issues, layer_summary, status
    except Exception as exc:
        # A derived family parent is intentionally partial; a full-name parse
        # failure is expected, not a defect. Validate it structurally instead —
        # it quarantines only when its family structure is broken.
        if is_derived_parent:
            struct_issues = derived_parent_structural_check(name, derived_children)
            status = "quarantined" if struct_issues else "valid"
            return struct_issues, {}, status
        exc_msg = str(exc).lower()
        issues = []
        if "component" in exc_msg and "coordinate" in exc_msg:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")
        return issues, {}, "quarantined"


async def drain_validation_backlog(batch_size: int = 50) -> dict[str, int]:
    """Drain every unvalidated StandardName through the admission gate.

    Standalone claim→validate→mark loop over ``claim_names_for_validation``
    (any name with a description and ``validated_at`` null). Deterministic and
    LLM-free — safe to run as a maintenance pass after ``--revalidate`` clears
    stamps, without composing or reviewing anything. Returns
    ``{"validated": n, "quarantined": m}``.
    """
    from imas_codex.standard_names.graph_ops import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    totals = {"validated": 0, "quarantined": 0}
    while True:
        token, items = await asyncio.to_thread(claim_names_for_validation, batch_size)
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = validate_name_candidate(entry)
                if status == "quarantined":
                    totals["quarantined"] += 1
                results.append(
                    {
                        "id": entry.get("id", ""),
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_backlog: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


# Matches the backlog claim timeout in graph_ops so a stale scoped claim
# expires on the same schedule.
_SCOPED_VALIDATION_CLAIM_TIMEOUT = "PT300S"


def claim_ids_for_validation(
    ids: Sequence[str], limit: int
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim named StandardNames for a scoped ISN re-validation.

    The id-scoped counterpart of ``claim_names_for_validation``: only nodes in
    *ids* whose ``validated_at`` is null are claimed, so a campaign re-stamp
    never touches names outside its batch. Returns ``(token, items)`` where the
    token must be passed to ``mark_names_validated`` or
    ``release_validation_claims``.
    """
    import uuid

    from imas_codex.graph.client import GraphClient

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
              AND sn.description IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            ids=list(ids),
            limit=limit,
            token=token,
            timeout=_SCOPED_VALIDATION_CLAIM_TIMEOUT,
        )
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested']
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.object AS object,
                   sn.physics_domain AS physics_domain,
                   sn.origin AS origin,
                   collect(DISTINCT src.id) AS source_ids,
                   collect(DISTINCT child.id) AS children
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


async def drain_validation_for_ids(
    ids: Sequence[str], *, batch_size: int = 50
) -> dict[str, Any]:
    """Re-run the deterministic ISN audit on an explicit id scope.

    Id-scoped counterpart of :func:`drain_validation_backlog`. Claims only the
    named nodes whose ``validated_at`` is null — a campaign clears the stamp on
    a batch before calling this — runs the shared admission gate
    (:func:`validate_name_candidate`), and re-stamps ``validation_issues`` /
    ``validation_status`` / ``validated_at``. LLM-free; a genuine defect (e.g. a
    unit inconsistency) re-quarantines instead of washing to 'valid'. Returns
    ``{"validated": n, "quarantined": m, "requarantined_ids": [...],
    "cleared_ids": [...]}``.
    """
    from imas_codex.standard_names.graph_ops import (
        mark_names_validated,
        release_validation_claims,
    )

    ids = list(ids)
    totals: dict[str, Any] = {
        "validated": 0,
        "quarantined": 0,
        "requarantined_ids": [],
        "cleared_ids": [],
    }
    if not ids:
        return totals
    while True:
        token, items = await asyncio.to_thread(
            claim_ids_for_validation, ids, batch_size
        )
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = validate_name_candidate(entry)
                sid = entry.get("id", "")
                if status == "quarantined":
                    totals["quarantined"] += 1
                    totals["requarantined_ids"].append(sid)
                else:
                    totals["cleared_ids"].append(sid)
                results.append(
                    {
                        "id": sid,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_for_ids: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


async def validate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Validate composed names via ISN grammar checks (claim loop).

    Graph-primary: claims unvalidated StandardName nodes, runs ISN
    three-layer validation + grammar round-trip, marks results on graph.
    Follows the claim/mark/release pattern from discovery workers.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_validate_worker")

    # Initialize finalize progress (3 steps: validate, consolidate, persist)
    state.finalize_stats.total = 3
    state.finalize_stats.status_text = "validating…"

    if state.dry_run:
        wlog.info("Dry run — skipping validation")
        count = sum(len(b.items) for b in state.extracted) if state.extracted else 0
        state.validate_stats.total = count
        state.validate_stats.processed = count
        state.stats["validate_skipped"] = True
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    _BATCH_SIZE = 50

    total_valid = 0
    total_invalid = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 50

    wlog.info("Starting validation claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_validation, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more unvalidated names — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for validation (token=%s)", len(items), token[:8])

        # Process the claimed batch
        try:
            results: list[dict[str, Any]] = []
            batch_invalid = 0

            for entry in items:
                name = entry.get("id", "")
                # Single shared admission gate (see validate_name_candidate) —
                # the same classification the pool generate_name path and the
                # sn edit rename path apply, so every name is judged identically.
                issues, layer_summary, status = validate_name_candidate(entry)
                state.audits_run += 1
                if any(i.startswith("audit:") or i.startswith("[") for i in issues):
                    state.audits_failed += 1
                if status == "quarantined":
                    batch_invalid += 1
                    wlog.debug("Validation quarantined %r: %s", name, issues[:2])
                results.append(
                    {
                        "id": name,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )

            # Mark results on graph (token-verified)
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            total_valid += marked
            total_invalid += batch_invalid
            state.validate_stats.processed += marked

            wlog.info(
                "Validated batch: %d marked, %d errors",
                marked,
                batch_invalid,
            )
            state.validate_stats.record_batch(marked)

        except Exception:
            wlog.warning("Validation batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_validation_claims, token)

    state.stats["validate_valid"] = total_valid
    state.stats["validate_invalid"] = total_invalid
    state.stats["audits_run"] = state.audits_run
    state.stats["audits_failed"] = state.audits_failed
    state.validate_stats.errors = total_invalid
    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()
    state.finalize_stats.processed = 1
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "validate",
                "description": f"{total_valid} valid  {total_invalid} invalid",
            }
        ]
    )


def _convert_fields_to_grammar(fields: dict) -> dict:
    """Convert string field values to grammar enum instances."""
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )

    enum_map = {
        "subject": Subject,
        "component": Component,
        "coordinate": Component,
        "position": Position,
        "process": Process,
        "transformation": Transformation,
        "geometric_base": GeometricBase,
        "object": Object,
        "binary_operator": BinaryOperator,
    }

    sn_fields: dict = {}
    for k, v in fields.items():
        if k == "physical_base":
            sn_fields[k] = v
        elif k in enum_map:
            sn_fields[k] = enum_map[k](v)
    return sn_fields


# =============================================================================
# CONSOLIDATE phase
# =============================================================================


async def consolidate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Graph-primary: queries all validated StandardNames from graph, runs
    consolidation analysis, marks approved names with ``consolidated_at``.
    Read-only query (no claims needed — single-pass batch analysis).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    state.finalize_stats.status_text = "consolidating…"

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.standard_names.consolidation import consolidate_candidates
    from imas_codex.standard_names.graph_ops import (
        get_validated_names,
        mark_names_consolidated,
    )

    # Always read from graph — this is the primary data source
    validated = await asyncio.to_thread(
        get_validated_names,
        ids_filter=getattr(state, "ids_filter", None),
    )

    if not validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    wlog.info("Consolidating %d validated candidates from graph", len(validated))
    state.consolidate_stats.total = len(validated)

    result = await asyncio.to_thread(consolidate_candidates, validated)

    # Mark approved names with consolidated_at on graph
    approved_ids = [e["id"] for e in result.approved if e.get("id")]
    if approved_ids:
        marked = await asyncio.to_thread(mark_names_consolidated, approved_ids)
        wlog.info("Marked %d names as consolidated", marked)

    # Log results
    wlog.info(
        "Consolidation: %d approved, %d conflicts, %d coverage gaps, %d reused",
        len(result.approved),
        len(result.conflicts),
        len(result.coverage_gaps),
        len(result.reused),
    )

    # Record stats
    state.stats["consolidation"] = result.stats
    if result.conflicts:
        for conflict in result.conflicts:
            wlog.warning(
                "Conflict: %s (%s) — %s",
                conflict.standard_name,
                conflict.conflict_type,
                conflict.details,
            )
    if result.coverage_gaps:
        wlog.info("Coverage gaps: %d unmapped source paths", len(result.coverage_gaps))

    state.consolidate_stats.processed = len(validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()
    state.finalize_stats.processed = 2
    conflicts_count = len(result.conflicts) if result.conflicts else 0
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "consolidate",
                "description": (
                    f"{len(result.approved)} names  {conflicts_count} conflicts"
                ),
            }
        ]
    )


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Compute embeddings for consolidated StandardNames (claim loop).

    Graph-primary: claims unembedded StandardNames, computes embeddings,
    writes results back to graph. Names are already persisted by compose —
    this worker handles the embedding enrichment pass.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    state.finalize_stats.status_text = "embedding…"

    if state.dry_run:
        wlog.info("Dry run — skipping embedding")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_embedding,
        mark_names_embedded,
        release_embedding_claims,
    )

    total_embedded = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 100

    wlog.info("Starting embedding claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_embedding, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more names needing embedding — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for embedding (token=%s)", len(items), token[:8])

        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            enriched = await asyncio.to_thread(embed_descriptions_batch, items)
            embed_batch = [
                {"id": e["id"], "embedding": e["embedding"]}
                for e in enriched
                if e.get("embedding")
            ]

            marked = await asyncio.to_thread(mark_names_embedded, token, embed_batch)
            total_embedded += marked

            wlog.info("Embedded batch: %d names", marked)
            state.persist_stats.processed += marked
            state.persist_stats.record_batch(marked)

        except Exception:
            wlog.warning("Embedding batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_embedding_claims, token)

    # Post-success cleanup: detach stale HAS_STANDARD_NAME for targeted paths
    # Only runs when --force/--paths regenerated names for specific paths
    if state.force and total_embedded > 0 and state.extracted:
        new_name_ids: set[str] = set()
        source_paths: set[str] = set()

        # Collect from graph — names we just embedded
        from imas_codex.graph.client import GraphClient

        def _get_recent_names():
            with GraphClient() as gc:
                results = gc.query(
                    """
                    MATCH (sn:StandardName)
                    WHERE sn.embedded_at IS NOT NULL
                      AND sn.name_stage = 'drafted'
                    RETURN sn.id AS id, sn.source_paths AS source_paths
                    """
                )
                for r in results:
                    new_name_ids.add(r["id"])
                    for p in r["source_paths"] or []:
                        source_paths.add(p)

        await asyncio.to_thread(_get_recent_names)

        if source_paths and new_name_ids:

            def _cleanup_stale():
                # Strip dd: prefix for IMASNode.id lookup
                bare_paths = [strip_dd_prefix(p) for p in source_paths]
                with GraphClient() as gc:
                    result = list(
                        gc.query(
                            """
                            UNWIND $paths AS path
                            MATCH (n:IMASNode {id: path})-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                            WHERE NOT (sn.id IN $keep_names)
                              AND coalesce(sn.name_stage, 'drafted') = 'drafted'
                            DELETE r
                            RETURN count(r) AS detached
                            """,
                            paths=bare_paths,
                            keep_names=list(new_name_ids),
                        )
                    )
                    return result[0]["detached"] if result else 0

            detached = await asyncio.to_thread(_cleanup_stale)
            if detached:
                wlog.info("Cleaned %d stale HAS_STANDARD_NAME relationships", detached)
                state.stats["stale_detached"] = detached

    state.stats["persist_embedded"] = total_embedded
    wlog.info("Persist complete: %d embedded", total_embedded)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.finalize_stats.processed = 3
    state.finalize_stats.status_text = "done"
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{total_embedded} embedded",
            }
        ]
    )
    state.finalize_stats.freeze_rate()


# =============================================================================
# Pool-mode batch processors (Phase 8)
# =============================================================================


async def compose_batch(
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    regen: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Shared implementation for compose and regen pool batch processors.

    Takes a list of pre-claimed items (dicts with ``path``, ``description``,
    ``physics_domain``, etc.) and runs the compose pipeline:
    prompt → LLM → grammar validate → persist.

    **H5 — batch-scope domain context:**
    Domain vocabulary is derived from the *batch items* rather than a
    run-scoped ``state.domain_filter``, so pooled-mode batches get domain
    context even without ``--physics-domain``.

    **H6 — soft drain on stop_event:**
    Checks ``stop_event.is_set()`` before the LLM call and returns early
    (after persisting any in-flight results) if the event fires.

    Args:
        batch: Pre-claimed source items from the graph claim query.
        mgr: Shared :class:`BudgetManager` for cost tracking.
        stop_event: Cooperative shutdown signal.
        regen: If True, set ``regen_increment`` on persisted candidates.

    Returns:
        Count of successfully composed/regenerated candidates.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_compose_self_refine,
        get_model,
        get_reasoning_effort,
    )
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    if not batch:
        return 0

    model = get_model("sn-compose")
    context = build_compose_context()

    # ── H5: batch-scope domain context ─────────────────────────────────
    def _scalar_domain(d: object) -> str | None:
        """Normalise physics_domain to a scalar string (handles list or str)."""
        if isinstance(d, list):
            return d[0] if d else None
        return d  # type: ignore[return-value]

    domains_in_batch = sorted(
        {
            _scalar_domain(item.get("physics_domain"))
            for item in batch
            if item.get("physics_domain")
        }
        - {None}
    )

    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab_parts: list[str] = []
    for dom in domains_in_batch:
        vocab = await asyncio.to_thread(build_domain_vocabulary_preseed, dom)
        if vocab:
            domain_vocab_parts.append(vocab)
    context["domain_vocabulary"] = "\n".join(domain_vocab_parts)

    # ── L4: Reviewer-theme extraction (batch-scoped) ───────────────────
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    for dom in domains_in_batch:
        themes = await asyncio.to_thread(extract_reviewer_themes, dom)
        reviewer_themes.extend(themes)
    context["reviewer_themes"] = reviewer_themes[:12]

    # ── K3: Scored-example injection ───────────────────────────────────
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(
                gc, physics_domains=domains_in_batch, axis="name"
            )

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    context["compose_scored_examples"] = compose_scored_examples

    # ── System prompt (cached per pool lifetime) ───────────────────────
    system_prompt = render_prompt("sn/generate_name_system", context)

    # ── Enrich items with DD context ───────────────────────────────────
    def _enrich():
        _enrich_batch_items(batch)

    await asyncio.to_thread(_enrich)

    # ── Search nearby existing names (per-item) ──────────────────────────
    # For each batch item, search using the item's description (or path
    # leaf) to get semantically relevant nearby names.  Deduplicate across
    # items and cap at 30 total to keep prompt size bounded.
    group_key = batch[0].get("path", "").split("/")[0] if batch else ""
    _nearby_seen: set[str] = set()
    nearby: list[dict] = []
    _PER_ITEM_K = 5
    _NEARBY_CAP = 30
    for item in batch:
        # Prefer item description for embedding query; fall back to path
        hint = item.get("description") or item.get("path", "")
        item_results = _search_nearby_names(hint, k=_PER_ITEM_K)
        for nr in item_results:
            nid = nr.get("id", "")
            if nid and nid not in _nearby_seen:
                _nearby_seen.add(nid)
                nearby.append(nr)
                if len(nearby) >= _NEARBY_CAP:
                    break
        if len(nearby) >= _NEARBY_CAP:
            break

    # ── IDS context ────────────────────────────────────────────────────
    ids_names = sorted(
        {
            item["path"].split("/")[0]
            for item in batch
            if item.get("path") and "/" in item["path"]
        }
    )
    ids_contexts: list[dict] = []
    for iname in ids_names:
        info = _enrich_ids_context(iname)
        if info:
            ids_contexts.append({"ids_name": iname, **info})

    # ── H6: pre-LLM stop check ────────────────────────────────────────
    if stop_event.is_set():
        return 0

    # ── B9: Reference exemplars — superseded by graph-backed
    # ``compose_scored_examples`` loaded above. The active prompt
    # template ``sn/generate_name_dd`` no longer renders a
    # ``reference_exemplars`` block (removed in commit 7a86069e), so
    # the per-batch ANN search has been retired here. The linear path
    # still uses ``_search_reference_exemplars`` for the names-only
    # template ``sn/generate_name_dd_names``.

    # ── B10: Cluster-aware existing-names roster ──────────────────────
    # Build name-only roster from any existing_name field carried on batch
    # items plus a graph query for SNs in the same IMASSemanticCluster.
    existing_names_set: set[str] = {
        item.get("existing_name") for item in batch if item.get("existing_name")
    }
    existing_names_set.discard(None)
    try:
        from imas_codex.graph.client import GraphClient

        path_ids = [item.get("path") for item in batch if item.get("path")]
        if path_ids:

            def _cluster_roster() -> list[str]:
                with GraphClient() as gc:
                    rows = gc.query(
                        """
                        UNWIND $paths AS p
                        MATCH (n:IMASNode {id: p})-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other:IMASNode)-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                        WHERE coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') <> 'superseded'
                        RETURN DISTINCT sn.id AS id
                        LIMIT 200
                        """,
                        paths=path_ids,
                    )
                    return [r["id"] for r in (rows or []) if r.get("id")]

            cluster_names = await asyncio.to_thread(_cluster_roster)
            existing_names_set.update(cluster_names)
    except Exception:
        logger.debug("Cluster-aware existing_names lookup failed", exc_info=True)
    existing_names = sorted(existing_names_set)[:200]

    # ── Render user prompt ─────────────────────────────────────────────
    user_context = {
        "items": batch,
        "ids_name": group_key,
        "ids_contexts": ids_contexts,
        "existing_names": existing_names,
        "cluster_context": "",
        "nearby_existing_names": nearby,
        "cocos_version": batch[0].get("cocos_version") if batch else None,
        "dd_version": batch[0].get("dd_version") if batch else None,
    }
    user_prompt = render_prompt("sn/generate_name_dd", {**context, **user_context})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── Budget reservation ─────────────────────────────────────────────
    # B12: reserve for retry_attempts + 1 LLM calls (re-enrichment retry)
    _max_retries = _retry_attempts()
    # Token-reuse hits from the FINAL attempt (carried to the VocabGap stamp).
    _token_reuse_hits: dict[tuple[str, str], Any] = {}
    estimated = len(batch) * 0.20 * (_max_retries + 1)
    phase_tag = "regen" if regen else "generate_name"
    lease = mgr.reserve(estimated, phase=phase_tag)
    if lease is None:
        # Fallback: tracking-only lease so charge_event still records
        # spend.  Without this, _spent stays at 0 and hard_exhausted()
        # never fires — causing the pool to run indefinitely.
        lease = mgr.reserve(0.0, phase=phase_tag)

    try:
        # ── B12: bounded retry loop for failed compositions ────────────
        # Mirror of the linear path's Delta H retry: on grammar parse
        # failure across any candidate, re-enrich items with expanded DD
        # context and re-prompt.  Bounded by _retry_attempts().
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0

        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                )
            except (ValueError, Exception) as compose_exc:
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Pool %s: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        phase_tag,
                        len(batch),
                        _exc_str[:200],
                    )
                    _mark_vocab_gap_sources(batch, _exc_str, "dd")
                    return 0
                raise
            result, cost, tokens = llm_out

            # Accumulate token/cost totals across retries
            _total_compose_cost += cost
            _attempt_tokens_in = getattr(llm_out, "input_tokens", 0) or 0
            _attempt_tokens_out = getattr(llm_out, "output_tokens", 0) or 0
            _attempt_cache_r = getattr(llm_out, "cache_read_tokens", 0) or 0
            _attempt_cache_w = getattr(llm_out, "cache_creation_tokens", 0) or 0
            _total_tokens_in += _attempt_tokens_in
            _total_tokens_out += _attempt_tokens_out
            _total_cache_read += _attempt_cache_r
            _total_cache_creation += _attempt_cache_w

            # Charge this attempt's cost immediately
            if lease:

                def _safe_sn_ids(_r=result) -> tuple[str, ...]:
                    ids: list[str] = []
                    for c in _r.candidates if _r else []:
                        try:
                            ids.append(c.compose_name())
                        except Exception:
                            ids.append(f"<compose-error:{c.segments.base_token}>")
                    return tuple(ids)

                _event = LLMCostEvent(
                    model=model,
                    tokens_in=_attempt_tokens_in,
                    tokens_out=_attempt_tokens_out,
                    tokens_cached_read=_attempt_cache_r,
                    tokens_cached_write=_attempt_cache_w,
                    sn_ids=_safe_sn_ids(),
                    batch_id=group_key,
                    phase=phase_tag,
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # Quick grammar round-trip check on all candidates
            _grammar_failures: list[str] = []
            try:
                from imas_standard_names.grammar import parse_standard_name

                for c in result.candidates:
                    try:
                        _name = c.compose_name()
                        parse_standard_name(_name)
                    except Exception:
                        try:
                            _grammar_failures.append(c.compose_name())
                        except Exception:
                            _grammar_failures.append(c.source_id)
            except ImportError:
                pass  # ISN not installed — skip check

            # Component-token reuse check (free local embeddings): flag any
            # candidate-new token semantically near a registered same-segment
            # token so the agent gets a chained chance to reuse-or-confirm.
            _token_reuse_hits = await asyncio.to_thread(
                _compute_token_reuse_hits, result.vocab_gaps
            )

            if (not _grammar_failures and not _token_reuse_hits) or (
                _compose_attempt >= _max_retries
            ):
                break

            # Re-enrich items with expanded hybrid search for retry
            logger.info(
                "Pool %s: composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits — re-composing with expanded DD context",
                phase_tag,
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
                len(_token_reuse_hits),
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    batch_tuples = [
                        (
                            item.get("path"),
                            item.get("description"),
                            item.get("physics_domain"),
                        )
                        for item in batch
                        if item.get("path")
                    ]
                    if batch_tuples:
                        hybrid_results = _hybrid_search_neighbours_batch(
                            gc,
                            batch_tuples,
                            search_k=_retry_k_expansion(),
                        )
                        item_idx = 0
                        for item in batch:
                            if not item.get("path"):
                                continue
                            if hybrid_results[item_idx]:
                                item["hybrid_neighbours"] = hybrid_results[item_idx]
                            item_idx += 1

            # Only re-enrich when grammar failed; a token-reuse-only retry needs
            # no expanded neighbour context (the agent just reuses-or-confirms).
            if _grammar_failures:
                await asyncio.to_thread(_re_enrich_expanded)

            _reason_parts = []
            if _grammar_failures:
                _reason_parts.append(
                    f"Previous attempt failed: grammar round-trip failed for "
                    f"{', '.join(_grammar_failures[:3])}. Consider expanded "
                    f"neighbour context and produce a different name."
                )
            if _token_reuse_hits:
                _reason_parts.append(_build_token_reuse_retry_reason(_token_reuse_hits))
            _retry_reason = "\n\n".join(_reason_parts)
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt("sn/generate_name_dd", retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        # Use accumulated totals for per-candidate cost attribution
        cost = _total_compose_cost
        tokens_in = _total_tokens_in
        tokens_out = _total_tokens_out
        tokens_cache_r = _total_cache_read
        tokens_cache_w = _total_cache_creation

        # ── Build candidates ───────────────────────────────────────────
        candidates: list[dict] = []
        wlog = logger  # plain logger; pool path has no WorkerLogAdapter
        source_kind = "dd"  # pool-mode generate-name is DD-only today
        # Deterministic provenance collapse: one canonical name per base quantity
        # so estimator facets (measured/reconstructed/reference) never drift apart.
        _prov_canon = provenance_canonical_names(result.candidates, log=wlog)
        for c in result.candidates:
            source_item = next(
                (item for item in batch if item.get("path") == c.source_id),
                None,
            )
            # User invariant (AGENTS.md "Unit safety"): the LLM never
            # decides units; the DD source must provide one.
            # - "1": dimensionless (ISN convention) — valid
            # - "-": DD dimensionless marker — normalize to "1"
            # - "mixed": heterogeneous dimensions — skip
            # - None/empty: enrichment couldn't resolve — skip
            raw_unit = source_item.get("unit") if source_item else None
            if raw_unit == "-":
                raw_unit = "1"  # normalize DD marker to ISN convention
            if raw_unit in ("mixed", None, ""):
                _skip_reason = (
                    "dd_unit_mixed_non_standard"
                    if raw_unit == "mixed"
                    else "dd_unit_unresolvable"
                )
                wlog.warning(
                    "compose(pool): %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                if c.source_id:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        with GraphClient() as _gc:
                            mark_source_skipped(
                                _gc,
                                c.source_id,
                                reason=_skip_reason,
                                detail=str(raw_unit),
                                source_type=source_kind,
                            )
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark pool source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                continue
            unit = raw_unit

            raw_domain = source_item.get("physics_domain") if source_item else None
            physics_domain = raw_domain or None
            source_domains = [raw_domain] if raw_domain else []

            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _inject_shape_parameter_surface(c, c.source_id, wlog)

            # B1/W4b: Pre-validation gate — reject malformed LLM output
            # before MERGE.  Mark the source as 'failed' so it is not
            # re-claimed forever.
            name_id = normalize_spelling(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = detect_value_provenance(c.source_id or "")
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]
            _well_formed, _reject_reason = is_well_formed_candidate(name_id)
            if not _well_formed:
                wlog.warning(
                    "Pool %s: pre-validation reject %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _reject_reason,
                )
                if c.source_id:
                    try:
                        from imas_codex.graph.client import GraphClient

                        with GraphClient() as gc:
                            gc.query(
                                "MATCH (sns:StandardNameSource {id: $id}) "
                                "SET sns.status = 'failed', "
                                "    sns.claimed_at = null, "
                                "    sns.claim_token = null",
                                id=f"{source_kind}:{c.source_id}",
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            # Non-nameable coordinate/infrastructure gate: a bare timestamp,
            # latency, counter, or metadata token is not a physics observable.
            # Composing it produces a doomed bare name that fails the semantic
            # gate and burns review + every refine rotation before exhausting.
            # Route the source to ``skipped`` (terminal) so it never re-claims.
            _non_nameable, _skip_reason = is_non_nameable_coordinate(name_id)
            if _non_nameable:
                wlog.warning(
                    "Pool %s: non-nameable skip %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _skip_reason,
                )
                if c.source_id:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        with GraphClient() as gc:
                            mark_source_skipped(
                                gc,
                                c.source_id,
                                reason=_skip_reason or "non_nameable_coordinate",
                                detail=f"bare non-nameable token: {name_id}",
                                source_type=source_kind,
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s skipped", c.source_id)
                continue

            # Deterministic source<->name consistency gate (tense + state
            # resolution). The LLM sometimes emits a species-level sibling
            # for a state-resolved source (R4 rotation: thermal_neutral_density
            # sourced from .../neutral/state/density_thermal) -- drop the
            # pairing so the source retries rather than persisting a
            # mis-attributed name. Same predicate the auto-attach path uses.
            if c.source_id:
                _paths = [
                    c.source_id,
                    *(p for p in (c.dd_paths or []) if p != c.source_id),
                ]
                _ok, _why = True, ""
                for _p in _paths:
                    _ok, _why = _is_attachment_consistent(
                        _p, name_id, existing_sources=[q for q in _paths if q != _p]
                    )
                    if not _ok:
                        break
                if not _ok:
                    wlog.warning(
                        "Source-name consistency reject: %r (%s)",
                        name_id[:80],
                        _why,
                    )
                    continue

            grammar_failed = False
            try:
                from imas_standard_names.grammar import (
                    compose_standard_name,
                    parse_standard_name,
                )

                parsed = parse_standard_name(name_id)
                normalized = compose_standard_name(parsed)
                if normalized != name_id:
                    name_id = normalized
                name_id = _dedup_adjacent_tokens(name_id)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug(
                    "Pool %s: grammar parse failed for %r — attempting B2 retry",
                    phase_tag,
                    name_id,
                )

                # B2: Single-shot grammar-failure retry.
                try:
                    retry_name, _r_cost, _r_ti, _r_to = await _grammar_retry(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if lease and _r_cost > 0:
                        _r_event = LLMCostEvent(
                            model=model,
                            tokens_in=_r_ti,
                            tokens_out=_r_to,
                            sn_ids=(name_id,),
                            batch_id=f"{group_key}-grammar-retry",
                            phase=phase_tag,
                            service="standard-names",
                        )
                        lease.charge_event(_r_cost, _r_event)
                    if retry_name and retry_name != name_id:
                        try:
                            parsed = parse_standard_name(retry_name)
                            normalized = compose_standard_name(parsed)
                            name_id = _dedup_adjacent_tokens(normalized)
                            grammar_failed = False
                            wlog.info(
                                "Pool %s: B2 grammar retry succeeded %r → %r",
                                phase_tag,
                                c.compose_name(),
                                name_id,
                            )
                        except Exception:
                            wlog.debug(
                                "Pool %s: B2 retry result still un-parseable",
                                phase_tag,
                            )
                except Exception:
                    wlog.debug(
                        "Pool %s: B2 grammar retry failed for %r",
                        phase_tag,
                        name_id,
                    )

            cand_description = normalize_description_text(c.description or "")

            # ── Free local self-refine pass (default off) ─────────────────
            # After compose + grammar normalization and BEFORE persist (so
            # before the paid review quorum), let the LOCAL compose model
            # critique its own name + description and emit an improved
            # candidate. Improve-or-no-op; the rewrite is re-validated and
            # discarded if it fails grammar. Local model → $0, so no budget
            # charge. OFF = byte-identical to the path without this block.
            if not grammar_failed and get_compose_self_refine():
                try:
                    _ref_name, _ref_desc = await _self_refine_candidate(
                        name_id,
                        cand_description,
                        c.segments.model_dump() if hasattr(c, "segments") else None,
                        source_item,
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if _ref_name != name_id:
                        wlog.info(
                            "Pool %s: self-refine %r → %r",
                            phase_tag,
                            name_id,
                            _ref_name,
                        )
                    name_id = _ref_name
                    cand_description = _ref_desc
                except Exception:
                    wlog.debug(
                        "Pool %s: self-refine failed for %r — keeping original",
                        phase_tag,
                        name_id,
                    )

            cand = {
                "id": name_id,
                "source_types": [source_kind],
                "source_id": c.source_id,
                "description": cand_description,
                "kind": c.kind,
                "source_paths": [
                    encode_source_path(source_kind, p) for p in (c.dd_paths or [])
                ],
                "reason": c.reason,
                "unit": unit,
                "physics_domain": physics_domain,
                "source_domains": source_domains,
                "cocos_transformation_type": cocos_type,
                "cocos": batch[0].get("cocos_version") if batch else None,
                "dd_version": batch[0].get("dd_version") if batch else None,
                **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
            }

            if regen:
                cand["regen_increment"] = True

            candidates.append(cand)

            # ── B3: Deterministic error-sibling minting ───────────────
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                try:
                    from imas_codex.standard_names.error_siblings import (
                        mint_error_siblings,
                    )

                    siblings = mint_error_siblings(
                        name_id,
                        error_node_ids=source_item["error_node_ids"],
                        unit=unit,
                        physics_domain=physics_domain,
                        cocos_type=cocos_type,
                        cocos_version=batch[0].get("cocos_version") if batch else None,
                        dd_version=batch[0].get("dd_version") if batch else None,
                    )
                    if siblings:
                        for s in siblings:
                            s["_from_error_sibling"] = True
                        candidates.extend(siblings)
                        wlog.debug(
                            "Pool %s: minted %d error siblings for %r",
                            phase_tag,
                            len(siblings),
                            name_id,
                        )
                except Exception:
                    wlog.debug(
                        "Pool %s: error-sibling minting failed for %r",
                        phase_tag,
                        name_id,
                        exc_info=True,
                    )

        # ── B8: Per-candidate cost / token attribution ────────────────
        # Pro-rata across LLM-generated candidates only — error siblings
        # are deterministic and carry no LLM cost.
        from datetime import UTC, datetime

        _llm_cands = [c for c in candidates if not c.get("_from_error_sibling")]
        n_cands = max(len(_llm_cands), 1)
        _llm_at = datetime.now(UTC).isoformat()
        for cand in candidates:
            if cand.get("_from_error_sibling"):
                continue
            cand["llm_cost"] = cost / n_cands
            cand["llm_model"] = model
            cand["llm_service"] = "standard-names"
            cand["llm_at"] = _llm_at
            cand["llm_tokens_in"] = tokens_in // n_cands
            cand["llm_tokens_out"] = tokens_out // n_cands
            cand["llm_tokens_cached_read"] = tokens_cache_r // n_cands
            cand["llm_tokens_cached_write"] = tokens_cache_w // n_cands

        # ── C1: Inline audits — populate validation_status ────────────
        try:
            from imas_codex.standard_names.audits import run_audits

            for cand in candidates:
                if cand.get("_from_error_sibling"):
                    cand.setdefault("validation_status", "valid")
                    continue
                src_paths = cand.get("source_paths") or []
                src_path = strip_dd_prefix(src_paths[0]) if src_paths else None
                try:
                    audit_issues = run_audits(
                        candidate=cand,
                        existing_sns_in_domain=None,
                        source_path=src_path,
                        source_cocos_type=cand.get("cocos_transformation_type"),
                    )
                except Exception:
                    audit_issues = []
                if cand.get("_grammar_retry_exhausted"):
                    audit_issues = list(audit_issues) + [
                        "audit:grammar_retry_exhausted"
                    ]
                cand["validation_issues"] = audit_issues
                cand["validation_status"] = (
                    "quarantined"
                    if _is_quarantined(list(audit_issues), {})
                    else "valid"
                )
        except Exception:
            wlog.debug("Pool %s: audits failed", phase_tag, exc_info=True)
            for cand in candidates:
                cand.setdefault("validation_status", "valid")

        # Strip private flag before persist
        for cand in candidates:
            cand.pop("_grammar_retry_exhausted", None)

        # ── B4: Vocab gaps — persist + clear sources ──────────────────
        if result.vocab_gaps:
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            gap_dicts = [
                {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                for vg in result.vocab_gaps
            ]
            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _stamp_dedup_decision(gap_dicts, _token_reuse_hits)
            try:
                await asyncio.to_thread(write_vocab_gaps, gap_dicts, source_kind)
                wlog.debug(
                    "Pool %s: persisted %d vocab gaps", phase_tag, len(gap_dicts)
                )
            except Exception:
                wlog.warning(
                    "Pool %s: write_vocab_gaps failed", phase_tag, exc_info=True
                )
            try:
                await asyncio.to_thread(
                    _update_sources_after_vocab_gap, gap_dicts, source_kind, wlog
                )
            except Exception:
                wlog.debug(
                    "Pool %s: vocab_gap source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # W29: auto-detect novel physical_base tokens.
        if result.candidates:
            try:
                auto_gaps = _auto_detect_physical_base_gaps(result.candidates)
                if auto_gaps:
                    from imas_codex.standard_names.graph_ops import write_vocab_gaps

                    await asyncio.to_thread(
                        write_vocab_gaps,
                        auto_gaps,
                        source_kind,
                        skip_segment_filter=True,
                    )
                    wlog.debug(
                        "Pool %s: auto-detected %d physical_base gaps",
                        phase_tag,
                        len(auto_gaps),
                    )
            except Exception:
                wlog.debug(
                    "Pool %s: physical_base gap detection failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── B5: Attachments — write edges + clear source claims ───────
        if result.attachments:
            try:
                attach_counts = await asyncio.to_thread(
                    _process_attachments_core, result.attachments, wlog
                )
                wlog.info(
                    "Pool %s: attached %d (rejected %d)",
                    phase_tag,
                    attach_counts.get("accepted", 0),
                    attach_counts.get("rejected", 0),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: attachment processing failed",
                    phase_tag,
                    exc_info=True,
                )
            try:
                await asyncio.to_thread(
                    _update_sources_after_attach,
                    result.attachments,
                    source_kind,
                    wlog,
                )
            except Exception:
                wlog.debug(
                    "Pool %s: attach source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── B6: Skipped sources — clear claims so they don't loop ─────
        if result.skipped:
            try:
                await asyncio.to_thread(
                    _update_sources_after_skip,
                    list(result.skipped),
                    source_kind,
                    wlog,
                )
                wlog.debug(
                    "Pool %s: marked %d sources as skipped",
                    phase_tag,
                    len(result.skipped),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: skipped source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── Persist ────────────────────────────────────────────────────
        if candidates:
            from imas_codex.standard_names.graph_ops import persist_generated_name_batch

            written = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch[0].get("dd_version"),
                cocos_version=batch[0].get("cocos_version"),
                run_id=mgr.run_id,
            )
            logger.debug("Pool %s: persisted %d candidates", phase_tag, written)

            # ── Emit per-item events ──────────────────────────────────
            if on_event is not None:
                for cand in candidates:
                    on_event(
                        {
                            "pool": "generate_name",
                            "name": cand["id"],
                            "source": (cand.get("source_paths") or [""])[0],
                            "dd_path": (cand.get("source_paths") or [""])[0],
                            "model": model,
                            "cost": cand.get("llm_cost", 0.0),
                        }
                    )

        return len(candidates)

    except Exception:
        logger.exception("Pool %s: compose batch failed", phase_tag)
        raise
    finally:
        if lease:
            lease.release_unused()


async def process_generate_name_batch(
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Pool-mode generate-name batch processor.

    Takes pre-claimed source items (dicts from the graph claim query),
    generates standard names via LLM, and persists results.

    Returns count of items successfully processed.
    """
    return await compose_batch(batch, mgr, stop_event, regen=False, on_event=on_event)


# Substrings that identify a deterministic grammar / vocab / schema failure
# in an LLM-proposed refined name.  These are NORMAL failed-refine outcomes
# (the model proposed an ungrammatical name), not crashes — the item is
# routed to the exhaust path rather than re-claimed and re-charged forever.
_GRAMMAR_FAILURE_MARKERS: tuple[str, ...] = (
    "not a registered",  # unregistered grammar token (base/qualifier/axis)
    "kind must be one of",  # legacy schema enum violation message
    "input should be",  # pydantic Literal enum violation (deterministic)
    "self-referential refined_from",  # refine produced an identical name
    "validation error for refinedname",  # pydantic RefinedName validation
)


def _is_refine_grammar_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic grammar/validation refine failure.

    Such failures (e.g. the LLM proposing an unregistered qualifier token) are
    a normal failed-refine outcome, not a crash: they must be routed to the
    exhaust path so the item stops re-claiming and re-burning paid budget.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _GRAMMAR_FAILURE_MARKERS)


def _is_refine_docs_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic ``RefinedDocs`` validation failure.

    A docs-refine that produces docs violating the ``RefinedDocs`` schema
    (e.g. consistently over/under the length bounds) is deterministic for a
    given item.  Reverting it to 'reviewed' would re-claim and re-burn paid
    budget forever, so route it to the docs-exhaust path instead.
    """
    return "validation error for refineddocs" in str(exc).lower()


async def process_refine_name_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of StandardNames for name refinement (Option B).

    For each item in the batch:
    1. Walk the REFINED_FROM chain via ``chain_history`` (already enriched).
    2. Decide whether to escalate (chain_length ≥ rotation_cap - 1).
    3. Optionally fan out targeted DD context (plan 39 Phase 1 — gated).
    4. Call LLM to produce a refined name (``RefinedName`` response model).
    5. Run the Phase 1.5 dup guard before persisting (plan 39 §5.2).
    6. Persist via ``persist_refined_name`` (new node + edge migration).
    7. On failure, release claims via ``release_refine_name_failed_claims``.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.canonical import find_name_key_duplicate
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.fanout import (
        CandidateContext,
        FanoutScope,
        assign_arm,
        load_settings as load_fanout_settings,
        run_fanout,
        should_trigger_fanout,
    )
    from imas_codex.standard_names.graph_ops import (
        _mark_refine_vocab_gap_exhausted,
        persist_refined_name,
        release_refine_name_failed_claims,
        resubmit_pinned_rename_for_review,
    )
    from imas_codex.standard_names.models import RefinedName

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    processed = 0

    # ── GraphClient lifecycle (plan 39 §10.1 I5) ─────────────────────
    # One client per cycle, reused by hybrid-neighbour search,
    # run_fanout, dup guard, and Fanout-node telemetry writes.
    # ``persist_refined_name`` opens its own client (different
    # transaction lifecycle) — that is intentional and unchanged.
    fanout_settings = load_fanout_settings()

    with GraphClient() as gc:
        for item in batch:
            if stop_event.is_set():
                break

            sn_id = item["id"]
            chain_length = item.get("chain_length", 0) or 0
            chain_history = item.get("chain_history", [])

            # ── Pinned-rename guard (never rewrite an operator's name) ──
            # A rename edit carries a fixed, operator-chosen name string.
            # Rewriting it is meaningless and destructive: re-emitting the
            # identical name trips the self-referential-refine guard and
            # decomposing a lexicalised base trips grammar validation — both
            # wrongly exhaust a correct name. Resubmit it to a fresh review
            # quorum instead (bounded), never spend an LLM rewrite on it.
            if (item.get("edit_mode") or "") == "rename":
                token = item.get("claim_token") or ""
                try:
                    outcome = await _asyncio.to_thread(
                        resubmit_pinned_rename_for_review,
                        sn_id=sn_id,
                        token=token,
                        rotation_cap=rotation_cap,
                    )
                except Exception:
                    logger.debug(
                        "refine_name: pinned-rename resubmit failed for %s", sn_id
                    )
                    outcome = ""
                logger.info(
                    "refine_name: pinned rename %s not rewritten — %s",
                    sn_id,
                    outcome or "no-op",
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": f"pinned_rename_{outcome or 'noop'}",
                            "model": "none",
                            "cost": 0.0,
                        }
                    )
                continue

            # ── Escalation decision ───────────────────────────────────
            escalate = chain_length >= rotation_cap - 1
            if escalate:
                model = DEFAULT_ESCALATION_MODEL
            else:
                # Refine tier (Sonnet 4.6 by default) — peeled off
                # [language] on 2026-05-03 after E3 acceptance audit
                # showed flash-lite refines lifted critiqued names at
                # ~5% vs ~42% for Sonnet compose.
                model = get_model("sn-refine")

            # ── Build prompt context ──────────────────────────────────
            path = item.get("source_paths", [""])[0] if item.get("source_paths") else ""

            # ── Load scored compose examples for refinement context ────
            from imas_codex.standard_names.example_loader import load_compose_examples

            try:
                _item_domain = item.get("physics_domain") or ""

                def _load_compose_examples(domain: str = _item_domain) -> list:
                    with GraphClient() as _gc:
                        return load_compose_examples(
                            _gc, physics_domains=[domain], axis="name"
                        )

                compose_scored_examples = await _asyncio.to_thread(
                    _load_compose_examples
                )
            except Exception:
                logger.debug(
                    "refine_name: scored-example load failed for %s",
                    sn_id,
                    exc_info=True,
                )
                compose_scored_examples = []

            # ── Parse vocab_gap_detail from JSON string (best-effort) ─────
            raw_vgd = item.get("vocab_gap_detail")
            vocab_gap_detail: dict | None = None
            if raw_vgd:
                try:
                    import json as _json

                    vocab_gap_detail = (
                        _json.loads(raw_vgd) if isinstance(raw_vgd, str) else raw_vgd
                    )
                except (ValueError, TypeError):
                    logger.debug(
                        "refine_name: could not parse vocab_gap_detail for %s", sn_id
                    )

            # validation_issues is stored as a list on the node directly
            validation_issues: list[str] | None = item.get("validation_issues") or None

            # Merge the cached compose context so the system prompt's
            # ``_grammar_reference.md`` include renders the closed-vocabulary
            # token map (``closed_vocab_full`` etc.). Without it the refiner
            # rewrites names with no vocabulary reference — the empty-grammar-
            # block silent gap. ``build_compose_context`` is ``_CONTEXT_CACHE``-
            # backed, so this is a dict lookup after the first call.
            from imas_codex.standard_names.context import build_compose_context

            prompt_context: dict[str, Any] = {
                **build_compose_context(),
                "item": item,
                "chain_history": chain_history,
                "chain_length": chain_length,
                "hybrid_neighbours": [],
                "fanout_evidence": "",
                "compose_scored_examples": compose_scored_examples,
                "vocab_gap_detail": vocab_gap_detail,
                "validation_issues": validation_issues,
            }

            # Attempt hybrid neighbour search (best-effort).  Uses the
            # cycle-scoped ``gc`` (plan 39 §10.1) — no fresh client.
            try:
                neighbours = _hybrid_search_neighbours(gc, path)
                prompt_context["hybrid_neighbours"] = [
                    {"path": n.get("path", ""), "description": n.get("description")}
                    for n in neighbours
                ]
            except Exception:
                logger.debug("Hybrid neighbour search failed for %s", sn_id)

            # ── DD path context (keywords, clusters, version history) ─
            if path:
                try:
                    _enrich_dd_path_context(gc, item, path)
                except Exception:
                    logger.debug(
                        "refine_name: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

            # ── Fan-out trigger gate (plan 39 §5.1) ──────────────────
            # Plumb reviewer_comments_per_dim_name from the claim batch
            # through the trigger predicate.  Gate on ALL of:
            # chain_length > 0, chain_history present (B12 enrichment),
            # at least one allow-listed dim contains a trigger keyword.
            reviewer_comments = item.get("reviewer_comments_per_dim_name")
            fanout_eligible, reviewer_excerpt = should_trigger_fanout(
                reviewer_comments_per_dim=reviewer_comments,
                chain_length=chain_length,
                chain_history=chain_history,
                keywords=fanout_settings.refine_trigger_keywords,
                dims=fanout_settings.refine_trigger_comment_dims,
                char_cap=fanout_settings.refine_trigger_comment_chars,
            )

            # ── Budget reservation (tiered, plan 39 §7.3 I1) ──────────
            # Snapshot ``original_reservation`` *before* any extension
            # so the cost-attribution invariant test can verify the
            # delta is fully accounted for via LLMCost batch_id rows.
            base_estimate = 0.20  # single-item refine
            fanout_pad = 0.0
            if (
                fanout_eligible
                and fanout_settings.enabled
                and fanout_settings.sites.get("refine_name", False)
            ):
                fanout_pad = fanout_settings.cost_estimate_for(escalate=escalate)
            estimated = base_estimate + fanout_pad
            lease = mgr.reserve(estimated, phase="refine_name")
            if lease is None:
                lease = mgr.reserve(0.0, phase="refine_name")
            original_reservation = lease.reserved if lease else 0.0

            # ── Optional fan-out (plan 39 Phase 1) ───────────────────
            fanout_evidence = ""
            fanout_run_id: str | None = None
            fanout_arm: str | None = None
            if fanout_eligible and lease is not None:
                fanout_arm = assign_arm(
                    sn_id,
                    chain_length,
                    arm_percent=fanout_settings.refine_fanout_arm_percent,
                )
                import uuid as _uuid

                fanout_run_id = str(_uuid.uuid4())
                physics_dom = item.get("physics_domain")
                if isinstance(physics_dom, list):
                    physics_dom = physics_dom[0] if physics_dom else None
                ids_filter = item.get("ids_name") or None
                candidate_ctx = CandidateContext(
                    sn_id=sn_id,
                    name=sn_id,
                    path=path,
                    description=item.get("description") or "",
                    physics_domain=physics_dom or "",
                    chain_length=chain_length,
                )
                scope = FanoutScope(
                    physics_domain=physics_dom,
                    ids_filter=ids_filter,
                )
                try:
                    fanout_evidence = await run_fanout(
                        site="refine_name",
                        candidate=candidate_ctx,
                        reviewer_excerpt=reviewer_excerpt,
                        scope=scope,
                        gc=gc,
                        parent_lease=lease,
                        settings=fanout_settings,
                        arm=fanout_arm,
                        escalate=escalate,
                        fanout_run_id=fanout_run_id,
                    )
                except Exception:
                    logger.exception(
                        "fan-out failed for %s — proceeding with empty evidence",
                        sn_id,
                    )
                    fanout_evidence = ""
            prompt_context["fanout_evidence"] = fanout_evidence

            # Load composition rules for system prompt
            from imas_codex.llm.prompt_loader import load_prompt_config as _lpc

            try:
                _rules_cfg = _lpc("sn_composition_rules")
                prompt_context["composition_rules"] = _rules_cfg.get(
                    "composition_rules", []
                )
            except Exception:
                prompt_context["composition_rules"] = []

            user_prompt = render_prompt("sn/refine_name_user", prompt_context)
            try:
                system_prompt = render_prompt("sn/refine_name_system", prompt_context)
            except Exception:
                logger.debug("refine_name: system prompt render failed for %s", sn_id)
                system_prompt = None

            # ── LLM call ──────────────────────────────────────────────
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=_messages,
                    response_model=RefinedName,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-refine"),
                )

                # acall_llm_structured returns an LLMResult that still supports
                # legacy tuple unpacking. Normalize telemetry from the real
                # result object and fall back only for older tuple-shaped tests.
                result_obj, cost, _tokens = llm_out
                (
                    llm_tokens_in,
                    llm_tokens_out,
                    llm_tokens_cached_read,
                    llm_tokens_cached_write,
                ) = _extract_llm_telemetry(
                    llm_out,
                    parsed_obj=result_obj,
                    tokens_fallback=_tokens,
                )

                # Charge cost to lease.  Stamp ``batch_id`` with the
                # ``fanout_run_id`` whenever fan-out fired so the
                # ``Fanout`` ↔ ``LLMCost`` join (plan 39 §8.3) works
                # for both arms.
                if lease:
                    _event = LLMCostEvent(
                        model=model,
                        tokens_in=llm_tokens_in,
                        tokens_out=llm_tokens_out,
                        tokens_cached_read=llm_tokens_cached_read,
                        tokens_cached_write=llm_tokens_cached_write,
                        sn_ids=(result_obj.name,),
                        phase=(
                            "refine_name+fanout" if fanout_run_id else "refine_name"
                        ),
                        service="standard-names",
                        batch_id=fanout_run_id,
                    )
                    lease.charge_event(cost, _event)

                # ── Phase 1.5 dup guard (plan 39 §5.2) ────────────
                # Deterministic name-key lookup AFTER B12 final
                # candidate, BEFORE persisting.  On hit we drop the
                # candidate and emit a ``dup_prevented`` log line.
                # Excludes ``old_name`` so the chain's predecessor is
                # never treated as a self-collision.
                dup_id = None
                try:
                    dup_id = find_name_key_duplicate(
                        gc,
                        result_obj.name,
                        exclude=sn_id,
                    )
                except Exception:
                    logger.debug(
                        "dup guard failed for %s — proceeding with persist",
                        sn_id,
                    )
                if dup_id:
                    logger.info(
                        "refine_name dup_prevented: %s → %s collides with %s",
                        sn_id,
                        result_obj.name,
                        dup_id,
                    )
                    # Release claim back to 'reviewed' so the cycle
                    # can pick it up again (with fresh feedback) or
                    # be marked superseded by manual review.
                    token = item.get("claim_token") or ""
                    try:
                        await _asyncio.to_thread(
                            release_refine_name_failed_claims,
                            sn_ids=[sn_id],
                            token=token,
                        )
                    except Exception:
                        logger.debug(
                            "release after dup_prevented failed for %s",
                            sn_id,
                        )
                    if on_event is not None:
                        on_event(
                            {
                                "pool": "refine_name",
                                "name": sn_id,
                                "old_name": sn_id,
                                "duplicate_of": dup_id,
                                "outcome": "dup_prevented",
                                "model": model,
                                "cost": cost,
                            }
                        )
                    continue

                # ── Persist ───────────────────────────────────────────
                await _asyncio.to_thread(
                    persist_refined_name,
                    old_name=sn_id,
                    new_name=result_obj.name,
                    description=normalize_description_text(result_obj.description),
                    kind=result_obj.kind,
                    unit=item.get("unit"),
                    physics_domain=(
                        (
                            (item.get("physics_domain") or [None])[0]
                            if isinstance(item.get("physics_domain"), list)
                            else item.get("physics_domain")
                        )
                        or None
                    ),
                    source_domains=(
                        item.get("source_domains")
                        if isinstance(item.get("source_domains"), list)
                        else None
                    ),
                    old_chain_length=chain_length,
                    model=model,
                    reason=result_obj.reason,
                    escalated=escalate,
                    run_id=mgr.run_id,
                )
                processed += 1
                logger.info(
                    "refine_name: %s → %s (chain_length=%d, model=%s)",
                    sn_id,
                    result_obj.name,
                    chain_length + 1,
                    model,
                )

                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": result_obj.name,
                            "old_name": sn_id,
                            "new_name": result_obj.name,
                            "chain_length": chain_length + 1,
                            "escalated": escalate,
                            "model": model,
                            "cost": cost,
                            "fanout_run_id": fanout_run_id,
                            "fanout_arm": fanout_arm,
                            "original_reservation": original_reservation,
                            "llm_tokens_in": llm_tokens_in,
                            "llm_tokens_out": llm_tokens_out,
                            "llm_tokens_cached_read": llm_tokens_cached_read,
                            "llm_tokens_cached_write": llm_tokens_cached_write,
                        }
                    )

            except Exception as exc:
                _exc_str = str(exc)
                # "no-op" means orphan_sweep already reverted this claim
                # while the LLM call was in flight — the graph is already
                # consistent; just warn and move on (no release needed).
                if "no-op" in _exc_str:
                    logger.warning(
                        "refine_name skipped (orphan_sweep beat us): %s", sn_id
                    )
                    continue
                token = item.get("claim_token") or ""
                # A grammar/validation failure (the LLM proposed an
                # ungrammatical name — e.g. an unregistered qualifier token) or
                # a self-referential refine is a NORMAL failed-refine outcome,
                # not a crash: the model keeps producing the same output for
                # this item.  Mark it exhausted (terminal) instead of reverting
                # to 'reviewed', which would re-claim and re-charge it on a paid
                # model every cycle — an infinite paid loop.  Log terminal
                # failures at WARNING; only genuinely unexpected errors get the
                # ERROR + traceback.
                is_terminal = _is_refine_grammar_failure(exc)
                if is_terminal:
                    logger.warning(
                        "refine_name failed (deterministic, marking exhausted) "
                        "for %s: %s",
                        sn_id,
                        _exc_str[:200],
                    )
                else:
                    logger.exception("refine_name failed for %s", sn_id)
                try:
                    if is_terminal:
                        await _asyncio.to_thread(
                            _mark_refine_vocab_gap_exhausted,
                            sn_id=sn_id,
                            token=token,
                            error_msg=str(exc)[:500],
                        )
                    else:
                        await _asyncio.to_thread(
                            release_refine_name_failed_claims,
                            sn_ids=[sn_id],
                            token=token,
                        )
                except Exception:
                    logger.debug(
                        "release/exhaust also failed for %s",
                        sn_id,
                    )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": "refine_failed",
                            "model": model,
                            "cost": 0.0,
                        }
                    )
            finally:
                # Always return the unused portion of the lease to the pool.
                # Without this the unspent remainder leaks every iteration
                # and the pool exhausts at ~25 % of cost_limit.
                if lease is not None:
                    try:
                        lease.release_unused()
                    except Exception:
                        pass

    return processed


# =============================================================================
# RD-quorum helper (shared by review_name + review_docs pool workers)
# =============================================================================


# In-process tally of RD-quorum reviews deferred because a ≥2-model quorum did
# NOT complete — a secondary reviewer failed (throttled or empty response), so
# only cycle 0 (or nothing) survived. Keyed by (run_id, review_axis). Accepting
# such a name on the single surviving review would silently advance it on ONE
# review when the profile demanded ≥2 — a critical correctness failure under
# provider throttling. Instead the review is deferred (the claim is released
# back to ``drafted``) and counted here so the run summary can surface the
# shortfall rather than let acceptance quietly degrade to single-model.
_quorum_incomplete_deferrals: dict[tuple[str, str], int] = {}


def _record_quorum_incomplete(run_id: str | None, review_axis: str) -> None:
    """Record one incomplete-quorum deferral for the (run_id, axis) tally."""
    key = (run_id or "", review_axis)
    _quorum_incomplete_deferrals[key] = _quorum_incomplete_deferrals.get(key, 0) + 1


def quorum_incomplete_snapshot(run_id: str | None) -> dict[str, int]:
    """Return ``{review_axis: deferrals}`` for *run_id* (empty when none)."""
    rid = run_id or ""
    return {
        axis: n for (r, axis), n in _quorum_incomplete_deferrals.items() if r == rid
    }


def reset_quorum_incomplete(run_id: str | None = None) -> None:
    """Clear incomplete-quorum tallies (all runs, or just *run_id*)."""
    if run_id is None:
        _quorum_incomplete_deferrals.clear()
        return
    for key in [k for k in _quorum_incomplete_deferrals if k[0] == (run_id or "")]:
        del _quorum_incomplete_deferrals[key]


async def _run_rd_quorum_cycles(
    *,
    sn_id: str,
    review_axis: str,
    response_model: Any,
    user_prompt: str,
    system_prompt: str,
    models: list[str],
    disagreement_threshold: float,
    rubric_dims: tuple[str, ...],
    lease: Any,
    phase: str,
    acall_llm_structured: Callable[..., Any],
    reasoning_effort: str | None = None,
    escalation_reasoning_effort: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Run the configured RD-quorum reviewer chain for a single StandardName.

    Calls each configured model in turn (cycle 0 = primary, cycle 1 =
    secondary, cycle 2 = escalator). Cycle 1 always runs when ≥2 models
    are configured. Cycle 2 runs only when (a) ≥3 models are configured
    AND (b) any rubric dimension differs between cycles 0 and 1 by more
    than *disagreement_threshold* (after normalising 0–20 scores to 0–1).

    Persists no graph state directly — the caller is responsible for
    writing the returned ``records`` list via
    :func:`~imas_codex.standard_names.graph_ops.write_reviews` and for
    the SN-side stage transition.

    Each cycle's cost is charged to *lease* via a fresh
    :class:`~imas_codex.standard_names.budget.LLMCostEvent` tagged with
    the cycle id (``c0``/``c1``/``c2``).

    Returns
    -------
    dict | None
        ``None`` when no cycle produced a parseable result (caller
        should release the claim). Otherwise a dict with keys:

        - ``records`` — list of Review record dicts (one per cycle) ready
          for :func:`write_reviews`. Share a single ``review_group_id``.
        - ``winning_score`` — float 0-1, the consensus score that should
          mirror onto the SN axis slot.
        - ``winning_scores`` — dict[str, float] of per-dim consensus.
        - ``winning_comments`` — str.
        - ``winning_comments_per_dim`` — dict | None.
        - ``canonical_model`` — str, the model attribution for the SN
          axis slot (always cycle 0's model — the chain anchor).
        - ``resolution_method`` — one of
          ``single_review`` / ``quorum_consensus`` /
          ``authoritative_escalation`` / ``max_cycles_reached``.
        - ``total_cost`` — float, sum of all cycles.
        - ``total_tokens_in`` / ``total_tokens_out`` — int.
    """
    import uuid as _uuid
    from datetime import datetime as _dt

    from imas_codex.standard_names.budget import LLMCostEvent

    review_group_id = str(_uuid.uuid4())
    cycles: list[dict[str, Any]] = []  # parsed cycle results (cycle_index 0..N)
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    async def _run_cycle(cycle_idx: int, model: str) -> dict[str, Any] | None:
        """Single LLM call. Returns parsed cycle dict or None on failure."""
        nonlocal total_cost, total_tokens_in, total_tokens_out
        # The escalator cycle (idx >= 2) fires only on a flagged disagreement,
        # so it gets a higher reasoning budget on just the contested items; the
        # base cycles run at the cheaper base effort (review is the cost-
        # dominant phase). Falls back to the base effort when unset.
        cycle_effort = (
            escalation_reasoning_effort
            if (cycle_idx >= 2 and escalation_reasoning_effort is not None)
            else reasoning_effort
        )
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
                service="standard-names",
                reasoning_effort=cycle_effort,
            )
            result_obj, cost, _tokens = llm_out
        except Exception:
            logger.exception(
                "rd_quorum %s cycle %d failed for %s (model=%s)",
                review_axis,
                cycle_idx,
                sn_id,
                model,
            )
            return None

        # Unwrap batch response: prompts ask for {"reviews": [...]}; we pass
        # one item at a time so reviews[0] is the per-item ReviewItem object.
        # A wrapper whose `reviews` is empty/None is a malformed/empty response
        # — treat it as a FAILED cycle (return None) rather than falling through
        # to `.scores` on the bare wrapper. That access raises AttributeError,
        # gets swallowed to score=0.0/tier='poor', and would persist a spurious
        # canonical review that drives a good name into refine/exhausted.
        _missing = object()
        reviews_list = getattr(result_obj, "reviews", _missing)
        if reviews_list is not _missing:
            if not reviews_list:
                logger.warning(
                    "rd_quorum %s cycle %d returned empty/no reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                return None
            try:
                result_obj = reviews_list[0]
            except (IndexError, TypeError):
                logger.warning(
                    "rd_quorum %s cycle %d returned unusable reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                return None

        tokens_in = int(getattr(llm_out, "input_tokens", 0) or 0)
        tokens_out = int(getattr(llm_out, "output_tokens", 0) or 0)
        cached_read = int(getattr(llm_out, "cache_read_tokens", 0) or 0)
        cached_write = int(getattr(llm_out, "cache_creation_tokens", 0) or 0)

        total_cost += cost
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out

        if lease is not None:
            try:
                lease.charge_event(
                    cost,
                    LLMCostEvent(
                        model=model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tokens_cached_read=cached_read,
                        tokens_cached_write=cached_write,
                        sn_ids=(sn_id,),
                        cycle=f"c{cycle_idx}",
                        phase=phase,
                        service="standard-names",
                    ),
                )
            except Exception:
                logger.debug(
                    "rd_quorum charge_event failed for %s c%d",
                    sn_id,
                    cycle_idx,
                    exc_info=True,
                )

        # Extract score and per-dim scores.
        try:
            score = float(result_obj.scores.score)
        except Exception:
            score = 0.0
        try:
            scores_dict = result_obj.scores.model_dump()
        except Exception:
            scores_dict = {}

        comments = getattr(result_obj, "reasoning", None) or ""
        comments_per_dim = None
        try:
            if getattr(result_obj, "comments", None) is not None:
                comments_per_dim = result_obj.comments.model_dump()
        except Exception:
            comments_per_dim = None

        return {
            "cycle_index": cycle_idx,
            "model": model,
            "score": score,
            "scores": scores_dict,
            "comments": comments,
            "comments_per_dim": comments_per_dim,
            "cost": cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached_read": cached_read,
            "cached_write": cached_write,
        }

    # ── Cycle 0 (primary, blind) ───────────────────────────────────────
    c0 = await _run_cycle(0, models[0])
    if c0 is not None:
        cycles.append(c0)

    # ── Cycle 1 (secondary, blind) — runs whenever ≥ 2 models ──────────
    c1 = None
    if len(models) >= 2:
        c1 = await _run_cycle(1, models[1])
        if c1 is not None:
            cycles.append(c1)

    # ── Quorum-completeness guard ──────────────────────────────────────
    # A profile with ≥2 models demands ≥2 successful reviews. If a secondary
    # reviewer failed (throttled / empty response) only cycle 0 — or nothing —
    # survived. A single surviving review must NOT be accepted for a name the
    # profile intended for multi-model adjudication, or throttling would
    # silently advance names on ONE review. Defer instead: return None so the
    # caller releases the claim back to ``drafted`` (same path it already takes
    # for a total failure), and count the deferral so the run summary can
    # surface the shortfall. Cycle 2 need not be considered — it only fires when
    # both base cycles already succeeded, which is a complete quorum.
    intended_model_count = len(models)
    successful_cycles = len(cycles)
    if intended_model_count >= 2 and successful_cycles < 2:
        logger.warning(
            "rd_quorum %s incomplete: %d/%d reviews succeeded for %s — "
            "deferring, NOT accepting on a single review",
            review_axis,
            successful_cycles,
            intended_model_count,
            sn_id,
        )
        _record_quorum_incomplete(run_id, review_axis)
        return None

    if not cycles:
        # Single-model profile whose only cycle failed — nothing to persist.
        return None

    # ── Per-dimension disagreement (cycle 2 gate) ──────────────────────
    disagreement = False
    if c0 is not None and c1 is not None:
        for dim in rubric_dims:
            s0 = float(c0["scores"].get(dim, 0)) / 20.0
            s1 = float(c1["scores"].get(dim, 0)) / 20.0
            if abs(s0 - s1) > disagreement_threshold:
                disagreement = True
                break

    # ── Cycle 2 (escalator) ────────────────────────────────────────────
    c2 = None
    if disagreement and len(models) >= 3:
        c2 = await _run_cycle(2, models[2])
        if c2 is not None:
            cycles.append(c2)

    # ── Determine winning score + resolution method ────────────────────
    canonical_model = models[0]  # SN axis attribution always cycle-0 model
    if len(cycles) == 1:
        # Reachable only for a single-model profile (len(models) == 1): a
        # ≥2-model profile with fewer than 2 successful cycles is deferred by
        # the quorum-completeness guard above and never reaches here, so a
        # single review is never a silent degradation of a multi-model quorum.
        winning = cycles[0]
        resolution_method = "single_review"
        winning_score = float(winning["score"])
        winning_scores = dict(winning["scores"])
        winning_comments = winning["comments"]
        winning_comments_per_dim = winning["comments_per_dim"]
    elif c2 is not None:
        # Escalator authoritative
        resolution_method = "authoritative_escalation"
        winning_score = float(c2["score"])
        winning_scores = dict(c2["scores"])
        winning_comments = c2["comments"]
        winning_comments_per_dim = c2["comments_per_dim"]
    else:
        # 2 cycles only (or escalator skipped/failed) — mean of c0 + c1
        winning_score = (float(c0["score"]) + float(c1["score"])) / 2.0
        winning_scores = {}
        for dim in rubric_dims:
            v0 = float(c0["scores"].get(dim, 0))
            v1 = float(c1["scores"].get(dim, 0))
            winning_scores[dim] = (v0 + v1) / 2.0
        # Merge comments preserving both reviewers' reasoning
        c0_comments = c0["comments"] or ""
        c1_comments = c1["comments"] or ""
        if c0_comments and c1_comments and c0_comments != c1_comments:
            winning_comments = f"[Primary] {c0_comments}\n[Secondary] {c1_comments}"
        else:
            winning_comments = c1_comments or c0_comments
        winning_comments_per_dim = c1["comments_per_dim"] or c0["comments_per_dim"]
        if disagreement:
            # Disputed but no escalator available
            resolution_method = "max_cycles_reached"
        else:
            resolution_method = "quorum_consensus"

    # ── Build Review records ───────────────────────────────────────────
    role_by_idx = {0: "primary", 1: "secondary", 2: "escalator"}
    method_by_idx = {0: None, 1: None, 2: None}
    if resolution_method == "single_review":
        method_by_idx[0] = "single_review"
    elif resolution_method == "authoritative_escalation":
        method_by_idx[2] = "authoritative_escalation"
    elif resolution_method == "max_cycles_reached":
        method_by_idx[1] = "max_cycles_reached"
    elif resolution_method == "quorum_consensus":
        method_by_idx[1] = "quorum_consensus"

    now_iso = _dt.utcnow().isoformat()
    records: list[dict[str, Any]] = []
    for c in cycles:
        idx = c["cycle_index"]
        score = float(c["score"])
        if score >= 0.85:
            tier = "outstanding"
        elif score >= 0.60:
            tier = "good"
        elif score >= 0.40:
            tier = "inadequate"
        else:
            tier = "poor"
        scores_json = json.dumps(c["scores"]) if c["scores"] else "{}"
        comments_per_dim_json = (
            json.dumps(c["comments_per_dim"]) if c["comments_per_dim"] else None
        )
        records.append(
            {
                "id": f"{sn_id}:{review_axis}:{review_group_id}:{idx}",
                "standard_name_id": sn_id,
                "model": c["model"],
                "reviewer_model": c["model"],
                "model_family": "other",
                "is_canonical": idx == 0,
                "score": score,
                "scores_json": scores_json,
                "tier": tier,
                "comments": c["comments"] or "",
                "comments_per_dim_json": comments_per_dim_json,
                "suggested_name": "",
                "suggestion_justification": "",
                "reviewed_at": now_iso,
                "review_axis": review_axis,
                "cycle_index": idx,
                "review_group_id": review_group_id,
                "resolution_role": role_by_idx.get(idx, "primary"),
                "resolution_method": method_by_idx.get(idx),
                "llm_model": c["model"],
                "llm_cost": c["cost"],
                "llm_tokens_in": c["tokens_in"],
                "llm_tokens_out": c["tokens_out"],
                "llm_tokens_cached_read": c["cached_read"],
                "llm_tokens_cached_write": c["cached_write"],
                "llm_at": now_iso,
                "llm_service": "standard-names",
            }
        )

    return {
        "records": records,
        "winning_score": float(winning_score),
        "winning_scores": winning_scores,
        "winning_comments": winning_comments,
        "winning_comments_per_dim": winning_comments_per_dim,
        "canonical_model": canonical_model,
        "resolution_method": resolution_method,
        "total_cost": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "review_group_id": review_group_id,
        "disagreement": disagreement,
    }


async def process_review_name_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for name review (RD-quorum).

    For each item in the batch:

    1. Build a name-axis review prompt via ``render_prompt("sn/review_names_user", ...)``.
    2. Run the configured RD-quorum chain
       (``[sn.review.names].models``):

       - **Cycle 0** (primary, blind) — always runs.
       - **Cycle 1** (secondary, blind) — runs when ≥ 2 models configured.
       - **Cycle 2** (escalator) — runs only when per-dimension disagreement
         between cycles 0 and 1 exceeds the configured threshold AND ≥ 3
         models are configured.
    3. Persist each cycle as a separate ``StandardNameReview`` node with
       proper ``cycle_index`` (0/1/2), ``review_group_id`` (UUID shared
       across cycles for the same SN), ``resolution_role``
       (primary/secondary/escalator) and ``resolution_method``
       (single_review / quorum_consensus / authoritative_escalation /
       max_cycles_reached).
    4. Persist via :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_name`
       with ``skip_review_node=True`` (review nodes already written) and the
       *winning* score so the SN axis slots mirror the consensus and the
       state machine transitions ``name_stage`` to ``'accepted'``,
       ``'reviewed'`` or ``'exhausted'``.
    5. Update review aggregates so ``review_count`` /
       ``review_disagreement`` reflect the multi-cycle group.
    6. On LLM error: release the claim via
       :func:`~imas_codex.standard_names.graph_ops.release_review_names_failed_claims`.

    Returns count of items successfully reviewed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
    )
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_name,
        release_review_names_failed_claims,
        update_review_aggregates,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    # ── Resolve review model chain ─────────────────────────────────────
    try:
        review_models = get_sn_review_names_models()
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        review_models = [DEFAULT_ESCALATION_MODEL]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""

        # ── Skip review for quarantined names ──────────────────────────
        # Names that failed ISN 3-layer validation already have a low-tier
        # verdict — running RD-quorum on them is pure waste. Persist a
        # zero-score review (cycle 0 only, single_review) directly.
        if item.get("validation_status") == "quarantined":
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=0.0,
                    scores={
                        "grammar": 0.0,
                        "semantic": 0.0,
                        "convention": 0.0,
                        "completeness": 0.0,
                    },
                    comments="quarantined: skipped reviewer LLM call",
                    comments_per_dim=None,
                    model="(skipped: quarantined)",
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_skipped_quarantined",
                            "sn_id": sn_id,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for quarantined %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # ── Semantic similarity gate (pre-LLM) ────────────────────────
        # Compute cosine similarity between name-as-text and description.
        # Names below the critical threshold are semantically ambiguous
        # (e.g. "co_passing_density" — density of what?).  Skip the
        # expensive LLM review and persist a synthetic low score that
        # routes the name to the refine_name pool.
        #
        # EXCEPT for deterministic parents: their description is the
        # canonical placeholder until ``GENERATE_DOCS`` runs, so the
        # cosine similarity is meaningless. Cosine sim between a
        # registered base token (``magnetic_field``) and a generic
        # placeholder string is always low — the gate would fire on
        # every deterministic parent that ever reached this point.
        # Deterministic parents auto-accept on the name axis anyway
        # (``seed_parent_sources`` writes ``name_stage='accepted'``);
        # this skip is defensive in case any drift puts one at
        # ``drafted``.
        from imas_codex.standard_names.audits import semantic_similarity_check
        from imas_codex.standard_names.defaults import (
            SEMANTIC_SIM_CRITICAL,
            SEMANTIC_SIM_SYNTHETIC_SCORE,
            SEMANTIC_SIM_WARNING,
        )

        sem_sim: float | None = None
        sem_issues: list[str] = []
        is_derived = item.get("origin") == "derived"
        if is_derived:
            # ── Phase 5: desc-name similarity gate for derived parents ────
            # For derived parents, the description may have been seeded from
            # a placeholder or a DD-derived string that doesn't align with
            # the name.  Compute desc_name_similarity and, if below
            # threshold, route to REFINE_DOCS instead of name scoring.
            # This keeps description-quality failures off the name axis.
            #
            # Unlike the general semantic_similarity_check (which uses the
            # quarantine-oriented SEMANTIC_SIM_CRITICAL threshold and routes
            # to REFINE_NAME), this gate uses the dedicated
            # desc_name_similarity_threshold and routes to REFINE_DOCS.
            from imas_codex.standard_names.desc_name_sim import (
                compute_desc_name_similarity,
                should_route_to_refine_docs,
            )

            desc_sim: float | None = None
            try:
                desc_sim = await _asyncio.to_thread(
                    compute_desc_name_similarity,
                    sn_id,
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: desc_name_similarity failed for derived %s",
                    sn_id,
                    exc_info=True,
                )

            if desc_sim is not None and should_route_to_refine_docs(desc_sim):
                logger.info(
                    "review_name: desc-name sim gate FIRED for derived %s "
                    "(sim=%.3f) — routing to REFINE_DOCS",
                    sn_id,
                    desc_sim,
                )
                from imas_codex.standard_names.graph_ops import mark_for_refine_docs

                try:
                    mark_for_refine_docs(
                        sn_id,
                        desc_name_similarity=desc_sim,
                        claim_token=claim_token,
                    )
                    processed += 1
                    if on_event:
                        on_event(
                            {
                                "type": "review_name_desc_sim_gate_routed_docs",
                                "sn_id": sn_id,
                                "desc_name_similarity": desc_sim,
                            }
                        )
                except Exception:
                    logger.debug(
                        "review_name: mark_for_refine_docs failed for %s",
                        sn_id,
                        exc_info=True,
                    )
                continue

            # Below threshold not triggered (sim is high enough, or embed
            # failed — treat as "gate not applicable"). Log and fall through
            # to normal name scoring.
            if desc_sim is not None:
                logger.debug(
                    "review_name: derived %s desc-name sim=%.3f (gate clear)",
                    sn_id,
                    desc_sim,
                )
        else:
            try:
                sem_sim, sem_issues = await _asyncio.to_thread(
                    semantic_similarity_check,
                    sn_id,
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: semantic_similarity_check failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # Persist semantic_sim on the node regardless of outcome
        if sem_sim is not None:
            try:
                from imas_codex.graph.client import GraphClient as _GC

                def _store_sim(_id=sn_id, _sim=sem_sim) -> None:
                    with _GC() as gc:
                        gc.query(
                            "MATCH (sn:StandardName {id: $id}) "
                            "SET sn.semantic_sim = $sim",
                            id=_id,
                            sim=_sim,
                        )

                await _asyncio.to_thread(_store_sim)
            except Exception:
                pass  # best-effort

        # Critical: skip LLM review, force into refine pipeline
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_CRITICAL:
            logger.info(
                "review_name: semantic gate FAILED for %s (sim=%.3f < %.2f) — "
                "routing to refine",
                sn_id,
                sem_sim,
                SEMANTIC_SIM_CRITICAL,
            )
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=SEMANTIC_SIM_SYNTHETIC_SCORE,
                    scores={
                        "grammar": 15.0 / 20.0,
                        "semantic": 2.0 / 20.0,
                        "convention": 15.0 / 20.0,
                        "completeness": 5.0 / 20.0,
                    },
                    comments=(
                        f"semantic_similarity_gate: sim={sem_sim:.3f} below "
                        f"critical {SEMANTIC_SIM_CRITICAL:.2f}. Name is "
                        f"semantically ambiguous — a reader cannot determine "
                        f"the measured quantity from the name alone."
                    ),
                    comments_per_dim={
                        "semantic": (
                            f"Name-to-description embedding similarity is "
                            f"{sem_sim:.3f}, well below the {SEMANTIC_SIM_CRITICAL:.2f} "
                            f"threshold. The name does not stand alone."
                        ),
                    },
                    model="(semantic_similarity_gate)",
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_semantic_gate_failed",
                            "sn_id": sn_id,
                            "semantic_sim": sem_sim,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for semantic-gated %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # Inject warning into item context for the reviewer
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_WARNING:
            sem_warning_note = (
                f"⚠️ SEMANTIC AMBIGUITY WARNING: Name-to-description "
                f"embedding similarity is {sem_sim:.3f} (threshold "
                f"{SEMANTIC_SIM_WARNING:.2f}). Verify that the name is "
                f"self-describing — can someone determine the measured "
                f"quantity from the name alone?"
            )
            item["semantic_warning"] = sem_warning_note

        # ── Build prompt context ───────────────────────────────────────
        from imas_codex.standard_names.context import (
            _build_enum_lists,
            fetch_review_neighbours,
        )

        try:
            neighbours = fetch_review_neighbours(item)
        except Exception:
            logger.debug(
                "review_name: neighbour fetch failed for %s", sn_id, exc_info=True
            )
            neighbours = {
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            }

        # ── Load scored examples for reviewer calibration ──────────────
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.example_loader import load_review_examples

        try:
            _item_domain = item.get("physics_domain") or ""

            def _load_review_examples(domain: str = _item_domain) -> list:
                with GraphClient() as _gc:
                    return load_review_examples(
                        _gc, physics_domains=[domain], axis="name"
                    )

            review_scored_examples = await _asyncio.to_thread(_load_review_examples)
        except Exception:
            logger.debug(
                "review_name: scored-example load failed for %s", sn_id, exc_info=True
            )
            review_scored_examples = []

        # ── DD path context for reviewer (keywords, clusters, version history) ─
        _source_paths = item.get("source_paths") or []
        if _source_paths:
            _first_src = (
                _source_paths[0] if isinstance(_source_paths, list) else _source_paths
            )
            if isinstance(_first_src, str):
                _first_src = strip_dd_prefix(_first_src)
            if _first_src:
                try:
                    from imas_codex.graph.client import GraphClient as _GCDD

                    def _do_dd_ctx(_p=_first_src, _it=item) -> None:
                        with _GCDD() as _gc:
                            _enrich_dd_path_context(_gc, _it, _p)

                    await _asyncio.to_thread(_do_dd_ctx)
                except Exception:
                    logger.debug(
                        "review_name: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        prompt_context: dict[str, Any] = {
            "items": [item],
            **neighbours,
            **_build_enum_lists(),
            "review_scored_examples": review_scored_examples,
        }
        try:
            user_prompt = render_prompt("sn/review_names_user", prompt_context)
            system_prompt = render_prompt("sn/review_names_system", prompt_context)
        except Exception:
            logger.debug("review_name: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Review the standard name: {item.get('id', sn_id)}\n"
                f"Description: {item.get('description', '')}"
            )
            system_prompt = (
                "You are a quality reviewer for IMAS standard names in "
                "fusion plasma physics."
            )

        # ── Budget reservation (cover all cycles) ──────────────────────
        # Per-item cost ~$0.05; reserve worst-case across the chain
        # (every model called) with a 1.3× safety margin.
        per_item_estimate = 0.05
        worst_case = per_item_estimate * len(review_models) * 1.3
        lease = mgr.reserve(worst_case, phase="review_name")
        if lease is None:
            lease = mgr.reserve(0.0, phase="review_name")

        # ── RD-quorum cycles ──────────────────────────────────────────
        try:
            quorum = await _run_rd_quorum_cycles(
                sn_id=sn_id,
                review_axis="names",
                response_model=StandardNameQualityReviewNameOnlyBatch,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                models=review_models,
                disagreement_threshold=disagreement_threshold,
                rubric_dims=("grammar", "semantic", "convention", "completeness"),
                lease=lease,
                phase="review_name",
                acall_llm_structured=acall_llm_structured,
                reasoning_effort=get_sn_review_reasoning_effort(),
                escalation_reasoning_effort=get_sn_review_escalation_reasoning_effort(),
                run_id=mgr.run_id,
            )

            if quorum is None:
                # No cycles produced a valid result — release claim.
                if claim_token:
                    try:
                        await _asyncio.to_thread(
                            release_review_names_failed_claims,
                            sn_ids=[sn_id],
                            claim_token=claim_token,
                            from_stage="drafted",
                            to_stage="drafted",
                        )
                    except Exception:
                        logger.debug(
                            "release_review_names_failed_claims also failed for %s",
                            sn_id,
                        )
                continue

            # ── Persist all cycle Review records ──────────────────────
            from imas_codex.standard_names.graph_ops import write_reviews

            await _asyncio.to_thread(write_reviews, quorum["records"])

            # ── Stage transition with winning score ────────────────────
            new_stage = await _asyncio.to_thread(
                persist_reviewed_name,
                sn_id=sn_id,
                claim_token=claim_token,
                score=quorum["winning_score"],
                scores=quorum["winning_scores"],
                comments=quorum["winning_comments"],
                comments_per_dim=quorum["winning_comments_per_dim"],
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
            )

            # ── Update aggregates so review_count / disagreement reflect group ─
            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_name: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_name",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_name: %s persist no-op (token mismatch?)", sn_id)

        except Exception:
            logger.exception("review_name failed for %s", sn_id)
            token = item.get("claim_token") or ""
            if token:
                try:
                    await _asyncio.to_thread(
                        release_review_names_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=token,
                        from_stage="drafted",
                        to_stage="drafted",
                    )
                except Exception:
                    logger.debug(
                        "release_review_names_failed_claims also failed for %s",
                        sn_id,
                    )
        finally:
            # Always return the unused portion of the lease to the pool.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# =============================================================================
# DD context enrichment for generate_docs
# =============================================================================

_DD_PATH_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH n, collect(DISTINCT {
    label: c.label, description: c.description, scope: c.scope
})[0..3] AS clusters
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n)
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
WITH n, clusters, parent,
     collect(DISTINCT {change_id: vc.id, change_type: vc.change_type}) AS changes
RETURN n.keywords AS keywords,
       n.documentation AS dd_documentation,
       n.description AS dd_description,
       n.units AS dd_units,
       parent.description AS parent_description,
       clusters,
       changes
"""

_DOCS_GEN_ENRICH_QUERY = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
RETURN sn.source_paths AS source_paths,
       sn.cocos_transformation_type AS cocos_label,
       sn.origin AS origin,
       collect(DISTINCT {
           id: imas.id,
           documentation: coalesce(imas.documentation, ''),
           description: coalesce(imas.description, ''),
           alias: imas.alias,
           unit: coalesce(imas.unit, '')
       }) AS dd_nodes
"""

# Children grounding for a derived-parent docs item. A derived parent is an
# abstraction over these concrete instances; generate_docs must GENERALISE over
# them (describe the shared quantity), not over-specialise to any one child.
_DOCS_GEN_PARENT_CHILDREN_QUERY = """
MATCH (c:StandardName)-[:HAS_PARENT]->(p:StandardName {id: $sn_id})
WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
RETURN c.id AS name,
       c.description AS description,
       c.unit AS unit,
       c.physics_domain AS physics_domain
ORDER BY c.id LIMIT 12
"""

_DOCS_GEN_NEARBY_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['accepted', 'approved']
  AND sn.physics_domain = $domain
  AND sn.id <> $sn_id
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS id,
       sn.description AS description,
       sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit
LIMIT 20
"""


def _enrich_dd_path_context(gc: Any, item: dict, source_path: str) -> None:
    """Enrich an item dict with DD path context (keywords, clusters, version history).

    Fetches lightweight context from the IMASNode for *source_path* and populates:

    - ``dd_keywords``: list of keyword strings from the IMASNode.
    - ``dd_clusters``: list of ``{label, description, scope}`` dicts.
    - ``dd_version_history``: list of ``{version, change_type}`` dicts.
    - ``dd_parent_description``: description of the parent structure node.

    Best-effort — failures are debug-logged and the item is left without
    the missing key.  Modifies *item* in-place.
    """
    try:
        rows = list(gc.query(_DD_PATH_CONTEXT_QUERY, path=source_path))
    except Exception:
        logger.debug(
            "_enrich_dd_path_context: query failed for %s", source_path, exc_info=True
        )
        return
    if not rows:
        return
    row = rows[0]

    # Keywords
    kw = row.get("keywords")
    if kw:
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        if kw:
            item["dd_keywords"] = kw

    # DD ground truth — the authoritative definition the reviewer verifies
    # documentation claims against (physics_accuracy grounding).
    dd_doc = row.get("dd_documentation")
    if dd_doc:
        item["dd_documentation"] = dd_doc
    dd_desc = row.get("dd_description")
    if dd_desc:
        item["dd_description"] = dd_desc
    dd_units = row.get("dd_units")
    if dd_units:
        item["dd_units"] = dd_units

    # Parent description
    parent_desc = row.get("parent_description")
    if parent_desc:
        item["dd_parent_description"] = parent_desc

    # Clusters
    clusters_raw = row.get("clusters") or []
    clusters = [
        {
            "label": c["label"],
            "description": c.get("description") or "",
            "scope": c.get("scope") or "",
        }
        for c in clusters_raw
        if c.get("label")
    ]
    if clusters:
        item["dd_clusters"] = clusters

    # Version history
    changes = row.get("changes") or []
    version_history = []
    for ch in changes:
        change_id = ch.get("change_id") or ""
        change_type = ch.get("change_type") or ""
        parts = change_id.rsplit(":", 1)
        version = parts[-1] if len(parts) >= 2 else ""
        if version and change_type:
            version_history.append({"version": version, "change_type": change_type})
    if version_history:
        item["dd_version_history"] = version_history


def _enrich_for_docs_gen(
    gc: Any, items: list[dict], cocos_params: dict | None = None
) -> None:
    """Enrich generate_docs items with DD source context.

    Populates per-item keys (all guarded by ``if`` in the prompt template):

    - ``source_paths``: bare DD paths (``dd:`` prefix stripped).
    - ``dd_source_docs``: list of ``{id, documentation, unit}`` from linked
      :class:`~imas_codex.graph.models.IMASNode` nodes.
    - ``dd_aliases``: list of alias strings from IMASNodes.
    - ``related_neighbours``: graph-relationship neighbours derived from the
      first DD source path (via :func:`_related_path_neighbours`).
    - ``nearest_peers``: vector-similar :class:`StandardName` neighbours from
      :func:`_search_nearby_names` formatted as ``{tag, unit, physics_domain,
      doc_short, cocos_label}``.
    - ``cocos_label``: COCOS transformation type from the SN node.
    - ``cocos_guidance``: rendered sign convention guidance for the LLM.
    - ``sibling_family``: the HAS_PARENT sibling family (parent, anchor,
      siblings) from :func:`~imas_codex.standard_names.context.fetch_sibling_family`
      — drives the parallel-structure directive in the docs prompts.

    Modifies *items* in-place.  Never raises — individual item failures are
    debug-logged and the item is left without the missing key.
    """
    for item in items:
        sn_id = item.get("id")
        if not sn_id:
            continue

        # ── 1. Source paths + linked IMASNode documentation ───────────
        try:
            rows = list(gc.query(_DOCS_GEN_ENRICH_QUERY, sn_id=sn_id))
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: graph query failed for %s",
                sn_id,
                exc_info=True,
            )
            continue

        if not rows:
            continue
        row = rows[0]

        raw_paths = row.get("source_paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        source_paths = [
            strip_dd_prefix(p) for p in raw_paths if p and isinstance(p, str)
        ]
        if source_paths:
            item["source_paths"] = source_paths

        dd_nodes = [n for n in (row.get("dd_nodes") or []) if n and n.get("id")]
        if dd_nodes:
            item["dd_source_docs"] = [
                {
                    "id": n["id"],
                    # Rich-first: the LLM-enriched description grounds the docs
                    # prompts; fall back to terse source documentation only when
                    # no enriched description exists.
                    "documentation": (
                        n.get("description") or n.get("documentation") or ""
                    ),
                    "unit": n.get("unit") or "",
                }
                for n in dd_nodes[:5]
            ]
            aliases = [n["alias"] for n in dd_nodes if n.get("alias")]
            if aliases:
                item["dd_aliases"] = aliases

            # Surface the DD ancestor lineage: the primary source leaf is often
            # a terse template stub, while the quantity's physics + evaluation
            # locus live on a parent quantity node up the HAS_PARENT chain
            # (e.g. "…rotation velocity … at the pedestal top"). Mirrors the
            # name-generation grounding so docs ground on the real physics/locus
            # instead of the stub. Nearest rich ancestor first; capped.
            primary = dd_nodes[0]["id"]
            try:
                anc_rows = list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=primary))
            except Exception:
                anc_rows = []
            lineage: list[dict[str, str]] = []
            seen_anc: set[str] = set()
            for a in anc_rows:
                text = (a.get("description") or "").strip() or (
                    a.get("documentation") or ""
                ).strip()
                if not text or text in seen_anc:
                    continue
                seen_anc.add(text)
                lineage.append({"path": a.get("path") or "", "text": text})
                if len(lineage) >= 4:
                    break
            if lineage:
                item["ancestor_context"] = lineage

        # ── 1a. Derived-parent children grounding ─────────────────────
        # A derived parent has no DD source of its own; its concrete physics
        # lives in its accepted children. Inject them so generate_docs writes a
        # GENERALISED parent doc (the shared quantity) rather than over-
        # specialising to one child's component/qualifier (observed defect:
        # ``perturbed_velocity`` documented as its single ``normalized_parallel``
        # child). The placeholder description is dropped (null) so only real
        # child meaning grounds the prompt.
        if row.get("origin") == "derived":
            try:
                from imas_codex.standard_names.defaults import (
                    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
                )

                kid_rows = list(gc.query(_DOCS_GEN_PARENT_CHILDREN_QUERY, sn_id=sn_id))
                children = [
                    {
                        "name": k["name"],
                        "description": (
                            None
                            if k.get("description")
                            == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
                            else k.get("description")
                        ),
                        "unit": k.get("unit"),
                        "physics_domain": k.get("physics_domain"),
                    }
                    for k in kid_rows
                ]
                if children:
                    item["derived_children"] = children
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: derived children fetch failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 1b. COCOS sign convention context ─────────────────────────
        cocos_label = row.get("cocos_label")
        if cocos_label:
            item["cocos_label"] = cocos_label
            if cocos_params:
                try:
                    from imas_codex.standard_names.context import (
                        render_cocos_guidance,
                    )

                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, cocos_params
                    )
                except Exception:
                    logger.debug(
                        "_enrich_for_docs_gen: cocos guidance render failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        # ── 2. Graph-relationship neighbours from first DD source path ─
        if source_paths:
            try:
                related = _related_path_neighbours(gc, source_paths[0])
                if related:
                    item["related_neighbours"] = related
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: related_neighbours failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 2b. DD path context (keywords, clusters, version history) ─
        if source_paths:
            try:
                _enrich_dd_path_context(gc, item, source_paths[0])
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: dd_path_context failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 3. Vector-similar SN neighbours ───────────────────────────
        description = item.get("description") or sn_id.replace("_", " ")
        try:
            peers_raw = _search_nearby_names(description, k=6)
            peers = [
                {
                    "tag": f"name:{p['id']}",
                    "unit": p.get("unit") or "",
                    "physics_domain": p.get("physics_domain") or "",
                    "doc_short": (p.get("description") or "")[:120],
                    "cocos_label": p.get("cocos_label") or "",
                }
                for p in peers_raw
                if p.get("id") and p.get("id") != sn_id
            ][:5]
            if peers:
                item["nearest_peers"] = peers
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: nearest_peers failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 4. Parent/child component context ─────────────────────────
        try:
            # Check if this SN is a component of a parent
            parent_rows = list(
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $sn_id})-[r:HAS_PARENT]->(parent:StandardName)
                    RETURN parent.id AS name,
                           parent.description AS description,
                           parent.documentation AS documentation,
                           r.axis AS axis
                    LIMIT 1
                    """,
                    sn_id=sn_id,
                )
            )
            if parent_rows:
                p = parent_rows[0]
                item["parent_sn"] = {
                    "name": p["name"],
                    "description": p.get("description") or "",
                    "documentation": p.get("documentation") or "",
                }
                item["component_axis"] = p.get("axis") or ""

            # Check if this SN has children (is a parent)
            child_rows = list(
                gc.query(
                    """
                    MATCH (child:StandardName)-[r:HAS_PARENT]->(sn:StandardName {id: $sn_id})
                    RETURN child.id AS name,
                           child.description AS description,
                           r.axis AS axis
                    ORDER BY child.id
                    """,
                    sn_id=sn_id,
                )
            )
            if child_rows:
                from imas_codex.standard_names.families import sort_by_axis_convention

                child_dicts = [
                    {
                        "name": c["name"],
                        "description": c.get("description") or "",
                        "axis": c.get("axis") or "",
                    }
                    for c in child_rows
                ]
                item["child_components"] = sort_by_axis_convention(child_dicts)
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: component context failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 5. Derivative / base-quantity context ─────────────────────────
        try:
            from imas_codex.standard_names.families import DD_DERIVATIVE_MAP

            parent_sn_data = item.get("parent_sn")
            if parent_sn_data:
                parent_name = parent_sn_data["name"]
                # Query parent unit and sibling derivatives sharing same parent
                bq_rows = list(
                    gc.query(
                        """
                        MATCH (parent:StandardName {id: $parent_name})
                        OPTIONAL MATCH (sib:StandardName)-[:HAS_PARENT]->(parent)
                        WHERE sib.id <> $sn_id
                        WITH parent, collect(sib.id) AS siblings
                        RETURN parent.unit AS unit, siblings
                        """,
                        parent_name=parent_name,
                        sn_id=sn_id,
                    )
                )
                if bq_rows:
                    bq_row = bq_rows[0]
                    item["base_quantity"] = {
                        "name": parent_name,
                        "description": parent_sn_data.get("description") or "",
                        "documentation": parent_sn_data.get("documentation") or "",
                        "unit": bq_row.get("unit") or "",
                    }
                    if sn_id in DD_DERIVATIVE_MAP:
                        numerator, denominator = DD_DERIVATIVE_MAP[sn_id]
                        raw_siblings = bq_row.get("siblings") or []
                        siblings = [s for s in raw_siblings if s]
                        item["derivative_context"] = {
                            "numerator": numerator,
                            "denominator": denominator,
                            "siblings": siblings,
                        }
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: derivative context failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 6. Sibling-family (parallel-structure) context ─────────────
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id, gc=gc)
            if family and family.get("siblings"):
                item["sibling_family"] = family
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: sibling family failed for %s",
                sn_id,
                exc_info=True,
            )


def _nearby_names_for_docs_gen(gc: Any, items: list[dict]) -> list[dict]:
    """Return accepted StandardNames in the same physics domain(s) as *items*.

    Queries up to 20 accepted SNs per unique domain found across the batch.
    Results are deduplicated and capped at 40 total.  Used to populate the
    batch-level ``nearby_existing_names`` context key in
    :func:`process_generate_docs_batch`.
    """
    domains: set[str] = set()
    for item in items:
        dom = item.get("physics_domain")
        if isinstance(dom, list):
            domains.update(dom)
        elif dom:
            domains.add(dom)

    nearby: list[dict] = []
    seen_ids: set[str] = set()
    batch_ids: set[str] = {item["id"] for item in items if item.get("id")}

    for domain in sorted(domains):
        # Use the first batch item id as the exclusion anchor; the query
        # already uses != so any id from the batch suffices.
        anchor = next(iter(batch_ids), "")
        try:
            rows = list(gc.query(_DOCS_GEN_NEARBY_QUERY, domain=domain, sn_id=anchor))
        except Exception:
            logger.debug(
                "_nearby_names_for_docs_gen: query failed for domain=%s",
                domain,
                exc_info=True,
            )
            continue
        for r in rows:
            nid = r.get("id")
            if nid and nid not in seen_ids and nid not in batch_ids:
                seen_ids.add(nid)
                nearby.append(dict(r))
                if len(nearby) >= 40:
                    break
        if len(nearby) >= 40:
            break

    return nearby


async def process_generate_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of accepted StandardNames for generate_docs (P4.1).

    For each item in the batch:

    1. Enrich items with DD source context via :func:`_enrich_for_docs_gen`
       and :func:`_nearby_names_for_docs_gen` (source_paths, dd_source_docs,
       dd_aliases, nearest_peers, related_neighbours, nearby_existing_names).
    2. Render prompt via ``render_prompt("sn/generate_docs_user", {...})`` with
       reviewer feedback (reviewer_score_name, reviewer_comments_name),
       chain history, and the new DD context as context.
    3. Use ``get_model("sn-docs")`` — docs generation model.
    4. Call ``acall_llm_structured`` with ``service="standard-names"`` and
       response_model=``GeneratedDocs``.
    5. Reserve budget and charge ``LLMCostEvent``.
    6. Persist via ``persist_generated_docs`` (transitions docs_stage → 'drafted').
    7. On error: ``release_generate_docs_failed_claims``.
    8. Stream per-item progress: name + first 100 chars of generated description.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.graph_ops import (
        persist_generated_docs,
        release_generate_docs_failed_claims,
    )
    from imas_codex.standard_names.models import GeneratedDocs

    model = get_model("sn-docs")
    processed = 0

    # ── Enrich batch with DD context (source paths, docs, peers) ──────────
    nearby_existing_names: list[dict] = []
    compose_scored_examples: list[dict] = []
    try:

        def _do_enrich() -> list[dict]:
            with GraphClient() as gc:
                # Load COCOS singleton params for sign guidance rendering
                cocos_params = None
                try:
                    from imas_codex.settings import get_dd_version

                    dd_version = get_dd_version()
                    cocos_rows = list(
                        gc.query(
                            """
                            MATCH (dv:DDVersion {id: $ver})-[:COCOS]->(c:COCOS)
                            RETURN properties(c) AS cocos_params
                            """,
                            ver=dd_version,
                        )
                    )
                    if cocos_rows:
                        cocos_params = cocos_rows[0]["cocos_params"]
                except Exception:
                    logger.debug(
                        "process_generate_docs_batch: COCOS params load failed",
                        exc_info=True,
                    )
                _enrich_for_docs_gen(gc, batch, cocos_params=cocos_params)
                return _nearby_names_for_docs_gen(gc, batch)

        nearby_existing_names = await _asyncio.to_thread(_do_enrich)
    except Exception:
        logger.debug("process_generate_docs_batch: enrichment failed", exc_info=True)

    # ── Load scored examples for docs generation ──────────────────────
    try:
        from imas_codex.standard_names.example_loader import load_compose_examples

        batch_domains = list(
            {
                item.get("physics_domain", "")
                for item in batch
                if item.get("physics_domain")
            }
        )

        def _load_docs_scored_examples() -> list[dict]:
            with GraphClient() as gc:
                return load_compose_examples(
                    gc, physics_domains=batch_domains, axis="docs"
                )

        compose_scored_examples = await _asyncio.to_thread(_load_docs_scored_examples)
        if compose_scored_examples:
            logger.info(
                "generate_docs: Loaded %d scored examples (domains=%s)",
                len(compose_scored_examples),
                batch_domains or "all",
            )
    except Exception:
        logger.debug(
            "process_generate_docs_batch: scored examples load failed",
            exc_info=True,
        )

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        chain_history = item.get("chain_history") or []
        # Fix D: the docs user template reads ``item.chain_history`` — ensure it
        # is present on the item (it is fetched into the local var, but the item
        # dict may not carry it), so refined-docs see predecessor history.
        item["chain_history"] = chain_history

        # ── Build prompt context ───────────────────────────────────────
        # Merge the cached compose context so the docs SYSTEM prompt's
        # ``_grammar_reference.md`` include renders the closed-vocabulary map
        # (otherwise the grammar block is empty). Cached after first call.
        from imas_codex.standard_names.context import (
            build_compose_context,
            locus_context_for,
        )

        prompt_context: dict[str, Any] = {
            **build_compose_context(),
            "item": item,
            "chain_history": chain_history,
            "nearby_existing_names": nearby_existing_names,
            "compose_scored_examples": compose_scored_examples,
            # Locus-defining cross-link context (PR-9): the ISN-registry gloss +
            # position-defining standard quantity for this name's locus, so the
            # docs prompt can link e.g. *_at_pedestal_top ->
            # normalized_poloidal_flux_coordinate_of_pedestal. None when the name
            # has no locus or the locus carries no gloss/defining quantity.
            "locus_context": locus_context_for(sn_id),
        }

        try:
            user_prompt = render_prompt("sn/generate_docs_user", prompt_context)
        except Exception:
            logger.debug("generate_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Generate description and documentation for the IMAS standard name: "
                f"{sn_id}\nKind: {item.get('kind', 'scalar')}\n"
                f"Unit: {item.get('unit', '')}"
            )

        try:
            system_prompt = render_prompt("sn/generate_docs_system", prompt_context)
        except Exception:
            logger.debug("generate_docs: system prompt render failed for %s", sn_id)
            system_prompt = None

        # ── Budget reservation ─────────────────────────────────────────
        estimated = 0.20
        lease = mgr.reserve(estimated, phase="generate_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="generate_docs")

        # ── LLM call ──────────────────────────────────────────────────
        try:
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=GeneratedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-docs"),
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="generate_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────────
            await _asyncio.to_thread(
                persist_generated_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_description_text(result_obj.description),
                documentation=normalize_prose_spelling(result_obj.documentation),
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            # ── Per-item progress ──────────────────────────────────────
            # Keep log preview short; pass full description to on_event so
            # terminal-aware clipping can use the available display width.
            _desc_log = result_obj.description[:100]
            logger.info(
                "\033[32mgenerate_docs\033[0m: %s — %s",
                sn_id,
                _desc_log,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "generate_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as _doc_exc:
            # A "token mismatch or node not found" persist error is an EXPECTED
            # race, not a failure: the orphan sweep reclaimed the item (docs
            # generation can outrun the claim TTL) or another worker took it.
            # The name is not lost — it stays at its prior stage and is
            # re-claimed. Log it quietly; reserve the full traceback for
            # genuinely unexpected errors.
            if "token mismatch or node not found" in str(_doc_exc):
                logger.warning(
                    "generate_docs claim lost for %s (reclaimed mid-generation) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
                logger.exception("generate_docs failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_generate_docs_failed_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug(
                    "release_generate_docs_failed_claims also failed for %s",
                    sn_id,
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


async def process_enrich_parents_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Synthesize real descriptions for placeholder derived parents.

    A derived parent materialised by ``_materialize_derived_parent_rows`` keeps
    the deterministic placeholder description until something writes a real one.
    That placeholder deadlocks it: REVIEW_NAME drops it (no real description),
    so it never earns a ``reviewer_score_name``, so generate_docs drops it too.
    This worker breaks the deadlock — for each claimed parent:

    1. Fetch the parent's live children (the concrete instances it abstracts
       over) for grounding.
    2. Render the children-grounded enrichment prompt and call
       ``get_model("sn-parent-enrich")`` (compose-tier; generate_docs rewrites
       the full documentation downstream).
    3. Charge the budget lease.
    4. Embed the synthesised description locally (free) so REVIEW_NAME's
       semantic-similarity check has an embedding immediately.
    5. Persist via :func:`persist_enriched_parent` (writes description +
       embedding, routes ``name_stage`` to ``'drafted'``).

    Returns the count of parents successfully enriched.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.embeddings.description import embed_description
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.graph_ops import (
        fetch_derived_parent_children,
        persist_enriched_parent,
        release_enrich_parents_claims,
    )
    from imas_codex.standard_names.models import EnrichedParentDescription

    model = get_model("sn-parent-enrich")
    processed = 0

    # ── Fetch grounding children for the whole batch (one round-trip) ─────
    try:
        children_by_parent = await _asyncio.to_thread(
            fetch_derived_parent_children, [item["id"] for item in batch]
        )
    except Exception:
        logger.debug(
            "process_enrich_parents_batch: children fetch failed", exc_info=True
        )
        children_by_parent = {}

    # Static grammar/context block (cached after first render) so the system
    # prompt's grammar include renders the closed-vocabulary map.
    _base_ctx = build_compose_context()

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        # The claim returns id as the StandardName id (== the name string).
        item.setdefault("name", sn_id)
        children = children_by_parent.get(sn_id, [])

        # A parent with no live children cannot be grounded — skip (the claim
        # gate already requires ≥1 child, so this is defensive). Release it.
        if not children:
            logger.debug("enrich_parents: %s has no live children — releasing", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                pass
            continue

        prompt_context: dict[str, Any] = {
            **_base_ctx,
            "item": item,
            "children": children,
        }
        try:
            user_prompt = render_prompt("sn/enrich_parent_user", prompt_context)
        except Exception:
            logger.debug("enrich_parents: user prompt render failed for %s", sn_id)
            _child_lines = "\n".join(
                f"- {c.get('name')}: {c.get('description') or ''}" for c in children
            )
            user_prompt = (
                f"Describe the derived parent standard name '{sn_id}' "
                f"(unit: {item.get('unit') or '—'}, kind: {item.get('kind') or 'scalar'}) "
                f"by generalising over its children:\n{_child_lines}"
            )
        try:
            system_prompt = render_prompt("sn/enrich_parent_system", prompt_context)
        except Exception:
            logger.debug("enrich_parents: system prompt render failed for %s", sn_id)
            system_prompt = None

        # ── Budget reservation ─────────────────────────────────────────
        estimated = 0.05
        lease = mgr.reserve(estimated, phase="enrich_parents")
        if lease is None:
            lease = mgr.reserve(0.0, phase="enrich_parents")

        try:
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=EnrichedParentDescription,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-parent-enrich"),
            )
            result_obj, cost, _tokens = llm_out

            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="enrich_parents",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            description = normalize_description_text(
                (result_obj.description or "").strip()
            )

            # Embed locally (free) so REVIEW_NAME has an embedding immediately.
            embedding = await _asyncio.to_thread(embed_description, description)

            await _asyncio.to_thread(
                persist_enriched_parent,
                sn_id=sn_id,
                claim_token=claim_token,
                description=description,
                embedding=embedding,
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            if on_event is not None:
                on_event(
                    {
                        "pool": "enrich_parents",
                        "name": sn_id,
                        "description": description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as _enrich_exc:
            if "token mismatch or node not found" in str(_enrich_exc):
                logger.warning(
                    "enrich_parents claim lost for %s (reclaimed mid-enrich) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
                logger.exception("enrich_parents failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug("release_enrich_parents_claims also failed for %s", sn_id)
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


async def process_review_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for docs review (RD-quorum).

    Mirrors :func:`process_review_name_batch` for the docs axis: runs the
    full ``[sn.review.docs].models`` chain (cycle 0 + cycle 1 unconditionally,
    cycle 2 only when per-dim disagreement exceeds threshold). Persists each
    cycle as a ``StandardNameReview`` node and transitions ``docs_stage`` via
    :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_docs`.

    Returns count of items successfully reviewed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_model,
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
    )
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_docs,
        release_review_docs_failed_claims,
        update_review_aggregates,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewDocsBatch,
        StandardNameQualityReviewDocsParentBatch,
    )

    # ── Resolve docs review chain ──────────────────────────────────────
    try:
        review_models = get_sn_review_docs_models()
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        try:
            review_models = get_sn_review_names_models()
        except (ValueError, IndexError):
            review_models = []
    if not review_models:
        # Refine tier (Sonnet 4.6) — defensive fallback only.
        review_models = [get_model("sn-refine")]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    processed = 0

    # ── Derived-parent children for parent-aware docs review ───────────
    # A derived parent is an abstraction over its children; the docs reviewer
    # must judge it on positioning + generalisation (does it correctly capture
    # the common quantity?), NOT as a standalone specific name. Inject the
    # children (origin='derived' only) so the review_docs_user parent block can
    # reframe the rubric — otherwise the quorum penalises abstractions for
    # lacking child-level specifics (observed: parent docs systematically <0.85).
    parent_children: dict[str, list] = {}
    try:
        from imas_codex.graph.client import GraphClient as _GCKids
        from imas_codex.standard_names.defaults import (
            DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER as _PH_DESC,
        )

        def _fetch_parent_children() -> dict[str, list]:
            with _GCKids() as _gc:
                rows = _gc.query(
                    """
                    MATCH (p:StandardName)
                    WHERE p.id IN $ids AND p.origin = 'derived'
                    MATCH (c:StandardName)-[:HAS_PARENT]->(p)
                    WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
                    WITH p, c ORDER BY c.id
                    RETURN p.id AS pid, collect({
                        name: c.id,
                        description: CASE WHEN c.description = $ph
                            THEN null ELSE c.description END,
                        unit: c.unit,
                        physics_domain: c.physics_domain
                    })[..12] AS children
                    """,
                    ids=[it["id"] for it in batch],
                    ph=_PH_DESC,
                )
                return {r["pid"]: list(r["children"]) for r in rows}

        parent_children = await _asyncio.to_thread(_fetch_parent_children)
    except Exception:
        logger.debug("review_docs: parent children fetch failed", exc_info=True)

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        if parent_children.get(sn_id):
            item["derived_children"] = parent_children[sn_id]

        # ── Build prompt context ───────────────────────────────────────
        from imas_codex.standard_names.context import fetch_review_neighbours

        try:
            neighbours = fetch_review_neighbours(item)
        except Exception:
            logger.debug(
                "review_docs: neighbour fetch failed for %s", sn_id, exc_info=True
            )
            neighbours = {
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            }

        # ── Load scored examples for reviewer calibration ──────────────
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.example_loader import load_review_examples

        try:
            _item_domain = item.get("physics_domain") or ""

            def _load_review_examples(domain: str = _item_domain) -> list:
                with GraphClient() as _gc:
                    return load_review_examples(
                        _gc, physics_domains=[domain], axis="docs"
                    )

            review_scored_examples = await _asyncio.to_thread(_load_review_examples)
        except Exception:
            logger.debug(
                "review_docs: scored-example load failed for %s", sn_id, exc_info=True
            )
            review_scored_examples = []

        # ── DD path context for docs reviewer ──────────────────────────
        _source_paths = item.get("source_paths") or []
        if _source_paths:
            _first_src = (
                _source_paths[0] if isinstance(_source_paths, list) else _source_paths
            )
            if isinstance(_first_src, str):
                _first_src = strip_dd_prefix(_first_src)
            if _first_src:
                try:
                    from imas_codex.graph.client import GraphClient as _GCDD

                    def _do_dd_ctx(_p=_first_src, _it=item) -> None:
                        with _GCDD() as _gc:
                            _enrich_dd_path_context(_gc, _it, _p)

                    await _asyncio.to_thread(_do_dd_ctx)
                except Exception:
                    logger.debug(
                        "review_docs: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        prompt_context: dict[str, Any] = {
            "item": item,
            **neighbours,
            "review_scored_examples": review_scored_examples,
        }
        # Derived parents are reviewed against the PARENT rubric (distinct
        # dimension set: generalization / positioning / physics_accuracy /
        # clarity), not the standalone-name docs rubric.
        _is_parent = bool(item.get("derived_children"))
        _user_tmpl = (
            "sn/review_docs_parent_user" if _is_parent else "sn/review_docs_user"
        )
        _system_tmpl = (
            "sn/review_docs_parent_system" if _is_parent else "sn/review_docs_system"
        )
        try:
            user_prompt = render_prompt(_user_tmpl, prompt_context)
            system_prompt = render_prompt(_system_tmpl, prompt_context)
        except Exception:
            logger.debug("review_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Review the documentation for standard name: {item.get('id', sn_id)}\n"
                f"Description: {item.get('description', '')}\n"
                f"Documentation: {item.get('documentation', '')}"
            )
            system_prompt = (
                "You are a quality reviewer for IMAS standard name "
                "documentation in fusion plasma physics."
            )

        # ── Budget reservation (cover all cycles) ──────────────────────
        per_item_estimate = 0.05
        worst_case = per_item_estimate * len(review_models) * 1.3
        lease = mgr.reserve(worst_case, phase="review_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="review_docs")

        # Derived parents use a SINGLE-model review (the chain anchor), not the
        # full RD-quorum: a structural abstraction reviewed against the
        # parent-aware rubric does not need multi-model adjudication, and the
        # quorum is the cost-dominant phase. One call (resolution_method=
        # 'single_review') cuts parent docs-review cost ~2-3x.
        _item_review_models = review_models[:1] if _is_parent else review_models
        _response_model = (
            StandardNameQualityReviewDocsParentBatch
            if _is_parent
            else StandardNameQualityReviewDocsBatch
        )
        _rubric_dims = (
            ("generalization", "positioning", "physics_accuracy", "clarity")
            if _is_parent
            else (
                "description_quality",
                "documentation_quality",
                "completeness",
                "physics_accuracy",
            )
        )

        try:
            quorum = await _run_rd_quorum_cycles(
                sn_id=sn_id,
                review_axis="docs",
                response_model=_response_model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                models=_item_review_models,
                disagreement_threshold=disagreement_threshold,
                rubric_dims=_rubric_dims,
                lease=lease,
                phase="review_docs",
                acall_llm_structured=acall_llm_structured,
                reasoning_effort=get_sn_review_reasoning_effort(),
                escalation_reasoning_effort=get_sn_review_escalation_reasoning_effort(),
                run_id=mgr.run_id,
            )

            if quorum is None:
                if claim_token:
                    try:
                        await _asyncio.to_thread(
                            release_review_docs_failed_claims,
                            sn_ids=[sn_id],
                            claim_token=claim_token,
                            from_stage="drafted",
                            to_stage="drafted",
                        )
                    except Exception:
                        logger.debug(
                            "release_review_docs_failed_claims also failed for %s",
                            sn_id,
                        )
                continue

            from imas_codex.standard_names.graph_ops import write_reviews

            await _asyncio.to_thread(write_reviews, quorum["records"])

            new_stage = await _asyncio.to_thread(
                persist_reviewed_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                score=quorum["winning_score"],
                scores=quorum["winning_scores"],
                comments=quorum["winning_comments"],
                comments_per_dim=quorum["winning_comments_per_dim"],
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
            )

            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_docs: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_docs",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_docs: %s persist no-op (token mismatch?)", sn_id)

        except Exception:
            logger.exception("review_docs failed for %s", sn_id)
            token = item.get("claim_token") or ""
            if token:
                try:
                    await _asyncio.to_thread(
                        release_review_docs_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=token,
                        from_stage="drafted",
                        to_stage="drafted",
                    )
                except Exception:
                    logger.debug(
                        "release_review_docs_failed_claims also failed for %s",
                        sn_id,
                    )
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# =============================================================================
# process_refine_docs_batch — DocsRevision snapshot + in-place docs update
# =============================================================================


async def process_refine_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of reviewed StandardNames for docs refinement (P4.3).

    For each item in the batch:

    1. Read ``docs_chain_length``, decide model (escalation on final attempt).
    2. Render prompt with current docs, reviewer feedback, and chain history.
    3. Call LLM to produce refined docs (``RefinedDocs`` response model).
    4. Persist via ``persist_refined_docs`` — snapshots old docs into
       ``DocsRevision``, updates SN in-place, advances chain.
    5. On failure: release claims via ``release_refine_docs_failed_claims``.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        _mark_refine_docs_exhausted,
        persist_refined_docs,
        release_refine_docs_failed_claims,
    )
    from imas_codex.standard_names.models import RefinedDocs

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        docs_chain_length = item.get("docs_chain_length", 0) or 0
        docs_chain_history = item.get("docs_chain_history", [])

        # ── Escalation decision ───────────────────────────────────
        escalate = docs_chain_length >= rotation_cap - 1
        if escalate:
            model = DEFAULT_ESCALATION_MODEL
        else:
            # Refine tier (Sonnet 4.6 by default) — see refine_name
            # comment + 2026-05-03 E3 acceptance audit.
            model = get_model("sn-refine")

        # ── Build prompt context ──────────────────────────────────
        prompt_context: dict[str, Any] = {
            "sn_name": sn_id,
            "description": item.get("description", ""),
            "documentation": item.get("documentation", ""),
            "kind": item.get("kind", "scalar"),
            "unit": item.get("unit", ""),
            "physics_domain": item.get("physics_domain", ""),
            "docs_chain_length": docs_chain_length,
            "docs_chain_history": docs_chain_history,
            "reviewer_score_docs": item.get("reviewer_score_docs"),
            "reviewer_comments_per_dim_docs": item.get(
                "reviewer_comments_per_dim_docs"
            ),
            "dd_paths": [],
            # Expert edit steering (imas-codex sn edit) — the refine_docs_user
            # template reads these as top-level context vars, not item.*,
            # because this prompt_context is a hand-built whitelist rather
            # than an `item` merge.
            "docs_hint": item.get("docs_hint"),
            "edit_reason": item.get("edit_reason"),
            "edit_origin": item.get("edit_origin"),
        }

        # Best-effort DD path enrichment
        try:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                dd_rows = gc.query(
                    """
                    MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: $sn_id})
                    WHERE n:IMASNode
                    RETURN n.id AS path, n.ids AS ids,
                           n.unit AS unit,
                           n.documentation AS documentation,
                           n.description AS description
                    LIMIT 5
                    """,
                    sn_id=sn_id,
                )
                prompt_context["dd_paths"] = [dict(r) for r in dd_rows]
        except Exception:
            logger.debug("refine_docs: DD path enrichment failed for %s", sn_id)

        # Parent-aware refine: inject children (origin='derived' only) so the
        # rewrite GENERALISES over them rather than re-hugging one child — the
        # parent-aware reviewer dock that routes single-child parents here is
        # for over-specialisation, which only children-grounding can fix.
        try:
            from imas_codex.graph.client import GraphClient as _GCRefKids
            from imas_codex.standard_names.defaults import (
                DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER as _PH_REF,
            )

            with _GCRefKids() as gc:
                krows = gc.query(
                    """
                    MATCH (p:StandardName {id: $sn_id}) WHERE p.origin = 'derived'
                    MATCH (c:StandardName)-[:HAS_PARENT]->(p)
                    WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
                    WITH c ORDER BY c.id
                    RETURN c.id AS name,
                           CASE WHEN c.description = $ph THEN null
                                ELSE c.description END AS description,
                           c.unit AS unit, c.physics_domain AS physics_domain
                    LIMIT 12
                    """,
                    sn_id=sn_id,
                    ph=_PH_REF,
                )
                kids = [dict(r) for r in krows]
                if kids:
                    prompt_context["derived_children"] = kids
        except Exception:
            logger.debug("refine_docs: derived children fetch failed for %s", sn_id)

        # Sibling-family context so the rewrite converges on the family's
        # parallel documentation template rather than drifting further.
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id)
            if family and family.get("siblings"):
                prompt_context["sibling_family"] = family
        except Exception:
            logger.debug("refine_docs: sibling family fetch failed for %s", sn_id)

        # Locus-defining cross-link context (PR-9), same as generate_docs.
        try:
            from imas_codex.standard_names.context import locus_context_for

            prompt_context["locus_context"] = locus_context_for(sn_id)
        except Exception:
            logger.debug("refine_docs: locus context fetch failed for %s", sn_id)

        try:
            user_prompt = render_prompt("sn/refine_docs_user", prompt_context)
        except Exception:
            logger.exception("refine_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Refine the documentation for standard name: {sn_id}\n"
                f"Current description: {item.get('description', '')}\n"
                f"Current documentation: {item.get('documentation', '')}\n"
                f"Reviewer feedback: {item.get('reviewer_comments_docs', '')}"
            )

        try:
            system_prompt = render_prompt("sn/refine_docs_system", prompt_context)
        except Exception:
            logger.debug("refine_docs: system prompt render failed for %s", sn_id)
            system_prompt = None

        # ── Budget reservation ─────────────────────────────────────
        estimated = 0.20
        lease = mgr.reserve(estimated, phase="refine_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="refine_docs")

        # ── LLM call ──────────────────────────────────────────────
        _messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": user_prompt}]
        )
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=RefinedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-refine"),
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="refine_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────
            await _asyncio.to_thread(
                persist_refined_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_description_text(result_obj.description),
                documentation=normalize_prose_spelling(result_obj.documentation),
                model=model,
                current_description=item.get("description") or "",
                current_documentation=item.get("documentation") or "",
                current_model=item.get("docs_model"),
                current_generated_at=(
                    str(item["docs_generated_at"])
                    if item.get("docs_generated_at")
                    else None
                ),
                reviewer_score_to_snapshot=item.get("reviewer_score_docs"),
                reviewer_comments_to_snapshot=item.get("reviewer_comments_docs"),
                reviewer_comments_per_dim_to_snapshot=item.get(
                    "reviewer_comments_per_dim_docs"
                ),
                run_id=mgr.run_id,
            )
            processed += 1

            desc_preview = result_obj.description[:80]
            logger.info(
                "\033[35mrefine_docs\033[0m: %s — %s (chain=%d, model=%s)",
                sn_id,
                desc_preview,
                docs_chain_length + 1,
                model,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "revision": docs_chain_length + 1,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as exc:
            # A deterministic ``RefinedDocs`` validation failure (docs that
            # consistently violate the schema length bounds) is a NORMAL
            # failed-refine outcome, not a crash — the model keeps producing
            # the same invalid docs for this item.  Mark it exhausted instead
            # of reverting to 'reviewed', which would re-claim and re-charge it
            # on a paid model every cycle (an infinite paid loop).  Log such
            # failures at WARNING; only unexpected errors get ERROR + traceback.
            is_terminal = _is_refine_docs_failure(exc)
            if is_terminal:
                logger.warning(
                    "refine_docs failed (deterministic, marking exhausted) for %s: %s",
                    sn_id,
                    str(exc)[:200],
                )
            else:
                logger.exception("refine_docs failed for %s", sn_id)
            try:
                if is_terminal:
                    await _asyncio.to_thread(
                        _mark_refine_docs_exhausted,
                        sn_id=sn_id,
                        token=claim_token,
                        error_msg=str(exc)[:500],
                    )
                else:
                    await _asyncio.to_thread(
                        release_refine_docs_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=claim_token,
                    )
            except Exception:
                logger.debug(
                    "release/exhaust refine_docs also failed for %s",
                    sn_id,
                )
            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "outcome": "refine_failed",
                        "model": model,
                        "cost": 0.0,
                    }
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# ═══════════════════════════════════════════════════════════════════════
# Embed worker — batch-embed StandardName nodes
# ═══════════════════════════════════════════════════════════════════════


async def process_embed_batch(
    items: list[dict[str, Any]],
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    on_event: Callable | None = None,
) -> int:
    """Batch-embed StandardName nodes.

    No LLM cost — uses embedding server only.  Does not charge budget.
    """
    from imas_codex.embeddings.description import embed_descriptions_batch
    from imas_codex.standard_names.graph_ops import (
        _compute_embed_hash,
        persist_embed_batch,
    )

    if stop_event.is_set():
        return 0

    # Build embed texts
    for item in items:
        desc = item.get("description")
        sn_id = item["id"]
        item["_embed_text"] = f"{sn_id} — {desc}" if desc else sn_id
        item["embed_text_hash"] = _compute_embed_hash(sn_id, desc)

    # Batch embed (no LLM cost — uses embedding server)
    await asyncio.to_thread(embed_descriptions_batch, items, text_field="_embed_text")

    # Filter to those that got embeddings
    to_persist = [
        {
            "id": it["id"],
            "embedding": it["embedding"],
            "embed_text_hash": it["embed_text_hash"],
        }
        for it in items
        if it.get("embedding") is not None
    ]

    if to_persist:
        written = await asyncio.to_thread(persist_embed_batch, to_persist)
    else:
        written = 0

    # Emit events for display
    if on_event:
        for it in to_persist:
            on_event(
                {
                    "pool": "embed_name",
                    "name": it["id"],
                }
            )

    return written
