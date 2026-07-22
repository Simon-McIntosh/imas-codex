"""Grammar segment classification — open vs closed vocabulary.

The ISN grammar distinguishes *closed* segments (fixed vocabulary — any token
outside the list is a real vocabulary gap) from *open* segments (free-form
compounds — any novel token is legitimate by design).

The LLM composer and our segment-edge writer both occasionally emit "missing
token" reports on open segments, which pollutes the ``VocabGap`` node
population with nonsensical entries.  The ISN release process then has to
manually filter these out.  This module is the single source of truth used by
codex to decide whether a reported gap is real.

Open segments are derived from ``SEGMENT_TOKEN_MAP`` in the installed
imas-standard-names package: any segment with an empty token list is treated
as open.  The LLM composer also reports structural ambiguity via a pseudo
segment ``grammar_ambiguity`` — these are grammar findings, not missing
tokens, and are likewise filtered.

When the ISN package is unavailable at import time we fall back to a
conservative empty set so all real-segment gaps are preserved.  Since
ISN rc21+, ``physical_base`` is intended to be a closed vocabulary; it is
therefore no longer in the fallback.
"""

from __future__ import annotations

from functools import lru_cache

# Hard fallback (used when imas-standard-names is unavailable at import time).
# ISN rc21+ closes ``physical_base``; no segment is guaranteed open by default.
_FALLBACK_OPEN_SEGMENTS: frozenset[str] = frozenset()

# Pseudo segments reported by the composer but that are not real grammar
# segments — these are structural findings, not missing tokens.  Treated as
# "open" for VocabGap filtering purposes.
PSEUDO_SEGMENTS: frozenset[str] = frozenset({"grammar_ambiguity"})

# Sentinel indicating ISN is not available — distinct from an empty set.
_ISN_UNAVAILABLE: frozenset[str] | None = None


def _load_segment_token_map() -> dict[str, tuple[str, ...]] | None:
    """Load the ISN SEGMENT_TOKEN_MAP, returning None if ISN is unavailable."""
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        return SEGMENT_TOKEN_MAP
    except ImportError:
        return None


# Synthetic segment label under which operator tokens are reported as "known".
# Operators are not a SEGMENT_TOKEN_MAP segment — they are a distinct grammar
# mechanism — so this label never appears in a real standard name; it exists
# only so the gap classifier can say "this token is a known operator, in the
# wrong slot" rather than "absent".
_OPERATOR_SEGMENT = "operator"


@lru_cache(maxsize=1)
def _operator_tokens() -> frozenset[str]:
    """Return the ISN operator vocabulary (derived from the grammar, not hardcoded).

    Operators (``flux_surface_averaged``, ``line_integrated``, ``normalized``,
    ``square``, ``derivative_with_respect_to``, ``gradient`` …) compose into
    names through the operator rendering engine (``<op>_of_<base>`` / postfix)
    rather than occupying a ``SEGMENT_TOKEN_MAP`` slot.  A composer that reports
    one as a missing *segment* token has mis-slotted a known operator, not found
    a genuine vocabulary gap — so the classifier must recognise these to avoid
    fabricating ``absent`` gaps (and retiring the source) for existing grammar.

    Returns an empty set when ISN is unavailable so real-segment gaps are
    preserved.
    """
    try:
        from imas_standard_names import get_grammar_context

        ctx = get_grammar_context()
        ops = ctx.get("grammar", {}).get("vocabularies", {}).get("operators", {})
        return frozenset(ops.keys())
    except Exception:
        return frozenset()


# Segments whose validity the ISN grammar resolves via lexical-compound
# matching rather than a flat token list.  For these, a token may be a valid
# base even when it is absent from ``SEGMENT_TOKEN_MAP`` (e.g.
# ``internal_inductance``, ``major_radius`` resolve to themselves through the
# parser).  All other segments are genuinely closed enums.
_PARSER_RESOLVED_SEGMENTS: frozenset[str] = frozenset(
    {"physical_base", "geometric_base"}
)


@lru_cache(maxsize=4096)
def resolved_base_segment(token: str) -> str | None:
    """Return the base segment the ISN grammar resolves *token* to, if self-resolving.

    The flat ``SEGMENT_TOKEN_MAP['physical_base']`` lists only the registered
    *atomic* base tokens.  The grammar additionally accepts lexical compounds
    (``internal_inductance``, ``major_radius``, ``minor_radius`` …) that
    ``parse_standard_name`` resolves to themselves but that never appear in the
    flat map.  Such compounds are valid bases, not vocabulary gaps.

    Returns the segment name (``"physical_base"`` or ``"geometric_base"``) when
    ``parse_standard_name(token)`` succeeds AND yields *token* itself as that
    base.  Returns ``None`` when the token decomposes to a *different* base
    (e.g. ``poloidal_magnetic_flux`` → ``magnetic_flux``), when the parser
    rejects it (genuine gap → ``UnknownBaseTokenError``), or when ISN is
    unavailable.  Decomposable / absent tokens are left to the surrounding
    classifier.

    Cached because it is called per gap during reconcile and per candidate
    during compose auto-detection.
    """
    if not token:
        return None
    try:
        from imas_standard_names.grammar import parse_standard_name
    except ImportError:
        return None
    try:
        parsed = parse_standard_name(token)
    except Exception:
        # UnknownBaseTokenError (genuine gap) or any parse failure.
        return None
    if getattr(parsed, "physical_base", None) == token:
        return "physical_base"
    if getattr(parsed, "geometric_base", None) == token:
        return "geometric_base"
    return None


def is_known_physical_base(token: str) -> bool:
    """Return True if the ISN grammar resolves *token* as a base in its own right.

    Thin boolean wrapper over :func:`resolved_base_segment` for callers that
    only need a yes/no on physical_base membership (e.g. compose
    auto-detection).
    """
    return resolved_base_segment(token) == "physical_base"


@lru_cache(maxsize=1)
def known_segments() -> frozenset[str] | None:
    """Return all valid ISN grammar segment names, or None if ISN unavailable.

    Includes both open and closed segments.  Use ``is_valid_segment()`` for
    per-segment checks.  Returns ``None`` when the ISN package cannot be
    imported — callers must handle this case conservatively.
    """
    stm = _load_segment_token_map()
    if stm is None:
        return _ISN_UNAVAILABLE
    try:
        return frozenset(stm.keys())
    except Exception:  # pragma: no cover — defensive
        return _ISN_UNAVAILABLE


def is_valid_segment(segment: str | None) -> bool:
    """Return True if *segment* is a recognized ISN grammar or pseudo segment.

    When ISN is unavailable, returns ``True`` conservatively so gaps are
    preserved rather than silently dropped.
    """
    if not segment:
        return False
    if segment in PSEUDO_SEGMENTS:
        return True
    segs = known_segments()
    if segs is None:
        return True  # ISN unavailable — assume valid to avoid data loss
    return segment in segs


@lru_cache(maxsize=1)
def open_segments() -> frozenset[str]:
    """Return the set of ISN grammar segments with no registered tokens.

    With ISN rc53+ all segments are closed (have registered tokens), so
    this should return an empty frozenset.  Retained as a runtime check
    against ISN regressions.
    """
    stm = _load_segment_token_map()
    if stm is None:
        return _FALLBACK_OPEN_SEGMENTS

    try:
        return frozenset(seg for seg, tokens in stm.items() if not tokens)
    except Exception:  # pragma: no cover — defensive
        return _FALLBACK_OPEN_SEGMENTS


def is_open_segment(segment: str | None) -> bool:
    """Return ``True`` if ``segment`` has no registered tokens or is a pseudo segment.

    Gaps reported against such segments should never materialise as
    :class:`VocabGap` nodes. With ISN rc53+ all real segments are closed,
    so only pseudo segments (``grammar_ambiguity``) return True.
    """
    if not segment:
        return False
    if segment in PSEUDO_SEGMENTS:
        return True
    return segment in open_segments()


@lru_cache(maxsize=1)
def _segment_token_index() -> dict[str, list[str]]:
    """Build a reverse index: token → list of segment names that contain it.

    Only closed-vocabulary segments (non-empty token lists in
    ``SEGMENT_TOKEN_MAP``) are indexed.  Open segments are excluded
    because every token is admissible there by definition.
    """
    stm = _load_segment_token_map()
    if stm is None:
        return {}

    index: dict[str, list[str]] = {}
    try:
        for seg, tokens in stm.items():
            if not tokens:
                continue  # open-vocabulary segment
            for tok in tokens:
                index.setdefault(tok, []).append(seg)
    except Exception:  # pragma: no cover — defensive
        return {}
    return index


def is_known_token(token: str) -> list[str]:
    """Return the closed-vocabulary segment names whose vocab contains *token*.

    Case-sensitive match against every closed segment in the ISN
    ``SEGMENT_TOKEN_MAP``.  Returns ``[]`` when the token is absent
    from all closed segments (either a true gap or an open-segment
    term).

    Multiple segments may be returned when the token legitimately
    appears in more than one closed vocabulary (e.g. orientation /
    qualifier overlap).

    For ``physical_base`` / ``geometric_base`` the flat map under-reports:
    the grammar accepts lexical compounds (``internal_inductance``,
    ``major_radius`` …) that resolve to themselves through
    ``parse_standard_name`` yet never appear in ``SEGMENT_TOKEN_MAP``.  Such
    a token is reported as known for the segment the parser resolves it to,
    so it is correctly classified ``false_positive`` rather than ``absent``.
    """
    found = list(_segment_token_index().get(token, []))
    # Augment with the parser-resolved base segment when the grammar accepts
    # the token as a self-resolving lexical-compound base absent from the flat
    # map (e.g. internal_inductance, major_radius).
    base_seg = resolved_base_segment(token)
    if base_seg is not None and base_seg not in found:
        found.append(base_seg)
    # Augment with the operator vocabulary (a grammar mechanism outside
    # SEGMENT_TOKEN_MAP): a token that is a known operator is not an absent gap
    # — the composer mis-slotted it, so it classifies as wrong-slot placement.
    if token in _operator_tokens() and _OPERATOR_SEGMENT not in found:
        found.append(_OPERATOR_SEGMENT)
    return found


def classify_gap(segment: str, token: str) -> tuple[str, list[str]]:
    """Classify a single vocabulary gap against the current ISN installation.

    Returns ``(category, actual_segments)`` where:

    - ``"false_positive"`` — token exists in the reported segment
    - ``"invalid_segment"`` — reported segment is not in ISN grammar
    - ``"open_segment"`` — reported segment has open vocabulary
    - ``"wrong_slot_placement"`` — token exists in exactly one other segment
    - ``"ambiguous_known_token"`` — token exists in multiple other segments
    - ``"decomposable"`` — compound token whose parts exist in other segments
    - ``"absent"`` — token is not in any closed segment (genuine gap)
    """
    if not is_valid_segment(segment):
        return "invalid_segment", []

    if is_open_segment(segment):
        return "open_segment", []

    segments_found = is_known_token(token)

    if segment in segments_found:
        return "false_positive", segments_found

    if not segments_found:
        # Before declaring absent, check if compound can be decomposed
        decomp_segs = _check_decomposable(token)
        if decomp_segs:
            return "decomposable", decomp_segs
        return "absent", []

    if len(segments_found) > 1:
        return "ambiguous_known_token", segments_found

    return "wrong_slot_placement", segments_found


# Gap categories that are NOT genuine vocabulary deficiencies: the token
# already exists (here or in another segment), decomposes into existing
# tokens, or sits in an open-vocabulary segment.  Only an ``absent``
# closed-segment gap warrants an ISN vocabulary addition — or retiring the
# source that reported it.
NON_ACTIONABLE_GAP_CATEGORIES: frozenset[str] = frozenset(
    {
        "false_positive",
        "invalid_segment",
        "open_segment",
        "wrong_slot_placement",
        "ambiguous_known_token",
        "decomposable",
    }
)


def is_actionable_gap(segment: str | None, token: str) -> bool:
    """Whether a reported gap names a genuinely-absent closed-segment token.

    True iff :func:`classify_gap` returns ``"absent"`` — the one category that
    both justifies an ISN vocabulary addition and warrants retiring the source
    to ``vocab_gap``.  Every other category is a composer mis-report (token in
    the wrong slot, decomposable into existing tokens, ambiguous, or a false
    positive) or an open-vocabulary segment: the source is still nameable, so
    it must not be stranded.
    """
    if not segment:
        return False
    return classify_gap(segment, token)[0] == "absent"


# Lexicalized physics compounds that must NOT be decomposed even though
# their prefixes match registered tokens.  These are single, irreducible
# physical concepts in the ISN physical_base registry.
ATOMIC_COMPOUNDS: frozenset[str] = frozenset(
    {
        "poloidal_flux",
        "poloidal_magnetic_flux",
        "magnetic_flux",
        "minor_radius",
        "major_radius",
        "cross_sectional_area",
        "safety_factor",
        "polarization_angle",
        "ellipticity_angle",
        "loop_voltage",
        "internal_inductance",
        "magnetic_field",
        "electric_field",
        "current_density",
        "power_density",
        "energy_density",
        "particle_flux",
        "heat_flux",
        "rotation_frequency",
        "magnetic_shear",
        "torque_density",
        "collisionality",
        "bootstrap_current",
    }
)


def _check_decomposable(token: str) -> list[str]:
    """Check if a compound token can be decomposed into existing vocabulary.

    Uses bounded left-to-right longest-prefix matching against all segment
    registries.  Returns the list of segments where parts were found, or
    empty list if the token cannot be decomposed.

    Skips tokens in :data:`ATOMIC_COMPOUNDS` to avoid false negatives on
    lexicalized physics terms.
    """
    if token in ATOMIC_COMPOUNDS:
        return []

    if "_" not in token:
        return []

    parts = token.split("_")
    if len(parts) < 2:
        return []

    index = _segment_token_index()
    if not index:
        return []

    # Try to cover ALL parts with registered tokens (greedy longest-prefix)
    matched_segments: list[str] = []
    i = 0
    while i < len(parts):
        found = False
        # Try longest prefix first (3-token, 2-token, 1-token)
        for width in range(min(3, len(parts) - i), 0, -1):
            candidate = "_".join(parts[i : i + width])
            segs = index.get(candidate, [])
            if segs:
                matched_segments.extend(segs)
                i += width
                found = True
                break
        if not found:
            return []  # Uncovered part — not fully decomposable

    # Only report decomposable if we matched tokens from ≥2 segments
    unique_segs = list(dict.fromkeys(matched_segments))
    if len(set(unique_segs)) >= 2:
        return unique_segs
    # Single-segment decomposition (e.g. two qualifiers) — still decomposable
    # if the compound doesn't exist as a registered token itself
    if unique_segs:
        return unique_segs

    return []


def filter_closed_segment_gaps(
    gaps: list[dict],
    *,
    segment_key: str = "segment",
) -> tuple[list[dict], list[dict]]:
    """Split gap records into (closed, open) by their grammar segment.

    ``gaps`` is a list of dicts with at least a ``segment`` key.  Returns the
    tuple ``(kept, dropped)`` — ``kept`` is emitted as ``VocabGap`` nodes,
    ``dropped`` is logged and discarded.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for g in gaps:
        if is_open_segment(g.get(segment_key)):
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


__all__ = [
    "ATOMIC_COMPOUNDS",
    "NON_ACTIONABLE_GAP_CATEGORIES",
    "PSEUDO_SEGMENTS",
    "classify_gap",
    "filter_closed_segment_gaps",
    "is_actionable_gap",
    "is_known_physical_base",
    "is_known_token",
    "is_open_segment",
    "is_valid_segment",
    "known_segments",
    "open_segments",
    "resolved_base_segment",
]
