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
    """Return the set of ISN grammar segments with open vocabulary.

    A segment is considered *open* when its closed-vocabulary token list is
    empty — any token is admissible by design (``physical_base`` is the
    canonical example).  Emitting a :class:`VocabGap` for such a segment is
    nonsensical.

    Results are memoised across the process lifetime because
    ``SEGMENT_TOKEN_MAP`` is immutable and cheap-but-not-free to introspect.
    """
    stm = _load_segment_token_map()
    if stm is None:
        return _FALLBACK_OPEN_SEGMENTS

    try:
        return frozenset(seg for seg, tokens in stm.items() if not tokens)
    except Exception:  # pragma: no cover — defensive
        return _FALLBACK_OPEN_SEGMENTS


def is_open_segment(segment: str | None) -> bool:
    """Return ``True`` if ``segment`` is open-vocab or a pseudo segment.

    Gaps reported against open or pseudo segments should never materialise as
    :class:`VocabGap` nodes: they do not indicate a missing closed-vocabulary
    token.
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
    """
    return list(_segment_token_index().get(token, []))


def classify_gap(segment: str, token: str) -> tuple[str, list[str]]:
    """Classify a single vocabulary gap against the current ISN installation.

    Returns ``(category, actual_segments)`` where:

    - ``"false_positive"`` — token exists in the reported segment
    - ``"invalid_segment"`` — reported segment is not in ISN grammar
    - ``"open_segment"`` — reported segment has open vocabulary
    - ``"wrong_slot_placement"`` — token exists in exactly one other segment
    - ``"ambiguous_known_token"`` — token exists in multiple other segments
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
        return "absent", []

    if len(segments_found) > 1:
        return "ambiguous_known_token", segments_found

    return "wrong_slot_placement", segments_found


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
    "PSEUDO_SEGMENTS",
    "classify_gap",
    "filter_closed_segment_gaps",
    "is_known_token",
    "is_open_segment",
    "is_valid_segment",
    "known_segments",
    "open_segments",
]
