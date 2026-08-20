"""Query tokenisation and tier policy for standard-name search.

Pure, dependency-light helpers used by the three-stream RRF in
``imas_codex.standard_names.search``.

The grammar stream partitions the ISN segments into three tiers and
applies tier-dependent RRF weights; :func:`filter_by_tier_policy`
documents the eligibility rules.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from packaging.version import Version

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

#: ISN connector stopwords. Carry no segment information; only inflate
#: keyword recall when left in the token stream.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {"of", "at", "from", "in", "to", "for", "the", "a", "an"}
)


_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def order_grammar_versions(versions: Iterable[str]) -> list[str]:
    """Return unique PEP 440 grammar versions from newest to oldest.

    Grammar snapshots use package versions, so string ordering is not valid:
    lexicographically, ``0.8.0rc9`` sorts after ``0.8.0rc66``.  Parsing with
    :class:`packaging.version.Version` preserves release-candidate, development,
    and local-version ordering.
    """
    return sorted(set(versions), key=Version, reverse=True)


def select_grammar_version(
    rows: Iterable[Mapping[str, Any]],
    *,
    preferred_version: str | None = None,
) -> str | None:
    """Select a token snapshot using runtime, active, then semantic ordering.

    An exact runtime snapshot is strongest because it is the installed grammar
    contract.  Otherwise the graph's single active version is authoritative.
    Semantic version ordering is only a recovery fallback for graphs without an
    active flag.
    """
    candidates = [
        (str(row.get("version") or row["v"]), bool(row.get("active", False)))
        for row in rows
        if row.get("version") or row.get("v")
    ]
    available = {version for version, _active in candidates}
    if preferred_version in available:
        return preferred_version

    active_versions = {version for version, active in candidates if active}
    if len(active_versions) > 1:
        raise ValueError(
            "multiple active grammar versions have token snapshots: "
            + ", ".join(order_grammar_versions(active_versions))
        )
    if active_versions:
        return next(iter(active_versions))
    ordered = order_grammar_versions(available)
    return ordered[0] if ordered else None


def tokenise_query(query: str) -> list[str]:
    """Lower-case, snake-split, drop ISN connector stopwords.

    >>> tokenise_query("x_magnetic_field")
    ['x', 'magnetic', 'field']
    >>> tokenise_query("electron_temperature_at_outboard_midplane")
    ['electron', 'temperature', 'outboard', 'midplane']
    >>> tokenise_query("")
    []

    Empty / whitespace-only inputs return an empty list. Non-ASCII
    characters are coerced via ``str.lower`` then split on the same
    regex; consumers should treat the output as ``list[str]``.
    """
    if not query or not query.strip():
        return []
    parts = _SPLIT_RE.split(query.lower())
    return [p for p in parts if p and p not in STOPWORDS]


# ---------------------------------------------------------------------------
# Tier policy
# ---------------------------------------------------------------------------

#: Tier 1 — physical concepts. Always contributes; can solely surface a result.
TIER1_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"physical_base", "subject", "geometric_base"}
)

#: Tier 2 — operational modifiers. Contributes only with a Tier-1 anchor
#: AND co-occurrence in the vector or keyword stream (strict AND-gate).
#: aggregation/orbit/population are discriminative name-prefix modifiers
#: (e.g. total_, trapped_, fast_) that narrow a physical anchor — they behave
#: like the other operational modifiers, so they join Tier 2.
TIER2_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "transformation",
        "component",
        "position",
        "process",
        "aggregation",
        "orbit",
        "population",
        # state (charge_state/internal_state) is a subject-refinement modifier
        # that narrows a species anchor exactly like population/orbit — Tier 2.
        "state",
    }
)

#: Tier 3 — geometric / device modifiers. Tie-break boost only; never
#: surfaces a candidate alone, requires Tier-1 anchor.
TIER3_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"coordinate", "geometry", "region", "device", "object"}
)

#: All segments stored as bare-name columns by ``_write_grammar_decomposition``.
ALL_TIER_SEGMENTS: Final[frozenset[str]] = (
    TIER1_SEGMENTS | TIER2_SEGMENTS | TIER3_SEGMENTS
)

#: RRF weights per tier. Chosen so a single Tier-1 hit at vector/keyword
#: rank 1 outranks an unbounded flood of Tier-2/3-only hits.
TIER_WEIGHT: Final[dict[int, float]] = {1: 1.0, 2: 0.5, 3: 0.25}


def tier_of(segment: str) -> int:
    """Return the tier (1/2/3) for *segment*. Unknown segments return 0."""
    if segment in TIER1_SEGMENTS:
        return 1
    if segment in TIER2_SEGMENTS:
        return 2
    if segment in TIER3_SEGMENTS:
        return 3
    return 0


def filter_by_tier_policy(
    by_id: dict[str, dict[int, list[tuple[str, int, float]]]],
    vector_hits: set[str],
    keyword_hits: set[str],
) -> set[str]:
    """Apply the strict tier-eligibility AND-gate.

    Returns the set of SN ids whose grammar-stream hits are admissible.

    Eligibility rules (strict AND-gate):

    - **Tier 1 hit, vector/keyword co-occurrence:** admitted (physical
      anchor + corroboration).
    - **Tier 1 hit, no vector/keyword co-occurrence:** dropped — pure
      grammar evidence on a Tier-1 segment is not enough on its own.
    - **Tier 2 hit, no Tier 1 anchor:** dropped.
    - **Tier 2 hit + Tier 1 anchor + (vector OR keyword) hit:** admitted.
    - **Tier 3 hit:** requires a Tier 1 anchor; vector/keyword
      co-occurrence is irrelevant for tier 3.

    ``test_tier2_requires_tier1_anchor_and_vk_cooccurrence`` pins the
    strict reading: even a Tier-1-bearing candidate must also appear in
    vector or keyword to qualify when the only evidence is grammar.
    Admitting Tier-1 hits unconditionally would let a single common
    physical token flood the result set.

    Args:
        by_id: ``{sn_id: {tier: [(segment, rank, weight), …]}}``.
        vector_hits: set of SN ids that appear in the vector stream.
        keyword_hits: set of SN ids that appear in the keyword stream.

    Returns:
        Set of SN ids admitted to the grammar-stream RRF input.
    """
    admitted: set[str] = set()
    for sn_id, tiers in by_id.items():
        has_t1 = 1 in tiers
        has_t2 = 2 in tiers
        has_t3 = 3 in tiers
        in_vk = sn_id in vector_hits or sn_id in keyword_hits
        if has_t1 and in_vk:
            # Anchor + corroboration — strongest case.
            admitted.add(sn_id)
        elif has_t2 and has_t1 and in_vk:
            # Tier-2 needs both anchor AND co-occurrence.
            admitted.add(sn_id)
        elif has_t3 and has_t1:
            # Tier-3 only contributes when anchored; vector/keyword optional.
            admitted.add(sn_id)
        # else: dropped
    return admitted
