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

import re
from collections.abc import Iterator, Set
from dataclasses import dataclass
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
OPERATOR_SEGMENT = "operator"

# The ISN *model layer* names the operator slot by how the operator attaches:
# a prefix lands in ``transformation``, a postfix in ``decomposition`` (see
# ``GrammarSegments._to_model_dict``).  A composer reporting a gap against one of
# those is naming the operator slot in the model's own vocabulary, so they are
# legitimate gap targets even though neither is a SEGMENT_TOKEN_MAP segment.
OPERATOR_SLOT_ALIASES: frozenset[str] = frozenset({"transformation", "decomposition"})


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


@lru_cache(maxsize=2)
def grammar_tokens_by_segment(
    *, include_operators: bool = True
) -> dict[str, tuple[str, ...]]:
    """Every ISN grammar token, keyed by the class that admits it.

    This is the vocabulary accessor for the whole module family.  It is the
    union of two sources that ISN keeps apart:

    - ``SEGMENT_TOKEN_MAP`` — the closed per-segment enums the parser slots
      tokens into (segments with an empty list are omitted; every token is
      admissible in an open segment by definition, so indexing it says nothing);
    - the operator registry — a grammar *mechanism* rather than a segment.
      Operators compose through the ordered ``operators`` expression list, so
      they appear in no ``SEGMENT_TOKEN_MAP`` slot and are exposed here under
      the synthetic :data:`OPERATOR_SEGMENT` class.

    A consumer reading only ``SEGMENT_TOKEN_MAP`` cannot see 51 legal tokens and
    reports them as missing vocabulary; reading only the operator registry
    cannot see the segment enums.  Every consumer that needs "is this token
    legal, and where does it belong" must come through here so the two halves
    can never drift apart again.

    ``include_operators=False`` returns the declared segment vocabularies
    without loading the separate operator context.  This is the lightweight
    path for a consumer that needs one segment class rather than the whole
    compositional grammar.  The default retains the complete vocabulary
    contract for general consumers.

    Returns an empty dict when ISN is unavailable — the rules built on it then
    turn off rather than falling back to a vocabulary snapshot that would rot.
    """
    out: dict[str, tuple[str, ...]] = {}

    stm = _load_segment_token_map()
    if stm is not None:
        try:
            for seg, tokens in stm.items():
                if not tokens:
                    continue  # open-vocabulary segment
                out[seg] = tuple(tokens)
        except Exception:  # pragma: no cover — defensive
            out = {}

    if include_operators:
        operators = _operator_tokens()
        if operators:
            out[OPERATOR_SEGMENT] = tuple(sorted(operators))

    return out


@lru_cache(maxsize=1)
def grammar_token_index() -> dict[str, tuple[str, ...]]:
    """Reverse of :func:`grammar_tokens_by_segment`: token → classes admitting it.

    A token may legitimately resolve to several classes (a qualifier that is
    also a component axis, say).  Order follows :func:`grammar_tokens_by_segment`
    so the reported class list is deterministic.
    """
    index: dict[str, list[str]] = {}
    for segment, tokens in grammar_tokens_by_segment().items():
        for token in tokens:
            index.setdefault(token, []).append(segment)
    return {token: tuple(segments) for token, segments in index.items()}


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

    The operator class is included (via :func:`grammar_tokens_by_segment`): a
    token that is a known operator is not an absent gap — the composer
    mis-slotted it, so it classifies as wrong-slot placement.
    """
    found = list(grammar_token_index().get(token, ()))
    # Augment with the parser-resolved base segment when the grammar accepts
    # the token as a self-resolving lexical-compound base absent from the flat
    # map (e.g. internal_inductance, major_radius).
    base_seg = resolved_base_segment(token)
    if base_seg is not None and base_seg not in found:
        found.append(base_seg)
    return found


#: Words by which a composer indexes one sample of a repeated structure.  A
#: standard name never carries them: every sample point of one object shares one
#: name, so the index lives in the DD path, not the vocabulary.
_ORDINAL_WORDS: frozenset[str] = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "initial",
        "final",
        "last",
        "start",
        "starting",
        "end",
        "ending",
        "intermediate",
    }
)

#: Nouns naming the sampled feature an ordinal indexes.  Required alongside an
#: ordinal before the ordinality rule fires, so an ordinary word that happens to
#: read as an ordinal (a token about a process *end state*, say) is not caught.
_SAMPLE_FEATURE_NOUNS: frozenset[str] = frozenset(
    {
        "point",
        "points",
        "sample",
        "samples",
        "node",
        "nodes",
        "centre",
        "center",
        "centroid",
        "position",
        "coordinate",
        "coordinates",
        "summit",
        "knot",
        "vertex",
        "corner",
        "index",
        "side",
    }
)

#: A DD field spelled as a letter and an index (``x1``, ``x2``) — the DD's own
#: name for a coordinate slot, carrying no physics.  Such a word is never
#: vocabulary: the name must say what the coordinate *is*.
_DD_INDEXED_FIELD = re.compile(r"^[A-Za-z]\d+$")


def _covering_registered_span(words: list[str], at: int) -> str | None:
    """The registered multi-word token spanning ``words[at]``, if one does.

    ``first_wall`` is registered, so the ``first`` inside
    ``first_wall_midplane`` is part of a grammar token rather than an ordinal.
    Checking this before reading a word as an ordinal is what stops the rule
    firing on the grammar's own vocabulary.
    """
    index = grammar_token_index()
    for width in range(2, _MAX_TOKEN_WIDTH + 1):
        for start in range(max(0, at - width + 1), at + 1):
            if start + width > len(words) or not start <= at < start + width:
                continue
            span = "_".join(words[start : start + width])
            if span in index:
                return span
    return None


def _strip_edge_connectives(words: list[str]) -> list[str]:
    """Drop grammar connectives left exposed at either end of a reduction.

    Removing ``third`` from ``third_point_of_line_of_sight`` and then the
    feature noun leaves a leading ``of``; the connectives *inside* a token
    (``line_of_sight``) are part of it and must stay.
    """
    connectives = _grammar_connectives()
    out = list(words)
    while out and out[0] in connectives:
        out.pop(0)
    while out and out[-1] in connectives:
        out.pop()
    return out


def _registered_singular(token: str) -> str | None:
    """The registered token ``token`` is an English plural of, if any.

    ``lines_of_sight`` pluralises the head word of the registered
    ``line_of_sight``, so a reduction landing on it can still name a real
    target.  Only accepted when the depluralised spelling is registered.
    """
    words = token.split("_")
    for i, word in enumerate(words):
        if not word.endswith("s") or len(word) < 4:
            continue
        candidate = "_".join([*words[:i], word[:-1], *words[i + 1 :]])
        if is_known_token(candidate):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class OrdinalForm:
    """A proposal that indexes one sample of a repeated structure.

    Attributes:
        ordinals: The ordinal words the token carries.
        reduced: The token with the ordinals dropped.
        locus: ``reduced`` with the sampled-feature nouns dropped too — the
            object the samples belong to, which is what a name may carry.
        target: The registered token one of those reductions lands on, or
            ``None`` when the locus itself is unregistered (an honest, and much
            narrower, vocabulary request than the ordinal compound).
    """

    ordinals: tuple[str, ...]
    reduced: str
    locus: str
    target: str | None


def ordinal_form(token: str) -> OrdinalForm | None:
    """Resolve ``token`` as an ordinal sample of a structure, or ``None``.

    Requires an ordinal word that no registered multi-word token covers, plus a
    sampled-feature noun for the ordinal to index.  Both conditions together are
    what keep the rule off the grammar's own vocabulary and off tokens whose
    ``end``/``final`` is a state rather than an index.
    """
    if not token or not grammar_token_index():
        return None
    words = token.split("_")
    ordinal_at = [
        i
        for i, word in enumerate(words)
        if word in _ORDINAL_WORDS and _covering_registered_span(words, i) is None
    ]
    if not ordinal_at:
        return None
    if not any(
        word in _SAMPLE_FEATURE_NOUNS
        for i, word in enumerate(words)
        if i not in ordinal_at
    ):
        return None

    kept = [word for i, word in enumerate(words) if i not in ordinal_at]
    reduced = "_".join(_strip_edge_connectives(kept))
    locus = "_".join(
        _strip_edge_connectives([w for w in kept if w not in _SAMPLE_FEATURE_NOUNS])
    )

    target: str | None = None
    for candidate in (reduced, locus):
        if not candidate:
            continue
        if is_known_token(candidate):
            target = candidate
            break
        singular = _registered_singular(candidate)
        if singular is not None:
            target = singular
            break

    return OrdinalForm(
        ordinals=tuple(words[i] for i in ordinal_at),
        reduced=reduced,
        locus=locus,
        target=target,
    )


def dd_indexed_field_words(token: str) -> tuple[str, ...]:
    """The DD coordinate-slot spellings (``x1``, ``x2``) ``token`` carries."""
    if not token:
        return ()
    return tuple(w for w in token.split("_") if _DD_INDEXED_FIELD.match(w))


def classify_gap(segment: str, token: str) -> tuple[str, list[str]]:
    """Classify a single vocabulary gap against the current ISN installation.

    Returns ``(category, actual_segments)`` where:

    - ``"false_positive"`` — token exists in the reported segment
    - ``"invalid_segment"`` — reported segment is not in ISN grammar
    - ``"open_segment"`` — reported segment has open vocabulary
    - ``"wrong_slot_placement"`` — token exists in exactly one other segment
    - ``"ambiguous_known_token"`` — token exists in multiple other segments
    - ``"rule_violation"`` — the spelling is forbidden however the vocabulary
      grows: it indexes one sample of a repeated structure, or it carries a DD
      coordinate-slot field name
    - ``"reuse"`` — a mechanical check resolves the token to a registered one
      (see :mod:`vocab_reuse`); overrides any embedding adjudication, since it
      derives the target rather than scoring a similarity
    - ``"decomposable"`` — compound token whose parts exist in other segments
    - ``"absent"`` — token is not in any closed segment (genuine gap)

    The three derived categories are ordered: a forbidden spelling is settled
    before any question of reuse, and reuse before decomposition — a word-order
    variant covers as a compound of its own words, and "compose it from those"
    would tell the composer to keep the wrong order.
    """
    if not is_valid_segment(segment):
        return "invalid_segment", []

    if is_open_segment(segment):
        return "open_segment", []

    segments_found = is_known_token(token)

    if segment in segments_found:
        return "false_positive", segments_found

    if not segments_found:
        if dd_indexed_field_words(token):
            return "rule_violation", []
        ordinal = ordinal_form(token)
        if ordinal is not None:
            return "rule_violation", (
                is_known_token(ordinal.target) if ordinal.target else []
            )
        from imas_codex.standard_names.vocab_reuse import registered_reuse

        reuse = registered_reuse(segment, token)
        if reuse is not None:
            return "reuse", list(reuse.target_segments)
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
# tokens, sits in an open-vocabulary segment, reuses a registered token under
# another spelling, or is forbidden however the vocabulary grows.  Only an
# ``absent`` closed-segment gap warrants an ISN vocabulary addition — or
# retiring the source that reported it.
NON_ACTIONABLE_GAP_CATEGORIES: frozenset[str] = frozenset(
    {
        "false_positive",
        "invalid_segment",
        "open_segment",
        "wrong_slot_placement",
        "ambiguous_known_token",
        "reuse",
        "rule_violation",
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


@lru_cache(maxsize=1)
def _registered_physical_bases() -> frozenset[str]:
    """Return the physical-base vocabulary declared by the installed grammar."""
    return frozenset(
        grammar_tokens_by_segment(include_operators=False).get("physical_base", ())
    )


class _RegisteredPhysicalBaseSet(Set[str]):
    """Read-only view over ISN's public physical-base vocabulary."""

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and value in _registered_physical_bases()

    def __iter__(self) -> Iterator[str]:
        return iter(_registered_physical_bases())

    def __len__(self) -> int:
        return len(_registered_physical_bases())


# Registered bases are indivisible during a cover walk.  The exported set lets
# callers check that classifier boundary directly; its contents come entirely
# from ISN rather than a second vocabulary snapshot in codex.
ATOMIC_COMPOUNDS: Set[str] = _RegisteredPhysicalBaseSet()


#: Widest span, in underscore-delimited words, a single registered token may
#: occupy when covering a compound.  Bounds the greedy walk; the longest
#: multi-word tokens in the grammar (``derivative_with_respect_to``,
#: ``cumulative_inside_flux_surface``) sit at this width.
_MAX_TOKEN_WIDTH = 5


@lru_cache(maxsize=1)
def _grammar_connectives() -> frozenset[str]:
    """Words that join grammar tokens without themselves being vocabulary.

    ``of`` is the joiner ISN's renderer places between an operator and its
    operand (``square_of_temperature``); ``and`` / ``to`` come from the binary
    operators' own declared separators (``_and_``, ``_to_``), read from the
    registry rather than assumed.  These carry no vocabulary of their own, so a
    compound must be allowed to step over them while being covered.
    """
    connectives = {"of"}
    try:
        from imas_standard_names import get_grammar_context

        operators = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    except Exception:
        return frozenset(connectives)
    for spec in operators.values():
        separator = spec.get("separator") if isinstance(spec, dict) else None
        if separator:
            connectives.update(w for w in separator.split("_") if w)
    return frozenset(connectives)


#: Inflections of an operator token the composer emits in place of the
#: registered spelling (``squared`` for ``square``, ``logarithmic`` for
#: ``logarithm``).  Only suffixes that leave the registered stem intact appear
#: here; the guard against an arbitrary word matching is that the remainder must
#: itself be a registered operator, not the suffix list.
_OPERATOR_INFLECTION_SUFFIXES: tuple[str, ...] = ("d", "ed", "s", "es", "ic")

#: The words a composer writes for a division — the two English spellings of a
#: quotient. ISN expresses it as the binary ``ratio`` operator over two operands,
#: so a compound spelled with either is a binary composition rather than a
#: compound base — provided BOTH operands are themselves fully registered.
#: Matched per whole word, so a token merely beginning with these letters
#: (``permeability``) is untouched.
_DIVISION_WORDS: tuple[str, ...] = ("over", "per")


def _division_at(words: list[str]) -> int | None:
    """Index of the division word in ``words``, or ``None`` if there is none.

    A division word inside a registered multi-word token is part of that token,
    not a quotient: the operator ``per_toroidal_mode`` spells one, and splitting
    a compound there would tear a registered operator in half.
    """
    for i, word in enumerate(words):
        if word in _DIVISION_WORDS and _covering_registered_span(words, i) is None:
            return i
    return None


#: Physics shorthand a composer writes inside a ratio operand instead of the
#: registered token, mapped to that token.  Deliberately tiny: every entry is a
#: symbol whose reading is unambiguous in a quotient of plasma quantities, and
#: each is dropped at runtime by :func:`_symbol_expansions` when its target is
#: not registered.  Ambiguous symbols are omitted on purpose — ``Z`` is a
#: vertical coordinate and a charge number, ``T`` a temperature and a time,
#: ``n`` a density and a mode number, and an honest ``absent`` that asks for the
#: token the composer needs costs less than a wrong expansion it will follow.
#:
#: - ``rho`` — the DD's ``rho_tor``, ISN's ``toroidal_flux_coordinate``. The
#:   dimensionful unnormalized coordinate is the carrier a quotient of
#:   gradients is written against.
#: - ``b``, ``b_field`` — the magnetic field, in the magnitude sense a scalar
#:   quotient uses.
#: - ``r`` — the major radius, the only radius a ``_over_r`` quotient of
#:   axisymmetric quantities divides by.
#: - ``grad`` — the composer's abbreviation of the registered ``gradient``
#:   operator, which it writes when spelling a quotient of gradients.
#:
#: Keys may span several words; the longest match at a position wins, so
#: ``b_field`` is read whole rather than as ``b`` followed by a stray ``field``.
_SYMBOL_EXPANSION_CANDIDATES: dict[str, str] = {
    "rho": "toroidal_flux_coordinate",
    "b": "magnetic_field",
    "b_field": "magnetic_field",
    "r": "major_radius",
    "grad": "gradient",
}


@lru_cache(maxsize=1)
def _symbol_expansions() -> dict[str, str]:
    """Symbol shorthand mapped to registered tokens, unregistered targets dropped."""
    index = grammar_token_index()
    if not index:
        return {}
    return {
        symbol: target
        for symbol, target in _SYMBOL_EXPANSION_CANDIDATES.items()
        if symbol not in index and is_known_token(target)
    }


def _expand_symbol_words(
    words: list[str],
) -> tuple[list[str], tuple[tuple[str, str], ...]] | None:
    """Replace symbol shorthand in ``words`` with the registered token's words.

    Matching is per whole span and case-insensitive (the shorthand is
    conventionally capitalised — ``B``, ``R``), longest span first, and only
    spans that are themselves unregistered are substituted, so a symbol the
    grammar happens to carry is left alone.  Returns ``None`` when nothing was
    expandable, so the caller can tell a retry apart from a no-op.
    """
    expansions = _symbol_expansions()
    if not expansions:
        return None
    widest = max(len(symbol.split("_")) for symbol in expansions)
    out: list[str] = []
    used: list[tuple[str, str]] = []
    i = 0
    while i < len(words):
        for width in range(min(widest, len(words) - i), 0, -1):
            span = "_".join(words[i : i + width])
            target = expansions.get(span.lower())
            if target is not None:
                out.extend(target.split("_"))
                used.append((span, target))
                i += width
                break
        else:
            out.append(words[i])
            i += 1
    if not used:
        return None
    return out, tuple(used)


def _registered_operator_stem(word: str) -> str | None:
    """Return the registered operator ``word`` spells, allowing an inflection.

    Exact matches win.  Otherwise a trailing inflection is stripped and the
    result accepted only when the remainder is itself a registered operator, so
    an arbitrary word ending in one of those letters cannot masquerade as one.
    """
    operators = _operator_tokens()
    if not operators:
        return None
    if word in operators:
        return word
    for suffix in _OPERATOR_INFLECTION_SUFFIXES:
        if len(word) > len(suffix) and word.endswith(suffix):
            stem = word[: -len(suffix)]
            if stem in operators:
                return stem
    return None


@dataclass(frozen=True, slots=True)
class OperatorComposition:
    """A token covered by registered operators applied to registered tokens.

    Attributes:
        operators: The registered operator tokens the compound spells, in the
            order they appear.  Inflected spellings are reported as their
            registered stem (``squared`` → ``square``).
        bases: The registered non-operator tokens the operators apply to.
        segments: Every grammar class the cover touched, operators included.
        binary_operator: The registered binary operator the spelling implies, set
            when the compound is a ratio written with ``over``.  ``None`` for a
            plain unary composition.
        operands: For a binary composition, the two operand sides — with any
            symbol shorthand expanded, so guidance quotes registered tokens.
            Empty for a unary one.
        symbol_expansions: The ``(symbol, registered token)`` substitutions the
            cover needed, so guidance can show the composer what its shorthand
            was read as.
    """

    operators: tuple[str, ...]
    bases: tuple[str, ...]
    segments: tuple[str, ...]
    binary_operator: str | None = None
    operands: tuple[str, ...] = ()
    symbol_expansions: tuple[tuple[str, str], ...] = ()


@lru_cache(maxsize=1)
def _infix_operator_splits() -> tuple[
    tuple[str, tuple[str, ...], tuple[str, ...]], ...
]:
    """Registered multi-word operators, split into every (head, tail) word pair.

    A composer writing natural word order puts the operand *inside* a multi-word
    operator — ``derivative_of_area_with_respect_to_poloidal_flux`` for the
    registered ``derivative_with_respect_to`` applied to ``area``.  The operator
    is present intact but interrupted, so a left-to-right walk cannot see it.

    Each entry is ``(operator, head_words, tail_words)``.  Longer heads come
    first so the most specific split is tried before a shorter one.  Derived from
    the registry, so a new multi-word operator is covered without a code change.
    """
    splits: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for operator in sorted(_operator_tokens()):
        words = operator.split("_")
        if len(words) < 2:
            continue
        for cut in range(len(words) - 1, 0, -1):
            splits.append((operator, tuple(words[:cut]), tuple(words[cut:])))
    splits.sort(key=lambda s: (-len(s[1]), -len(s[2])))
    return tuple(splits)


@lru_cache(maxsize=1)
def _cover_index() -> dict[str, tuple[str, ...]]:
    """:func:`grammar_token_index` plus the lexical-compound bases the walk needs.

    The flat map lists atomic bases only, so a compound the grammar resolves to
    itself (``major_radius``) is invisible to a span lookup.  Resolving arbitrary
    spans through the parser is not an option — one parse costs tens of
    milliseconds and the walk tries several spans per word — so exactly the
    compounds a symbol expansion can introduce are resolved once, here, and
    folded in.
    """
    index = grammar_token_index()
    if not index:
        return {}
    merged = dict(index)
    for target in _SYMBOL_EXPANSION_CANDIDATES.values():
        if target in merged or "_" not in target:
            continue
        segment = resolved_base_segment(target)
        if segment is not None:
            merged[target] = (segment,)
    return merged


def _cover_words(words: list[str]) -> list[tuple[str, tuple[str, ...]]] | None:
    """Cover a plain word sequence, no infix or ratio handling. See _cover_token."""
    index = _cover_index()
    if not index:
        return None

    connectives = _grammar_connectives()
    cover: list[tuple[str, tuple[str, ...]]] = []
    i = 0
    while i < len(words):
        if words[i] in connectives:
            i += 1
            continue
        for width in range(min(_MAX_TOKEN_WIDTH, len(words) - i), 0, -1):
            span = "_".join(words[i : i + width])
            classes = index.get(span)
            if classes:
                cover.append((span, classes))
                i += width
                break
            # A one-word span may be an inflected operator spelling.
            if width == 1:
                stem = _registered_operator_stem(span)
                if stem is not None:
                    cover.append((stem, (OPERATOR_SEGMENT,)))
                    i += 1
                    break
        else:
            return None  # uncovered word — the compound is not fully registered
    return cover or None


def _cover_infix_operator(
    words: list[str],
) -> list[tuple[str, tuple[str, ...]]] | None:
    """Cover a sequence spelling a multi-word operator around its operand.

    Requires the operator's head at the very start (operators are prefixes) and
    its tail as a whole span later, with everything between and after covered by
    registered tokens.  Both halves of a registered operator, in order, is a
    narrow enough signal that an unrelated compound cannot satisfy it.
    """
    for operator, head, tail in _infix_operator_splits():
        n_head, n_tail = len(head), len(tail)
        if len(words) <= n_head + n_tail:
            continue
        if tuple(words[:n_head]) != head:
            continue
        for start in range(n_head, len(words) - n_tail + 1):
            if tuple(words[start : start + n_tail]) != tail:
                continue
            inner = _cover_words(words[n_head:start])
            outer = _cover_words(words[start + n_tail :])
            if inner is None and words[n_head:start]:
                continue
            if outer is None and words[start + n_tail :]:
                continue
            return [
                (operator, (OPERATOR_SEGMENT,)),
                *(inner or []),
                *(outer or []),
            ]
    return None


def _cover_token(token: str) -> list[tuple[str, tuple[str, ...]]] | None:
    """Cover ``token`` with registered tokens, or return None if it cannot be.

    Greedy longest-first walk over the underscore-delimited words, matching each
    span against :func:`grammar_token_index` — which carries operators as well
    as the segment enums, so an operator span is matchable.  Grammar
    connectives (``of``, and the binary separators) are stepped over.  Matching
    is per whole span, never per substring, so a word that merely *contains* a
    registered token's letters cannot match it.

    Returns the ``(span, classes)`` pairs covering the token, or ``None`` when
    any word is left uncovered.  Falls back to matching a multi-word operator
    spelled around its operand when the straight walk cannot cover the words.
    """
    if not grammar_token_index():
        return None
    words = token.split("_")
    return _cover_words(words) or _cover_infix_operator(words)


@lru_cache(maxsize=1)
def _registered_binary_operator() -> str | None:
    """The registered binary operator a division spells, or None if there is none.

    Looked up in the registry by kind rather than named, so codex does not carry
    ISN's spelling of it.
    """
    try:
        from imas_standard_names import get_grammar_context

        operators = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    except Exception:
        return None
    for token, spec in sorted(operators.items()):
        if isinstance(spec, dict) and spec.get("kind") == "binary":
            if spec.get("separator") == "_to_":
                return token
    return None


def _ratio_composition(token: str) -> OperatorComposition | None:
    """Cover a compound spelled as a division, e.g. ``<a>_over_<b>``.

    ISN expresses a division as the binary ratio operator over two operand
    strings, so such a compound is not a missing base — it is a binary
    expression the composer wrote as one word.

    BOTH operands must cover completely.  A single uncovered word on either side
    means the composer needs a token there, and guessing at the composition would
    suppress that request.  An operand that will not cover as written is retried
    once with its symbol shorthand expanded (:func:`_symbol_expansions`), which
    is what lets ``gradient_rho_squared_over_B_squared`` resolve to a ratio of
    two registered operands instead of asking for ``rho`` and ``B`` as tokens;
    the substitutions are carried on the result so guidance can quote them.

    **``over`` is deliberately NOT treated as a connective the cover walk steps
    over.**  Skipping it would let a division fall through to the unary path and
    classify as an ordinary compound, which reads as a fix — the token stops
    being ``absent`` — while emitting guidance that tells the composer to fold
    the operators into one base and silently drops the division.  Confident wrong
    guidance costs more than an honest ``absent``: the composer follows it and
    mints a name that means something else.  A division is a *binary* expression,
    so it either resolves here, with both operands intact and the ratio named, or
    it stays ``absent`` and asks for the token it actually needs.
    """
    words = token.split("_")
    at = _division_at(words)
    if at is None:
        return None
    binary = _registered_binary_operator()
    if binary is None:
        return None

    left_words, right_words = words[:at], words[at + 1 :]
    if not left_words or not right_words:
        return None

    expansions: list[tuple[str, str]] = []
    sides: list[tuple[list[str], list[tuple[str, tuple[str, ...]]]]] = []
    for side_words in (left_words, right_words):
        cover = _cover_words(side_words) or _cover_infix_operator(side_words)
        if cover is None:
            expanded = _expand_symbol_words(side_words)
            if expanded is None:
                return None
            side_words, used = expanded
            cover = _cover_words(side_words) or _cover_infix_operator(side_words)
            if cover is None:
                return None
            expansions.extend(used)
        sides.append((side_words, cover))

    operators: list[str] = []
    bases: list[str] = []
    segments: list[str] = []
    for _side_words, cover in sides:
        for span, classes in cover:
            segments.extend(classes)
            (operators if OPERATOR_SEGMENT in classes else bases).append(span)

    return OperatorComposition(
        operators=(binary, *operators),
        bases=tuple(bases),
        segments=tuple(dict.fromkeys([OPERATOR_SEGMENT, *segments])),
        binary_operator=binary,
        operands=tuple("_".join(side_words) for side_words, _cover in sides),
        symbol_expansions=tuple(expansions),
    )


@lru_cache(maxsize=4096)
def operator_composition(token: str) -> OperatorComposition | None:
    """Return how ``token`` is expressible as registered operators, if it is.

    A compound the composer reported as missing vocabulary is often the grammar
    it already has, spelled as one word: ``inverse_square`` is the operators
    ``inverse`` and ``square``; ``line_integrated_density`` is the operator
    ``line_integrated`` on the base ``density``.  Such a token is not a
    vocabulary deficiency — routing the operators through ``operators``
    composes the name today.

    Returns ``None`` when the token is a registered base, when any part is
    unregistered, when no part is an operator, or when ISN is unavailable.
    """
    if not token or token in ATOMIC_COMPOUNDS:
        return None

    ratio = _ratio_composition(token)
    if ratio is not None:
        return ratio

    cover = _cover_token(token)
    if cover is None:
        return None

    operators: list[str] = []
    bases: list[str] = []
    segments: list[str] = []
    for span, classes in cover:
        segments.extend(classes)
        if OPERATOR_SEGMENT in classes:
            operators.append(span)
        else:
            bases.append(span)

    if not operators:
        return None  # ordinary multi-segment compound — not an operator matter

    return OperatorComposition(
        operators=tuple(operators),
        bases=tuple(bases),
        segments=tuple(dict.fromkeys(segments)),
    )


def _check_decomposable(token: str) -> list[str]:
    """Check if a compound token can be decomposed into existing vocabulary.

    Uses bounded left-to-right longest-prefix matching against every grammar
    class — operators included, so a compound spelling registered operators
    (``inverse_square``, ``flux_surface_averaged_square_magnetic_field``) is
    reported as decomposable rather than as missing vocabulary.  Returns the
    list of classes where parts were found, or an empty list if the token
    cannot be covered.

    Skips registered bases because they are already complete grammar tokens.
    """
    if token in ATOMIC_COMPOUNDS:
        return []

    # Operator-bearing spellings — inflected single words, ratios written with
    # ``over``, multi-word operators split around their operand — all resolve
    # through one function so this classifier and a caller asking for the
    # composition directly can never disagree about what is expressible.
    comp = operator_composition(token)
    if comp is not None:
        return list(comp.segments)

    # What remains is an operator-free compound of ordinary segment tokens.
    if "_" not in token:
        return []

    cover = _cover_token(token)
    if cover is None:
        return []

    matched_segments: list[str] = []
    for _span, classes in cover:
        matched_segments.extend(classes)

    # Any cover is a decomposition: a single-class cover (two qualifiers, two
    # operators) is as composable as a cross-class one, given the compound is
    # not itself registered — the caller checked that before asking.
    return list(dict.fromkeys(matched_segments))


@dataclass(frozen=True, slots=True)
class GapVerdict:
    """A gap classification together with the guidance a composer can act on.

    :func:`classify_gap` answers "is this a real deficiency"; a model that gets
    only that answer back re-proposes the same token.  This carries the rest:
    which registered operators the token spells, which classes its parts belong
    to, and one line of prose naming the slot to use instead.

    Attributes:
        category: The :func:`classify_gap` category.
        segments: The grammar classes the token or its parts resolve to.
        operators: Registered operator tokens the proposal spells, if any.
        bases: Registered non-operator tokens the operators apply to.
        guidance: One line, safe to hand back to a model as retry feedback.
        reuse_target: The registered token this proposal should reuse, set for a
            ``reuse`` verdict and for a ``rule_violation`` whose ordinal-free
            reduction lands on a registered locus.
    """

    category: str
    segments: tuple[str, ...]
    operators: tuple[str, ...]
    bases: tuple[str, ...]
    guidance: str
    reuse_target: str | None = None


def _operator_routing_advice(operators: tuple[str, ...], bases: tuple[str, ...]) -> str:
    """Render the "use the operator slot" half of a verdict's guidance."""
    many = len(operators) > 1
    advice = (
        f"{', '.join(operators)} {'are registered operators' if many else 'is a registered operator'}"
        f" — route {'them' if many else 'it'} through the outer-to-inner "
        f"operators list (the live registry supplies each kind)"
    )
    if bases:
        noun = "tokens" if len(bases) > 1 else "token"
        advice += f", applied to the registered {noun} {', '.join(bases)}"
    return advice


def _reuse_finding(segment: str, token: str):
    """The mechanical reuse resolution for ``token``, or ``None``.

    Imported lazily: :mod:`vocab_reuse` reads the vocabulary through this module,
    so a module-level import would close a cycle.
    """
    from imas_codex.standard_names.vocab_reuse import registered_reuse

    return registered_reuse(segment, token)


def _reuse_guidance(finding) -> str:
    """Render a reuse finding as one line of retry feedback."""
    from imas_codex.standard_names.vocab_reuse import reuse_guidance

    return reuse_guidance(finding)


def _absent_narrowing(token: str) -> str:
    """Narrow an absent compound's request to the words that are not registered.

    A compound gap otherwise asks for the whole compound, and a hundred such asks
    are a handful of missing atoms wearing prefixes.  Two words are never offered
    as tokens to request: the division word, because listing it invites folding a
    quotient into one base, and a bare index, because an isotope or ordinal digit
    is not vocabulary.
    """
    index = grammar_token_index()
    words = token.split("_")
    if len(words) < 2 or not index:
        return ""

    at = _division_at(words)
    if at is not None:
        binary = _registered_binary_operator()
        sides = ("_".join(words[:at]), "_".join(words[at + 1 :]))
        unresolved = [s for s in sides if s and not _cover_words(s.split("_"))]
        if binary and unresolved:
            return (
                f" It is a division: keep it as the binary operator "
                f"'{binary}' over two operands and request only the operand that "
                f"is unregistered ({', '.join(repr(s) for s in unresolved)}) — "
                f"never fold the quotient into one base token."
            )
        return ""

    connectives = _grammar_connectives()
    known = [w for w in words if w in index]
    unknown = [
        w
        for w in words
        if w not in index
        and w not in connectives
        and not w.isdigit()
        and not _DD_INDEXED_FIELD.match(w)
    ]
    if not known or not unknown:
        return ""
    return (
        f" Its words {', '.join(repr(w) for w in known)} are already registered — "
        f"request only {', '.join(repr(w) for w in unknown)}, and compose the rest."
    )


def describe_gap(segment: str, token: str) -> GapVerdict:
    """Classify a gap and render guidance that tells a composer what to do.

    Wraps :func:`classify_gap` — the category always agrees with it — and adds
    the operator/base decomposition plus a single line of prose naming the slot
    the token belongs in.  Intended for retry feedback: a composer told only
    that a name failed has no way to converge, whereas one told that ``square``
    is an operator rather than a qualifier can fix the name on the next attempt.
    """
    category, segments_found = classify_gap(segment, token)
    comp = operator_composition(token)
    operators = comp.operators if comp is not None else ()
    bases = comp.bases if comp is not None else ()
    reuse_target: str | None = None

    if category == "false_positive":
        guidance = (
            f"'{token}' is already a registered {segment} token — it is not a "
            f"vocabulary gap; use it directly."
        )
    elif category == "invalid_segment":
        legal = ", ".join(sorted(reportable_segments()))
        guidance = (
            f"'{segment}' is not a grammar segment class. Report the gap "
            f"against one of: {legal}."
        )
    elif category == "open_segment":
        guidance = (
            f"the {segment} segment has no fixed vocabulary, so '{token}' "
            f"needs no registration."
        )
    elif category == "wrong_slot_placement" and segments_found == [OPERATOR_SEGMENT]:
        guidance = (
            f"'{token}' is a registered OPERATOR, not a {segment} — "
            f"{_operator_routing_advice((token,), ())}."
        )
    elif category in {"wrong_slot_placement", "ambiguous_known_token"}:
        where = " or ".join(segments_found)
        guidance = (
            f"'{token}' is a registered {where} token, not a {segment} — "
            f"place it in the {where} slot."
        )
    elif category == "rule_violation" and dd_indexed_field_words(token):
        fields = ", ".join(f"'{w}'" for w in dd_indexed_field_words(token))
        guidance = (
            f"{fields} in '{token}' names a DD coordinate slot, not a physical "
            f"quantity — a standard name says what the coordinate is (the axis it "
            f"runs along, the locus it measures), never the DD's field spelling. "
            f"Re-propose without it."
        )
    elif category == "rule_violation" and (ordinal := ordinal_form(token)) is not None:
        which = ", ".join(f"'{o}'" for o in ordinal.ordinals)
        if ordinal.target is not None:
            reuse_target = ordinal.target
            guidance = (
                f"'{token}' indexes one sample of a repeated structure; "
                f"ordinality never enters a standard name — every sample of one "
                f"object shares one name. Drop {which} and use the registered "
                f"'{ordinal.target}'."
            )
        elif ordinal.locus:
            guidance = (
                f"'{token}' indexes one sample of a repeated structure; "
                f"ordinality never enters a standard name — every sample of one "
                f"object shares one name. Name the object the samples belong to: "
                f"drop {which}, which leaves '{ordinal.locus}'. That locus is "
                f"itself unregistered, so request that single token if you need it "
                f"— not the ordinal compound."
            )
        else:
            guidance = (
                f"'{token}' indexes one sample of a repeated structure and names "
                f"no object; ordinality never enters a standard name — every "
                f"sample of one object shares one name. Name the object whose "
                f"samples these are, in the {segment} slot, and drop {which}."
            )
    elif (
        category == "reuse" and (finding := _reuse_finding(segment, token)) is not None
    ):
        reuse_target = finding.target
        guidance = _reuse_guidance(finding)
    elif category == "decomposable" and comp is not None and comp.binary_operator:
        left, right = comp.operands
        guidance = (
            f"'{token}' is a division, not a base token: express it with the "
            f"binary operator '{comp.binary_operator}' in operators — first "
            f"operand '{left}' in the base fields, second operand '{right}' in "
            f"secondary_operand."
        )
        if comp.symbol_expansions:
            read_as = ", ".join(
                f"'{symbol}' as '{target}'" for symbol, target in comp.symbol_expansions
            )
            guidance += (
                f" The shorthand was read as {read_as} — spell the registered "
                f"token in the operand, and keep the division: it is two operands "
                f"under '{comp.binary_operator}', not one compound base."
            )
    elif category == "decomposable" and operators:
        guidance = f"'{token}' is not a single token: {_operator_routing_advice(operators, bases)}."
    elif category == "decomposable":
        cover = _cover_token(token) or []
        parts = ", ".join(
            f"'{span}' ({' or '.join(classes)})" for span, classes in cover
        )
        guidance = (
            f"'{token}' decomposes into the registered tokens {parts} — "
            f"compose it from those rather than requesting a new token."
        )
    else:
        # Absent, and the fallback for a derived category whose deriving check
        # no longer fires — the grammar can be swapped underneath a caller, and
        # a stale "needs a new token" line is safer than a stale claim of reuse.
        guidance = (
            f"'{token}' is in no grammar class and does not decompose into "
            f"registered tokens; naming this needs a new {segment} token in "
            f"imas-standard-names."
        )
        guidance += _absent_narrowing(token)

    return GapVerdict(
        category=category,
        segments=tuple(segments_found),
        operators=operators,
        bases=bases,
        guidance=guidance,
        reuse_target=reuse_target,
    )


def clear_grammar_caches() -> None:
    """Drop every cached view of the ISN grammar in this module.

    The module memoises the grammar aggressively because the classifier runs per
    gap and per candidate.  A caller that swaps the ISN vocabulary underneath it
    — tests mocking ``SEGMENT_TOKEN_MAP`` or the operator registry — must reset
    all of it, and enumerating the caches at each such site is what lets a newly
    added cache go on serving stale vocabulary.  Adding a cache here is the only
    place that list has to be kept.
    """
    from imas_codex.standard_names.vocab_reuse import clear_reuse_caches

    clear_reuse_caches()
    for cache in (
        _operator_tokens,
        _grammar_connectives,
        _infix_operator_splits,
        _registered_binary_operator,
        _registered_physical_bases,
        _symbol_expansions,
        _cover_index,
        grammar_tokens_by_segment,
        grammar_token_index,
        known_segments,
        open_segments,
        reportable_segments,
        resolved_base_segment,
        operator_composition,
    ):
        # A caller may have substituted a plain callable for one of these; only
        # the memoised ones have anything to drop.
        clear = getattr(cache, "cache_clear", None)
        if clear is not None:
            clear()


@lru_cache(maxsize=1)
def reportable_segments() -> frozenset[str]:
    """Grammar classes a vocabulary gap may legitimately be reported against.

    Every class in :func:`grammar_tokens_by_segment` (the segment enums plus
    :data:`OPERATOR_SEGMENT`), the pseudo segments the composer uses for
    structural findings, and the :data:`OPERATOR_SLOT_ALIASES` under which the
    ISN model layer names that same slot.  Empty when ISN is unavailable, in
    which case callers must not constrain — there is nothing to constrain
    against.

    **Deliberately wider than :func:`known_segments`, and the two must not be
    merged.**  They answer different questions: this one is "what may a gap be
    *reported* against", ``known_segments`` is "what does the *parser* slot tokens
    into".  The reporting side is wider because the ISN model layer owns slots the
    parser has no segment for — a composer reporting against ``transformation``
    has named the operator slot correctly in the model's vocabulary.  Narrowing
    this to ``known_segments`` rejects those reports and makes the response model
    fail on valid composer output; seven tests pin that.
    """
    classes = set(grammar_tokens_by_segment())
    if not classes:
        return frozenset()
    return frozenset(classes | set(PSEUDO_SEGMENTS) | set(OPERATOR_SLOT_ALIASES))


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
    "OPERATOR_SEGMENT",
    "PSEUDO_SEGMENTS",
    "GapVerdict",
    "OperatorComposition",
    "OrdinalForm",
    "classify_gap",
    "clear_grammar_caches",
    "dd_indexed_field_words",
    "describe_gap",
    "filter_closed_segment_gaps",
    "grammar_token_index",
    "grammar_tokens_by_segment",
    "is_actionable_gap",
    "is_known_physical_base",
    "is_known_token",
    "is_open_segment",
    "is_valid_segment",
    "known_segments",
    "open_segments",
    "operator_composition",
    "ordinal_form",
    "reportable_segments",
    "resolved_base_segment",
]
