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


@lru_cache(maxsize=1)
def grammar_tokens_by_segment() -> dict[str, tuple[str, ...]]:
    """Every ISN grammar token, keyed by the class that admits it.

    This is the vocabulary accessor for the whole module family.  It is the
    union of two sources that ISN keeps apart:

    - ``SEGMENT_TOKEN_MAP`` — the closed per-segment enums the parser slots
      tokens into (segments with an empty list are omitted; every token is
      admissible in an open segment by definition, so indexing it says nothing);
    - the operator registry — a grammar *mechanism* rather than a segment.
      Operators compose through ``operator_token`` + ``operator_kind``, so they
      appear in no ``SEGMENT_TOKEN_MAP`` slot and are exposed here under the
      synthetic :data:`OPERATOR_SEGMENT` class.

    A consumer reading only ``SEGMENT_TOKEN_MAP`` cannot see 51 legal tokens and
    reports them as missing vocabulary; reading only the operator registry
    cannot see the segment enums.  Every consumer that needs "is this token
    legal, and where does it belong" must come through here so the two halves
    can never drift apart again.

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

#: The word a composer writes for a division. ISN expresses it as the binary
#: ``ratio`` operator over two operands, so a compound spelled with it is a
#: binary composition rather than a compound base — provided BOTH operands are
#: themselves fully registered.
_RATIO_WORD = "over"


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
        operands: For a binary composition, the two operand sides as written.
            Empty for a unary one.
    """

    operators: tuple[str, ...]
    bases: tuple[str, ...]
    segments: tuple[str, ...]
    binary_operator: str | None = None
    operands: tuple[str, ...] = ()


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


def _cover_words(words: list[str]) -> list[tuple[str, tuple[str, ...]]] | None:
    """Cover a plain word sequence, no infix or ratio handling. See _cover_token."""
    index = grammar_token_index()
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
    suppress that request: ``gradient_rho_squared_over_B_squared`` fails here
    because ``rho`` and ``B`` are symbol shorthand registered nowhere.

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
    if _RATIO_WORD not in words:
        return None
    binary = _registered_binary_operator()
    if binary is None:
        return None

    at = words.index(_RATIO_WORD)
    left_words, right_words = words[:at], words[at + 1 :]
    if not left_words or not right_words:
        return None

    left = _cover_words(left_words) or _cover_infix_operator(left_words)
    right = _cover_words(right_words) or _cover_infix_operator(right_words)
    if left is None or right is None:
        return None

    operators: list[str] = []
    bases: list[str] = []
    segments: list[str] = []
    for span, classes in [*left, *right]:
        segments.extend(classes)
        (operators if OPERATOR_SEGMENT in classes else bases).append(span)

    return OperatorComposition(
        operators=(binary, *operators),
        bases=tuple(bases),
        segments=tuple(dict.fromkeys([OPERATOR_SEGMENT, *segments])),
        binary_operator=binary,
        operands=("_".join(left_words), "_".join(right_words)),
    )


@lru_cache(maxsize=4096)
def operator_composition(token: str) -> OperatorComposition | None:
    """Return how ``token`` is expressible as registered operators, if it is.

    A compound the composer reported as missing vocabulary is often the grammar
    it already has, spelled as one word: ``inverse_square`` is the operators
    ``inverse`` and ``square``; ``line_integrated_density`` is the operator
    ``line_integrated`` on the base ``density``.  Such a token is not a
    vocabulary deficiency — routing the operators through ``operator_token``
    composes the name today.

    Returns ``None`` when the token is a lexicalized compound listed in
    :data:`ATOMIC_COMPOUNDS`, when any part is unregistered, when no part is an
    operator, or when ISN is unavailable.
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

    Skips tokens in :data:`ATOMIC_COMPOUNDS` to avoid false negatives on
    lexicalized physics terms.
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
    """

    category: str
    segments: tuple[str, ...]
    operators: tuple[str, ...]
    bases: tuple[str, ...]
    guidance: str


def _operator_routing_advice(operators: tuple[str, ...], bases: tuple[str, ...]) -> str:
    """Render the "use the operator slot" half of a verdict's guidance."""
    many = len(operators) > 1
    advice = (
        f"{', '.join(operators)} {'are registered operators' if many else 'is a registered operator'}"
        f" — route {'them' if many else 'it'} through operator_token "
        f"(with operator_kind from the registry)"
    )
    if bases:
        noun = "tokens" if len(bases) > 1 else "token"
        advice += f", applied to the registered {noun} {', '.join(bases)}"
    return advice


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
    elif category == "decomposable" and comp is not None and comp.binary_operator:
        left, right = comp.operands
        guidance = (
            f"'{token}' is a division, not a base token: express it with the "
            f"binary operator_token '{comp.binary_operator}' — first operand "
            f"'{left}' in base_token, second operand '{right}' in secondary_base."
        )
    elif category == "decomposable" and operators:
        guidance = f"'{token}' is not a single token: {_operator_routing_advice(operators, bases)}."
    elif category == "decomposable":
        where = ", ".join(segments_found)
        guidance = (
            f"'{token}' decomposes into tokens already registered across "
            f"{where} — compose it from those rather than requesting a new token."
        )
    else:  # absent
        guidance = (
            f"'{token}' is in no grammar class and does not decompose into "
            f"registered tokens; naming this needs a new {segment} token in "
            f"imas-standard-names."
        )

    return GapVerdict(
        category=category,
        segments=tuple(segments_found),
        operators=operators,
        bases=bases,
        guidance=guidance,
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
    for cache in (
        _operator_tokens,
        _grammar_connectives,
        _infix_operator_splits,
        _registered_binary_operator,
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
    "classify_gap",
    "clear_grammar_caches",
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
    "reportable_segments",
    "resolved_base_segment",
]
