"""Jev selects each grammar segment from the closed ISN vocabulary.

Where the composer *generates* a name and the grammar then judges it, this module
*selects*: given a DD source (its path, its documentation and its enriched
description), Jev answers one Choice per grammar slot over that slot's closed
ISN vocabulary, and code renders the chosen tokens through the ISN renderer.
Two properties follow from that order rather than from a filter. The result can
never hold a token the grammar lacks, because every option offered is a
vocabulary token and an optional slot left empty is ``none`` rather than a
guessed word. And the same source yields the same tokens for the same answer,
because nothing between the answer and the name invents a segment.

Selection design
----------------
Four rules shape every question, so the model is asked only what it can answer
well:

- **The base is mandatory.** Every grammar name carries a physical base, so the
  base Choice offers no ``none``.
- **Every option shows its meaning.** An option's text is the token together
  with its definition, read from ISN's public vocabulary loaders (physical
  bases, geometry carriers, the locus registry) or, where none of those carries
  one, from the concept registry's one-sentence definition or the source
  token's registered aliases. Near-synonyms a bare token leaves
  indistinguishable (``density`` against ``number_density``) are told apart by
  their definitions. :func:`definition_coverage` reports, per slot, how many
  offered tokens carry no definition from any of those sources.
- **The base comes from an embedding shortlist.** ISN's base vocabulary is
  ranked against the shortlisted source's description by embedding similarity
  through the project's embedding client, and the base Choice is offered over
  the top :data:`SHORTLIST_SIZE` only. A base the shortlist missed is a
  retrieval miss, recorded on the item so it can be told apart from a judgement
  miss: the item carries both the shortlist and whether its expected base was
  in it.
- **The slots are asked in sequence, grammar-valid only.** The base is chosen
  first; each later slot then offers only tokens that compose with the tokens
  already chosen, checked through
  :func:`imas_codex.standard_names.grammar_adapter.compose_canonical_ir`, the
  one module that decides what the grammar accepts. Optional slots keep
  ``none``, and their question says most names leave the slot empty.

Slots and how each maps onto ISN
-------------------------------
The plan names six slots; ISN keeps its vocabulary and its intermediate
representation under different keys, so the mapping is stated once here and
nowhere else:

======================  ==============================  ==============================
plan slot               ISN vocabulary (segment class)  IR field of ``StandardNameIR``
======================  ==============================  ==============================
``physical_base``       ``SEGMENT_TOKEN_MAP['physical_base']``
                                                        ``base`` (``BaseKind.QUANTITY``)
``subject``             ``SEGMENT_TOKEN_MAP['subject']``
                                                        ``qualifiers`` (``Qualifier``)
``component``           ``SEGMENT_TOKEN_MAP['component']``
                                                        ``projection`` (``AxisProjection``)
``operator``            the operator registry (``OPERATOR_SEGMENT``)
                                                        ``operators`` (``OperatorApplication``)
``locus``               the locus registry (its ``type`` field)
                                                        ``locus`` (``LocusRef``)
``process``             ``SEGMENT_TOKEN_MAP['process']``
                                                        ``mechanism`` (``Process``)
======================  ==============================  ==============================

Every vocabulary token set is read from
:func:`imas_codex.standard_names.segments.grammar_tokens_by_segment` (and its
:func:`~imas_codex.standard_names.segments.locus_types` accessor for the locus
registry), which read ISN live. This module holds no vocabulary snapshot, so it
cannot drift from the grammar it renders through. The same rule governs the
definitions offered as option text: they are read from ISN's loaders at call
time, never copied here.

Asking order and render order
-----------------------------
:data:`SLOT_ORDER` is the order the Choices are asked: the base first, then the
remaining five slots in the order ISN renders them. The renderer's own order is
projection, qualifiers, base, locus, mechanism with the operator stack wrapped
around that core, so the module derives its asking order from the renderer
rather than from the plan's listing.

Large vocabularies as a tree
----------------------------
A slot whose vocabulary ISN keeps as one flat enum is offered as a single
Choice over its tokens. A slot whose vocabulary ISN already partitions — the
operator registry by ``kind`` (unary prefix / unary postfix / binary), the locus
registry by ``type`` (entity / position / region / geometry) — is walked as a
tree: one Choice over the partitions, then one Choice per visited partition over
its tokens. The tree comes from structure the vocabulary carries; nothing here
authors a taxonomy over an unpartitioned enum.

A path is scored by the geometric mean of the per-level choice probabilities,
and a path's score becomes a slot selection's score.

Rendering, and refusal
----------------------
The chosen tokens build a :class:`StandardNameIR`, which
:func:`imas_codex.standard_names.grammar_adapter.compose_canonical_ir` renders
and proves — it composes, canonicalises and requires the parse to preserve the
semantic IR. That adapter is the one module that decides what the grammar
accepts, so the refusal rule is stated once there rather than re-derived here. A
name the grammar refuses is recorded as a failure for that source and never
repaired: a repaired string would be a name the model did not choose.

This module writes no graph state and nothing in the pipeline calls it yet.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.standard_names.grammar_adapter import compose_canonical_ir
from imas_codex.standard_names.segments import (
    OPERATOR_SEGMENT,
    grammar_tokens_by_segment,
    locus_types,
)

logger = logging.getLogger(__name__)

#: Decisions service tag, shared with the other standard-names stages.
SERVICE = "standard-names"

#: Pipeline-cost step these calls are booked to.
COST_STEP = "sn_segment_selection"

#: The seat whose model answers the segment Choices.
MODEL_SEAT = "sn-judge"

#: Paths kept at each level of a slot's tree.
BEAM = 3

#: The option that leaves a slot empty. A Choice always offers it, so an absent
#: optional segment is an answer rather than a token the model invented.
NONE = "none"

#: Question name the answer to one tree node comes back under.
QUESTION_NAME = "segment"

#: How many base tokens the embedding shortlist offers, ranked by similarity.
SHORTLIST_SIZE = 20

#: Plan slot → the ISN segment class whose vocabulary fills it. ``locus`` and
#: ``operator`` are absent here because their vocabulary is a registry rather
#: than a flat ``SEGMENT_TOKEN_MAP`` slot.
SLOT_PHYSICAL_BASE = "physical_base"
SLOT_SUBJECT = "subject"
SLOT_COMPONENT = "component"
SLOT_OPERATOR = "operator"
SLOT_LOCUS = "locus"
SLOT_PROCESS = "process"

#: Slot asking order: the mandatory base first, then every other slot in the
#: order ISN renders it (component, subject, locus, process, with the operator
#: wrapping the core last).
SLOT_ORDER: tuple[str, ...] = (
    SLOT_PHYSICAL_BASE,
    SLOT_COMPONENT,
    SLOT_SUBJECT,
    SLOT_LOCUS,
    SLOT_PROCESS,
    SLOT_OPERATOR,
)

_SEGMENT_CLASS: dict[str, str] = {
    SLOT_PHYSICAL_BASE: "physical_base",
    SLOT_SUBJECT: "subject",
    SLOT_COMPONENT: "component",
    SLOT_PROCESS: "process",
}

#: Locus registry ``type`` values, in the order the tree offers them.
_LOCUS_TYPE_ORDER: tuple[str, ...] = ("entity", "position", "region", "geometry")

#: The source key carrying the item's expected base, when the caller knows it.
#: Used only to record whether that base reached the shortlist — a retrieval
#: miss — so it never changes which options are offered.
EXPECTED_BASE_KEY = "expected_base"


class SegmentRefusal(RuntimeError):
    """One source's chosen tokens do not render to a name the grammar accepts."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Path:
    """One candidate for a slot: the token chosen and the tokens on the way to it.

    ``token`` is ``None`` when the path ends on :data:`NONE`. ``score`` is the
    geometric mean of the per-level choice probabilities of ``route`` combined
    with the terminal choice, so a deeper path is not rewarded for its depth.
    """

    token: str | None
    score: float
    route: tuple[str, ...] = ()


@dataclass
class SlotSelection:
    """The ranked paths for one slot and one source; ``paths[0]`` is the pick."""

    slot: str
    paths: list[Path]

    @property
    def token(self) -> str | None:
        return self.paths[0].token if self.paths else None

    @property
    def score(self) -> float:
        return self.paths[0].score if self.paths else 0.0


@dataclass
class SourceSelection:
    """One source's selections, the name they render to, and any refusal.

    A slot the model never answered (its call failed) is absent from ``slots``.
    A ``name`` is ``None`` exactly when ``failure`` is set.

    ``shortlist`` is the ranked base shortlist the base Choice was offered over,
    and ``base_in_shortlist`` says whether the item's expected base (carried on
    the source under :data:`EXPECTED_BASE_KEY`) appeared in it. That is the
    retrieval miss; a name whose base was offered and not chosen is a judgement
    miss, and the two are told apart by this field rather than by the miss count
    alone.
    """

    source: Mapping[str, Any]
    slots: dict[str, SlotSelection] = field(default_factory=dict)
    name: str | None = None
    shortlist: tuple[str, ...] = ()
    base_in_shortlist: bool | None = None
    failure: str | None = None

    def chosen(self) -> dict[str, str | None]:
        """The chosen token per addressed slot, ``None`` where the slot is empty."""
        return {slot: sel.token for slot, sel in self.slots.items()}


@dataclass
class SelectionRun:
    """Every source's selection plus the decisions cost the run booked.
    """

    items: list[SourceSelection]
    cost_usd: float


# ---------------------------------------------------------------------------
# What each option means
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _vocabulary_definitions() -> dict[str, str]:
    """Token → definition for the vocabulary entries ISN declares one on.

    Read from the public loaders at call time, so a definition added to ISN's
    vocabulary reaches the options without a change here. Returns an empty
    mapping when ISN is unavailable, so the option text then falls through to
    the concept registry and the aliases rather than to a stale copy.
    """
    try:
        from imas_standard_names.grammar.vocab_loaders import (
            load_geometry_carriers,
            load_locus_registry,
            load_physical_bases,
        )
    except ImportError:  # pragma: no cover - ISN absent
        return {}
    definitions: dict[str, str] = {}
    try:
        for token, entry in load_locus_registry().loci.items():
            definitions[token] = entry.definition
        for token, entry in load_physical_bases().bases.items():
            definitions[token] = entry.definition
        for token, entry in load_geometry_carriers().carriers.items():
            definitions[token] = entry.definition
    except Exception:  # pragma: no cover - ISN restructured
        return {}
    return {token: text for token, text in definitions.items() if text}


@lru_cache(maxsize=1)
def _token_aliases() -> dict[str, tuple[str, ...]]:
    """Token → its registered aliases and abbreviations, from ISN's loaders."""
    try:
        from imas_standard_names.grammar.vocab_loaders import (
            load_geometry_carriers,
            load_locus_registry,
            load_physical_bases,
        )
    except ImportError:  # pragma: no cover - ISN absent
        return {}
    aliases: dict[str, tuple[str, ...]] = {}
    try:
        for token, entry in load_locus_registry().loci.items():
            registered = set(entry.abbreviations)
            aliases[token] = tuple(sorted(registered))
        for token, entry in load_physical_bases().bases.items():
            aliases[token] = tuple(sorted(entry.aliases))
        for token, entry in load_geometry_carriers().carriers.items():
            aliases[token] = tuple(sorted(entry.aliases))
    except Exception:  # pragma: no cover - ISN restructured
        return {}
    return {token: names for token, names in aliases.items() if names}


@lru_cache(maxsize=1)
def _concept_entries() -> dict[str, tuple[str, str]]:
    """Token → ``(definition, phrase)`` for tokens the concept registry covers.

    The registry carries a phrase for every token it covers and a definition
    only for a token whose vocabulary entry has none (operators carry none), so
    this is where an operator's meaning comes from. Returns an empty mapping
    when the registry cannot be read, and the option text then falls through to
    the aliases.
    """
    try:
        from imas_standard_names.grammar.vocab_loaders import load_concepts
    except ImportError:  # pragma: no cover - ISN absent
        return {}
    try:
        concepts = load_concepts().concepts
    except Exception:  # pragma: no cover - the concept registry is optional
        return {}
    return {
        token: (entry.definition, entry.phrase) for token, entry in concepts.items()
    }


def _has_definition(token: str) -> bool:
    """Whether any ISN source carries meaning for ``token``."""
    if _vocabulary_definitions().get(token):
        return True
    definition, phrase = _concept_entries().get(token, ("", ""))
    return bool(definition or phrase)


def option_text(token: str) -> str:
    """The text a Choice offers for ``token``: the token and what it means.

    The meaning is taken, in order, from the token's vocabulary definition, the
    concept registry's definition, the concept registry's phrase, and the
    token's registered aliases. A token with none of those still appears, as its
    own words, so the option set is never silently narrowed — but it is counted
    by :func:`definition_coverage` as an option the model must judge from its
    spelling.
    """
    definition = _vocabulary_definitions().get(token)
    if definition:
        return f"{token} — {definition}"
    concept_definition, phrase = _concept_entries().get(token, ("", ""))
    if concept_definition:
        return f"{token} — {' '.join(concept_definition.split())}"
    if phrase:
        return f"{token} — {phrase}"
    aliases = _token_aliases().get(token)
    if aliases:
        return f"{token} (also called {', '.join(aliases)})"
    return token.replace("_", " ")


def definition_coverage() -> dict[str, dict[str, int]]:
    """Per slot, how many offered tokens carry a definition and how many do not.

    The count is the honest one the plan asks for: a token whose only meaning is
    its registered aliases is reported as carrying none, because an alias is a
    second spelling rather than a definition.
    """
    coverage: dict[str, dict[str, int]] = {}
    for slot in SLOT_ORDER:
        tokens = _slot_tokens(slot)
        missing = [token for token in tokens if not _has_definition(token)]
        coverage[slot] = {
            "offered": len(tokens),
            "with_definition": len(tokens) - len(missing),
            "without_definition": len(missing),
        }
    return coverage


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    """One choice point: a partition, or a leaf token.

    ``children`` is empty for a leaf; then ``key`` names the token. A partition
    node's ``key`` names the partition and its ``description`` is the word the
    Choice offers for it.
    """

    key: str
    description: str
    children: tuple[_Node, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def child(self, key: str) -> _Node | None:
        for node in self.children:
            if node.key == key:
                return node
        return None


def _none_leaf() -> _Node:
    return _Node(NONE, "none of the offered tokens fits this slot")


def _leaf_nodes(tokens: Sequence[str], *, include_none: bool = True) -> tuple[_Node, ...]:
    """Token leaves, each carrying its meaning, plus the ``none`` leaf.

    ``include_none`` is False for the base slot, which every name must fill.
    """
    leaves = [_Node(token, option_text(token)) for token in tokens]
    if include_none:
        leaves.append(_none_leaf())
    return tuple(leaves)


def _group_by(
    tokens: Sequence[str], key_of: Mapping[str, str], order: Sequence[str]
) -> list[tuple[str, list[str]]]:
    """Partition ``tokens`` by ``key_of``, ordered by ``order`` then first sight.

    A token whose key is unknown is dropped with a warning rather than grouped
    under an invented label — the vocabulary's own metadata is the only
    authority on which bucket a token belongs to.
    """
    buckets: dict[str, list[str]] = {}
    for token in tokens:
        key = key_of.get(token)
        if key is None:
            logger.warning("segment token %r carries no grouping key; skipped", token)
            continue
        buckets.setdefault(key, []).append(token)
    ordered: list[tuple[str, list[str]]] = []
    for key in order:
        if key in buckets:
            ordered.append((key, sorted(buckets.pop(key))))
    for key in sorted(buckets):
        ordered.append((key, sorted(buckets[key])))
    return ordered


def _operator_kinds() -> dict[str, str]:
    """Operator token → its registry ``kind``, from ISN's operator registry."""
    from imas_standard_names import get_grammar_context

    ops = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    return {
        token: str(spec["kind"])
        for token, spec in ops.items()
        if isinstance(spec, dict) and spec.get("kind")
    }


#: Operator kinds a single chosen token can render. A binary operator needs two
#: operands and one token cannot supply them, so it is not offered as a slot.
_RENDERABLE_OPERATOR_KINDS: tuple[str, ...] = ("unary_prefix", "unary_postfix")


def _slot_tokens(slot: str) -> list[str]:
    """The closed vocabulary for ``slot``, read from ISN, sorted for determinism."""
    if slot == SLOT_LOCUS:
        return sorted(locus_types())
    vocabulary = grammar_tokens_by_segment()
    if slot == SLOT_OPERATOR:
        kinds = _operator_kinds()
        return sorted(
            token
            for token in vocabulary.get(OPERATOR_SEGMENT, ())
            if kinds.get(token) in _RENDERABLE_OPERATOR_KINDS
        )
    return sorted(vocabulary.get(_SEGMENT_CLASS[slot], ()))


@lru_cache(maxsize=512)
def _build_tree(
    slot: str, allowed: tuple[str, ...] | None, include_none: bool
) -> _Node:
    """Build the choice tree for one slot over ``allowed`` (or its whole vocabulary).

    A slot ISN partitions (operator by ``kind``, locus by ``type``) becomes a
    two-level tree: a Choice over the partitions, then a Choice over each
    visited partition's tokens. A slot ISN keeps as one flat enum becomes a
    single Choice over its tokens. The root is only a container; the first asked
    question is over the root's children.
    """
    tokens = list(allowed) if allowed is not None else _slot_tokens(slot)
    if slot == SLOT_OPERATOR:
        groups = _group_by(tokens, _operator_kinds(), _RENDERABLE_OPERATOR_KINDS)
        description = "the operator's arity and position in the rendered name"
    elif slot == SLOT_LOCUS:
        groups = _group_by(tokens, locus_types(), _LOCUS_TYPE_ORDER)
        description = "the locus registry type of the referenced thing"
    else:
        groups = []

    if not groups:
        return _Node(slot, slot, _leaf_nodes(tokens, include_none=include_none))

    partitions = tuple(
        _Node(key, f"{key} ({description})", _leaf_nodes(members, include_none=include_none))
        for key, members in groups
    )
    if include_none:
        # An optional slot is answered "none" at its first Choice too, so a
        # source that uses none of the partitions is never forced into one.
        partitions = (*partitions, _none_leaf())
    return _Node(slot, slot, partitions)


def build_slot_tree(
    slot: str, *, allowed: Sequence[str] | None = None, include_none: bool = True
) -> _Node:
    """The choice tree for one slot, over ``allowed`` when given.

    ``allowed`` is the grammar-filtered token set a caller computed for a
    specific source; ``None`` means the slot's whole vocabulary. ``include_none``
    is False only for the mandatory base.
    """
    key = tuple(allowed) if allowed is not None else None
    return _build_tree(slot, key, include_none)


# ---------------------------------------------------------------------------
# Asking Jev
# ---------------------------------------------------------------------------


def _node_question(node: _Node, *, mandatory: bool) -> dict[str, Any]:
    """The one Choice asked at ``node``: which child fits the source.

    The instructions differ by whether the slot may be left empty. The base slot
    offers no ``none`` at all, so its instructions must not mention one; every
    other slot offers it and says so, because most names leave most slots empty.
    """
    criteria = {child.key: child.description for child in node.children}
    if mandatory:
        instructions = (
            "Choose the option that best fits the source described in the state. "
            "Every standard name carries a base, so one option must be chosen."
        )
    else:
        instructions = (
            "Choose the option that best fits the source described in the state. "
            "Choose 'none' when no offered token applies — most standard names "
            "leave this slot empty."
        )
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


def _node_state(
    source: Mapping[str, Any], slot: str, route: Sequence[str]
) -> dict[str, Any]:
    """The judgement context for one source at one node of a slot's tree."""
    return {
        "slot": slot,
        "route": list(route),
        "source": {
            "path": source.get("path") or "",
            "description": source.get("description") or "",
            "documentation": source.get("documentation") or "",
        },
    }


async def _expand(
    node: _Node,
    sources: Sequence[Mapping[str, Any]],
    *,
    slot: str,
    route: Sequence[str],
    mandatory: bool,
    model: str,
    cost: Any | None,
    spent: list[float],
    concurrency: int,
    budget_seconds: float | None,
) -> list[list[tuple[str, float]] | None]:
    """Ask ``node``'s Choice once per source; return each source's child weights.

    One :func:`decide_batch` call carries every source at this node, because
    they share the node's question. A source whose call failed is reported by
    ``judgment.py`` as ``None``. Every child's probability is returned, sorted
    high to low, so the caller prunes to the beam; the endpoint's chosen option
    is the first of them. The call's summed reported cost is appended to
    ``spent``, so the caller can report the run's cost whether or not a tracker
    was supplied.
    """
    states = [_node_state(source, slot, route) for source in sources]
    results, batch_cost = await decide_batch(
        states,
        _node_question(node, mandatory=mandatory),
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
        cost=cost,
        step=COST_STEP,
    )
    spent.append(batch_cost)
    return [_child_weights(node, result) for result in results]


def _child_weights(
    node: _Node, result: tuple[dict[str, Any], float] | None
) -> list[tuple[str, float]] | None:
    """One answer's (child key, probability) pairs, high first, or ``None``.

    Only children the node actually offers are kept, so an answer cannot widen
    the tree; a probability the endpoint reported as ``0.00`` is kept at that
    weight rather than dropped, since the model still ranks it.
    """
    if result is None:
        return None
    answer = result[0].get(QUESTION_NAME) or {}
    probabilities = answer.get("probabilities") or {}
    offered = {child.key for child in node.children}
    pairs = [
        (key, float(probability))
        for key, probability in probabilities.items()
        if key in offered
    ]
    if not pairs:
        chosen = answer.get("choice")
        if chosen in offered:
            pairs = [(chosen, 0.0)]
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return pairs


# ---------------------------------------------------------------------------
# The beam search
# ---------------------------------------------------------------------------


async def _search_slot(
    root: _Node,
    sources: Sequence[Mapping[str, Any]],
    *,
    model: str,
    cost: Any | None,
    spent: list[float],
    concurrency: int,
    budget_seconds: float | None,
) -> list[list[Path]]:
    """Walk ``root``'s tree for every source; return each source's ranked paths.

    A path is ``(node, cumulative probability, route)``. At each level the
    frontier's interior nodes are grouped so one Choice is asked per node, the
    answers multiply into the paths, and each source's paths are pruned to the
    top :data:`BEAM` by cumulative probability. Because every path from the
    root reaches the same depth, ranking by the product is ranking by the
    geometric mean, and the reported score is that geometric mean.

    A failed expansion contributes no child. A source that still holds a leaf
    path keeps it; a source left with nothing but interior nodes narrows to an
    empty beam and yields no path for the slot, which the caller records as the
    failure it is. The walk therefore always terminates: no node is re-queued,
    so a decisions owner that reports failure for every call costs one pass and
    never re-asks the same Choice.
    """
    mandatory = root.key == SLOT_PHYSICAL_BASE
    frontier: list[list[tuple[_Node, float, tuple[str, ...]]]] = [
        [(root, 1.0, ())] for _ in sources
    ]
    depth = 0

    while True:
        by_node: dict[
            str, tuple[_Node, list[int], list[float], list[tuple[str, ...]]]
        ] = {}
        for index, paths in enumerate(frontier):
            for node, probability, route in paths:
                if node.is_leaf:
                    continue
                entry = by_node.setdefault(node.key, (node, [], [], []))
                entry[1].append(index)
                entry[2].append(probability)
                entry[3].append(route)
        if not by_node:
            break
        depth += 1

        grown: list[list[tuple[_Node, float, tuple[str, ...]]]] = [[] for _ in sources]
        for node, indices, probabilities, routes in by_node.values():
            answers = await _expand(
                node,
                [sources[i] for i in indices],
                slot=root.key,
                route=routes[0],
                mandatory=mandatory,
                model=model,
                cost=cost,
                spent=spent,
                concurrency=concurrency,
                budget_seconds=budget_seconds,
            )
            for position, index in enumerate(indices):
                weights = answers[position]
                if weights is None:
                    continue
                for child_key, weight in weights:
                    child = node.child(child_key)
                    if child is None:  # pragma: no cover - _child_weights filters
                        continue
                    grown[index].append(
                        (
                            child,
                            probabilities[position] * weight,
                            (*routes[position], child_key),
                        )
                    )

        next_frontier: list[list[tuple[_Node, float, tuple[str, ...]]]] = []
        for index, paths in enumerate(frontier):
            kept = [path for path in paths if path[0].is_leaf]
            kept.extend(grown[index])
            # An interior node whose every expansion failed contributes no
            # child, and a source with nothing but interior nodes then has no
            # path left. Re-queuing those nodes would re-ask the same Choice on
            # every pass without ever narrowing, so the beam narrows to empty:
            # the source carries no selection for this slot, and the caller
            # records the missing slot as the empty-beam failure.
            kept.sort(key=lambda item: (-item[1], item[0].key))
            next_frontier.append(kept[:BEAM])
        frontier = next_frontier

    ranked: list[list[Path]] = []
    for paths in frontier:
        leaves: list[Path] = []
        for node, probability, route in paths:
            if not node.is_leaf:
                continue
            score = probability ** (1.0 / depth) if depth else probability
            leaves.append(
                Path(
                    token=None if node.key == NONE else node.key,
                    score=score,
                    route=route,
                )
            )
        leaves.sort(key=lambda path: -path.score)
        ranked.append(leaves[:BEAM])
    return ranked


# ---------------------------------------------------------------------------
# Grammar-valid tokens only
# ---------------------------------------------------------------------------


def _composes(
    chosen: Mapping[str, str | None], slot: str, token: str
) -> bool:
    """Whether ``token`` composes with the tokens already chosen for ``chosen``.

    The test is the grammar's own: the trial token set is rendered through
    :func:`imas_codex.standard_names.grammar_adapter.compose_canonical_ir` by way
    of :func:`render_name`, so a token is admitted only when the adapter accepts
    the name it produces. Nothing else is consulted, and no rule is re-derived
    here.
    """
    trial = dict(chosen)
    trial[slot] = token
    try:
        render_name(trial)
    except SegmentRefusal:
        return False
    return True


@lru_cache(maxsize=4096)
def _admissible_cached(
    slot: str, chosen_items: tuple[tuple[str, str | None], ...]
) -> tuple[str, ...]:
    chosen = dict(chosen_items)
    return tuple(
        token for token in _slot_tokens(slot) if _composes(chosen, slot, token)
    )


def admissible(slot: str, chosen: Mapping[str, str | None]) -> tuple[str, ...]:
    """The slot's tokens that compose with the tokens already chosen.

    Cached on the (slot, chosen tokens) pair because the same pair recurs across
    a batch: every source whose earlier answers matched shares one computation,
    and the token list is the slot's whole vocabulary each time.
    """
    items = tuple(sorted((key, value) for key, value in chosen.items() if value))
    return _admissible_cached(slot, items)


# ---------------------------------------------------------------------------
# The base shortlist
# ---------------------------------------------------------------------------


def _embed(texts: list[str]) -> Sequence[Sequence[float]]:
    """Embed ``texts`` through the project's embedding client.

    One indirection so the retrieval can be stubbed in a test without loading a
    model: the identity of the client is this function, not every embedded
    choice. Raises whatever the client raises — a retrieval that cannot be done
    is not silently turned into an empty shortlist.
    """
    from imas_codex.embeddings.embed import get_encoder

    return get_encoder().embed_texts(texts)


def _source_text(source: Mapping[str, Any]) -> str:
    """The text a source is retrieved by: its description, then documentation."""
    parts = [
        str(source.get("description") or ""),
        str(source.get("documentation") or ""),
    ]
    text = " ".join(part for part in parts if part).strip()
    return text or str(source.get("path") or "").replace("/", " ").strip()


def _retrieval_text(token: str) -> str:
    """The text a base token is retrieved by: its words and what it means."""
    words = token.replace("_", " ")
    definition = _vocabulary_definitions().get(token)
    if definition:
        return f"{words}: {definition}"
    concept_definition, phrase = _concept_entries().get(token, ("", ""))
    if concept_definition:
        return f"{words}: {' '.join(concept_definition.split())}"
    if phrase:
        return f"{words}: {phrase}"
    return words


def base_shortlists(
    sources: Sequence[Mapping[str, Any]], *, size: int = SHORTLIST_SIZE
) -> list[list[str]]:
    """Rank the base vocabulary against each source; return the top ``size`` each.

    The ranking is the cosine similarity of the source's description against
    each base token's meaning, taken as a dot product because the encoder
    normalises its vectors. The shortlist is a *retrieval* step and nothing more:
    it narrows what the base Choice is offered over, and the caller records
    whether the item's expected base survived it, so a base the retrieval missed
    is never confused with one the model passed over.
    """
    tokens = _slot_tokens(SLOT_PHYSICAL_BASE)
    if not tokens:
        return [[] for _ in sources]
    size = max(1, min(size, len(tokens)))
    texts = [_retrieval_text(token) for token in tokens]
    queries = [_source_text(source) for source in sources]
    vectors = [
        [float(value) for value in vector] for vector in _embed([*texts, *queries])
    ]
    base_vectors = vectors[: len(tokens)]
    query_vectors = vectors[len(tokens) :]
    shortlists: list[list[str]] = []
    for query in query_vectors:
        scored = [
            (sum(left * right for left, right in zip(query, vector)), token)
            for vector, token in zip(base_vectors, tokens)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        shortlists.append([token for _, token in scored[:size]])
    return shortlists


# ---------------------------------------------------------------------------
# The asking sequence
# ---------------------------------------------------------------------------


async def _search_groups(
    slot: str,
    sources: Sequence[Mapping[str, Any]],
    allowed_per_source: Sequence[Sequence[str]],
    *,
    include_none: bool,
    model: str,
    cost: Any | None,
    spent: list[float],
    concurrency: int,
    budget_seconds: float | None,
) -> list[list[Path]]:
    """Walk ``slot``'s tree for each source over its own allowed tokens.

    Sources whose allowed sets are identical share one tree and one set of
    Choices, so a batch pays for the common cases once rather than once per
    source. Results are scattered back into input order.
    """
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, allowed in enumerate(allowed_per_source):
        groups.setdefault(tuple(allowed), []).append(index)

    ranked: list[list[Path]] = [[] for _ in sources]
    for allowed, indices in groups.items():
        tree = build_slot_tree(slot, allowed=allowed, include_none=include_none)
        subset = await _search_slot(
            tree,
            [sources[index] for index in indices],
            model=model,
            cost=cost,
            spent=spent,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
        )
        for position, index in enumerate(indices):
            ranked[index] = subset[position]
    return ranked


async def _select_sequence(
    sources: Sequence[Mapping[str, Any]],
    *,
    shortlists: Sequence[Sequence[str]],
    model: str,
    cost: Any | None,
    spent: list[float],
    concurrency: int,
    budget_seconds: float | None,
) -> tuple[dict[str, list[list[Path]]], list[dict[str, str | None]]]:
    """Ask every slot in sequence, each offering only grammar-valid tokens.

    The base is asked first, over ``shortlists``. Each later slot's allowed set
    is then derived from that source's own chosen tokens so far, so a token is
    offered only when it composes with them. The pick at each slot feeds the
    next slot's filter, which is what makes the sequence conditional rather than
    six independent questions.
    """
    chosen: list[dict[str, str | None]] = [{} for _ in sources]
    selections: dict[str, list[list[Path]]] = {}
    for slot in SLOT_ORDER:
        if slot == SLOT_PHYSICAL_BASE:
            allowed_per_source: list[Sequence[str]] = list(shortlists)
            include_none = False
        else:
            allowed_per_source = [
                admissible(slot, entry) for entry in chosen
            ]
            include_none = True
        ranked = await _search_groups(
            slot,
            sources,
            allowed_per_source,
            include_none=include_none,
            model=model,
            cost=cost,
            spent=spent,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
        )
        selections[slot] = ranked
        for index, paths in enumerate(ranked):
            if paths:
                chosen[index][slot] = paths[0].token
    return selections, chosen


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _build_ir(chosen: Mapping[str, str | None]) -> Any:
    """Build the IR from the chosen tokens; raise :class:`SegmentRefusal` on a gap.

    The maps onto ISN's IR are the ones the module docstring states. A slot left
    empty contributes nothing; a name with no ``physical_base`` has no IR.
    """
    from imas_standard_names.grammar.ir import (
        AxisProjection,
        BaseKind,
        LocusRef,
        LocusType,
        OperatorApplication,
        OperatorKind,
        Process,
        ProjectionShape,
        Qualifier,
        QuantityOrCarrier,
        StandardNameIR,
    )

    base_token = chosen.get(SLOT_PHYSICAL_BASE)
    if not base_token:
        raise SegmentRefusal("no physical_base was chosen")

    operators = []
    operator_token = chosen.get(SLOT_OPERATOR)
    if operator_token:
        kind = _operator_kinds().get(operator_token)
        if kind not in _RENDERABLE_OPERATOR_KINDS:
            raise SegmentRefusal(
                f"operator {operator_token!r} has no renderable kind ({kind!r})"
            )
        operators.append(
            OperatorApplication(
                kind=OperatorKind(kind),
                op=operator_token,
                bare_prefix=kind == "unary_prefix",
            )
        )

    qualifiers = []
    subject = chosen.get(SLOT_SUBJECT)
    if subject:
        qualifiers.append(Qualifier(token=subject))

    projection = None
    component = chosen.get(SLOT_COMPONENT)
    if component:
        projection = AxisProjection(axis=component, shape=ProjectionShape.COMPONENT)

    locus = None
    locus_token = chosen.get(SLOT_LOCUS)
    if locus_token:
        locus_type = locus_types().get(locus_token)
        if locus_type is None:
            raise SegmentRefusal(f"locus {locus_token!r} carries no registry type")
        relation = _locus_relation(locus_token, locus_type)
        locus = LocusRef(
            relation=relation, token=locus_token, type=LocusType(locus_type)
        )

    mechanism = None
    process = chosen.get(SLOT_PROCESS)
    if process:
        mechanism = Process(token=process)

    return StandardNameIR(
        operators=operators,
        projection=projection,
        qualifiers=qualifiers,
        base=QuantityOrCarrier(token=base_token, kind=BaseKind.QUANTITY),
        locus=locus,
        mechanism=mechanism,
    )


def _locus_relation(token: str, locus_type: str) -> Any:
    """A relation the IR's own matrix permits for a locus of this registry type.

    The IR's ``LOCUS_RELATION_MATRIX`` is the authority on which relation a locus
    type admits; the registry's ``allowed_relations`` is not that set — a
    geometry-typed locus lists ``at`` there while the IR permits only ``of`` — so
    reading the registry would build a ``LocusRef`` the IR refuses. ``of`` is
    preferred where permitted, as the intrinsic-geometry reading.
    """
    from imas_standard_names.grammar.ir import (
        LOCUS_RELATION_MATRIX,
        LocusRelation,
        LocusType,
    )

    try:
        permitted = LOCUS_RELATION_MATRIX[LocusType(locus_type)]
    except ValueError as exc:
        raise SegmentRefusal(
            f"locus {token!r} has unknown registry type {locus_type!r}"
        ) from exc
    if not permitted:
        raise SegmentRefusal(f"locus type {locus_type!r} permits no relation")
    if LocusRelation.OF in permitted:
        return LocusRelation.OF
    return sorted(permitted, key=lambda relation: relation.value)[0]


def render_name(chosen: Mapping[str, str | None]) -> str:
    """Render the chosen tokens to a name the grammar accepts, or refuse.

    Two refusals are possible and both are the grammar's, not this module's:
    the IR cannot be built (a slot holds a token with no place in the IR), or
    :func:`~imas_codex.standard_names.grammar_adapter.compose_canonical_ir`
    refuses the IR it is handed. That adapter is the one place that composes an
    IR and proves the rendered name canonical and IR-preserving, so this module
    does not re-implement the compose-then-parse pair beside it. Neither refusal
    is repaired — the caller records the failure for that source.
    """
    try:
        ir = _build_ir(chosen)
    except SegmentRefusal:
        raise
    except Exception as exc:  # noqa: BLE001 - a refusal is data
        raise SegmentRefusal(f"the IR refused the chosen tokens: {exc}") from exc
    try:
        return compose_canonical_ir(ir)
    except Exception as exc:  # noqa: BLE001 - a refusal is data
        raise SegmentRefusal(f"the grammar refused the IR: {exc}") from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def select_segments(
    sources: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    cost: Any | None = None,
    concurrency: int = 8,
    budget_seconds: float | None = None,
    shortlist_size: int = SHORTLIST_SIZE,
) -> SelectionRun:
    """Select every grammar slot for each source and render the name.

    Each source is a mapping carrying ``path``, ``description`` (its enriched
    description) and ``documentation``; it may also carry ``expected_base`` when
    the caller knows the base the source should select, which is recorded on the
    item rather than used to steer the offer. Every slot's tree is walked for the
    whole batch through the shared decisions owner
    (:func:`imas_codex.discovery.base.judgment.decide_batch`) under the
    ``sn-judge`` seat and the ``standard-names`` service, so the retry policy,
    the concurrency bound and the cost booking live in one place. The base is
    asked first over an embedding shortlist; every later slot offers only tokens
    that compose with the tokens already chosen.

    Args:
        sources: The DD sources to select segments for.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        cost: Optional run cost tracker; every call's reported cost is booked
            to it under ``sn_segment_selection``.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time budget per Choice node; ``None`` waits.
        shortlist_size: How many base tokens the shortlist offers the base
            Choice.

    Returns:
        A :class:`SelectionRun` whose ``items`` are one
        :class:`SourceSelection` per source, in input order, and whose
        ``cost_usd`` is the summed reported cost of every call.

    Raises:
        SegmentRefusal: Only from an explicitly invoked render; a rendering
            is recorded on the item's ``failure``, never raised here.
    """
    if model is None:
        from imas_codex.settings import get_model

        model = get_model(MODEL_SEAT)

    items = [SourceSelection(source=source) for source in sources]
    if not items:
        return SelectionRun(items=[], cost_usd=0.0)

    spent: list[float] = []
    shortlists = base_shortlists(sources, size=shortlist_size)
    selections, _ = await _select_sequence(
        sources,
        shortlists=shortlists,
        model=model,
        cost=cost,
        spent=spent,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
    )

    for index, item in enumerate(items):
        item.shortlist = tuple(shortlists[index])
        expected = sources[index].get(EXPECTED_BASE_KEY)
        item.base_in_shortlist = (
            None if not expected else expected in item.shortlist
        )
        for slot in SLOT_ORDER:
            paths = selections[slot][index]
            if not paths:
                continue
            item.slots[slot] = SlotSelection(slot=slot, paths=paths)
        try:
            item.name = render_name(item.chosen())
        except SegmentRefusal as exc:
            item.failure = str(exc)

    return SelectionRun(items=items, cost_usd=sum(spent))
