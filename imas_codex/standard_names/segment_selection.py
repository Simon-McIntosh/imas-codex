"""Jev selects each grammar segment from the closed ISN vocabulary.

Where the composer *generates* a name and the grammar then judges it, this module
*selects*: given a DD source (its path, its documentation and its enriched
description), Jev answers one Choice per grammar slot over that slot's closed
ISN vocabulary, and code renders the chosen tokens through the ISN renderer.
Two properties follow from that order rather than from a filter. The result can
never hold a token the grammar lacks, because every option offered is a
vocabulary token and a slot left empty is ``none`` rather than a guessed word.
And the same source yields the same tokens for the same answer, because nothing
between the answer and the name invents a segment.

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
cannot drift from the grammar it renders through.

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
and a beam of :data:`BEAM` keeps the best paths at each level, which is the
hierarchical-classification method the plan fixes.

Rendering, and refusal
----------------------
The chosen tokens build a :class:`StandardNameIR`, which
:func:`imas_standard_names.grammar.render.compose` renders. The rendered name is
then checked with :func:`imas_standard_names.grammar.parse` at ``strict=True`` —
ISN's authoritative validity oracle. A name the grammar refuses is recorded as a
failure for that source and never repaired: a repaired string would be a name
the model did not choose.

This module writes no graph state and nothing in the pipeline calls it yet.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
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

#: Paths kept by the beam at each level.
BEAM = 3

#: The option that leaves a slot empty. A Choice always offers it, so an absent
#: segment is an answer rather than a token the model invented.
NONE = "none"

#: Question name the answer to one tree node comes back under.
QUESTION_NAME = "segment"

#: Plan slot → the ISN segment class whose vocabulary fills it. ``locus`` and
#: ``operator`` are absent here because their vocabulary is a registry rather
#: than a flat ``SEGMENT_TOKEN_MAP`` slot.
SLOT_PHYSICAL_BASE = "physical_base"
SLOT_SUBJECT = "subject"
SLOT_COMPONENT = "component"
SLOT_OPERATOR = "operator"
SLOT_LOCUS = "locus"
SLOT_PROCESS = "process"

#: Slot order, outermost name fragment first; also the order the Choices are asked.
SLOT_ORDER: tuple[str, ...] = (
    SLOT_PHYSICAL_BASE,
    SLOT_SUBJECT,
    SLOT_COMPONENT,
    SLOT_OPERATOR,
    SLOT_LOCUS,
    SLOT_PROCESS,
)

_SEGMENT_CLASS: dict[str, str] = {
    SLOT_PHYSICAL_BASE: "physical_base",
    SLOT_SUBJECT: "subject",
    SLOT_COMPONENT: "component",
    SLOT_PROCESS: "process",
}

#: Locus registry ``type`` values, in the order the tree offers them.
_LOCUS_TYPE_ORDER: tuple[str, ...] = ("entity", "position", "region", "geometry")


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
    """

    source: Mapping[str, Any]
    slots: dict[str, SlotSelection] = field(default_factory=dict)
    name: str | None = None
    failure: str | None = None

    def chosen(self) -> dict[str, str | None]:
        """The chosen token per addressed slot, ``None`` where the slot is empty."""
        return {slot: sel.token for slot, sel in self.slots.items()}


@dataclass
class SelectionRun:
    """Every source's selection plus the decisions cost the run booked."""

    items: list[SourceSelection]
    cost_usd: float


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


def _leaf_nodes(tokens: Sequence[str]) -> tuple[_Node, ...]:
    """Token leaves plus the ``none`` leaf every choice offers."""
    leaves = [_Node(token, token) for token in tokens]
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


def build_slot_tree(slot: str) -> _Node:
    """The choice tree for one slot, built from the structure its vocabulary carries.

    A slot ISN partitions (operator by ``kind``, locus by ``type``) becomes a
    two-level tree: a Choice over the partitions, then a Choice over each
    visited partition's tokens. A slot ISN keeps as one flat enum becomes a
    single Choice over its tokens. The root is only a container; the first asked
    question is over the root's children.
    """
    tokens = _slot_tokens(slot)
    if slot == SLOT_OPERATOR:
        groups = _group_by(tokens, _operator_kinds(), _RENDERABLE_OPERATOR_KINDS)
        description = "the operator's arity and position in the rendered name"
    elif slot == SLOT_LOCUS:
        groups = _group_by(tokens, locus_types(), _LOCUS_TYPE_ORDER)
        description = "the locus registry type of the referenced thing"
    else:
        groups = []

    if not groups:
        return _Node(slot, slot, _leaf_nodes(tokens))

    partitions = tuple(
        _Node(key, f"{key} ({description})", _leaf_nodes(members))
        for key, members in groups
    )
    return _Node(slot, slot, partitions)


# ---------------------------------------------------------------------------
# Asking Jev
# ---------------------------------------------------------------------------


def _node_question(node: _Node) -> dict[str, Any]:
    """The one Choice asked at ``node``: which child fits the source."""
    criteria = {child.key: child.description for child in node.children}
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "Choose the option that best fits the source described in the "
                "state. Answer 'none' when no token fits the slot."
            ),
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
    model: str,
    cost: Any | None,
    spent: list[float],
    concurrency: int,
    budget_seconds: float | None,
) -> list[list[tuple[str, float]] | None]:
    """Ask ``node``'s Choice once per source; return each source's child weights.

    One :func:`decide_batch` call carries every source at this node, because
    they share the node's question. A source whose call failed (``None). Every
    child's probability is returned, sorted high to low, so the caller prunes to
    the beam; the endpoint's chosen option is the first of them. The call's
    summed reported cost is appended to ``spent``, so the caller can report the
    run's cost whether or not a tracker was supplied.
    """
    states = [_node_state(source, slot, route) for source in sources]
    results, batch_cost = await decide_batch(
        states,
        _node_question(node),
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
        cost=cost,
        step=COST_STEP,
    )
    spent.append(batch_cost)
    weights: list[list[tuple[str, float]] | None] = []
    for result in results:
        weights.append(_child_weights(node, result))
    return weights


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

    A source that fails an expansion keeps the paths it already holds, so a
    single failed call narrows the beam rather than discarding the slot.
    """
    # frontier[i] = list of (node, probability product, route)
    frontier: list[list[tuple[_Node, float, tuple[str, ...]]]] = [
        [(root, 1.0, ())] for _ in sources
    ]
    depth = 0

    while True:
        # Which (node, sources) to expand: every interior node in a frontier.
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

        # Answers land here, keyed by source, as new (node, probability, route).
        grown: list[list[tuple[_Node, float, tuple[str, ...]]]] = [[] for _ in sources]
        for node, indices, probabilities, routes in by_node.values():
            answers = await _expand(
                node,
                [sources[i] for i in indices],
                slot=root.key,
                route=routes[0],
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

        # Carry leaf paths forward and prune each source's beam.
        next_frontier: list[list[tuple[_Node, float, tuple[str, ...]]]] = []
        for index, paths in enumerate(frontier):
            kept = [path for path in paths if path[0].is_leaf]
            kept.extend(grown[index])
            if not kept:
                kept = list(paths)  # every expansion failed: hold what we had
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
    ISN's strict parser rejects the rendered name. Neither is repaired — the
    caller records the failure for that source.
    """
    from imas_standard_names.grammar import parse
    from imas_standard_names.grammar.render import compose

    try:
        ir = _build_ir(chosen)
    except SegmentRefusal:
        raise
    except Exception as exc:  # noqa: BLE001 - a refusal is data
        raise SegmentRefusal(f"the IR refused the chosen tokens: {exc}") from exc
    try:
        name = compose(ir)
    except Exception as exc:  # noqa: BLE001 - a refusal is data
        raise SegmentRefusal(f"compose refused the IR: {exc}") from exc
    try:
        parse(name, strict=True)
    except Exception as exc:  # noqa: BLE001 - a refusal is data
        raise SegmentRefusal(f"grammar refused {name!r}: {exc}") from exc
    return name


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
) -> SelectionRun:
    """Select every grammar slot for each source and render the name.

    Each source is a mapping carrying ``path``, ``description`` (its enriched
    description) and ``documentation``. Every slot's tree is walked for the
    whole batch through the shared decisions owner
    (:func:`imas_codex.discovery.base.judgment.decide_batch`) under the
    ``sn-judge`` seat and the ``standard-names`` service, so the retry policy,
    the concurrency bound and the cost booking live in one place.

    Args:
        sources: The DD sources to select segments for.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        cost: Optional run cost tracker; every call's reported cost is booked
            to it under ``sn_segment_selection``.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time budget per Choice node; ``None`` waits.

    Returns:
        A :class:`SelectionRun` whose ``items`` are one
        :class:`SourceSelection` per source, in input order, and whose
        ``cost_usd`` is the summed reported cost of every call.

    Raises:
        SegmentRefusal: Only from an explicitly invoked render; a rendering
            refusal is recorded on the item's ``failure``, never raised here.
    """
    if model is None:
        from imas_codex.settings import get_model

        model = get_model(MODEL_SEAT)

    items = [SourceSelection(source=source) for source in sources]
    if not items:
        return SelectionRun(items=[], cost_usd=0.0)

    spent: list[float] = []

    trees = {slot: build_slot_tree(slot) for slot in SLOT_ORDER}
    selections: dict[str, list[list[Path]]] = {}
    for slot in SLOT_ORDER:
        ranked = await _search_slot(
            trees[slot],
            sources,
            model=model,
            cost=cost,
            spent=spent,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
        )
        selections[slot] = ranked

    for index, item in enumerate(items):
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
