"""Deterministic DD candidate retrieval for signal-source mapping.

Builds, for each signal source, a shortlist of real Data Dictionary nodes to
choose an IMAS target from. Retrieval calls no language model: each source
description is embedded once and searched against the DD with
:func:`~imas_codex.graph.dd_search.hybrid_dd_search`.

A source may be routed to several IDSs (the IDS choice that precedes
retrieval). Each routed IDS is searched as its own *arm* — a separate
``hybrid_dd_search`` call restricted to that IDS — and the arms are merged by
score and deduplicated on path, so a path returned by more than one arm yields
a single candidate carrying the union of the arms that produced it. A source
with no routed IDSs runs a single unscoped arm.

:class:`Candidate` composes the :class:`SearchHit` the search returns rather
than copying its fields, and adds the arms that produced the hit and the
documentation of its parent DD node.

On top of retrieval the module owns the candidate stage's judgment and routing:
:func:`route_ids` asks a decisions model which IDSs would hold a source's
values, :func:`judge_candidates` asks one batched ``same_quantity`` question per
candidate, and :func:`route` orders the judged candidates and decides whether
the source is selected, escalated or rejected. A decisions transport failure
returns no route, leaving the source unjudged for a retry.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from imas_codex.discovery.base.llm import DecisionsValidationError, call_decisions
from imas_codex.graph.dd_search import hybrid_dd_search, related_dd_search
from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit
from imas_codex.settings import (
    RouteThresholds,
    get_mapping_route_thresholds,
    get_model,
)

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.mapping import PipelineCost

logger = logging.getLogger(__name__)

#: Default shortlist size per source.
DEFAULT_K: Final[int] = 20

#: Identifier of the arm that searches the whole DD, used when a source carries
#: no routed IDSs. Real IDS names never take this value.
UNSCOPED_ARM: Final[str] = "*"

#: Service tag for the candidate stage's decisions calls and their API key.
JUDGMENT_SERVICE: Final[str] = "imas-mapping"

#: Number of most probable IDSs the routing choice returns.
ROUTED_IDS: Final[int] = 3

#: Number of a source's top-ranked candidates whose cross-IDS clusters seed the
#: sibling expansion.
CLUSTER_SEED_COUNT: Final[int] = 5

#: Maximum new cross-IDS siblings added to one source's shortlist.
CLUSTER_SIBLING_CAP: Final[int] = 10

#: Identifier of the arm that returns a path as a cross-IDS cluster sibling.
CLUSTER_ARM: Final[str] = "cluster"

#: Characters of a DD node's documentation carried into a judgment.
_CANDIDATE_DOC_CHARS: Final[int] = 400


@dataclass(frozen=True, slots=True)
class Candidate:
    """A DD node retrieved for one signal source.

    Composes the :class:`SearchHit` from ``hybrid_dd_search`` — its path, IDS,
    documentation, units, data type, physics domain and score are read from
    ``hit`` and never copied — and records the retrieval arms that returned it
    and the documentation of its parent DD node.
    """

    hit: SearchHit
    arms: frozenset[str] = frozenset()
    parent_documentation: str | None = None


@dataclass(frozen=True, slots=True)
class PairJudgment:
    """One candidate judged against a signal source by the decisions model.

    ``p_same_quantity`` is the model's probability that the DD field at
    ``path`` correctly holds the values the source provides, ``model`` the
    decisions model that produced it and ``judged_at`` an ISO-8601 UTC stamp.
    """

    path: str
    p_same_quantity: float
    model: str
    judged_at: str


@dataclass(frozen=True, slots=True)
class Route:
    """The routing decision for one judged source.

    ``decision`` is ``"selected"`` when a select threshold is set and any
    candidate reaches it, ``"no_candidate"`` when a floor is set and no score
    reaches it, and ``"escalated"`` otherwise. ``shortlist`` is the top
    candidates in Jev order (descending ``p_same_quantity``). ``selected`` holds
    the path of every candidate at or above the select threshold.
    """

    decision: str
    shortlist: tuple[PairJudgment, ...]
    selected: frozenset[str] = frozenset()


@dataclass
class _MergedHit:
    """A hit kept during the arm merge, with the arms that returned it."""

    hit: SearchHit
    arms: set[str] = field(default_factory=set)


def retrieve_candidates(
    sources: Mapping[str, str],
    ids_by_source: Mapping[str, Sequence[str]],
    *,
    gc: GraphClient,
    k: int = DEFAULT_K,
    dd_version: int | None = None,
) -> dict[str, list[Candidate]]:
    """Retrieve a ranked DD shortlist for every signal source.

    Args:
        sources: Source id -> description text. Every description is embedded
            in a single encoder call.
        ids_by_source: Source id -> routed IDS names. A source absent from the
            mapping, or mapping to an empty sequence, is searched unscoped.
        gc: Active graph client.
        k: Shortlist size per source.
        dd_version: DD major version to scope the search to.

    Returns:
        Source id -> candidates ordered by descending score, at most ``k``.
    """
    source_ids = list(sources)
    result: dict[str, list[Candidate]] = {sid: [] for sid in source_ids}
    if not source_ids:
        return result

    texts = [sources[sid] or "" for sid in source_ids]

    from imas_codex.embeddings.encoder import Encoder

    encoder = Encoder()
    embeddings = encoder.embed_texts(texts)

    merged: dict[str, dict[str, _MergedHit]] = {}
    for index, sid in enumerate(source_ids):
        description = texts[index]
        if not description.strip():
            continue
        routed = list((ids_by_source or {}).get(sid) or [])
        merged[sid] = _merge_arms(
            gc,
            description,
            _as_vector(embeddings[index]),
            routed,
            k=k,
            dd_version=dd_version,
        )

    parent_docs = _fetch_parent_documentation(
        gc, sorted({path for hits in merged.values() for path in hits})
    )

    for sid, hits in merged.items():
        ordered = sorted(hits.items(), key=lambda item: item[1].hit.score, reverse=True)
        result[sid] = [
            Candidate(
                hit=entry.hit,
                arms=frozenset(entry.arms),
                parent_documentation=parent_docs.get(path),
            )
            for path, entry in ordered[:k]
        ]
    return result


def _merge_arms(
    gc: GraphClient,
    description: str,
    embedding: list[float],
    routed_ids: Sequence[str],
    *,
    k: int,
    dd_version: int | None,
) -> dict[str, _MergedHit]:
    """Search each arm and merge its hits by score, deduplicated on path."""
    arm_ids = list(routed_ids) or [UNSCOPED_ARM]
    merged: dict[str, _MergedHit] = {}
    for arm in arm_ids:
        ids_filter = None if arm == UNSCOPED_ARM else arm
        hits = hybrid_dd_search(
            gc,
            description,
            ids_filter=ids_filter,
            dd_version=dd_version,
            k=k,
            embedding=embedding,
        )
        for hit in hits:
            existing = merged.get(hit.path)
            if existing is None:
                merged[hit.path] = _MergedHit(hit=hit, arms={arm})
            else:
                if hit.score > existing.hit.score:
                    existing.hit = hit
                existing.arms.add(arm)
    return merged


def _fetch_parent_documentation(
    gc: GraphClient, paths: Sequence[str]
) -> dict[str, str | None]:
    """Return documentation of each path's parent IMASNode, in one query."""
    if not paths:
        return {}
    rows = gc.query(
        """
        UNWIND $paths AS pid
        MATCH (p:IMASNode {id: pid})
        OPTIONAL MATCH (p)-[:HAS_PARENT]->(parent:IMASNode)
        RETURN p.id AS id, parent.documentation AS parent_documentation
        """,
        paths=list(paths),
    )
    return {row["id"]: row.get("parent_documentation") for row in rows or []}


def _as_vector(embedding: object) -> list[float]:
    """Coerce one encoder row to a plain float list."""
    tolist = getattr(embedding, "tolist", None)
    if callable(tolist):
        return list(tolist())
    return list(embedding)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pair judgment and routing through the decisions model
# ---------------------------------------------------------------------------
# Each source costs two decisions requests: one Choice that routes it to the
# three most probable IDSs, and one batched same_quantity judgment over its
# retrieved candidates. Both travel the shared ``call_decisions`` layer; no
# second client is built here. A transport failure returns no route so the
# source stays unjudged for a later retry, while a contract violation is raised
# and never mistaken for a usable judgment.


def _judgment_templates() -> dict[str, Any]:
    """Render the candidate-judgment question templates from their prompt file."""
    import json

    from imas_codex.llm.prompt_loader import render_prompt

    return json.loads(render_prompt("mapping/candidate_judgment", {}))


def _call_judgment(
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    *,
    service: str,
    step: str,
    cost: PipelineCost | None = None,
) -> dict[str, Any] | None:
    """Run one decisions call, returning ``None`` when the transport fails.

    A malformed answer (a :class:`DecisionsValidationError`) is raised, not
    swallowed: it is a deterministic contract violation to reject, never a
    reason to leave a source silently unjudged. Any other failure returns
    ``None`` so the source is neither selected nor rejected and can be retried.

    The transport's reported ``usage.cost`` is added to ``cost`` under ``step``
    when a tracker is passed, the way the map pipeline's ``_acall_llm`` does.
    """
    try:
        answers, spent = call_decisions(model, state, questions, service=service)
    except DecisionsValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on any transport fault
        logger.warning("candidate decisions call failed: %s", exc)
        return None
    if cost is not None:
        cost.add(step, spent, 0)
    return answers


def _ids_criteria(gc: GraphClient) -> dict[str, str]:
    """Every IDS id keyed to its one-line description, drawn from the graph."""
    rows = gc.query(
        "MATCH (i:IDS) RETURN i.id AS id, i.description AS description ORDER BY i.id"
    )
    return {
        row["id"]: (row.get("description") or row["id"])[:240] for row in rows or []
    }


def route_ids(
    description: str,
    *,
    gc: GraphClient,
    model: str | None = None,
    service: str = JUDGMENT_SERVICE,
    cost: PipelineCost | None = None,
) -> list[str] | None:
    """Ask the decisions model which IDSs would hold a source's values.

    One Choice question is asked over every IDS in the graph, each IDS's name
    and one-line description offered as its criterion. Returns the three most
    probable IDSs, drawn from the criteria that were offered, or ``None`` when
    the decisions call fails.

    Args:
        description: The signal source's description text.
        gc: Active graph client, read for the IDS criteria.
        model: Decisions model id; defaults to the mapping-candidates seat.
        service: Service tag for the API key and headers.
        cost: Optional run cost tracker; the decisions call's reported cost is
            added to it under the ``candidate_route`` step.
    """
    criteria = _ids_criteria(gc)
    if not criteria:
        return None
    routing = dict(_judgment_templates()["ids_routing"])
    routing["criteria"] = criteria
    resolved_model = model or get_model("mapping-candidates")
    state = {"signal_source": {"description": description}}
    answers = _call_judgment(
        resolved_model,
        state,
        {"ids_routing": routing},
        service=service,
        step="candidate_route",
        cost=cost,
    )
    if answers is None:
        return None
    probabilities = answers["ids_routing"].get("probabilities") or {}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:ROUTED_IDS]]


def _judgment_state(
    source: Mapping[str, Any],
    facility: Mapping[str, Any],
    candidates: Sequence[Candidate],
) -> dict[str, Any]:
    """Build the facility, source and candidate blocks a judgment is asked over."""
    return {
        "facility": dict(facility),
        "signal_source": dict(source),
        "candidates": [
            {
                "path": candidate.hit.path,
                "ids_name": candidate.hit.ids_name,
                "documentation": (candidate.hit.documentation or "")[
                    :_CANDIDATE_DOC_CHARS
                ],
                "unit": candidate.hit.units,
                "data_type": candidate.hit.data_type,
                "physics_domain": candidate.hit.physics_domain,
                "parent_documentation": (candidate.parent_documentation or "")[
                    :_CANDIDATE_DOC_CHARS
                ],
            }
            for candidate in candidates
        ],
    }


def judge_candidates(
    source: Mapping[str, Any],
    facility: Mapping[str, Any],
    candidates: Sequence[Candidate],
    *,
    model: str | None = None,
    service: str = JUDGMENT_SERVICE,
    cost: PipelineCost | None = None,
    step: str = "candidate_judgment",
) -> list[PairJudgment] | None:
    """Judge every candidate's quantity against one source in a single call.

    Sends one batched decisions request carrying the facility, source and
    candidate blocks, and one ``same_quantity`` question per candidate. Returns
    one :class:`PairJudgment` per candidate in candidate order, or ``None`` when
    the decisions call fails, so the source stays unjudged for a retry.

    Args:
        source: The signal source block (id, description, and related fields).
        facility: The facility block the judgment is grounded on.
        candidates: The retrieved candidates to judge, in shortlist order.
        model: Decisions model id; defaults to the mapping-candidates seat.
        service: Service tag for the API key and headers.
        cost: Optional run cost tracker; the decisions call's reported cost is
            added to it under ``step``.
        step: Pipeline-cost step name the call's reported cost is booked to.
    """
    if not candidates:
        return []
    template = _judgment_templates()["same_quantity"]
    questions: dict[str, Any] = {}
    for index in range(len(candidates)):
        questions[f"same_quantity_{index}"] = {
            **template,
            "instructions": template["instructions"].format(
                ref=f"candidates[{index}].path"
            ),
        }
    resolved_model = model or get_model("mapping-candidates")
    state = _judgment_state(source, facility, candidates)
    answers = _call_judgment(
        resolved_model,
        state,
        questions,
        service=service,
        step=step,
        cost=cost,
    )
    if answers is None:
        return None
    judged_at = datetime.now(UTC).isoformat()
    return [
        PairJudgment(
            path=candidates[index].hit.path,
            p_same_quantity=float(answers[f"same_quantity_{index}"]["noul"]),
            model=resolved_model,
            judged_at=judged_at,
        )
        for index in range(len(candidates))
    ]


def route(
    judgments: Sequence[PairJudgment] | None,
    thresholds: RouteThresholds | None = None,
) -> Route | None:
    """Route one source from its judged candidates and the routing thresholds.

    Orders the judgments by descending ``p_same_quantity`` and takes the top
    ``shortlist_size`` as the shortlist. A source is selected when a select
    threshold is set and any candidate reaches it — every candidate at or above
    the threshold is returned in ``selected``, not only the best, because a
    near-tie is expected when a quantity has several homes. It is
    ``no_candidate`` only when a floor is set and no score reaches it;
    everything else is escalated. ``None`` judgments — a failed decisions call —
    yield ``None``: no route, and the source is not selected or rejected.

    Args:
        judgments: The source's candidate judgments, or ``None`` on failure.
        thresholds: Routing thresholds; defaults to the configured ones.
    """
    if judgments is None:
        return None
    limits = thresholds or get_mapping_route_thresholds()
    ordered = sorted(
        judgments, key=lambda judgment: judgment.p_same_quantity, reverse=True
    )
    shortlist = tuple(ordered[: limits.shortlist_size])
    best = ordered[0].p_same_quantity if ordered else None
    if (
        limits.select_threshold is not None
        and best is not None
        and best >= limits.select_threshold
    ):
        decision = "selected"
        selected = frozenset(
            judgment.path
            for judgment in ordered
            if judgment.p_same_quantity >= limits.select_threshold
        )
    elif limits.floor_threshold is not None and (
        best is None or best < limits.floor_threshold
    ):
        decision = "no_candidate"
        selected = frozenset()
    else:
        decision = "escalated"
        selected = frozenset()
    return Route(decision=decision, shortlist=shortlist, selected=selected)


def _sibling_hit(path: str, ids_name: str, documentation: str) -> SearchHit:
    """Build the ``SearchHit`` a cross-IDS cluster sibling joins as."""
    return SearchHit(
        path=path,
        ids_name=ids_name,
        documentation=documentation,
        score=0.0,
        rank=1,
        search_mode=SearchMode.AUTO,
    )


def expand_cluster_siblings(
    source: Mapping[str, Any],
    facility: Mapping[str, Any],
    candidates: Sequence[Candidate],
    judgments: Sequence[PairJudgment],
    *,
    gc: GraphClient,
    model: str | None = None,
    service: str = JUDGMENT_SERVICE,
    cost: PipelineCost | None = None,
    dd_version: int | None = None,
) -> tuple[list[Candidate], list[PairJudgment]]:
    """Expand a source's top candidates through their cross-IDS DD clusters.

    The seed set is the ``CLUSTER_SEED_COUNT`` candidates with the highest
    first-call ``p_same_quantity``. Each seed's cluster siblings in other IDSs
    (``related_dd_search`` with ``relationship_types="cluster"``) become new
    candidates carrying the ``CLUSTER_ARM`` arm. A sibling already in the
    shortlist is dropped, so the union is deduplicated on path, and the new
    siblings are capped at ``CLUSTER_SIBLING_CAP``. Each seed's
    ``SearchHit.see_also`` is filled with the sibling paths its own cluster
    contributed. The new siblings are judged in a second decisions call booked
    to the ``candidate_cluster_judgment`` cost step, because the sibling set
    depends on the first call's ranking.

    Returns the new sibling candidates and their judgments — both empty when no
    seed has a sibling, and empty when the second call's transport fails, so the
    source still routes on its first-call candidates.
    """
    ranked = sorted(
        judgments, key=lambda judgment: judgment.p_same_quantity, reverse=True
    )
    seed_paths = [judgment.path for judgment in ranked[:CLUSTER_SEED_COUNT]]
    if not seed_paths:
        return [], []

    by_path = {candidate.hit.path: candidate for candidate in candidates}
    shortlist_paths = set(by_path)
    sibling_ids: dict[str, str] = {}
    sibling_docs: dict[str, str] = {}
    new_order: list[str] = []
    for seed in seed_paths:
        result = related_dd_search(
            gc, seed, relationship_types="cluster", dd_version=dd_version
        )
        seed_paths_here: list[str] = []
        for hit in result.hits:
            if hit.relationship_type != CLUSTER_ARM:
                continue
            seed_paths_here.append(hit.path)
            if hit.path in shortlist_paths:
                continue
            if hit.path not in sibling_ids:
                sibling_ids[hit.path] = hit.ids
                sibling_docs[hit.path] = hit.doc
                new_order.append(hit.path)
        if seed in by_path:
            by_path[seed].hit.see_also = list(dict.fromkeys(seed_paths_here))

    new_paths = new_order[:CLUSTER_SIBLING_CAP]
    if not new_paths:
        return [], []

    parent_docs = _fetch_parent_documentation(gc, new_paths)
    siblings = [
        Candidate(
            hit=_sibling_hit(path, sibling_ids[path], sibling_docs.get(path) or ""),
            arms=frozenset({CLUSTER_ARM}),
            parent_documentation=parent_docs.get(path),
        )
        for path in new_paths
    ]
    sibling_judgments = judge_candidates(
        source,
        facility,
        siblings,
        model=model,
        service=service,
        cost=cost,
        step="candidate_cluster_judgment",
    )
    if sibling_judgments is None:
        return [], []
    return siblings, sibling_judgments
