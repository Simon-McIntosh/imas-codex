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
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from imas_codex.graph.dd_search import hybrid_dd_search
from imas_codex.search.search_strategy import SearchHit

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

#: Default shortlist size per source.
DEFAULT_K: Final[int] = 20

#: Identifier of the arm that searches the whole DD, used when a source carries
#: no routed IDSs. Real IDS names never take this value.
UNSCOPED_ARM: Final[str] = "*"


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
