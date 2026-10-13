"""Reciprocal Rank Fusion, shared by every search tool.

A candidate's contribution from one stream is ``1 / (k + rank + 1)``, summed
over the streams that rank it, so the consensus of the streams scores a
candidate rather than the magnitude of any one stream's score. The cosine and
BM25 scales the callers fuse are not comparable, and fusing ranks removes the
scale mismatch instead of weighting one against the other.
"""

from __future__ import annotations

__all__ = ["RRF_K", "rrf_fuse", "rrf_fuse_in_input_order"]

#: Default RRF damping constant (Cormack et al. 2009).
RRF_K: int = 60


def _rrf_mass(streams: list[list[dict]], k: int) -> dict[str, float]:
    """Fused mass per id: the summed ``1 / (k + rank + 1)`` over the streams.

    The arithmetic both fusions share, so the two cannot drift. An empty id is
    skipped; a stream entry is either a row dict carrying an ``id`` key or a
    bare id string.
    """
    mass: dict[str, float] = {}
    for stream in streams:
        for rank, row in enumerate(stream):
            item_id = row.get("id") if isinstance(row, dict) else row
            if not item_id:
                continue
            mass[item_id] = mass.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return mass


def rrf_fuse(streams: list[list[dict]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked id-lists.

    Each stream contributes ``1 / (k + rank + 1)`` per id. Tie-break on
    descending fused mass, then ascending id. A stream entry is either a row
    dict carrying an ``id`` key or a bare id string.
    """
    return sorted(_rrf_mass(streams, k).items(), key=lambda r: (-r[1], r[0]))


def rrf_fuse_in_input_order(
    streams: list[list[dict]], order: list[str], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion that breaks ties by a caller-supplied order.

    The fused mass is :func:`rrf_fuse`'s, so the two agree on every candidate
    the streams can separate. They differ only on ties: equal-mass candidates
    keep the sequence the caller gives in *order* rather than sorting on id.

    *order* also names the complete candidate set. A caller fusing alternate
    orderings of one retrieval passes that retrieval's own order as *order*, so
    the fusion is total over it: an id named in *order* but ranked by no stream
    is carried at zero mass, after every ranked candidate, in *order* sequence.
    A caller that wants ties broken on id passes its ids sorted.
    """
    mass = _rrf_mass(streams, k)
    position: dict[str, int] = {}
    for index, item_id in enumerate(order):
        position.setdefault(item_id, index)
        mass.setdefault(item_id, 0.0)
    unranked = len(position)
    return sorted(
        mass.items(),
        key=lambda r: (-r[1], position.get(r[0], unranked), r[0]),
    )
