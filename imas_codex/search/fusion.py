"""Reciprocal Rank Fusion, shared by every search tool.

A candidate's contribution from one stream is ``1 / (k + rank + 1)``, summed
over the streams that rank it, so the consensus of the streams scores a
candidate rather than the magnitude of any one stream's score. The cosine and
BM25 scales the callers fuse are not comparable, and fusing ranks removes the
scale mismatch instead of weighting one against the other.
"""

from __future__ import annotations

__all__ = ["RRF_K", "rrf_fuse"]

#: Default RRF damping constant (Cormack et al. 2009).
RRF_K: int = 60


def rrf_fuse(streams: list[list[dict]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked id-lists.

    Each stream contributes ``1 / (k + rank + 1)`` per id. Tie-break on
    descending fused mass, then ascending id. A stream entry is either a row
    dict carrying an ``id`` key or a bare id string.
    """
    mass: dict[str, float] = {}
    for stream in streams:
        for rank, row in enumerate(stream):
            item_id = row.get("id") if isinstance(row, dict) else row
            if not item_id:
                continue
            mass[item_id] = mass.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(mass.items(), key=lambda r: (-r[1], r[0]))
