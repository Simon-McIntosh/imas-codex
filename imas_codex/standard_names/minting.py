"""Mint the standard-name review set from a set of DD source paths.

The review pipeline speaks two currencies: extraction/mop-up speaks DD paths,
while export/PR/merge/approval speak standard-name ids. :func:`mint_sn_list`
bridges them deterministically over graph state:

1. **Base join** — every live ``StandardName`` reached by a ``PRODUCED_NAME``
   edge from the ``StandardNameSource`` for an input DD path. The edge is the
   authority; the ``source_paths`` list on the name is a projection of it and
   holds prefixed source ids (``dd:<path>``), so it is not a valid join key for
   an input of bare DD paths.
2. **Immediate-family closure** — approving a name without its immediate
   family is incoherent, so each touched name pulls in its one-hop family:
   its ``HAS_PARENT`` parent(s), that parent's direct children (siblings), and
   its own direct children. One hop in each direction, **non-transitive**.
   Sharing a grammar base token, ``IN_CLUSTER`` membership and ``REFINED_FROM``
   lineage are deliberately NOT closure edges — they are too broad.

Dead names (``superseded``/``exhausted``) are excluded from both the base join
and the closure. A DD path with no linked live name is *reported* in
``unmatched_paths`` rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MintResult:
    """Result of minting an SN set from DD paths.

    Attributes:
        names: Sorted, de-duplicated standard-name ids in the batch (base join
            plus immediate-family closure).
        unmatched_paths: Input DD paths that no live standard name references —
            reported so a caller can surface coverage gaps, never dropped.
    """

    names: list[str] = field(default_factory=list)
    unmatched_paths: list[str] = field(default_factory=list)


# name_stage values that are dead and never belong in a review batch.
_DEAD_STAGES = ["superseded", "exhausted"]


def mint_sn_list(dd_paths: list[str], *, gc: object | None = None) -> MintResult:
    """Mint the standard-name batch for *dd_paths* (base join + family closure).

    Args:
        dd_paths: Bare DD source paths (``<ids>/<path>``) to resolve to names.
        gc: Optional graph client (anything exposing ``.query(cypher, **params)``
            returning a list of row mappings). Injected for testing; when None a
            fresh :class:`GraphClient` is opened.

    Returns:
        A :class:`MintResult` with the sorted batch ids and any unmatched paths.
    """
    paths = list(dict.fromkeys(dd_paths))  # de-dup, preserve order
    if not paths:
        return MintResult(names=[], unmatched_paths=[])

    if gc is not None:
        return _mint_with_client(gc, paths)
    from imas_codex.graph.client import GraphClient

    with GraphClient() as client:
        return _mint_with_client(client, paths)


def _mint_with_client(gc: object, paths: list[str]) -> MintResult:
    # 1. Base join: live names produced by the sources for these DD paths.
    #    Joined over the PRODUCED_NAME edge, which is the authority — the
    #    ``source_paths`` list on the name is a projection of it and cannot be
    #    the join key: it stores prefixed source ids (``dd:<path>``), matching
    #    the StandardNameSource id scheme, so an input of bare DD paths never
    #    intersects it.
    base_rows = gc.query(
        """
        UNWIND $paths AS p
        MATCH (src:StandardNameSource {id: 'dd:' + p})-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE NOT coalesce(sn.name_stage, '') IN $dead
        RETURN p AS path, collect(DISTINCT sn.id) AS ids
        """,
        paths=paths,
        dead=_DEAD_STAGES,
    )
    base_ids: set[str] = set()
    matched_paths: set[str] = set()
    for row in base_rows or []:
        ids = row.get("ids") or []
        if not ids:
            continue
        base_ids.update(ids)
        matched_paths.add(row["path"])

    batch: set[str] = set(base_ids)

    # 2. Immediate-family closure: one hop up (parent), sideways (siblings),
    #    and down (own children); non-transitive; dead names excluded.
    if base_ids:
        fam_rows = gc.query(
            """
            MATCH (x:StandardName) WHERE x.id IN $base_ids
            OPTIONAL MATCH (x)-[:HAS_PARENT]->(parent:StandardName)
              WHERE NOT coalesce(parent.name_stage, '') IN $dead
            OPTIONAL MATCH (sib:StandardName)-[:HAS_PARENT]->(parent)
              WHERE NOT coalesce(sib.name_stage, '') IN $dead
            OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(x)
              WHERE NOT coalesce(child.name_stage, '') IN $dead
            WITH collect(DISTINCT parent.id) + collect(DISTINCT sib.id)
                 + collect(DISTINCT child.id) AS fam
            RETURN [f IN fam WHERE f IS NOT NULL] AS fam_ids
            """,
            base_ids=sorted(base_ids),
            dead=_DEAD_STAGES,
        )
        for row in fam_rows or []:
            batch.update(row.get("fam_ids") or [])

    unmatched = [p for p in paths if p not in matched_paths]
    return MintResult(names=sorted(batch), unmatched_paths=unmatched)
