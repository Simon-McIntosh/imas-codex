"""Deterministic topological ordering for per-domain catalog entries.

Implements Kahn's topological sort over the ordering-parent relation
derived from ``HAS_PARENT`` and ``HAS_ERROR`` graph edges, with
alphabetic tie-break and clean-root / orphan queue separation.

Pure function of ``(entry-ids, in-domain-edges, cross-domain-edge
presence)`` — does not touch the graph. Cyclic identities are returned as
explicit exclusions while every acyclic identity remains orderable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderingExclusion:
    """One identity withheld because its hierarchy contains a cycle."""

    name: str
    relationships: tuple[tuple[str, str, str], ...]

    @property
    def detail(self) -> str:
        """Name the exact graph relationships that form the cycle."""
        rendered = "; ".join(
            f"{source} -[{edge_type}]-> {target}"
            for source, target, edge_type in self.relationships
        )
        return f"hierarchy cycle relationships: {rendered}"


@dataclass(frozen=True)
class OrderingResult:
    """Acyclic catalog entries plus identity-bearing cycle exclusions."""

    entries: tuple[dict, ...]
    exclusions: tuple[OrderingExclusion, ...]


def order_entries_by_hierarchy(
    entries: list[dict],
    edges: list[tuple[str, str, str]],
    *,
    cross_domain_parent_ids: set[str] | None = None,
) -> OrderingResult:
    """Order entries by deterministic topological traversal.

    Parameters
    ----------
    entries:
        List of entry dicts, each with at least an ``"name"`` key
        (or ``"id"``).
    edges:
        List of ``(src_id, tgt_id, edge_type)`` tuples where
        *edge_type* ∈ ``{"HAS_PARENT", "HAS_ERROR"}``.
        All edges are **in-domain** (both endpoints present in
        *entries*).
    cross_domain_parent_ids:
        Set of entry IDs (names) in *entries* whose full-graph
        ordering-parent lives **outside** this domain.  These are
        placed in the orphan queue instead of the clean-roots queue
        when their in-domain in-degree is zero.

    Returns
    -------
    Acyclic entries re-ordered so that every entry appears after all its
    in-domain ordering-parents, with alphabetic tie-break, plus one exclusion
    for each identity participating in a hierarchy cycle. Removing the cycle
    participants before the final traversal preserves every acyclic entry,
    including descendants of a withheld cycle.
    """
    if cross_domain_parent_ids is None:
        cross_domain_parent_ids = set()

    # Build entry lookup by name
    entry_by_name: dict[str, dict] = {}
    for e in entries:
        name = e.get("name") or e.get("id", "")
        entry_by_name[name] = e

    all_names = set(entry_by_name.keys())

    # ── Build ordering-parent → child adjacency ────────────────────
    # Unified ordering-parent relation:
    #   HAS_PARENT: src -[:HAS_PARENT]-> tgt  ⇒  tgt is parent of src
    #   HAS_ERROR:    src -[:HAS_ERROR]-> tgt     ⇒  src is parent of tgt
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = dict.fromkeys(all_names, 0)

    for src, tgt, edge_type in edges:
        if src not in all_names or tgt not in all_names:
            continue  # skip edges with endpoints outside this domain
        if edge_type == "HAS_PARENT":
            # tgt is ordering-parent of src
            children[tgt].append(src)
            in_degree[src] += 1
        elif edge_type == "HAS_ERROR":
            # src is ordering-parent of tgt
            children[src].append(tgt)
            in_degree[tgt] += 1

    # ── Seed queues ────────────────────────────────────────────────
    clean_roots: list[str] = []
    orphan_queue: list[str] = []

    for name in sorted(all_names):
        if in_degree[name] == 0:
            if name in cross_domain_parent_ids:
                orphan_queue.append(name)
            else:
                clean_roots.append(name)

    # Queues are maintained sorted (alphabetic tie-break)
    clean_roots.sort()
    orphan_queue.sort()

    # ── Kahn's drain ───────────────────────────────────────────────
    result: list[dict] = []
    emitted: set[str] = set()

    while clean_roots or orphan_queue:
        # Pop from clean-roots first, else orphan
        if clean_roots:
            current = clean_roots.pop(0)
        else:
            current = orphan_queue.pop(0)

        if current in emitted:
            continue
        emitted.add(current)
        result.append(entry_by_name[current])

        # Decrement children; newly-ready children go to clean-roots
        for child in sorted(children.get(current, [])):
            in_degree[child] -= 1
            if in_degree[child] == 0 and child not in emitted:
                # Children always inherit clean-root queue status
                _insort(clean_roots, child)

    # ── Cycle isolation ────────────────────────────────────────────
    if len(emitted) != len(all_names):
        stuck = all_names - emitted
        cyclic_components = _find_cyclic_components(stuck, children)
        cyclic_names = set().union(*cyclic_components)
        exclusions: list[OrderingExclusion] = []
        for component in cyclic_components:
            relationships = tuple(
                sorted(
                    (src, tgt, edge_type)
                    for src, tgt, edge_type in edges
                    if src in component and tgt in component
                )
            )
            exclusions.extend(
                OrderingExclusion(name=name, relationships=relationships)
                for name in sorted(component)
            )

        remaining_entries = [
            entry for name, entry in entry_by_name.items() if name not in cyclic_names
        ]
        remaining_edges = [
            edge
            for edge in edges
            if edge[0] not in cyclic_names and edge[1] not in cyclic_names
        ]
        ordered_remaining = order_entries_by_hierarchy(
            remaining_entries,
            remaining_edges,
            cross_domain_parent_ids=cross_domain_parent_ids - cyclic_names,
        )
        return OrderingResult(
            entries=ordered_remaining.entries,
            exclusions=tuple(exclusions) + ordered_remaining.exclusions,
        )

    return OrderingResult(entries=tuple(result), exclusions=())


def _find_cyclic_components(
    nodes: set[str],
    children: dict[str, list[str]],
) -> list[set[str]]:
    """Return strongly connected components that contain a directed cycle."""
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for child in sorted(set(children.get(node, [])) & nodes):
            if child not in indexes:
                visit(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif child in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[child])

        if lowlinks[node] != indexes[node]:
            return

        component: set[str] = set()
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.add(member)
            if member == node:
                break
        if len(component) > 1 or node in children.get(node, []):
            components.append(component)

    for node in sorted(nodes):
        if node not in indexes:
            visit(node)

    return sorted(components, key=lambda component: sorted(component))


def _insort(sorted_list: list[str], value: str) -> None:
    """Insert *value* into *sorted_list* maintaining sorted order."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    sorted_list.insert(lo, value)
