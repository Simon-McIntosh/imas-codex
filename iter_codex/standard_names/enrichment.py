"""DD enrichment layer for standard name generation.

Navigates DD graph query results to build rich context for standard name
generation.  Implements primary cluster selection (resolving many-to-many
cluster memberships to exactly one per path) and groups paths **globally**
by (cluster × unit × species_context) for unit-safe, species-aware batching.

Data flow::

    sources/dd.py (graph query, multi-cluster rows)
        → enrich_paths()              (classify, deduplicate, select primary cluster)
        → group_by_concept_and_unit() (global grouping, batch splitting)
        → list[ExtractionBatch]       (ready for compose worker)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from iter_codex.standard_names.families import VectorFamily, detect_families
from iter_codex.standard_names.sources.base import ExtractionBatch, SourceCandidate
from iter_codex.standard_names.sources.dd_qualifier import qualify_dd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species / entity context extraction.
#
# Paths like edge_transport/model/ggd/electrons/... and
# edge_transport/model/ggd/ion/... share a cluster and unit but describe
# physically distinct quantities.  Extracting species context lets
# group_by_concept_and_unit() split these into separate batches so the
# LLM generates distinct standard names.
# ---------------------------------------------------------------------------

#: DD path segments that indicate a specific plasma species or entity.
#: When these appear in a path, the standard name MUST distinguish them.
_SPECIES_SEGMENTS: dict[str, str] = {
    "electrons": "electron",
    "electron": "electron",
    "ion": "ion",
    "ions": "ion",
    "neutral": "neutral",
    "neutrals": "neutral",
    "fast_ion": "fast_ion",
    "fast_ions": "fast_ion",
    "total_ion_energy": "total_ion",
    "momentum": "momentum",
}


def extract_species_context(path: str) -> str | None:
    """Extract species/entity context from a DD path hierarchy.

    Scans path segments for known species identifiers and returns the
    canonical species label. Returns None if no species context is found.

    Examples:
        >>> extract_species_context("edge_transport/model/ggd/electrons/energy/v_parallel/values")
        'electron'
        >>> extract_species_context("edge_transport/model/ggd/ion/energy/v_parallel/values")
        'ion'
        >>> extract_species_context("equilibrium/time_slice/profiles_1d/psi")
        None
    """
    segments = path.split("/")
    for seg in segments:
        if seg in _SPECIES_SEGMENTS:
            return _SPECIES_SEGMENTS[seg]
    return None


# ---------------------------------------------------------------------------
# Domain reclassification for magnetics IDS paths.
#
# Paths from the ``magnetics`` IDS are assigned ``physics_domain`` by the DD
# enrichment layer, which often routes them into ``equilibrium`` (via the
# constraint subtree) or ``general``.  For standard-name generation these
# paths should live in ``magnetic_field_diagnostics``.
# ---------------------------------------------------------------------------

#: Magnetics IDS subtrees that should be reclassified to
#: ``magnetic_field_diagnostics``.
MAGNETICS_DIAGNOSTICS_SUBTREES: tuple[str, ...] = (
    "magnetics/bpol_probe",
    "magnetics/flux_loop",
    "magnetics/rogowski_coil",
    "magnetics/diamagnetic_flux",
    "magnetics/b_field_",
    "magnetics/ip",
    "magnetics/method",
    "magnetics/time",
    "magnetics/shunt",
)


def reclassify_magnetics_domain(row: dict) -> None:
    """Override ``physics_domain`` for magnetics-IDS paths.

    Only modifies rows whose ``ids_name`` is ``magnetics``.  Any path under
    the ``magnetics`` IDS that currently has a domain other than
    ``magnetic_field_diagnostics`` is reclassified.

    Args:
        row: Enriched path dict (mutated in place).
    """
    ids_name = row.get("ids_name") or ""
    if ids_name != "magnetics":
        return

    current_domain = row.get("physics_domain") or ""
    if current_domain == "magnetic_field_diagnostics":
        return  # already correct

    row["physics_domain"] = "magnetic_field_diagnostics"


# ---------------------------------------------------------------------------
# Scope priority for primary cluster selection (most specific first).
# ---------------------------------------------------------------------------

_SCOPE_PRIORITY: dict[str, int] = {
    "ids": 0,
    "domain": 1,
    "global": 2,
}
_DEFAULT_SCOPE_RANK = 3  # missing or unrecognised scope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_primary_cluster(clusters: list[dict]) -> dict | None:
    """Choose ONE primary cluster per path from its many-to-many memberships.

    Resolution order (most specific first — best for per-item context):

    1. IDS-scope cluster (most specific) — check ``scope`` field
    2. Domain-scope cluster
    3. Global-scope cluster

    Within same scope: highest ``similarity_score`` (if available), else
    first by ``cluster_label`` (deterministic tie-break).

    Args:
        clusters: List of dicts with keys ``cluster_id``, ``cluster_label``,
            ``cluster_description``, ``scope`` (ids/domain/global),
            ``similarity_score`` (optional).

    Returns:
        The selected primary cluster dict, or ``None`` if *clusters* is empty.
    """
    if not clusters:
        return None
    if len(clusters) == 1:
        return clusters[0]

    def _sort_key(c: dict) -> tuple[int, float, str]:
        scope_rank = _SCOPE_PRIORITY.get((c.get("scope") or ""), _DEFAULT_SCOPE_RANK)
        # Negate similarity so higher scores sort first.
        sim = -(c.get("similarity_score") or 0.0)
        label = c.get("cluster_label") or ""
        return (scope_rank, sim, label)

    return min(clusters, key=_sort_key)


# ---------------------------------------------------------------------------
# Reversed scope priority for grouping (widest first).
# ---------------------------------------------------------------------------

_GROUPING_SCOPE_PRIORITY: dict[str, int] = {
    "global": 0,
    "domain": 1,
    "ids": 2,
}


def select_grouping_cluster(clusters: list[dict]) -> dict | None:
    """Choose the best cluster for batch grouping.

    Uses REVERSED priority: global → domain → IDS (widest first) to ensure
    equivalent paths across different IDSs land in the same batch.

    Returns:
        The selected grouping cluster dict, or ``None`` if *clusters* is empty.
    """
    if not clusters:
        return None
    if len(clusters) == 1:
        return clusters[0]

    def _sort_key(c: dict) -> tuple[int, float, str]:
        scope_rank = _GROUPING_SCOPE_PRIORITY.get(
            (c.get("scope") or ""), _DEFAULT_SCOPE_RANK
        )
        sim = -(c.get("similarity_score") or 0.0)
        label = c.get("cluster_label") or ""
        return (scope_rank, sim, label)

    return min(clusters, key=_sort_key)


def enrich_paths(paths: list[dict]) -> list[dict]:
    """Enrich DD paths with classification and primary cluster selection.

    The enriched graph query in ``sources/dd.py`` may return **multiple rows
    per path** (one per cluster membership).  This function:

    1. Deduplicates rows to one entry per unique path.
    2. Collects all cluster memberships for each path.
    3. Qualifies each path via :func:`qualify_dd` (eligible / skip).
    4. Selects primary cluster from multi-cluster memberships.
    5. Attaches enrichment metadata.

    Returns:
        Only eligible paths, each with ``primary_cluster_id``,
        ``primary_cluster_label``, ``primary_cluster_description``, and
        ``all_clusters`` attached.
    """
    if not paths:
        return []

    # --- Step 1+2: deduplicate rows, collect clusters -----------------------
    path_base: dict[str, dict] = {}  # path → first row (node attributes)
    path_clusters: dict[str, list[dict]] = defaultdict(list)

    for row in paths:
        p = row.get("path", "")
        if not p:
            continue

        # First row seen becomes the canonical base row.
        if p not in path_base:
            path_base[p] = dict(row)

        # Collect cluster membership if present.
        cid = row.get("cluster_id")
        if cid:
            # Avoid adding the same cluster twice (defensive).
            existing_ids = {c["cluster_id"] for c in path_clusters[p]}
            if cid not in existing_ids:
                path_clusters[p].append(
                    {
                        "cluster_id": cid,
                        "cluster_label": row.get("cluster_label") or "",
                        "cluster_description": row.get("cluster_description") or "",
                        "scope": row.get("cluster_scope") or "",
                        "similarity_score": row.get("similarity_score"),
                    }
                )

    # --- Step 3+4+5: qualify, select primary cluster, attach enrichment ----
    enriched: list[dict] = []
    skip_count = 0

    for path, base_row in path_base.items():
        candidate = SourceCandidate.from_dd_row(base_row)
        qualification = qualify_dd(candidate)

        if not qualification.eligible:
            skip_count += 1
            continue

        # Reclassify magnetics-IDS paths to magnetic_field_diagnostics
        reclassify_magnetics_domain(base_row)

        # Select primary cluster (IDS-local, for per-item context)
        clusters = path_clusters.get(path, [])
        primary = select_primary_cluster(clusters)

        base_row["primary_cluster_id"] = primary["cluster_id"] if primary else None
        base_row["primary_cluster_label"] = (
            primary["cluster_label"] if primary else None
        )
        base_row["primary_cluster_description"] = (
            primary["cluster_description"] if primary else None
        )
        base_row["all_clusters"] = clusters

        # Select grouping cluster (global/domain preferred, for batch formation)
        grouping = select_grouping_cluster(clusters)
        base_row["grouping_cluster_id"] = grouping["cluster_id"] if grouping else None
        base_row["grouping_cluster_label"] = (
            grouping["cluster_label"] if grouping else None
        )

        enriched.append(base_row)

    logger.info(
        "Enriched %d quantity paths (skipped %d) from %d raw rows",
        len(enriched),
        skip_count,
        len(paths),
    )
    return enriched


def build_batch_context(
    items: list[dict], group_key: str, cocos_version: int | None = None
) -> str:
    """Build rich context summary for a batch.

    Includes cluster label, authoritative unit, path count, cross-IDS
    summary, concept description, and cluster sibling preview.
    """
    parts: list[str] = []

    # Derive cluster label from items (preferred) or group_key (fallback)
    if group_key.startswith("unclustered/"):
        parts.append("Unclustered paths")
        inner = group_key[len("unclustered/") :]
        last_slash = inner.rfind("/")
        if last_slash > 0:
            parent = inner[:last_slash]
            parts.append(f"Parent structure: {parent}")
    else:
        cluster_label = items[0].get("grouping_cluster_label") or items[0].get(
            "primary_cluster_label"
        )
        if cluster_label:
            parts.append(f"Cluster: {cluster_label}")
        else:
            parts.append(f"Group: {group_key}")

    # Authoritative unit.
    unit = items[0].get("unit") or "dimensionless"
    parts.append(f"Authoritative unit: {unit}")

    # Path count.
    parts.append(f"{len(items)} paths sharing this concept")

    # Cross-IDS summary.
    ids_names = sorted({item.get("ids_name", "unknown") for item in items})
    if len(ids_names) > 1:
        parts.append(f"Cross-IDS: {', '.join(ids_names)}")
    elif ids_names:
        parts.append(f"IDS: {ids_names[0]}")

    # Cluster description.
    desc = items[0].get("primary_cluster_description")
    if desc:
        parts.append(f"Concept: {desc}")

    # Sibling preview.
    siblings = items[0].get("cluster_siblings", [])
    if siblings:
        sib_strs = [f"  {s['path']} ({s.get('unit', '?')})" for s in siblings[:5]]
        parts.append("Cross-IDS siblings:\n" + "\n".join(sib_strs))

    # COCOS context
    cocos_labels = {
        item.get("cocos_label") for item in items if item.get("cocos_label")
    }
    if cocos_labels:
        labels_str = ", ".join(sorted(cocos_labels))
        parts.append(f"COCOS transformation types: {labels_str}")
        if cocos_version:
            parts.append(f"COCOS convention: {cocos_version}")

    return "\n".join(parts)


def _detect_families_from_items(items: list[dict]) -> list[VectorFamily]:
    """Run family detection on enriched items.

    Converts enriched item dicts to the ``{path, unit}`` format expected
    by :func:`detect_families` and returns detected families.
    """
    family_inputs = [
        {"path": item["path"], "unit": item.get("unit") or ""}
        for item in items
        if item.get("path")
    ]
    if not family_inputs:
        return []
    return detect_families(family_inputs)


def load_accepted_sibling_standard_names(
    paths: list[str],
    gc: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Map DD source paths to the established accepted StandardName each has.

    Resolution is exact-first, pattern-fallback:

    * Exact: an accepted StandardName bound to that source path itself.
    * Fallback: a sibling leaf that carries no accepted name of its own still
      has an established spelling — the family's spelling is the dominant
      accepted name bound to sources of the same ``<parent>/<leaf>`` DD shape
      across instruments (e.g. every ``*/position/r`` resolves to
      ``radial_coordinate_of_measurement_position``). The family detector
      groups by the parent's last segment, so this is the same key.

    Only ``name_stage = 'accepted'`` is admitted - a name at drafted,
    reviewed, exhausted or superseded is not an established spelling and must
    not be offered as one.

    Returns ``{source_path: {name, description, unit, physics_domain}}``.
    """
    from iter_codex.graph.client import GraphClient

    nonempty = [p for p in paths if p]
    if not nonempty:
        return {}
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = client.query(
            """
            UNWIND $paths AS p
            MATCH (sns:StandardNameSource {id: 'dd:' + p})-[:PRODUCED_NAME]->(
              sn:StandardName)
            WHERE sn.name_stage = 'accepted'
            RETURN p AS path,
                   sn.id AS name,
                   coalesce(sn.description, '') AS description,
                   coalesce(sn.unit, '') AS unit,
                   sn.physics_domain AS physics_domain
            """,
            paths=nonempty,
        )
        resolved = {
            str(r["path"]): {
                "name": str(r["name"]),
                "description": str(r.get("description") or ""),
                "unit": str(r.get("unit") or ""),
                "physics_domain": r.get("physics_domain"),
            }
            for r in rows or []
            if r.get("path")
        }
        missing = [p for p in nonempty if p not in resolved]
        if missing:
            # Fallback: inherit the family's established spelling from the
            # dominant accepted name sharing the same <parent>/<leaf> shape.
            suffixes = [
                f"/{p.split('/')[-2]}/{p.rsplit('/', 1)[-1]}"
                for p in missing
                if "/" in p
            ]
            if suffixes:
                pattern_rows = client.query(
                    """
                    UNWIND $suffixes AS suffix
                    MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(
                      sn:StandardName)
                    WHERE sns.id ENDS WITH suffix
                      AND sn.name_stage = 'accepted'
                    WITH suffix, sn, count(*) AS cnt
                    ORDER BY cnt DESC
                    WITH suffix,
                         collect({
                           name: sn.id,
                           description: coalesce(sn.description, ''),
                           unit: coalesce(sn.unit, ''),
                           physics_domain: sn.physics_domain
                         })[0] AS pick
                    RETURN suffix, pick
                    """,
                    suffixes=list(dict.fromkeys(suffixes)),
                )
                picks = {
                    str(r["suffix"]): r["pick"]
                    for r in pattern_rows or []
                    if r.get("suffix") and r.get("pick")
                }
                for path, suffix in zip(missing, suffixes, strict=False):
                    pick = picks.get(suffix)
                    if pick:
                        resolved[path] = {
                            "name": str(pick.get("name") or ""),
                            "description": str(pick.get("description") or ""),
                            "unit": str(pick.get("unit") or ""),
                            "physics_domain": pick.get("physics_domain"),
                        }
        return resolved
    finally:
        if own:
            client.close()


def attach_family_accepted_siblings(
    items: list[dict[str, Any]],
    *,
    gc: Any | None = None,
) -> None:
    """Tag family-member items with their ACCEPTED siblings' standard names.

    Family detection groups DD paths structurally, so a family conveys member
    *paths*, not the spellings the family has already settled on. For each item
    that is a member of a detected vector/geometric family, this resolves the
    accepted StandardName bound to every sibling source path and writes
    ``item["family_accepted_siblings"]`` — an axis-ordered list of ``{name,
    description, axis, path}`` (same per-entry fields and
    :func:`sort_by_axis_convention` ordering as the docs-side
    ``child_components`` injection, mirrored in the other direction).

    The family is reconstructed from the DD containment structure (the item's
    parent via ``HAS_PARENT`` and the parent's other children), so a family
    member composed alone — a pool claim batch is grouped by cluster/unit and
    may not carry its axis siblings — still sees them here.

    Defensive: any detection or graph failure leaves items untouched and the
    tag simply absent.
    """
    from iter_codex.standard_names.families import (
        detect_families,
        sort_by_axis_convention,
    )

    try:
        paths = [item["path"] for item in items if item.get("path")]
        if not paths:
            return
        from iter_codex.graph.client import GraphClient

        own = gc is None
        client = GraphClient() if own else gc
        try:
            # Containment is modelled upward in this graph (a node points at
            # its parent via HAS_PARENT), so the candidate siblings of a path
            # are the other children of the same parent node. Coordinate
            # family members (position/r, position/z, position/phi) are
            # themselves STRUCTURE containers whose axis suffix is what
            # detect_families keys on, so structure children are NOT excluded
            # here — family detection upstream filters by axis suffix.
            sibling_rows = client.query(
                """
                UNWIND $paths AS p
                MATCH (n:IMASNode {id: p})-[:HAS_PARENT]->(parent:IMASNode)
                OPTIONAL MATCH (parent)<-[:HAS_PARENT]-(sib:IMASNode)
                RETURN p AS path, collect(DISTINCT sib.id) AS sibling_ids
                """,
                paths=paths,
            )
            siblings_by_path: dict[str, list[str]] = {
                str(r["path"]): [str(s) for s in (r.get("sibling_ids") or []) if s]
                for r in sibling_rows or []
            }
        finally:
            if own:
                client.close()

        candidate_bysrc: dict[str, list[dict[str, str]]] = {}
        for item in items:
            path = item.get("path")
            if not path:
                continue
            local = [{"path": path, "unit": str(item.get("unit") or "")}]
            for spath in siblings_by_path.get(path, []):
                if spath != path:
                    local.append({"path": spath, "unit": ""})
            candidate_bysrc[path] = local
        if not candidate_bysrc:
            return
        family_inputs = [entry for local in candidate_bysrc.values() for entry in local]
        member_of: dict[str, VectorFamily] = {}
        for family in detect_families(family_inputs):
            for member in family.members:
                member_of[member.dd_path] = family
        # Families that contain at least one in-scope item (dedup by identity).
        owned_families: list[VectorFamily] = []
        seen_ids: set[int] = set()
        for path in candidate_bysrc:
            family = member_of.get(path)
            if family is not None and id(family) not in seen_ids:
                seen_ids.add(id(family))
                owned_families.append(family)
        all_sibling_paths = sorted(
            {m.dd_path for f in owned_families for m in f.members}
        )
        if not all_sibling_paths:
            return
        accepted = load_accepted_sibling_standard_names(all_sibling_paths, gc=gc)
        if not accepted:
            return
        for item in items:
            path = item.get("path")
            family = member_of.get(path)
            if family is None:
                continue
            entries = [
                {
                    "name": accepted[p.dd_path]["name"],
                    "description": accepted[p.dd_path]["description"],
                    "axis": p.axis,
                    "unit": accepted[p.dd_path].get("unit") or "",
                    "physics_domain": accepted[p.dd_path].get("physics_domain"),
                    "path": p.dd_path,
                }
                for p in family.members
                if p.dd_path != path and p.dd_path in accepted
            ]
            if entries:
                item["family_accepted_siblings"] = sort_by_axis_convention(entries)
    except Exception:
        logger.debug(
            "family accepted-sibling context failed",
            exc_info=True,
        )


def group_by_concept_and_unit(
    items: list[dict],
    max_batch_size: int = 25,
    existing_names: set[str] | None = None,
    max_tokens: int | None = None,
) -> list[ExtractionBatch]:
    """Group enriched paths by (primary_cluster × unit) **globally**.

    Critical design decisions:

    * **Global grouping** — same concept across IDSs → same batch → same
      name.
    * **Primary cluster** — each path appears in exactly ONE batch (no
      Cartesian product from multi-cluster membership).
    * **Mixed-unit clusters** split into separate batches.
    * **Oversized groups** split into chunks of *max_batch_size*.
    * **Unclustered paths** sub-grouped by ``parent_path``.

    Args:
        items: Enriched path dicts (output of :func:`enrich_paths`).
        max_batch_size: Maximum concepts per batch (token budget guard).
        existing_names: Known standard names for dedup awareness.
        max_tokens: When set, apply a pre-flight token check that
            binary-splits any batch exceeding this estimated token count.

    Returns:
        List of :class:`ExtractionBatch` objects ready for the compose
        worker.
    """
    if existing_names is None:
        existing_names = set()

    if not items:
        return []

    # --- Family-aware pre-grouping -----------------------------------------
    # Detect vector, geometric, and derivative families BEFORE the normal
    # cluster×unit grouping.  Family members are forced into a single batch
    # even when they have different units (critical for geometric families
    # where r(m) and phi(rad) must be named together).
    family_items: dict[str, list[dict]] = {}  # family key → items
    family_paths: set[str] = set()  # paths consumed by a family

    families = _detect_families_from_items(items)
    for family in families:
        family_id = f"{family.parent_path}:{family.family_type}"
        member_paths = {m.dd_path for m in family.members}
        family_group: list[dict] = []
        for item in items:
            if item.get("path") in member_paths:
                # Tag item with family metadata
                member = next(m for m in family.members if m.dd_path == item["path"])
                sibling_paths = sorted(
                    m.dd_path for m in family.members if m.dd_path != item["path"]
                )
                item["family_type"] = family.family_type
                item["family_axis"] = member.axis
                item["family_siblings"] = sibling_paths
                item["family_parent_name"] = family.parent_name
                family_group.append(item)
                family_paths.add(item["path"])
        if family_group:
            family_items[f"family:{family_id}"] = family_group

    # --- Build groups: (grouping_cluster_id / unit / species_context) -------
    # Uses grouping cluster (global/domain preferred) and cluster ID (not label)
    # to ensure cross-IDS paths sharing the same concept land in one batch.
    # Species context is included so paths for different plasma species
    # (electrons, ion, neutral, …) that share a cluster and unit are split
    # into distinct batches and receive distinct standard names.
    groups: dict[str, list[dict]] = defaultdict(list)

    # Pre-seed family groups
    groups.update(family_items)

    for item in items:
        # Skip items already consumed by a family
        if item.get("path") in family_paths:
            continue

        cluster_id = item.get("grouping_cluster_id")
        unit = item.get("unit") or "dimensionless"
        species_context = extract_species_context(item.get("path") or "")
        item["species_context"] = species_context

        if cluster_id:
            if species_context:
                group_key = f"{cluster_id}/{unit}/{species_context}"
            else:
                group_key = f"{cluster_id}/{unit}"
        else:
            # Unclustered: sub-group by IDS + parent for coherent batches.
            ids_name = item.get("ids_name") or "unknown"
            parent = item.get("parent_path") or "root"
            if species_context:
                group_key = f"unclustered/{ids_name}/{parent}/{unit}/{species_context}"
            else:
                group_key = f"unclustered/{ids_name}/{parent}/{unit}"

        groups[group_key].append(item)

    # --- Split oversized groups and build batches ---------------------------
    batches: list[ExtractionBatch] = []

    for group_key in sorted(groups):
        group_items = groups[group_key]

        # Chunk into max_batch_size slices.
        chunks = [
            group_items[i : i + max_batch_size]
            for i in range(0, len(group_items), max_batch_size)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            batch_key = group_key if len(chunks) == 1 else f"{group_key}#{chunk_idx}"
            context = build_batch_context(chunk, group_key)

            batches.append(
                ExtractionBatch(
                    source="dd",
                    group_key=batch_key,
                    items=chunk,
                    context=context,
                    existing_names=existing_names,
                )
            )

    logger.info(
        "Grouped %d paths into %d batches (max_batch_size=%d)",
        len(items),
        len(batches),
        max_batch_size,
    )

    # Pre-flight token check: split batches that exceed token budget
    if max_tokens is not None:
        from iter_codex.standard_names.batching import pre_flight_token_check

        batches = pre_flight_token_check(batches, max_tokens=max_tokens)

    return batches


# ---------------------------------------------------------------------------
# Name-only grouping — coarser batches, broader concept bins.
# ---------------------------------------------------------------------------


def build_name_only_context(items: list[dict], group_key: str) -> str:
    """Build compact context for a ``name_only`` batch.

    Name-only batches use ``(physics_domain × unit)`` as the grouping
    key and may contain items from many clusters and IDSs.  The prompt
    asks the LLM to identify natural sub-groups within the batch, so
    the context summarises the domain-level shape rather than a single
    cluster.
    """
    parts: list[str] = []

    domain = items[0].get("physics_domain") or "unspecified"
    unit = items[0].get("unit") or "dimensionless"

    parts.append(f"Physics domain: {domain}")
    parts.append(f"Authoritative unit: {unit}")
    parts.append(f"{len(items)} paths sharing (domain, unit)")

    ids_names = sorted({item.get("ids_name") or "unknown" for item in items})
    if len(ids_names) > 1:
        parts.append(f"Cross-IDS: {', '.join(ids_names)}")
    elif ids_names:
        parts.append(f"IDS: {ids_names[0]}")

    # Surface the cluster diversity so the LLM knows to look for
    # natural sub-groups when composing names.
    cluster_labels = {
        (item.get("grouping_cluster_label") or item.get("primary_cluster_label"))
        for item in items
        if item.get("grouping_cluster_label") or item.get("primary_cluster_label")
    }
    cluster_labels.discard(None)
    if cluster_labels:
        labels_preview = sorted(cluster_labels)[:8]
        suffix = (
            f" (+{len(cluster_labels) - len(labels_preview)} more)"
            if len(cluster_labels) > len(labels_preview)
            else ""
        )
        parts.append("Represented clusters: " + "; ".join(labels_preview) + suffix)

    return "\n".join(parts)


def group_for_name_only(
    items: list[dict],
    batch_size: int = 50,
    existing_names: set[str] | None = None,
    max_tokens: int | None = None,
) -> list[ExtractionBatch]:
    """Group enriched paths by ``(physics_domain × unit)`` for name-only mode.

    This is the name-only batching strategy: coarser grouping than
    :func:`group_by_concept_and_unit` to amortise the system-prompt
    cost across many more items per LLM call.  Empirically this
    collapses the ~35 %% singleton tail of (cluster × unit) batching
    into dense bins (mean ≈ 13 items at ``batch_size=50``) while
    preserving unit safety.

    Args:
        items: Enriched path dicts (output of :func:`enrich_paths`).
        batch_size: Maximum items per batch.  Items beyond this cap
            are split into chunks that keep the same group key but
            append a ``#<idx>`` suffix for traceability.
        existing_names: Known standard names for dedup awareness.
        max_tokens: When set, apply a pre-flight token check that
            binary-splits any batch exceeding this estimated token count.

    Returns:
        List of :class:`ExtractionBatch` objects with ``mode="names"``.
    """
    if existing_names is None:
        existing_names = set()

    if not items:
        return []

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        domain = item.get("physics_domain") or "unspecified"
        unit = item.get("unit") or "dimensionless"
        # `ids_name` is NOT part of the key — we intentionally merge
        # across IDSs to encourage cross-IDS name reuse.
        group_key = f"name_only/{domain}/{unit}"
        groups[group_key].append(item)

    batches: list[ExtractionBatch] = []
    for group_key in sorted(groups):
        group_items = groups[group_key]
        chunks = [
            group_items[i : i + batch_size]
            for i in range(0, len(group_items), batch_size)
        ]
        for chunk_idx, chunk in enumerate(chunks):
            batch_key = group_key if len(chunks) == 1 else f"{group_key}#{chunk_idx}"
            context = build_name_only_context(chunk, group_key)
            batches.append(
                ExtractionBatch(
                    source="dd",
                    group_key=batch_key,
                    items=chunk,
                    context=context,
                    existing_names=existing_names,
                    mode="names",
                )
            )

    logger.info(
        "Name-only grouping: %d paths → %d batches (batch_size=%d, mean=%.1f)",
        len(items),
        len(batches),
        batch_size,
        len(items) / len(batches) if batches else 0.0,
    )

    # Pre-flight token check: split batches that exceed token budget
    if max_tokens is not None:
        from iter_codex.standard_names.batching import pre_flight_token_check

        batches = pre_flight_token_check(batches, max_tokens=max_tokens)

    return batches
