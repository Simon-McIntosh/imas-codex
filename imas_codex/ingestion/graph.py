"""Graph relationship creation for ingested content.

Post-ingestion processing to create relationships between Chunk nodes,
DataReference nodes, and data entities (SignalNode, TDIFunction, IMASNode).

Architecture:
    Chunk -[:CONTAINS_REF]-> DataReference -[:RESOLVES_TO_*]-> Entity

DataReference nodes preserve the exact string found in content for provenance,
while typed resolution relationships link to actual data entities.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Pattern to strip channel/index suffixes for fuzzy matching
CHANNEL_SUFFIX_PATTERN = re.compile(r"[_:](?:CHANNEL_?)?\d+$", re.IGNORECASE)


def _normalize_for_matching(raw_path: str) -> str:
    """Normalize a path for fuzzy SignalNode matching.

    Strips channel indices and numeric suffixes to match tree structure.
    E.g., \\ATLAS::DT196_MHD_001:CHANNEL_006 -> \\ATLAS::DT196_MHD:CHANNEL
    """
    normalized = raw_path.upper().rstrip(":.")
    while CHANNEL_SUFFIX_PATTERN.search(normalized):
        normalized = CHANNEL_SUFFIX_PATTERN.sub("", normalized)
    return normalized


def _generate_ref_id(facility: str, ref_type: str, raw_string: str) -> str:
    """Generate DataReference ID: facility:type:hash."""
    content = raw_string.encode("utf-8")
    hash_suffix = hashlib.md5(content).hexdigest()[:12]
    return f"{facility}:{ref_type}:{hash_suffix}"


def link_chunks_to_edas_signals(
    graph_client: GraphClient | None = None,
    example_ids: list[str] | None = None,
    *,
    facility_id: str,
) -> dict[str, int]:
    """Backfill stored code reads and resolve literal EDAS signal keys.

    Reads only stored CodeChunk text. It never contacts a facility or reruns
    discovery, so ingestion and backfill share the same extraction path.
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.ingestion.extractors.edas import extract_edas_references

    if "edas" not in (get_facility(facility_id).get("data_systems") or {}):
        raise ValueError(f"EDAS is not configured for {facility_id}")

    selection = (
        "AND c.code_example_id IN $example_ids" if example_ids is not None else ""
    )
    with _get_client(graph_client) as client:
        chunks = client.query(
            "MATCH (c:CodeChunk {facility_id: $facility}) "
            f"WHERE c.text IS NOT NULL {selection} "
            "RETURN c.id AS id, c.text AS text",
            facility=facility_id,
            example_ids=example_ids,
        )
        rows = []
        for chunk in chunks:
            for ref in extract_edas_references(chunk["text"]):
                rows.append(
                    {
                        "chunk_id": chunk["id"],
                        "id": _generate_ref_id(
                            facility_id, ref.ref_type, ref.raw_string
                        ),
                        "raw_string": ref.raw_string,
                        "ref_type": ref.ref_type,
                        "category": ref.category,
                        "data_name": ref.data_name,
                        "signal_path": (
                            ref.raw_string
                            if ref.category
                            else f"{ref.ref_type.removeprefix('edas_').upper()}/{ref.data_name}"
                        ),
                    }
                )
        if not rows:
            return {"chunks": len(chunks), "references": 0, "resolved": 0}
        resolved = 0
        for offset in range(0, len(rows), 200):
            result = client.query(
                """
                UNWIND $rows AS item
                MATCH (c:CodeChunk {id: item.chunk_id})
                MERGE (d:DataReference {id: item.id})
                SET d.facility_id = $facility,
                    d.ref_type = item.ref_type,
                    d.raw_string = item.raw_string,
                    d.edas_category = item.category,
                    d.edas_data_name = item.data_name
                MERGE (c)-[:CONTAINS_REF]->(d)
                WITH d, item
                MATCH (f:Facility {id: $facility})
                MERGE (d)-[:AT_FACILITY]->(f)
                WITH d, item
                OPTIONAL MATCH (s:FacilitySignal {
                    facility_id: $facility, data_source_path: item.signal_path
                })
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                    MERGE (d)-[:RESOLVES_TO_FACILITY_SIGNAL]->(s))
                RETURN count(s) AS resolved
                """,
                rows=rows[offset : offset + 200],
                facility=facility_id,
            )
            resolved += result[0]["resolved"] if result else 0
        return {
            "chunks": len(chunks),
            "references": len(rows),
            "resolved": resolved,
        }


@contextlib.contextmanager
def _get_client(graph_client: GraphClient | None = None) -> Iterator[GraphClient]:
    """Context manager that uses provided client or creates a new one."""
    if graph_client is not None:
        yield graph_client
    else:
        from imas_codex.graph import GraphClient

        client = GraphClient()
        with client:
            yield client


def link_chunks_to_imas_paths(
    graph_client: GraphClient | None = None,
    example_ids: list[str] | None = None,
) -> int:
    """Link extracted DD paths to existing IMASNode paths only.

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.
        example_ids: If provided, only process chunks belonging to these
            CodeExample IDs. Otherwise processes all chunks (legacy mode).

    Returns:
        Number of path references resolved
    """
    with _get_client(graph_client) as client:
        count = _link_chunk_references(
            client, "imas_paths", "IMASNode", "REFERENCES_IMAS", example_ids
        )[1]
        logger.info("Resolved %d IMAS DD path references", count)
        return count


def link_chunks_to_ids_roots(
    graph_client: GraphClient | None = None,
    example_ids: list[str] | None = None,
) -> int:
    """Link bare IDS names to IDS roots and refuse missing named links."""
    with _get_client(graph_client) as client:
        mentions, linked = _link_chunk_references(
            client, "related_ids", "IDS", "REFERENCES_IDS", example_ids
        )
        if mentions:
            selection = (
                "WHERE c.code_example_id IN $example_ids"
                if example_ids is not None
                else ""
            )
            params = {"example_ids": example_ids} if example_ids is not None else {}
            missing_rows = client.query(
                f"MATCH (c:CodeChunk) {selection} "
                "UNWIND c.related_ids AS name "
                "OPTIONAL MATCH (c)-[:REFERENCES_IDS]->(root:IDS {id: name}) "
                "WITH root WHERE root IS NULL RETURN count(*) AS missing",
                **params,
            )
            missing = missing_rows[0]["missing"] if missing_rows else mentions - linked
            if missing:
                raise ValueError(
                    f"No IDS roots linked for {missing} of {mentions} named references"
                )
        logger.info("Resolved %d IDS root references", linked)
        return linked


def _link_chunk_references(
    client: GraphClient,
    property_name: str,
    target_label: str,
    relationship: str,
    example_ids: list[str] | None,
) -> tuple[int, int]:
    """Count references, then merge links only to existing target nodes."""
    selection = (
        "WHERE c.code_example_id IN $example_ids" if example_ids is not None else ""
    )
    params = {"example_ids": example_ids} if example_ids is not None else {}
    counts = client.query(
        f"MATCH (c:CodeChunk) {selection} "
        f"RETURN sum(size(coalesce(c.{property_name}, []))) AS mentions",
        **params,
    )
    mentions = (counts[0]["mentions"] or 0) if counts else 0
    if not mentions:
        return 0, 0
    links = client.query(
        f"MATCH (c:CodeChunk) {selection} "
        f"UNWIND c.{property_name} AS reference "
        f"MATCH (target:{target_label} {{id: reference}}) "
        f"MERGE (c)-[:{relationship}]->(target) "
        "RETURN count(*) AS linked",
        **params,
    )
    return mentions, links[0]["linked"] if links else 0


def link_examples_to_facility(
    graph_client: GraphClient | None = None,
    example_ids: list[str] | None = None,
) -> int:
    """Create AT_FACILITY relationships from CodeExample nodes.

    Args:
        graph_client: Optional GraphClient instance.
        example_ids: If provided, only process these CodeExample IDs.

    Returns:
        Number of relationships created
    """
    if example_ids is not None:
        cypher = """
            MATCH (e:CodeExample)
            WHERE e.id IN $example_ids AND e.facility_id IS NOT NULL
            MATCH (f:Facility {id: e.facility_id})
            MERGE (e)-[:AT_FACILITY]->(f)
            RETURN count(*) AS created
        """
        params = {"example_ids": example_ids}
    else:
        cypher = """
            MATCH (e:CodeExample)
            WHERE e.facility_id IS NOT NULL
            MATCH (f:Facility {id: e.facility_id})
            MERGE (e)-[:AT_FACILITY]->(f)
            RETURN count(*) AS created
        """
        params = {}

    with _get_client(graph_client) as client:
        result = client.query(cypher, **params)
        count = result[0]["created"] if result else 0
        logger.info("Created %d AT_FACILITY relationships", count)
        return count


def link_chunks_to_data_nodes(
    graph_client: GraphClient | None = None,
    example_ids: list[str] | None = None,
) -> int:
    """Create DataReference nodes and link to DataNodes for MDSplus paths.

    For chunks with mdsplus_paths metadata:
    1. Creates DataReference nodes (deduplicated by facility:type:hash)
    2. Creates CONTAINS_REF relationships from CodeChunk -> DataReference
    3. Computes normalized_path for fuzzy matching
    4. Creates RESOLVES_TO_NODE relationships from DataReference -> SignalNode

    Args:
        graph_client: Optional GraphClient instance.
        example_ids: If provided, only process chunks belonging to these
            CodeExample IDs. Otherwise processes all chunks (legacy mode).

    Returns:
        Number of DataReference nodes created or linked
    """
    scoped = example_ids is not None

    with _get_client(graph_client) as client:
        import time as _time

        link_times: dict[str, float] = {}

        # Materialize DataReference nodes from chunk path strings.
        t_s = _time.monotonic()
        if scoped:
            create_refs_simple = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                  AND c.mdsplus_paths IS NOT NULL
                MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
                WHERE e.facility_id IS NOT NULL
                UNWIND c.mdsplus_paths AS path
                WITH DISTINCT e.facility_id AS facility, path
                MERGE (d:DataReference {raw_string: path, facility_id: facility})
                ON CREATE SET
                    d.id = facility + ':mdsplus_path:' + path,
                    d.ref_type = 'mdsplus_path'
                WITH d, facility
                MATCH (f:Facility {id: facility})
                MERGE (d)-[:AT_FACILITY]->(f)
                RETURN count(d) AS refs_created
            """
            params: dict = {"example_ids": example_ids}
        else:
            create_refs_simple = """
                MATCH (c:CodeChunk)
                WHERE c.mdsplus_paths IS NOT NULL
                MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
                WHERE e.facility_id IS NOT NULL
                UNWIND c.mdsplus_paths AS path
                WITH DISTINCT e.facility_id AS facility, path
                MERGE (d:DataReference {raw_string: path, facility_id: facility})
                ON CREATE SET
                    d.id = facility + ':mdsplus_path:' + path,
                    d.ref_type = 'mdsplus_path'
                WITH d, facility
                MATCH (f:Facility {id: facility})
                MERGE (d)-[:AT_FACILITY]->(f)
                RETURN count(d) AS refs_created
            """
            params = {}
        result = client.query(create_refs_simple, **params)
        refs_created = result[0]["refs_created"] if result else 0
        link_times["create_refs"] = _time.monotonic() - t_s
        logger.info("Created/matched %d DataReference nodes", refs_created)

        # Connect each chunk to its extracted path references.
        t_s = _time.monotonic()
        if scoped:
            contains_ref = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                  AND c.mdsplus_paths IS NOT NULL
                MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
                WHERE e.facility_id IS NOT NULL
                UNWIND c.mdsplus_paths AS path
                WITH c, e.facility_id AS facility, path
                MATCH (d:DataReference {raw_string: path, facility_id: facility})
                MERGE (c)-[:CONTAINS_REF]->(d)
                RETURN count(*) AS contains_created
            """
        else:
            contains_ref = """
                MATCH (c:CodeChunk)
                WHERE c.mdsplus_paths IS NOT NULL
                MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
                WHERE e.facility_id IS NOT NULL
                UNWIND c.mdsplus_paths AS path
                With c, e.facility_id AS facility, path
                MATCH (d:DataReference {raw_string: path, facility_id: facility})
                MERGE (c)-[:CONTAINS_REF]->(d)
                RETURN count(*) AS contains_created
            """
        result = client.query(contains_ref, **params)
        contains_count = result[0]["contains_created"] if result else 0
        link_times["contains_ref"] = _time.monotonic() - t_s
        logger.info("Created %d CONTAINS_REF relationships", contains_count)

        # Normalize only references in this batch.
        t_s = _time.monotonic()
        if scoped:
            # Only normalize refs we just created/matched
            refs_to_normalize = client.query(
                """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                  AND c.mdsplus_paths IS NOT NULL
                MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
                UNWIND c.mdsplus_paths AS path
                WITH DISTINCT e.facility_id AS facility, path
                MATCH (d:DataReference {raw_string: path, facility_id: facility})
                WHERE d.normalized_path IS NULL
                RETURN d.id AS id, d.raw_string AS raw
                """,
                **params,
            )
        else:
            refs_to_normalize = client.query("""
                MATCH (d:DataReference {ref_type: 'mdsplus_path'})
                WHERE d.normalized_path IS NULL
                RETURN d.id AS id, d.raw_string AS raw
            """)
        if refs_to_normalize:
            updates = [
                {"id": r["id"], "normalized": _normalize_for_matching(r["raw"])}
                for r in refs_to_normalize
            ]
            client.query(
                """
                UNWIND $updates AS u
                MATCH (d:DataReference {id: u.id})
                SET d.normalized_path = u.normalized
                """,
                updates=updates,
            )
            logger.info("Computed normalized_path for %d refs", len(updates))
        link_times["normalize"] = _time.monotonic() - t_s

        # Resolve references to facility signal nodes.
        # Two-phase: exact match first (uses path+facility_id index),
        # then fuzzy match for remaining (facility-scoped to avoid
        # catastrophic O(all_refs × all_signals) cross-product).
        t_s = _time.monotonic()
        if scoped:
            # Exact path matching uses the facility and path index.
            resolve_exact = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                  AND c.mdsplus_paths IS NOT NULL
                MATCH (c)-[:CONTAINS_REF]->(d:DataReference {ref_type: 'mdsplus_path'})
                WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
                WITH DISTINCT d
                MATCH (t:SignalNode {facility_id: d.facility_id, path: d.raw_string})
                MERGE (d)-[:RESOLVES_TO_NODE]->(t)
                RETURN count(*) AS resolved
            """
            # Fuzzy matching stays within unresolved facility references.
            # Uses CALL {} subquery with LIMIT 1 per DataReference to avoid
            # O(refs × signals) cross-product that caused 14-26s stalls.
            resolve_fuzzy = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                  AND c.mdsplus_paths IS NOT NULL
                MATCH (c)-[:CONTAINS_REF]->(d:DataReference {ref_type: 'mdsplus_path'})
                WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
                WITH DISTINCT d
                CALL {
                    WITH d
                    MATCH (t:SignalNode {facility_id: d.facility_id})
                    WHERE t.path ENDS WITH substring(d.raw_string, 1)
                       OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
                       OR toUpper(t.path) = d.normalized_path
                    RETURN t LIMIT 1
                }
                MERGE (d)-[:RESOLVES_TO_NODE]->(t)
                RETURN count(*) AS resolved
            """
        else:
            # Exact path matching uses the facility and path index.
            resolve_exact = """
                MATCH (d:DataReference {ref_type: 'mdsplus_path'})
                WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
                WITH d
                MATCH (t:SignalNode {facility_id: d.facility_id, path: d.raw_string})
                MERGE (d)-[:RESOLVES_TO_NODE]->(t)
                RETURN count(*) AS resolved
            """
            # Fuzzy matching stays within unresolved facility references.
            # Uses CALL {} subquery with LIMIT 1 per DataReference.
            resolve_fuzzy = """
                MATCH (d:DataReference {ref_type: 'mdsplus_path'})
                WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
                WITH d
                CALL {
                    WITH d
                    MATCH (t:SignalNode {facility_id: d.facility_id})
                    WHERE t.path ENDS WITH substring(d.raw_string, 1)
                       OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
                       OR toUpper(t.path) = d.normalized_path
                    RETURN t LIMIT 1
                }
                MERGE (d)-[:RESOLVES_TO_NODE]->(t)
                RETURN count(*) AS resolved
            """
        result = client.query(resolve_exact, **params)
        exact_count = result[0]["resolved"] if result else 0
        link_times["resolve_exact"] = _time.monotonic() - t_s
        t_s = _time.monotonic()
        result = client.query(resolve_fuzzy, **params)
        fuzzy_count = result[0]["resolved"] if result else 0
        link_times["resolve_fuzzy"] = _time.monotonic() - t_s
        resolved_count = exact_count + fuzzy_count
        logger.info(
            "Created %d RESOLVES_TO_NODE relationships (exact=%d, fuzzy=%d)",
            resolved_count,
            exact_count,
            fuzzy_count,
        )

        # Carry resolved signal mappings through to IMAS paths.
        t_s = _time.monotonic()
        if scoped:
            imas_q = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                MATCH (c)-[:CONTAINS_REF]->(dr:DataReference)-[:RESOLVES_TO_NODE]->(dn:SignalNode)
                WHERE NOT (dr)-[:RESOLVES_TO_IMAS_PATH]->()
                MATCH (dn)-[:MEMBER_OF]->(sg:SignalSource)-[:MAPS_TO_IMAS]->(ip:IMASNode)
                MERGE (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip)
                RETURN count(*) AS linked
            """
        else:
            imas_q = """
                MATCH (dr:DataReference)-[:RESOLVES_TO_NODE]->(dn:SignalNode)
                WHERE NOT (dr)-[:RESOLVES_TO_IMAS_PATH]->()
                MATCH (dn)-[:MEMBER_OF]->(sg:SignalSource)-[:MAPS_TO_IMAS]->(ip:IMASNode)
                MERGE (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip)
                RETURN count(*) AS linked
            """
        result = client.query(imas_q, **params)
        imas_linked = result[0]["linked"] if result else 0
        link_times["imas_path"] = _time.monotonic() - t_s
        if imas_linked:
            logger.info("Created %d RESOLVES_TO_IMAS_PATH relationships", imas_linked)

        # Connect TDI call references to function nodes.
        t_s = _time.monotonic()
        if scoped:
            tdi_q = """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                MATCH (c)-[:CONTAINS_REF]->(dr:DataReference {ref_type: 'tdi_call'})
                WHERE NOT (dr)-[:CALLS_TDI_FUNCTION]->()
                WITH DISTINCT dr
                MATCH (tdi:TDIFunction {facility_id: dr.facility_id})
                WHERE tdi.name = dr.raw_string
                   OR dr.raw_string CONTAINS tdi.name
                MERGE (dr)-[:CALLS_TDI_FUNCTION]->(tdi)
                RETURN count(*) AS linked
            """
        else:
            tdi_q = """
                MATCH (dr:DataReference {ref_type: 'tdi_call'})
                WHERE NOT (dr)-[:CALLS_TDI_FUNCTION]->()
                MATCH (tdi:TDIFunction {facility_id: dr.facility_id})
                WHERE tdi.name = dr.raw_string
                   OR dr.raw_string CONTAINS tdi.name
                MERGE (dr)-[:CALLS_TDI_FUNCTION]->(tdi)
                RETURN count(*) AS linked
            """
        result = client.query(tdi_q, **params)
        tdi_linked = result[0]["linked"] if result else 0
        link_times["tdi_func"] = _time.monotonic() - t_s
        if tdi_linked:
            logger.info("Created %d CALLS_TDI_FUNCTION relationships", tdi_linked)

        # Update ref_count on CodeChunk nodes
        t_s = _time.monotonic()
        if scoped:
            client.query(
                """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                MATCH (c)-[r:CONTAINS_REF]->(d:DataReference)
                WITH c, count(r) AS ref_count
                SET c.ref_count = ref_count
                """,
                **params,
            )
        else:
            client.query("""
                MATCH (c:CodeChunk)-[r:CONTAINS_REF]->(d:DataReference)
                With c, count(r) AS ref_count
                SET c.ref_count = ref_count
            """)
        link_times["ref_count"] = _time.monotonic() - t_s

        link_detail = " ".join(
            f"{k}={v:.1f}s" for k, v in link_times.items() if v >= 0.1
        )
        if link_detail:
            logger.info("link_chunks_to_data_nodes timing: [%s]", link_detail)

        return refs_created


def link_example_mdsplus_paths(
    graph_client: GraphClient,
    example_id: str,
) -> int:
    """Create DataReference nodes and link to DataNodes for a specific example.

    Args:
        graph_client: GraphClient instance
        example_id: ID of the CodeExample

    Returns:
        Number of DataReference nodes created/linked
    """
    # Create DataReference nodes and CONTAINS_REF relationships
    result = graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
        WHERE c.mdsplus_paths IS NOT NULL AND e.facility_id IS NOT NULL
        UNWIND c.mdsplus_paths AS path
        WITH c, e.facility_id AS facility, path
        MERGE (d:DataReference {raw_string: path, facility_id: facility})
        ON CREATE SET
            d.id = facility + ':mdsplus_path:' + path,
            d.ref_type = 'mdsplus_path'
        MERGE (c)-[:CONTAINS_REF]->(d)
        WITH d, facility
        MATCH (f:Facility {id: facility})
        MERGE (d)-[:AT_FACILITY]->(f)
        RETURN count(DISTINCT d) AS refs_created
        """,
        example_id=example_id,
    )
    refs_created = result[0]["refs_created"] if result else 0

    # Resolve to DataNodes (facility-scoped, two-phase)
    # Exact matching uses the facility and path index.
    graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
              -[:CONTAINS_REF]->(d:DataReference {ref_type: 'mdsplus_path'})
        WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
        WITH DISTINCT d
        MATCH (t:SignalNode {facility_id: d.facility_id, path: d.raw_string})
        MERGE (d)-[:RESOLVES_TO_NODE]->(t)
        """,
        example_id=example_id,
    )
    # Fuzzy matching stays within unresolved facility references.
    graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
              -[:CONTAINS_REF]->(d:DataReference {ref_type: 'mdsplus_path'})
        WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
        WITH DISTINCT d
        MATCH (t:SignalNode {facility_id: d.facility_id})
        WHERE t.path ENDS WITH substring(d.raw_string, 1)
           OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
        MERGE (d)-[:RESOLVES_TO_NODE]->(t)
        """,
        example_id=example_id,
    )

    # Update ref_count
    graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
              -[r:CONTAINS_REF]->(d:DataReference)
        WITH c, count(r) AS ref_count
        SET c.ref_count = ref_count
        """,
        example_id=example_id,
    )

    return refs_created


__all__ = [
    "link_chunks_to_imas_paths",
    "link_chunks_to_ids_roots",
    "link_chunks_to_data_nodes",
    "link_example_mdsplus_paths",
    "link_examples_to_facility",
]
