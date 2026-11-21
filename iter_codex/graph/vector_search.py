"""Neo4j 2026.01 SEARCH clause builder for vector similarity queries.

Generates Cypher 25 MATCH + SEARCH syntax, replacing legacy
db.index.vector.queryNodes() procedure calls.

In-index pre-filtering (``WHERE`` inside ``SEARCH``) runs against the ANN
candidate selection itself, so a selective predicate does not lose its
matches to a global top-k cut.  It requires the predicate's properties to
be registered as additional vector index properties (the ``WITH [...]``
clause of ``CREATE VECTOR INDEX``); a predicate over a property the index
does not carry is only valid as a post-filter.
"""

from __future__ import annotations


def build_vector_search(
    index: str,
    label: str,
    *,
    where_clauses: list[str] | None = None,
    prefilter_clauses: list[str] | None = None,
    k: str = "$k",
    node_alias: str = "n",
    score_alias: str = "score",
    embedding_param: str = "$embedding",
) -> str:
    """Build a Cypher 25 MATCH + SEARCH clause for vector similarity.

    Generates the correct Neo4j 2026.01 SEARCH syntax::

        CYPHER 25
        MATCH (n:Label)
        SEARCH n IN (
          VECTOR INDEX index_name
          FOR $embedding
          WHERE n.filter_prop = $val
          LIMIT $k
        ) SCORE AS score
        WHERE n.prop = $val

    Args:
        index: Vector index name (e.g. 'imas_node_embedding').
        label: Node label (e.g. 'IMASNode').
        where_clauses: Filter expressions applied as post-filters
            after the ANN candidate selection.  Both property filters
            (``n.facility_id = $f``) and relationship pattern predicates
            (``NOT (n)-[:REL]->(:Other)``) are valid here.
        prefilter_clauses: Property predicates applied inside ``SEARCH``,
            before the ANN cut.  Each referenced property must be
            registered as an additional vector index property, otherwise
            the query fails to plan.
        k: Cypher expression for the ANN candidate limit
            (default: "$k").  Can be a literal like "20".
        node_alias: Variable name for the matched node.
        score_alias: Variable name for the similarity score.
        embedding_param: Cypher expression for the query embedding
            (default: "$embedding").

    Returns:
        A complete query prefix starting with ``CYPHER 25``.
        Append OPTIONAL MATCH, WITH, and RETURN clauses as needed.
    """
    parts = [
        "CYPHER 25",
        f"MATCH ({node_alias}:{label})",
        f"SEARCH {node_alias} IN (",
        f"  VECTOR INDEX {index}",
        f"  FOR {embedding_param}",
    ]

    if prefilter_clauses:
        parts.append(f"  WHERE {' AND '.join(prefilter_clauses)}")

    parts.extend(
        [
            f"  LIMIT {k}",
            f") SCORE AS {score_alias}",
        ]
    )

    if where_clauses:
        parts.append(f"WHERE {' AND '.join(where_clauses)}")

    return "\n".join(parts)
