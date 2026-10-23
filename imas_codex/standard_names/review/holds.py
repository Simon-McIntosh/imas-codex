"""Review vetoes remain separate from judged scores and have one graph owner."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

HOLD_KINDS = frozenset(
    {
        "registry_conformance",
        "banned_prose",
        "documentation_contradiction",
        "link_target_mismatch",
        "docs_lint_error",
        "cascade_contradiction",
    }
)


def finding_key(kind: str, finding: str) -> str:
    """Use stable text identity without losing the finding shown to a reader."""
    if kind not in HOLD_KINDS:
        raise ValueError(f"Unknown review hold kind: {kind}")
    normalized = re.sub(r"\s+", " ", finding).strip().casefold()
    if not normalized:
        raise ValueError("A review hold needs a finding")
    return hashlib.sha256(f"{kind}\n{normalized}".encode()).hexdigest()


def hold_id(name_id: str, axis: str, kind: str, finding: str) -> str:
    if axis not in {"name", "docs"}:
        raise ValueError(f"Unknown review hold axis: {axis}")
    return f"{name_id}:{axis}:{finding_key(kind, finding)}"


@contextmanager
def _client(graph: Any | None) -> Iterator[Any]:
    if graph is not None:
        yield graph
    else:
        with GraphClient() as client:
            yield client


def raise_hold(
    name_id: str,
    axis: str,
    kind: str,
    finding: str,
    evidence: str,
    raised_by: str,
    *,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Raise or refresh a finding; a dismissed identity stays dismissed."""
    if not evidence.strip() or not raised_by.strip():
        raise ValueError("A review hold needs evidence and an origin")
    identity = hold_id(name_id, axis, kind, finding)
    with _client(graph) as client:
        rows = client.query(
            """
            MATCH (sn:StandardName {id: $name_id})
            MERGE (hold:ReviewHold {id: $id})
            ON CREATE SET hold.axis = $axis, hold.kind = $kind,
                          hold.finding_key = $finding_key,
                          hold.raised_at = datetime(), hold.status = 'open'
            SET hold.finding = CASE WHEN hold.status = 'dismissed'
                                    THEN hold.finding ELSE $finding END,
                hold.evidence = CASE WHEN hold.status = 'dismissed'
                                     THEN hold.evidence ELSE $evidence END,
                hold.raised_by = CASE WHEN hold.status = 'dismissed'
                                      THEN hold.raised_by ELSE $raised_by END,
                hold.status = CASE WHEN hold.status = 'dismissed'
                                   THEN 'dismissed' ELSE 'open' END
            MERGE (sn)-[:HAS_HOLD]->(hold)
            RETURN hold{.*} AS hold
            """,
            name_id=name_id,
            id=identity,
            axis=axis,
            kind=kind,
            finding_key=finding_key(kind, finding),
            finding=finding,
            evidence=evidence,
            raised_by=raised_by,
        )
    if not rows:
        raise ValueError(f"StandardName not found: {name_id}")
    return rows[0]["hold"]


def sync_holds(
    name_id: str,
    axis: str,
    kinds: set[str] | frozenset[str],
    pending: list[dict[str, str]] | tuple[dict[str, str], ...],
    *,
    graph: Any | None = None,
    raise_fn: Any = None,
    clear_fn: Any = None,
) -> list[str]:
    """Raise current findings and clear stale findings owned by these checks."""
    raise_fn = raise_hold if raise_fn is None else raise_fn
    clear_fn = clear_holds_not_raised if clear_fn is None else clear_fn
    raised: list[str] = []
    raised_keys: set[str] = set()
    with _client(graph) as client:
        for hold in pending:
            record = raise_fn(
                name_id,
                axis,
                hold["kind"],
                hold["finding"],
                hold["evidence"],
                hold["raised_by"],
                graph=client,
            )
            raised.append(record["id"])
            raised_keys.add(finding_key(hold["kind"], hold["finding"]))
        if kinds:
            clear_fn(name_id, axis, set(kinds), raised_keys, graph=client)
    return raised


def clear_holds_not_raised(
    name_id: str,
    axis: str,
    checked_kinds: set[str],
    raised_keys: set[str],
    *,
    graph: Any | None = None,
) -> int:
    """Clear only open findings from checks that actually ran this review."""
    if not checked_kinds <= HOLD_KINDS:
        raise ValueError("Unknown review hold kind")
    if axis not in {"name", "docs"}:
        raise ValueError(f"Unknown review hold axis: {axis}")
    with _client(graph) as client:
        rows = client.query(
            """
            MATCH (:StandardName {id: $name_id})-[:HAS_HOLD]->(hold:ReviewHold)
            WHERE hold.axis = $axis AND hold.status = 'open'
              AND hold.kind IN $checked_kinds
              AND NOT hold.finding_key IN $raised_keys
            SET hold.status = 'cleared', hold.triaged_at = datetime(),
                hold.triaged_by = 'review-check'
            RETURN count(hold) AS cleared
            """,
            name_id=name_id,
            axis=axis,
            checked_kinds=sorted(checked_kinds),
            raised_keys=sorted(raised_keys),
        )
        _promote_if_eligible(client, name_id, axis)
    return rows[0]["cleared"] if rows else 0


def dismiss_hold(
    hold_id: str, by: str, reason: str, *, graph: Any | None = None
) -> bool:
    """A reasoned manual dismissal permanently suppresses this finding key."""
    if not by.strip() or not reason.strip():
        raise ValueError("Dismissing a hold requires a triager and reason")
    with _client(graph) as client:
        rows = client.query(
            """
            MATCH (sn:StandardName)-[:HAS_HOLD]->(hold:ReviewHold {id: $id})
            WHERE hold.status <> 'dismissed'
            SET hold.status = 'dismissed', hold.triaged_by = $by,
                hold.triaged_at = datetime(), hold.triage_reason = $reason
            RETURN hold.id AS id, hold.axis AS axis, sn.id AS name_id
            """,
            id=hold_id,
            by=by,
            reason=reason,
        )
        if rows:
            _promote_if_eligible(client, rows[0]["name_id"], rows[0]["axis"])
    return bool(rows)


def _promote_if_eligible(client: Any, name_id: str, axis: str) -> None:
    """Release a cleared or dismissed document only through the canonical gate.

    The same predicate, not a second copy of its terms: a document reached by
    clearing its last hold must clear exactly the bar the stranded-promotion
    pass applies, including an accepted name axis, an unquarantined row and a
    reachable winning docs review. A row reachable only by score would accept a
    document the normal gate never would.
    """
    if axis != "docs":
        return
    from imas_codex.standard_names.graph_ops import (
        PROMOTE_STRANDED_DOCS_WHERE,
        docs_review_eligibility_params,
    )

    client.query(
        f"""
        MATCH (sn:StandardName {{id: $name_id}})
        WHERE {PROMOTE_STRANDED_DOCS_WHERE}
        SET sn.updated_at = datetime(), {docs_acceptance_set()}
        RETURN sn.id AS id
        """,
        name_id=name_id,
        min_score=DEFAULT_MIN_SCORE,
        **docs_review_eligibility_params(),
    )


def open_holds(
    name_id: str, axis: str, *, graph: Any | None = None
) -> list[dict[str, Any]]:
    """Return the current actionable findings in deterministic order."""
    if axis not in {"name", "docs"}:
        raise ValueError(f"Unknown review hold axis: {axis}")
    with _client(graph) as client:
        rows = client.query(
            """
            MATCH (:StandardName {id: $name_id})-[:HAS_HOLD]->(hold:ReviewHold)
            WHERE hold.axis = $axis AND hold.status = 'open'
            RETURN hold{.*} AS hold
            ORDER BY hold.raised_at, hold.id
            """,
            name_id=name_id,
            axis=axis,
        )
    return [row["hold"] for row in rows]


def has_open_docs_hold_where(alias: str = "sn") -> str:
    """Cypher predicate for an open documentation veto on a bound name."""
    if not alias.isidentifier():
        raise ValueError("Cypher alias must be an identifier")
    return (
        "EXISTS { MATCH (" + alias + ")-[:HAS_HOLD]->(hold:ReviewHold) "
        "WHERE hold.axis = 'docs' AND hold.status = 'open' }"
    )


def docs_acceptance_where(alias: str = "sn") -> str:
    """Cypher fragment shared by acceptance decisions and downstream readers."""
    return f"({alias}.reviewer_score_docs >= $min_score AND NOT {has_open_docs_hold_where(alias)})"


def docs_acceptance_set(alias: str = "sn") -> str:
    """Apply acceptance and resolve an open edit in the same graph write."""
    if not alias.isidentifier():
        raise ValueError("Cypher alias must be an identifier")
    return (
        f"{alias}.docs_stage = 'accepted', "
        f"{alias}.edit_status = CASE WHEN {alias}.edit_status = 'open' "
        f"THEN 'applied' ELSE {alias}.edit_status END"
    )


def docs_accepted(
    score: float, min_score: float, name_id: str, *, graph: Any | None = None
) -> bool:
    """Accept only a passing judged score without an open documentation hold."""
    return score >= min_score and not open_holds(name_id, "docs", graph=graph)
