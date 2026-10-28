"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import NameStage, RefineStopReason, SNRunStatus
from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.doc_links import (
    find_name_links,
    find_name_references,
    find_prose_bracket_spans,
)
from imas_codex.standard_names.ledger import reattach_produced_name_edges
from imas_codex.standard_names.protection import (
    ProtectedDeletionError,
    automatic_deletion_protections,
    filter_automatic_deletion_candidates,
    refuse_protected_automatic_deletion,
)
from imas_codex.standard_names.provenance_lifecycle import (
    deletion_change_cypher,
    deletion_change_params,
)
from imas_codex.standard_names.review.holds import (
    docs_acceptance_set,
    docs_acceptance_where,
    docs_accepted,
    has_open_docs_hold_where,
    open_holds,
)
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

_UNINITIALIZED_SERVICES = object()
_services: Any = _UNINITIALIZED_SERVICES


def wire_services(services: Any) -> None:
    """Install entrypoint collaborators for graph operations."""
    global _services
    _services = services


def get_services() -> Any:
    """Return the wired collaborators, installing the composition root on first read.

    Every storage-module reader reaches the collaborators through this function,
    so wiring here is what makes a reader independent of some other module
    having imported the graph facade first. ``None`` is a deliberate refusal —
    it says the root was explicitly removed, not that it was never installed —
    and still raises.
    """
    if _services is _UNINITIALIZED_SERVICES:
        from imas_codex.standard_names import graph_ops  # noqa: F401

    if _services is _UNINITIALIZED_SERVICES or _services is None:
        raise RuntimeError("Graph services are not wired; import graph_ops first")
    return _services


def services_for_call() -> Any:
    """Return the wired collaborators, installing the composition root on first call."""
    return get_services()


logger = logging.getLogger(__name__)


_STABLE_BINDING_NAME_STAGES: frozenset[str] = frozenset(
    {
        NameStage.drafted.value,
        NameStage.reviewed.value,
        NameStage.accepted.value,
        NameStage.approved.value,
    }
)


_TERMINAL_BINDING_NAME_STAGES: frozenset[str] = frozenset(
    {
        NameStage.superseded.value,
        NameStage.exhausted.value,
        NameStage.contested.value,
    }
)


_SETTLED_SOURCE_BINDING_STAGES: frozenset[str] = frozenset(
    {
        NameStage.superseded.value,
        NameStage.exhausted.value,
    }
)


_CLAIM_TIMEOUT_SECONDS = 300
DEFAULT_POOL_BATCH_SIZE = 25
_CLAIM_VERIFY_SETTLE_SECONDS = 0.5

_PHASE_TO_POOL: dict[str, str] = {
    "generate": "compose",
    "generate_name": "compose",
    "compose": "compose",
    "regen": "refine_name",
    "refine_name": "refine_name",
    "refine_docs": "refine_docs",
    "validate": "validate",
    "validate_name": "validate",
    "review_names": "review",
    "review_docs": "review",
    "review": "review",
    "enrich": "enrich",
    "enrich_links": "enrich",
}

_PHASE_TO_STANDARD_NAME_COST_FIELD: dict[str, str] = {
    "generate_name": "llm_cost_generate_name",
    "review_name": "llm_cost_review_name",
    "refine_name": "llm_cost_refine_name",
    "generate_docs": "llm_cost_generate_docs",
    "review_docs": "llm_cost_review_docs",
    "refine_docs": "llm_cost_refine_docs",
}


def docs_review_winner_query_body() -> str:
    """Return the shared traversal that selects a winning docs group."""
    return """
        MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        WHERE review.review_axis = 'docs'
          AND review.review_group_id IS NOT NULL
          AND NOT EXISTS {
              MATCH (sn)-[:HAS_REVIEW]->(non_winner:StandardNameReview)
              WHERE non_winner.review_axis = 'docs'
                AND non_winner.resolution_method
                    IN $docs_review_non_winning_methods
          }
        WITH review.review_group_id AS review_group_id,
             max(CASE WHEN coalesce(review.is_canonical, false)
                 THEN 1 ELSE 0 END) AS canonical_group,
             max(review.reviewed_at) AS group_reviewed_at,
             [item IN collect({
                 review_id: review.id,
                 reviewed_at: review.reviewed_at,
                 resolution_method: review.resolution_method,
                 score: review.score
             }) WHERE item.resolution_method IN $docs_review_winning_methods]
                 AS winning_records
        WHERE size(winning_records) > 0
        UNWIND winning_records AS winning_record
        WITH review_group_id, canonical_group, group_reviewed_at, winning_record
        ORDER BY canonical_group DESC, group_reviewed_at DESC,
                 review_group_id DESC, winning_record.reviewed_at DESC,
                 winning_record.review_id DESC
        WITH collect({
            review_group_id: review_group_id,
            review_id: winning_record.review_id,
            resolution_method: winning_record.resolution_method,
            score: winning_record.score
        })[0] AS winner
        RETURN winner
    """


DERIVED_PARENT_CLEANUP_DELETION_LIMIT = 80


class DerivedParentCleanupRefusal(RuntimeError):
    """A structural cleanup cannot prove that its deletion is safe."""


class _TransactionQuery:
    """Expose a Neo4j transaction through the graph-query interface."""

    def __init__(self, transaction: Any) -> None:
        self._transaction = transaction

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run one query without leaving the caller-owned transaction."""
        return [dict(record) for record in self._transaction.run(cypher, **params)]


def _ensure_json(value: Any) -> str | None:
    """Ensure a value is a JSON string, not a raw dict/list (Neo4j rejects Maps)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _ensure_list(value: Any) -> list[str]:
    """Coerce *value* to a list of strings (for multi-valued fields).

    - ``None`` / empty string → ``[]``
    - scalar string → ``[value]``
    - list → passed through
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _scalar_domain(value: Any) -> str | None:
    """Coerce a possibly-list ``physics_domain`` value to a scalar string.

    Defensive helper for reads from the graph during the schema migration
    window — legacy nodes may still have a list-valued ``physics_domain``
    before ``sn clear`` is run.  Returns the first non-empty entry from
    a list, or the string itself, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    return None


def reclassify_standard_name_domain(
    name: str,
    domain: str,
    *,
    reason: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reassign one name's domain property and relationship atomically.

    The domain write is one statement, so the property, the source set and
    the edges move together or not at all. The change record is written
    afterwards, so a failed domain write records no change for a domain
    that did not move.
    """
    from imas_codex.core.physics_domain import PhysicsDomain

    name = (name or "").strip()
    reason = (reason or "").strip()
    if not name:
        return {"ok": False, "reason": "a standard name is required"}
    if not reason:
        return {"ok": False, "reason": "--reason is mandatory for a domain move"}
    from imas_codex.standard_names.domain_decision import (
        DomainCorrection,
        DomainDecision,
        write_name_domain,
    )

    try:
        target_domain = PhysicsDomain(domain).value
    except ValueError:
        valid = ", ".join(item.value for item in PhysicsDomain)
        return {
            "ok": False,
            "reason": f"unknown physics domain {domain!r}; one of: {valid}",
        }

    with get_services().GraphClient() as gc:
        rows = list(
            gc.query(
                "MATCH (name:StandardName {id: $id}) "
                "RETURN name.physics_domain AS domain, "
                "name.name_stage AS stage",
                id=name,
            )
        )
        if not rows:
            return {"ok": False, "reason": f"standard name {name!r} not found"}

        before = rows[0]
        result = {
            "ok": True,
            "name": name,
            "stage": before["stage"],
            "from_domain": before["domain"],
            "to_domain": target_domain,
            "dry_run": dry_run,
        }
        if dry_run or before["domain"] == target_domain:
            result["noop"] = before["domain"] == target_domain
            return result

        write_name_domain(
            gc,
            name,
            DomainDecision(
                name=name,
                candidates=(),
                choice=None,
                probability=None,
                model=None,
                decided_at=None,
                undecided_reason="a recorded correction names the domain",
            ),
            correction=DomainCorrection(domain=target_domain, reason=reason),
        )
        change_id = f"sn-change:{uuid.uuid4()}"
        mutation = list(
            gc.query(
                """
                MATCH (name:StandardName {id: $id})
                SET name.updated_at = datetime()
                CREATE (change:StandardNameChange {
                  id: $change_id, from_name: name.id, to_name: name.id,
                  operation: 'reclassify_domain', reason: $reason,
                  origin: 'catalog_edit', changed_at: datetime($changed_at),
                  internal: true
                })
                MERGE (name)-[:HAS_INTERNAL_CHANGE]->(change)
                RETURN name.id AS id
                """,
                id=name,
                change_id=change_id,
                reason=reason,
                changed_at=datetime.now(UTC).isoformat(),
            )
        )
        if [row.get("id") for row in mutation] != [name]:
            raise RuntimeError(
                "standard name changed or disappeared during domain reclassification"
            )
        result["change_id"] = change_id
        return result


_PHYSICS_ABBREVIATIONS: dict[str, str] = {
    "ExB": "e_cross_b",
    "BxGradB": "b_cross_grad_b",
}


def normalize_name_id(name: str) -> str:
    """Normalize a standard name ID: expand physics abbreviations, force lowercase.

    Applies known physics abbreviation expansions (case-sensitive match on the
    abbreviation, e.g. ``ExB`` → ``e_cross_b``) and then forces the entire
    string to lowercase.  This must be called before grammar round-trip
    validation so that ISN's parser sees only valid lowercase tokens.

    Args:
        name: Raw standard name ID as produced by the LLM.

    Returns:
        Normalized lowercase name ID with physics abbreviations expanded.
    """
    for abbrev, expansion in _PHYSICS_ABBREVIATIONS.items():
        name = name.replace(abbrev, expansion)
    return name.lower()


@functools.lru_cache(maxsize=1)
def _public_segment_token_index() -> dict[str, tuple[str, ...]]:
    """Return the public grammar's token-to-segment index.

    ``StandardNameIR.qualifiers`` deliberately preserve authored order without
    imposing the legacy flat segment model.  The graph's compatibility columns
    still need the narrower segment labels for search filters, so this derives
    them from ``get_grammar_context()`` instead of duplicating grammar
    vocabulary in codex.
    """
    from imas_standard_names import get_grammar_context

    index: dict[str, list[str]] = {}
    for section in get_grammar_context().get("vocabulary_sections", []):
        segment = section.get("segment")
        if segment not in get_services()._GRAMMAR_SEGMENT_COLUMNS:
            continue
        for token in section.get("tokens") or []:
            index.setdefault(str(token), []).append(str(segment))
    return {token: tuple(segments) for token, segments in index.items()}


@functools.lru_cache(maxsize=1)
def _public_grammar_segment_order() -> tuple[str, ...]:
    """Return segment names in their public grammar render order."""
    from imas_standard_names import get_grammar_context

    return tuple(
        str(section["segment"])
        for section in get_grammar_context().get("vocabulary_sections", [])
        if section.get("segment")
    )


def _segments_from_ir(ir: Any) -> dict[str, str | None]:
    """Project a strict ISN IR into legacy scalar graph columns.

    The IR is the structural authority.  For a binary root, the first operand
    supplies compatibility base/qualifier/projection/locus/mechanism columns;
    the binary expression itself remains losslessly represented by
    ``HAS_PARENT`` edges.  ``transformation`` is the outermost operator only,
    so consumers that still expect one scalar can inspect it without mistaking
    that projection for the complete operator tree.
    """
    columns: dict[str, str | None] = dict.fromkeys(
        get_services()._GRAMMAR_SEGMENT_COLUMNS
    )
    operators = list(getattr(ir, "operators", ()) or ())
    primary = ir
    if operators:
        outer = operators[0]
        columns["transformation"] = get_services()._coerce_segment_value(
            getattr(outer, "op", None)
        )
        # A binary operator can sit below any number of unary wrappers, and a
        # binary's primary operand can itself be another binary expression.
        # Follow the first argument at every such node until the compatibility
        # view reaches a real leaf instead of the IR's structural placeholder.
        while True:
            primary_operators = list(getattr(primary, "operators", ()) or ())
            primary_binary = next(
                (
                    operator
                    for operator in primary_operators
                    if getattr(operator, "args", None)
                ),
                None,
            )
            if primary_binary is None:
                break
            primary = primary_binary.args[0]

    base = getattr(primary, "base", None)
    base_token = get_services()._coerce_segment_value(getattr(base, "token", None))
    base_kind = getattr(base, "kind", None)
    if base_token:
        is_geometry = (
            isinstance(base_kind, StrEnum) and base_kind is base_kind.__class__.GEOMETRY
        )
        column = "geometric_base" if is_geometry else "physical_base"
        columns[column] = base_token

    projection = getattr(primary, "projection", None)
    if projection is not None:
        shape = get_services()._coerce_segment_value(getattr(projection, "shape", None))
        column = "coordinate" if shape == "coordinate" else "component"
        columns[column] = get_services()._coerce_segment_value(
            getattr(projection, "axis", None)
        )

    token_index = _public_segment_token_index()
    qualifier_values: dict[str, list[str]] = {}
    qualifiers = list(getattr(primary, "qualifiers", ()) or ())

    locus = getattr(primary, "locus", None)
    if locus is not None:
        qualifiers.extend(list(getattr(locus, "qualifiers", ()) or ()))
        locus_token = get_services()._coerce_segment_value(
            getattr(locus, "token", None)
        )
        locus_type = get_services()._coerce_segment_value(getattr(locus, "type", None))
        locus_column = {
            "position": "position",
            "region": "region",
            "geometry": "geometry",
            "entity": "object",
        }.get(locus_type)
        if locus_column and locus_token:
            columns[locus_column] = locus_token

    for qualifier in qualifiers:
        token = get_services()._coerce_segment_value(getattr(qualifier, "token", None))
        if not token:
            continue
        for segment in token_index.get(token, ()):
            qualifier_values.setdefault(segment, []).append(token)
            break
    for segment, tokens in qualifier_values.items():
        columns[segment] = "_".join(tokens)

    mechanism = getattr(primary, "mechanism", None)
    if mechanism is not None:
        columns["process"] = get_services()._coerce_segment_value(
            getattr(mechanism, "token", None)
        )

    return columns


def _parse_grammar(name: str) -> dict[str, Any]:
    """Parse ``name`` with the ISN grammar API.

    Accept/reject and segment extraction use the public strict ``parse`` /
    ``compose`` contract.  Scalar columns are a compatibility projection of
    the lossless IR; failure of the optional flat Pydantic projection is never
    treated as invalid and cannot clear a valid ordered operator expression.

    Returns a dict with ``grammar_parse_version`` (ISN package version
    string), ``validation_diagnostics_json`` (JSON array of diagnostic
    objects, ``"[]"`` when the IR parser rejects the name), and all
    bare-name segment columns. When strict parsing rejects the name the segment
    columns are all ``None``; the name's non-compliance is
    recorded authoritatively by ``validation_status='quarantined'`` at
    validate time (:func:`validate_worker`), not by a separate flag.
    """
    try:
        import dataclasses

        import imas_standard_names
        from imas_standard_names import compose, parse

        version: str = imas_standard_names.__version__
    except ImportError:
        return {
            "grammar_parse_version": None,
            "validation_diagnostics_json": None,
            **dict.fromkeys(get_services()._GRAMMAR_SEGMENT_COLUMNS),
        }

    try:
        result = parse(name, strict=True)
        rendered = compose(result.ir)
        if rendered != name:
            raise ValueError(f"strict grammar round-trip produced {rendered!r}")
        diagnostics = []
        for diagnostic in result.diagnostics:
            if dataclasses.is_dataclass(diagnostic):
                diagnostics.append(dataclasses.asdict(diagnostic))
            elif hasattr(diagnostic, "model_dump"):
                diagnostics.append(diagnostic.model_dump(mode="json"))
            else:
                diagnostics.append(str(diagnostic))
        diags = json.dumps(diagnostics)
    except Exception as exc:
        logger.debug("ISN strict grammar rejected '%s': %s", name, exc)
        return {
            "grammar_parse_version": version,
            "validation_diagnostics_json": "[]",
            **dict.fromkeys(get_services()._GRAMMAR_SEGMENT_COLUMNS),
        }

    return {
        "grammar_parse_version": version,
        "validation_diagnostics_json": diags,
        **_segments_from_ir(result.ir),
    }


def _semantic_ir_expression(ir: Any) -> str:
    """Render a recursive public grammar IR as a compact semantic expression.

    The public composer remains the authority for leaf spelling.  Operator
    applications are rendered as ``operator(argument, ...)`` so operand scope
    stays visible to reviewers instead of being flattened into compatibility
    columns.
    """
    from imas_standard_names import compose

    operators = list(getattr(ir, "operators", ()) or ())
    if not operators:
        return compose(ir)

    outer = operators[0]
    arguments = list(getattr(outer, "args", ()) or ())
    if arguments:
        rendered_arguments = ", ".join(
            _semantic_ir_expression(argument) for argument in arguments
        )
        return f"{outer.op}({rendered_arguments})"

    operand = ir.model_copy(update={"operators": operators[1:]})
    return f"{outer.op}({_semantic_ir_expression(operand)})"


def strict_review_grammar_context(name: str) -> dict[str, Any]:
    """Return lossless public-ISN grammar evidence for name review.

    Strict parsing and canonical equality are delegated to the shared lossless
    adapter.  The compatibility projection is derived from that same recursive
    IR, while ``semantic_ir`` keeps nested operator scope explicit.
    """
    import imas_standard_names

    from imas_codex.standard_names.grammar_adapter import parse_canonical_name

    parsed = parse_canonical_name(name)
    fields = _segments_from_ir(parsed.ir)
    return {
        "grammar_parse_version": imas_standard_names.__version__,
        "grammar_round_trip": parsed.name,
        "grammar_projection": [
            {"field": field, "value": value}
            for field, value in fields.items()
            if value is not None
        ],
        "semantic_ir": _semantic_ir_expression(parsed.ir),
        "semantic_ir_json": json.dumps(
            parsed.ir.model_dump(mode="json"), sort_keys=True
        ),
        **fields,
    }


def _compute_link_status(links: list[str] | None) -> str | None:
    """Determine link resolution status from link prefixes.

    Returns 'resolved' if all links are ``name:`` or URL prefixed,
    'unresolved' if any link has ``dd:`` prefix (pending resolution),
    or None if no links exist.
    """
    if not links:
        return None
    has_unresolved = any(link.startswith("dd:") for link in links)
    return "unresolved" if has_unresolved else "resolved"


_LINK_RE = re.compile(
    r"""
    \[([^\]]*?)\]\(name:([a-z0-9_]+)\)  |  # markdown link [text](name:xxx)
    \bname:([a-z0-9_]+)\b                   # bare name:xxx reference
    """,
    re.VERBOSE,
)


_DD_LINK_RE = re.compile(
    r"""
    \[([^\]]*?)\]\(dd:([^\)]+)\)  |  # markdown link [text](dd:path)
    \bdd:([a-z0-9_/]+)\b             # bare dd:path reference
    """,
    re.VERBOSE,
)


def _extract_links_from_docs(
    documentation: str | None,
    *,
    self_name: str | None = None,
) -> list[str] | None:
    """Extract structured links from documentation markdown text.

    Looks for ``name:xxx`` and ``dd:path`` references in the documentation
    and returns an order-preserving deduplicated list of link strings
    suitable for the ``links`` field on StandardName nodes.

    When ``self_name`` is set, a ``name:<self_name>`` reference is
    dropped: prose may legitimately mention the current entry by name
    in one or more paragraphs, but the structured ``links`` index is a
    cross-reference list and a self-loop is meaningless. The ISNC
    SQLite catalog enforces UNIQUE(name, link) and a self-reference
    that survived deduplication used to crash ``validate_catalog``.
    """
    if not documentation:
        return None

    links: list[str] = []
    seen: set[str] = set()
    self_link = f"name:{self_name}" if self_name else None

    for m in _LINK_RE.finditer(documentation):
        name_id = m.group(2) or m.group(3)
        if name_id:
            ref = f"name:{name_id}"
            if ref == self_link:
                continue
            if ref not in seen:
                links.append(ref)
                seen.add(ref)

    for m in _DD_LINK_RE.finditer(documentation):
        dd_path = m.group(2) or m.group(3)
        if dd_path:
            ref = f"dd:{dd_path}"
            if ref not in seen:
                links.append(ref)
                seen.add(ref)

    return links if links else None


_pipeline_buckets_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}


_historical_cpi_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}


__all__ = [
    "Any",
    "Counter",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_ORPHAN_SWEEP_TIMEOUT_S",
    "DEFAULT_REFINE_ROTATIONS",
    "DERIVED_PARENT_CLEANUP_DELETION_LIMIT",
    "DerivedParentCleanupRefusal",
    "GraphClient",
    "GraphSchema",
    "NameStage",
    "Path",
    "ProtectedDeletionError",
    "RefineStopReason",
    "SNRunStatus",
    "StrEnum",
    "UTC",
    "_CLAIM_TIMEOUT_SECONDS",
    "_DD_LINK_RE",
    "_LINK_RE",
    "_PHYSICS_ABBREVIATIONS",
    "_SETTLED_SOURCE_BINDING_STAGES",
    "_STABLE_BINDING_NAME_STAGES",
    "_TERMINAL_BINDING_NAME_STAGES",
    "_TransactionQuery",
    "_compute_link_status",
    "_ensure_json",
    "_ensure_list",
    "_extract_links_from_docs",
    "_historical_cpi_cache",
    "_parse_grammar",
    "_pipeline_buckets_cache",
    "_public_grammar_segment_order",
    "_public_segment_token_index",
    "_scalar_domain",
    "_segments_from_ir",
    "_semantic_ir_expression",
    "annotations",
    "apply_signed_manifest",
    "automatic_deletion_protections",
    "datetime",
    "defaultdict",
    "deletion_change_cypher",
    "deletion_change_params",
    "deque",
    "docs_acceptance_set",
    "docs_acceptance_where",
    "docs_accepted",
    "filter_automatic_deletion_candidates",
    "find_name_links",
    "find_name_references",
    "find_prose_bracket_spans",
    "functools",
    "has_open_docs_hold_where",
    "hashlib",
    "inspect",
    "json",
    "logger",
    "logging",
    "normalize_name_id",
    "nullcontext",
    "open_holds",
    "os",
    "re",
    "reattach_produced_name_edges",
    "reclassify_standard_name_domain",
    "refuse_protected_automatic_deletion",
    "retry_on_deadlock",
    "strict_review_grammar_context",
    "suppress",
    "time",
    "uuid",
]
