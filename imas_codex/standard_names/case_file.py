"""One case file per standard name or unnamed DD source, built by code.

A case file is the whole of what a generator, a reference set or a benchmark is
allowed to read about one quantity: the text of every DD source bound to it and
of every structure enclosing those sources up to the root, the units and
coordinates the sources declare, the other names of its family with their
definitions, its nearest neighbours in the catalog by subject and by locus with
what separates each from it, the grammar's segments and the vocabulary, registry
entry and conventions a name for this quantity may use, the incumbent name,
definition and documentation where one exists, and the reason an adjudicator
recorded for it. Nothing here calls a model and nothing writes to the graph:
every statement is a read, so building a case file for every member of a group
changes nothing stored.

The record (:class:`CaseFile`) carries its content, its digest and its own
completeness verdict together. The digest covers the content alone, so a rebuild
over unchanged graph text reproduces it. The verdict names every channel that is
missing, so a reader refuses a record whose enclosing-structure text was never
appended to that channel instead of reading the empty block as "nothing to say".
A channel that legitimately has nothing to report says so in prose ("no family",
"no neighbour"), which is what makes the two states distinguishable.

Assembly reuses the repository's context owners rather than re-deriving them:

* the enclosing chain and the statements its structures make --
  ``ancestor_statements`` in
  :mod:`imas_codex.standard_names.worker_tasks.worker_common`, the rule a review
  and a compose prompt already share;
* family -- :func:`~imas_codex.standard_names.context.fetch_sibling_family`;
* neighbours by subject --
  :func:`~imas_codex.standard_names.context.fetch_review_neighbours`, whose
  ``channel_status`` separates "no neighbour" from "lookup failed", and by locus
  -- :func:`~imas_codex.standard_names.context.fetch_locus_neighbours`;
* grammar, vocabulary, registry entry and conventions --
  :func:`~imas_codex.standard_names.context.build_compose_context`,
  :func:`~imas_codex.standard_names.context.conventions_context_for` and
  :func:`~imas_codex.standard_names.context.locus_context_for`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CASE_FILE_SCHEMA = "imas-codex.standard-name-case-file"
CASE_FILE_VERSION = 1

MAX_ANCESTOR_DEPTH = 24

_WRITE_CLAUSES = (
    " CREATE ",
    " MERGE ",
    " SET ",
    " DELETE ",
    " REMOVE ",
    " DROP ",
)

WITHHELD_CHANNELS = (
    "source_text",
    "enclosing_structure_text",
    "family",
    "neighbours",
    "grammar",
    "incumbent",
)

_REASON_FIELDS = (
    "review_axis",
    "comments",
    "suggested_name",
    "suggestion_justification",
    "score",
    "reviewed_at",
    "resolution_method",
)

_TARGET_QUERY = """
// CASE_FILE_TARGET
MATCH (sn:StandardName {id: $id})
OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
WITH sn, review
ORDER BY coalesce(review.reviewed_at, datetime('1970-01-01T00:00:00Z')) DESC,
         coalesce(review.id, '')
WITH sn, [r IN collect(review) WHERE r IS NOT NULL] AS reviews
RETURN properties(sn) AS name,
       CASE WHEN size(reviews) = 0 THEN null ELSE properties(reviews[0]) END
         AS reason
"""

_SOURCE_TEXT_QUERY = """
// CASE_FILE_SOURCE_TEXT
UNWIND $paths AS pid
OPTIONAL MATCH (n:IMASNode {id: pid})
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coordinate:IMASCoordinateSpec)
RETURN pid AS id,
       n IS NOT NULL AS exists,
       n.unit AS unit,
       n.ndim AS ndim,
       n.data_type AS data_type,
       n.documentation AS documentation,
       n.description AS description,
       [c IN collect(DISTINCT coordinate.id) WHERE c IS NOT NULL]
         AS coordinates
"""

_SOURCE_CHAIN_QUERY = f"""
// CASE_FILE_SOURCE_CHAIN
UNWIND $paths AS pid
OPTIONAL MATCH path = (n:IMASNode {{id: pid}})-[:HAS_PARENT*1..{MAX_ANCESTOR_DEPTH}]->
                      (ancestor:IMASNode)
WITH pid, ancestor, length(path) AS depth
ORDER BY pid, depth
WITH pid,
     [r IN collect({{path: ancestor.id, depth: depth,
                     documentation: ancestor.documentation,
                     description: ancestor.description}})
      WHERE r.path IS NOT NULL] AS ancestors
RETURN pid AS id, ancestors
"""

_SOURCE_ROOT_QUERY = f"""
// CASE_FILE_SOURCE_ROOT
UNWIND $paths AS pid
OPTIONAL MATCH (n:IMASNode {{id: pid}})-[:HAS_PARENT*0..{MAX_ANCESTOR_DEPTH}]->
                (root:IMASNode)
WHERE NOT (root)-[:HAS_PARENT]->(:IMASNode)
RETURN pid AS id, [r IN collect(DISTINCT root.id) WHERE r IS NOT NULL] AS roots
"""

_FAMILY_SEED_QUERY = """
// CASE_FILE_FAMILY_SEED
MATCH (sn:StandardName {id: $id})
RETURN sn.id AS id, coalesce(sn.physical_base, '') AS physical_base,
       coalesce(sn.name_stage, '') AS name_stage
"""


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_id: str = CASE_FILE_SCHEMA
    version: int = CASE_FILE_VERSION
    target: dict[str, Any]
    content: dict[str, Any]
    completeness: dict[str, Any]
    digest: str
    tokens: int
    provenance: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Digest and size
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def case_file_digest(content: Mapping[str, Any]) -> str:
    """Return the digest of a case file's content.

    Only content is covered: the digest states "this is the text the generator
    was given", and a verdict or a build stamp beside it must not change that
    statement.
    """
    return hashlib.sha256(_canonical(content).encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    """Return the project's token estimate for *text*.

    The ``len(text) // 4`` rule the batching module owns, reused so a case
    file's size and a batch budget are one measure.
    """
    from imas_codex.standard_names.batching import estimate_tokens as _estimate

    return _estimate(text)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def completeness_verdict(content: Mapping[str, Any]) -> dict[str, Any]:
    """Judge one case file's content, naming every channel that is missing.

    A channel is missing when it was asked for and did not answer: a status of
    ``unavailable``, a bound source whose node or text is absent, an enclosing
    chain that never reached the root, or a source whose enclosing structures
    carry no text while it is not itself the root. A channel that answered
    "nothing here" -- a family of one, no neighbour in the catalog -- is present
    and complete, because the record states it in prose.
    """
    missing: list[str] = []
    checked = [
        "incumbent",
        "sources",
        "enclosing",
        "family",
        "neighbours:by_subject",
        "neighbours:by_locus",
        "grammar",
        "grammar:vocabulary",
        "grammar:registry",
        "grammar:conventions",
    ]
    target = content.get("target") or {}
    kind = target.get("kind")

    if kind == "name" and not content.get("incumbent"):
        missing.append("incumbent")

    sources = list(content.get("sources") or ())
    if kind == "name" and not sources:
        missing.append("bound_source")
    if kind == "source" and not sources:
        missing.append("sources")

    for source in sources:
        source_id = str(source.get("id") or "")
        if not source.get("exists"):
            missing.append(f"source_node:{source_id}")
            continue
        if not str(source.get("text") or "").strip():
            missing.append(f"source_text:{source_id}")
        enclosing = source.get("enclosing") or {}
        chain = list(enclosing.get("chain") or ())
        statements = list(enclosing.get("statements") or ())
        if not chain and not enclosing.get("is_root"):
            missing.append(f"enclosing_structure_text:{source_id}")
        elif len(chain) > 1 and not statements:
            missing.append(f"enclosing_structure_text:{source_id}")
        if chain and not enclosing.get("chain_to_root"):
            missing.append(f"chain_to_root:{source_id}")

    family = content.get("family") or {}
    if (
        family.get("status") not in {"members", "none"}
        or not str(family.get("statement") or "").strip()
    ):
        missing.append("family")

    neighbours = content.get("neighbours") or {}
    for channel in ("by_subject", "by_locus"):
        block = neighbours.get(channel) or {}
        if (
            block.get("status") not in {"ok", "none"}
            or not str(block.get("statement") or "").strip()
        ):
            missing.append(f"neighbours:{channel}")

    grammar = content.get("grammar") or {}
    if kind == "name":
        if grammar.get("status") not in {"parsed", "unparsed"}:
            missing.append("grammar")
        else:
            for channel in ("vocabulary", "registry", "conventions"):
                block = grammar.get(channel) or {}
                if block.get("status") not in {"ok", "none"}:
                    missing.append(f"grammar:{channel}")
    elif grammar.get("status") != "not_a_name":
        missing.append("grammar")

    verdict = "complete" if not missing else "incomplete"
    return {
        "verdict": verdict,
        "usable": verdict == "complete",
        "missing": sorted(missing),
        "checked": checked,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_source(source: Mapping[str, Any], lines: list[str]) -> None:
    lines.append(
        f"  source {source.get('id')}: unit="
        f"{source.get('unit') or '(none declared)'} ndim={source.get('ndim')} "
        f"coordinates={list(source.get('coordinates') or ())}"
    )
    lines.append(f"    text: {source.get('text') or '(no text recorded)'}")
    enclosing = source.get("enclosing") or {}
    for statement in enclosing.get("statements") or ():
        lines.append(f"    enclosing {statement.get('path')}: {statement.get('text')}")
    if enclosing.get("status") != "ok" or not enclosing.get("statements"):
        lines.append(
            f"    enclosing: {enclosing.get('statement') or '(not assembled)'}"
        )
    chain = list(enclosing.get("chain") or ())
    lines.append(
        f"    chain: {len(chain)} structures, root="
        f"{enclosing.get('root') or '(not reached)'}"
    )


def _render_neighbours(block: Mapping[str, Any], lines: list[str]) -> None:
    lines.append(f"  {block.get('statement') or '(not assembled)'}")
    for neighbour in block.get("members") or ():
        lines.append(
            f"  {neighbour.get('name')}: "
            f"{neighbour.get('description') or '(no definition)'} -- separates by: "
            f"{neighbour.get('separation') or '(not recorded)'}"
        )


def render_case_file(record: CaseFile | Mapping[str, Any]) -> str:
    """Render a case file as the text its consumers read.

    Every content section appears, including the channels that answered
    "nothing here", so the rendering is also a readable account of what a
    generator was shown and what it was not.
    """
    content = record.content if isinstance(record, CaseFile) else record["content"]
    target = content.get("target") or {}
    lines: list[str] = [f"CASE FILE: {target.get('id', '')} ({target.get('kind', '')})"]

    incumbent = content.get("incumbent")
    lines.append("")
    lines.append("Incumbent name, definition, documentation:")
    if incumbent:
        lines.append(f"  name: {incumbent.get('name', '')}")
        lines.append(f"  kind: {incumbent.get('kind') or '(none recorded)'}")
        lines.append(f"  unit: {incumbent.get('unit') or '(none declared)'}")
        lines.append(
            f"  definition: {incumbent.get('description') or '(none recorded)'}"
        )
        lines.append(
            f"  documentation: {incumbent.get('documentation') or '(none recorded)'}"
        )
    else:
        lines.append("  none (the target is an unnamed source, or none is recorded)")

    reason = content.get("adjudicator_reason")
    lines.append("")
    lines.append("Adjudicator's recorded reason:")
    if reason:
        for key, value in reason.items():
            if value in (None, "", [], {}):
                continue
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  none recorded.")

    lines.append("")
    lines.append("Bound sources and their enclosing structures:")
    sources = list(content.get("sources") or ())
    if not sources:
        lines.append("  none (no DD source is bound to this target).")
    for source in sources:
        _render_source(source, lines)

    family = content.get("family") or {}
    lines.append("")
    lines.append("Family:")
    lines.append(f"  {family.get('statement') or '(not assembled)'}")
    for member in family.get("members") or ():
        lines.append(
            f"  {member.get('name')}: {member.get('description') or '(no definition)'} "
            f"[differs on {member.get('differs_on') or 'nothing recorded'}]"
        )

    neighbours = content.get("neighbours") or {}
    lines.append("")
    lines.append("Nearest neighbours by subject:")
    _render_neighbours(neighbours.get("by_subject") or {}, lines)
    lines.append("")
    lines.append("Nearest neighbours by locus:")
    _render_neighbours(neighbours.get("by_locus") or {}, lines)

    grammar = content.get("grammar") or {}
    lines.append("")
    lines.append("Grammar for this quantity:")
    if grammar.get("status") == "parsed":
        lines.append(f"  segments: {json.dumps(grammar.get('segments') or {})}")
    else:
        lines.append(f"  segments: {grammar.get('statement') or '(not parsed)'}")
    for channel in ("vocabulary", "registry", "conventions"):
        block = grammar.get(channel) or {}
        lines.append(f"  {channel}: {block.get('statement') or '(not assembled)'}")
        for entry in block.get("entries") or ():
            lines.append(f"    {json.dumps(entry, ensure_ascii=False, sort_keys=True)}")

    verdict = completeness_verdict(content)
    lines.append("")
    lines.append(
        f"Completeness: {verdict['verdict']}; usable={str(verdict['usable']).lower()}"
    )
    lines.append(
        f"Missing: {', '.join(verdict['missing']) if verdict['missing'] else 'none'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# What separates two names
# ---------------------------------------------------------------------------

_FACT_LABELS = {
    "base.token": "base",
    "base.kind": "base kind",
    "locus.token": "locus",
    "locus.type": "locus type",
    "locus.relation": "locus relation",
    "projection.token": "projection",
    "mechanism.token": "mechanism",
}


def _fact_label(path: str) -> str:
    if path in _FACT_LABELS:
        return _FACT_LABELS[path]
    parts = path.split(".")
    if parts[0] == "qualifiers":
        return "qualifier"
    if parts[0] == "operators":
        return "operator"
    return " ".join(part for part in parts if part != "token")


def _flatten_ir(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a parsed name's IR dump into readable scalar facts."""
    facts: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            facts.update(_flatten_ir(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            facts.update(_flatten_ir(child, f"{prefix}.{index}"))
    elif value not in (None, "", False):
        facts[prefix] = str(value)
    return facts


def name_facts(name: str) -> dict[str, str]:
    """Return one name's grammar facts, or an empty dict when it does not parse."""
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        return _flatten_ir(parse_canonical_name(name).ir.model_dump(mode="json"))
    except Exception:
        logger.debug("case_file: %r does not parse", name, exc_info=True)
        return {}


def separation(own: Mapping[str, Any], other: Mapping[str, Any]) -> str:
    """State what separates *other* from *own*, fact by fact.

    Both sides carry ``facts`` and ``unit``. Every grammar fact whose value
    differs, or that only one side carries, is named. When no fact differs the
    two names read identically through the grammar, and the statement says so
    instead of leaving the comparison empty -- a locus neighbour at the same
    grammar reading is separated by its DD locus alone.
    """
    own_facts = own.get("facts") or {}
    other_facts = other.get("facts") or {}
    parts: list[str] = []
    labelled: dict[str, str] = {}
    for path in sorted(set(own_facts) | set(other_facts)):
        left = own_facts.get(path) or ""
        right = other_facts.get(path) or ""
        if left == right:
            continue
        label = _fact_label(path)
        labelled[label] = (
            f"{label} is {right or '(absent)'} where this name's is "
            f"{left or '(absent)'}"
        )
    parts.extend(labelled[label] for label in sorted(labelled))
    own_unit = own.get("unit") or ""
    other_unit = other.get("unit") or ""
    if own_unit != other_unit:
        parts.append(
            f"unit is {other_unit or '(none declared)'} where this name's is "
            f"{own_unit or '(none declared)'}"
        )
    if not parts:
        return "no grammar fact differs; it is separated only by its DD source locus"
    return "; ".join(parts)


def _plain(value: Any) -> Any:
    """Convert a graph value into JSON-safe primitives."""
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def assert_read_only(statements: Iterable[str]) -> None:
    """Raise when any statement would write to the graph.

    Assembly is a read: a case file that persisted anything would change the
    catalog for every later run. The check is on the statement text before it is
    sent, so a writer added to this module fails here.
    """
    for statement in statements:
        upper = f" {' '.join(str(statement).split()).upper()} "
        for clause in _WRITE_CLAUSES:
            if clause in upper:
                raise ValueError(
                    "case file assembly may only read the graph; found "
                    f"{clause.strip()!r} in a statement"
                )


class _Reader:
    """Send every case-file statement through the read-only guard."""

    def __init__(self, gc: Any, statements: list[str]) -> None:
        self._gc = gc
        self.statements = statements

    def query(self, statement: str, **params: Any) -> list[Mapping[str, Any]]:
        assert_read_only([statement])
        self.statements.append(statement)
        return [dict(row) for row in self._gc.query(statement, **params) or ()]


# ---------------------------------------------------------------------------
# One source and its enclosing chain
# ---------------------------------------------------------------------------


def _source_text(row: Mapping[str, Any]) -> str:
    """Return the source's own text: documentation first, then description."""
    for field in ("documentation", "description"):
        text = " ".join(str(row.get(field) or "").split())
        if text:
            return text
    return ""


def _enclosing_block(
    path: str,
    ancestors: Sequence[Mapping[str, Any]],
    roots: Sequence[str],
    *,
    withheld: bool,
) -> dict[str, Any]:
    """Assemble one source's chain to the root and the statements it makes.

    The statement rule is the extraction owner's
    (:func:`~imas_codex.standard_names.worker_tasks.worker_common.ancestor_statements`):
    documentation then description, whitespace collapsed, a statement a nearer
    structure already made dropped. The full chain is kept beside the
    statements, so a structure that states nothing still appears in the record
    and the chain's completion is a fact the verdict can read.
    """
    from imas_codex.standard_names.worker_tasks.worker_common import (
        ancestor_statements,
    )

    ordered = sorted(ancestors, key=lambda row: row.get("depth") or 0)
    chain = [
        {"path": str(row.get("path") or ""), "depth": _plain(row.get("depth"))}
        for row in ordered
    ]
    statements = (
        []
        if withheld
        else [
            {"path": str(entry.get("path") or ""), "text": str(entry.get("text") or "")}
            for entry in ancestor_statements(ordered, limit=None)
        ]
    )
    reachable = {str(root) for root in roots}
    root = chain[-1]["path"] if chain else path
    chain_to_root = bool(reachable) and root in reachable
    is_root = not chain and path in reachable
    if withheld:
        status = "absent"
        statement = "the enclosing-structure text was withheld"
    elif is_root:
        status = "ok"
        statement = "the source is the root: no enclosing structure exists"
    elif not statements:
        status = "absent"
        statement = "the enclosing structures state no text"
    else:
        status = "ok"
        statement = f"{len(statements)} enclosing statements, nearest first"
    return {
        "chain": chain,
        "statements": statements,
        "chain_to_root": chain_to_root,
        "root": root if chain_to_root else None,
        "is_root": is_root,
        "status": status,
        "statement": statement,
    }


def _read_sources(
    reader: _Reader,
    paths: Sequence[str],
    withheld: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = list(paths)
    if not wanted:
        return []
    texts = {
        str(row.get("id") or ""): row
        for row in reader.query(_SOURCE_TEXT_QUERY, paths=wanted)
    }
    chains = {
        str(row.get("id") or ""): list(row.get("ancestors") or ())
        for row in reader.query(_SOURCE_CHAIN_QUERY, paths=wanted)
    }
    roots = {
        str(row.get("id") or ""): list(row.get("roots") or ())
        for row in reader.query(_SOURCE_ROOT_QUERY, paths=wanted)
    }
    sources: list[dict[str, Any]] = []
    for path in wanted:
        row = texts.get(path) or {}
        sources.append(
            {
                "id": f"dd:{path}",
                "exists": bool(row.get("exists")),
                "text": "" if "source_text" in withheld else _source_text(row),
                "unit": _plain(row.get("unit")) or None,
                "ndim": _plain(row.get("ndim")),
                "data_type": _plain(row.get("data_type")),
                "coordinates": [
                    _plain(coordinate) for coordinate in row.get("coordinates") or ()
                ],
                "enclosing": _enclosing_block(
                    path,
                    chains.get(path) or [],
                    roots.get(path) or [],
                    withheld="enclosing_structure_text" in withheld,
                ),
            }
        )
    return sources


def _incumbent(props: Mapping[str, Any]) -> dict[str, Any]:
    """Read the standing name, definition and documentation from the node."""
    return {
        key: _plain(props.get(key))
        for key in (
            "id",
            "kind",
            "unit",
            "description",
            "documentation",
            "physical_base",
            "name_stage",
        )
    } | {"name": _plain(props.get("id")), "source_paths": _source_paths(props)}


def _source_paths(props: Mapping[str, Any]) -> list[str]:
    """Return the DD paths bound to a name, without the ``dd:`` prefix."""
    return sorted(
        str(path)[3:] if str(path).startswith("dd:") else str(path)
        for path in props.get("source_paths") or ()
        if path
    )


def _reason_block(reason: Any) -> dict[str, Any] | None:
    """Read the adjudicator's recorded reason from the canonical review."""
    if not reason:
        return None
    plain = _plain(reason)
    block = {field: plain.get(field) for field in _REASON_FIELDS if plain.get(field)}
    return block or None


# ---------------------------------------------------------------------------
# Grammar, family and neighbours
# ---------------------------------------------------------------------------


def _grammar_block(
    name: str | None,
    compose_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the grammar's reading of the name, its vocabulary and rules.

    The vocabulary is narrowed to the closed-vocabulary segments whose token
    lists contain a token this name uses; a name whose base belongs to an open
    class selects no such segment, and the statement says so. Conventions and
    the locus registry entry come from the context owners.
    """
    if not name:
        not_a_name = {"status": "none", "statement": "not a name", "entries": []}
        return {
            "status": "not_a_name",
            "statement": "the target is an unnamed source: no grammar reading applies",
            "segments": {},
            "vocabulary": dict(not_a_name),
            "registry": dict(not_a_name),
            "conventions": dict(not_a_name),
        }

    facts = name_facts(name)
    status = "parsed" if facts else "unparsed"

    sections: list[dict[str, Any]] = []
    vocabulary_status = "ok"
    try:
        context = compose_context
        if context is None:
            from imas_codex.standard_names.context import build_compose_context

            context = build_compose_context()
        sections = [
            dict(section)
            for section in context.get("vocabulary_sections") or ()
        ]
    except Exception:
        logger.debug("case_file: compose context unavailable", exc_info=True)
        vocabulary_status = "unavailable"

    used = {value for path, value in facts.items() if path.endswith(".token")}
    selected = [
        {
            "segment": section.get("segment"),
            "tokens": list(section.get("tokens") or ()),
        }
        for section in sections
        if used & {str(token) for token in section.get("tokens") or ()}
    ]
    if vocabulary_status != "ok":
        vocabulary = {
            "status": "unavailable",
            "statement": "the grammar's vocabulary could not be read",
            "entries": [],
        }
    elif selected:
        vocabulary = {
            "status": "ok",
            "statement": (
                f"{len(selected)} of the grammar's {len(sections)} closed segments "
                "admit a token this name uses"
            ),
            "entries": selected,
        }
    else:
        vocabulary = {
            "status": "ok",
            "statement": (
                "no closed-vocabulary segment admits a token this name uses; its "
                "base belongs to an open class"
            ),
            "entries": [],
        }

    from imas_codex.standard_names.context import (
        conventions_context_for,
        locus_context_for,
    )

    registry_entries: list[dict[str, Any]] = []
    base = facts.get("base.token")
    if base:
        registry_entries.append({"registry": "physical_base", "token": base})
    try:
        locus = locus_context_for(name)
    except Exception:
        logger.debug("case_file: locus registry unavailable", exc_info=True)
        locus = None
    if locus:
        registry_entries.append({"registry": "locus", **locus})
    registry = {
        "status": "ok" if registry_entries else "none",
        "statement": (
            f"{len(registry_entries)} registry entries selected for this quantity"
            if registry_entries
            else "no registry entry is selected for this name"
        ),
        "entries": registry_entries,
    }

    try:
        conventions = conventions_context_for(name)
    except Exception:
        logger.debug("case_file: conventions unavailable", exc_info=True)
        conventions = None
    convention_entries: list[dict[str, Any]] = []
    for kind in ("concepts", "notation", "templates"):
        for entry in (conventions or {}).get(kind) or ():
            convention_entries.append({"kind": kind, **entry})
    if convention_entries:
        conventions_block = {
            "status": "ok",
            "statement": (
                f"{len(convention_entries)} ruled conventions this name cites"
            ),
            "entries": convention_entries,
        }
    elif facts:
        conventions_block = {
            "status": "none",
            "statement": "the catalog has ruled no convention for this name",
            "entries": [],
        }
    else:
        conventions_block = {
            "status": "none",
            "statement": (
                "the installed grammar cannot parse this name, so no convention "
                "is selected"
            ),
            "entries": [],
        }

    return {
        "status": status,
        "statement": (
            f"{len(facts)} grammar facts"
            if facts
            else "the installed grammar cannot parse this name"
        ),
        "segments": facts,
        "vocabulary": vocabulary,
        "registry": registry,
        "conventions": conventions_block,
    }


def _family_block(
    name: str | None,
    client: Any,
    reader: _Reader | None,
    family_reader: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Assemble the other names of the family and what separates each.

    ``fetch_sibling_family`` returns ``None`` both for a family of one and for
    a failed query, so the seed is read first: a name in the graph whose family
    is empty states "no family", while a seed that cannot be read reports the
    channel unavailable and the verdict refuses it.
    """
    if not name:
        return {
            "status": "none",
            "statement": "the target is an unnamed source: it has no name family",
            "members": [],
        }
    if reader is None:
        return {
            "status": "unavailable",
            "statement": "no graph reader was given for the family channel",
            "members": [],
        }
    seeds = reader.query(_FAMILY_SEED_QUERY, id=name)
    if not seeds:
        return {
            "status": "unavailable",
            "statement": "the name is absent from the graph, so its family could not be read",
            "members": [],
        }
    if family_reader is None:
        from imas_codex.standard_names.context import fetch_sibling_family

        family_reader = fetch_sibling_family
    try:
        family = family_reader(name, gc=client)
    except Exception:
        logger.debug("case_file: family lookup failed for %s", name, exc_info=True)
        family = None
    base = str((seeds[0] or {}).get("physical_base") or "")
    if not family:
        return {
            "status": "none",
            "statement": (
                "no family: no other live name shares this name's pattern"
                + (f" over {base}" if base else "")
            ),
            "members": [],
        }
    members = [
        {
            "name": member.get("name"),
            "description": member.get("description") or "",
            "differs_on": member.get("operator_kind") or "nothing recorded",
            "axis": member.get("axis") or "",
        }
        for member in family.get("siblings") or ()
    ]
    parent = family.get("parent")
    return {
        "status": "members",
        "statement": (
            f"{len(members)} other name(s) of its family"
            + (
                f", over parent {parent.get('name')}"
                if parent
                else ", with no parent node"
            )
        ),
        "parent": parent,
        "anchor": family.get("anchor"),
        "members": members,
    }


def _neighbours_block(
    name: str | None,
    description: str,
    facts: Mapping[str, str],
    unit: str | None,
    paths: Sequence[str],
    client: Any,
    neighbour_reader: Callable[..., Any] | None,
    locus_reader: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Assemble nearest neighbours by subject and by locus, with separations."""
    own = {"facts": dict(facts), "unit": unit}

    def row(entry: Mapping[str, Any], source: str) -> dict[str, Any]:
        other_name = str(entry.get("id") or entry.get("name") or "")
        other_unit = _plain(entry.get("unit"))
        return {
            "name": other_name,
            "description": _plain(entry.get("description")) or "",
            "unit": other_unit,
            "source": source,
            "score": _plain(entry.get("score")),
            "shared_depth": _plain(entry.get("shared_depth")),
            "separation": separation(
                own,
                {"facts": name_facts(other_name), "unit": other_unit},
            ),
        }

    if not name:
        by_subject: dict[str, Any] = {
            "status": "none",
            "statement": "the target is an unnamed source: subject neighbours need a name",
            "members": [],
        }
    else:
        if neighbour_reader is None:
            from imas_codex.standard_names.context import fetch_review_neighbours

            neighbour_reader = fetch_review_neighbours
        try:
            channels = (
                neighbour_reader(
                    {
                        "id": name,
                        "description": description,
                        "physical_base": facts.get("base.token"),
                        "source_paths": [f"dd:{path}" for path in paths],
                    },
                    gc=client,
                )
                or {}
            )
        except Exception:
            logger.debug(
                "case_file: subject neighbourhood failed for %s", name, exc_info=True
            )
            channels = {}
        statuses = channels.get("channel_status") or {}
        attempted = [
            channel
            for channel in ("vector", "same_base")
            if statuses.get(channel) != "skipped"
        ]
        members: list[dict[str, Any]] = []
        seen: set[str] = set()
        for channel, source in (
            ("vector_neighbours", "subject embedding"),
            ("same_base_neighbours", "same physical base"),
        ):
            for entry in channels.get(channel) or ():
                entry_id = str(_plain(entry.get("id")) or "")
                if not entry_id or entry_id == name or entry_id in seen:
                    continue
                seen.add(entry_id)
                members.append(row(entry, source))
        if not attempted:
            by_subject = {
                "status": "none",
                "statement": (
                    "the name records no definition or physical base from which "
                    "subject neighbours could be found"
                ),
                "members": [],
            }
        elif any(statuses.get(channel) == "unavailable" for channel in attempted):
            by_subject = {
                "status": "unavailable",
                "statement": (
                    "a subject-neighbour lookup failed, so no absence can be claimed"
                ),
                "members": [],
            }
        elif members:
            by_subject = {
                "status": "ok",
                "statement": f"{len(members)} nearest name(s) by subject",
                "members": members,
            }
        else:
            by_subject = {
                "status": "none",
                "statement": "no neighbour by subject in the catalog",
                "members": [],
            }

    if not paths:
        by_locus: dict[str, Any] = {
            "status": "none",
            "statement": (
                "the target binds no DD source path, so it has no locus neighbour"
            ),
            "members": [],
        }
    else:
        if locus_reader is None:
            from imas_codex.standard_names.context import fetch_locus_neighbours

            locus_reader = fetch_locus_neighbours
        try:
            locus_rows = locus_reader(
                [f"dd:{path}" for path in paths], sn_id=name or "", gc=client
            )
        except Exception:
            logger.debug(
                "case_file: locus neighbourhood failed for %s", name, exc_info=True
            )
            locus_rows = None
        if locus_rows is None:
            by_locus = {
                "status": "unavailable",
                "statement": "the locus-neighbour lookup failed",
                "members": [],
            }
        elif locus_rows:
            by_locus = {
                "status": "ok",
                "statement": (
                    f"{len(locus_rows)} nearest name(s) by locus, deepest shared "
                    "path first"
                ),
                "members": [
                    row(entry, "locus")
                    for entry in locus_rows
                    if str(entry.get("id") or "") != (name or "")
                ],
            }
        else:
            by_locus = {
                "status": "none",
                "statement": "no other name shares this target's DD locus subtree",
                "members": [],
            }

    return {"by_subject": by_subject, "by_locus": by_locus}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _open_client(gc: Any) -> tuple[Any, bool]:
    if gc is not None:
        return gc, False
    from imas_codex.graph.client import GraphClient

    return GraphClient(), True


def _withheld_answer(channel: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "statement": f"{channel} was withheld from this record on purpose",
        "members": [],
    }


def _withheld_grammar() -> dict[str, Any]:
    return {
        "status": "withheld",
        "statement": "the grammar reading was withheld from this record on purpose",
        "segments": {},
        "vocabulary": _withheld_answer("the vocabulary"),
        "registry": _withheld_answer("the registry entries"),
        "conventions": _withheld_answer("the conventions"),
    }


def _checked_withheld(withhold: Sequence[str]) -> list[str]:
    withheld = sorted(set(withhold))
    unknown = sorted(set(withheld) - set(WITHHELD_CHANNELS))
    if unknown:
        raise ValueError(
            f"unknown withheld channel(s) {unknown}; expected a subset of "
            f"{list(WITHHELD_CHANNELS)}"
        )
    return withheld


def _assemble(
    target: Mapping[str, Any],
    *,
    incumbent: dict[str, Any] | None,
    reason: dict[str, Any] | None,
    sources: Sequence[Mapping[str, Any]],
    family: Mapping[str, Any],
    neighbours: Mapping[str, Any],
    grammar: Mapping[str, Any],
    withheld: Sequence[str],
    statements: Sequence[str],
) -> CaseFile:
    content: dict[str, Any] = {
        "target": dict(target),
        "incumbent": incumbent,
        "adjudicator_reason": reason,
        "sources": [dict(source) for source in sources],
        "family": dict(family),
        "neighbours": dict(neighbours),
        "grammar": dict(grammar),
        "withheld_channels": list(withheld),
    }
    verdict = completeness_verdict(content)
    rendered = render_case_file({"content": content})
    return CaseFile(
        target=dict(target),
        content=content,
        completeness=verdict,
        digest=case_file_digest(content),
        tokens=estimate_tokens(rendered),
        provenance={
            "graph_statements": len(statements),
            "model_calls": 0,
            "graph_writes": 0,
        },
    )


def _normalize_paths(paths: Iterable[str]) -> list[str]:
    return sorted(
        {
            str(path)[3:] if str(path).startswith("dd:") else str(path)
            for path in paths
            if str(path).strip()
        }
    )


def build_name_case_file(
    name: str,
    *,
    gc: Any = None,
    withhold: Sequence[str] = (),
    compose_context: Mapping[str, Any] | None = None,
    family_reader: Callable[..., Any] | None = None,
    neighbour_reader: Callable[..., Any] | None = None,
    locus_reader: Callable[..., Any] | None = None,
) -> CaseFile:
    """Build the one case file for a live standard name.

    Args:
        name: The standard name to build the record for.
        gc: A graph client to read through; one is opened and closed when
            omitted.
        withhold: Channels to leave out on purpose, named in
            :data:`WITHHELD_CHANNELS`. Used to show the completeness verdict
            refuses a record whose text is absent.
        compose_context: The grammar context to read the vocabulary from;
            defaults to :func:`~imas_codex.standard_names.context.build_compose_context`.
        family_reader: Override for
            :func:`~imas_codex.standard_names.context.fetch_sibling_family`.
        neighbour_reader: Override for
            :func:`~imas_codex.standard_names.context.fetch_review_neighbours`.
        locus_reader: Override for
            :func:`~imas_codex.standard_names.context.fetch_locus_neighbours`.

    Raises:
        ValueError: The name is not in the graph, or a withheld channel is not
            one of :data:`WITHHELD_CHANNELS`.
    """
    withheld = _checked_withheld(withhold)
    statements: list[str] = []
    client, own = _open_client(gc)
    try:
        reader = _Reader(client, statements)
        rows = reader.query(_TARGET_QUERY, id=name)
        if not rows:
            raise ValueError(f"no StandardName {name!r} exists to hold a case file")
        props = dict(rows[0].get("name") or {})
        facts = name_facts(name)
        paths = _source_paths(props)
        sources = _read_sources(reader, paths, withheld)
        grammar = (
            _withheld_grammar()
            if "grammar" in withheld
            else _grammar_block(name, compose_context)
        )
        family = (
            _withheld_answer("the family")
            if "family" in withheld
            else _family_block(name, client, reader, family_reader)
        )
        neighbours = (
            {
                "by_subject": _withheld_answer("the subject neighbours"),
                "by_locus": _withheld_answer("the locus neighbours"),
            }
            if "neighbours" in withheld
            else _neighbours_block(
                name,
                _plain(props.get("description")) or "",
                facts,
                _plain(props.get("unit")),
                paths,
                client,
                neighbour_reader,
                locus_reader,
            )
        )
        return _assemble(
            {"kind": "name", "id": name, "paths": paths},
            incumbent=None if "incumbent" in withheld else _incumbent(props),
            reason=_reason_block(rows[0].get("reason")),
            sources=sources,
            family=family,
            neighbours=neighbours,
            grammar=grammar,
            withheld=withheld,
            statements=statements,
        )
    finally:
        if own:
            try:
                client.close()
            except Exception:
                logger.debug("case_file: client close failed", exc_info=True)


def build_source_case_file(
    source_paths: Iterable[str],
    *,
    gc: Any = None,
    withhold: Sequence[str] = (),
    locus_reader: Callable[..., Any] | None = None,
) -> CaseFile:
    """Build the one case file for an unnamed DD source or source set.

    An unnamed source has no incumbent, no family and no grammar reading of its
    own name; its record states those plainly rather than leaving empty blocks.
    Its sources, their enclosing chains, declared units and coordinates, and its
    locus neighbours are assembled exactly as for a named target, so a path that
    has no name yet is read the same way once it acquires one.
    """
    withheld = _checked_withheld(withhold)
    paths = _normalize_paths(source_paths)
    if not paths:
        raise ValueError("a case file for a source needs at least one DD path")
    statements: list[str] = []
    client, own = _open_client(gc)
    try:
        reader = _Reader(client, statements)
        sources = _read_sources(reader, paths, withheld)
        grammar = (
            _withheld_grammar() if "grammar" in withheld else _grammar_block(None, None)
        )
        family = (
            _withheld_answer("the family")
            if "family" in withheld
            else _family_block(None, client, reader, None)
        )
        neighbours = (
            {
                "by_subject": _withheld_answer("the subject neighbours"),
                "by_locus": _withheld_answer("the locus neighbours"),
            }
            if "neighbours" in withheld
            else _neighbours_block(
                None, "", {}, None, paths, client, None, locus_reader
            )
        )
        return _assemble(
            {"kind": "source", "id": f"dd:{paths[0]}", "paths": paths},
            incumbent=None,
            reason=None,
            sources=sources,
            family=family,
            neighbours=neighbours,
            grammar=grammar,
            withheld=withheld,
            statements=statements,
        )
    finally:
        if own:
            try:
                client.close()
            except Exception:
                logger.debug("case_file: client close failed", exc_info=True)


def build_case_file(target: str | Mapping[str, Any], **kwargs: Any) -> CaseFile:
    """Build a case file for a name, or for a mapping carrying ``paths``."""
    if isinstance(target, Mapping):
        return build_source_case_file(target.get("paths") or (), **kwargs)
    return build_name_case_file(str(target), **kwargs)
