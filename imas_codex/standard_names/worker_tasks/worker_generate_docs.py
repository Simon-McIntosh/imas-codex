"""Documentation worker tasks and their explicit entrypoint collaborators."""

from __future__ import annotations

import asyncio
import logging
import re as _re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


def _enrich_dd_path_context(
    _worker_dependency, /, gc: Any, item: dict, source_path: str
) -> None:
    """Enrich an item dict with DD path context (keywords, clusters, version history).

    Fetches lightweight context from the IMASNode for *source_path* and populates:

    - ``dd_keywords``: list of keyword strings from the IMASNode.
    - ``dd_clusters``: list of ``{label, description, scope}`` dicts.
    - ``dd_version_history``: list of ``{version, change_type}`` dicts.
    - ``dd_parent_description``: description of the parent structure node.

    Best-effort — failures are debug-logged and the item is left without
    the missing key.  Modifies *item* in-place.
    """
    _DD_PATH_CONTEXT_QUERY = _worker_dependency("_DD_PATH_CONTEXT_QUERY")
    try:
        rows = list(gc.query(_DD_PATH_CONTEXT_QUERY, path=source_path))
    except Exception:
        logger.debug(
            "_enrich_dd_path_context: query failed for %s", source_path, exc_info=True
        )
        return
    if not rows:
        return
    row = rows[0]

    # Keywords
    kw = row.get("keywords")
    if kw:
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        if kw:
            item["dd_keywords"] = kw

    # DD ground truth — the authoritative definition the reviewer verifies
    # documentation claims against (physics_accuracy grounding).
    dd_doc = row.get("dd_documentation")
    if dd_doc:
        item["dd_documentation"] = dd_doc
    dd_desc = row.get("dd_description")
    if dd_desc:
        item["dd_description"] = dd_desc
    dd_units = row.get("dd_units")
    if dd_units:
        item["dd_units"] = dd_units

    # Parent description
    parent_desc = row.get("parent_description")
    if parent_desc:
        item["dd_parent_description"] = parent_desc

    # Clusters
    clusters_raw = row.get("clusters") or []
    clusters = [
        {
            "label": c["label"],
            "description": c.get("description") or "",
            "scope": c.get("scope") or "",
        }
        for c in clusters_raw
        if c.get("label")
    ]
    if clusters:
        item["dd_clusters"] = clusters

    # Version history
    changes = row.get("changes") or []
    version_history = []
    for ch in changes:
        change_id = ch.get("change_id") or ""
        change_type = ch.get("change_type") or ""
        parts = change_id.rsplit(":", 1)
        version = parts[-1] if len(parts) >= 2 else ""
        if version and change_type:
            version_history.append({"version": version, "change_type": change_type})
    if version_history:
        item["dd_version_history"] = version_history


def _enrich_for_docs_gen(
    _worker_dependency, /, gc: Any, items: list[dict], cocos_params: dict | None = None
) -> None:
    """Enrich generate_docs items with DD source context.

    Populates per-item keys (all guarded by ``if`` in the prompt template):

    - ``source_paths``: bare DD paths (``dd:`` prefix stripped).
    - ``dd_source_docs``: list of ``{id, documentation, unit}`` from linked
      :class:`~imas_codex.graph.models.IMASNode` nodes.
    - ``dd_aliases``: list of alias strings from IMASNodes.
    - ``related_neighbours``: graph-relationship neighbours derived from the
      first DD source path (via :func:`_related_path_neighbours`).
    - ``nearest_peers``: vector-similar :class:`StandardName` neighbours from
      :func:`_search_nearby_names` formatted as ``{tag, unit, physics_domain,
      doc_short, cocos_label}``.
    - ``cocos_label``: COCOS transformation type from the SN node.
    - ``cocos_guidance``: rendered sign convention guidance for the LLM.
    - ``sibling_family``: the HAS_PARENT sibling family (parent, anchor,
      siblings) from :func:`~imas_codex.standard_names.context.fetch_sibling_family`
      — drives the parallel-structure directive in the docs prompts.

    Modifies *items* in-place.  Never raises — individual item failures are
    debug-logged and the item is left without the missing key.
    """
    _ANCESTOR_CONTEXT_QUERY = _worker_dependency("_ANCESTOR_CONTEXT_QUERY")
    _DOCS_GEN_ENRICH_QUERY = _worker_dependency("_DOCS_GEN_ENRICH_QUERY")
    _DOCS_GEN_PARENT_CHILDREN_QUERY = _worker_dependency(
        "_DOCS_GEN_PARENT_CHILDREN_QUERY"
    )
    _enrich_dd_path_context = _worker_dependency("_enrich_dd_path_context")
    _related_path_neighbours = _worker_dependency("_related_path_neighbours")
    _search_nearby_names = _worker_dependency("_search_nearby_names")
    strip_dd_prefix = _worker_dependency("strip_dd_prefix")
    for item in items:
        sn_id = item.get("id")
        if not sn_id:
            continue

        # ── 1. Source paths + linked IMASNode documentation ───────────
        try:
            rows = list(gc.query(_DOCS_GEN_ENRICH_QUERY, sn_id=sn_id))
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: graph query failed for %s",
                sn_id,
                exc_info=True,
            )
            continue

        if not rows:
            continue
        row = rows[0]

        raw_paths = row.get("source_paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        source_paths = [
            strip_dd_prefix(p) for p in raw_paths if p and isinstance(p, str)
        ]
        if source_paths:
            item["source_paths"] = source_paths

        dd_nodes = [n for n in (row.get("dd_nodes") or []) if n and n.get("id")]
        if dd_nodes:
            from imas_codex.settings import get_dd_version
            from imas_codex.standard_names.dd_resolutions import resolve_dd_rows

            batch = resolve_dd_rows(
                [
                    {
                        "path": node["id"],
                        "unit": node.get("unit"),
                        "documentation": node.get("documentation"),
                    }
                    for node in dd_nodes
                ],
                dd_version=get_dd_version(),
            )
            if batch.refusals:
                logger.warning(
                    "Documentation context omitted %d changed DD fields: %s",
                    len(batch.refusals),
                    ", ".join(
                        f"{refusal.path}:{refusal.field.value}"
                        for refusal in batch.refusals
                    ),
                )
            item["dd_source_docs"] = [
                {
                    "id": n["id"],
                    # Rich-first: the LLM-enriched description grounds the docs
                    # prompts; fall back to terse source documentation only when
                    # no enriched description exists.
                    "documentation": n.get("description")
                    or context.documentation
                    or "",
                    "unit": context.unit or "",
                    "raw_dd_context": context.as_pipeline_item()["raw_dd_context"],
                    "published_dd_context": context.as_pipeline_item()[
                        "published_dd_context"
                    ],
                    "dd_resolution_ids": list(context.applied_resolution_ids),
                    "dd_resolution_converged_ids": list(
                        context.converged_resolution_ids
                    ),
                    "dd_resolution_manifest_digest": context.manifest_digest,
                }
                for resolved in batch.resolved
                if resolved.row_index < 5
                for n, context in [(dd_nodes[resolved.row_index], resolved.context)]
            ]
            aliases = [n["alias"] for n in dd_nodes if n.get("alias")]
            if aliases:
                item["dd_aliases"] = aliases

            # Surface the DD ancestor lineage: the primary source leaf is often
            # a terse template stub, while the quantity's physics + evaluation
            # locus live on a parent quantity node up the HAS_PARENT chain
            # (e.g. "…rotation velocity … at the pedestal top"). Mirrors the
            # name-generation grounding so docs ground on the real physics/locus
            # instead of the stub. Nearest rich ancestor first; capped.
            primary = dd_nodes[0]["id"]
            try:
                anc_rows = list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=primary))
            except Exception:
                anc_rows = []
            lineage: list[dict[str, str]] = []
            seen_anc: set[str] = set()
            for a in anc_rows:
                text = (a.get("description") or "").strip() or (
                    a.get("documentation") or ""
                ).strip()
                if not text or text in seen_anc:
                    continue
                seen_anc.add(text)
                lineage.append({"path": a.get("path") or "", "text": text})
                if len(lineage) >= 4:
                    break
            if lineage:
                item["ancestor_context"] = lineage

        # ── 1a. Derived-parent children grounding ─────────────────────
        # A derived parent has no DD source of its own; its concrete physics
        # lives in its accepted children. Inject them so generate_docs writes a
        # GENERALISED parent doc (the shared quantity) rather than over-
        # specialising to one child's component/qualifier (observed defect:
        # ``perturbed_velocity`` documented as its single ``normalized_parallel``
        # child). The placeholder description is dropped (null) so only real
        # child meaning grounds the prompt.
        if row.get("origin") == "derived":
            try:
                from imas_codex.standard_names.defaults import (
                    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
                )

                kid_rows = list(gc.query(_DOCS_GEN_PARENT_CHILDREN_QUERY, sn_id=sn_id))
                children = [
                    {
                        "name": k["name"],
                        "description": (
                            None
                            if k.get("description")
                            == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
                            else k.get("description")
                        ),
                        "unit": k.get("unit"),
                        "physics_domain": k.get("physics_domain"),
                    }
                    for k in kid_rows
                ]
                if children:
                    item["derived_children"] = children
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: derived children fetch failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 1b. COCOS sign convention context ─────────────────────────
        cocos_label = row.get("cocos_label")
        if cocos_label:
            item["cocos_label"] = cocos_label
            if cocos_params:
                try:
                    from imas_codex.standard_names.context import (
                        render_cocos_guidance,
                    )

                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, cocos_params
                    )
                except Exception:
                    logger.debug(
                        "_enrich_for_docs_gen: cocos guidance render failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        # ── 2. Graph-relationship neighbours from first DD source path ─
        if source_paths:
            try:
                related = _related_path_neighbours(gc, source_paths[0])
                if related:
                    item["related_neighbours"] = related
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: related_neighbours failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 2b. DD path context (keywords, clusters, version history) ─
        if source_paths:
            try:
                _enrich_dd_path_context(gc, item, source_paths[0])
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: dd_path_context failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 3. Vector-similar SN neighbours ───────────────────────────
        description = item.get("description") or sn_id.replace("_", " ")
        try:
            peers_raw = _search_nearby_names(description, k=6)
            peers = [
                {
                    "tag": f"name:{p['id']}",
                    "unit": p.get("unit") or "",
                    "physics_domain": p.get("physics_domain") or "",
                    "doc_short": (p.get("description") or "")[:120],
                    "cocos_label": p.get("cocos_label") or "",
                }
                for p in peers_raw
                if p.get("id") and p.get("id") != sn_id
            ][:5]
            if peers:
                item["nearest_peers"] = peers
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: nearest_peers failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 4. Parent/child component context ─────────────────────────
        try:
            # Check if this SN is a component of a parent
            parent_rows = list(
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $sn_id})-[r:HAS_PARENT]->(parent:StandardName)
                    RETURN parent.id AS name,
                           parent.description AS description,
                           parent.documentation AS documentation,
                           r.axis AS axis
                    LIMIT 1
                    """,
                    sn_id=sn_id,
                )
            )
            if parent_rows:
                p = parent_rows[0]
                item["parent_sn"] = {
                    "name": p["name"],
                    "description": p.get("description") or "",
                    "documentation": p.get("documentation") or "",
                }
                item["component_axis"] = p.get("axis") or ""

            # Check if this SN has children (is a parent)
            child_rows = list(
                gc.query(
                    """
                    MATCH (child:StandardName)-[r:HAS_PARENT]->(sn:StandardName {id: $sn_id})
                    RETURN child.id AS name,
                           child.description AS description,
                           r.axis AS axis
                    ORDER BY child.id
                    """,
                    sn_id=sn_id,
                )
            )
            if child_rows:
                from imas_codex.standard_names.families import sort_by_axis_convention

                child_dicts = [
                    {
                        "name": c["name"],
                        "description": c.get("description") or "",
                        "axis": c.get("axis") or "",
                    }
                    for c in child_rows
                ]
                item["child_components"] = sort_by_axis_convention(child_dicts)
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: component context failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 5. Derivative / base-quantity context ─────────────────────────
        try:
            from imas_codex.standard_names.families import DD_DERIVATIVE_MAP

            parent_sn_data = item.get("parent_sn")
            if parent_sn_data:
                parent_name = parent_sn_data["name"]
                # Query parent unit and sibling derivatives sharing same parent
                bq_rows = list(
                    gc.query(
                        """
                        MATCH (parent:StandardName {id: $parent_name})
                        OPTIONAL MATCH (sib:StandardName)-[:HAS_PARENT]->(parent)
                        WHERE sib.id <> $sn_id
                        WITH parent, collect(sib.id) AS siblings
                        RETURN parent.unit AS unit, siblings
                        """,
                        parent_name=parent_name,
                        sn_id=sn_id,
                    )
                )
                if bq_rows:
                    bq_row = bq_rows[0]
                    item["base_quantity"] = {
                        "name": parent_name,
                        "description": parent_sn_data.get("description") or "",
                        "documentation": parent_sn_data.get("documentation") or "",
                        "unit": bq_row.get("unit") or "",
                    }
                    if sn_id in DD_DERIVATIVE_MAP:
                        numerator, denominator = DD_DERIVATIVE_MAP[sn_id]
                        raw_siblings = bq_row.get("siblings") or []
                        siblings = [s for s in raw_siblings if s]
                        item["derivative_context"] = {
                            "numerator": numerator,
                            "denominator": denominator,
                            "siblings": siblings,
                        }
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: derivative context failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 6. Sibling-family (parallel-structure) context ─────────────
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id, gc=gc)
            if family and family.get("siblings"):
                item["sibling_family"] = family
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: sibling family failed for %s",
                sn_id,
                exc_info=True,
            )


def _nearby_names_for_docs_gen(
    _worker_dependency, /, gc: Any, items: list[dict]
) -> list[dict]:
    """Return accepted StandardNames in the same physics domain(s) as *items*.

    Queries up to 20 accepted SNs per unique domain found across the batch.
    Results are deduplicated and capped at 40 total.  Used to populate the
    batch-level ``nearby_existing_names`` context key in
    :func:`process_generate_docs_batch`.
    """
    _DOCS_GEN_NEARBY_QUERY = _worker_dependency("_DOCS_GEN_NEARBY_QUERY")
    domains: set[str] = set()
    for item in items:
        dom = item.get("physics_domain")
        if isinstance(dom, list):
            domains.update(dom)
        elif dom:
            domains.add(dom)

    nearby: list[dict] = []
    seen_ids: set[str] = set()
    batch_ids: set[str] = {item["id"] for item in items if item.get("id")}

    for domain in sorted(domains):
        # Use the first batch item id as the exclusion anchor; the query
        # already uses != so any id from the batch suffices.
        anchor = next(iter(batch_ids), "")
        try:
            rows = list(gc.query(_DOCS_GEN_NEARBY_QUERY, domain=domain, sn_id=anchor))
        except Exception:
            logger.debug(
                "_nearby_names_for_docs_gen: query failed for domain=%s",
                domain,
                exc_info=True,
            )
            continue
        for r in rows:
            nid = r.get("id")
            if nid and nid not in seen_ids and nid not in batch_ids:
                seen_ids.add(nid)
                nearby.append(dict(r))
                if len(nearby) >= 40:
                    break
        if len(nearby) >= 40:
            break

    return nearby


def _fill_template_sentence(
    _worker_dependency, /, sentence: str, fields: dict[str, str]
) -> str:
    """Substitute a name's own segments into one template sentence.

    Every ``{placeholder}`` the sentence carries is filled from the name. A
    sentence whose name supplies no value for one of its placeholders renders
    nothing, so a half-filled opening is never written.
    """
    filled = sentence.strip()
    for placeholder in set(_re.findall(r"\{([^{}]+)\}", filled)):
        if placeholder not in fields:
            return ""
        filled = filled.replace(f"{{{placeholder}}}", fields[placeholder])
    return filled


def _instantiated_template_sentences(
    _worker_dependency, /, name: str, templates: tuple[Any, ...]
) -> tuple[str | None, tuple[str, ...]]:
    """Fill the template sentences the grammar selects for one name.

    The grammar owns which sentences a name selects and in what order: each
    selected template's opening, then its note once per distinct note. Each
    sentence renders only where the name itself supplies its placeholders, so
    a name that does not fill its template's fields keeps the written opening.

    Returns the first renderable opening and the notes that render.
    """
    _fill_template_sentence = _worker_dependency("_fill_template_sentence")
    from imas_standard_names.grammar.parser import ParseError, parse
    from imas_standard_names.grammar.terms import template_sentences
    from imas_standard_names.grammar.vocab_loaders import load_subjects

    try:
        ir = parse(name, strict=True).ir
    except ParseError:
        return None, ()
    subject_tokens = [q.token for q in ir.qualifiers if q.token in load_subjects()]
    fields: dict[str, str] = {}
    if len(subject_tokens) == 1:
        fields["subject"] = subject_tokens[0].replace("_", " ")
    if ir.locus is not None:
        fields["locus"] = (
            f"{ir.locus.relation.value} {ir.locus.token.replace('_', ' ')}"
        )
    openings = {template.documentation.strip() for template in templates}
    opening: str | None = None
    notes: list[str] = []
    for sentence in template_sentences(name):
        filled = _fill_template_sentence(sentence, fields)
        if not filled:
            continue
        if sentence.strip() in openings:
            if opening is None:
                opening = filled
        elif filled not in notes:
            notes.append(filled)
    return opening, tuple(notes)


def _documentation_with_registry_form(
    _worker_dependency,
    /,
    name: str,
    documentation: str,
    conventions: dict[str, Any] | None,
) -> str:
    """Write the selected openings, notes and canonical forms before persistence."""
    _instantiated_template_sentences = _worker_dependency(
        "_instantiated_template_sentences"
    )
    normalize_prose_spelling = _worker_dependency("normalize_prose_spelling")
    from imas_standard_names.grammar.terms import AppliedConventions

    from imas_codex.standard_names.docs_gates import (
        _DISPLAY_MATH_RE,
        HouseStyleFindingClass,
        _is_canonical_equation,
        _locus_registry_tokens,
        _opening_line,
        house_style_findings,
        locus_definition_sentence,
    )

    written = normalize_prose_spelling(documentation)
    if not conventions:
        return written
    applied = AppliedConventions.model_validate(conventions)
    opening, notes = _instantiated_template_sentences(name, applied.templates)
    missing_notes = [note for note in notes if note not in written]
    if opening is not None or missing_notes:
        lines = written.splitlines()
        model_opening = _opening_line(written)
        first = next(
            (i for i, line in enumerate(lines) if line.strip() == model_opening), None
        )
        if first is None:
            written = "\n\n".join(
                ([opening] if opening is not None else []) + missing_notes
            )
        else:
            block = [opening if opening is not None else lines[first], *missing_notes]
            lines[first : first + 1] = ["\n\n".join(block)]
            written = "\n".join(lines)

    concepts = {f"concept:{concept.token}": concept for concept in applied.concepts}
    for match in reversed(list(_DISPLAY_MATH_RE.finditer(written))):
        findings = house_style_findings(applied, documentation=f"$${match.group(1)}$$")
        concept = next(
            (
                concepts[finding.registry_entry]
                for finding in findings
                if finding.registry_entry in concepts
                and finding.finding_class
                in {
                    HouseStyleFindingClass.NON_CANONICAL_EQUATION,
                    HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT,
                }
            ),
            None,
        )
        if concept is not None:
            written = (
                written[: match.start()]
                + f"$${concept.equation}$$"
                + written[match.end() :]
            )

    for concept in applied.concepts:
        if concept.restates_equation and not any(
            _is_canonical_equation(concept, match.group(1))
            for match in _DISPLAY_MATH_RE.finditer(written)
        ):
            written = f"{written.rstrip()}\n\n$${concept.equation}$$"

    locus_tokens = _locus_registry_tokens()
    for concept in applied.concepts:
        if concept.token not in locus_tokens:
            continue
        sentence = locus_definition_sentence(concept)
        if sentence and sentence not in written:
            written = f"{written.rstrip()}\n\n{sentence}"
        break
    return written


async def process_generate_docs_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of accepted StandardNames for generate_docs.

    For each item in the batch:

    1. Enrich items with DD source context via :func:`_enrich_for_docs_gen`
       and :func:`_nearby_names_for_docs_gen` (source_paths, dd_source_docs,
       dd_aliases, nearest_peers, related_neighbours, nearby_existing_names).
    2. Render prompt via ``render_prompt("sn/generate_docs_user", {...})`` with
       reviewer feedback (reviewer_score_name, reviewer_comments_name),
       chain history, and the new DD context as context.
    3. Use ``get_model("sn-docs")`` — docs generation model.
    4. Call ``acall_llm_structured`` with ``service="standard-names"`` and
       response_model=``GeneratedDocs``.
    5. Reserve budget and charge ``LLMCostEvent``.
    6. Persist via ``persist_generated_docs`` (transitions docs_stage → 'drafted').
    7. On error: ``release_generate_docs_failed_claims``.
    8. Stream per-item progress: name + first 100 chars of generated description.

    Returns count of items successfully processed.
    """
    _DOCS_GEN_COCOS_PARAMS_QUERY = _worker_dependency("_DOCS_GEN_COCOS_PARAMS_QUERY")
    _documentation_with_registry_form = _worker_dependency(
        "_documentation_with_registry_form"
    )
    _enrich_for_docs_gen = _worker_dependency("_enrich_for_docs_gen")
    _nearby_names_for_docs_gen = _worker_dependency("_nearby_names_for_docs_gen")
    bind_attempt_exposure = _worker_dependency("bind_attempt_exposure")
    charge_billable_exception = _worker_dependency("charge_billable_exception")
    model_provider_exposure = _worker_dependency("model_provider_exposure")
    normalize_description_text = _worker_dependency("normalize_description_text")
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        persist_generated_docs,
        release_generate_docs_failed_claims,
    )
    from imas_codex.standard_names.models import GeneratedDocs

    model = get_model("sn-docs")
    processed = 0

    # ── Enrich batch with DD context (source paths, docs, peers) ──────────
    nearby_existing_names: list[dict] = []
    compose_scored_examples: list[dict] = []
    try:

        def _do_enrich() -> list[dict]:
            with GraphClient() as gc:
                # Load COCOS singleton params for sign guidance rendering
                cocos_params = None
                try:
                    from imas_codex.settings import get_dd_version

                    dd_version = get_dd_version()
                    cocos_rows = list(
                        gc.query(
                            _DOCS_GEN_COCOS_PARAMS_QUERY,
                            ver=dd_version,
                        )
                    )
                    if cocos_rows:
                        cocos_params = cocos_rows[0]["cocos_params"]
                except Exception:
                    logger.debug(
                        "process_generate_docs_batch: COCOS params load failed",
                        exc_info=True,
                    )
                _enrich_for_docs_gen(gc, batch, cocos_params=cocos_params)
                return _nearby_names_for_docs_gen(gc, batch)

        nearby_existing_names = await _asyncio.to_thread(_do_enrich)
    except Exception:
        logger.debug("process_generate_docs_batch: enrichment failed", exc_info=True)

    # ── Load scored examples for docs generation ──────────────────────
    try:
        from imas_codex.standard_names.example_loader import load_compose_examples

        batch_domains = list(
            {
                item.get("physics_domain", "")
                for item in batch
                if item.get("physics_domain")
            }
        )

        def _load_docs_scored_examples() -> list[dict]:
            with GraphClient() as gc:
                return load_compose_examples(
                    gc, physics_domains=batch_domains, axis="docs"
                )

        compose_scored_examples = await _asyncio.to_thread(_load_docs_scored_examples)
        if compose_scored_examples:
            logger.info(
                "generate_docs: Loaded %d scored examples (domains=%s)",
                len(compose_scored_examples),
                batch_domains or "all",
            )
    except Exception:
        logger.debug(
            "process_generate_docs_batch: scored examples load failed",
            exc_info=True,
        )

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        chain_history = item.get("chain_history") or []
        # The docs user template reads ``item.chain_history``. Keep it on the
        # item so refined documentation sees predecessor history.
        item["chain_history"] = chain_history

        # ── Build prompt context ───────────────────────────────────────
        # Merge the cached compose context so the docs SYSTEM prompt's
        # ``_grammar_reference.md`` include renders the closed-vocabulary map
        # (otherwise the grammar block is empty). Cached after first call.
        from imas_codex.standard_names.context import (
            build_compose_context,
            conventions_context_for,
            locus_context_for,
        )

        prompt_context: dict[str, Any] = {
            **build_compose_context(),
            "item": item,
            "chain_history": chain_history,
            "nearby_existing_names": nearby_existing_names,
            "compose_scored_examples": compose_scored_examples,
            # Locus-defining cross-link context: the ISN-registry gloss and
            # position-defining standard quantity for this name's locus, so the
            # docs prompt can link e.g. *_at_pedestal_top ->
            # normalized_poloidal_flux_coordinate_of_pedestal. None when the name
            # has no locus or the locus carries no gloss/defining quantity.
            "locus_context": locus_context_for(sn_id),
            "conventions": conventions_context_for(sn_id),
        }

        try:
            user_prompt = render_prompt("sn/generate_docs_user", prompt_context)
        except Exception:
            logger.debug("generate_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Generate description and documentation for the IMAS standard name: "
                f"{sn_id}\nKind: {item.get('kind', 'scalar')}\n"
                f"Unit: {item.get('unit', '')}"
            )

        try:
            system_prompt = render_prompt("sn/generate_docs_system", prompt_context)
        except Exception:
            logger.debug("generate_docs: system prompt render failed for %s", sn_id)
            system_prompt = None

        _messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": user_prompt}]
        )
        maximum_exposure = model_provider_exposure(
            model,
            _messages,
            response_model=GeneratedDocs,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="generate_docs")
        if lease is None:
            continue

        # ── LLM call ──────────────────────────────────────────────────
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=GeneratedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-docs"),
                before_attempt=bind_attempt_exposure(
                    lease, model, _messages, response_model=GeneratedDocs
                ),
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="generate_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────────
            await _asyncio.to_thread(
                persist_generated_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_description_text(result_obj.description),
                documentation=_documentation_with_registry_form(
                    sn_id, result_obj.documentation, prompt_context["conventions"]
                ),
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            # ── Per-item progress ──────────────────────────────────────
            # Keep log preview short; pass full description to on_event so
            # terminal-aware clipping can use the available display width.
            _desc_log = result_obj.description[:100]
            _worker_dependency("logger").info(
                "\033[32mgenerate_docs\033[0m: %s — %s",
                sn_id,
                _desc_log,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "generate_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as _doc_exc:
            charge_billable_exception(
                lease,
                _doc_exc,
                model=model,
                sn_ids=(sn_id,),
                phase="generate_docs",
            )
            # A "token mismatch or node not found" persist error is an EXPECTED
            # race, not a failure: the orphan sweep reclaimed the item (docs
            # generation can outrun the claim TTL) or another worker took it.
            # The name is not lost — it stays at its prior stage and is
            # re-claimed. Log it quietly; reserve the full traceback for
            # genuinely unexpected errors.
            if "token mismatch or node not found" in str(_doc_exc):
                logger.warning(
                    "generate_docs claim lost for %s (reclaimed mid-generation) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
                logger.exception("generate_docs failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_generate_docs_failed_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug(
                    "release_generate_docs_failed_claims also failed for %s",
                    sn_id,
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed
