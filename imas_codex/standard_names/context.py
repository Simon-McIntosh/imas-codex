"""Rich grammar context for SN compose prompts.

Imports grammar context from imas_standard_names public API
(``get_grammar_context()``) and augments with codex-specific data
(curated examples, tokamak parameter ranges, enum lists).

Caches assembled context in-process (module-level dict).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached context builder
# ---------------------------------------------------------------------------

_CONTEXT_CACHE: dict[str, Any] | None = None


def _get_isn_context() -> dict[str, Any]:
    """Return the ISN grammar context (cached by ISN internally)."""
    from imas_standard_names.grammar.context import get_grammar_context

    return get_grammar_context()


def build_compose_context() -> dict[str, Any]:
    """Build rich context dict for sn/generate_name_system.md template.

    Pulls all grammar, vocabulary, field-guidance, tag, and applicability
    data from ISN's ``get_grammar_context()`` public API, then augments
    with codex-specific data (curated examples, tokamak ranges, enum lists).

    Returns keys needed by both system and user prompts:
    - grammar_rules: canonical pattern, order constraint, template rules
    - vocabulary: per-segment token lists with descriptions
    - segment_descriptions: detailed segment usage guidance
    - field_guidance: per-field content rules and validation
    - examples: curated standard name examples (YAML)
    - tokamak_ranges: machine parameter data for grounding
    - exclusive_pairs: mutually exclusive segment pairs
    - naming_guidance, kind_definitions, anti_patterns, quick_start,
      common_patterns, critical_distinctions, vocabulary_usage_stats,
      base_requirements, type_specific_requirements, documentation_guidance
    - enum lists: subjects, positions, etc. (for user prompt backward compat)
    """
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    # Single call to ISN's public API provides all grammar context
    isn = _get_isn_context()

    ctx: dict[str, Any] = {}

    # Grammar rules (from ISN)
    ctx["canonical_pattern"] = isn["canonical_pattern"]
    ctx["segment_order"] = isn["segment_order"]
    ctx["template_rules"] = isn["template_rules"]
    ctx["exclusive_pairs"] = isn["exclusive_pairs"]

    # Vocabulary with descriptions (from ISN)
    ctx["vocabulary_sections"] = isn["vocabulary_sections"]

    # Segment descriptions and usage guidance (from ISN)
    ctx["segment_descriptions"] = isn["segment_descriptions"]

    # Field guidance for documentation generation (from ISN)
    ctx["field_guidance"] = isn["field_guidance"]

    # Applicability rules (from ISN)
    ctx["applicability"] = isn["applicability"]

    # New ISN-provided keys
    ctx["naming_guidance"] = isn["naming_guidance"]
    ctx["kind_definitions"] = isn["kind_definitions"]
    ctx["anti_patterns"] = isn["anti_patterns"]
    ctx["quick_start"] = isn["quick_start"]
    ctx["common_patterns"] = isn["common_patterns"]
    ctx["critical_distinctions"] = isn["critical_distinctions"]
    ctx["vocabulary_usage_stats"] = isn["vocabulary_usage_stats"]
    ctx["base_requirements"] = isn["base_requirements"]
    ctx["type_specific_requirements"] = isn["type_specific_requirements"]
    ctx["documentation_guidance"] = isn["documentation_guidance"]

    # Codex-specific data (not from ISN)
    ctx["examples"] = _load_curated_examples()
    ctx["tokamak_ranges"] = _load_tokamak_ranges()

    # W2: full closed-vocabulary token map (per-segment) — injected verbatim
    # into the prompt so the LLM never has to guess whether a token is a
    # closed-vocab member.  This is the primary defence against decomposition
    # failures (closed tokens absorbed into physical_base).
    ctx["closed_vocab_full"] = _load_closed_vocab_full()

    # W2: curated examples + anti-patterns from the W0 snapshot YAMLs.  These
    # are static, cacheable, and survive `sn clear`; the graph-driven
    # `compose_scored_examples` injection still complements them at runtime
    # once the graph repopulates.
    ctx["w0_curated_examples"] = _load_w0_curated_examples()
    ctx["decomposition_anti_patterns"] = _load_decomposition_anti_patterns()

    # Physics domain enum (for prompt context — LLM doesn't set it but
    # needs domain awareness for better naming decisions)
    from imas_codex.core.physics_domain import PhysicsDomain

    ctx["physics_domains"] = [e.value for e in PhysicsDomain]

    # NC composition rules (for _nc_rules.md include in system prompt)
    from imas_codex.llm.prompt_loader import load_prompt_config

    try:
        _rules_cfg = load_prompt_config("sn_composition_rules")
        ctx["composition_rules"] = _rules_cfg.get("composition_rules", [])
    except Exception:
        logger.warning(
            "Failed to load sn_composition_rules.yaml; NC rules will be empty"
        )
        ctx["composition_rules"] = []

    # Component-token reuse threshold — surfaced to the prompt so the agent
    # sees the same cosine bar the compose retry loop applies when it flags a
    # candidate-new token as a likely synonym of a registered one.
    from imas_codex.settings import get_sn_dedup_threshold

    ctx["dedup_similarity_threshold"] = get_sn_dedup_threshold()

    # Bare enum lists (backward compat for user prompt)
    ctx.update(_build_enum_lists())

    _CONTEXT_CACHE = ctx
    return ctx


def clear_context_cache() -> None:
    """Clear cached context (for testing)."""
    global _CONTEXT_CACHE
    _CONTEXT_CACHE = None


# ---------------------------------------------------------------------------
# Curated examples
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_curated_examples() -> list[dict[str, Any]]:
    """Load all curated standard name examples from imas_standard_names resources."""
    import imas_standard_names

    pkg_path = Path(imas_standard_names.__path__[0])
    examples_dir = pkg_path / "resources" / "standard_name_examples"

    if not examples_dir.exists():
        logger.warning("No curated examples directory at %s", examples_dir)
        return []

    examples = []
    for yml_path in sorted(examples_dir.rglob("*.yml")):
        try:
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "name" in data:
                # Add the category from directory name
                data["category"] = yml_path.parent.name
                examples.append(data)
        except Exception:
            logger.debug("Failed to load example: %s", yml_path)

    logger.info("Loaded %d curated standard name examples", len(examples))
    return examples


# ---------------------------------------------------------------------------
# W2: Full closed-vocabulary injection
# ---------------------------------------------------------------------------

# Aliased segments in the ISN SEGMENT_TOKEN_MAP that share an identical token
# list — emit only the canonical name to avoid duplicating ~400 tokens in the
# rendered prompt (and to keep the cached system prompt deterministic).
_SEGMENT_ALIASES: dict[str, str] = {
    # alias -> canonical
    "coordinate": "component",
    "object": "device",
    "position": "geometry",
}


@lru_cache(maxsize=1)
def _load_closed_vocab_full() -> list[dict[str, Any]]:
    """Return every closed-vocabulary segment with its FULL token list.

    The returned structure is a list of dicts ordered for stable, cache-friendly
    rendering::

        [
          {"segment": "component", "aliases": ["coordinate"],
           "tokens": ["binormal", "normal", ..., "z"]},
          ...
        ]

    Tokens within a segment are sorted alphabetically.  Segments with
    empty token lists (if any) are omitted.

    The data source is :data:`imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP`
    which is the single source of truth used by the parser, the
    ``is_known_token`` primitive, and the decomposition audit.  When the
    package is unavailable an empty list is returned so prompt rendering
    degrades gracefully rather than raising.
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
    except ImportError:
        logger.warning("imas_standard_names not available — closed_vocab_full empty")
        return []

    # Group aliased segments under their canonical name.
    canonical_to_aliases: dict[str, list[str]] = {}
    for segment in SEGMENT_TOKEN_MAP:
        canonical = _SEGMENT_ALIASES.get(segment, segment)
        if canonical == segment:
            canonical_to_aliases.setdefault(canonical, [])
        else:
            canonical_to_aliases.setdefault(canonical, []).append(segment)

    out: list[dict[str, Any]] = []
    for segment in sorted(canonical_to_aliases):
        tokens = SEGMENT_TOKEN_MAP.get(segment) or ()
        if not tokens:
            continue  # skip segments with no tokens
        out.append(
            {
                "segment": segment,
                "aliases": sorted(canonical_to_aliases[segment]),
                "tokens": sorted(tokens),
            }
        )
    return out


# ---------------------------------------------------------------------------
# W2: W0 snapshot — curated examples + decomposition anti-patterns
# ---------------------------------------------------------------------------


def _w0_examples_path() -> Path:
    return Path(__file__).parent / "examples_curated.yaml"


def _anti_patterns_path() -> Path:
    return Path(__file__).parent / "anti_patterns.yaml"


@lru_cache(maxsize=1)
def _load_w0_curated_examples() -> dict[str, list[dict[str, Any]]]:
    """Load the W0 snapshot ``examples_curated.yaml`` for prompt injection.

    Returns a dict keyed by tier — ``outstanding``, ``good``, ``adequate``,
    ``inadequate``, ``poor`` — each mapping to a list of example entries with
    ``id``, ``description``, ``documentation``, ``reviewer_comments_name``,
    ``grammar_decomposition``, etc.  When the YAML file is absent or
    malformed an empty dict is returned.

    The compose system prompt template selects the strongest entries
    (top of ``outstanding`` and ``good``) for the cacheable
    "EXEMPLAR DECOMPOSITIONS" section.
    """
    path = _w0_examples_path()
    if not path.exists():
        logger.warning("W0 curated examples not found at %s", path)
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return {
            tier: [e for e in (entries or []) if isinstance(e, dict)]
            for tier, entries in data.items()
            if isinstance(entries, list)
        }
    except Exception:
        logger.exception("Failed to load W0 curated examples from %s", path)
        return {}


@lru_cache(maxsize=1)
def _load_decomposition_anti_patterns() -> list[dict[str, Any]]:
    """Load ``anti_patterns.yaml`` — curated decomposition-failure exemplars.

    Each entry contains ``bad_name``, ``issue_category``, ``reviewer_comment``,
    ``absorbed_tokens``, ``correct_decomposition``, and ``rewritten_name``.
    See ``imas_codex/standard_names/anti_patterns.yaml`` for the schema and
    ``tests/standard_names/test_anti_patterns_yaml.py`` for the validator.
    """
    path = _anti_patterns_path()
    if not path.exists():
        logger.warning("Decomposition anti-patterns YAML missing at %s", path)
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            return []
        return [e for e in data if isinstance(e, dict) and e.get("bad_name")]
    except Exception:
        logger.exception("Failed to load decomposition anti-patterns from %s", path)
        return []


# ---------------------------------------------------------------------------
# Tokamak parameters
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_tokamak_ranges() -> dict[str, dict[str, Any]]:
    """Load tokamak machine parameters for documentation grounding."""
    import imas_standard_names

    pkg_path = Path(imas_standard_names.__path__[0])
    params_dir = pkg_path / "resources" / "tokamak_parameters"

    if not params_dir.exists():
        logger.warning("No tokamak parameters directory at %s", params_dir)
        return {}

    machines: dict[str, dict[str, Any]] = {}
    for yml_path in sorted(params_dir.glob("*.yml")):
        if yml_path.name in ("schema.yml", "README.md"):
            continue
        try:
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "machine" in data:
                machines[data["machine"]] = data
        except Exception:
            logger.debug("Failed to load tokamak params: %s", yml_path)

    logger.info("Loaded %d tokamak parameter sets", len(machines))
    return machines


# ---------------------------------------------------------------------------
# Backward-compatible enum lists
# ---------------------------------------------------------------------------


def _build_enum_lists() -> dict[str, list[str]]:
    """Build bare enum lists for user prompt template variables.

    MUST cover every closed grammar segment — reviewers judge tokens
    "unregistered" when a segment is missing here (R3 rotation finding:
    'thermal'/'launched' wrongly docked -4 because the population and
    qualifier registries were absent from the reviewer context).
    """
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )
    from imas_standard_names.grammar.model_types import (
        Aggregation,
        Orbit,
        Population,
    )
    from imas_standard_names.grammar.vocab_loaders import load_qualifiers

    return {
        "subjects": [e.value for e in Subject],
        "positions": [e.value for e in Position],
        "components": [e.value for e in Component],
        "coordinates": [e.value for e in Component],  # same enum
        "processes": [e.value for e in Process],
        "transformations": [e.value for e in Transformation],
        "geometric_bases": [e.value for e in GeometricBase],
        "objects": [e.value for e in Object],
        "binary_operators": [e.value for e in BinaryOperator],
        "populations": [e.value for e in Population],
        "orbits": [e.value for e in Orbit],
        "aggregations": [e.value for e in Aggregation],
        "qualifiers": sorted(load_qualifiers()),
    }


def build_domain_vocabulary_preseed(domain: str | None) -> str:
    """Build a vocabulary pre-seed section for compose prompts.

    Queries the graph for all StandardName nodes in *domain* that are
    live (``name_stage`` not superseded/exhausted) AND
    ``validation_status = 'valid'``, returning up to 40 canonical
    ``(name, description-first-sentence)`` pairs.

    Ordered by name_stage priority (accepted > reviewed > refining > drafted),
    then alphabetically by name.

    Returns empty string when *domain* is None or no names are found.
    """
    if not domain:
        return ""

    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.physics_domain = $domain
                  AND NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted']
                  AND sn.validation_status = 'valid'
                RETURN sn.id AS name,
                       sn.description AS description,
                       sn.name_stage AS name_stage
                ORDER BY
                    CASE sn.name_stage
                        WHEN 'accepted' THEN 0
                        WHEN 'reviewed' THEN 1
                        WHEN 'refining' THEN 2
                        WHEN 'drafted' THEN 3
                    END,
                    sn.id
                LIMIT 40
                """,
                domain=domain,
            )
            if not rows:
                return ""

            lines = []
            for row in rows:
                name = row.get("name", "")
                desc = row.get("description", "")
                # Take first sentence only
                first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""
                lines.append(f"- `{name}`: {first_sentence}")

            return "\n".join(lines)
    except Exception:
        logger.debug("Domain vocabulary preseed unavailable", exc_info=True)
        return ""


def render_cocos_guidance(label: str, cocos_params: dict) -> str:
    """Render sign guidance for a transformation label using COCOS node properties.

    Args:
        label: COCOS transformation label (e.g., 'psi_like')
        cocos_params: Properties dict from the COCOS graph node
            (sigma_bp, psi_increasing_outward, phi_increasing_ccw, etc.)

    Returns:
        Rendered guidance string for the LLM prompt.
    """
    from imas_codex.llm.prompt_loader import load_prompt_config

    config = load_prompt_config("cocos_sign_guidance")
    label_config = config.get("labels", {}).get(label)
    if not label_config:
        return config.get("generic_fallback", "")

    guidance = label_config["guidance"]

    # Substitute raw Sauter parameters directly
    for param in ("sigma_bp", "sigma_r_phi_z", "sigma_rho_theta_phi", "e_bp"):
        guidance = guidance.replace(f"{{{param}}}", str(cocos_params.get(param, "?")))

    # Resolve template variables from COCOS node boolean/sign properties
    for var_name, var_spec in label_config.get("variables", {}).items():
        source_prop = var_spec["from"]
        source_val = cocos_params.get(source_prop)
        # Normalize lookup key: booleans → "true"/"false", numbers → str
        if isinstance(source_val, bool):
            lookup_key = str(source_val).lower()
        else:
            lookup_key = str(source_val)
        replacement = var_spec.get(lookup_key, f"[unknown {source_prop}]")
        guidance = guidance.replace(f"{{{var_name}}}", replacement)

    return guidance


# ---------------------------------------------------------------------------
# Reviewer neighbourhood (third-party-critic context)
# ---------------------------------------------------------------------------


def _path_ids_prefix(path: str) -> str | None:
    """Extract the leading IDS segment from a DD path.

    ``equilibrium/time_slice/0/global_quantities/ip`` -> ``equilibrium``.
    Returns ``None`` for empty input.
    """
    if not path:
        return None
    head = path.split("/", 1)[0]
    return head or None


def fetch_review_neighbours(
    sn: dict[str, Any],
    *,
    gc: Any = None,
    n_vector: int = 5,
    n_same_base: int = 3,
    n_same_path: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch nearest-neighbour SNs to inject into reviewer prompts.

    Returns three lists used as third-party comparators by the reviewer:

    * ``vector_neighbours`` — up to ``n_vector`` accepted SNs nearest to the
      candidate description by embedding similarity (vector index lookup).
    * ``same_base_neighbours`` — up to ``n_same_base`` accepted SNs sharing
      the candidate's ``physical_base`` token (sibling-by-base comparator).
    * ``same_path_neighbours`` — up to ``n_same_path`` accepted SNs whose
      ``source_paths`` share the candidate's leading IDS prefix.
    * ``sibling_family`` — the candidate's HAS_PARENT sibling family
      (:func:`fetch_sibling_family`) for the parallel-structure check;
      ``None`` when the candidate has no qualifying family.

    All result lists exclude the candidate itself by ``id``. On any failure
    (no graph client, missing index, etc.) the corresponding list is empty
    and the function logs at DEBUG level — never raises.

    Each entry dict contains:
        ``id, name, description, kind, unit, score`` (score only for vector).

    The candidate ``sn`` dict must contain at least ``id``; ``description``
    drives the vector lookup, ``physical_base`` the same-base lookup, and
    ``source_paths`` the same-path lookup.
    """
    sn_id = sn.get("id") or sn.get("name") or ""
    desc = sn.get("description") or sn.get("name") or sn.get("id") or ""
    physical_base = (
        sn.get("physical_base")
        or sn.get("base_token")
        or (sn.get("grammar_segments") or {}).get("physical_base")
    )
    source_paths = sn.get("source_paths") or []
    ids_prefix = next(
        (p for p in (_path_ids_prefix(sp) for sp in source_paths) if p), None
    )

    out: dict[str, Any] = {
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
        "sibling_family": None,
    }

    own_gc = False
    _gc_ctx: Any = None
    if gc is None:
        try:
            from imas_codex.graph.client import GraphClient

            _gc_ctx = GraphClient()
            gc = _gc_ctx.__enter__() if hasattr(_gc_ctx, "__enter__") else _gc_ctx
            own_gc = True
        except Exception:
            logger.debug("fetch_review_neighbours: GraphClient unavailable")
            return out

    try:
        # --- Vector nearest --------------------------------------------------
        if desc:
            try:
                from imas_codex.standard_names.search import (
                    search_standard_names_vector,
                )

                rows = search_standard_names_vector(
                    desc, k=n_vector + 1, gc=gc, include_superseded=False
                )
                out["vector_neighbours"] = [r for r in rows if r.get("id") != sn_id][
                    :n_vector
                ]
            except Exception:
                logger.debug("fetch_review_neighbours: vector lookup failed")

        # --- Same physical_base ---------------------------------------------
        if physical_base:
            try:
                rows = (
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WHERE sn.physical_base = $base
                          AND sn.id <> $sn_id
                          AND coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') = 'accepted'
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id,
                               sn.name AS name,
                               sn.description AS description,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit
                        ORDER BY sn.id
                        LIMIT $k
                        """,
                        base=physical_base,
                        sn_id=sn_id,
                        k=n_same_base,
                    )
                    or []
                )
                out["same_base_neighbours"] = [dict(r) for r in rows]
            except Exception:
                logger.debug("fetch_review_neighbours: same-base lookup failed")

        # --- Same DD IDS prefix ---------------------------------------------
        if ids_prefix:
            try:
                rows = (
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WHERE sn.id <> $sn_id
                          AND coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') = 'accepted'
                          AND ANY(p IN sn.source_paths WHERE p STARTS WITH $prefix)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id,
                               sn.name AS name,
                               sn.description AS description,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit
                        ORDER BY sn.id
                        LIMIT $k
                        """,
                        sn_id=sn_id,
                        prefix=ids_prefix + "/",
                        k=n_same_path,
                    )
                    or []
                )
                out["same_path_neighbours"] = [dict(r) for r in rows]
            except Exception:
                logger.debug("fetch_review_neighbours: same-path lookup failed")

        # --- HAS_PARENT sibling family (parallel-structure comparator) -------
        if sn_id:
            try:
                out["sibling_family"] = fetch_sibling_family(sn_id, gc=gc)
            except Exception:
                logger.debug("fetch_review_neighbours: sibling-family lookup failed")

        return out
    finally:
        if own_gc and _gc_ctx is not None:
            try:
                if hasattr(_gc_ctx, "__exit__"):
                    _gc_ctx.__exit__(None, None, None)
                else:
                    gc.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sibling-family (parallel-structure) context
# ---------------------------------------------------------------------------

_SIBLING_FAMILY_QUERY = """
MATCH (sn:StandardName {id: $sn_id})-[r:HAS_PARENT]->(p:StandardName)
WHERE r.operator_kind IN ['projection', 'qualifier', 'coordinate', 'locus']
OPTIONAL MATCH (sib:StandardName)-[rs:HAS_PARENT]->(p)
WHERE sib.id <> $sn_id
  AND rs.operator_kind IN ['projection', 'qualifier', 'coordinate', 'locus']
  AND coalesce(sib.name_stage, '') <> 'superseded'
WITH p, r,
     [s IN collect({
         id: sib.id,
         description: sib.description,
         documentation: sib.documentation,
         docs_stage: coalesce(sib.docs_stage, ''),
         reviewer_score_docs: sib.reviewer_score_docs,
         operator_kind: rs.operator_kind,
         axis: rs.axis
     }) WHERE s.id IS NOT NULL] AS sibs
RETURN p.id AS parent_id,
       p.description AS parent_description,
       p.documentation AS parent_documentation,
       coalesce(p.docs_stage, '') AS parent_docs_stage,
       r.operator_kind AS member_operator_kind,
       sibs
ORDER BY size(sibs) DESC, parent_id
LIMIT 1
"""


def _doc_opening(text: str | None, limit: int = 240) -> str:
    """First ``limit`` characters of *text*, cut at a word boundary."""
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    head, _, _ = cut.rpartition(" ")
    return (head or cut) + " …"


def fetch_sibling_family(
    sn_id: str,
    *,
    gc: Any = None,
    max_siblings: int = 12,
) -> dict[str, Any] | None:
    """Return the HAS_PARENT sibling family of *sn_id* for prompt grounding.

    The family is the set of live (non-superseded) StandardNames sharing a
    HAS_PARENT parent with the candidate via a structural operator edge
    (``operator_kind`` in projection / qualifier / coordinate). When the
    candidate has several such parents the largest family wins.

    Returns ``None`` when the name has no qualifying family (or on any
    graph failure — never raises). Otherwise a dict:

    - ``parent``: ``{name, description, docs_accepted}``
    - ``anchor``: the documentation-template authority — the parent when its
      docs are accepted and its description is real, else the docs-accepted
      sibling with the highest ``reviewer_score_docs``; ``None`` when the
      family has no accepted member yet (defer semantics — siblings are
      still returned for structural comparison).
    - ``siblings``: up to *max_siblings* entries of ``{name, operator_kind,
      axis, description, documentation_opening, docs_stage}``.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    own_gc = False
    if gc is None:
        try:
            from imas_codex.graph.client import GraphClient

            gc = GraphClient()
            own_gc = True
        except Exception:
            logger.debug("fetch_sibling_family: GraphClient unavailable")
            return None

    try:
        rows = list(gc.query(_SIBLING_FAMILY_QUERY, sn_id=sn_id) or [])
    except Exception:
        logger.debug("fetch_sibling_family: query failed for %s", sn_id)
        rows = []
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass

    if not rows:
        return None
    row = rows[0]
    raw_sibs = [s for s in (row.get("sibs") or []) if s.get("id")]
    if not raw_sibs:
        return None

    placeholder = DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER

    def _real(text: str | None) -> str:
        t = (text or "").strip()
        return "" if t == placeholder else t

    parent_desc = _real(row.get("parent_description"))
    parent_docs_accepted = row.get("parent_docs_stage") == "accepted" and bool(
        parent_desc
    )

    siblings = [
        {
            "name": s["id"],
            "operator_kind": s.get("operator_kind") or "",
            "axis": s.get("axis") or "",
            "description": _real(s.get("description")),
            "documentation_opening": _doc_opening(s.get("documentation")),
            "docs_stage": s.get("docs_stage") or "",
        }
        for s in sorted(raw_sibs, key=lambda s: (s.get("operator_kind") or "", s["id"]))
    ][:max_siblings]

    anchor: dict[str, Any] | None = None
    if parent_docs_accepted:
        anchor = {
            "name": row["parent_id"],
            "description": parent_desc,
            "documentation": _doc_opening(row.get("parent_documentation"), 2000),
            "is_parent": True,
        }
    else:
        accepted = [
            s
            for s in raw_sibs
            if s.get("docs_stage") == "accepted" and _real(s.get("description"))
        ]
        if accepted:
            best = max(
                accepted,
                key=lambda s: (
                    s.get("reviewer_score_docs") or 0.0,
                    len(_real(s.get("description"))),
                ),
            )
            anchor = {
                "name": best["id"],
                "description": _real(best.get("description")),
                "documentation": _doc_opening(best.get("documentation"), 2000),
                "is_parent": False,
            }

    return {
        "parent": {
            "name": row["parent_id"],
            "description": parent_desc,
            "docs_accepted": parent_docs_accepted,
        },
        "anchor": anchor,
        "siblings": siblings,
    }


def locus_context_for(sn_id: str) -> dict[str, Any] | None:
    """Resolve the ISN locus-registry context for a name's evaluation locus.

    Returns ``{token, description, defining_quantity}`` for the locus of
    *sn_id* when the installed ISN registry supplies a DD-anchored gloss or a
    position-defining standard quantity; ``None`` otherwise. Injected into the
    docs prompt so the locus-defining cross-link rule (PR-9) can link a name to
    the standard quantity that fixes its locus (e.g. ``*_at_pedestal_top`` ->
    ``normalized_poloidal_flux_coordinate_of_pedestal``) — data-driven: the
    mapping lives in the ISN vocab, the rule lives in the prompt.
    """
    try:
        from imas_standard_names.grammar.parser import parse as _isn_parse
        from imas_standard_names.grammar.vocab_loaders import (
            load_locus_registry as _load_locus_registry,
        )

        ir = getattr(_isn_parse(sn_id), "ir", None)
        if ir is None or getattr(ir, "locus", None) is None:
            return None
        token = ir.locus.token
        entry = _load_locus_registry().loci.get(token)
        if entry is None or not (entry.description or entry.defining_quantity):
            return None
        return {
            "token": token,
            "description": entry.description or "",
            "defining_quantity": entry.defining_quantity or "",
        }
    except Exception:
        logger.debug("locus_context_for: failed for %s", sn_id, exc_info=True)
        return None
