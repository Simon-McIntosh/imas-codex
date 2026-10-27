"""Documentation worker tasks and their explicit entrypoint collaborators."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


def _canonical_docs_review_item(
    _worker_dependency, /, item: dict[str, Any]
) -> dict[str, Any]:
    """Copy the prompt-bearing fields while excluding claim/lifecycle state."""
    _DOCS_REVIEW_VOLATILE_ITEM_FIELDS = _worker_dependency(
        "_DOCS_REVIEW_VOLATILE_ITEM_FIELDS"
    )
    import copy

    return {
        key: copy.deepcopy(value)
        for key, value in item.items()
        if key not in _DOCS_REVIEW_VOLATILE_ITEM_FIELDS
    }


def _bound_docs_review_dd_paths(
    _worker_dependency, /, item: dict[str, Any]
) -> list[str]:
    """List every DD path bound to the name, preserving source order.

    Only DD paths are returned: the paths of ``dd`` source bindings, the
    ``dd_source_paths`` a prepared review item carries, or ``source_paths``
    entries carrying the ``dd:`` prefix. A facility signal path names no DD
    text, so it never becomes a required DD path. ``dd_source_paths`` is this
    function's own output for the item, kept beside the ``dd:``-stripped
    ``source_paths`` a prepared review item is normalised to, so the DD
    provenance survives that normalisation.
    """
    DD_PREFIX = _worker_dependency("DD_PREFIX")
    strip_dd_prefix = _worker_dependency("strip_dd_prefix")
    bindings = item.get("source_bindings") or []
    paths = [
        binding.get("dd_path") or binding.get("source_id") or binding.get("id")
        for binding in bindings
        if binding and binding.get("source_type") == "dd"
    ]
    if not paths:
        paths = item.get("dd_source_paths")
        if paths is None:
            paths = item.get("source_paths") or []
            if isinstance(paths, str):
                try:
                    paths = json.loads(paths)
                except (json.JSONDecodeError, TypeError):
                    paths = [paths]
            if isinstance(paths, str):
                paths = [paths]
            paths = [
                path
                for path in paths
                if isinstance(path, str) and path.startswith(DD_PREFIX)
            ]
    if isinstance(paths, str):
        paths = [paths]
    return list(
        dict.fromkeys(
            strip_dd_prefix(path) for path in paths if isinstance(path, str) and path
        )
    )


def _set_docs_review_dd_source_paths(
    _worker_dependency, /, item: dict[str, Any]
) -> list[str]:
    """Normalise an item's DD paths, keeping the DD provenance beside them.

    ``source_paths`` is set to the bare DD paths so the DD-source read and the
    review prompt see paths without the ``dd:`` prefix. ``dd_source_paths``
    records the same list, so a later :func:`_bound_docs_review_dd_paths` call
    on the normalised item still recognises the paths as DD rather than as
    unlabelled ``source_paths`` entries. A facility signal path is dropped:
    it names no DD text.
    """
    _bound_docs_review_dd_paths = _worker_dependency("_bound_docs_review_dd_paths")
    dd_paths = _bound_docs_review_dd_paths(item)
    item["dd_source_paths"] = dd_paths
    item["source_paths"] = dd_paths
    return dd_paths


def _has_dd_source_reference(_worker_dependency, /, item: dict[str, Any]) -> bool:
    """Report whether an item names DD source text to compare claims against.

    A DD path reaches the item either as a ``dd`` source binding or as a
    ``dd:``-prefixed entry, which a binding of another source type can also
    carry. Items whose sources are all derived or manual name no DD text, so
    there is nothing for a claim comparison to read.
    """
    DD_PREFIX = _worker_dependency("DD_PREFIX")
    _bound_docs_review_dd_paths = _worker_dependency("_bound_docs_review_dd_paths")
    if _bound_docs_review_dd_paths(item):
        return True
    paths = item.get("source_paths")
    entries: list[Any] = list(paths) if isinstance(paths, (list, tuple)) else []
    bindings = item.get("source_bindings")
    if isinstance(bindings, (list, tuple)):
        entries.extend(bindings)
    for entry in entries:
        if isinstance(entry, str):
            if entry.startswith(DD_PREFIX):
                return True
            continue
        if isinstance(entry, dict) and any(
            isinstance(entry.get(key), str) and entry[key].startswith(DD_PREFIX)
            for key in ("dd_path", "source_id", "path", "id")
        ):
            return True
    return False


def _load_docs_review_source_item(_worker_dependency, /, sn_id: str) -> dict[str, Any]:
    """Read the same prompt-bearing projection returned by a docs claim."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (sn)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            RETURN sn.id AS id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   c.id AS cluster_id,
                   sn.physics_domain AS physics_domain,
                   sn.validation_status AS validation_status,
                   sn.tags AS tags,
                   coalesce(sn.docs_chain_length, 0) AS docs_chain_length,
                   sn.docs_stage AS docs_stage,
                   sn.edit_mode AS edit_mode,
                   sn.name_hint AS name_hint,
                   sn.docs_hint AS docs_hint,
                   sn.edit_reason AS edit_reason,
                   sn.edit_origin AS edit_origin,
                   sn.origin AS origin,
                   sn.run_id AS run_id,
                   sn.source_paths AS source_paths,
                   [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) | {
                       id: source.id,
                       source_type: source.source_type,
                       source_id: source.source_id,
                       dd_path: source.dd_path,
                       dd_version: source.dd_version
                   }] AS source_bindings
            """,
            id=sn_id,
        )
    if len(rows) != 1:
        raise ValueError(f"documentation review target {sn_id!r} is not unique")
    return dict(rows[0])


def _load_docs_review_parent_children(
    _worker_dependency, /, item: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read the canonical live children for one derived parent candidate."""
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER = _worker_dependency(
        "DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER"
    )
    if item.get("origin") != "derived":
        return []
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (p:StandardName {id: $id, origin: 'derived'})
            MATCH (c:StandardName)-[:HAS_PARENT]->(p)
            WHERE NOT (coalesce(c.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested'])
            WITH c ORDER BY c.id
            RETURN collect({
                name: c.id,
                description: CASE WHEN c.description = $placeholder
                    THEN null ELSE c.description END,
                unit: c.unit,
                physics_domain: c.physics_domain
            })[..12] AS children
            """,
            id=item["id"],
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    return list(rows[0].get("children") or []) if rows else []


def _load_docs_review_examples(
    _worker_dependency, /, item: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read the scored documentation examples used by the reviewer prompt."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_review_examples

    domain = item.get("physics_domain") or ""
    with GraphClient() as gc:
        return list(load_review_examples(gc, physics_domains=[domain], axis="docs"))


def _enrich_docs_review_dd_context(_worker_dependency, /, item: dict[str, Any]) -> None:
    """Attach the same first-source DD context used at worker dispatch.

    A name with no bound source has no DD path to read, which is a valid
    state rather than an error: review proceeds on the name, description, and
    documentation alone. Reaching for a first element that is not there would
    abort preparation and fail the whole quorum for that name.
    """
    _enrich_dd_path_context = _worker_dependency("_enrich_dd_path_context")
    strip_dd_prefix = _worker_dependency("strip_dd_prefix")
    source_paths = item.get("source_paths") or []
    if isinstance(source_paths, list):
        first_source = source_paths[0] if source_paths else None
    else:
        first_source = source_paths
    if isinstance(first_source, str):
        first_source = strip_dd_prefix(first_source)
    if not first_source:
        return
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        _enrich_dd_path_context(gc, item, first_source)


def _docs_review_request_identity(
    _worker_dependency, /, payload: dict[str, Any]
) -> str:
    """Hash a canonical rendered request and its priced reviewer sequence."""
    import hashlib

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _bounded_docs_escalation_factory(
    _worker_dependency,
    /,
    *,
    base_user_prompt: str,
    system_prompt: str,
    response_model: type[Any],
    input_token_bound: int,
) -> Callable[[list[dict[str, Any]]], str]:
    """Render the canonical critique request within a proven input bound.

    The canonical budget infrastructure proves input tokens are no greater
    than the complete serialized UTF-8 byte length plus chat framing. This
    factory binary-searches the critique prefix against that same complete
    request serialization, so the actual escalator request can never exceed
    the cataloged model input limit used for admission pricing.
    """
    if input_token_bound <= 0:
        raise ValueError("escalator input token bound must be positive")
    from imas_codex.standard_names.budget import _CHAT_FRAMING_TOKEN_ALLOWANCE

    serialized_byte_bound = input_token_bound - _CHAT_FRAMING_TOKEN_ALLOWANCE
    if serialized_byte_bound <= 0:
        raise ValueError("escalator input allowance cannot cover chat framing")
    marker = "\n\n## Prior reviewer critiques for authoritative escalation\n"
    schema = response_model.model_json_schema()

    def _serialized_size(user_prompt: str) -> int:
        return len(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_schema": schema,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def _render(prior_reviews: list[dict[str, Any]]) -> str:
        critique = json.dumps(
            prior_reviews,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        prefix = base_user_prompt + marker
        if _serialized_size(prefix) > serialized_byte_bound:
            raise ValueError("base escalation request exceeds catalog input bound")
        low, high = 0, len(critique)
        while low < high:
            midpoint = (low + high + 1) // 2
            if _serialized_size(prefix + critique[:midpoint]) <= serialized_byte_bound:
                low = midpoint
            else:
                high = midpoint - 1
        return prefix + critique[:low]

    return _render


async def prepare_docs_review_request(
    _worker_dependency, /, item: dict[str, Any]
) -> dict[str, Any]:
    """Render and price one canonical docs-review request without mutation.

    This is the sole docs-review prompt construction path. Admission and the
    claimed worker both call it, so context, templates, reviewer models,
    response schema, effort settings, possible escalator exposure, and request
    identity cannot drift through duplicated preparation code.
    """
    _bounded_docs_escalation_factory = _worker_dependency(
        "_bounded_docs_escalation_factory"
    )
    _canonical_docs_review_item = _worker_dependency("_canonical_docs_review_item")
    _docs_review_request_identity = _worker_dependency("_docs_review_request_identity")
    _enrich_docs_review_dd_context = _worker_dependency(
        "_enrich_docs_review_dd_context"
    )
    _load_docs_review_examples = _worker_dependency("_load_docs_review_examples")
    _load_docs_review_parent_children = _worker_dependency(
        "_load_docs_review_parent_children"
    )
    _load_docs_review_source_item = _worker_dependency("_load_docs_review_source_item")
    _set_docs_review_dd_source_paths = _worker_dependency(
        "_set_docs_review_dd_source_paths"
    )
    model_provider_exposure = _worker_dependency("model_provider_exposure")
    import asyncio as _asyncio

    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_model,
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
    )
    from imas_codex.standard_names.context import (
        conventions_context_for,
        fetch_review_neighbours,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewDocsBatch,
        StandardNameQualityReviewDocsParentBatch,
    )

    source_item = item
    if item.get("docs_stage") in {"accepted", "reviewed", "exhausted"}:
        source_item = await _asyncio.to_thread(
            _load_docs_review_source_item, str(item.get("id") or "")
        )
    prepared_item = _canonical_docs_review_item(source_item)
    try:
        review_models = list(get_sn_review_docs_models())
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        try:
            review_models = list(get_sn_review_names_models())
        except (ValueError, IndexError):
            review_models = []
    if not review_models:
        review_models = [get_model("sn-refine")]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    try:
        children = await _asyncio.to_thread(
            _load_docs_review_parent_children, prepared_item
        )
    except Exception:
        logger.debug("review_docs: parent children fetch failed", exc_info=True)
        children = []
    if children:
        prepared_item["derived_children"] = children

    try:
        neighbours = await _asyncio.to_thread(fetch_review_neighbours, prepared_item)
    except Exception:
        logger.debug(
            "review_docs: neighbour fetch failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )
        neighbours = {
            "vector_neighbours": [],
            "same_base_neighbours": [],
            "same_path_neighbours": [],
        }
    try:
        review_examples = await _asyncio.to_thread(
            _load_docs_review_examples, prepared_item
        )
    except Exception:
        logger.debug(
            "review_docs: scored-example load failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )
        review_examples = []
    try:
        await _asyncio.to_thread(_enrich_docs_review_dd_context, prepared_item)
    except Exception:
        logger.debug(
            "review_docs: dd_path_context failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )

    prompt_context: dict[str, Any] = {
        "item": prepared_item,
        **neighbours,
        "review_scored_examples": review_examples,
    }
    prompt_context["conventions"] = conventions_context_for(prepared_item["id"])
    is_parent = bool(prepared_item.get("derived_children"))
    user_template = "sn/review_docs_parent_user" if is_parent else "sn/review_docs_user"
    system_template = (
        "sn/review_docs_parent_system" if is_parent else "sn/review_docs_system"
    )
    user_prompt = render_prompt(user_template, prompt_context)
    system_prompt = render_prompt(system_template, prompt_context)
    if is_parent and prompt_context["conventions"]:
        user_prompt += "\n\n" + render_prompt("sn/_conventions", prompt_context)
    response_model = (
        StandardNameQualityReviewDocsParentBatch
        if is_parent
        else StandardNameQualityReviewDocsBatch
    )
    rubric_dims = (
        ("generalization", "positioning", "physics_accuracy", "clarity")
        if is_parent
        else (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        )
    )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    mandatory_exposures = [
        model_provider_exposure(
            model,
            base_messages,
            response_model=response_model,
            provider_attempts=1,
        )
        for model in review_models[:2]
    ]
    conditional_exposure = 0.0
    escalation_input_token_bound = None
    escalation_prompt_factory = None
    if len(review_models) >= 3:
        from imas_codex.discovery.base.llm import get_catalog_model_info

        catalog_bound = get_catalog_model_info(review_models[2]).get("max_input_tokens")
        if not isinstance(catalog_bound, int) or catalog_bound <= 0:
            raise ValueError("escalator catalog input allowance is not bounded")
        escalation_input_token_bound = catalog_bound
        escalation_prompt_factory = _bounded_docs_escalation_factory(
            base_user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_model=response_model,
            input_token_bound=catalog_bound,
        )
        worst_case_prompt = escalation_prompt_factory(
            [{"critique": "x" * catalog_bound}]
        )
        escalation_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": worst_case_prompt},
        ]
        conditional_exposure = model_provider_exposure(
            review_models[2],
            escalation_messages,
            response_model=response_model,
            provider_attempts=1,
        )
    exposures = [*mandatory_exposures]
    if conditional_exposure > 0:
        exposures.append(conditional_exposure)

    reasoning_effort = get_sn_review_reasoning_effort()
    escalation_reasoning_effort = get_sn_review_escalation_reasoning_effort()
    identity_payload = {
        "id": prepared_item["id"],
        "models": review_models,
        "messages": base_messages,
        "response_schema": response_model.model_json_schema(),
        "rubric_dims": list(rubric_dims),
        "disagreement_threshold": disagreement_threshold,
        "reasoning_effort": reasoning_effort,
        "escalation_reasoning_effort": escalation_reasoning_effort,
        "expected_exposures": exposures,
        "mandatory_exposures": mandatory_exposures,
        "conditional_exposure": conditional_exposure,
        "escalation_input_token_bound": escalation_input_token_bound,
    }
    veto_item = _canonical_docs_review_item(prepared_item)
    _set_docs_review_dd_source_paths(veto_item)
    if veto_item["source_paths"]:
        from imas_codex.standard_names.review.pipeline import _fetch_review_dd_context

        try:
            await _asyncio.to_thread(_fetch_review_dd_context, [veto_item])
        except Exception:
            logger.warning(
                "review_docs: DD source batch read failed for %s",
                veto_item["id"],
                exc_info=True,
            )
    return {
        **identity_payload,
        "veto_item": veto_item,
        "request_identity": _docs_review_request_identity(identity_payload),
        "expected_exposure": sum(exposures),
        "provider_policy_ceiling_is_separate": True,
        "response_model": response_model,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "escalation_prompt_factory": escalation_prompt_factory,
    }


async def _release_docs_review_claim(
    _worker_dependency, /, item: dict[str, Any]
) -> None:
    """Return one unreviewed documentation claim to the drafted pool."""
    import asyncio as _asyncio

    from imas_codex.standard_names.graph_store.graph_claim_release import (
        release_review_docs_failed_claims,
    )

    claim_token = item.get("claim_token") or ""
    if not claim_token:
        return
    await _asyncio.to_thread(
        release_review_docs_failed_claims,
        sn_ids=[item["id"]],
        claim_token=claim_token,
        from_stage="drafted",
        to_stage="drafted",
    )


def _documentation_conformance_note(
    _worker_dependency, /, item: dict[str, Any]
) -> str | None:
    """Name the registry findings that bar a reviewed document from acceptance.

    A review score judges the prose; it does not check the opening against the
    selected template, a forbidden equation variant, or an equation for a
    registered concept written other than in its canonical form. Those are
    decided here, on the same findings the audit report carries. A canonical
    equation is the converged statement and is accepted. A name the grammar cannot parse
    selects no registry entry and so carries no finding.
    """
    from imas_standard_names.grammar.parser import ParseError

    from imas_codex.standard_names.docs_gates import (
        HouseStyleFindingClass,
        conformance_findings,
    )

    barring = {
        HouseStyleFindingClass.TEMPLATE_OPENING,
        HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT,
        HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS,
        HouseStyleFindingClass.LOCUS_DEFINITION,
        HouseStyleFindingClass.NON_CANONICAL_EQUATION,
    }
    try:
        findings = conformance_findings(
            item["id"],
            description=item.get("description") or "",
            documentation=item.get("documentation") or "",
        )
    except ParseError:
        logger.debug("review_docs: no registry selection for %s", item["id"])
        return None
    details = [
        f"{finding.finding_class.value}: {finding.detail} [{finding.registry_entry}]"
        for finding in findings
        if finding.finding_class in barring
    ]
    if not details:
        return None
    return (
        "documentation does not conform to the conventions registry — "
        "correct each finding: " + "; ".join(details)
    )


async def _documentation_banned_prose_note(
    _worker_dependency, /, item: dict[str, Any], mgr: BudgetManager
) -> str | None:
    """Hold a policy candidate only when prose adjudication confirms it."""
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.prose_adjudicator import _adjudicate_one
    from imas_codex.standard_names.prose_policy import (
        BANNED_PROSE_PATTERNS,
        banned_prose_findings,
    )

    text = f"{item.get('description') or ''}\n{item.get('documentation') or ''}"
    findings = banned_prose_findings(text)
    if not any(findings.values()):
        return None
    try:
        genuine = await _adjudicate_one(
            get_model("sn-prose-adjudicator"),
            item["id"],
            text,
            findings,
            reasoning_effort=get_reasoning_effort("sn-prose-adjudicator", "low"),
            service="standard-names",
            budget_manager=mgr,
        )
    except Exception:
        logger.warning(
            "review_docs: prose adjudication failed for %s; holding candidate",
            item["id"],
            exc_info=True,
        )
        genuine = True
    if not genuine:
        return None
    for category, patterns in BANNED_PROSE_PATTERNS.items():
        for pattern in patterns:
            if match := pattern.search(text):
                return f"banned prose ({category}): {match.group(0)!r}"
    return None


async def _documentation_contradiction_note(
    _worker_dependency, /, item: dict[str, Any], mgr: BudgetManager
) -> str | None:
    """Compare authored claims with bound DD text before accepting them."""
    _bound_docs_review_dd_paths = _worker_dependency("_bound_docs_review_dd_paths")
    _has_dd_source_reference = _worker_dependency("_has_dd_source_reference")
    strip_dd_prefix = _worker_dependency("strip_dd_prefix")
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.review import jev_review

    required_paths = _bound_docs_review_dd_paths(item)
    sources = item.get("dd_source_docs") or []
    if not sources and not required_paths and item.get("dd_documentation"):
        sources = [{"documentation": item["dd_documentation"]}]
    present_paths = {
        strip_dd_prefix(str(source.get("id") or source.get("path") or ""))
        for source in sources
    }
    missing_paths = [path for path in required_paths if path not in present_paths]
    if missing_paths:
        logger.warning(
            "review_docs: DD source text missing for %s at %s",
            item["id"],
            ", ".join(missing_paths),
        )
        return "DD source text unavailable at " + ", ".join(missing_paths)
    empty_paths = [
        str(source.get("id") or source.get("path") or "DD source")
        for source in sources
        if not (source.get("documentation") or source.get("description"))
    ]
    if empty_paths:
        logger.warning(
            "review_docs: DD source text empty for %s at %s",
            item["id"],
            ", ".join(empty_paths),
        )
        return "DD source text unavailable at " + ", ".join(empty_paths)
    if not sources:
        if _has_dd_source_reference(item):
            logger.warning("review_docs: DD source text unavailable for %s", item["id"])
            return "DD source text unavailable; claim comparison could not run"
        return None

    states = [
        jev_review.build_docs_state({**item, "dd_source_docs": [source]})
        for source in sources
    ]
    claims = states[0]["documentation_claims"]
    if not claims:
        return None
    questions = {
        key: question
        for key, question in jev_review.docs_question_table(
            jev_review._default_checks(), claims
        ).items()
        if key.startswith("check:claim_contradicts_source:")
    }
    lease = mgr.reserve(0.001 * len(sources), phase="review_docs")
    if lease is None:
        logger.warning("review_docs: claim comparison unfunded for %s", item["id"])
        return "claim comparison unavailable: run budget refused the call"
    try:
        model = get_model("sn-judge")
        results, _ = await jev_review.decide_batch(
            states, questions, model=model, service=jev_review.SERVICE
        )
        if len(results) != len(sources):
            logger.warning("review_docs: claim result count differs for %s", item["id"])
            return "claim comparison unavailable: Jev omitted a DD source"
        contradictions = []
        unavailable = []
        for source, state, result in zip(sources, states, results, strict=True):
            source_path = str(source.get("id") or source.get("path") or "DD source")
            if result is None:
                unavailable.append(source_path)
                continue
            answers, spent = result
            lease.charge_event(
                spent,
                LLMCostEvent(
                    model=model,
                    tokens_in=0,
                    tokens_out=0,
                    sn_ids=(item["id"],),
                    phase="review_docs",
                ),
            )
            verdict = jev_review.aggregate_docs_review(
                state, jev_review._default_checks(), answers
            )
            if verdict["contradiction_veto"]:
                dd_text = source.get("documentation") or source.get("description") or ""
                contradictions.append(
                    f"{source_path}: {verdict['contradiction_reason']}; DD: {dd_text}"
                )
            elif any(
                answer is None
                or answer.get("choice") not in {"supports", "says nothing"}
                for answer in verdict["claim_answers"]
            ):
                unavailable.append(source_path)
        if contradictions:
            return "contradicts source: " + " | ".join(contradictions)
        if unavailable:
            logger.warning(
                "review_docs: claim answer missing for %s at %s",
                item["id"],
                ", ".join(unavailable),
            )
            return "claim comparison unavailable at " + ", ".join(unavailable)
        return None
    except Exception:
        logger.warning(
            "review_docs: claim comparison errored for %s; holding candidate",
            item["id"],
            exc_info=True,
        )
        return "claim comparison unavailable: Jev error"
    finally:
        lease.release_unused()


async def process_review_docs_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for docs review (RD-quorum).

    Mirrors :func:`process_review_name_batch` for the docs axis: runs the
    full ``[sn.review.docs].models`` chain (cycle 0 + cycle 1 unconditionally,
    cycle 2 only when per-dim disagreement exceeds threshold). Persists each
    cycle as a ``StandardNameReview`` node and transitions ``docs_stage`` via
    :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_docs`.

    Returns count of items successfully reviewed.
    """
    _documentation_banned_prose_note = _worker_dependency(
        "_documentation_banned_prose_note"
    )
    _documentation_conformance_note = _worker_dependency(
        "_documentation_conformance_note"
    )
    _documentation_contradiction_note = _worker_dependency(
        "_documentation_contradiction_note"
    )
    _persist_dd_gap_evidence = _worker_dependency("_persist_dd_gap_evidence")
    _persisting_quorum_cycles = _worker_dependency("_persisting_quorum_cycles")
    _release_docs_review_claim = _worker_dependency("_release_docs_review_claim")
    _review_dd_source_bindings = _worker_dependency("_review_dd_source_bindings")
    prepare_docs_review_request = _worker_dependency("prepare_docs_review_request")
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_store.graph_catalog import (
        update_review_aggregates,
    )
    from imas_codex.standard_names.graph_store.graph_review_persistence import (
        persist_reviewed_docs,
    )

    processed = 0

    for item in batch:
        if stop_event.is_set():
            await _release_docs_review_claim(item)
            continue

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        lease = None

        try:
            request = await prepare_docs_review_request(item)

            from functools import partial

            from imas_codex.standard_names.review.cascade import adjudicate_review

            quorum = await adjudicate_review(
                "docs",
                request.get("veto_item", item),
                request["models"],
                partial(
                    _persisting_quorum_cycles,
                    sn_id=sn_id,
                    review_axis="docs",
                    response_model=request["response_model"],
                    user_prompt=request["user_prompt"],
                    system_prompt=request["system_prompt"],
                    disagreement_threshold=request["disagreement_threshold"],
                    rubric_dims=tuple(request["rubric_dims"]),
                    lease=lease,
                    budget_manager=mgr,
                    phase="review_docs",
                    acall_llm_structured=acall_llm_structured,
                    reasoning_effort=request["reasoning_effort"],
                    escalation_reasoning_effort=request["escalation_reasoning_effort"],
                    run_id=mgr.run_id,
                    escalation_prompt_factory=request.get("escalation_prompt_factory"),
                    review_group_id=None,
                ),
                mgr=mgr,
            )

            if quorum is None:
                await _release_docs_review_claim(item)
                continue

            score = quorum["winning_score"]
            comments = quorum["winning_comments"]
            comments_per_dim = quorum["winning_comments_per_dim"]
            conformance_note = _documentation_conformance_note(item)
            prose_note = await _documentation_banned_prose_note(item, mgr)
            contradiction_note = await _documentation_contradiction_note(
                request.get("veto_item", item), mgr
            )
            checked_holds = (
                ("registry_conformance", conformance_note, "conventions-registry"),
                ("banned_prose", prose_note, "prose-adjudicator"),
                (
                    "documentation_contradiction",
                    contradiction_note,
                    "jev-contradiction",
                ),
            )
            review_holds: list[dict[str, str]] = []
            for kind, note, raised_by in checked_holds:
                if note is None:
                    continue
                evidence = json.dumps(
                    {
                        "claim": note,
                        "documentation": item.get("documentation") or "",
                        "dd_source_docs": item.get("dd_source_docs") or [],
                    },
                    sort_keys=True,
                    default=str,
                )
                review_holds.append(
                    {
                        "kind": kind,
                        "finding": note,
                        "evidence": evidence,
                        "raised_by": raised_by,
                    }
                )
            if any(note is not None for _, note, _ in checked_holds):
                notes = [
                    note
                    for note in (conformance_note, prose_note, contradiction_note)
                    if note
                ]
                comments = "\n".join(filter(None, (comments, *notes)))
                comments_per_dim = dict(comments_per_dim or {})
                if conformance_note is not None:
                    comments_per_dim["registry_conformance"] = conformance_note
                if prose_note is not None:
                    comments_per_dim["banned_prose"] = prose_note
                if contradiction_note is not None:
                    comments_per_dim["contradicts_source"] = contradiction_note
                logger.info(
                    "review_docs: %s found documentation veto — %s",
                    sn_id,
                    "; ".join(notes),
                )
            new_stage = await _asyncio.to_thread(
                persist_reviewed_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                claim_seq=item.get("claim_seq"),
                score=score,
                scores=quorum["winning_scores"],
                comments=comments,
                comments_per_dim=comments_per_dim,
                review_holds=review_holds,
                checked_hold_kinds={kind for kind, _, _ in checked_holds},
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
                resolution_method=quorum["resolution_method"],
                reviewer_chain_size=len(request["models"]),
            )
            if new_stage:
                source_versions = _review_dd_source_bindings(item)
                await _asyncio.to_thread(
                    _persist_dd_gap_evidence,
                    quorum["dd_gaps"],
                    set(source_versions),
                    phase="review_docs",
                    reporter="review-docs",
                    source_versions=source_versions,
                )

            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_docs: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_docs",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_docs: %s persist no-op (token mismatch?)", sn_id)

        except asyncio.CancelledError:
            await _release_docs_review_claim(item)
            raise
        except Exception:
            logger.exception("review_docs failed for %s", sn_id)
            try:
                await _release_docs_review_claim(item)
            except Exception:
                logger.debug("documentation claim cleanup also failed for %s", sn_id)
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed
