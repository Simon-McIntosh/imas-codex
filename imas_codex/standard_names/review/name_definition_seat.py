"""Chat review seat: does a definition state a qualifier its name does not.

The name and its definition are compared by a review model through the shared
LLM call layer, one batch of items per call. A flagged item becomes a review
hold in the docs review, so a name whose definition narrows, normalises,
weights, aggregates or generalises the quantity is not accepted until a
reviewer answers it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

QUALIFIERS = (
    "normalisation",
    "counting-convention",
    "weighting",
    "integration",
    "restriction-to-one-object",
    "family-of-quantities",
    "agrees",
)

BATCH_SIZE = 10


class NameDefinitionVerdict(BaseModel):
    """One item's verdict."""

    name: str = Field(description="The standard name, echoed back unchanged.")
    verdict: str = Field(description="`flag` or `pass`.")
    qualifier: str = Field(
        description="One of the qualifier kinds, or `agrees` when the verdict is pass."
    )
    reasoning: str = Field(
        description=(
            "The wording in the definition that states the extra qualifier, or "
            "the one quantity both texts share."
        )
    )


class NameDefinitionBatch(BaseModel):
    """One call's verdicts, in the order the items were given."""

    verdicts: list[NameDefinitionVerdict]


def review_model() -> str:
    """The configured docs review seat's primary reviewer."""

    from imas_codex.settings import get_sn_review_docs_models

    return get_sn_review_docs_models()[0]


def _item_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "standard_name": item.get("id") or item.get("name") or "",
        "description": item.get("description") or "",
        "documentation": item.get("documentation") or "",
        "unit": item.get("unit") or "",
        "derived_children": list(item.get("derived_children") or []),
    }


async def _judge_batch(
    batch: list[dict[str, Any]],
    *,
    seat: str,
    system_prompt: str,
    reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], float]:
    """One call's verdicts for one batch, plus its cost."""

    import asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt

    user_prompt = render_prompt(
        "sn/review_name_definition_user",
        {"items": [_item_payload(item) for item in batch]},
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def unanswered(reason: str) -> list[dict[str, Any]]:
        return [
            {
                "name": _item_payload(item)["standard_name"],
                "verdict": "unanswered",
                "qualifier": None,
                "reasoning": reason,
            }
            for item in batch
        ]

    try:
        result, cost, _ = await acall_llm_structured(
            model=seat,
            messages=messages,
            response_model=NameDefinitionBatch,
            service="standard-names",
            max_retries=2,
            reasoning_effort=reasoning_effort,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("name definition seat failed on a batch", exc_info=True)
        return unanswered("the review seat returned no verdict"), 0.0

    returned = {verdict.name: verdict for verdict in result.verdicts}
    verdicts: list[dict[str, Any]] = []
    for item in batch:
        name = _item_payload(item)["standard_name"]
        verdict = returned.get(name)
        if verdict is None:
            verdicts.extend(unanswered("the review seat returned no verdict for this name"))
            continue
        qualifier = verdict.qualifier if verdict.qualifier in QUALIFIERS else None
        flagged = verdict.verdict.strip().lower() == "flag" or qualifier not in (
            None,
            "agrees",
        )
        verdicts.append(
            {
                "name": name,
                "verdict": "flag" if flagged else "pass",
                "qualifier": qualifier,
                "reasoning": verdict.reasoning,
            }
        )
    return verdicts, cost


async def judge_name_definitions(
    items: list[dict[str, Any]],
    *,
    model: str | None = None,
    batch_size: int = BATCH_SIZE,
    concurrency: int = 6,
    reasoning_effort: str | None = None,
    progress_path: Path | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Compare each item's name with its definition, returning verdicts and cost.

    A call that fails is reported as an unanswered item rather than as a pass, so
    a reviewer sees the gap instead of a silent acceptance. ``progress_path``
    appends each batch's verdicts as they land, so a run cut short keeps what it
    had already paid for.
    """

    import asyncio

    from imas_codex.llm.prompt_loader import render_prompt

    seat = model or review_model()
    system_prompt = render_prompt("sn/review_name_definition_system", {})
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    limit = asyncio.Semaphore(max(1, concurrency))

    async def one(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        async with limit:
            return await _judge_batch(
                batch,
                seat=seat,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
            )

    pending = [asyncio.create_task(one(batch)) for batch in batches]
    verdicts: list[dict[str, Any]] = []
    total_cost = 0.0
    try:
        for finished in asyncio.as_completed(pending):
            batch_verdicts, batch_cost = await finished
            verdicts.extend(batch_verdicts)
            total_cost += batch_cost
            if progress_path is not None:
                with progress_path.open("a", encoding="utf-8") as handle:
                    for verdict in batch_verdicts:
                        handle.write(json.dumps(verdict, sort_keys=True) + "\n")
                logger.info(
                    "name definition seat: %s of %s verdicts recorded",
                    len(verdicts),
                    len(items),
                )
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
    return verdicts, total_cost


def name_definition_note(item: dict[str, Any], verdict: dict[str, Any]) -> str | None:
    """The review hold a verdict raises, or None when it passes.

    An unanswered verdict holds the candidate: a seat call that failed must not
    read as an acceptance.
    """

    outcome = verdict.get("verdict")
    if outcome == "pass":
        return None
    if outcome != "flag":
        return (
            "name-versus-definition review did not answer: "
            f"{verdict.get('reasoning') or 'no verdict returned'}"
        )
    qualifier = verdict.get("qualifier") or "unspecified-qualifier"
    return (
        f"definition states a {qualifier} the name does not: "
        f"{verdict.get('reasoning') or 'the definition quantifies something else'}"
    )
