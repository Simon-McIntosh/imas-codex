"""Structured review doubles for worker-wiring tests with fixed draft answers."""

import json
import re

from iter_codex.discovery.signals.enrichment import DescriptionReview


def with_description_review(draft_call):
    """Accept fixture drafts with target-bound citations, preserving real wiring.

    These doubles test callers and result routing. Semantic rejection and citation
    validity are exercised by the grounded enrichment tests.
    """

    async def respond(*, messages, response_model, **kwargs):
        if response_model is not DescriptionReview:
            return await draft_call(
                messages=messages, response_model=response_model, **kwargs
            )
        payload = json.loads(messages[1]["content"])
        references = {}
        index = None
        for key, line in payload["context_items"].items():
            match = re.fullmatch(r"### Signal (\d+)", line)
            if match:
                index = int(match[1])
            elif index is not None and line.startswith("accessor: "):
                references[index] = key
        return (
            DescriptionReview(
                results=[
                    {
                        "signal_index": int(index),
                        "statements": [
                            {
                                "draft_index": position,
                                "text": text,
                                "context_items": [references[int(index)]],
                            }
                            for position, text in enumerate(statements, 1)
                        ],
                    }
                    for index, statements in payload["drafts"].items()
                ]
            ),
            0.0,
            0,
        )

    return respond
