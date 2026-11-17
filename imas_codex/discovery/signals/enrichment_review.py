"""Coalesce simultaneous local reviews without sharing citation identities."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from imas_codex.discovery.signals.enrichment import DescriptionReview


class ReviewCallError(ValueError):
    """One request's share of a failed pooled call's consumed telemetry."""

    def __init__(self, message, cost, tokens):
        super().__init__(message)
        self.cost, self.tokens = cost, tokens


@dataclass
class ReviewRequest:
    call: object
    kwargs: dict
    future: asyncio.Future


class ReviewBatcher:
    """Collect a bounded group; a short deadline also flushes the final tail."""

    def __init__(self, size: int):
        self.size = size
        self.pending: list[ReviewRequest] = []
        self.timer = None

    async def submit(self, call, kwargs):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append(ReviewRequest(call, kwargs, future))
        if len(self.pending) == self.size:
            if self.timer:
                self.timer.cancel()
            self.flush()
        elif self.timer is None:
            self.timer = loop.call_later(2.0, self.flush)
        return await future

    def flush(self):
        requests, self.pending = self.pending, []
        self.timer = None
        if requests:
            # Every request awaits its own future until this task completes.
            asyncio.create_task(self.deliver(requests))

    async def deliver(self, requests):
        cost, tokens = 0.0, 0
        try:
            if len(requests) == 1:
                request = requests[0]
                values = [await request.call(**request.kwargs)]
            else:
                contexts, drafts, allowed, bindings = {}, {}, {}, {}
                for request_index, request in enumerate(requests):
                    data = json.loads(request.kwargs["messages"][1]["content"])
                    prefix = f"request-{request_index}-"
                    contexts.update(
                        {
                            prefix + key: text
                            for key, text in data["context_items"].items()
                        }
                    )
                    for index, statements in data["drafts"].items():
                        merged_index = len(bindings) + 1
                        bindings[merged_index] = (request_index, int(index), prefix)
                        drafts[merged_index] = statements
                        allowed[merged_index] = [
                            prefix + key for key in data["context_items"]
                        ]
                kwargs = dict(requests[0].kwargs)
                kwargs["messages"] = [
                    kwargs["messages"][0],
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "context_items": contexts,
                                "drafts": drafts,
                                "allowed_context_items": allowed,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
                merged, cost, tokens = await requests[0].call(**kwargs)
                indices = [row.signal_index for row in merged.results]
                if set(indices) != set(bindings) or len(indices) != len(bindings):
                    raise ValueError(
                        "batched review must cover each signal exactly once"
                    )
                outputs = [[] for _ in requests]
                for row in merged.results:
                    request_index, index, prefix = bindings[row.signal_index]
                    row = row.model_copy(deep=True)
                    row.signal_index = index
                    for statement in row.statements:
                        # A citation to another request cannot become a local ID.
                        references = [
                            item.strip("[]") for item in statement.context_items
                        ]
                        statement.context_items = [
                            item[len(prefix) :] if item.startswith(prefix) else item
                            for item in references
                        ]
                    outputs[request_index].append(row)
                values = [
                    (
                        DescriptionReview(results=rows),
                        cost / len(requests),
                        tokens // len(requests) + (index < tokens % len(requests)),
                    )
                    for index, rows in enumerate(outputs)
                ]
            for request, value in zip(requests, values, strict=True):
                if not request.future.done():
                    request.future.set_result(value)
        except BaseException as exc:
            from imas_codex.discovery.base.llm import ProviderBudgetExhausted

            cost += getattr(exc, "cost", 0) or 0
            tokens += getattr(exc, "tokens", 0) or 0
            for index, request in enumerate(requests):
                if not request.future.done():
                    error = exc
                    if (
                        len(requests) > 1
                        and isinstance(exc, ValueError)
                        and not isinstance(exc, ProviderBudgetExhausted)
                        and not getattr(exc, "telemetry_failure", False)
                    ):
                        error = ReviewCallError(
                            str(exc),
                            cost / len(requests),
                            tokens // len(requests) + (index < tokens % len(requests)),
                        )
                    request.future.set_exception(error)


_BATCHERS = WeakKeyDictionary()


async def batched_review(call, size, **kwargs):
    """Share only reviews with identical model, controls and system contract."""
    loop = asyncio.get_running_loop()
    controls = {
        key: value
        for key, value in kwargs.items()
        if key not in ("messages", "response_model")
    }
    key = (size, json.dumps(controls, sort_keys=True), kwargs["messages"][0]["content"])
    batchers = _BATCHERS.setdefault(loop, {})
    return await batchers.setdefault(key, ReviewBatcher(size)).submit(call, kwargs)
