"""Worker task implementations for the standard-name pipeline.

Task functions that accept the composition root resolver also accept ordinary
caller arguments directly. Internal callers may still pass a resolver first.
"""

from __future__ import annotations

import functools
import inspect
from importlib import import_module
from typing import Any


def _resolve_dependency(name: str) -> Any:
    from imas_codex.standard_names.workers import _worker_dependency

    return _worker_dependency(name)


def _bind_task(function: Any) -> Any:
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_bound(*args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]):
                return await function(*args, **kwargs)
            return await function(_resolve_dependency, *args, **kwargs)

        wrapper = async_bound
    else:

        @functools.wraps(function)
        def bound(*args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]):
                return function(*args, **kwargs)
            return function(_resolve_dependency, *args, **kwargs)

        wrapper = bound

    for attribute in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(function, attribute):
            setattr(wrapper, attribute, getattr(function, attribute))
    return wrapper


for _name in (
    "worker_common",
    "worker_compose_and_validate",
    "worker_compose_batches",
    "worker_enrich_parents",
    "worker_extraction_and_enrichment",
    "worker_generate_docs",
    "worker_refine_docs",
    "worker_refine_names",
    "worker_review_docs",
    "worker_review_names",
):
    _module = import_module(f".{_name}", __name__)
    for _symbol, _function in vars(_module).copy().items():
        if (
            not callable(_function)
            or inspect.isclass(_function)
            or getattr(_function, "__module__", None) != _module.__name__
        ):
            continue
        try:
            _parameters = iter(inspect.signature(_function).parameters.values())
            _first = next(_parameters)
        except (TypeError, ValueError, StopIteration):
            continue
        if _first.name == "_worker_dependency":
            setattr(_module, _symbol, _bind_task(_function))
