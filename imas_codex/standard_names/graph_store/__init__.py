"""Graph storage domains for standard names.

Storage functions accept an explicit service object for internal composition and
also provide the ordinary caller signature through the wired graph root.
"""

from __future__ import annotations

import functools
import inspect
from importlib import import_module
from typing import Any


def _bind_services(function: Any, signature: inspect.Signature) -> Any:
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_bound(*args: Any, **kwargs: Any) -> Any:
            if "services" not in signature.bind_partial(*args, **kwargs).arguments:
                from .graph_foundation import get_services

                kwargs["services"] = get_services()
            return await function(*args, **kwargs)

        wrapper = async_bound
    else:

        @functools.wraps(function)
        def bound(*args: Any, **kwargs: Any) -> Any:
            if "services" not in signature.bind_partial(*args, **kwargs).arguments:
                from .graph_foundation import get_services

                kwargs["services"] = get_services()
            return function(*args, **kwargs)

        wrapper = bound

    for attribute in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(function, attribute):
            setattr(wrapper, attribute, getattr(function, attribute))
    return wrapper


for _name in (
    "graph_foundation",
    "graph_audits",
    "graph_catalog",
    "graph_catalog_reset",
    "graph_claim_release",
    "graph_claim_scope",
    "graph_cost_and_claims",
    "graph_doc_links",
    "graph_docs_persistence",
    "graph_lineage",
    "graph_parent_lifecycle",
    "graph_refinement",
    "graph_review_persistence",
    "graph_source_bindings",
    "graph_supersession",
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
            _signature = inspect.signature(_function)
        except (TypeError, ValueError):
            continue
        if "services" in _signature.parameters:
            setattr(_module, _symbol, _bind_services(_function, _signature))
