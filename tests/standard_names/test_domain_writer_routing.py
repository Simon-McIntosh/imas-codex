"""Every pipeline domain writer routes through the one writer.

A standard name's ``physics_domain`` is its stored Jev decision and its
``source_domains`` is the value the one writer sets with it. The compose,
consolidation, catalog-import and export paths must therefore stop selecting a
domain themselves: none of them calls the ranking selectors
(``select_primary_domain``, ``maybe_promote_domain``), and the path that
writes a name's domain goes through
:func:`imas_codex.standard_names.domain_decision.write_name_domain`.

The assertions read each routed module's source through its import, so
restoring a selector call in a routed function — the declared negative control
— turns the module-wide check red.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

#: Routed module -> the functions this node routed, by import path.
ROUTED_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "imas_codex.standard_names.consolidation": (
        "consolidate_candidates",
        "_resolve_physics_domains",
        "_merge_duplicates",
    ),
    "imas_codex.standard_names.catalog_import": (
        "check_catalog",
        "record_catalog_import_provenance",
    ),
    "imas_codex.standard_names.export": (
        "_graph_node_to_entry_dict",
        "_candidate_physics_domains",
    ),
    "imas_codex.standard_names.workers": (
        "compose_worker",
        "compose_batch",
        "consolidate_worker",
        "_write_composed_name_domains",
    ),
}

#: The selectors no routed function may call once routed.
BANNED_SELECTORS = ("select_primary_domain", "maybe_promote_domain")

#: The one writer every domain-setting routed path must call.
ONE_WRITER = "write_name_domain"


def _function_source(module_name: str, func_name: str) -> str:
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return inspect.getsource(func)


@pytest.mark.parametrize(
    ("module_name", "func_name"),
    [(m, f) for m, funcs in ROUTED_FUNCTIONS.items() for f in funcs],
)
def test_routed_function_calls_no_selector(module_name: str, func_name: str) -> None:
    """No routed function selects a domain itself."""
    source = _function_source(module_name, func_name)
    for banned in BANNED_SELECTORS:
        assert banned not in source, (
            f"{module_name}.{func_name} still calls {banned}; the name's domain "
            f"must come from the one writer"
        )


@pytest.mark.parametrize("module_name", list(ROUTED_FUNCTIONS))
def test_routed_module_calls_no_selector(module_name: str) -> None:
    """The routed modules carry no selector call anywhere in their source."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    for banned in BANNED_SELECTORS:
        assert banned not in source, (
            f"{module_name} still references {banned}; route it through {ONE_WRITER}"
        )


def test_compose_sets_domains_through_the_one_writer() -> None:
    """A composed name's domain is set by the one writer, not carried in."""
    helper = _function_source(
        "imas_codex.standard_names.workers", "_write_composed_name_domains"
    )
    assert ONE_WRITER in helper, (
        f"the compose domain path must set the name's domain by calling {ONE_WRITER}"
    )
    for caller in ("compose_worker", "compose_batch"):
        source = _function_source("imas_codex.standard_names.workers", caller)
        assert "_write_composed_name_domains" in source, (
            f"{caller} must hand its persisted names to the one writer"
        )
