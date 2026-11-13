"""Service wiring for standard-name worker tasks.

Task implementations and public worker entry points live in ``worker_tasks``.
The task package uses this composition root for collaborators supplied by other
standard-name modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from importlib import import_module

from imas_codex.standard_names.budget import (
    bind_attempt_exposure,
    charge_billable_exception,
    model_provider_exposure,
)
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    SEMANTIC_SIM_GATE_MODEL,
    SEMANTIC_SIM_GATE_RESOLUTION_METHOD,
)
from imas_codex.standard_names.provenance import detect_value_provenance
from imas_codex.standard_names.source_paths import (
    DD_PREFIX,
    encode_source_path,
    strip_dd_prefix,
)

logger = logging.getLogger(__name__)

#: Collaborators a task module requests by name, declared rather than read out
#: of the module namespace: the set a task may ask for is then readable here,
#: and a name this root does not carry raises ``KeyError`` exactly as the
#: namespace it replaces did. Each value is read when it is requested, so a
#: caller that replaces one of these names on this module is the one a task
#: sees; holding the imported object here would serve the import instead.
_WORKER_DEPENDENCIES: dict[str, Callable[[], object]] = {
    "DD_PREFIX": lambda: DD_PREFIX,
    "DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER": (
        lambda: DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    ),
    "SEMANTIC_SIM_GATE_MODEL": lambda: SEMANTIC_SIM_GATE_MODEL,
    "SEMANTIC_SIM_GATE_RESOLUTION_METHOD": lambda: SEMANTIC_SIM_GATE_RESOLUTION_METHOD,
    "_name_definition_note": lambda: _task_collaborator(
        "worker_review_docs", "_name_definition_note"
    ),
    "asyncio": lambda: asyncio,
    "bind_attempt_exposure": lambda: bind_attempt_exposure,
    "charge_billable_exception": lambda: charge_billable_exception,
    "detect_value_provenance": lambda: detect_value_provenance,
    "encode_source_path": lambda: encode_source_path,
    "json": lambda: json,
    "logger": lambda: logger,
    "model_provider_exposure": lambda: model_provider_exposure,
    "strip_dd_prefix": lambda: strip_dd_prefix,
}


def _task_collaborator(module: str, name: str) -> object:
    """Read a collaborator that a task module defines, when a task asks for it.

    A task module's own collaborators cannot be imported here at module load,
    because the package imports this root lazily from inside a task call. The
    attribute is read from the module at request time, so a caller that
    replaces the name on the owning module is still the one a task sees.
    """
    return getattr(
        import_module(f"imas_codex.standard_names.worker_tasks.{module}"), name
    )


def _worker_dependency(name: str):
    """Resolve a non-task collaborator through the namespace that holds it."""
    return _WORKER_DEPENDENCIES[name]()
