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


def _worker_dependency(name: str):
    """Resolve a non-task collaborator through the namespace that holds it."""
    return _WORKER_DEPENDENCIES[name]()
