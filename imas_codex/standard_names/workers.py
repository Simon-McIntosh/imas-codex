"""Service wiring for standard-name worker tasks.

Task implementations and public worker entry points live in ``worker_tasks``.
The task package uses this composition root for collaborators supplied by other
standard-name modules.
"""

from __future__ import annotations

import asyncio
import json
import logging

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
#: namespace it replaces did.
_WORKER_DEPENDENCIES: dict[str, object] = {
    "DD_PREFIX": DD_PREFIX,
    "DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER": (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    ),
    "SEMANTIC_SIM_GATE_MODEL": SEMANTIC_SIM_GATE_MODEL,
    "SEMANTIC_SIM_GATE_RESOLUTION_METHOD": SEMANTIC_SIM_GATE_RESOLUTION_METHOD,
    "asyncio": asyncio,
    "bind_attempt_exposure": bind_attempt_exposure,
    "charge_billable_exception": charge_billable_exception,
    "detect_value_provenance": detect_value_provenance,
    "encode_source_path": encode_source_path,
    "json": json,
    "logger": logger,
    "model_provider_exposure": model_provider_exposure,
    "strip_dd_prefix": strip_dd_prefix,
}


def _worker_dependency(name: str):
    """Resolve a non-task collaborator through the composition root."""
    return _WORKER_DEPENDENCIES[name]
