"""Service wiring for standard-name worker tasks.

Task implementations and public worker entry points live in ``worker_tasks``.
The task package uses this composition root for collaborators supplied by other
standard-name modules.
"""

from __future__ import annotations

import asyncio  # noqa: F401 - resolved by task modules through _worker_dependency
import json  # noqa: F401 - resolved by task modules through _worker_dependency
import logging

from imas_codex.standard_names.budget import (
    bind_attempt_exposure,  # noqa: F401
    charge_billable_exception,  # noqa: F401
    model_provider_exposure,  # noqa: F401
)
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,  # noqa: F401
    SEMANTIC_SIM_GATE_MODEL,  # noqa: F401
    SEMANTIC_SIM_GATE_RESOLUTION_METHOD,  # noqa: F401
)
from imas_codex.standard_names.provenance import detect_value_provenance  # noqa: F401
from imas_codex.standard_names.source_paths import (
    DD_PREFIX,  # noqa: F401
    encode_source_path,  # noqa: F401
    strip_dd_prefix,  # noqa: F401
)


def _worker_dependency(name: str):
    """Resolve a non-task collaborator through the composition root."""
    return globals()[name]


logger = logging.getLogger(__name__)
