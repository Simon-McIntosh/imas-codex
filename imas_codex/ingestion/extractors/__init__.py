"""Content extractors for entity recognition during ingestion.

Extractors scan chunk text for domain-specific references:
- IDS: IMAS IDS references (equilibrium, core_profiles, etc.)
- MDSplus: Tree paths and TDI function calls
- Units: Physical units mentioned in text
- Conventions: Sign conventions and COCOS references
"""

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from imas_codex.ingestion.graph import (
    link_chunks_to_data_nodes,
    link_chunks_to_edas_signals,
)

from .edas import EDASReference, extract_edas_references
from .ids import extract_ids_references, get_known_ids
from .mdsplus import MDSplusReference, extract_mdsplus_paths
from .units import extract_conventions, extract_units


class ReferenceHandler(NamedTuple):
    """Code reference extractor and its graph linker."""

    extractor: Callable[[str], list[Any]]
    linker: Callable[..., Any]


_MDSPLUS_HANDLER = ReferenceHandler(extract_mdsplus_paths, link_chunks_to_data_nodes)
REFERENCE_HANDLERS: dict[str, ReferenceHandler] = {
    "edas": ReferenceHandler(extract_edas_references, link_chunks_to_edas_signals),
    "mdsplus": _MDSPLUS_HANDLER,
    "tdi": _MDSPLUS_HANDLER,
}


def reference_handlers_for_systems(
    data_systems: Mapping[str, Any],
) -> tuple[ReferenceHandler, ...]:
    """Select each configured extractor and linker once per facility."""
    return tuple(
        dict.fromkeys(
            REFERENCE_HANDLERS[name]
            for name in data_systems
            if name in REFERENCE_HANDLERS
        )
    )


__all__ = [
    "EDASReference",
    "MDSplusReference",
    "REFERENCE_HANDLERS",
    "ReferenceHandler",
    "extract_conventions",
    "extract_edas_references",
    "extract_ids_references",
    "extract_mdsplus_paths",
    "extract_units",
    "get_known_ids",
    "reference_handlers_for_systems",
]
