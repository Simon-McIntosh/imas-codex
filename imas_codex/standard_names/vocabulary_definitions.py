"""Definitions declared by the closed standard-name vocabulary."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def vocabulary_definitions(*, include_loci: bool = False) -> dict[str, str]:
    """Map defined base and carrier tokens, optionally including loci.

    Tokens without a nonblank definition are absent from the map.
    """
    try:
        from imas_standard_names.grammar import vocab_loaders
    except ImportError:  # pragma: no cover - ISN absent
        return {}

    definitions: dict[str, str] = {}
    try:
        if include_loci:
            for token, entry in vocab_loaders.load_locus_registry().loci.items():
                text = (entry.definition or "").strip()
                if text:
                    definitions[token] = text
        for token, entry in vocab_loaders.load_physical_bases().bases.items():
            text = (entry.definition or "").strip()
            if text:
                definitions[token] = text
        for token, entry in vocab_loaders.load_geometry_carriers().carriers.items():
            text = (entry.definition or "").strip()
            if text:
                definitions[token] = text
    except Exception:  # pragma: no cover - ISN restructured
        return {}
    return definitions
