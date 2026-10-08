"""IMAS IDS names and data dictionary paths extracted from code.

Scans text for IMAS IDS references (equilibrium, core_profiles, etc.)
using regex patterns. Works on code, documents, and wiki pages.
"""

import re

from imas_codex.discovery.base.imas_patterns import (
    extract_ids_names,
    extract_imas_paths,
    get_all_ids_names,
    normalize_imas_path,
)

# Regex patterns for IMAS IDS detection
IDS_PATTERNS = [
    # Python: ids_factory.new("equilibrium")
    r'\.new\(["\'](\w+)["\']\)',
    # Python: factory.equilibrium()
    r"factory\.(\w+)\(\)",
    # String literals that are IDS names
    r'["\'](\w+)["\']',
]


def get_known_ids() -> frozenset[str]:
    """Get the set of valid IDS names from the data dictionary.

    Delegates to the shared ``imas_codex.discovery.base.imas_patterns``
    module for the canonical IDS name list.

    Returns:
        Frozen set of lowercase IDS names
    """
    return frozenset(get_all_ids_names())


def extract_ids_references(text: str) -> set[str]:
    """Extract IMAS IDS references from text.

    Delegates to the shared ``imas_codex.discovery.base.imas_patterns``
    module which uses the same IDS name list across the entire pipeline.

    Args:
        text: Text to scan for IDS references

    Returns:
        Set of IDS names found
    """
    return extract_ids_names(text)


_SEGMENT = r"[A-Za-z_][A-Za-z_0-9]*(?:\([^()\n]*\)|\[[^][\n]*\])?"
_FORTRAN_CHAIN = re.compile(rf"\b([A-Za-z_][A-Za-z_0-9]*)((?:%{_SEGMENT})+)", re.I)
_FORTRAN_INDEX = re.compile(r"\([^()\n]*\)")


def extract_imas_path_references(
    text: str, related_ids: set[str] | None = None
) -> list[str]:
    """Return IDS-rooted DD paths from dotted, slash, and Fortran chains.

    A generic Fortran ``ids`` variable needs exactly one known IDS name in the
    same chunk. Ambiguous variables are left unresolved rather than guessed.
    Graph linking separately requires an existing IMASNode for every path.
    """
    paths = set(extract_imas_paths(text))
    names = related_ids if related_ids is not None else extract_ids_references(text)
    known = get_known_ids()
    for match in _FORTRAN_CHAIN.finditer(text):
        root = match.group(1).lower()
        if root == "ids" and len(names) == 1:
            root = next(iter(names))
        if root not in known:
            continue
        chain = _FORTRAN_INDEX.sub("", match.group(2)).replace("%", "/")
        paths.add(normalize_imas_path(root + chain))
    return sorted(paths)


__all__ = [
    "extract_ids_references",
    "extract_imas_path_references",
    "get_known_ids",
]
