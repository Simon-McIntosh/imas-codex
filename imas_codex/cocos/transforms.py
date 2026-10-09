"""COCOS path transforms for IMAS Data Dictionary version conversions.

Integrates with imas-python's internal path lists to identify which
IMAS paths require sign flips when converting between DD versions.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _dd_cocos_labels(dd_version: str | None) -> dict[str, str]:
    """Read the DD's field labels once for each requested version."""
    import imas.dd_zip

    tree = imas.dd_zip.dd_etree(dd_version)
    root = tree.getroot() if hasattr(tree, "getroot") else tree
    labels: dict[str, str] = {}
    for ids in root:
        ids_name = ids.get("name")
        if not ids_name:
            continue
        for field in ids.iter("field"):
            path = field.get("path")
            label = field.get("cocos_transformation_type") or field.get(
                "cocos_label_transformation"
            )
            if path and label:
                labels[f"{ids_name}/{path}"] = label
    return labels


def dd_cocos_class(path: str, *, dd_version: str | None = None) -> str | None:
    """Resolve a DD field's COCOS class from its own or an ancestor's label.

    ``path`` includes the IDS name. A signal's ``data`` leaf commonly inherits
    the class of its enclosing signal structure.
    """
    labels = _dd_cocos_labels(dd_version)
    parts = path.strip("/").split("/")
    for end in range(len(parts), 1, -1):
        parent = "/".join(parts[:end])
        if parent in labels:
            return labels[parent]
    # The published DD XML omits this label, although the flux is a ψ signal.
    if len(parts) >= 3 and parts[:3] == ["magnetics", "flux_loop", "flux"]:
        return "psi_like"
    return None


@lru_cache(maxsize=1)
def _get_sign_flip_paths() -> dict[str, frozenset[str]]:
    """Load sign flip paths from imas-python.

    Returns:
        Dict mapping IDS name to set of paths requiring sign flip.
    """
    try:
        from imas.ids_convert import _3to4_sign_flip_paths

        return {
            ids_name: frozenset(paths)
            for ids_name, paths in _3to4_sign_flip_paths.items()
        }
    except ImportError:
        return {}


def path_needs_cocos_transform(ids_name: str, path: str) -> bool:
    """Check if path requires COCOS sign flip between DD3/DD4.

    The IMAS Data Dictionary changed from COCOS 11 to COCOS 17 at version 4.0.0.
    Certain paths require sign flips when converting between these conventions.

    Args:
        ids_name: Name of the IDS (e.g., "equilibrium", "core_profiles")
        path: Relative path within the IDS (e.g., "time_slice/global_quantities/psi_axis")

    Returns:
        True if the path requires a sign flip, False otherwise.

    Example:
        >>> path_needs_cocos_transform("equilibrium", "time_slice/global_quantities/psi_axis")
        True
        >>> path_needs_cocos_transform("equilibrium", "time_slice/global_quantities/ip")
        False
    """
    flip_paths = _get_sign_flip_paths()
    ids_paths = flip_paths.get(ids_name.lower(), frozenset())
    return path in ids_paths


def get_sign_flip_paths(ids_name: str) -> list[str]:
    """Get all paths requiring sign flip for an IDS.

    Args:
        ids_name: Name of the IDS (e.g., "equilibrium")

    Returns:
        List of paths requiring sign flip for DD3->DD4 conversion.

    Example:
        >>> paths = get_sign_flip_paths("equilibrium")
        >>> "time_slice/global_quantities/psi_axis" in paths
        True
    """
    flip_paths = _get_sign_flip_paths()
    ids_paths = flip_paths.get(ids_name.lower(), frozenset())
    return sorted(ids_paths)


def list_ids_with_sign_flips() -> list[str]:
    """Get all IDS names that have paths requiring sign flips.

    Returns:
        Sorted list of IDS names with sign flip paths.
    """
    flip_paths = _get_sign_flip_paths()
    return sorted(flip_paths.keys())
