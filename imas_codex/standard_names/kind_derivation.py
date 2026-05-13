"""Auto-derive ``kind`` from the standard name string (D5/P0.3).

Deterministic pattern-match overriding the LLM's ``kind`` field.
The LLM defaults to ``scalar`` for everything; this module inspects
the name tokens to assign the structurally correct ``StandardNameKind``.

All returned values are validated against the LinkML ``StandardNameKind``
enum at import time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Valid Kind enum values (imported lazily to avoid circular imports at
# module level; validated once on first call).
_VALID_KINDS: frozenset[str] | None = None


def _load_valid_kinds() -> frozenset[str]:
    """Load the StandardNameKind enum values from generated models.

    Always unions with the full fallback set so a stale/partial enum load
    (e.g. a long-running process holding an old ``models.py``) cannot
    silently disable pattern rules.
    """
    global _VALID_KINDS
    if _VALID_KINDS is not None:
        return _VALID_KINDS
    fallback = frozenset(
        {
            "scalar",
            "vector",
            "tensor",
            "eigenfunction",
            "spectrum",
            "complex",
            "metadata",
        }
    )
    try:
        from imas_codex.graph.models import StandardNameKind

        loaded = frozenset(e.value for e in StandardNameKind)
        _VALID_KINDS = loaded | fallback
    except Exception:
        _VALID_KINDS = fallback
    return _VALID_KINDS


def derive_kind(name: str) -> str:
    """Return the most specific ``StandardNameKind`` value for *name*.

    Pattern rules (evaluated in order — first match wins):

    1. Leading axis qualifier for vector projection → ``scalar`` (a component
       is a scalar projection of the parent vector, not the vector itself).
       Detected via ISN grammar parse when available, with regex fallback
       for names starting with axis tokens like ``radial_``, ``toroidal_``, etc.
    2. ``_tensor`` token (e.g. ``metric_tensor``, ``stress_tensor``) →
       ``tensor``
    3. ``_eigenfunction`` → ``eigenfunction``
    4. endswith ``_spectrum`` → ``spectrum``
    5. ``real_part`` or ``imaginary_part`` → ``complex``
    6. default → ``scalar``
    """
    valid = _load_valid_kinds()
    name_lower = name.lower()

    # 1. Component of a vector — the component itself is a scalar projection.
    # Short form: axis qualifier prefix like `toroidal_magnetic_field`.
    _AXIS_TOKENS = {
        "radial",
        "toroidal",
        "poloidal",
        "parallel",
        "perpendicular",
        "normal",
        "tangential",
        "vertical",
        "horizontal",
        "binormal",
        "x",
        "y",
        "z",
        "r",
        "phi",
    }
    _VECTOR_BASES = {
        "magnetic_field",
        "electric_field",
        "velocity",
        "current_density",
        "heat_flux",
        "momentum_flux",
        "force",
        "surface_normal",
        "acceleration",
        "displacement",
        "rotation_frequency",
    }
    first_token = name_lower.split("_", 1)[0]
    if first_token in _AXIS_TOKENS and "scalar" in valid:
        rest = name_lower[len(first_token) + 1 :]
        for vb in _VECTOR_BASES:
            if rest == vb or rest.endswith(f"_{vb}"):
                return "scalar"

    # 2. Tensor
    # Match tokens like _tensor_, _tensor (end of name), but NOT
    # names that merely mention tensor in a qualifier
    if "_tensor" in name_lower:
        # Check it's a real tensor reference (not e.g. "tensor_product_of_...")
        # by verifying _tensor is at the end or followed by _
        import re

        if re.search(r"_tensor(?:_|$)", name_lower):
            if "tensor" in valid:
                return "tensor"

    # 3. Eigenfunction
    if "_eigenfunction" in name_lower or name_lower == "eigenfunction":
        if "eigenfunction" in valid:
            return "eigenfunction"

    # 4. Spectrum
    if name_lower.endswith("_spectrum"):
        if "spectrum" in valid:
            return "spectrum"

    # 5. Complex part (real/imaginary)
    if "real_part" in name_lower or "imaginary_part" in name_lower:
        if "complex" in valid:
            return "complex"

    # 6. Default
    return "scalar"


# Mapping from extended local kinds → ISN's discriminator.
# ISN now supports {scalar, vector, tensor, complex, metadata}.
# Eigenfunction and spectrum are still codex-only extended kinds
# that collapse to scalar for ISN validation.
_ISN_KIND_MAP: dict[str, str] = {
    "scalar": "scalar",
    "vector": "vector",
    "tensor": "tensor",
    "eigenfunction": "scalar",
    "spectrum": "scalar",
    "complex": "complex",
    "metadata": "metadata",
}


def to_isn_kind(kind: str | None) -> str:
    """Map a local extended kind value to one ISN's discriminator accepts.

    Defaults to ``scalar`` for unknown values so validation never crashes
    on a kind the ISN library doesn't recognise.
    """
    if not kind:
        return "scalar"
    return _ISN_KIND_MAP.get(kind, "scalar")
