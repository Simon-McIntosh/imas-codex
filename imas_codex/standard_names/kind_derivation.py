"""Auto-derive ``kind`` from the standard name string.

Deterministic classification overriding the LLM's ``kind`` field (the LLM
defaults to ``scalar`` for everything).  The structural kind of a name is
authoritative in ISN: each ``physical_base`` token is declared ``scalar`` /
``vector`` / ``tensor`` in the ISN base registry, and a projected component
(or a magnitude reduction) of a vector is itself a scalar.  Kind is the
STRUCTURAL classification and mirrors the ISN catalog ``Kind`` enum exactly
(single vocabulary, LinkML ``StandardNameKind`` is the generated source) —
semantic categories are not kinds: an eigenfunction or a spectrum is
structurally a scalar unless it names a complex part or carries vector
topology.

All returned values are validated against the LinkML ``StandardNameKind``
enum at import time.
"""

from __future__ import annotations

import logging
import re
from functools import cache, lru_cache

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


# --- ISN structural classification -----------------------------------------
#
# ISN is the single source of truth for whether a base token has vector-like
# structure. We ask its strict public grammar whether the registered magnitude
# reduction accepts the base and cache that behavioral result per token.


# Decomposition operators that collapse a vector/tensor to a scalar, so the
# result is scalar regardless of the underlying base's kind.
@lru_cache(maxsize=1)
def _magnitude_operator() -> str | None:
    """Return the registered scalar-reducing postfix used for kind probing."""
    try:
        from imas_standard_names import get_grammar_context

        operators = (
            get_grammar_context()
            .get("grammar", {})
            .get("vocabularies", {})
            .get("operators", {})
        )
        metadata = operators.get("magnitude")
        if metadata and metadata.get("kind") == "unary_postfix":
            return "magnitude"
    except Exception:
        pass
    return None


@cache
def _base_accepts_magnitude(base: str) -> bool:
    """Ask the public strict grammar whether *base* has vector-like structure."""
    operator = _magnitude_operator()
    if not operator:
        return False
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        parse_canonical_name(f"{base}_{operator}")
    except Exception:
        return False
    return True


def _isn_structural_kind(name: str) -> str | None:
    """Return the ISN-derived structural kind of *name*, or ``None``.

    - A single projected axis (vector component or coordinate) or a magnitude
      reduction is a scalar projection of its parent vector/tensor → ``scalar``.
    - Otherwise the kind is the ``physical_base`` token's declared kind in the
      ISN registry (``scalar`` / ``vector`` / ``tensor``).

    Returns ``None`` when the name cannot be parsed or its base is not a
    registered ``physical_base`` (geometry-carrier-only forms fall through to
    ``scalar``; binary-operator / unparseable forms return ``None`` so the
    caller applies its default).
    """
    try:
        from imas_standard_names.grammar import parse_standard_name

        # The flat facade is intentional here: this function reads legacy
        # component/decomposition/base fields and does not decide validity.
        parsed = parse_standard_name(name)
    except Exception:
        return None
    # A projected component or coordinate axis is a scalar projection of the
    # parent vector/tensor, not the vector itself.
    if parsed.component is not None or parsed.coordinate is not None:
        return "scalar"
    # Magnitude collapses vector/tensor structure to a scalar. Complex-valued
    # parts are promoted by the higher-priority caller rule.
    if getattr(parsed, "decomposition", None) == _magnitude_operator():
        return "scalar"
    base = parsed.physical_base
    if base is None:
        # Geometry carriers (position, coordinate, unit_vector, …) have no
        # declared base kind in ISN; codex treats them as scalar-valued.
        return "scalar"
    return "vector" if _base_accepts_magnitude(str(base)) else "scalar"


def derive_kind(name: str) -> str:
    """Return the most specific ``StandardNameKind`` value for *name*.

    Resolution order (first match wins):

    1. **Tensor** — an ISN tensor base (e.g. ``metric_tensor``) or a codex
       ``_tensor`` compound the ISN registry does not model (e.g.
       ``reynolds_stress_tensor``, ``maxwell_stress_tensor``).
    2. **Complex** — ``real_part`` / ``imaginary_part`` mark a component of
       a complex-valued pair.
    3. **Scalar** — a projected component/coordinate axis or a magnitude
       reduction of a vector/tensor (per the ISN parse).
    4. **Vector** — an ISN vector base (``magnetic_field``, ``velocity``,
       ``current_density``, …) with no scalar-reducing projection.
    5. Default → ``scalar``.

    Eigenfunction / spectrum names carry no kind of their own — they are
    semantic categories, structurally scalar (or complex via rule 2, or
    vector via projection topology), matching what the catalog export has
    always emitted for them.
    """
    valid = _load_valid_kinds()
    name_lower = name.lower()

    structural = _isn_structural_kind(name_lower)

    # 1. Tensor — ISN-registered tensor base, or codex `_tensor` compound.
    if "tensor" in valid:
        if structural == "tensor":
            return "tensor"
        # ISN only registers ``metric_tensor``; keep the substring heuristic
        # for stress/pressure tensors ISN does not model as bases.
        if re.search(r"_tensor(?:_|$)", name_lower):
            return "tensor"

    # 2. Complex part.
    if ("real_part" in name_lower or "imaginary_part" in name_lower) and (
        "complex" in valid
    ):
        return "complex"

    # 3. Scalar projection (component / coordinate axis or magnitude).
    if structural == "scalar" and "scalar" in valid:
        return "scalar"

    # 4. Vector base with no scalar-reducing projection.
    if structural == "vector" and "vector" in valid:
        return "vector"

    # 5. Default.
    return "scalar"


# Kind values map to ISN's discriminator one-to-one now that the local
# vocabulary mirrors the ISN Kind enum. The retired extended kinds
# (eigenfunction / spectrum) stay as legacy entries so stale graph data
# written before the vocabulary was unified still collapses to scalar.
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
    """Map a stored kind value to one ISN's discriminator accepts.

    Defaults to ``scalar`` for unknown values so validation never crashes
    on a kind the ISN library doesn't recognise.
    """
    if not kind:
        return "scalar"
    return _ISN_KIND_MAP.get(kind, "scalar")
