"""Lossless access to the public IMAS standard-name grammar.

The legacy ``parse_standard_name`` facade projects an ordered expression into
flat fields and can therefore reject a valid expression tree.  Pipeline
validity and canonicality decisions use this module instead: strict public
parsing produces the lossless IR, the public composer renders that IR, and the
rendered spelling must equal the input exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from imas_standard_names import StandardNameIR, compose, parse


@dataclass(frozen=True, slots=True)
class ParsedCanonicalName:
    """A canonical name paired with its lossless ordered grammar tree."""

    name: str
    ir: StandardNameIR


def parse_canonical_name(name: str) -> ParsedCanonicalName:
    """Strictly parse *name* and require exact lossless canonical equality."""
    result = parse(name, strict=True)
    canonical = compose(result.ir)
    if canonical != name:
        raise ValueError(
            f"name {name!r} is not canonical; ordered grammar renders {canonical!r}"
        )
    return ParsedCanonicalName(name=name, ir=result.ir)


def compose_canonical_ir(ir: StandardNameIR) -> str:
    """Compose *ir* and prove the emitted spelling through the strict parser."""
    name = compose(ir)
    try:
        parse_canonical_name(name)
        return name
    except ValueError as exc:
        # The public strict parser can supply the unique canonical flat
        # segment ordering for an otherwise structured IR (for example a
        # projection combined with a bare-prefix transformation). Accept only
        # that parser-owned spelling and prove it through the same strict gate.
        canonical = getattr(exc, "canonical_form", None)
        if not canonical:
            raise
        return parse_canonical_name(canonical).name


def is_canonical_name(name: str) -> bool:
    """Return whether *name* is a strict, losslessly canonical grammar name."""
    try:
        parse_canonical_name(name)
    except (TypeError, ValueError):
        return False
    return True
