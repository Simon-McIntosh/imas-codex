"""Lossless access to the public IMAS standard-name grammar.

The legacy ``parse_standard_name`` facade projects an ordered expression into
flat fields and can therefore reject a valid expression tree.  Pipeline
validity and canonicality decisions use this module instead: strict public
parsing produces the lossless IR, the public composer renders that IR, and the
rendered spelling must equal the input exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imas_standard_names import StandardNameIR, compose, parse


@dataclass(frozen=True, slots=True)
class ParsedCanonicalName:
    """A canonical name paired with its lossless ordered grammar tree."""

    name: str
    ir: StandardNameIR


def _ir_payload(ir: StandardNameIR) -> dict[str, Any]:
    """Return a stable public-model representation for semantic comparison."""
    return ir.model_dump(mode="json")


def _has_scoped_operator_tree(ir: StandardNameIR) -> bool:
    """Return whether *ir* contains an operator with explicit operands."""
    return any(operator.args for operator in ir.operators)


def _flat_canonicalization_is_semantic(
    expected: StandardNameIR,
    parsed: StandardNameIR,
) -> bool:
    """Allow one token-preserving public normalization of a flat bare prefix.

    The public parser can absorb a bare-prefix operator token into either the
    projection axis or the qualifier sequence.  Every untouched IR field and
    every remaining operator must be identical.  Binary or explicit-argument
    trees never take this compatibility path: their recursive IR is the
    semantic contract.
    """
    if _has_scoped_operator_tree(expected) or _has_scoped_operator_tree(parsed):
        return False

    expected_data = _ir_payload(expected)
    parsed_data = _ir_payload(parsed)
    if any(
        expected_data[field] != parsed_data[field]
        for field in ("base", "locus", "mechanism")
    ):
        return False

    if not expected.operators:
        return False
    operator = expected.operators[-1]
    if (
        operator.kind.value != "unary_prefix"
        or not operator.bare_prefix
        or operator.args
    ):
        return False
    remaining = [item.model_dump(mode="json") for item in expected.operators[:-1]]
    parsed_operators = [item.model_dump(mode="json") for item in parsed.operators]
    if remaining != parsed_operators:
        return False

    expected_projection = expected_data["projection"]
    parsed_projection = parsed_data["projection"]
    expected_qualifiers = expected_data["qualifiers"]
    parsed_qualifiers = parsed_data["qualifiers"]

    projection_absorbed = (
        expected_projection is not None
        and parsed_projection is not None
        and parsed_projection["shape"] == expected_projection["shape"]
        and parsed_projection["axis"] == f"{operator.op}_{expected_projection['axis']}"
        and parsed_qualifiers == expected_qualifiers
    )
    if projection_absorbed:
        return True

    if parsed_projection != expected_projection:
        return False
    for qualifier_index, qualifier in enumerate(parsed_qualifiers):
        if qualifier != {"token": operator.op, "category": None}:
            continue
        without_operator = [
            item
            for position, item in enumerate(parsed_qualifiers)
            if position != qualifier_index
        ]
        if without_operator == expected_qualifiers:
            return True
    return False


def _require_semantic_ir(
    expected: StandardNameIR,
    parsed: StandardNameIR,
    *,
    name: str,
) -> None:
    """Require parse-after-compose to preserve recursive operator semantics."""
    if _ir_payload(expected) == _ir_payload(parsed):
        return
    if _flat_canonicalization_is_semantic(expected, parsed):
        return
    raise ValueError(f"name {name!r} round-trips textually but changes the semantic IR")


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
    """Compose *ir* and prove strict spelling and semantic IR preservation."""
    name = compose(ir)
    try:
        parsed = parse_canonical_name(name)
    except ValueError as exc:
        # The public strict parser can supply the unique canonical flat
        # segment ordering for an otherwise structured IR (for example a
        # projection combined with a bare-prefix transformation). Accept only
        # that parser-owned spelling and prove it through the same strict gate.
        canonical = getattr(exc, "canonical_form", None)
        if not canonical:
            raise
        parsed = parse_canonical_name(canonical)
        name = parsed.name
    _require_semantic_ir(ir, parsed.ir, name=name)
    return name


def is_canonical_name(name: str) -> bool:
    """Return whether *name* is a strict, losslessly canonical grammar name."""
    try:
        parse_canonical_name(name)
    except (TypeError, ValueError):
        return False
    return True
