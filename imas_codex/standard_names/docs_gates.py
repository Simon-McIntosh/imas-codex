"""Deterministic content gates for standard-name documentation."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from imas_codex.graph.models import COCOSLabelTransformation
from imas_codex.standard_names.audits import (
    _dimension_container,
    _dimensions_overlap,
    _unit_dimensions,
    latex_def_check,
)
from imas_codex.standard_names.doc_links import find_name_references
from imas_codex.units import resolve_dd_unit
from imas_codex.units.dd_unit_exceptions import canonical_or_none

MIN_DOCUMENTATION_WORDS = 40

NORMATIVE_GATE_NAMES: tuple[str, ...] = (
    "defining_equation",
    "symbol_definitions",
    "relationship_link",
    "sign_convention",
)

DOCUMENTATION_GATE_NAMES: tuple[str, ...] = (
    *NORMATIVE_GATE_NAMES,
    "link_hygiene",
    "minimum_word_count",
)

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'\N{RIGHT SINGLE QUOTATION MARK}][A-Za-z]+)*")

_DEFINING_RELATION_RE = re.compile(r"=|\\equiv\b|\\propto\b|\\int\b")
_VALID_SIGN_RE = re.compile(
    r"^Sign convention: Positive (?:when|for) .+\.$|"
    r"^Sign convention: Positive [A-Za-z$\\].+\.$",
    re.DOTALL,
)
_COCOS_NUMBER_RE = re.compile(
    r"\bCOCOS(?:[- ]*(?:convention|number|version))?[- ]*#?\d+\b",
    re.IGNORECASE,
)
_COCOS_TRANSFORMATION_LABEL_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(re.escape(label.value) for label in COCOSLabelTransformation)})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"\[(?:condition|physical condition|quantity)\]", re.I)
_VALID_LINK_TARGET_RE = re.compile(
    r"(?:name:[a-z0-9_]+|#[a-z0-9_]+|dd:[a-z0-9_/]+|https?://\S+)$"
)
_NAME_LINK_TARGET_RE = re.compile(r"(?:name:|#)[a-z0-9_]+$")
_LATEX_EXPONENT_RE = re.compile(r"\$\^\{?(-?\d+)\}?\$")
_UNIT_AFTER_IN_RE = re.compile(
    r"\b(?:in|measured in|unit(?:s)? of)\s+"
    r"([A-Za-z\N{GREEK CAPITAL LETTER OMEGA}]+(?:[./^*-][A-Za-z0-9\N{GREEK CAPITAL LETTER OMEGA}+^.-]+)*)",
    re.IGNORECASE,
)
_UNIT_AFTER_VALUE_RE = re.compile(
    r"(?:\d+(?:\.\d+)?(?:\s*\\times\s*10\^\{?-?\d+\}?)?\s+)"
    r"([A-Za-z\N{GREEK CAPITAL LETTER OMEGA}]+(?:[./^*-][A-Za-z0-9\N{GREEK CAPITAL LETTER OMEGA}+^.-]+)*)"
)


class DocumentationGateOutcome(StrEnum):
    """One deterministic gate result, including unavailable authority."""

    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True)
class DocumentationGateResult:
    """One gate outcome with its deterministic explanation."""

    outcome: DocumentationGateOutcome
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", DocumentationGateOutcome(self.outcome))
        if not self.reason.strip():
            raise ValueError("Documentation gate result requires a reason")


@dataclass(frozen=True)
class DocumentationPhysicsContext:
    """Authoritative facts available to documentation physics checks."""

    dd_path: str | None = None
    declared_unit: str | None = None
    cocos_transformation_type: str | None = None
    cocos_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DocumentationGateScore:
    """Gate outcome vector and aggregate counts for one documentation string."""

    gate_vector: dict[str, DocumentationGateResult]
    word_count: int
    physics_context: DocumentationPhysicsContext | None = None

    def __post_init__(self) -> None:
        if tuple(self.gate_vector) != DOCUMENTATION_GATE_NAMES:
            raise ValueError(
                "Documentation gate vector must contain every gate in canonical order"
            )
        if not all(
            isinstance(result, DocumentationGateResult)
            for result in self.gate_vector.values()
        ):
            raise TypeError("Documentation gate vector values must be gate results")

    @property
    def passed_count(self) -> int:
        """Return the number of authoritative checks that passed."""

        return sum(
            result.outcome is DocumentationGateOutcome.PASS
            for result in self.gate_vector.values()
        )

    @property
    def failed_count(self) -> int:
        """Return the number of authoritative contradictions."""

        return sum(
            result.outcome is DocumentationGateOutcome.FAIL
            for result in self.gate_vector.values()
        )

    @property
    def not_evaluable_count(self) -> int:
        """Return the number of checks lacking authority for evaluation."""

        return sum(
            result.outcome is DocumentationGateOutcome.NOT_EVALUABLE
            for result in self.gate_vector.values()
        )

    @property
    def evaluable_count(self) -> int:
        """Return the denominator for pass-rate aggregation."""

        return self.passed_count + self.failed_count

    @property
    def total_count(self) -> int:
        """Return the number of gate outcomes, including unavailable checks."""

        return len(self.gate_vector)


def _without_markup(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), text)
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(" ", text)
    return re.sub(r"[`*_#>]", " ", text)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(_without_markup(text)))


def _braced_group(text: str, start: int) -> tuple[str, int] | None:
    """Return one balanced LaTeX braced group and its exclusive end."""

    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _expand_latex_fractions(expression: str) -> str | None:
    """Translate balanced ``\\frac`` forms into ordinary division."""

    output: list[str] = []
    position = 0
    while position < len(expression):
        if not expression.startswith(r"\frac", position):
            output.append(expression[position])
            position += 1
            continue
        numerator = _braced_group(expression, position + len(r"\frac"))
        if numerator is None:
            return None
        denominator = _braced_group(expression, numerator[1])
        if denominator is None:
            return None
        expanded_numerator = _expand_latex_fractions(numerator[0])
        expanded_denominator = _expand_latex_fractions(denominator[0])
        if expanded_numerator is None or expanded_denominator is None:
            return None
        output.append(f"(({expanded_numerator})/({expanded_denominator}))")
        position = denominator[1]
    return "".join(output)


def _unit_from_definition(clause: str) -> str | None:
    """Extract one explicitly stated symbol unit from its prose clause."""

    normalized = _LATEX_EXPONENT_RE.sub(r"^\1", clause)
    lowered = normalized.lower()
    if "dimensionless" in lowered or "unit vector" in lowered:
        return "1"
    candidates = [
        *(match.group(1) for match in _UNIT_AFTER_IN_RE.finditer(normalized)),
        *(match.group(1) for match in _UNIT_AFTER_VALUE_RE.finditer(normalized)),
    ]
    for candidate in candidates:
        canonical = canonical_or_none(candidate.rstrip(".,;:"))
        if canonical is not None:
            return canonical
    return None


def _symbol_unit_bindings(text: str) -> dict[str, str]:
    """Bind LaTeX symbols to units explicitly stated beside their definitions."""

    text = _LATEX_EXPONENT_RE.sub(r"^\1", text)
    matches = list(_INLINE_MATH_RE.finditer(text))
    bindings: dict[str, str] = {}
    for index, match in enumerate(matches):
        symbol = match.group(1).strip()
        if not symbol or any(marker in symbol for marker in ("=", r"\propto")):
            continue
        clause_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        clause = text[match.end() : clause_end]
        clause = re.split(r"[.\n]", clause, maxsplit=1)[0]
        unit = _unit_from_definition(clause)
        if unit is not None:
            bindings[symbol] = unit
    return bindings


def _subject_symbol(text: str) -> str | None:
    """Return the quantity symbol introduced by the opening prose."""

    for symbol in _INLINE_MATH_RE.findall(text):
        symbol = symbol.strip()
        if re.search(r"[A-Za-z\\]", symbol) and not any(
            marker in symbol for marker in ("=", r"\propto", r"\times")
        ):
            return symbol
    return None


def _latex_expression_tree(
    expression: str,
    bindings: Mapping[str, str],
) -> tuple[ast.Expression, dict[str, str]] | None:
    """Translate a conservative multiplicative LaTeX expression to Python AST."""

    expanded = _expand_latex_fractions(expression)
    if expanded is None:
        return None
    placeholders: dict[str, str] = {}
    for index, (symbol, unit) in enumerate(
        sorted(bindings.items(), key=lambda item: len(item[0]), reverse=True)
    ):
        if symbol not in expanded:
            continue
        placeholder = f"unit_symbol_{index}"
        expanded = expanded.replace(symbol, placeholder)
        placeholders[placeholder] = unit
    expanded = re.sub(r"\\(?:int|oint)\b", "", expanded)
    expanded = expanded.replace(r"\times", "*").replace(r"\cdot", "*")
    expanded = expanded.replace(r"\left", "").replace(r"\right", "")
    expanded = expanded.replace(r"\,", " ").replace(r"\;", " ")
    expanded = expanded.replace(r"\!", " ").replace(r"\pi", "1")
    expanded = expanded.replace("{", "(").replace("}", ")")
    expanded = re.sub(r"(?<=\w)\s+(?=\w)", "*", expanded)
    expanded = re.sub(r"\s+", "", expanded)
    expanded = re.sub(r"(?<=\))(?=\()", "*", expanded)
    expanded = re.sub(r"(?<=\d)(?=unit_symbol_)", "*", expanded)
    expanded = re.sub(r"(?<=\))(?=unit_symbol_)", "*", expanded)
    expanded = re.sub(r"(?<=unit_symbol_\d)(?=\()", "*", expanded)
    if not expanded or re.search(
        r"\\|[A-Za-z_]", re.sub(r"unit_symbol_\d+", "", expanded)
    ):
        return None
    try:
        tree = ast.parse(expanded, mode="eval")
    except SyntaxError:
        return None
    return tree, placeholders


def _evaluate_dimension_tree(
    node: ast.AST,
    placeholders: Mapping[str, str],
) -> Any | None:
    """Evaluate the dimensional algebra of a restricted expression AST."""

    if isinstance(node, ast.Expression):
        return _evaluate_dimension_tree(node.body, placeholders)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return _dimension_container("dimensionless")
    if isinstance(node, ast.Name) and node.id in placeholders:
        dimensions = _unit_dimensions({placeholders[node.id]})
        if dimensions is None or len(dimensions) != 1:
            return None
        return _dimension_container(next(iter(dimensions)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _evaluate_dimension_tree(node.operand, placeholders)
    if not isinstance(node, ast.BinOp):
        return None
    left = _evaluate_dimension_tree(node.left, placeholders)
    right = _evaluate_dimension_tree(node.right, placeholders)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Add | ast.Sub):
        return left if left == right else None
    if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
        if isinstance(node.right.value, int | float):
            return left**node.right.value
    return None


def _expression_dimensions(
    expression: str,
    bindings: Mapping[str, str],
) -> set[str] | None:
    parsed = _latex_expression_tree(expression, bindings)
    if parsed is None:
        return None
    dimensions = _evaluate_dimension_tree(parsed[0], parsed[1])
    return {str(dimensions)} if dimensions is not None else None


def _defining_equation_result(
    text: str,
    physics_context: DocumentationPhysicsContext | None,
) -> DocumentationGateResult:
    """Compare one stated defining relation with the DD-declared unit."""

    if physics_context is None or not physics_context.declared_unit:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.NOT_EVALUABLE,
            reason="DD-declared unit is unavailable",
        )
    declared_unit = resolve_dd_unit(
        physics_context.dd_path or "",
        physics_context.declared_unit,
    )
    declared_dimensions = (
        _unit_dimensions({declared_unit}) if declared_unit is not None else None
    )
    if declared_dimensions is None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.NOT_EVALUABLE,
            reason="DD-declared unit has no resolvable dimensionality",
        )

    equations = [equation.strip() for equation in _DISPLAY_MATH_RE.findall(text)]
    if len(equations) != 1 or not _DEFINING_RELATION_RE.search(equations[0]):
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="documentation lacks exactly one dimensionally checkable defining relation",
        )
    relation = re.split(r"=|\\equiv\b", equations[0], maxsplit=1)
    if len(relation) != 2:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="defining relation does not state a dimensional equality",
        )
    subject = _subject_symbol(text)
    if subject is None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="documentation does not identify the relation's DD quantity symbol",
        )
    bindings = _symbol_unit_bindings(text)
    bindings[subject] = declared_unit
    left_dimensions = _expression_dimensions(relation[0], bindings)
    right_dimensions = _expression_dimensions(relation[1], bindings)
    if left_dimensions is None or right_dimensions is None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="defining relation cannot reproduce the declared unit from its stated symbol units",
        )
    if _dimensions_overlap(left_dimensions, right_dimensions):
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.PASS,
            reason="defining relation reproduces the DD-declared unit",
        )
    return DocumentationGateResult(
        outcome=DocumentationGateOutcome.FAIL,
        reason="defining relation dimensions contradict the DD-declared unit",
    )


def _symbols_are_defined(text: str) -> bool:
    return not latex_def_check({"documentation": text})


def _has_relationship_link(text: str) -> bool:
    """Return whether the documentation links to another standard name."""

    return any(
        _NAME_LINK_TARGET_RE.fullmatch(target.strip())
        for _, target in _MARKDOWN_LINK_RE.findall(text)
    )


def _sign_convention_result(
    text: str,
    physics_context: DocumentationPhysicsContext | None,
) -> DocumentationGateResult:
    """Require sign prose exactly when authoritative COCOS metadata does."""

    transformation_type = (
        physics_context.cocos_transformation_type
        if physics_context is not None
        else None
    )
    leaked_label = _COCOS_TRANSFORMATION_LABEL_RE.search(text)
    leaked_authoritative_label = transformation_type and re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(transformation_type)}(?![A-Za-z0-9_])",
        text,
        re.IGNORECASE,
    )
    if _COCOS_NUMBER_RE.search(text) or leaked_label or leaked_authoritative_label:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="documentation exposes catalog-internal COCOS metadata",
        )
    if not transformation_type:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.NOT_EVALUABLE,
            reason="COCOS transformation class is unavailable",
        )

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    sign_paragraphs = [part for part in paragraphs if "sign convention" in part.lower()]
    if transformation_type == COCOSLabelTransformation.one_like.value:
        return _predicate_result(
            not sign_paragraphs,
            pass_reason="COCOS-invariant quantity omits sign-convention prose",
            fail_reason="COCOS-invariant quantity states a sign convention",
        )
    if not sign_paragraphs:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="COCOS-sensitive quantity omits a sign convention",
        )
    if len(sign_paragraphs) != 1 or sign_paragraphs[0] != paragraphs[-1]:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason="sign-convention prose has noncanonical placement or form",
        )
    sign = sign_paragraphs[0]
    return _predicate_result(
        bool(_VALID_SIGN_RE.fullmatch(sign) and not _PLACEHOLDER_RE.search(sign)),
        pass_reason="COCOS-sensitive quantity has canonical sign-convention prose",
        fail_reason="sign-convention prose has noncanonical placement or form",
    )


def _links_are_hygienic(text: str) -> bool:
    for match in _MARKDOWN_LINK_RE.finditer(text):
        label, target = match.groups()
        if not label.strip() or not _VALID_LINK_TARGET_RE.fullmatch(target.strip()):
            return False
    without_links = _MARKDOWN_LINK_RE.sub("", text)
    return not any(
        reference.syntax == "bare_bracket"
        for reference in find_name_references(without_links)
    )


def _predicate_result(
    passed: bool,
    *,
    pass_reason: str,
    fail_reason: str,
) -> DocumentationGateResult:
    if passed:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.PASS,
            reason=pass_reason,
        )
    return DocumentationGateResult(
        outcome=DocumentationGateOutcome.FAIL,
        reason=fail_reason,
    )


def score_documentation(
    documentation: str,
    *,
    physics_context: DocumentationPhysicsContext | None = None,
) -> DocumentationGateScore:
    """Score one documentation body against the live deterministic gates.

    Equation and symbol gates describe observable content. Sign conventions
    are required for COCOS-sensitive transformation classes, forbidden for the
    invariant class, and unevaluable when the transformation class is absent.
    Catalog-visible text must not expose COCOS numbers or transformation labels.
    """

    text = documentation.strip()
    words = _word_count(text)

    predicates = {
        "symbol_definitions": _symbols_are_defined(text),
        "relationship_link": _has_relationship_link(text),
        "link_hygiene": _links_are_hygienic(text),
        "minimum_word_count": words >= MIN_DOCUMENTATION_WORDS,
    }
    reasons = {
        "symbol_definitions": (
            "every mathematical symbol has a prose definition",
            "at least one mathematical symbol lacks a prose definition",
        ),
        "relationship_link": (
            "documentation contains a standard-name relationship link",
            "documentation contains no standard-name relationship link",
        ),
        "link_hygiene": (
            "all links resolve through supported target syntax",
            "documentation contains malformed or bare name references",
        ),
        "minimum_word_count": (
            "documentation meets the minimum word count",
            "documentation is below the minimum word count",
        ),
    }
    gates = {
        "defining_equation": _defining_equation_result(text, physics_context),
        "symbol_definitions": _predicate_result(
            predicates["symbol_definitions"],
            pass_reason=reasons["symbol_definitions"][0],
            fail_reason=reasons["symbol_definitions"][1],
        ),
        "relationship_link": _predicate_result(
            predicates["relationship_link"],
            pass_reason=reasons["relationship_link"][0],
            fail_reason=reasons["relationship_link"][1],
        ),
        "sign_convention": _sign_convention_result(text, physics_context),
        "link_hygiene": _predicate_result(
            predicates["link_hygiene"],
            pass_reason=reasons["link_hygiene"][0],
            fail_reason=reasons["link_hygiene"][1],
        ),
        "minimum_word_count": _predicate_result(
            predicates["minimum_word_count"],
            pass_reason=reasons["minimum_word_count"][0],
            fail_reason=reasons["minimum_word_count"][1],
        ),
    }
    assert tuple(gates) == DOCUMENTATION_GATE_NAMES
    return DocumentationGateScore(
        gate_vector=gates,
        word_count=words,
        physics_context=physics_context,
    )
