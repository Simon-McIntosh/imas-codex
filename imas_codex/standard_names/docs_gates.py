"""Deterministic content gates for standard-name documentation."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from imas_standard_names.grammar.terms import AppliedConventions, select_conventions

from imas_codex.graph.models import COCOSLabelTransformation
from imas_codex.standard_names.audits import (
    _dimension_container,
    _dimensions_overlap,
    _unit_dimensions,
    latex_def_check,
)
from imas_codex.standard_names.doc_links import (
    NameLink,
    find_name_references,
    find_prose_bracket_phrases,
    find_self_links,
)
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
            outcome=DocumentationGateOutcome.NOT_EVALUABLE,
            reason="relation cannot bind the DD quantity symbol to its declared unit",
        )
    bindings = _symbol_unit_bindings(text)
    bindings[subject] = declared_unit
    left_dimensions = _expression_dimensions(relation[0], bindings)
    right_dimensions = _expression_dimensions(relation[1], bindings)
    if left_dimensions is None or right_dimensions is None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.NOT_EVALUABLE,
            reason="relation cannot bind every symbol to a stated unit",
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


def _bracketed_prose_phrase(text: str) -> str | None:
    """Return the first bracketed phrase that names no standard name, if any.

    A bracket that holds notation — a unit such as ``[m^-3]`` or ``[K]``, a
    numeric interval such as ``[0, 1]`` — or that sits inside mathematics or
    code, or that is the label of an already formed Markdown link, is not a
    documentation prose bracket and raises nothing here.
    """

    phrases = find_prose_bracket_phrases(text)
    return phrases[0] if phrases else None


def _self_referential_link(text: str, documenting_name: str | None) -> NameLink | None:
    """Return the first name link pointing at the documented name itself."""

    if not documenting_name:
        return None
    links = find_self_links(text, documenting_name)
    return links[0] if links else None


def _link_hygiene_result(
    text: str,
    documenting_name: str | None = None,
) -> DocumentationGateResult:
    """Report the first link defect in *text*, naming the offending construct.

    A link is defective when its target syntax is unsupported or its label is
    empty, when a bare ``[identity]`` bracket is written instead of a link,
    when a bracketed phrase stands where a link belongs, or when a ``name:``
    link targets the documented name itself. The reason carries the offending
    phrase or link so a refiner can act on it without re-reading the document.
    """

    for match in _MARKDOWN_LINK_RE.finditer(text):
        label, target = match.groups()
        if not label.strip() or not _VALID_LINK_TARGET_RE.fullmatch(target.strip()):
            return DocumentationGateResult(
                outcome=DocumentationGateOutcome.FAIL,
                reason=(
                    "link has an unsupported target or an empty label: "
                    f"[{label}]({target})"
                ),
            )
    without_links = _MARKDOWN_LINK_RE.sub("", text)
    bare = next(
        (
            reference
            for reference in find_name_references(without_links)
            if reference.syntax == "bare_bracket"
        ),
        None,
    )
    if bare is not None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason=(
                f"bare bracket names a standard name but carries no link: [{bare.name}]"
            ),
        )
    phrase = _bracketed_prose_phrase(text)
    if phrase is not None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason=f"bracketed phrase names no standard name: [{phrase}]",
        )
    self_link = _self_referential_link(text, documenting_name)
    if self_link is not None:
        return DocumentationGateResult(
            outcome=DocumentationGateOutcome.FAIL,
            reason=(
                f"link '{self_link.label}' targets the documented name itself: "
                f"name:{self_link.target}"
            ),
        )
    return DocumentationGateResult(
        outcome=DocumentationGateOutcome.PASS,
        reason="all links resolve through supported target syntax",
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


class HouseStyleFindingClass(StrEnum):
    """The conformance finding classes a written name can raise."""

    FORBIDDEN_EQUATION_VARIANT = "forbidden_equation_variant"
    FORBIDDEN_SYMBOL_SPELLING = "forbidden_symbol_spelling"
    FORBIDDEN_LOCUS_GLOSS = "forbidden_locus_gloss"
    LOCUS_DEFINITION = "locus_definition"
    TEMPLATE_OPENING = "template_opening"
    TEMPLATE_NOTE = "template_note"
    NON_CANONICAL_EQUATION = "non_canonical_equation"


@dataclass(frozen=True)
class HouseStyleFinding:
    """One conformance finding against a selected registry entry."""

    finding_class: HouseStyleFindingClass
    registry_entry: str
    detail: str


_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")


def _template_opening_pattern(template: str) -> re.Pattern[str] | None:
    """Compile a documented template into an opening pattern.

    Each ``{placeholder}`` is replaced by a wildcard, so the pattern checks the
    fixed wording an opening must carry without binding a substitution the
    registry does not source.
    """
    literal = _TEMPLATE_PLACEHOLDER_RE.sub(".+", template.strip())
    if not literal:
        return None
    try:
        return re.compile(literal, re.DOTALL)
    except re.error:
        return None


def _opening_line(text: str) -> str:
    """Return the first non-empty line, the opening the template governs."""

    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


@dataclass(frozen=True)
class RegistryTemplateTexts:
    """Selected template text and its presence in one documentation entry."""

    opening: str
    candidate_openings: tuple[tuple[Any, str], ...]
    opening_matches: bool
    rendered_texts: tuple[str, ...]
    missing_notes: tuple[tuple[Any, str], ...]


def registry_template_texts(
    templates: tuple[Any, ...], documentation: str
) -> RegistryTemplateTexts:
    """Find selected template openings and notes carried by documentation."""

    opening = _opening_line(documentation)
    lines = tuple(line.strip() for line in documentation.splitlines())
    candidates: list[tuple[Any, str]] = []
    opening_matches = False
    rendered: list[str] = []
    missing_notes: list[tuple[Any, str]] = []
    seen_notes: set[str] = set()
    for template in templates:
        pattern = _template_opening_pattern(template.documentation)
        if pattern is not None:
            candidates.append((template, template.documentation))
            opening_matches |= pattern.match(opening) is not None
            if pattern.fullmatch(opening):
                rendered.append(opening)
        note = str(getattr(template, "note", None) or "").strip()
        if not note:
            continue
        note_pattern = _template_opening_pattern(note)
        if note_pattern is not None:
            rendered.extend(line for line in lines if note_pattern.fullmatch(line))
        if note not in seen_notes:
            seen_notes.add(note)
            if note not in documentation:
                missing_notes.append((template, note))
    return RegistryTemplateTexts(
        opening=opening,
        candidate_openings=tuple(candidates),
        opening_matches=opening_matches,
        rendered_texts=tuple(rendered),
        missing_notes=tuple(missing_notes),
    )


def _display_math_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _DISPLAY_MATH_RE.finditer(text)]


def _forbidden_variant_findings(
    concepts: tuple[Any, ...],
    text: str,
) -> list[HouseStyleFinding]:
    """Flag a forbidden variant of a selected concept, named by its locus.

    A variant matching inside a display equation is a written equation the
    registry forbids; one matching in prose is a forbidden gloss of a concept.
    """

    findings: list[HouseStyleFinding] = []
    spans = _display_math_spans(text)
    for concept in concepts:
        for pattern in concept.forbidden_variants:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                continue
            if match is None:
                continue
            inside_equation = any(start <= match.start() < end for start, end in spans)
            finding_class = (
                HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT
                if inside_equation
                else HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS
            )
            findings.append(
                HouseStyleFinding(
                    finding_class=finding_class,
                    registry_entry=f"concept:{concept.token}",
                    detail=(
                        f"'{match.group(0).strip()}' is a forbidden variant of "
                        f"'{concept.phrase}'"
                    ),
                )
            )
            break
    return findings


def _locus_registry_tokens() -> frozenset[str]:
    """The tokens the locus registry defines, so every locus is covered."""

    from imas_standard_names.grammar.vocab_loaders import load_locus_registry

    return frozenset(load_locus_registry().loci)


def locus_definition_sentence(concept: Any) -> str:
    """Render a locus's defining sentence from its registry entry.

    The sentence is the locus registry's ``definition``, followed by the
    conventions entry's canonical equation and the symbols that equation uses.
    The registry, not the name, supplies every part, so the sentence reads the
    same for each name evaluated at the locus whatever its subject.
    """

    definition = concept.definition.strip()
    if not definition.endswith("."):
        definition = f"{definition}."
    equation = concept.equation.strip()
    if not equation:
        return definition
    clause = f"Where ${equation}$"
    if concept.symbols:
        symbols = "; ".join(
            f"${symbol.symbol}$ is {symbol.meaning}" for symbol in concept.symbols
        )
        clause = f"{clause}, {symbols}."
    else:
        clause = f"{clause}."
    return f"{definition} {clause}"


def _collapsed(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Determiners are stripped before the word that governs a locus phrase is
# read, so ``the plasma boundary`` and ``the plasma-boundary surface`` both
# leave the same governor to inspect.
_LOCUS_DETERMINERS = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "its", "their", "any"}
)

# A phrase governed by one of these binds the locus into a larger noun phrase,
# so it names where a quantity is evaluated rather than what the locus is.
_LOCUS_PREPOSITIONS = frozenset(
    {
        "at",
        "of",
        "on",
        "in",
        "into",
        "near",
        "around",
        "to",
        "from",
        "by",
        "for",
        "through",
        "along",
        "onto",
        "between",
        "across",
        "within",
        "against",
        "beyond",
        "below",
        "above",
        "over",
        "under",
        "outside",
        "inside",
        "beneath",
        "adjacent",
    }
)

_LOCUS_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]*")


def _defines_locus(sentence: str, subject_pattern: re.Pattern[str]) -> bool:
    """Whether the locus phrase is the subject a copula predicates on.

    A sentence defines the locus when the locus phrase is what the copula
    asserts about — ``The plasma boundary is …``, ``The plasma-boundary surface
    is …``. A phrase bound into a larger noun phrase by a preposition —
    ``the density at the plasma boundary is …`` — evaluates a quantity there
    and states nothing about what the locus is, so it is not a second
    definition and is left alone.
    """

    for match in subject_pattern.finditer(sentence):
        preceding = _LOCUS_WORD_RE.findall(sentence[: match.start()])
        if preceding and preceding[-1].lower() in _LOCUS_DETERMINERS:
            preceding.pop()
        if preceding and preceding[-1].lower() in _LOCUS_PREPOSITIONS:
            continue
        return True
    return False


def _locus_definition_findings(
    concepts: tuple[Any, ...],
    text: str,
) -> list[HouseStyleFinding]:
    """Flag a locus sentence that departs from the locus registry's definition.

    A locus is defined once, in the locus registry. A name evaluated at it may
    and does state that definition, but a sentence whose subject is the locus
    and which defines it in other words is a second definition and is refined
    until it carries the registry's. A sentence that merely places the locus in
    a prepositional phrase — naming where a quantity is evaluated rather than
    what the locus is — is not a definition and is left alone. The locus's
    forbidden variants arrive through the concept's variants and are not
    reported again here.
    """

    locus_tokens = _locus_registry_tokens()
    sentences = re.split(r"(?<=[.;])\s+", text)
    findings: list[HouseStyleFinding] = []
    for concept in concepts:
        if concept.token not in locus_tokens:
            continue
        definition = _collapsed(concept.definition)
        if not definition:
            continue
        phrase = r"[\s-]+".join(
            re.escape(word) for word in concept.phrase.split() if word
        )
        if not phrase:
            continue
        subject = re.compile(
            rf"(?<![\w-]){phrase}(?!-)(?:\s+[A-Za-z][A-Za-z-]*)*\s+(?:is|are)\b",
            re.IGNORECASE,
        )
        for sentence in sentences:
            if definition in _collapsed(sentence):
                continue
            if not _defines_locus(sentence, subject):
                continue
            findings.append(
                HouseStyleFinding(
                    finding_class=HouseStyleFindingClass.LOCUS_DEFINITION,
                    registry_entry=f"concept:{concept.token}",
                    detail=(
                        f"a sentence defining '{concept.phrase}' does not carry "
                        f"the locus registry definition '{concept.definition}'"
                    ),
                )
            )
            break
    return findings


_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")

# Commands that size, space or decorate an expression rather than name an
# operation, so two spellings of one equation differ only in them.
_LATEX_LAYOUT_COMMANDS = frozenset(
    {
        "left",
        "right",
        "big",
        "Big",
        "bigg",
        "Bigg",
        "displaystyle",
        "frac",
        "dfrac",
        "tfrac",
        "mathrm",
        "mathbf",
        "mathcal",
        "mathit",
        "text",
        "operatorname",
        "bar",
        "hat",
        "tilde",
        "vec",
        "overline",
        "quad",
        "qquad",
        "lvert",
        "rvert",
        "cdot",
        "times",
    }
)
_LATEX_COMMAND_ALIASES = {"oint": "int", "oiint": "int", "iint": "int"}


def _equation_operation_commands(equation: str, symbols: set[str]) -> set[str]:
    """Name the operations a LaTeX expression performs, spelling aside.

    A command that is part of a declared symbol, or that only lays the
    expression out, is not an operation.
    """

    return {
        _LATEX_COMMAND_ALIASES.get(command, command)
        for command in _LATEX_COMMAND_RE.findall(equation)
        if command not in symbols and command not in _LATEX_LAYOUT_COMMANDS
    }


_LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\S")
_LATEX_SPACING_TOKENS = frozenset(
    {"\\,", "\\;", "\\:", "\\!", "\\ ", "\\left", "\\right", "\\displaystyle"}
)
_TOKEN_END = "\x1f"


def _equation_tokens(equation: str) -> list[str]:
    """Split LaTeX into tokens, dropping whitespace and spacing or sizing commands."""

    return [
        token
        for token in _LATEX_TOKEN_RE.findall(equation)
        if token not in _LATEX_SPACING_TOKENS
    ]


def _canonical_equation_pattern(concept: Any) -> re.Pattern[str]:
    """Compile a concept's canonical equation into a pattern over its tokens.

    The canonical equation fixes every operator, delimiter and ordering; the
    symbols the registry declares for it stand for the quantity being written
    about, so each matches any run of tokens. Spacing and sizing commands do
    not take part.
    """

    symbol_tokens = [
        tuple(_equation_tokens(symbol.symbol))
        for symbol in concept.symbols
        if symbol.symbol.strip()
    ]
    tokens = _equation_tokens(concept.equation)
    pieces: list[str] = []
    index = 0
    while index < len(tokens):
        matched = next(
            (
                symbol
                for symbol in sorted(symbol_tokens, key=len, reverse=True)
                if symbol and tuple(tokens[index : index + len(symbol)]) == symbol
            ),
            None,
        )
        if matched is not None:
            pieces.append(f"(?:[^{_TOKEN_END}]+{_TOKEN_END})+?")
            index += len(matched)
            continue
        pieces.append(re.escape(tokens[index]) + _TOKEN_END)
        index += 1
    return re.compile("".join(pieces))


def _is_canonical_equation(concept: Any, equation: str) -> bool:
    tokens = _equation_tokens(equation.strip().rstrip(".,;"))
    joined = "".join(token + _TOKEN_END for token in tokens)
    return _canonical_equation_pattern(concept).fullmatch(joined) is not None


def _concept_equation_signature(concept: Any) -> set[str]:
    """Return the operation commands that identify an equation of a concept.

    The left-hand side names the operator being applied, so it identifies the
    concept whatever form the right-hand side is written in. A concept whose
    left-hand side carries no operation command is identified by its whole
    canonical equation.
    """

    symbols: set[str] = set()
    for symbol in concept.symbols:
        symbols.update(_LATEX_COMMAND_RE.findall(symbol.symbol))
    left, _, _ = concept.equation.partition("=")
    signature = _equation_operation_commands(left, symbols)
    if signature:
        return signature
    return _equation_operation_commands(concept.equation, symbols)


def _non_canonical_equation_findings(
    concepts: tuple[Any, ...],
    text: str,
) -> list[HouseStyleFinding]:
    """Flag a display equation for a concept written other than canonically.

    A concept has one canonical equation, so a display equation performing its
    operation in any other form is a second form of it. The canonical form is
    the converged statement and is never reported, whatever the entry's
    ``restates_equation`` value. An equation a forbidden variant already
    matches is reported as that variant and not a second time.
    """

    findings: list[HouseStyleFinding] = []
    for concept in concepts:
        signature = _concept_equation_signature(concept)
        if not signature:
            continue
        symbols = {
            command
            for symbol in concept.symbols
            for command in _LATEX_COMMAND_RE.findall(symbol.symbol)
        }
        for equation in _DISPLAY_MATH_RE.findall(text):
            if not signature <= _equation_operation_commands(equation, symbols):
                continue
            if _is_canonical_equation(concept, equation):
                continue
            if any(
                _variant_matches(pattern, f"$${equation}$$")
                for pattern in concept.forbidden_variants
            ):
                continue
            findings.append(
                HouseStyleFinding(
                    finding_class=HouseStyleFindingClass.NON_CANONICAL_EQUATION,
                    registry_entry=f"concept:{concept.token}",
                    detail=(
                        f"a display equation for '{concept.phrase}' is not "
                        f"written in its canonical form '{concept.equation}'"
                    ),
                )
            )
            break
    return findings


def _variant_matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return False


def _forbidden_symbol_findings(
    notation: tuple[Any, ...],
    text: str,
) -> list[HouseStyleFinding]:
    """Flag a forbidden spelling of a selected quantity's canonical symbol."""

    findings: list[HouseStyleFinding] = []
    for entry in notation:
        for spelling in entry.forbidden_spellings:
            if spelling and spelling in text:
                findings.append(
                    HouseStyleFinding(
                        finding_class=HouseStyleFindingClass.FORBIDDEN_SYMBOL_SPELLING,
                        registry_entry=f"notation:{entry.token}",
                        detail=(
                            f"'{spelling}' is a forbidden spelling of the "
                            f"canonical symbol '{entry.symbol}'"
                        ),
                    )
                )
                break
    return findings


def _template_findings(
    templates: tuple[Any, ...],
    documentation: str,
) -> list[HouseStyleFinding]:
    """Flag a documentation body that departs from its selected templates.

    One pattern can select several templates, and an opening need only follow
    one of them. Only an opening matching none is a finding, so a name with two
    selected templates is not reported for the one it did not follow. Each
    distinct note a selected template carries is required as well: the notes
    are what keep near-duplicate names apart, so documentation omitting one is
    reported.
    """

    findings: list[HouseStyleFinding] = []
    selected = registry_template_texts(templates, documentation)
    if selected.candidate_openings and not selected.opening_matches:
        required = "; ".join(text for _, text in selected.candidate_openings)
        findings.append(
            HouseStyleFinding(
                finding_class=HouseStyleFindingClass.TEMPLATE_OPENING,
                registry_entry=f"template:{selected.candidate_openings[0][0].base}",
                detail=(
                    f"documentation opens as '{selected.opening[:60]}', which does not "
                    f"follow any selected template: {required}"
                ),
            )
        )
    for template, note in selected.missing_notes:
        findings.append(
            HouseStyleFinding(
                finding_class=HouseStyleFindingClass.TEMPLATE_NOTE,
                registry_entry=f"template:{template.base}",
                detail=(
                    "documentation omits the note its selected template "
                    f"carries: {note}"
                ),
            )
        )
    return findings


def house_style_findings(
    conventions: AppliedConventions,
    *,
    description: str = "",
    documentation: str = "",
) -> list[HouseStyleFinding]:
    """Return the registry conformance findings for one written name.

    The registry supplies what a name's own gate cannot decide: the canonical
    equation form and its forbidden variants, the canonical symbol per quantity,
    and the opening and note each pattern template requires. The forbidden locus
    gloss arrives through the same concept variants, so a gloss equating two loci
    is reported without a separate lookup. A locus is defined once in its
    registry, so a sentence defining it in other words is reported as well.
    """

    prose = f"{description}\n\n{documentation}"
    return [
        *_forbidden_variant_findings(conventions.concepts, prose),
        *_locus_definition_findings(conventions.concepts, prose),
        *_forbidden_symbol_findings(conventions.notation, prose),
        *_template_findings(conventions.templates, documentation),
        *_non_canonical_equation_findings(conventions.concepts, prose),
    ]


def conformance_findings(
    name: str,
    *,
    description: str = "",
    documentation: str = "",
) -> list[HouseStyleFinding]:
    """Select a name's registry entries and return its conformance findings."""

    conventions = select_conventions(name)
    return house_style_findings(
        conventions,
        description=description,
        documentation=documentation,
    )


def score_documentation(
    documentation: str,
    *,
    physics_context: DocumentationPhysicsContext | None = None,
    documenting_name: str | None = None,
) -> DocumentationGateScore:
    """Score one documentation body against the live deterministic gates.

    Equation and symbol gates describe observable content. Sign conventions
    are required for COCOS-sensitive transformation classes, forbidden for the
    invariant class, and unevaluable when the transformation class is absent.
    Catalog-visible text must not expose COCOS numbers or transformation labels.
    The hygiene gate reads *documenting_name*, when the caller supplies it, to
    catch a link that points back at the name it documents.
    """

    text = documentation.strip()
    words = _word_count(text)

    predicates = {
        "symbol_definitions": _symbols_are_defined(text),
        "relationship_link": _has_relationship_link(text),
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
        "link_hygiene": _link_hygiene_result(text, documenting_name),
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
