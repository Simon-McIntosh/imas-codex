"""Deterministic content gates for standard-name documentation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from imas_codex.standard_names.audits import latex_def_check
from imas_codex.standard_names.doc_links import find_name_references

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
_COCOS_NUMBER_RE = re.compile(r"\bCOCOS(?:[- ]?\d+)\b", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[(?:condition|physical condition|quantity)\]", re.I)
_VALID_LINK_TARGET_RE = re.compile(
    r"(?:name:[a-z0-9_]+|#[a-z0-9_]+|dd:[a-z0-9_/]+|https?://\S+)$"
)
_NAME_LINK_TARGET_RE = re.compile(r"(?:name:|#)[a-z0-9_]+$")


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


def _has_defining_equation(text: str) -> bool:
    equations = [equation.strip() for equation in _DISPLAY_MATH_RE.findall(text)]
    return len(equations) == 1 and bool(_DEFINING_RELATION_RE.search(equations[0]))


def _symbols_are_defined(text: str) -> bool:
    return not latex_def_check({"documentation": text})


def _has_relationship_link(text: str) -> bool:
    """Return whether the documentation links to another standard name."""

    return any(
        _NAME_LINK_TARGET_RE.fullmatch(target.strip())
        for _, target in _MARKDOWN_LINK_RE.findall(text)
    )


def _valid_sign_convention(text: str) -> bool:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    sign_paragraphs = [part for part in paragraphs if "sign convention" in part.lower()]
    if not sign_paragraphs:
        return True
    if len(sign_paragraphs) != 1 or sign_paragraphs[0] != paragraphs[-1]:
        return False
    sign = sign_paragraphs[0]
    return bool(
        _VALID_SIGN_RE.fullmatch(sign)
        and not _COCOS_NUMBER_RE.search(sign)
        and not _PLACEHOLDER_RE.search(sign)
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
    are conditional: absence passes because applicability requires structured
    quantity metadata, while any convention present must use the canonical
    standalone final paragraph.
    """

    text = documentation.strip()
    words = _word_count(text)

    predicates = {
        "defining_equation": _has_defining_equation(text),
        "symbol_definitions": _symbols_are_defined(text),
        "relationship_link": _has_relationship_link(text),
        "sign_convention": _valid_sign_convention(text),
        "link_hygiene": _links_are_hygienic(text),
        "minimum_word_count": words >= MIN_DOCUMENTATION_WORDS,
    }
    assert tuple(predicates) == DOCUMENTATION_GATE_NAMES
    reasons = {
        "defining_equation": (
            "exactly one display equation contains a defining relation",
            "documentation lacks exactly one display equation with a defining relation",
        ),
        "symbol_definitions": (
            "every mathematical symbol has a prose definition",
            "at least one mathematical symbol lacks a prose definition",
        ),
        "relationship_link": (
            "documentation contains a standard-name relationship link",
            "documentation contains no standard-name relationship link",
        ),
        "sign_convention": (
            "sign-convention prose is absent or has canonical placement and form",
            "sign-convention prose has noncanonical placement or form",
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
        gate: _predicate_result(
            passed,
            pass_reason=reasons[gate][0],
            fail_reason=reasons[gate][1],
        )
        for gate, passed in predicates.items()
    }
    return DocumentationGateScore(
        gate_vector=gates,
        word_count=words,
        physics_context=physics_context,
    )
