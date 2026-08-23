"""Deterministic content gates for standard-name documentation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from imas_codex.standard_names.audits import latex_def_check

MIN_DOCUMENTATION_WORDS = 40
MAX_DOCUMENTATION_WORDS = 160

NORMATIVE_GATE_NAMES: tuple[str, ...] = (
    "physical_meaning",
    "defining_equation",
    "symbol_definitions",
    "scope",
    "exclusions_or_distinctions",
    "essential_relationships",
    "sign_convention",
)

DOCUMENTATION_GATE_NAMES: tuple[str, ...] = (
    *NORMATIVE_GATE_NAMES,
    "link_hygiene",
    "minimum_word_count",
    "maximum_word_count",
)

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BARE_NAME_BRACKET_RE = re.compile(r"(?<!\\)\[([a-z][a-z0-9_]+)\](?!\()")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'\N{RIGHT SINGLE QUOTATION MARK}][A-Za-z]+)*")

_DEFINITION_RE = re.compile(
    r"\b(?:is|are|denotes?|represents?|quantifies?|describes?|"
    r"defined as|refers to|corresponds to)\b",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(
    r"\b(?:scope|context|within|inside|outside|between|relative to|"
    r"with respect to|at the|on the|through|boundary|surface|region|domain|"
    r"averaged over|integrated over|summed over)\b",
    re.IGNORECASE,
)
_DISTINCTION_RE = re.compile(
    r"\b(?:excludes?|excluding|does not include|distinct from|rather than|"
    r"in contrast to|boundary|endpoint|normalization|reference|relative to|"
    r"from .{1,80} to)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_RE = re.compile(
    r"\b(?:relates? to|relationship|depends? on|proportional to|"
    r"integral of|gradient of|derivative of|normalized by|defined from|"
    r"obtained from)\b",
    re.IGNORECASE,
)
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


@dataclass(frozen=True)
class DocumentationGateScore:
    """Boolean gate vector and aggregate counts for one documentation string."""

    gate_vector: dict[str, bool]
    passed_count: int
    total_count: int
    word_count: int


def _without_markup(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), text)
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(" ", text)
    return re.sub(r"[`*_#>]", " ", text)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(_without_markup(text)))


def _has_physical_meaning(text: str) -> bool:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return False
    opening = _without_markup(paragraphs[0])
    return len(_WORD_RE.findall(opening)) >= 8 and bool(_DEFINITION_RE.search(opening))


def _has_defining_equation(text: str) -> bool:
    equations = [equation.strip() for equation in _DISPLAY_MATH_RE.findall(text)]
    return len(equations) == 1 and bool(_DEFINING_RELATION_RE.search(equations[0]))


def _symbols_are_defined(text: str) -> bool:
    return not latex_def_check({"documentation": text})


def _has_essential_relationship(text: str) -> bool:
    return bool(_MARKDOWN_LINK_RE.search(text) or _RELATIONSHIP_RE.search(text))


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
    return not _BARE_NAME_BRACKET_RE.search(without_links)


def score_documentation(documentation: str) -> DocumentationGateScore:
    """Score one documentation body against the live deterministic gates.

    Equation and symbol gates describe observable content. Sign conventions
    are conditional: absence passes because applicability requires structured
    quantity metadata, while any convention present must use the canonical
    standalone final paragraph.
    """

    text = documentation.strip()
    words = _word_count(text)
    scope_present = bool(_SCOPE_RE.search(_without_markup(text)))
    distinction_present = bool(_DISTINCTION_RE.search(_without_markup(text)))

    gates = {
        "physical_meaning": _has_physical_meaning(text),
        "defining_equation": _has_defining_equation(text),
        "symbol_definitions": _symbols_are_defined(text),
        "scope": scope_present,
        "exclusions_or_distinctions": distinction_present,
        "essential_relationships": _has_essential_relationship(text),
        "sign_convention": _valid_sign_convention(text),
        "link_hygiene": _links_are_hygienic(text),
        "minimum_word_count": words >= MIN_DOCUMENTATION_WORDS,
        "maximum_word_count": words <= MAX_DOCUMENTATION_WORDS,
    }
    assert tuple(gates) == DOCUMENTATION_GATE_NAMES
    return DocumentationGateScore(
        gate_vector=gates,
        passed_count=sum(gates.values()),
        total_count=len(gates),
        word_count=words,
    )
