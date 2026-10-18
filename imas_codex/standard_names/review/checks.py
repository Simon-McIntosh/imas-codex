"""Named review checks: the taxonomy of every failure a review can raise.

Each review failure the name and documentation rubrics can find is one entry in
``imas_codex/llm/prompts/sn/review_checks.yaml``. A check carries the axis it
belongs to (the name or the documentation), the review dimension it docks, the
review-state evidence fields it reads, a retrieval flag (whether its failure
calls for gathering more context), and how it is decided:

* a Jev question — a Noul whose true case is the defect, or a categorising
  Choice where the defect is one of several forms; or
* the :class:`~imas_codex.standard_names.review.audits.AuditReport` finding
  class that already decides it in code.

A deterministic check is never re-asked as a Jev question, and no check carries
both a question and an audit finding class or neither. The taxonomy is data; the
loader below validates it against the review state the pipeline actually builds.
"""

from __future__ import annotations

import ast
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    HouseStyleFindingClass,
)

CHECKS_PATH = (
    Path(__file__).resolve().parents[2]
    / "llm"
    / "prompts"
    / "sn"
    / "review_checks.yaml"
)

# The finding families an ``AuditReport`` can carry. The lexical lint and link
# integrity emit their own ``finding_type`` strings; the deterministic
# documentation gates emit the gate name; the registry conformance checks emit
# ``convention_violation`` with the class named first in their detail, so the
# class name is what a check reads. Near-duplicate groups are a distinct field.
_LEXICAL_FINDING_TYPES = ("round_trip_failure", "convention_violation")
_LINK_FINDING_TYPES = ("unresolved", "dead_link", "missing_reverse")

AUDIT_FINDING_CLASSES: frozenset[str] = frozenset(
    {
        *_LEXICAL_FINDING_TYPES,
        *_LINK_FINDING_TYPES,
        *DOCUMENTATION_GATE_NAMES,
        *(member.value for member in HouseStyleFindingClass),
        "duplicate_component",
    }
)

NAME_DIMENSIONS = frozenset({"grammar", "semantic", "convention", "completeness"})
DOCS_DIMENSIONS = frozenset(
    {"description_quality", "documentation_quality", "completeness", "physics_accuracy"}
)

# Item-attribute variables the review pipeline assigns fields onto. The
# carried-field set is read from the pipeline by walking its AST for keys set on
# these names, the extract query's RETURN aliases, the enrichment context-key
# tuple, and the prompt context dict — never from a hand-written list.
_ITEM_VARIABLES = ("item", "item_data", "name", "stub")
_ALWAYS_PRESENT = ("_computed_hash", "validation_issues")

_RETURN_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)")


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# The one polarity a Noul question may declare: its true case is the defect the
# check names. The reader fails a Noul on a high probability, so this is the
# value the loader requires and the only value it accepts.
NOUL_TRUE_CASE = "defect"


class JevQuestionKind(StrEnum):
    """The shape of the judgment a failed check asks of Jev."""

    NOUL = "noul"
    CHOICE = "choice"


class JevQuestion(BaseModel):
    """One question whose failing case is the defect the check names."""

    kind: JevQuestionKind
    text: str
    options: list[str] = Field(default_factory=list)
    conforming: list[str] = Field(default_factory=list)
    # A Noul carries its polarity explicitly: the field states that the
    # question's true case is the defect the check names. The reader fails a
    # Noul when its probability is high, so a question whose true case is the
    # acceptable form would be read backwards. The loader requires every Noul
    # to declare this field, and this value, so the one convention the reader
    # applies is stated in the taxonomy rather than inferred from the wording.
    true_case: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> JevQuestion:
        if self.kind is JevQuestionKind.CHOICE:
            if len(self.options) < 2:
                raise ValueError("a Choice question requires at least two options")
            if (
                not self.conforming
                or any(not option.strip() for option in self.conforming)
                or not set(self.conforming) <= set(self.options)
            ):
                raise ValueError(
                    "a Choice question requires conforming options from its options"
                )
        elif self.options:
            raise ValueError("a Noul question carries no options")
        elif self.conforming:
            raise ValueError("a Noul question carries no conforming options")
        return self


class ReviewCheck(BaseModel):
    """One named check in the review taxonomy."""

    id: str
    axis: str
    dimension: str
    evidence_fields: list[str]
    retrieval: bool
    question: JevQuestion | None = None
    audit_finding_class: str | None = None

    @model_validator(mode="after")
    def _decided_exactly_once(self) -> ReviewCheck:
        has_question = self.question is not None
        has_audit = self.audit_finding_class is not None
        if has_question == has_audit:
            raise ValueError(
                f"check {self.id!r} must carry exactly one of a Jev question or "
                "an audit finding class"
            )
        if self.axis not in {"name", "docs"}:
            raise ValueError(f"check {self.id!r} has unknown axis {self.axis!r}")
        valid_dims = NAME_DIMENSIONS if self.axis == "name" else DOCS_DIMENSIONS
        if self.dimension not in valid_dims:
            raise ValueError(
                f"check {self.id!r} dimension {self.dimension!r} is not a "
                f"{self.axis}-axis dimension"
            )
        if not self.evidence_fields:
            raise ValueError(f"check {self.id!r} reads no evidence field")
        if self.audit_finding_class is not None and (
            self.audit_finding_class not in AUDIT_FINDING_CLASSES
        ):
            raise ValueError(
                f"check {self.id!r} names audit finding class "
                f"{self.audit_finding_class!r}, which no audit emits"
            )
        return self


class ReasonMapEntry(BaseModel):
    """One distinct human edit reason mapped to a check, or left unmapped."""

    reason: str
    count: int
    check_id: str | None = None
    unmapped_reason: str | None = None


def _pipeline_source() -> str:
    return (Path(__file__).resolve().parent / "pipeline.py").read_text(encoding="utf-8")


def review_state_fields() -> frozenset[str]:
    """Return the field set the review state carries, read from the pipeline.

    Derived rather than hand-listed. Walking the pipeline's AST yields the keys
    the enrichment attaches to each review item (as subscripts and ``.get``
    reads on the item variables), every bare-string element of a
    ``*_CONTEXT_KEYS`` tuple, and the keys of the prompt ``context`` dict. The
    extract query's ``RETURN`` aliases are read from its query string.
    """

    tree = ast.parse(_pipeline_source())
    fields: set[str] = set(_ALWAYS_PRESENT)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if node.target is not None else [])
            )
            for target in targets:
                key = _item_subscript_key(target)
                if key:
                    fields.add(key)
                if isinstance(target, ast.Name) and target.id.endswith("_CONTEXT_KEYS"):
                    fields.update(_string_elements(node.value))
                if isinstance(target, ast.Name) and target.id == "context":
                    fields.update(_dict_string_keys(node.value))
        elif isinstance(node, ast.Call):
            fields.update(_get_keys(node))

    for query in _query_strings(tree):
        fields.update(_RETURN_ALIAS_RE.findall(query))

    if len(fields) < 25:
        raise RuntimeError(
            "review-state field derivation found only "
            f"{len(fields)} fields — the pipeline source was not read as expected"
        )
    return frozenset(fields)


def _item_subscript_key(target: ast.AST) -> str | None:
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        if target.value.id in _ITEM_VARIABLES:
            return _const_str(target.slice)
    return None


def _get_keys(call: ast.Call) -> set[str]:
    keys: set[str] = set()
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Name)
        and func.value.id in _ITEM_VARIABLES
        and call.args
    ):
        key = _const_str(call.args[0])
        if key:
            keys.add(key)
    return keys


def _string_elements(node: ast.AST | None) -> set[str]:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return {s for el in node.elts if (s := _const_str(el))}
    return set()


def _dict_string_keys(node: ast.AST | None) -> set[str]:
    if isinstance(node, ast.Dict):
        return {k for key in node.keys if (k := _const_str(key))}
    return set()


def _query_strings(tree: ast.AST) -> list[str]:
    return [
        _const_str(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "MATCH (sn:StandardName)" in node.value
    ]


def load_review_checks(
    path: Path | None = None,
) -> tuple[list[ReviewCheck], list[ReasonMapEntry]]:
    """Load and validate the taxonomy, returning its checks and reason map."""

    data: dict[str, Any] = yaml.safe_load(
        (path or CHECKS_PATH).read_text(encoding="utf-8")
    )
    checks = [ReviewCheck(**entry) for entry in data.get("checks", [])]
    for check in checks:
        _require_noul_true_case(check)
    reason_map = [ReasonMapEntry(**entry) for entry in data.get("human_reason_map", [])]
    return checks, reason_map


def _require_noul_true_case(check: ReviewCheck) -> None:
    """Refuse a Noul question that does not declare the reader's one polarity.

    The reader fails a Noul when its probability is at or above the midpoint, so
    it reads the question's true case as the defect. A Noul that omits the field,
    or names any value but the defect, would be read backwards without saying so;
    refusing it here keeps the convention single and stated in the data.
    """
    question = check.question
    if question is None or question.kind is not JevQuestionKind.NOUL:
        return
    if question.true_case != NOUL_TRUE_CASE:
        raise ValueError(
            f"check {check.id!r} asks a Noul question whose true_case is "
            f"{question.true_case!r}; every Noul question must declare "
            f"true_case: {NOUL_TRUE_CASE}"
        )
