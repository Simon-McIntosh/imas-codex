"""Pydantic models for standard name pipeline LLM responses."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

# Valid enum values for IR segment fields.  Plain ``str`` is used on the
# Pydantic model (instead of ``Literal``) because Anthropic/OpenRouter
# rejects ``anyOf: [{enum: [...]}, {type: null}]`` patterns in structured
# output schemas.  Validators below enforce the allowed values.
_BASE_KINDS = {"quantity", "geometry"}
_PROJECTION_SHAPES = {"component", "coordinate"}
_LOCUS_RELATIONS = {"of", "at", "over"}
_LOCUS_TYPES = {"entity", "position", "region", "geometry"}
_OPERATOR_KINDS = {"unary_prefix", "unary_postfix"}
_ENTRY_KINDS = {"scalar", "vector", "metadata"}


class GrammarSegments(BaseModel):
    """IR grammar segment fields — the LLM's output target.

    Separated into its own sub-model to keep the per-object property
    count under Anthropic/OpenRouter's undocumented structured-output
    schema limit (~13 properties per ``$defs`` item).
    """

    base_token: str = Field(
        description=(
            "Physical quantity or geometry carrier token, "
            "e.g. 'temperature', 'position'"
        )
    )
    base_kind: str = Field(
        description="'quantity' for physical_base, 'geometry' for geometric_base"
    )

    projection_axis: str | None = Field(
        default=None,
        description=(
            "Axis token for component/coordinate projection, e.g. 'radial', 'toroidal'"
        ),
    )
    projection_shape: str | None = Field(
        default=None,
        description=(
            "'component' for physical quantities, 'coordinate' for geometric bases"
        ),
    )

    qualifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Species or source-entity qualifier tokens, "
            "e.g. ['electron'] or ['thermal', 'ion']"
        ),
    )

    locus_token: str | None = Field(
        default=None,
        description="Location reference token, e.g. 'magnetic_axis', 'flux_loop'",
    )
    locus_relation: str | None = Field(
        default=None, description="Locus preposition: 'of', 'at', or 'over'"
    )
    locus_type: str | None = Field(
        default=None,
        description="Locus classification: 'entity', 'position', 'region', or 'geometry'",
    )

    process_token: str | None = Field(
        default=None,
        description="Causal process token for due_to_ suffix, e.g. 'collisions'",
    )

    operator_token: str | None = Field(
        default=None,
        description=(
            "Unary operator token, e.g. 'time_derivative', 'tendency', 'magnitude'"
        ),
    )
    operator_kind: str | None = Field(
        default=None,
        description="Operator position: 'unary_prefix' or 'unary_postfix'",
    )

    @model_validator(mode="after")
    def _validate_enum_fields(self) -> GrammarSegments:
        """Enforce allowed values for enum-like str fields."""
        if self.base_kind not in _BASE_KINDS:
            raise ValueError(
                f"base_kind must be one of {_BASE_KINDS}, got '{self.base_kind}'"
            )
        if (
            self.projection_shape is not None
            and self.projection_shape not in _PROJECTION_SHAPES
        ):
            raise ValueError(
                f"projection_shape must be one of {_PROJECTION_SHAPES}, "
                f"got '{self.projection_shape}'"
            )
        if (
            self.locus_relation is not None
            and self.locus_relation not in _LOCUS_RELATIONS
        ):
            raise ValueError(
                f"locus_relation must be one of {_LOCUS_RELATIONS}, "
                f"got '{self.locus_relation}'"
            )
        if self.locus_type is not None and self.locus_type not in _LOCUS_TYPES:
            raise ValueError(
                f"locus_type must be one of {_LOCUS_TYPES}, got '{self.locus_type}'"
            )
        if self.operator_kind is not None and self.operator_kind not in _OPERATOR_KINDS:
            raise ValueError(
                f"operator_kind must be one of {_OPERATOR_KINDS}, "
                f"got '{self.operator_kind}'"
            )
        return self

    @model_validator(mode="after")
    def _validate_base_token(self) -> GrammarSegments:
        """Validate base_token is a registered physical_base or geometric_base."""
        try:
            from imas_standard_names import get_grammar_context
        except ImportError:
            return self

        ctx = get_grammar_context()
        vocab = ctx.get("vocabulary_sections", [])

        if self.base_kind == "quantity":
            pb_section = next(
                (s for s in vocab if s["segment"] == "physical_base"), None
            )
            tokens = pb_section.get("tokens", []) if pb_section else []
            if tokens and self.base_token not in tokens:
                raise ValueError(
                    f"base_token '{self.base_token}' is not a registered "
                    f"physical_base. Use a vocab_gap entry instead."
                )
        elif self.base_kind == "geometry":
            gb_section = next(
                (s for s in vocab if s["segment"] == "geometric_base"), None
            )
            tokens = gb_section.get("tokens", []) if gb_section else []
            if tokens and self.base_token not in tokens:
                raise ValueError(
                    f"base_token '{self.base_token}' is not a registered "
                    f"geometric_base."
                )
        return self

    @model_validator(mode="after")
    def _validate_projection_axis(self) -> GrammarSegments:
        """Validate projection_axis against closed component/coordinate vocab."""
        if self.projection_axis is None:
            return self
        try:
            from imas_standard_names import get_grammar_context
        except ImportError:
            return self

        ctx = get_grammar_context()
        vocab = ctx.get("vocabulary_sections", [])

        segment = "component" if self.projection_shape == "component" else "coordinate"
        section = next((s for s in vocab if s["segment"] == segment), None)
        tokens = section.get("tokens", []) if section else []
        if tokens and self.projection_axis not in tokens:
            raise ValueError(
                f"projection_axis '{self.projection_axis}' is not a registered "
                f"{segment} token."
            )
        return self

    @model_validator(mode="after")
    def _validate_qualifiers(self) -> GrammarSegments:
        """Validate qualifier tokens against all grammar vocabularies.

        The ``qualifiers`` field can hold tokens from subject, qualifier,
        component, or coordinate segments — the ISN compose step validates
        actual grammar compatibility.  We only reject tokens that appear
        in *no* grammar section at all.
        """
        if not self.qualifiers:
            return self
        try:
            from imas_standard_names import get_grammar_context
        except ImportError:
            return self

        ctx = get_grammar_context()
        vocab = ctx.get("vocabulary_sections", [])
        allowed: set[str] = set()
        for section in vocab:
            allowed.update(section.get("tokens", []))
        if allowed:
            for q in self.qualifiers:
                if q not in allowed:
                    raise ValueError(
                        f"qualifier '{q}' is not a registered grammar token."
                    )
        return self

    def to_ir(self) -> Any:
        """Build ISN StandardNameIR from segment fields."""
        from imas_standard_names.grammar.ir import (
            AxisProjection,
            BaseKind,
            LocusRef,
            LocusRelation,
            LocusType,
            OperatorApplication,
            OperatorKind,
            Process,
            ProjectionShape,
            Qualifier,
            QuantityOrCarrier,
            StandardNameIR,
        )

        base = QuantityOrCarrier(token=self.base_token, kind=BaseKind(self.base_kind))

        projection = None
        if self.projection_axis is not None and self.projection_shape is not None:
            projection = AxisProjection(
                axis=self.projection_axis,
                shape=ProjectionShape(self.projection_shape),
            )

        qualifiers = [Qualifier(token=q) for q in self.qualifiers]

        locus = None
        if (
            self.locus_token is not None
            and self.locus_relation is not None
            and self.locus_type is not None
        ):
            # Auto-correct invalid relation+type combos per ISN rules:
            # entity→of, position→at|of, region→over, geometry→of
            _VALID_RELATIONS: dict[str, list[str]] = {
                "entity": ["of"],
                "position": ["at", "of"],
                "region": ["over"],
                "geometry": ["of"],
            }
            rel = self.locus_relation
            lt = self.locus_type
            allowed = _VALID_RELATIONS.get(lt, ["of"])
            if rel not in allowed:
                rel = allowed[0]

            locus = LocusRef(
                relation=LocusRelation(rel),
                token=self.locus_token,
                type=LocusType(lt),
            )

        mechanism = None
        if self.process_token is not None:
            mechanism = Process(token=self.process_token)

        operators: list[OperatorApplication] = []
        if self.operator_token is not None and self.operator_kind is not None:
            operators = [
                OperatorApplication(
                    kind=OperatorKind(self.operator_kind),
                    op=self.operator_token,
                )
            ]

        return StandardNameIR(
            operators=operators,
            projection=projection,
            qualifiers=qualifiers,
            base=base,
            locus=locus,
            mechanism=mechanism,
        )

    def compose_name(self) -> str:
        """Build canonical standard name string via ISN compose()."""
        from imas_standard_names.grammar.render import compose

        return compose(self.to_ir())


# Module-level constant: segment field names used by flat-wrap validators.
_GRAMMAR_SEGMENT_FIELDS = frozenset(GrammarSegments.model_fields)


class StandardNameCandidate(BaseModel):
    """A single standard name candidate — LLM fills grammar segments.

    The ``segments`` sub-model contains all IR grammar fields. This
    split keeps each JSON schema ``$defs`` item under Anthropic's
    ~13-property limit for structured output.
    """

    source_id: str = Field(description="Source entity ID (DD path or signal ID)")
    segments: GrammarSegments = Field(description="ISN grammar segment fields")

    # --- Non-IR fields ---
    description: str = Field(
        default="",
        description="1-line ≤120 char summary of the physical quantity",
    )
    kind: str = Field(default="scalar", description="Entry kind")
    dd_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    reason: str = Field(description="Brief justification (≤25 words)")

    @model_validator(mode="before")
    @classmethod
    def _wrap_flat_segments(cls, data: Any) -> Any:
        """Auto-wrap flat segment fields into a ``segments`` sub-dict.

        Allows callers to pass ``base_token='temperature'`` at the top
        level instead of ``segments={'base_token': 'temperature', ...}``.
        """
        if not isinstance(data, dict):
            return data
        if "segments" in data:
            return data
        seg_keys = _GRAMMAR_SEGMENT_FIELDS & data.keys()
        if seg_keys:
            segments = {k: data.pop(k) for k in seg_keys}
            data["segments"] = segments
        return data

    @model_validator(mode="after")
    def _validate_kind(self) -> StandardNameCandidate:
        if self.kind not in _ENTRY_KINDS:
            raise ValueError(f"kind must be one of {_ENTRY_KINDS}, got '{self.kind}'")
        return self

    # --- Convenience accessors delegating to segments ---

    @property
    def base_token(self) -> str:
        return self.segments.base_token

    @property
    def base_kind(self) -> str:
        return self.segments.base_kind

    @property
    def projection_axis(self) -> str | None:
        return self.segments.projection_axis

    @property
    def projection_shape(self) -> str | None:
        return self.segments.projection_shape

    @property
    def qualifiers(self) -> list[str]:
        return self.segments.qualifiers

    @property
    def locus_token(self) -> str | None:
        return self.segments.locus_token

    @property
    def locus_relation(self) -> str | None:
        return self.segments.locus_relation

    @property
    def locus_type(self) -> str | None:
        return self.segments.locus_type

    @property
    def process_token(self) -> str | None:
        return self.segments.process_token

    @property
    def operator_token(self) -> str | None:
        return self.segments.operator_token

    @property
    def operator_kind(self) -> str | None:
        return self.segments.operator_kind

    def to_ir(self) -> Any:
        """Delegate to segments."""
        return self.segments.to_ir()

    def compose_name(self) -> str:
        """Delegate to segments."""
        return self.segments.compose_name()


class StandardNameVocabGap(BaseModel):
    """A path where naming requires vocabulary expansion."""

    source_id: str = Field(description="DD path that needs naming")
    segment: str = Field(
        description="Grammar segment missing a token (e.g., 'subject', 'position')"
    )
    needed_token: str = Field(
        description="Proposed token value for the grammar segment"
    )
    reason: str = Field(description="Why this token is needed for naming this path")


class StandardNameAttachment(BaseModel):
    """A DD path that should attach to an existing standard name without regeneration."""

    source_id: str = Field(description="DD path to attach")
    standard_name: str = Field(description="Existing standard name to attach to")
    reason: str = Field(description="Why this path maps to this existing name")


class StandardNameComposeBatch(BaseModel):
    """LLM response for a batch of standard name compositions."""

    candidates: list[StandardNameCandidate]
    attachments: list[StandardNameAttachment] = Field(
        default_factory=list,
        description=(
            "DD paths that map to existing standard names — attach without regeneration. "
            "Use when a path measures the exact same quantity as an existing name."
        ),
    )
    skipped: list[str] = Field(
        default_factory=list, description="Source IDs skipped (not physics quantities)"
    )
    vocab_gaps: list[StandardNameVocabGap] = Field(
        default_factory=list,
        description="Paths where naming requires vocabulary expansion in imas-standard-names",
    )

    @model_validator(mode="before")
    @classmethod
    def _rescue_failed_candidates(cls, data: Any) -> Any:
        """Validate candidates individually; move vocab-gap failures to vocab_gaps.

        Without this, a single candidate with an unregistered grammar token
        fails Pydantic validation for the entire batch, marking ALL sources
        as vocab_gap.  This validator catches per-candidate errors and
        converts them to explicit vocab_gap entries, preserving valid
        candidates in the same batch.
        """
        if not isinstance(data, dict):
            return data
        raw_candidates = data.get("candidates")
        if not raw_candidates or not isinstance(raw_candidates, list):
            return data

        valid: list[dict] = []
        rescued_gaps: list[dict] = []

        for raw in raw_candidates:
            if not isinstance(raw, dict):
                valid.append(raw)
                continue
            try:
                StandardNameCandidate.model_validate(raw)
                valid.append(raw)
            except (ValidationError, ValueError) as exc:
                exc_str = str(exc)
                if "not a registered" not in exc_str:
                    # Non-vocab-gap error — keep for normal batch failure
                    valid.append(raw)
                    continue
                # Extract source_id and failed token info from the error
                source_id = raw.get("source_id", "unknown")
                segments = raw.get("segments", raw)
                segment, token = _extract_gap_from_error(exc_str, segments)
                rescued_gaps.append(
                    {
                        "source_id": source_id,
                        "segment": segment,
                        "needed_token": token,
                        "reason": f"LLM proposed unregistered {segment} token",
                    }
                )
                logger.info(
                    "Rescued candidate %s from batch failure: %s '%s' not registered",
                    source_id,
                    segment,
                    token,
                )

        if rescued_gaps:
            data["candidates"] = valid
            existing_gaps = data.get("vocab_gaps", [])
            if isinstance(existing_gaps, list):
                data["vocab_gaps"] = existing_gaps + rescued_gaps
            else:
                data["vocab_gaps"] = rescued_gaps
            logger.warning(
                "Batch rescue: %d candidates → vocab_gap, %d valid preserved",
                len(rescued_gaps),
                len(valid),
            )

        return data


def _extract_gap_from_error(exc_str: str, segments: dict[str, Any]) -> tuple[str, str]:
    """Extract (segment_name, token_value) from a vocab-gap validation error.

    Parses error messages like:
      "qualifier 'cumulative' is not a registered grammar token."
      "base_token 'foo' is not a registered physical_base."
      "projection_axis 'bar' is not a registered component token."
    """
    import re

    # Pattern: "<field_or_segment> '<token>' is not a registered"
    m = re.search(r"(\w+)\s+'([^']+)'\s+is not a registered", exc_str)
    if m:
        field_name = m.group(1)
        token = m.group(2)
        # Map field names to grammar segments
        segment_map = {
            "base_token": "physical_base",
            "projection_axis": "component",
            "qualifier": "qualifier",
            "locus_token": "geometry",
            "process_token": "process",
            "operator_token": "qualifier",
        }
        return segment_map.get(field_name, field_name), token

    # Fallback: find the first unknown token from segments
    for field in ["base_token", "qualifiers", "projection_axis", "locus_token"]:
        val = segments.get(field)
        if isinstance(val, str) and val:
            return field, val
        if isinstance(val, list):
            for v in val:
                if isinstance(v, str):
                    return field, v
    return "unknown", "unknown"


# =============================================================================
# Publish models — YAML catalog export (Feature 08)
# =============================================================================


class StandardNameProvenance(BaseModel):
    """Provenance metadata for a standard name entry."""

    source: str = Field(description="Source type: dd or signal")
    source_id: str = Field(description="Source entity ID")
    ids_name: str | None = Field(default=None, description="IDS name (for DD source)")
    generated_by: str = Field(
        default="imas-codex", description="Tool that generated this"
    )


class StandardNamePublishEntry(BaseModel):
    """A single standard name entry ready for YAML catalog export."""

    name: str = Field(description="The standard name")
    kind: str = Field(
        default="scalar", description="Name kind: scalar, vector, or metadata"
    )
    unit: str | None = Field(default=None, description="SI unit string")
    status: str = Field(default="drafted", description="Entry status")
    physics_domain: str | None = Field(
        default=None,
        description="Primary physics domain (scalar, promoted by rank).",
    )
    source_domains: list[str] = Field(
        default_factory=list,
        description=(
            "All physics domains that have contributed a source to "
            "this StandardName (append-only, deduplicated)."
        ),
    )
    description: str = Field(default="", description="Human-readable description")
    # Rich fields
    documentation: str | None = Field(
        default=None, description="Rich documentation with LaTeX"
    )
    links: list[str] = Field(default_factory=list, description="Related standard names")
    dd_paths: list[str] = Field(
        default_factory=list, description="Mapped IMAS DD paths"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints"
    )
    validity_domain: str | None = Field(
        default=None, description="Physical region where valid"
    )
    cocos_transformation_type: str | None = Field(
        default=None,
        description="COCOS transformation type (e.g., psi_like, ip_like). Null for non-COCOS quantities.",
    )
    cocos: int | None = Field(
        default=None,
        description="COCOS convention index (e.g. 11, 17). Null for non-COCOS quantities.",
    )
    provenance: StandardNameProvenance = Field(description="Generation provenance")


class StandardNamePublishBatch(BaseModel):
    """A batch of entries to publish as a PR."""

    group_key: str = Field(description="Batch group key (IDS name or domain)")
    entries: list[StandardNamePublishEntry]
    confidence_tier: str = Field(description="high, medium, or low")


# =============================================================================
# Cross-model review models
# =============================================================================


class StandardNameReviewItem(BaseModel):
    """Review of a single standard name candidate."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    reason: str = Field(description="Justification for the review")
    revised_name: str | None = Field(
        default=None, description="Suggested revised name, if any"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameReviewBatch(BaseModel):
    """LLM response for reviewing a batch of standard name candidates."""

    reviews: list[StandardNameReviewItem]


class StandardNameQualityComments(BaseModel):
    """Per-dimension comments for the full 6-dimensional review rubric."""

    grammar: str | None = Field(default=None, description="Comment on grammar score")
    semantic: str | None = Field(default=None, description="Comment on semantic score")
    documentation: str | None = Field(
        default=None, description="Comment on documentation score"
    )
    convention: str | None = Field(
        default=None, description="Comment on convention score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )
    compliance: str | None = Field(
        default=None, description="Comment on compliance score"
    )


class StandardNameQualityCommentsNameOnly(BaseModel):
    """Per-dimension comments for the 4-dimensional name-only review rubric."""

    grammar: str | None = Field(default=None, description="Comment on grammar score")
    semantic: str | None = Field(default=None, description="Comment on semantic score")
    convention: str | None = Field(
        default=None, description="Comment on convention score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )


class StandardNameQualityCommentsDocs(BaseModel):
    """Per-dimension comments for the 4-dimensional docs review rubric.

    Note: uses independent dimension names (description_quality etc.),
    NOT a subset of the full 6-dim names.
    """

    description_quality: str | None = Field(
        default=None, description="Comment on description quality score"
    )
    documentation_quality: str | None = Field(
        default=None, description="Comment on documentation quality score"
    )
    completeness: str | None = Field(
        default=None, description="Comment on completeness score"
    )
    physics_accuracy: str | None = Field(
        default=None, description="Comment on physics accuracy score"
    )


# =============================================================================
# Unified quality review models (used by both mint and benchmark)
# =============================================================================


class StandardNameQualityScore(BaseModel):
    """6-dimensional quality score for a standard name entry."""

    grammar: int = Field(ge=0, le=20, description="Grammar correctness (0-20)")
    semantic: int = Field(ge=0, le=20, description="Semantic accuracy (0-20)")
    documentation: int = Field(ge=0, le=20, description="Documentation quality (0-20)")
    convention: int = Field(ge=0, le=20, description="Naming conventions (0-20)")
    completeness: int = Field(ge=0, le=20, description="Entry completeness (0-20)")
    compliance: int = Field(
        ge=0, le=20, description="Prompt instruction compliance (0-20)"
    )

    @property
    def total(self) -> int:
        return (
            self.grammar
            + self.semantic
            + self.documentation
            + self.convention
            + self.completeness
            + self.compliance
        )

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 6 dimensions / 120."""
        return self.total / 120.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReview(BaseModel):
    """Review of a single standard name with quality scoring."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScore = Field(description="6-dimensional quality scores")
    comments: StandardNameQualityComments | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    reasoning: str = Field(description="Specific justification per dimension")
    revised_name: str | None = Field(
        default=None, description="Suggested revised name, if any"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    suggested_name: str | None = Field(
        default=None,
        description=(
            "Reviewer-recommended improved name when the candidate could be "
            "improved; null when no better name is offered."
        ),
    )
    suggestion_justification: str | None = Field(
        default=None,
        description=(
            "1–3 sentence justification for suggested_name. Null when "
            "suggested_name is null."
        ),
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewBatch(BaseModel):
    """LLM response for quality-scored review of a batch."""

    reviews: list[StandardNameQualityReview]


# =============================================================================
# Name-only review — 4-dimensional rubric for --name-only cycles
# =============================================================================


class StandardNameQualityScoreNameOnly(BaseModel):
    """4-dimensional quality score for name-only review mode.

    Scores the name itself (grammar, semantic, convention, completeness)
    without penalising missing documentation or compliance, which are
    intentionally deferred in name-only generation cycles. Normalised
    over 80 rather than 120.
    """

    grammar: int = Field(ge=0, le=20, description="Grammar correctness (0-20)")
    semantic: int = Field(ge=0, le=20, description="Semantic accuracy (0-20)")
    convention: int = Field(ge=0, le=20, description="Naming conventions (0-20)")
    completeness: int = Field(ge=0, le=20, description="Entry completeness (0-20)")

    @property
    def total(self) -> int:
        return self.grammar + self.semantic + self.convention + self.completeness

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 4 dimensions / 80."""
        return self.total / 80.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReviewNameOnly(BaseModel):
    """Review of a single standard name using the 4-dimensional rubric."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScoreNameOnly = Field(
        description="4-dimensional quality scores"
    )
    comments: StandardNameQualityCommentsNameOnly | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    reasoning: str = Field(description="Specific justification per dimension")
    revised_name: str | None = Field(
        default=None, description="Suggested revised name, if any"
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None, description="Revised grammar fields"
    )
    suggested_name: str | None = Field(
        default=None,
        description=(
            "Reviewer-recommended improved name when the candidate could be "
            "improved; null when no better name is offered."
        ),
    )
    suggestion_justification: str | None = Field(
        default=None,
        description=(
            "1–3 sentence justification for suggested_name, grounded in ISN "
            "grammar and the per-item DD context. Null when suggested_name is null."
        ),
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewNameOnlyBatch(BaseModel):
    """LLM response for name-only quality-scored review of a batch."""

    reviews: list[StandardNameQualityReviewNameOnly]


# =============================================================================
# Docs review — 4-dimensional rubric for --target docs cycles
# =============================================================================


class StandardNameQualityScoreDocs(BaseModel):
    """4-dimensional quality score for docs review mode.

    Scores the generated documentation (description, documentation body,
    completeness of doc fields, and physics accuracy of prose) without
    re-scoring the name itself — the name was already reviewed in a prior
    ``--target names`` cycle. Normalised over 80 rather than 120.
    """

    description_quality: int = Field(
        ge=0, le=20, description="Clarity and precision of short description (0-20)"
    )
    documentation_quality: int = Field(
        ge=0,
        le=20,
        description="Documentation body: equations, variables, sign conventions (0-20)",
    )
    completeness: int = Field(
        ge=0,
        le=20,
        description="Required doc fields filled (links, aliases, cross-refs) (0-20)",
    )
    physics_accuracy: int = Field(
        ge=0,
        le=20,
        description="Physics correctness of documentation prose and equations (0-20)",
    )

    @property
    def total(self) -> int:
        return (
            self.description_quality
            + self.documentation_quality
            + self.completeness
            + self.physics_accuracy
        )

    @property
    def score(self) -> float:
        """Normalized quality score (0-1). Sum of 4 dimensions / 80."""
        return self.total / 80.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85:
            return "outstanding"
        elif s >= 0.65:
            return "good"
        elif s >= 0.40:
            return "inadequate"
        return "poor"


class StandardNameQualityReviewDocs(BaseModel):
    """Review of a single standard name's docs using the 4-dimensional rubric."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScoreDocs = Field(
        description="4-dimensional docs quality scores"
    )
    comments: StandardNameQualityCommentsDocs | None = Field(
        default=None, description="Per-dimension reviewer comments"
    )
    reasoning: str = Field(description="Specific justification per dimension")
    revised_description: str | None = Field(
        default=None, description="Suggested revised description, if any"
    )
    revised_documentation: str | None = Field(
        default=None, description="Suggested revised documentation body"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewDocsBatch(BaseModel):
    """LLM response for docs quality-scored review of a batch."""

    reviews: list[StandardNameQualityReviewDocs]


# =============================================================================
# Enrichment models — documentation iteration (Phase 3D)
# =============================================================================


class StandardNameEnrichItem(BaseModel):
    """Enrichment result for a single standard name."""

    standard_name: str = Field(
        description="The standard name (must match input exactly)"
    )
    description: str = Field(description="One sentence definition, <120 chars")
    documentation: str = Field(
        description="Rich docs with LaTeX, links, typical values"
    )
    links: list[str] = Field(
        default_factory=list, description="Related standard names (name:xxx format)"
    )
    validity_domain: str | None = Field(
        default=None, description="Physical region where quantity is meaningful"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Physical constraints on the quantity"
    )


class StandardNameEnrichBatch(BaseModel):
    """LLM response for enriching a batch of standard names."""

    items: list[StandardNameEnrichItem]


# =============================================================================
# Refine-pipeline response models (Phase 8.1)
# =============================================================================


class RefinedName(BaseModel):
    """LLM response model for a single refine_name call.

    Uses ``GrammarSegments`` sub-model (same as ``StandardNameCandidate``).
    The ``confidence`` field is intentionally absent — removed in Phase 8.1.
    """

    segments: GrammarSegments = Field(description="ISN grammar segment fields")

    # --- Non-IR fields ---
    description: str = Field(
        ..., description="One-sentence physics definition (≤ 120 chars, no LaTeX)"
    )
    kind: str = Field(default="scalar", description="Entry kind")
    reason: str = Field(
        default="",
        description="Brief justification for how this addresses reviewer concerns",
    )

    model_config = {"extra": "ignore", "populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def _wrap_flat_segments(cls, data: Any) -> Any:
        """Auto-wrap flat segment fields into ``segments`` sub-dict."""
        if not isinstance(data, dict):
            return data
        if "segments" in data:
            return data
        seg_keys = _GRAMMAR_SEGMENT_FIELDS & data.keys()
        if seg_keys:
            segments = {k: data.pop(k) for k in seg_keys}
            data["segments"] = segments
        return data

    @model_validator(mode="after")
    def _validate_kind(self) -> RefinedName:
        if self.kind not in _ENTRY_KINDS:
            raise ValueError(f"kind must be one of {_ENTRY_KINDS}, got '{self.kind}'")
        return self

    # --- Convenience accessors delegating to segments ---

    @property
    def base_token(self) -> str:
        return self.segments.base_token

    @property
    def base_kind(self) -> str:
        return self.segments.base_kind

    @property
    def projection_axis(self) -> str | None:
        return self.segments.projection_axis

    @property
    def projection_shape(self) -> str | None:
        return self.segments.projection_shape

    @property
    def qualifiers(self) -> list[str]:
        return self.segments.qualifiers

    @property
    def locus_token(self) -> str | None:
        return self.segments.locus_token

    @property
    def locus_relation(self) -> str | None:
        return self.segments.locus_relation

    @property
    def locus_type(self) -> str | None:
        return self.segments.locus_type

    @property
    def process_token(self) -> str | None:
        return self.segments.process_token

    @property
    def operator_token(self) -> str | None:
        return self.segments.operator_token

    @property
    def operator_kind(self) -> str | None:
        return self.segments.operator_kind

    def to_ir(self) -> Any:
        """Delegate to segments."""
        return self.segments.to_ir()

    def compose_name(self) -> str:
        """Delegate to segments."""
        return self.segments.compose_name()

    @property
    def name(self) -> str:
        """Backwards-compatible name property using IR compose."""
        return self.compose_name()


class RefinedDocs(BaseModel):
    """LLM response model for a single refine_docs call.

    Mirrors the ``StandardNameEnrichItem`` documentation fields but is
    targeted at single-name docs-refine calls rather than batched enrichment.
    """

    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description=(
            "1-3 sentence technical description of the physical quantity "
            "(American spelling, no LaTeX, ≤ 500 chars)."
        ),
    )
    documentation: str = Field(
        ..., description="Full documentation text with LaTeX, typical values, context"
    )
    links: list[str] = Field(
        default_factory=list,
        description="Related standard names (name:xxx or dd:path format)",
    )

    model_config = {"extra": "ignore"}


class GeneratedDocs(BaseModel):
    """LLM response model for a single generate_docs call.

    The model is constrained to produce ONLY documentation content
    (description + documentation).  It must NOT change the name, kind,
    unit, or any other identity field — those are fixed by the
    accepted name_stage.
    """

    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description=(
            "1-3 sentence technical description of the physical quantity "
            "(American spelling, no LaTeX, ≤ 500 chars)."
        ),
    )
    documentation: str = Field(
        ...,
        min_length=20,
        description=(
            "Rich markdown documentation covering physical meaning, governing "
            "equations (LaTeX), typical values, measurement methods, and "
            "cross-references to related standard names."
        ),
    )

    model_config = {"extra": "ignore"}
