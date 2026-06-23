"""Pydantic models for standard name pipeline LLM responses."""

from __future__ import annotations

import functools
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
_OPERATOR_KINDS = {"unary_prefix", "unary_postfix", "binary"}
_ENTRY_KINDS = {"scalar", "vector", "metadata"}

# Binary operators combine two operands into ``<op>_of_<A>_<sep>_<B>`` (e.g.
# ``ratio_of_electron_temperature_to_ion_temperature``). ISN's model layer
# expects the model-form token (``ratio_of``) in the ``binary_operator`` field;
# its operators vocabulary registers the bare IR form (``ratio``). Codex accepts
# either spelling from the LLM and normalises to the model form.
_BINARY_OP_MODEL = {"ratio_of", "product_of", "difference_of"}
_BINARY_BARE_TO_MODEL = {
    "ratio": "ratio_of",
    "product": "product_of",
    "difference": "difference_of",
}


@functools.cache
def _operator_registry_kinds() -> dict[str, str]:
    """Map each registered operator token to its registry ``kind``.

    Returns ``{token: "unary_prefix" | "unary_postfix" | "binary"}`` from the
    public grammar context. The registry ``kind`` is authoritative for whether
    an operator is a prefix transformation or a postfix decomposition; it is
    the only operator routing codex needs because the ISN model layer
    (``compose_standard_name``) owns the bare-vs-``_of_`` prefix distinction.
    """
    try:
        from imas_standard_names import get_grammar_context
    except ImportError:  # pragma: no cover - ISN always present in this repo
        return {}
    ctx = get_grammar_context()
    ops = ctx.get("grammar", {}).get("vocabularies", {}).get("operators", {})
    return {token: meta.get("kind", "") for token, meta in ops.items()}


@functools.cache
def _coord_indexed_operator_tokens() -> frozenset[str]:
    """Operator tokens that bind a *coordinate* index.

    These require ``operator_coordinate`` to be set; emitting one without the
    coordinate drops it and mints a malformed name
    (``derivative_with_respect_to_of_volume``). The coordinate-indexed operators
    are the ``indexed`` *prefix* operators (``derivative_with_respect_to``);
    indexed *postfix* operators (``bessel_0``/``bessel_1``/``fourier_coefficient``)
    are order/mode-indexed and carry their index in the token itself, so they do
    NOT use ``operator_coordinate``. Pulled from the live registry so a new
    coordinate-indexed operator is picked up without a code change.
    """
    try:
        from imas_standard_names import get_grammar_context
    except ImportError:  # pragma: no cover - ISN always present in this repo
        return frozenset()
    ctx = get_grammar_context()
    ops = ctx.get("grammar", {}).get("vocabularies", {}).get("operators", {})
    return frozenset(
        token
        for token, meta in ops.items()
        if meta.get("indexed") and meta.get("kind") == "unary_prefix"
    )


@functools.cache
def _projection_axis_tokens() -> dict[str, frozenset[str]]:
    """Registered component/coordinate axis tokens, keyed by projection shape.

    Used to detect compound axis tokens (e.g. ``normalized_radial``) that ISN
    forms by fusing a bare-prefix transformation with an axis. Pulled from the
    live grammar vocabulary — no hardcoded token list.
    """
    try:
        from imas_standard_names import get_grammar_context
    except ImportError:  # pragma: no cover - ISN always present in this repo
        return {"component": frozenset(), "coordinate": frozenset()}
    ctx = get_grammar_context()
    out: dict[str, frozenset[str]] = {}
    for shape in ("component", "coordinate"):
        section = next(
            (
                s
                for s in ctx.get("vocabulary_sections", [])
                if s.get("segment") == shape
            ),
            None,
        )
        out[shape] = frozenset(section.get("tokens", []) if section else [])
    return out


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
    locus_value: str | None = Field(
        default=None,
        description=(
            "Numeric value for value-parameterized at-positions, underscores "
            "as decimal separator (e.g. '0_95' for q95 → "
            "at_normalized_poloidal_magnetic_flux_equal_to_0_95). Requires "
            "locus_relation='at' and locus_type='position'."
        ),
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
        description="Operator position: 'unary_prefix', 'unary_postfix', or 'binary'",
    )

    operator_coordinate: str | None = Field(
        default=None,
        description=(
            "Bound coordinate index of an INDEXED operator. Set ONLY with an "
            "indexed operator_token: for 'derivative_with_respect_to' this is the "
            "coordinate the derivative is taken against, as a registered "
            "coordinate carrier (e.g. 'poloidal_magnetic_flux_coordinate', "
            "'toroidal_flux_coordinate', 'normalized_poloidal_flux_coordinate', "
            "'radial_coordinate'). Example: dVolume/dpsi → base_token='volume', "
            "operator_token='derivative_with_respect_to', "
            "operator_coordinate='poloidal_magnetic_flux_coordinate' → "
            "derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume. "
            "Null for non-indexed operators."
        ),
    )

    secondary_base: str | None = Field(
        default=None,
        description=(
            "Second operand of a binary operator, as a fully-composed "
            "standard-name string. Set ONLY when operator_token is a binary "
            "operator (ratio_of/product_of/difference_of). The first operand is "
            "built from base_token (+ qualifiers); this is the second. "
            "Example: for ratio_of_electron_temperature_to_ion_temperature set "
            "base_token='temperature', qualifiers=['electron'], "
            "operator_token='ratio_of', secondary_base='ion_temperature'. "
            "Use this for ratios/products/differences instead of inventing a "
            "compound base token like 'velocity_over_magnetic_field'."
        ),
    )

    def _binary_model_op(self) -> str | None:
        """Return the ISN model-form binary operator token, or None.

        Recognises both the model spelling (``ratio_of``) and the bare IR
        spelling (``ratio``); falls back to the operator registry ``kind`` so a
        future binary token is picked up without a code change.
        """
        tok = self.operator_token
        if tok is None:
            return None
        if tok in _BINARY_OP_MODEL:
            return tok
        if tok in _BINARY_BARE_TO_MODEL:
            return _BINARY_BARE_TO_MODEL[tok]
        if _operator_registry_kinds().get(tok) == "binary":
            return _BINARY_BARE_TO_MODEL.get(tok, f"{tok}_of")
        return None

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

    @model_validator(mode="after")
    def _validate_binary_operator(self) -> GrammarSegments:
        """A binary operator needs a second operand and vice versa.

        The operand strings themselves are validated by the ISN model layer at
        compose time (it re-parses each operand), so codex only enforces the
        pairing here.
        """
        binary_op = self._binary_model_op()
        if binary_op is not None and not self.secondary_base:
            raise ValueError(
                f"binary operator '{self.operator_token}' requires a "
                f"secondary_base (the second operand)."
            )
        if self.secondary_base and binary_op is None:
            raise ValueError(
                "secondary_base requires a binary operator_token "
                "(ratio_of / product_of / difference_of)."
            )
        return self

    @model_validator(mode="after")
    def _validate_operator_coordinate(self) -> GrammarSegments:
        """Coordinate-index consistency for indexed operators.

        A coordinate-indexed operator (``derivative_with_respect_to``) needs
        ``operator_coordinate``; without it the coordinate is silently dropped
        and a malformed ``derivative_with_respect_to_of_<base>`` is minted.
        Conversely, ``operator_coordinate`` only makes sense with such an
        operator.
        """
        coord_indexed = _coord_indexed_operator_tokens()
        if self.operator_coordinate and self.operator_token is None:
            raise ValueError(
                "operator_coordinate requires an (indexed) operator_token."
            )
        # Only enforce the requires-coordinate rule when the registry actually
        # knows the operator is coord-indexed (avoids false positives if the
        # registry is unavailable).
        if (
            coord_indexed
            and self.operator_token in coord_indexed
            and not self.operator_coordinate
        ):
            raise ValueError(
                f"operator '{self.operator_token}' is coordinate-indexed and "
                f"requires operator_coordinate (e.g. "
                f"'poloidal_magnetic_flux_coordinate'). Emitting it without the "
                f"coordinate drops the index and mints a malformed name."
            )
        return self

    def _to_model_dict(self) -> dict[str, Any]:
        """Build the flat ISN ``StandardName`` model dict from segment fields.

        Canonical joining is DELEGATED to the ISN model layer rather than
        re-derived in codex: a prefix operator goes in ``transformation`` and
        ISN decides bare (``volume_averaged_X``) vs ``_of_``
        (``time_derivative_of_X``); a postfix goes in ``decomposition``. The
        operator kind is taken from the registry (authoritative) and falls
        back to the LLM-supplied ``operator_kind`` only for unregistered
        tokens. Qualifiers fold into the base compound in the order the model
        emitted them (ISN re-parses and enforces canonical order downstream).

        One codex-side adjustment remains: a bare-prefix transformation that
        co-occurs with a projection axis fuses into a single registered
        compound axis token (``normalized`` + ``radial`` ->
        ``normalized_radial``), because ISN forbids ``transformation`` and
        ``component`` together. The fusion is detected from the live axis
        vocabulary, not a hardcoded list.
        """
        d: dict[str, Any] = {}

        # Base (+ qualifiers folded as a canonical-order compound prefix).
        base = self.base_token
        if self.qualifiers:
            base = "_".join([*self.qualifiers, base])

        # Binary operator short-circuit: operand A is the (qualifier-folded)
        # base; operand B is the secondary_base string. ISN re-parses both
        # operand strings and renders ``<op>_of_<A>_<sep>_<B>``. Projection and
        # locus on a binary expression are not modelled here (rare); a due_to_
        # process still attaches.
        binary_model_op = self._binary_model_op()
        if binary_model_op is not None and self.secondary_base:
            d["binary_operator"] = binary_model_op
            d["physical_base"] = base
            d["secondary_base"] = self.secondary_base
            if self.process_token is not None:
                d["process"] = self.process_token
            return d

        if self.base_kind == "geometry":
            d["geometric_base"] = base
        else:
            d["physical_base"] = base

        projection_axis = self.projection_axis
        projection_shape = self.projection_shape or "component"

        # Operator → transformation (prefix) / decomposition (postfix).
        if self.operator_token is not None:
            kind = (
                _operator_registry_kinds().get(self.operator_token)
                or self.operator_kind
            )
            # Indexed operators carry a bound coordinate fused into the token,
            # exactly as the ISN parser/model layer represents them
            # (``derivative_with_respect_to_<coord>``). Without this the
            # coordinate is silently dropped and a malformed
            # ``derivative_with_respect_to_of_<base>`` is minted.
            op_token = self.operator_token
            if self.operator_coordinate:
                op_token = f"{self.operator_token}_{self.operator_coordinate}"
            if kind == "unary_postfix":
                d["decomposition"] = op_token
            else:
                # Prefix transformation. Fuse with the projection axis when the
                # compound is a registered axis token (ISN represents
                # normalized_radial as one component, and rejects
                # transformation + component together).
                compound = (
                    f"{op_token}_{projection_axis}"
                    if projection_axis is not None
                    else None
                )
                if compound is not None and compound in _projection_axis_tokens().get(
                    projection_shape, frozenset()
                ):
                    projection_axis = compound
                else:
                    d["transformation"] = op_token

        # Projection → component / coordinate.
        if projection_axis is not None:
            if projection_shape == "coordinate":
                d["coordinate"] = projection_axis
            else:
                d["component"] = projection_axis

        # Locus → object / position(+value) / geometry / region. Map by type
        # (and relation for the position split), mirroring ISN's locus matrix;
        # an out-of-matrix relation is normalised by the type-based mapping.
        if (
            self.locus_token is not None
            and self.locus_relation is not None
            and self.locus_type is not None
        ):
            lt = self.locus_type
            if lt == "entity":
                d["object"] = self.locus_token
            elif lt == "region":
                d["region"] = self.locus_token
            elif lt == "position":
                if self.locus_relation == "of":
                    d["geometry"] = self.locus_token
                else:  # 'at' (or normalised from an invalid relation)
                    d["position"] = self.locus_token
                    if self.locus_value is not None:
                        d["position_value"] = self.locus_value
            else:  # geometry-type locus
                d["geometry"] = self.locus_token

        # Mechanism → process.
        if self.process_token is not None:
            d["process"] = self.process_token

        return d

    def compose_name(self) -> str:
        """Build the canonical standard name via the ISN model-layer composer.

        Delegates to ISN's public ``compose_standard_name``, which owns the
        bare-vs-``_of_`` prefix routing, postfix joins, qualifier ordering, and
        compound-axis fusion — so codex inherits ISN's grammar authority rather
        than re-deriving it in a low-level IR patch.
        """
        from imas_standard_names.grammar import compose_standard_name

        return compose_standard_name(self._to_model_dict())

    def to_ir(self) -> Any:
        """Return the ISN IR for this segment set's canonical name.

        Derived by parsing the canonical name so the IR is always consistent
        with :meth:`compose_name`. Raises if the segments do not form an
        expressible canonical name (callers already guard ``compose_name``).
        """
        from imas_standard_names.grammar.parser import parse

        return parse(self.compose_name()).ir


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
    token: str = Field(description="Proposed token value for the grammar segment")
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
                        "token": token,
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
# Description review — 4-dimensional rubric for compose-time descriptions
# =============================================================================


class StandardNameQualityScoreDescription(BaseModel):
    """4-dimensional quality score for the SHORT compose-time description.

    Scores the one-line description that a name-generation model emits
    alongside the name (NOT the longer enrichment ``documentation``). Used
    by the benchmark to turn the short description into a scored
    discriminator once names converge across models. Normalised over 80.
    """

    physics_accuracy: int = Field(
        ge=0,
        le=20,
        description=(
            "No hallucinated physics; consistent with the DD source context "
            "provided (0-20)"
        ),
    )
    specificity: int = Field(
        ge=0,
        le=20,
        description=(
            "Says what the quantity IS — species, location, conditions — not "
            "generic filler (0-20)"
        ),
    )
    consistency: int = Field(
        ge=0,
        le=20,
        description=(
            "Description and name describe the SAME quantity; flag drift (0-20)"
        ),
    )
    concision: int = Field(
        ge=0,
        le=20,
        description=(
            "One-to-two sentences, no boilerplate, no units-in-prose restating "
            "the unit field (0-20)"
        ),
    )

    @property
    def total(self) -> int:
        return (
            self.physics_accuracy + self.specificity + self.consistency + self.concision
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


class StandardNameQualityReviewDescription(BaseModel):
    """Review of one compose-time description using the 4-dimensional rubric."""

    source_id: str = Field(description="Source entity ID being reviewed")
    standard_name: str = Field(description="The standard name under review")
    scores: StandardNameQualityScoreDescription = Field(
        description="4-dimensional description quality scores"
    )
    reasoning: str = Field(
        description="One-line justification covering the four dimensions"
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


class StandardNameQualityReviewDescriptionBatch(BaseModel):
    """LLM response for compose-time description quality review of a batch."""

    reviews: list[StandardNameQualityReviewDescription]


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


class EnrichedParentDescription(BaseModel):
    """LLM response model for a single enrich_parents call.

    A derived parent abstracts over its accepted ``HAS_PARENT`` children; this
    response carries ONLY a concise description GENERALISED over those
    children — the common physical quantity they share.  It must NOT invent
    physics beyond what the children attest, and must NOT alter the name, unit,
    kind, or any identity field (all fixed by the derived parent).  The full
    long-form documentation is produced later by generate_docs.
    """

    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description=(
            "1-2 sentence technical description of the common physical quantity "
            "the parent's children share — the generalised meaning, not any one "
            "child's specifics (American spelling, no LaTeX, no markdown links, "
            "<= 500 chars)."
        ),
    )

    model_config = {"extra": "ignore"}
