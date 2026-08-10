"""Validation and canonical identity for typed LLM prompt context."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from imas_codex.llm.context_models import (
    AuthorityClass,
    ContextClaim,
    ContextField,
    ContextScope,
    ContextValue,
    ContextValueKind,
    EnvelopeFingerprints,
    FacetState,
    ItemFingerprints,
    PromptEnvelope,
    PromptItemEnvelope,
    SemanticFacet,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_VALUE_CHANNELS = {
    ContextValueKind.text: "text_value",
    ContextValueKind.integer: "integer_value",
    ContextValueKind.number: "number_value",
    ContextValueKind.boolean: "boolean_value",
    ContextValueKind.text_list: "text_items",
    ContextValueKind.structured: "fields",
}
_AUTHORITY_PLACEMENTS = {
    "source_facts": AuthorityClass.pinned_source_fact,
    "approved_resolutions": AuthorityClass.approved_local_resolution,
    "reviewer_intent": AuthorityClass.reviewer_intent,
    "comparators": AuthorityClass.non_binding_comparator,
    "provenance": AuthorityClass.mutable_provenance,
}
_ITEM_SCOPES = {ContextScope.exact_item}
_COMPARATOR_SCOPES = {ContextScope.exact_item, ContextScope.family}
_BATCH_COMPARATOR_SCOPES = {ContextScope.batch, ContextScope.global_static}
_MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000


def _decode_and_inspect_image(
    data_base64: str, location: str
) -> tuple[bytes, str, int, int]:
    """Decode and validate one bounded, single-frame image payload."""
    if len(data_base64) > ((_MAX_ATTACHMENT_BYTES + 2) // 3) * 4:
        raise ContextEnvelopeError(
            f"{location}.data_base64 exceeds the hard byte bound"
        )
    try:
        content = b64decode(data_base64, validate=True)
    except (Base64Error, ValueError) as exc:
        raise ContextEnvelopeError(f"{location}.data_base64 is invalid") from exc
    if not content or len(content) > _MAX_ATTACHMENT_BYTES:
        raise ContextEnvelopeError(
            f"{location} decoded content exceeds the hard byte bound"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                media_type = image.get_format_mimetype()
                width, height = image.size
                if not media_type or not media_type.startswith("image/"):
                    raise ContextEnvelopeError(
                        f"{location} has an unsupported image signature"
                    )
                if (
                    getattr(image, "is_animated", False)
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ContextEnvelopeError(
                        f"{location} animated or multi-frame images are unsupported"
                    )
                if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                    raise ContextEnvelopeError(
                        f"{location} image dimensions exceed the hard pixel bound"
                    )
                image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ContextEnvelopeError(f"{location} is not a safe supported image") from exc
    return content, media_type, width, height


class ContextEnvelopeError(ValueError):
    """Base error for invalid authority envelopes."""


class AuthorityPlacementError(ContextEnvelopeError):
    """A claim was placed in a channel that cannot carry its authority class."""


class AuthorityConflictError(ContextEnvelopeError):
    """Source-backed claims disagree about the same semantic kind."""


class ObligationDerivationError(ContextEnvelopeError):
    """A semantic obligation is not derived from matching authority claims."""


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json", by_alias=True, exclude_none=True))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain(item) for item in value]
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ContextEnvelopeError("Canonical context cannot contain NaN or infinity")
    return value


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for a typed context payload."""
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_fingerprint(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical typed JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(digest: str, location: str) -> None:
    if not _DIGEST_RE.fullmatch(digest):
        raise ContextEnvelopeError(f"{location} must be a lowercase SHA-256 digest")


def _populated_channels(value: ContextValue | ContextField) -> set[str]:
    populated: set[str] = set()
    for channel in _VALUE_CHANNELS.values():
        channel_value = getattr(value, channel, None)
        if channel_value is not None:
            populated.add(channel)
    return populated


def _validate_typed_value(value: ContextValue | ContextField, location: str) -> None:
    expected = _VALUE_CHANNELS[ContextValueKind(_enum_value(value.value_kind))]
    populated = _populated_channels(value)
    if populated != {expected}:
        raise ContextEnvelopeError(
            f"{location} must populate only {expected}; found {sorted(populated)}"
        )
    if expected in {"text_items", "fields"} and not getattr(value, expected):
        raise ContextEnvelopeError(f"{location}.{expected} must not be empty")
    if isinstance(value, ContextValue) and value.fields:
        names: set[str] = set()
        for index, field in enumerate(value.fields):
            if field.name in names:
                raise ContextEnvelopeError(
                    f"{location}.fields contains duplicate name {field.name!r}"
                )
            names.add(field.name)
            _validate_typed_value(field, f"{location}.fields[{index}]")


def _validate_claim(claim: ContextClaim, location: str) -> None:
    for field_name in (
        "claim_id",
        "kind",
        "source_ref",
        "source_field",
        "source_version",
    ):
        if not getattr(claim, field_name).strip():
            raise ContextEnvelopeError(f"{location}.{field_name} must not be blank")
    _validate_digest(claim.source_digest, f"{location}.source_digest")
    _validate_typed_value(claim.value, f"{location}.value")


def _validate_claim_list(
    claims: Sequence[ContextClaim],
    *,
    expected_class: AuthorityClass,
    allowed_scopes: set[ContextScope],
    location: str,
) -> None:
    for index, claim in enumerate(claims):
        claim_location = f"{location}[{index}]"
        _validate_claim(claim, claim_location)
        actual_class = AuthorityClass(_enum_value(claim.authority_class))
        if actual_class is not expected_class:
            raise AuthorityPlacementError(
                f"{claim_location} carries {actual_class.value}, expected "
                f"{expected_class.value}"
            )
        actual_scope = ContextScope(_enum_value(claim.scope))
        if actual_scope not in allowed_scopes:
            allowed = ", ".join(sorted(scope.value for scope in allowed_scopes))
            raise AuthorityPlacementError(
                f"{claim_location} has scope {actual_scope.value}; allowed: {allowed}"
            )


def _claim_value(claim: ContextClaim) -> str:
    return canonical_json(claim.value)


def _validate_authority_consistency(item: PromptItemEnvelope, location: str) -> None:
    by_kind: dict[str, list[ContextClaim]] = {}
    for claim in [*item.source_facts, *item.approved_resolutions]:
        by_kind.setdefault(claim.kind, []).append(claim)
    for kind, claims in by_kind.items():
        distinct = {_claim_value(claim) for claim in claims}
        if len(distinct) > 1:
            refs = ", ".join(sorted(claim.source_ref for claim in claims))
            raise AuthorityConflictError(
                f"{location} has conflicting source authority for {kind!r}: {refs}"
            )

    authority_by_id = {
        claim.claim_id: claim
        for claim in [*item.source_facts, *item.approved_resolutions]
    }
    for facet_name in item.obligations.__class__.model_fields:
        facet = getattr(item.obligations, facet_name)
        _validate_facet(
            facet_name,
            facet,
            authority_by_id=authority_by_id,
            location=f"{location}.obligations.{facet_name}",
        )

    known_facets = {
        facet_name: facet.value
        for facet_name in item.obligations.__class__.model_fields
        if (facet := getattr(item.obligations, facet_name))
        if FacetState(_enum_value(facet.state)) is FacetState.known
    }
    for claim in item.reviewer_intent:
        expected = known_facets.get(claim.kind)
        if expected is None:
            continue
        value = claim.value.text_value
        if value != expected:
            raise AuthorityConflictError(
                f"{location}.reviewer_intent cannot rewrite known {claim.kind!r} "
                f"from {expected!r} to {value!r}"
            )


def _validate_facet(
    facet_name: str,
    facet: SemanticFacet,
    *,
    authority_by_id: Mapping[str, ContextClaim],
    location: str,
) -> None:
    state = FacetState(_enum_value(facet.state))
    source_ids = facet.source_claim_ids or []
    if state is not FacetState.known:
        if facet.value is not None or source_ids:
            raise ObligationDerivationError(
                f"{location} {state.value} state cannot carry a value or sources"
            )
        return
    if facet.value is None or not facet.value.strip():
        raise ObligationDerivationError(f"{location} known state requires a value")
    if not source_ids:
        raise ObligationDerivationError(
            f"{location} known state requires source_claim_ids"
        )
    for source_id in source_ids:
        claim = authority_by_id.get(source_id)
        if claim is None:
            raise ObligationDerivationError(
                f"{location} references unknown authority claim {source_id!r}"
            )
        if claim.kind != facet_name:
            raise ObligationDerivationError(
                f"{location} source {source_id!r} has kind {claim.kind!r}"
            )
        if claim.value.text_value != facet.value:
            raise ObligationDerivationError(
                f"{location} value does not match source claim {source_id!r}"
            )


def _validate_item(item: PromptItemEnvelope, index: int) -> None:
    location = f"batch_items[{index}]"
    if not item.item_id.strip():
        raise ContextEnvelopeError(f"{location}.item_id must not be blank")
    for field_name, expected_class in _AUTHORITY_PLACEMENTS.items():
        claims = getattr(item, field_name)
        allowed_scopes = (
            _COMPARATOR_SCOPES if field_name == "comparators" else _ITEM_SCOPES
        )
        _validate_claim_list(
            claims,
            expected_class=expected_class,
            allowed_scopes=allowed_scopes,
            location=f"{location}.{field_name}",
        )
    if item.mutable_candidate is not None:
        _validate_claim_list(
            [item.mutable_candidate],
            expected_class=AuthorityClass.mutable_provenance,
            allowed_scopes=_ITEM_SCOPES,
            location=f"{location}.mutable_candidate",
        )
    attachment_ids: set[str] = set()
    for attachment_index, attachment in enumerate(item.attachments or []):
        attachment_location = f"{location}.attachments[{attachment_index}]"
        for field_name in ("attachment_id", "media_type"):
            if not getattr(attachment, field_name).strip():
                raise ContextEnvelopeError(
                    f"{attachment_location}.{field_name} must not be blank"
                )
        if attachment.attachment_id in attachment_ids:
            raise ContextEnvelopeError(
                f"{location}.attachments contains duplicate attachment_id"
            )
        attachment_ids.add(attachment.attachment_id)
        _validate_digest(
            attachment.content_digest, f"{attachment_location}.content_digest"
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (attachment.byte_length, attachment.width, attachment.height)
        ):
            raise ContextEnvelopeError(
                f"{attachment_location} dimensions and byte_length must be positive integers"
            )
        content, derived_media_type, derived_width, derived_height = (
            _decode_and_inspect_image(attachment.data_base64, attachment_location)
        )
        if len(content) != attachment.byte_length:
            raise ContextEnvelopeError(
                f"{attachment_location}.byte_length does not match decoded content"
            )
        if hashlib.sha256(content).hexdigest() != attachment.content_digest:
            raise ContextEnvelopeError(
                f"{attachment_location}.content_digest does not match decoded content"
            )
        if attachment.media_type != derived_media_type:
            raise ContextEnvelopeError(
                f"{attachment_location}.media_type does not match decoded content"
            )
        if (attachment.width, attachment.height) != (derived_width, derived_height):
            raise ContextEnvelopeError(
                f"{attachment_location} dimensions do not match decoded content"
            )
    _validate_authority_consistency(item, location)


def _coerce_model[ModelT: BaseModel](
    model_type: type[ModelT], value: ModelT | Mapping[str, Any]
) -> ModelT:
    if isinstance(value, model_type):
        return value
    try:
        return model_type.model_validate_json(canonical_json(value))
    except (TypeError, ValidationError) as exc:
        raise ContextEnvelopeError(str(exc)) from exc


def validate_envelope(
    envelope: PromptEnvelope | Mapping[str, Any],
) -> PromptEnvelope:
    """Coerce and validate a prompt envelope without graph or provider access."""
    validated = _coerce_model(PromptEnvelope, envelope)
    for field_name in (
        "schema_version",
        "callsite_id",
        "service",
        "seat",
        "task_kind",
        "policy_id",
    ):
        if not getattr(validated, field_name).strip():
            raise ContextEnvelopeError(f"{field_name} must not be blank")
    if not validated.static_context:
        raise ContextEnvelopeError("static_context must not be empty")
    for index, static_ref in enumerate(validated.static_context):
        if not static_ref.name.strip() or not static_ref.source_version.strip():
            raise ContextEnvelopeError(
                f"static_context[{index}] name and source_version must not be blank"
            )
        _validate_digest(
            static_ref.source_digest, f"static_context[{index}].source_digest"
        )
    if not validated.batch_items:
        raise ContextEnvelopeError("batch_items must not be empty")
    item_ids = [item.item_id for item in validated.batch_items]
    if len(item_ids) != len(set(item_ids)):
        raise ContextEnvelopeError("batch_items must have unique item_id values")
    claim_ids: list[str] = []
    for index, item in enumerate(validated.batch_items):
        _validate_item(item, index)
        claim_ids.extend(
            claim.claim_id
            for claim in (
                *item.source_facts,
                *item.approved_resolutions,
                *item.reviewer_intent,
                *item.comparators,
                *item.provenance,
            )
        )
        if item.mutable_candidate is not None:
            claim_ids.append(item.mutable_candidate.claim_id)
    _validate_claim_list(
        validated.batch_comparators,
        expected_class=AuthorityClass.non_binding_comparator,
        allowed_scopes=_BATCH_COMPARATOR_SCOPES,
        location="batch_comparators",
    )
    claim_ids.extend(claim.claim_id for claim in validated.batch_comparators)
    if len(claim_ids) != len(set(claim_ids)):
        raise ContextEnvelopeError("claim_id values must be unique within an envelope")
    return validated


def _item_payloads(item: PromptItemEnvelope) -> tuple[dict[str, Any], ...]:
    authority = {
        "item_id": item.item_id,
        "source_facts": item.source_facts,
        "approved_resolutions": item.approved_resolutions,
        "obligations": item.obligations,
        "reviewer_intent": item.reviewer_intent,
    }
    comparators = {"item_id": item.item_id, "comparators": item.comparators}
    provenance = {
        "item_id": item.item_id,
        "provenance": item.provenance,
        "mutable_candidate": item.mutable_candidate,
        "attachments": [
            {
                "attachment_id": attachment.attachment_id,
                "media_type": attachment.media_type,
                "content_digest": attachment.content_digest,
                "byte_length": attachment.byte_length,
                "width": attachment.width,
                "height": attachment.height,
            }
            for attachment in (item.attachments or [])
        ],
    }
    return authority, comparators, provenance


def fingerprint_item(item: PromptItemEnvelope) -> ItemFingerprints:
    """Return fingerprints whose values do not depend on batch position."""
    authority, comparators, provenance = _item_payloads(item)
    return ItemFingerprints(
        item_id=item.item_id,
        authority_fingerprint=canonical_fingerprint(authority),
        comparator_fingerprint=canonical_fingerprint(comparators),
        provenance_fingerprint=canonical_fingerprint(provenance),
    )


def fingerprint_envelope(
    envelope: PromptEnvelope | Mapping[str, Any],
) -> EnvelopeFingerprints:
    """Validate an envelope and compute independent canonical fingerprints."""
    validated = validate_envelope(envelope)
    static_payload = {
        "schema_version": validated.schema_version,
        "callsite_id": validated.callsite_id,
        "service": validated.service,
        "seat": validated.seat,
        "task_kind": validated.task_kind,
        "policy_id": validated.policy_id,
        "static_context": validated.static_context,
    }
    authority_items: list[dict[str, Any]] = []
    comparator_items: list[dict[str, Any]] = []
    provenance_items: list[dict[str, Any]] = []
    for item in validated.batch_items:
        authority, comparators, provenance = _item_payloads(item)
        authority_items.append(authority)
        comparator_items.append(comparators)
        provenance_items.append(provenance)
    static_fingerprint = canonical_fingerprint(static_payload)
    authority_fingerprint = canonical_fingerprint(authority_items)
    comparator_fingerprint = canonical_fingerprint(
        {
            "items": comparator_items,
            "batch_comparators": validated.batch_comparators,
        }
    )
    provenance_fingerprint = canonical_fingerprint(provenance_items)
    request_fingerprint = canonical_fingerprint(
        {
            "static": static_fingerprint,
            "authority": authority_fingerprint,
            "comparators": comparator_fingerprint,
            "provenance": provenance_fingerprint,
        }
    )
    return EnvelopeFingerprints(
        static_fingerprint=static_fingerprint,
        authority_fingerprint=authority_fingerprint,
        comparator_fingerprint=comparator_fingerprint,
        provenance_fingerprint=provenance_fingerprint,
        request_fingerprint=request_fingerprint,
    )


__all__ = [
    "AuthorityConflictError",
    "AuthorityPlacementError",
    "ContextEnvelopeError",
    "ObligationDerivationError",
    "canonical_fingerprint",
    "canonical_json",
    "fingerprint_envelope",
    "fingerprint_item",
    "validate_envelope",
]
