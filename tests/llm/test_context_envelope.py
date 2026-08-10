"""Authority separation and canonical identity for typed prompt envelopes."""

from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from imas_codex.llm.context_envelope import (
    AuthorityConflictError,
    AuthorityPlacementError,
    ContextEnvelopeError,
    ObligationDerivationError,
    canonical_fingerprint,
    canonical_json,
    fingerprint_envelope,
    fingerprint_item,
    validate_envelope,
)

_SOURCE_DIGEST = "a" * 64
_TEMPLATE_DIGEST = "b" * 64


def _image_bytes(
    *,
    width: int = 10,
    height: int = 20,
    image_format: str = "PNG",
    animated: bool = False,
) -> bytes:
    output = BytesIO()
    first = Image.new("RGB", (width, height), color="navy")
    if animated:
        second = Image.new("RGB", (width, height), color="gold")
        first.save(
            output,
            format=image_format,
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
        )
    else:
        first.save(output, format=image_format)
    return output.getvalue()


def _text_value(value: str) -> dict[str, object]:
    return {"value_kind": "text", "text_value": value}


def _claim(
    claim_id: str,
    authority_class: str,
    kind: str,
    value: str,
    *,
    scope: str = "exact_item",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "authority_class": authority_class,
        "kind": kind,
        "value": _text_value(value),
        "source_ref": f"source:{claim_id}",
        "source_field": kind,
        "source_version": "source-release",
        "source_digest": _SOURCE_DIGEST,
        "scope": scope,
    }


def _unknown_facet() -> dict[str, object]:
    return {"state": "unknown", "source_claim_ids": []}


def _obligations() -> dict[str, object]:
    facets = {
        name: _unknown_facet()
        for name in (
            "quantity",
            "carrier",
            "representation",
            "coordinate_frame",
            "owner",
            "locus",
            "axis",
            "section_plane",
            "unit",
            "dd_version",
            "kind",
            "physics_domain",
            "cocos",
            "family",
        )
    }
    facets["owner"] = {
        "state": "known",
        "value": "active_conductor",
        "source_claim_ids": ["owner-source"],
    }
    return facets


@pytest.fixture
def envelope_data() -> dict[str, object]:
    return {
        "schema_version": "prompt-context",
        "callsite_id": "compose_name",
        "service": "standard-names",
        "seat": "sn-compose",
        "task_kind": "name_composition",
        "policy_id": "source_authority",
        "static_context": [
            {
                "name": "sn/generate_name_system",
                "kind": "template",
                "role": "system",
                "source_version": "template-release",
                "source_digest": _TEMPLATE_DIGEST,
            }
        ],
        "batch_items": [
            {
                "item_id": "equilibrium/path",
                "source_facts": [
                    _claim(
                        "owner-source",
                        "pinned_source_fact",
                        "owner",
                        "active_conductor",
                    )
                ],
                "approved_resolutions": [],
                "obligations": _obligations(),
                "reviewer_intent": [],
                "comparators": [
                    _claim(
                        "nearby-path",
                        "non_binding_comparator",
                        "owner",
                        "passive_loop",
                        scope="family",
                    )
                ],
                "provenance": [
                    _claim(
                        "prior-description",
                        "mutable_provenance",
                        "description",
                        "Generated description",
                    )
                ],
            }
        ],
        "batch_comparators": [],
    }


def test_valid_envelope_is_strict_frozen_and_explicit(
    envelope_data: dict[str, object],
) -> None:
    envelope = validate_envelope(envelope_data)

    assert envelope.model_config["strict"] is True
    assert envelope.model_config["frozen"] is True
    assert envelope.batch_items[0].obligations.section_plane.state == "unknown"
    with pytest.raises(ValidationError):
        envelope.service = "other"


def test_unknown_context_keys_are_rejected(envelope_data: dict[str, object]) -> None:
    envelope_data["undeclared_context"] = "must fail"

    with pytest.raises(ContextEnvelopeError, match="undeclared_context"):
        validate_envelope(envelope_data)


def test_authority_requires_source_version_and_digest(
    envelope_data: dict[str, object],
) -> None:
    claim = envelope_data["batch_items"][0]["source_facts"][0]  # type: ignore[index]
    claim["source_version"] = ""  # type: ignore[index]

    with pytest.raises(ContextEnvelopeError, match="source_version"):
        validate_envelope(envelope_data)

    claim["source_version"] = "source-release"  # type: ignore[index]
    claim["source_digest"] = "not-a-digest"  # type: ignore[index]
    with pytest.raises(ContextEnvelopeError, match="SHA-256"):
        validate_envelope(envelope_data)


def test_comparator_cannot_be_placed_in_source_authority(
    envelope_data: dict[str, object],
) -> None:
    claim = envelope_data["batch_items"][0]["source_facts"][0]  # type: ignore[index]
    claim["authority_class"] = "non_binding_comparator"  # type: ignore[index]

    with pytest.raises(AuthorityPlacementError, match="expected pinned_source_fact"):
        validate_envelope(envelope_data)


def test_identity_authority_must_have_exact_item_scope(
    envelope_data: dict[str, object],
) -> None:
    claim = envelope_data["batch_items"][0]["source_facts"][0]  # type: ignore[index]
    claim["scope"] = "batch"  # type: ignore[index]

    with pytest.raises(AuthorityPlacementError, match="allowed: exact_item"):
        validate_envelope(envelope_data)


def test_conflicting_source_and_resolution_authority_fails(
    envelope_data: dict[str, object],
) -> None:
    item = envelope_data["batch_items"][0]  # type: ignore[index]
    item["approved_resolutions"] = [  # type: ignore[index]
        _claim(
            "owner-resolution",
            "approved_local_resolution",
            "owner",
            "passive_loop",
        )
    ]

    with pytest.raises(AuthorityConflictError, match="conflicting source authority"):
        validate_envelope(envelope_data)


def test_known_obligation_must_match_pinned_authority(
    envelope_data: dict[str, object],
) -> None:
    owner = envelope_data["batch_items"][0]["obligations"]["owner"]  # type: ignore[index]
    owner["value"] = "passive_loop"  # type: ignore[index]

    with pytest.raises(ObligationDerivationError, match="does not match"):
        validate_envelope(envelope_data)


def test_optional_absence_is_explicit_unknown_or_not_applicable(
    envelope_data: dict[str, object],
) -> None:
    plane = envelope_data["batch_items"][0]["obligations"]["section_plane"]  # type: ignore[index]
    plane["state"] = "not_applicable"  # type: ignore[index]
    assert (
        validate_envelope(envelope_data).batch_items[0].obligations.section_plane.value
        is None
    )

    plane["value"] = "poloidal"  # type: ignore[index]
    with pytest.raises(ObligationDerivationError, match="cannot carry"):
        validate_envelope(envelope_data)


def test_comparator_drift_changes_only_comparator_and_request_fingerprints(
    envelope_data: dict[str, object],
) -> None:
    before = fingerprint_envelope(envelope_data)
    changed = deepcopy(envelope_data)
    comparator = changed["batch_items"][0]["comparators"][0]  # type: ignore[index]
    comparator["value"]["text_value"] = "diagnostic_owner"  # type: ignore[index]
    after = fingerprint_envelope(changed)

    assert after.static_fingerprint == before.static_fingerprint
    assert after.authority_fingerprint == before.authority_fingerprint
    assert after.provenance_fingerprint == before.provenance_fingerprint
    assert after.comparator_fingerprint != before.comparator_fingerprint
    assert after.request_fingerprint != before.request_fingerprint


def test_provenance_drift_is_independent(envelope_data: dict[str, object]) -> None:
    before = fingerprint_envelope(envelope_data)
    changed = deepcopy(envelope_data)
    provenance = changed["batch_items"][0]["provenance"][0]  # type: ignore[index]
    provenance["value"]["text_value"] = "Different generated prose"  # type: ignore[index]
    after = fingerprint_envelope(changed)

    assert after.static_fingerprint == before.static_fingerprint
    assert after.authority_fingerprint == before.authority_fingerprint
    assert after.comparator_fingerprint == before.comparator_fingerprint
    assert after.provenance_fingerprint != before.provenance_fingerprint


def test_attachment_digest_and_dimensions_are_validated(
    envelope_data: dict[str, object],
) -> None:
    content = _image_bytes()
    envelope_data["batch_items"][0]["attachments"] = [  # type: ignore[index]
        {
            "attachment_id": "image:one",
            "media_type": "image/png",
            "content_digest": hashlib.sha256(content).hexdigest(),
            "data_base64": base64.b64encode(content).decode(),
            "byte_length": len(content),
            "width": 10,
            "height": 20,
        }
    ]

    before = fingerprint_envelope(envelope_data)
    changed = deepcopy(envelope_data)
    changed["batch_items"][0]["attachments"][0]["width"] = 11  # type: ignore[index]
    with pytest.raises(ContextEnvelopeError, match="dimensions do not match"):
        fingerprint_envelope(changed)
    assert before.provenance_fingerprint

    changed = deepcopy(envelope_data)
    changed["batch_items"][0]["attachments"][0]["content_digest"] = "b" * 64  # type: ignore[index]
    with pytest.raises(ContextEnvelopeError, match="does not match"):
        validate_envelope(changed)


def test_animated_images_are_rejected(envelope_data: dict[str, object]) -> None:
    content = _image_bytes(image_format="GIF", animated=True)
    envelope_data["batch_items"][0]["attachments"] = [  # type: ignore[index]
        {
            "attachment_id": "image:animated",
            "media_type": "image/gif",
            "content_digest": hashlib.sha256(content).hexdigest(),
            "data_base64": base64.b64encode(content).decode(),
            "byte_length": len(content),
            "width": 10,
            "height": 20,
        }
    ]

    with pytest.raises(ContextEnvelopeError, match="multi-frame"):
        validate_envelope(envelope_data)


def test_item_fingerprint_does_not_depend_on_batch_position(
    envelope_data: dict[str, object],
) -> None:
    first = validate_envelope(envelope_data).batch_items[0]
    second_data = deepcopy(envelope_data["batch_items"][0])  # type: ignore[index]
    second_data["item_id"] = "equilibrium/other"  # type: ignore[index]
    for channel in (
        "source_facts",
        "comparators",
        "provenance",
    ):
        for claim in second_data[channel]:  # type: ignore[index]
            claim["claim_id"] = f"other-{claim['claim_id']}"
    second_data["obligations"]["owner"]["source_claim_ids"] = [  # type: ignore[index]
        "other-owner-source"
    ]
    reordered = deepcopy(envelope_data)
    reordered["batch_items"] = [second_data, deepcopy(envelope_data["batch_items"][0])]  # type: ignore[index]
    second = validate_envelope(reordered).batch_items[1]

    assert fingerprint_item(first) == fingerprint_item(second)


def test_batch_order_changes_request_identity(envelope_data: dict[str, object]) -> None:
    second_data = deepcopy(envelope_data["batch_items"][0])  # type: ignore[index]
    second_data["item_id"] = "equilibrium/other"  # type: ignore[index]
    for channel in ("source_facts", "comparators", "provenance"):
        for claim in second_data[channel]:  # type: ignore[index]
            claim["claim_id"] = f"other-{claim['claim_id']}"
    second_data["obligations"]["owner"]["source_claim_ids"] = [  # type: ignore[index]
        "other-owner-source"
    ]
    ordered = deepcopy(envelope_data)
    ordered["batch_items"].append(second_data)  # type: ignore[union-attr]
    reversed_items = deepcopy(ordered)
    reversed_items["batch_items"].reverse()  # type: ignore[union-attr]

    assert (
        fingerprint_envelope(ordered).request_fingerprint
        != fingerprint_envelope(reversed_items).request_fingerprint
    )


def test_canonical_json_is_key_order_independent() -> None:
    left = {"beta": [2, 1], "alpha": {"value": True}}
    right = {"alpha": {"value": True}, "beta": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_fingerprint(left) == canonical_fingerprint(right)
