"""Round trips for source convention records in the generated graph models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from imas_codex.graph.models import (
    COCOSClass,
    ConventionObservation,
    ConventionStatus,
    FacilitySignal,
    SignalSource,
)
from imas_codex.graph.schema import GraphSchema


def test_source_convention_fields_round_trip() -> None:
    now = datetime(2026, 10, 9, tzinfo=UTC)
    values = {
        "id": "jet:magnetics:flux",
        "facility_id": "jet",
        "group_key": "magnetics:flux",
        "status": "enriched",
        "convention_dependent": True,
        "p_convention_dependent": 0.91,
        "cocos_class": "psi_like",
        "cocos_class_distribution": '{"psi_like": 0.91, "ip_like": 0.09}',
        "convention_writer": "raw diagnostic",
        "cocos_survivors": [3],
        "cocos": 3,
        "cocos_confidence": 0.78,
        "convention_status": "needs_followup",
        "convention_question": "Which handedness was used?",
        "convention_judged_at": now,
        "convention_probed_at": now,
        "convention_settled_at": now,
        "convention_claimed_at": now,
        "convention_claim_token": "claim-token",
    }
    source = SignalSource.model_validate(values)
    restored = SignalSource.model_validate(source.model_dump(mode="json"))

    for field, value in values.items():
        assert getattr(restored, field) == value
    assert restored.cocos_class == COCOSClass.psi_like
    assert restored.convention_status == ConventionStatus.needs_followup


def test_channel_sign_round_trip_without_signal_values() -> None:
    signal = FacilitySignal.model_validate(
        {
            "id": "jet:magnetics:flux:1",
            "facility_id": "jet",
            "accessor": "flux(1)",
            "channel_sign": -1,
        }
    )
    data = signal.model_dump(mode="json", exclude_none=True)
    assert FacilitySignal.model_validate(data).channel_sign == -1
    assert "value" not in data
    assert "values" not in data


def test_observation_round_trip_and_edge() -> None:
    observation = ConventionObservation.model_validate(
        {
            "id": "jet:magnetics:flux:123:flux",
            "facility_id": "jet",
            "observed_for": "jet:magnetics:flux",
            "shot": 123,
            "quantity": "flux_loop_response",
            "observed_sign": -1,
            "plasma_current_sign": 1,
            "data_access": "jet:ppf",
            "observed_at": datetime(2026, 10, 9, tzinfo=UTC),
        }
    )
    restored = ConventionObservation.model_validate(observation.model_dump(mode="json"))
    assert restored == observation

    relationships = GraphSchema().get_relationships_from("ConventionObservation")
    assert any(
        rel.slot_name == "observed_for"
        and rel.to_class == "SignalSource"
        and rel.cypher_type == "OBSERVED_FOR"
        for rel in relationships
    )


def test_cocos_class_rejects_undeclared_value() -> None:
    assert set(COCOSClass) == {
        COCOSClass.none,
        COCOSClass.psi_like,
        COCOSClass.dodpsi_like,
        COCOSClass.ip_like,
        COCOSClass.b0_like,
        COCOSClass.q_like,
        COCOSClass.tor_angle_like,
        COCOSClass.pol_angle_like,
        COCOSClass.compound,
    }
    with pytest.raises(ValidationError):
        SignalSource.model_validate(
            {
                "id": "jet:magnetics:flux",
                "facility_id": "jet",
                "group_key": "magnetics:flux",
                "status": "enriched",
                "cocos_class": "outside_enum",
            }
        )
