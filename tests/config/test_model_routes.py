"""Every declared language-model seat resolves to its serving endpoint."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
import pytest

from iter_codex import settings
from iter_codex.discovery.base import llm
from tests.config.test_model_seats import _declared_models

_DECLARED = [
    (seat, model)
    for seat, model in _declared_models().items()
    if seat != "embedding.model"
]


def test_seat_scan_covers_all_route_classes():
    seats = dict(_DECLARED)
    assert len(seats) >= 50
    assert seats["sn-compose.model"].startswith("local/")
    assert seats["cluster-labels.model"].startswith("openrouter/")
    assert seats["discovery-relevance.model"].startswith("typesafe/")


@pytest.mark.parametrize("seat,model", _DECLARED, ids=[seat for seat, _ in _DECLARED])
def test_every_declared_seat_resolves_to_its_endpoint(seat, model, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "test-provider-key")
    section = seat.removesuffix(".model")
    if section in settings.MODEL_SECTIONS:
        config = settings.get_model_config(section)
        assert config["model"] == model, seat
    else:
        settings._validate_model_identifier(model, seat)
        config = settings.get_model_endpoint(model) or {}

    if model.startswith("typesafe/"):
        assert not config.get("api_base"), seat
        received = []

        def send(client, request, **kwargs):
            received.append((str(request.url), json.loads(request.content)))
            return httpx.Response(
                200, json={"answers": {}, "usage": {"cost": 0}}, request=request
            )

        monkeypatch.setattr(httpx.Client, "send", send)
        llm.call_decisions(model, {}, {}, service="seat-routing", max_retries=1)
        assert received == [
            (
                "https://openrouter.ai/api/alpha/decisions",
                {"model": model, "state": {}, "questions": {}},
            )
        ], seat
        return

    request = llm._build_kwargs(
        model,
        "test-provider-key",
        [{"role": "user", "content": "Reply with ok"}],
        None,
        1,
        None,
        10,
        api_base=config.get("api_base"),
    )
    assert request["model"] == model, seat
    if model.startswith("local/"):
        local_base = settings._get_section("model-routes")["ambix-local"]["api-base"]
        assert request["api_base"] == local_base, seat
        assert urlparse(request["api_base"]).hostname, seat
    else:
        assert model.startswith("openrouter/"), seat
        assert "api_base" not in request, seat


@pytest.mark.parametrize("model", ["retired-alias", "cc:opus", "anthropic/unqualified"])
def test_unrecognised_seat_route_names_the_seat(model, monkeypatch):
    seat = "scratch-unrecognised-seat"
    monkeypatch.setattr(settings, "MODEL_SECTIONS", settings.MODEL_SECTIONS | {seat})
    monkeypatch.setitem(settings._MODEL_DEFAULTS, seat, model)
    original = settings._get_section
    monkeypatch.setattr(
        settings,
        "_get_section",
        lambda section: {"model": model} if section == seat else original(section),
    )
    with pytest.raises(ValueError, match=seat) as refusal:
        settings.get_model_config(seat)
    assert model in str(refusal.value)
    print(f"ROUTE_REFUSAL {refusal.value}")


def test_local_seat_without_endpoint_names_the_seat(monkeypatch):
    seat = "sn-compose"
    original = settings._get_section
    monkeypatch.setattr(
        settings,
        "_get_section",
        lambda section: (
            {"model": "local/missing-endpoint"}
            if section == seat
            else original(section)
        ),
    )
    with pytest.raises(ValueError, match=seat):
        settings.get_model_config(seat)


def test_local_model_cannot_fall_through_to_provider(monkeypatch):
    monkeypatch.setattr(settings, "_MODEL_ENDPOINTS", {})
    with pytest.raises(ValueError, match="requires a configured endpoint"):
        llm._build_kwargs("local/missing", "key", [], None, 1, None, 10)
