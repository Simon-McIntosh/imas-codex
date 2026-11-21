"""Health checks probe the configured model endpoint without generating tokens."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest


class TestLlmHealthCheckLocalFreeEndpoint:
    """A local-free seat is probed at its own endpoint."""

    API_BASE = "http://test-router:18802/v1"

    def _mock_models(self, models):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": models}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("iter_codex.settings.get_model_config")
    @patch("urllib.request.urlopen")
    def test_probes_models_listing(self, mock_urlopen, mock_config):
        """The models listing of the seat's api-base is probed without completion."""
        from iter_codex.discovery.base.services import llm_health_check

        mock_config.return_value = {"model": "local/x", "api_base": self.API_BASE}
        mock_urlopen.return_value = self._mock_models([{"id": "local/x"}])

        healthy, _ = llm_health_check("discovery-score")

        assert healthy is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == f"{self.API_BASE}/models"
        assert mock_urlopen.call_args.kwargs["timeout"] == 10

    @patch("iter_codex.settings.get_model_config")
    @patch("urllib.request.urlopen")
    def test_unreachable_endpoint_is_unhealthy(self, mock_urlopen, mock_config):
        from iter_codex.discovery.base.services import llm_health_check

        mock_config.return_value = {"model": "local/x", "api_base": self.API_BASE}
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        healthy, detail = llm_health_check("discovery-score")

        assert healthy is False
        assert "refused" in detail


class TestDecisionsHealthCheck:
    """The decisions route is probed with a zero-cost authenticated request."""

    KEY_URL = "https://openrouter.ai/api/v1/key"

    def _mock_key_info(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"data": {"label": "test", "usage": 0.0}}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_authenticated_key_info_probe_is_healthy(self, mock_urlopen, monkeypatch):
        from iter_codex.discovery.base.services import decisions_health_check

        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-test")
        mock_urlopen.return_value = self._mock_key_info()

        healthy, detail = decisions_health_check()

        assert healthy is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == self.KEY_URL
        assert req.get_header("Authorization") == "Bearer sk-test"
        assert mock_urlopen.call_args.kwargs["timeout"] == 10

    @patch("urllib.request.urlopen")
    def test_refused_route_is_unhealthy(self, mock_urlopen, monkeypatch):
        from iter_codex.discovery.base.services import decisions_health_check

        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-test")
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        healthy, detail = decisions_health_check()

        assert healthy is False
        assert "refused" in detail

    @patch("urllib.request.urlopen")
    def test_unauthorized_credential_is_unhealthy(self, mock_urlopen, monkeypatch):
        from iter_codex.discovery.base.services import decisions_health_check

        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-test")
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=self.KEY_URL, code=401, msg="unauthorized", hdrs=None, fp=None
        )

        healthy, detail = decisions_health_check()

        assert healthy is False
        assert "401" in detail


class TestLlmHealthCheckWithDecisions:
    """The panel's llm row reflects both the score seat and the decisions route."""

    API_BASE = "http://test-router:18802/v1"

    def _mock_models(self, models):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": models}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("iter_codex.discovery.base.services.llm_health_check")
    @patch("urllib.request.urlopen")
    def test_healthy_when_both_reachable(self, mock_urlopen, mock_chat, monkeypatch):
        from iter_codex.discovery.base.services import (
            llm_health_check_with_decisions,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-test")
        mock_chat.return_value = (True, "titan")
        resp = MagicMock()
        resp.read.return_value = b'{"data": {}}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        healthy, detail = llm_health_check_with_decisions("discovery-score")

        assert healthy is True
        # The all-healthy path keeps the seat's detail (what the panel showed
        # before the decisions probe existed), not the decisions route host.
        assert detail == "titan"
        # The decisions probe still ran and reached the route's key-info endpoint.
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://openrouter.ai/api/v1/key"

    @patch("iter_codex.discovery.base.services.llm_health_check")
    @patch("urllib.request.urlopen")
    def test_refused_decisions_probe_names_decisions(
        self, mock_urlopen, mock_chat, monkeypatch
    ):
        from iter_codex.discovery.base.services import (
            llm_health_check_with_decisions,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-test")
        mock_chat.return_value = (True, "titan")
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        healthy, detail = llm_health_check_with_decisions("discovery-score")

        assert healthy is False
        assert "decisions" in detail
        assert "refused" in detail

    @patch("iter_codex.settings.get_model")
    @patch("iter_codex.settings.get_model_config")
    @patch("urllib.request.urlopen")
    def test_refused_local_seat_names_score(
        self, mock_urlopen, mock_config, mock_model
    ):
        from iter_codex.discovery.base.services import (
            llm_health_check_with_decisions,
        )

        mock_config.return_value = {"model": "local/x", "api_base": self.API_BASE}
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        healthy, detail = llm_health_check_with_decisions("discovery-score")

        assert healthy is False
        assert detail.startswith("score (")
        # The decisions route is not probed once the score seat is down.
        assert mock_urlopen.call_count == 1


@pytest.mark.parametrize(
    "model,expected_url",
    [
        ("openrouter/example/chat", "https://openrouter.ai/api/v1/key"),
        ("typesafe/example-decision", "https://decisions.example.test/api/v1/key"),
    ],
)
def test_seat_health_uses_its_own_origin(model, expected_url, monkeypatch):
    from iter_codex.discovery.base import llm, services

    monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "test-key")
    monkeypatch.setattr(
        llm, "DECISIONS_ENDPOINT", "https://decisions.example.test/api/alpha/decisions"
    )
    with (
        patch(
            "iter_codex.settings.get_model_config",
            return_value={"model": model, "api_base": None},
        ),
        patch("urllib.request.urlopen", return_value=MagicMock()) as send,
    ):
        assert services.llm_health_check("discovery-score")[0]
    assert send.call_count == 1
    assert send.call_args.args[0].full_url == expected_url


@pytest.mark.parametrize(
    "model", ["openrouter/example/chat", "typesafe/example-decision"]
)
def test_shared_provider_is_probed_once_per_poll(model, monkeypatch):
    from iter_codex.discovery.base.services import llm_health_check_with_decisions

    monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "test-key")
    with (
        patch(
            "iter_codex.settings.get_model_config",
            return_value={"model": model, "api_base": None},
        ),
        patch("urllib.request.urlopen", return_value=MagicMock()) as send,
    ):
        assert llm_health_check_with_decisions("discovery-score")[0]
    assert send.call_count == 1
    assert send.call_args.args[0].full_url == "https://openrouter.ai/api/v1/key"


def test_distinct_provider_origins_each_get_one_probe(monkeypatch):
    from iter_codex.discovery.base import llm, services

    monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "test-key")
    monkeypatch.setattr(
        llm, "DECISIONS_ENDPOINT", "https://decisions.example.test/api/alpha/decisions"
    )
    with (
        patch(
            "iter_codex.settings.get_model_config",
            return_value={"model": "openrouter/example/chat", "api_base": None},
        ),
        patch("urllib.request.urlopen", return_value=MagicMock()) as send,
    ):
        assert services.llm_health_check_with_decisions("discovery-score")[0]
    assert [call.args[0].full_url for call in send.call_args_list] == [
        "https://openrouter.ai/api/v1/key",
        "https://decisions.example.test/api/v1/key",
    ]


def test_local_router_and_provider_each_get_one_probe(monkeypatch):
    from iter_codex.discovery.base.services import llm_health_check_with_decisions

    monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "test-key")
    reply = MagicMock()
    reply.__enter__.return_value.read.return_value = b'{"data":[{"id":"model"}]}'
    with (
        patch(
            "iter_codex.settings.get_model_config",
            return_value={
                "model": "local/example",
                "api_base": "http://router.test/v1",
            },
        ),
        patch("urllib.request.urlopen", return_value=reply) as send,
    ):
        assert llm_health_check_with_decisions("discovery-score")[0]
    assert [call.args[0].full_url for call in send.call_args_list] == [
        "http://router.test/v1/models",
        "https://openrouter.ai/api/v1/key",
    ]


def test_provider_health_uses_the_requested_service_key(monkeypatch):
    from iter_codex.discovery.base.services import llm_health_check_with_decisions

    monkeypatch.setenv("OPENROUTER_API_KEY_STANDARD_NAMES", "test-service-key")
    with (
        patch(
            "iter_codex.settings.get_model_config",
            return_value={"model": "openrouter/example/chat", "api_base": None},
        ),
        patch("urllib.request.urlopen", return_value=MagicMock()) as send,
    ):
        assert llm_health_check_with_decisions("sn-docs", service="standard-names")[0]
    assert send.call_count == 1
    assert (
        send.call_args.args[0].get_header("Authorization") == "Bearer test-service-key"
    )
