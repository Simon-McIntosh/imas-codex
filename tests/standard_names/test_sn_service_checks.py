"""Unit tests for the SN run SERVERS-row endpoint health checks.

``_check_local_llm`` probes the local GPU vLLM server configured in
``[sn-compose]``; ``_check_openrouter`` probes OpenRouter key validity.
The critical regression covered here: a vLLM server launched with
``--api-key`` returns 401 on unauthenticated routes — the probe must
send the key and must classify an HTTP response as *reachable* (an
HTTPError means the server is up), never as "unreachable".
"""

from __future__ import annotations

import asyncio
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from imas_codex.cli.sn import (
    _check_local_llm,
    _check_openrouter,
    _require_local_compose_ready,
    _run_can_generate_names,
    _run_sn_cmd,
    _sn_llm_service_checks,
)

_COMPOSE_CFG = {
    "model": "hosted_vllm/deepseek-v4-flash",
    "api_base": "http://gpu-node:18800/v1",
    "api_key_env": "TEST_GPU_KEY",
}


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://gpu-node:18800/v1/models",
        code=code,
        msg="",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(b""),
    )


def _ok_response(body: bytes = b"{}") -> MagicMock:
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCheckLocalLLM:
    """Local GPU endpoint probe classification."""

    def _run(self, urlopen_effect, key: str | None = "sk-test"):
        env = {"TEST_GPU_KEY": key} if key else {}
        with (
            patch("imas_codex.settings.get_model_config", return_value=_COMPOSE_CFG),
            patch.dict("os.environ", env, clear=False),
            patch("urllib.request.urlopen") as urlopen,
        ):
            if key is None:
                import os

                os.environ.pop("TEST_GPU_KEY", None)
            if isinstance(urlopen_effect, Exception):
                urlopen.side_effect = urlopen_effect
            else:
                urlopen.return_value = urlopen_effect
            result = _check_local_llm()
            return result, urlopen

    def test_healthy_returns_model_short_name(self) -> None:
        (healthy, detail), _ = self._run(_ok_response())
        assert healthy is True
        assert detail == "deepseek-v4-flash"

    def test_probe_sends_authorization_header(self) -> None:
        """Regression: unauthenticated probes 401 against vLLM --api-key
        servers and were misreported as unreachable."""
        _, urlopen = self._run(_ok_response())
        request = urlopen.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer sk-test"

    def test_401_with_key_is_auth_error_not_unreachable(self) -> None:
        (healthy, detail), _ = self._run(_http_error(401))
        assert healthy is False
        assert detail == "auth error"

    def test_401_without_key_is_key_missing(self) -> None:
        (healthy, detail), _ = self._run(_http_error(401), key=None)
        assert healthy is False
        assert detail == "key missing"

    def test_http_500_reports_code(self) -> None:
        (healthy, detail), _ = self._run(_http_error(500))
        assert healthy is False
        assert detail == "HTTP 500"

    def test_connection_refused_is_down(self) -> None:
        err = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        (healthy, detail), _ = self._run(err)
        assert healthy is False
        assert detail == "down"

    def test_timeout_is_timeout(self) -> None:
        err = urllib.error.URLError(TimeoutError("timed out"))
        (healthy, detail), _ = self._run(err)
        assert healthy is False
        assert detail == "timeout"

    def test_no_api_base_is_not_configured(self) -> None:
        cfg = {"model": "openrouter/x", "api_base": None, "api_key_env": None}
        with patch("imas_codex.settings.get_model_config", return_value=cfg):
            healthy, detail = _check_local_llm()
        assert healthy is False
        assert detail == "not configured"


class TestCheckOpenRouter:
    """OpenRouter key probe classification."""

    def _run(self, urlopen_effect, key: str | None = "sk-or-test"):
        env_patch = {"OPENROUTER_API_KEY_IMAS_CODEX": key} if key else {}
        with (
            patch.dict("os.environ", env_patch, clear=False),
            patch("urllib.request.urlopen") as urlopen,
        ):
            if key is None:
                import os

                os.environ.pop("OPENROUTER_API_KEY_IMAS_CODEX", None)
            if isinstance(urlopen_effect, Exception):
                urlopen.side_effect = urlopen_effect
            else:
                urlopen.return_value = urlopen_effect
            return _check_openrouter()

    def test_no_key(self) -> None:
        healthy, detail = self._run(_ok_response(), key=None)
        assert healthy is False
        assert detail == "no key"

    def test_healthy_with_remaining_credit(self) -> None:
        body = b'{"data": {"limit_remaining": 323.4}}'
        healthy, detail = self._run(_ok_response(body))
        assert healthy is True
        assert detail == "ok $323"

    def test_healthy_without_limit(self) -> None:
        healthy, detail = self._run(_ok_response(b'{"data": {}}'))
        assert healthy is True
        assert detail == "ok"

    def test_402_is_no_credit(self) -> None:
        healthy, detail = self._run(_http_error(402))
        assert healthy is False
        assert detail == "no credit"

    def test_429_is_rate_limited(self) -> None:
        healthy, detail = self._run(_http_error(429))
        assert healthy is False
        assert detail == "rate limited"

    def test_401_is_auth_error(self) -> None:
        healthy, detail = self._run(_http_error(401))
        assert healthy is False
        assert detail == "auth error"

    def test_network_error_is_unreachable(self) -> None:
        err = urllib.error.URLError(OSError("no route to host"))
        healthy, detail = self._run(err)
        assert healthy is False
        assert detail == "unreachable"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"only": "compose"}, True),
        ({"skip_generate": True}, False),
        ({"docs_only": True}, False),
        ({"flush": True}, False),
        ({"only": "review"}, False),
        ({"only": "review_name"}, False),
        ({"only": "reconcile"}, False),
        ({"only": "attach"}, False),
    ],
)
def test_run_capability_identifies_local_compose_dependency(
    kwargs: dict[str, Any], expected: bool
) -> None:
    """Only runs able to dispatch generation depend on local compose."""
    assert _run_can_generate_names(**kwargs) is expected


def test_generation_registers_local_compose_as_critical() -> None:
    """The Rich monitor gates generation on the configured local endpoint."""
    with patch(
        "imas_codex.cli.sn._configured_local_compose_required", return_value=True
    ):
        checks = _sn_llm_service_checks(
            use_rich=True,
            dry_run=False,
            can_generate_names=True,
        )

    gpu = next(item for item in checks if item[0] == "gpu")
    assert gpu[1] is _check_local_llm
    assert gpu[2]["critical"] is True


@pytest.mark.parametrize(
    "run_kwargs",
    [
        {"can_generate_names": False},
        {"can_generate_names": True, "dry_run": True},
    ],
)
def test_non_generation_routes_do_not_probe_local_compose(
    run_kwargs: dict[str, bool],
) -> None:
    """Review, docs, maintenance, and previews stay independent of Ambix."""
    with patch(
        "imas_codex.cli.sn._configured_local_compose_required",
        side_effect=AssertionError("local compose route inspected"),
    ):
        checks = _sn_llm_service_checks(
            use_rich=True,
            dry_run=run_kwargs.get("dry_run", False),
            can_generate_names=run_kwargs["can_generate_names"],
        )

    assert all(name != "gpu" for name, _, _ in checks)


def test_failed_local_readiness_blocks_without_paid_fallback() -> None:
    """A local outage fails closed before any OpenRouter route is considered."""
    with (
        patch(
            "imas_codex.cli.sn._configured_local_compose_required",
            return_value=True,
        ),
        patch(
            "imas_codex.cli.sn._check_local_llm",
            return_value=(False, "down"),
        ),
        patch(
            "imas_codex.cli.sn._check_openrouter",
            side_effect=AssertionError("paid fallback attempted"),
        ) as paid_fallback,
        pytest.raises(click.ClickException, match="no paid provider fallback"),
    ):
        _require_local_compose_ready()

    paid_fallback.assert_not_called()


def test_initial_local_readiness_blocks_generation_until_stopped() -> None:
    """An unhealthy critical probe cannot race a generation worker launch."""
    run_pools = AsyncMock()
    paid_fallback = MagicMock(side_effect=AssertionError("paid fallback attempted"))

    async def wait_forever() -> bool:
        await asyncio.Event().wait()
        return True

    monitor = MagicMock()
    monitor.await_services_ready = AsyncMock(side_effect=wait_forever)

    def run_discovery(config, async_main):
        gpu = next(item for item in config.extra_service_checks if item[0] == "gpu")
        assert gpu[2]["critical"] is True
        stop_event = asyncio.Event()
        stop_event.set()
        return asyncio.run(async_main(stop_event, monitor))

    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch(
            "imas_codex.cli.sn._configured_local_compose_required",
            return_value=True,
        ),
        patch("imas_codex.cli.sn._check_openrouter", new=paid_fallback),
        patch("imas_codex.cli.discover.common.use_rich_output", return_value=True),
        patch("imas_codex.cli.discover.common.setup_logging"),
        patch(
            "imas_codex.cli.discover.common.run_discovery", side_effect=run_discovery
        ),
        patch(
            "imas_codex.standard_names.display.SN6PoolDisplay", return_value=MagicMock()
        ),
        patch("imas_codex.standard_names.loop.run_sn_pools", new=run_pools),
    ):
        result = _run_sn_cmd(
            cost_limit=1.0,
            per_domain_limit=None,
            dry_run=False,
            quiet=False,
        )

    assert result is None
    run_pools.assert_not_awaited()
    paid_fallback.assert_not_called()
