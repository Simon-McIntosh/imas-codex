"""The Neo4j reachability probe must not authenticate without a credential.

The probe runs at collection time whenever any collected item is marked
``graph``, ``integration`` or ``requires_graph`` — markers that ordinary
``-m "not slow and not graph"`` runs do not deselect. A checkout that supplies
no credential resolves the active profile to the packaged development
placeholder, so the probe would authenticate against the shared project graph
with a password that cannot work. Repeated across pytest sessions those failed
logins trip the server's failed-attempt limiter and lock out whoever is using
the graph for real, so the probe has to decide it cannot connect without
trying.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _conftest_module():
    """Return the loaded root conftest module object.

    Iterating a snapshot — a lazy import elsewhere can grow ``sys.modules``
    mid-scan.
    """
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).resolve() == _CONFTEST:
            return module
    raise AssertionError("root conftest module is not loaded")


@pytest.fixture
def conftest_module(monkeypatch):
    """The root conftest with its per-session reachability cache cleared."""
    module = _conftest_module()
    monkeypatch.setattr(module, "_neo4j_available", None, raising=False)
    monkeypatch.delenv("IMAS_CODEX_TEST_NEO4J_URI", raising=False)
    return module


def _profile_with(password: str):
    """A resolved profile carrying *password*, without touching a server.

    The guard reads only the credential, so a stub keeps the test independent
    of the profile dataclass's other required fields.
    """
    return SimpleNamespace(password=password)


class TestProbeCredentialGuard:
    def test_placeholder_credential_makes_no_connection_attempt(
        self, conftest_module, monkeypatch
    ):
        """The decisive property: nothing is constructed, so nothing authenticates."""
        from imas_codex.graph import profiles

        monkeypatch.setattr(
            profiles,
            "resolve_neo4j",
            lambda **_kwargs: _profile_with(profiles.DEFAULT_PASSWORD),
        )

        def _must_not_be_called(*_args, **_kwargs):
            raise AssertionError(
                "the probe built a client despite having no credential — "
                "it would have authenticated against the project graph"
            )

        monkeypatch.setattr(conftest_module, "_neo4j_probe_client", _must_not_be_called)

        with patch("imas_codex.graph.client.GraphClient.__post_init__") as post_init:
            assert conftest_module._check_neo4j() is False

        assert post_init.call_count == 0

    def test_empty_credential_makes_no_connection_attempt(
        self, conftest_module, monkeypatch
    ):
        """An unset credential is as unusable as the placeholder."""
        from imas_codex.graph import profiles

        monkeypatch.setattr(
            profiles, "resolve_neo4j", lambda **_kwargs: _profile_with("")
        )

        with patch("imas_codex.graph.client.GraphClient.__post_init__") as post_init:
            assert conftest_module._check_neo4j() is False

        assert post_init.call_count == 0

    def test_unresolvable_profile_makes_no_connection_attempt(
        self, conftest_module, monkeypatch
    ):
        """A checkout that cannot resolve a profile at all must not guess one."""
        from imas_codex.graph import profiles

        def _raise(**_kwargs):
            raise RuntimeError("no profile configured")

        monkeypatch.setattr(profiles, "resolve_neo4j", _raise)

        with patch("imas_codex.graph.client.GraphClient.__post_init__") as post_init:
            assert conftest_module._check_neo4j() is False

        assert post_init.call_count == 0

    def test_real_credential_still_probes(self, conftest_module, monkeypatch):
        """No behaviour change where a credential exists — the probe still runs."""
        from imas_codex.graph import profiles

        monkeypatch.setattr(
            profiles,
            "resolve_neo4j",
            lambda **_kwargs: _profile_with("a-real-configured-secret"),
        )

        probed: list[bool] = []

        class _Reachable:
            def get_stats(self):
                probed.append(True)
                return {}

            def close(self):
                return None

        monkeypatch.setattr(
            conftest_module, "_neo4j_probe_client", lambda: _Reachable()
        )

        assert conftest_module._check_neo4j() is True
        assert probed == [True]

    def test_explicit_test_endpoint_is_never_blocked(
        self, conftest_module, monkeypatch
    ):
        """An explicit endpoint is an opt-in and cannot reach the project graph.

        It stays usable with no password so an auth-disabled local server
        still probes normally.
        """
        monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", "bolt://127.0.0.1:7687")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        assert conftest_module._graph_credential_is_configured() is True


class TestSkipReason:
    def test_missing_credential_reason_names_the_cause(
        self, conftest_module, monkeypatch
    ):
        """A skipped graph test says why in one line a reader can act on."""
        from imas_codex.graph import profiles

        monkeypatch.setattr(
            profiles,
            "resolve_neo4j",
            lambda **_kwargs: _profile_with(profiles.DEFAULT_PASSWORD),
        )

        reason = conftest_module._neo4j_unavailable_reason()

        assert "No Neo4j credential is configured" in reason
        assert "placeholder" in reason

    def test_configured_credential_keeps_the_unreachable_reason(
        self, conftest_module, monkeypatch
    ):
        """A configured-but-down graph must not be reported as unconfigured."""
        from imas_codex.graph import profiles

        monkeypatch.setattr(
            profiles,
            "resolve_neo4j",
            lambda **_kwargs: _profile_with("a-real-configured-secret"),
        )

        assert (
            conftest_module._neo4j_unavailable_reason()
            == "Configured project Neo4j graph is not available"
        )
