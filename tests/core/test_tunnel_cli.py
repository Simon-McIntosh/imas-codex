from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.tunnel import (
    SERVICE_MANIFEST_PREFIX,
    _build_systemd_service_content,
    _get_tunnel_ports,
    _installed_service_supports_request,
    _service_selected_services,
    tunnel,
)
from imas_codex.remote.tunnel import SSH_TUNNEL_OPTS, discover_compute_node_local


class TestTunnelServiceHelpers:
    def test_build_systemd_service_content_uses_runtime_service_runner(self):
        with patch(
            "imas_codex.cli.tunnel.shutil.which",
            side_effect=lambda cmd: (
                "/usr/bin/uv" if cmd == "uv" else "/usr/bin/autossh"
            ),
        ):
            content = _build_systemd_service_content(
                "iter",
                neo4j_only=False,
                embed_only=False,
                llm_only=False,
            )

        assert "tunnel service-run iter" in content
        assert "98dci4-gpu-0002" not in content
        assert "-L 17687:" not in content
        assert "WatchdogSec" not in content

    def test_installed_service_supports_subset_request(self, tmp_path):
        service_file = tmp_path / "imas-codex-tunnel-iter.service"
        service_file.write_text(
            "ExecStart=/usr/bin/uv run --project /repo imas-codex tunnel service-run iter\n"
        )

        with patch("imas_codex.cli.tunnel._service_file", return_value=service_file):
            assert _installed_service_supports_request(
                "iter",
                neo4j_only=True,
                embed_only=False,
                llm_only=False,
            )
            assert _installed_service_supports_request(
                "iter",
                neo4j_only=False,
                embed_only=True,
                llm_only=False,
            )

    def test_installed_service_rejects_missing_service_flags(self, tmp_path):
        service_file = tmp_path / "imas-codex-tunnel-iter.service"
        service_file.write_text(
            "ExecStart=/usr/bin/uv run --project /repo imas-codex tunnel service-run iter --neo4j\n"
        )

        with patch("imas_codex.cli.tunnel._service_file", return_value=service_file):
            assert _installed_service_supports_request(
                "iter",
                neo4j_only=True,
                embed_only=False,
                llm_only=False,
            )
            assert not _installed_service_supports_request(
                "iter",
                neo4j_only=False,
                embed_only=True,
                llm_only=False,
            )

    def test_docs_only_emits_docs_server_port(self):
        ports = _get_tunnel_ports(
            "iter",
            neo4j=False,
            embed=False,
            llm=False,
            vllm=False,
            docs=True,
            emit_status=False,
        )
        assert ports == [(8765, 8765, "docs", "127.0.0.1", "L")]

    def test_installed_service_rejects_docs_when_absent(self, tmp_path):
        service_file = tmp_path / "imas-codex-tunnel-iter.service"
        service_file.write_text(
            "ExecStart=/usr/bin/uv run --project /repo imas-codex tunnel service-run iter --llm\n"
        )

        with patch("imas_codex.cli.tunnel._service_file", return_value=service_file):
            assert not _installed_service_supports_request(
                "iter",
                neo4j_only=False,
                embed_only=False,
                llm_only=False,
                docs_only=True,
            )

    def test_build_systemd_service_content_emits_manifest(self):
        with patch(
            "imas_codex.cli.tunnel.shutil.which",
            side_effect=lambda cmd: (
                "/usr/bin/uv" if cmd == "uv" else "/usr/bin/autossh"
            ),
        ):
            # No flags → manifest lists every current service.
            all_content = _build_systemd_service_content(
                "iter",
                neo4j_only=False,
                embed_only=False,
                llm_only=False,
            )
            # Subset flags → manifest lists only those.
            subset_content = _build_systemd_service_content(
                "iter",
                neo4j_only=True,
                embed_only=False,
                llm_only=False,
                docs_only=True,
            )

        assert (
            SERVICE_MANIFEST_PREFIX + "docs embed ink llm neo4j vllm wsl-clip"
            in all_content
        )
        assert SERVICE_MANIFEST_PREFIX + "docs neo4j" in subset_content

    def test_service_selected_services_reads_manifest(self):
        text = (
            f"{SERVICE_MANIFEST_PREFIX}neo4j docs\n"
            "[Unit]\n"
            "ExecStart=/usr/bin/uv run --project /repo imas-codex "
            "tunnel service-run iter --neo4j --docs\n"
        )
        assert _service_selected_services(text) == {"neo4j", "docs"}

    def test_legacy_flagless_unit_excludes_docs(self, tmp_path):
        # A unit installed before SERVICE_MANIFEST_PREFIX existed (no manifest
        # line, no flags) must NOT claim docs support — otherwise `tunnel start
        # --docs` silently starts a stale unit that doesn't forward 8765.
        service_file = tmp_path / "imas-codex-tunnel-iter.service"
        service_file.write_text(
            "ExecStart=/usr/bin/uv run --project /repo imas-codex tunnel service-run iter\n"
        )

        with patch("imas_codex.cli.tunnel._service_file", return_value=service_file):
            # Pre-docs services still claimed (back-compat).
            assert _installed_service_supports_request(
                "iter",
                neo4j_only=True,
                embed_only=False,
                llm_only=False,
            )
            # Docs is not in the legacy frozen set — must return False so the
            # caller falls through to ad-hoc or prompts a reinstall.
            assert not _installed_service_supports_request(
                "iter",
                neo4j_only=False,
                embed_only=False,
                llm_only=False,
                docs_only=True,
            )


class TestTunnelStart:
    def test_tunnel_start_uses_matching_systemd_service(self):
        runner = CliRunner()

        with (
            patch(
                "imas_codex.cli.tunnel._installed_service_supports_request",
                return_value=True,
            ),
            patch("imas_codex.cli.tunnel._run_systemctl_user") as mock_systemctl,
        ):
            result = runner.invoke(tunnel, ["start", "iter"])

        assert result.exit_code == 0
        assert "Starting systemd tunnel service for iter" in result.output
        mock_systemctl.assert_called_once_with(["start", "imas-codex-tunnel-iter"])


class TestComputeNodeDiscovery:
    def test_ssh_tunnel_opts_no_clear_all_forwardings(self):
        # ClearAllForwardings=yes must NOT be used — it clears our own -L
        # forwards (confirmed OpenSSH 8.9 behaviour).
        assert "ClearAllForwardings=yes" not in SSH_TUNNEL_OPTS

    def test_local_discovery_uses_configured_job_name(self):
        calls = []

        def _run(args, **kwargs):
            calls.append(args)

            class Result:
                returncode = 0
                stdout = "98dci4-gpu-0002\n"

            return Result()

        with patch("imas_codex.remote.tunnel.subprocess.run", side_effect=_run):
            node = discover_compute_node_local("codex-neo4j")

        assert node == "98dci4-gpu-0002"
        assert len(calls) == 1
        assert calls[0][2] == "codex-neo4j"
