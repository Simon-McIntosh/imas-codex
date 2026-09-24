import subprocess
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.tunnel import (
    SERVICE_MANIFEST_PREFIX,
    _build_foreground_tunnel_command,
    _build_systemd_service_content,
    _get_tunnel_ports,
    _installed_service_supports_request,
    _is_remote_clipboard_active,
    _probe_reverse_ssh_forward,
    _resolve_reverse_nodes,
    _reverse_forward_answers,
    _service_selected_services,
    _supervised_links,
    _terminate_tunnel_process,
    tunnel,
)
from imas_codex.remote.tunnel import (
    SSH_TUNNEL_OPTS,
    _local_forward_port,
    _ssh_forwarded_local_ports,
    discover_compute_node_local,
    local_forward_spec,
)


class TestTunnelServiceHelpers:
    def test_foreground_tunnel_binds_forward_to_ipv4_loopback(self):
        with patch(
            "imas_codex.cli.tunnel.shutil.which", return_value="/usr/bin/autossh"
        ):
            command, _env = _build_foreground_tunnel_command(
                "iter",
                [(8765, 8765, "docs", "127.0.0.1", "L")],
            )

        forward_index = command.index("-L")
        assert command[forward_index + 1] == "127.0.0.1:8765:127.0.0.1:8765"
        assert command.count("ExitOnForwardFailure=yes") == 0

    def test_reverse_forward_keeps_the_connection_on_a_rejected_bind(self):
        """A rejected reverse bind must not take the forward tunnels down.

        The supervisor probes each reverse forward end to end and restarts on a
        real failure, which also catches a bind that succeeded onto a dead local
        target — something exiting on bind failure could never see. A repeated
        ssh option keeps its first value, so a stricter one appended after the
        shared options would be inert rather than an override.
        """
        with patch(
            "imas_codex.cli.tunnel.shutil.which", return_value="/usr/bin/autossh"
        ):
            command, _env = _build_foreground_tunnel_command(
                "iter",
                [(2490, 2490, "wsl-clip", "localhost", "R")],
            )

        assert "ExitOnForwardFailure=no" in command
        assert "ExitOnForwardFailure=yes" not in command
        reverse_index = command.index("-R")
        assert command[reverse_index + 1] == "2490:localhost:2490"

    def test_reverse_ssh_forward_health_ignores_an_unreachable_remote(self):
        """Failing to reach the remote is not evidence against the forward.

        Treating it as a failure would restart the tunnel — and drop every
        forward sharing the connection — whenever the remote is briefly away.
        """
        with patch("imas_codex.cli.tunnel._probe_reverse_ssh_forward") as probe:
            probe.return_value = "unreachable"
            assert _reverse_forward_answers("iter", "wsl-ssh", 2222)
            probe.return_value = "up"
            assert _reverse_forward_answers("iter", "wsl-ssh", 2222)
            probe.return_value = "stale"
            assert not _reverse_forward_answers("iter", "wsl-ssh", 2222)
            probe.return_value = "down"
            assert not _reverse_forward_answers("iter", "wsl-ssh", 2222)

    def test_reverse_forward_health_reads_a_banner_not_a_login(self):
        """The verdict must describe the forward, not the client's keys.

        A login test reports a working forward as broken whenever the client
        stops authorising the remote's key, and the supervisor restarts on this
        verdict.
        """
        completed = MagicMock(returncode=0)
        with patch(
            "imas_codex.cli.tunnel.subprocess.run", return_value=completed
        ) as run:
            _probe_reverse_ssh_forward("iter", 2222)

        probe = run.call_args[0][0][-1]
        assert "/dev/tcp/127.0.0.1/2222" in probe
        assert "SSH-" in probe
        assert "BatchMode" not in probe

    def test_unreapable_tunnel_child_does_not_wedge_supervisor(self):
        child = MagicMock()
        child.pid = 42
        child.poll.return_value = None
        child.wait.side_effect = [
            subprocess.TimeoutExpired("ssh", 10),
            subprocess.TimeoutExpired("ssh", 5),
        ]

        with patch("imas_codex.cli.tunnel.os.killpg") as killpg:
            _terminate_tunnel_process(child)

        assert killpg.call_count == 2

    def test_remote_clipboard_health_check_uses_remote_loopback(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
        with patch(
            "imas_codex.cli.tunnel.subprocess.run", return_value=completed
        ) as run:
            assert _is_remote_clipboard_active("iter", 2490)

        assert run.call_args.args[0][-2:] == [
            "3",
            "http://127.0.0.1:2490/health",
        ]

    def test_remote_clipboard_health_check_rejects_failed_request(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=7, stdout="", stderr="connection refused"
        )
        with patch("imas_codex.cli.tunnel.subprocess.run", return_value=completed):
            assert not _is_remote_clipboard_active("iter", 2490)

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
        assert "After=wsl-clip-server.service" in content
        assert "Wants=wsl-clip-server.service" in content

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

    def test_reverse_ssh_forward_targets_the_local_sshd_port(self):
        """The remote dial port and the client's sshd port are different numbers.

        Forwarding the remote port to itself lands on a port nothing listens on,
        and the remote sshd accepts the connection on its own listener before
        the channel to the client is opened — so the failure presents as a
        connection that opens and sends no banner, indistinguishable from an
        sshd that is down.
        """
        with (
            patch("imas_codex.cli.tunnel._discover_compute_node", return_value=None),
            patch("imas_codex.cli.tunnel._discover_vllm_node", return_value=None),
        ):
            ports = _get_tunnel_ports(
                "iter",
                neo4j=False,
                embed=False,
                llm=False,
                emit_status=False,
            )

        reverse_ssh = [entry for entry in ports if entry[2] == "wsl-ssh"]
        assert reverse_ssh == [(2222, 22, "wsl-ssh", "localhost", "R")]

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


class TestReverseNodes:
    """Reverse forwards held on every login node a shell can land on.

    A reverse forward binds only on the node its ssh session reached, while the
    gateway places an interactive shell on any login node, so a clipboard bound
    on one node is invisible from the others.
    """

    SSH_CONFIG = {
        "hostname": "98dci4-srv-1006.iter.org",
        "proxyjump": "sdcc-login.iter.org",
        "user": "mcintos",
    }
    NODES = [f"98dci4-srv-100{i}" for i in range(1, 7)]

    def test_all_expands_to_known_nodes_minus_the_primary(self):
        with (
            patch(
                "imas_codex.cli.tunnel._ssh_client_config",
                return_value=self.SSH_CONFIG,
            ),
            patch("imas_codex.cli.tunnel._known_login_nodes", return_value=self.NODES),
        ):
            nodes = _resolve_reverse_nodes("iter", ["all"])

        names = [name for name, _ in nodes]
        assert "98dci4-srv-1006.iter.org" not in names
        assert names == [f"98dci4-srv-100{i}.iter.org" for i in range(1, 6)]
        assert nodes[2][1] == (
            "-o",
            "ProxyJump=sdcc-login.iter.org",
            "-l",
            "mcintos",
            "98dci4-srv-1003.iter.org",
        )

    def test_named_node_is_deduplicated_and_needs_no_jump_when_direct(self):
        config = {"hostname": "98dci4-srv-1006.iter.org", "proxyjump": "none"}
        with patch("imas_codex.cli.tunnel._ssh_client_config", return_value=config):
            nodes = _resolve_reverse_nodes(
                "iter", ["98dci4-srv-1003", "98dci4-srv-1003.iter.org"]
            )

        assert nodes == [("98dci4-srv-1003.iter.org", ("98dci4-srv-1003.iter.org",))]

    def test_no_nodes_requested_resolves_nothing(self):
        with patch("imas_codex.cli.tunnel._ssh_client_config") as config:
            assert _resolve_reverse_nodes("iter", []) == []
        config.assert_not_called()

    def test_extra_nodes_carry_only_the_reverse_forwards(self):
        ports = [
            (8765, 8765, "docs", "127.0.0.1", "L"),
            (2490, 2490, "wsl-clip", "localhost", "R"),
            (2222, 22, "wsl-ssh", "localhost", "R"),
        ]
        node = ("98dci4-srv-1003.iter.org", ("98dci4-srv-1003.iter.org",))

        links = _supervised_links("iter", ports, [node])

        assert links[0] == ("iter", "iter", ports)
        assert links[1] == (node[0], node[1], ports[1:])

    def test_no_extra_links_without_reverse_forwards(self):
        ports = [(8765, 8765, "docs", "127.0.0.1", "L")]
        node = ("98dci4-srv-1003.iter.org", ("98dci4-srv-1003.iter.org",))

        assert _supervised_links("iter", ports, [node]) == [("iter", "iter", ports)]

    def test_tunnel_command_ends_with_the_node_destination(self):
        target = ("-o", "ProxyJump=gw", "-l", "me", "node.example")
        with patch(
            "imas_codex.cli.tunnel.shutil.which", return_value="/usr/bin/autossh"
        ):
            command, _env = _build_foreground_tunnel_command(
                target, [(2490, 2490, "wsl-clip", "localhost", "R")]
            )

        assert command[-5:] == list(target)

    def test_service_unit_passes_reverse_nodes_to_the_supervisor(self):
        with patch("imas_codex.cli.tunnel.shutil.which", return_value="/usr/bin/uv"):
            content = _build_systemd_service_content(
                "iter", False, False, False, reverse_nodes=("all",)
            )

        exec_start = next(
            line for line in content.splitlines() if line.startswith("ExecStart=")
        )
        assert exec_start.endswith("service-run iter --reverse-node all")


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


class TestTunnelProcessInspection:
    def test_local_forward_spec_binds_to_ipv4_loopback(self):
        assert (
            local_forward_spec(17687, "compute-node", 7687)
            == "127.0.0.1:17687:compute-node:7687"
        )

    def test_local_forward_port_accepts_supported_ssh_forms(self):
        assert _local_forward_port("8765:127.0.0.1:8765") == 8765
        assert _local_forward_port("127.0.0.1:8765:127.0.0.1:8765") == 8765
        assert _local_forward_port("[::1]:8765:127.0.0.1:8765") == 8765
        assert _local_forward_port("not-a-forward") is None

    def test_ssh_forwarded_ports_reads_command_lines_only(self, tmp_path):
        commands = {
            "101": ["/usr/bin/ssh", "-N", "-L", "127.0.0.1:8765:host:8765"],
            "102": ["/usr/bin/autossh", "-L17687:host:7687", "iter"],
            "103": ["/usr/bin/python", "-L", "9999:host:9999"],
        }
        for pid, args in commands.items():
            pid_dir = tmp_path / pid
            pid_dir.mkdir()
            (pid_dir / "cmdline").write_bytes(
                b"\0".join(arg.encode() for arg in args) + b"\0"
            )

        assert _ssh_forwarded_local_ports(tmp_path) == {8765, 17687}
