"""Tests for remote SSH command construction in service CLI modules.

Regression gates for ITER XDR security fix: SSH commands must never use
``echo <b64> | base64 -d | bash`` or any base64 encoding pattern.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# ===========================================================================
# TestRunOnNode
# ===========================================================================


class TestRunOnNode:
    """Tests for _run_on_node in imas_codex.cli.services."""

    def test_generates_ssh_command(self):
        """_run_on_node must call _run_remote with an SSH command targeting the node."""
        with patch("imas_codex.cli.services._run_remote") as mock_run:
            mock_run.return_value = "output"
            from imas_codex.cli.services import _run_on_node

            result = _run_on_node("gpu-0001", "pgrep -u $USER -f embed", timeout=30)

            assert result == "output"
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "gpu-0001" in cmd
            assert "ssh" in cmd.lower() or "StrictHostKeyChecking" in cmd

    def test_no_base64_in_command(self):
        """The SSH command must not contain the word 'base64'."""
        with patch("imas_codex.cli.services._run_remote") as mock_run:
            mock_run.return_value = ""
            from imas_codex.cli.services import _run_on_node

            _run_on_node("gpu-0001", "pgrep -u $USER -f embed", timeout=30)

            cmd = mock_run.call_args[0][0]
            assert "base64" not in cmd

    def test_no_echo_base64_pipe(self):
        """The 'echo ... | base64 -d' pattern must not appear in the SSH command."""
        with patch("imas_codex.cli.services._run_remote") as mock_run:
            mock_run.return_value = ""
            from imas_codex.cli.services import _run_on_node

            _run_on_node("gpu-0001", "pgrep -u $USER -f embed", timeout=30)

            cmd = mock_run.call_args[0][0]
            assert "base64 -d" not in cmd


# ===========================================================================
# TestSubmitServiceJob
# ===========================================================================


class TestSubmitServiceJob:
    """Tests for _submit_service_job in imas_codex.cli.services."""

    @staticmethod
    def _patched_submit(mock_run_side_effect=None):
        """Return context managers that mock all external dependencies."""
        if mock_run_side_effect is None:
            mock_run_side_effect = ["/home/user", "Submitted batch job 12345"]
        return (
            patch("imas_codex.cli.services._stop_login_services"),
            patch(
                "imas_codex.cli.services._run_remote",
                side_effect=mock_run_side_effect,
            ),
            patch(
                "imas_codex.cli.services._general_partition_name",
                return_value="batch",
            ),
            patch(
                "imas_codex.cli.services._embedding_target",
                return_value=(
                    {"name": "gpu"},
                    {"location": "gpu-0001"},
                ),
            ),
            patch(
                "imas_codex.cli.services._gpu_entry",
                return_value={"location": "gpu-0001"},
            ),
            patch(
                "imas_codex.cli.services._get_node_state",
                return_value=("idle", "none"),
            ),
        )

    def test_generates_sbatch_command(self):
        """_submit_service_job must call _run_remote with an sbatch command."""
        patches = self._patched_submit()
        with (
            patches[0],
            patches[1] as mock_run,
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            from imas_codex.cli.services import _submit_service_job

            _submit_service_job(
                "test-job",
                "echo hello",
                cpus=4,
                mem="8G",
                gpus=1,
            )

            # At least two calls: (1) echo $HOME, (2) the submit command
            assert mock_run.call_count >= 2
            submit_cmd = mock_run.call_args_list[1][0][0]
            assert "sbatch" in submit_cmd

    def test_no_base64_in_submit(self):
        """The sbatch submission command must not contain 'base64'."""
        patches = self._patched_submit()
        with (
            patches[0],
            patches[1] as mock_run,
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            from imas_codex.cli.services import _submit_service_job

            _submit_service_job(
                "test-job",
                "echo hello",
                cpus=4,
                mem="8G",
                gpus=1,
            )

            submit_cmd = mock_run.call_args_list[1][0][0]
            assert "base64" not in submit_cmd

    def test_script_content_includes_service_command(self):
        """The generated SLURM script must embed the service command."""
        captured_cmds: list[str] = []

        def capturing_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            if "echo $HOME" in cmd:
                return "/home/user"
            return "Submitted batch job 12345"

        patches = self._patched_submit()
        with (
            patches[0],
            patch("imas_codex.cli.services._run_remote", side_effect=capturing_run),
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            from imas_codex.cli.services import _submit_service_job

            _submit_service_job(
                "test-job",
                "echo hello",
                cpus=4,
                mem="8G",
                gpus=1,
            )

        # At least one captured command must relate to the submit step
        assert len(captured_cmds) >= 2
        submit_cmd = captured_cmds[1]
        # The submit command must reference the job name and sbatch
        assert "test-job" in submit_cmd
        assert "sbatch" in submit_cmd


class TestEmbeddingTarget:
    """Configured embedding locations select their matching GPU target."""

    @staticmethod
    def _compute_config() -> dict:
        return {
            "scheduler": {
                "partitions": [
                    {
                        "name": "betelgeuse",
                        "gpu_type": "NVIDIA H200",
                        "gpus_per_node": 8,
                        "reservation": "gpu_0003_grpA",
                    },
                    {
                        "name": "titan",
                        "gpu_type": "Tesla P100-PCIE-16GB",
                        "gpus_per_node": 8,
                    },
                ]
            },
            "gpus": [
                {
                    "location": "98dci4-gpu-0003",
                    "model": "NVIDIA H200",
                    "count": 8,
                    "current_use": "embed_server",
                },
                {
                    "location": "98dci4-gpu-0002",
                    "model": "Tesla P100-PCIE-16GB",
                    "count": 8,
                },
            ],
        }

    def test_titan_location_emits_titan_node_without_reservation(self):
        """A Titan location wins even when the H200 partition is listed first."""
        location = SimpleNamespace(
            is_compute=True,
            scheduler="slurm",
            partition="titan",
            facility="iter",
        )
        with (
            patch("imas_codex.settings.get_embedding_location", return_value="titan"),
            patch(
                "imas_codex.remote.locations.resolve_location",
                return_value=location,
            ),
            patch(
                "imas_codex.cli.services._compute_config",
                return_value=self._compute_config(),
            ),
            patch("imas_codex.cli.services._stop_login_services"),
            patch(
                "imas_codex.cli.services._get_node_state",
                return_value=("idle", "none"),
            ),
            patch(
                "imas_codex.cli.services._run_remote",
                side_effect=["/home/user", "Submitted batch job 12345"],
            ) as mock_run,
        ):
            from imas_codex.cli.services import _submit_service_job

            _submit_service_job(
                "codex-embed",
                "echo start",
                cpus=2,
                mem="16G",
                gpus=1,
            )

        submit_cmd = mock_run.call_args_list[1][0][0]
        assert "#SBATCH --partition=titan" in submit_cmd
        assert "#SBATCH --nodelist=98dci4-gpu-0002" in submit_cmd
        assert "#SBATCH --reservation=" not in submit_cmd

    def test_h200_location_preserves_partition_reservation(self):
        """A configured H200 location retains its scheduler reservation."""
        location = SimpleNamespace(
            is_compute=True,
            scheduler="slurm",
            partition="betelgeuse",
            facility="iter",
        )
        with (
            patch(
                "imas_codex.settings.get_embedding_location",
                return_value="betelgeuse",
            ),
            patch(
                "imas_codex.remote.locations.resolve_location",
                return_value=location,
            ),
            patch(
                "imas_codex.cli.services._compute_config",
                return_value=self._compute_config(),
            ),
        ):
            from imas_codex.cli.services import _embedding_target

            partition, gpu = _embedding_target()

        assert partition["name"] == "betelgeuse"
        assert partition["reservation"] == "gpu_0003_grpA"
        assert gpu["location"] == "98dci4-gpu-0003"


class TestEmbedStop:
    """Embedding shutdown cleans the node that owns the live job."""

    def test_location_change_cleans_live_job_node(self):
        """Stopping after a config move does not leave workers on the old node."""

        def run_remote(command, **_kwargs):
            if "systemctl" in command:
                raise subprocess.CalledProcessError(1, command)
            return ""

        live_job = {
            "job_id": "12345",
            "state": "RUNNING",
            "node": "98dci4-gpu-0003",
        }
        with (
            patch("imas_codex.cli.embed._get_embed_job", return_value=live_job),
            patch("imas_codex.cli.embed._cancel_service_job", return_value=True),
            patch("imas_codex.cli.embed._run_remote", side_effect=run_remote),
            patch("imas_codex.cli.embed.time.sleep"),
            patch(
                "imas_codex.cli.services._gpu_entry",
                return_value={"location": "98dci4-gpu-0002"},
            ) as configured_target,
            patch(
                "imas_codex.cli.services._run_on_node", return_value="9876\n"
            ) as run_on_node,
            patch("imas_codex.cli.services._kill_embed_orphans") as kill_orphans,
        ):
            from imas_codex.cli.embed import embed_stop

            embed_stop.callback()

        configured_target.assert_not_called()
        assert run_on_node.call_args.args[0] == "98dci4-gpu-0003"
        kill_orphans.assert_called_once_with("98dci4-gpu-0003")


# ===========================================================================
# TestLlmCliBase64
# ===========================================================================


class TestLlmCliBase64:
    """Anti-base64 gates for llm_cli.py remote command construction."""

    def test_launcher_no_base64(self):
        """Launcher script delivery must not use 'echo ... | base64 -d'."""
        commands_sent: list[str] = []

        def capture(cmd, **kwargs):
            commands_sent.append(cmd)
            return ""

        with (
            patch("imas_codex.cli.llm_cli._run_llm_remote", side_effect=capture),
            patch("imas_codex.cli.llm_cli._llm_port", return_value=4000),
            patch("imas_codex.cli.llm_cli._llm_ssh", return_value="login01"),
            patch("imas_codex.cli.llm_cli._SERVICES_DIR", "/services"),
            patch("imas_codex.cli.llm_cli._PROJECT", "/project"),
            # Prevent actual subprocess.run / start_db from firing
            patch("imas_codex.cli.llm_cli.subprocess.run"),
            patch("imas_codex.cli.llm_cli.os.environ.get", return_value=None),
        ):
            try:
                from imas_codex.cli.llm_cli import _deploy_login_llm_direct

                _deploy_login_llm_direct()
            except Exception:
                pass  # incomplete mocking is fine — we only need the commands

        launcher_write_cmds = [c for c in commands_sent if "launcher" in c.lower()]
        for cmd in launcher_write_cmds:
            assert "base64" not in cmd, (
                f"base64 found in launcher write command: {cmd!r}"
            )

    def test_systemd_install_no_base64(self):
        """Systemd unit installation must not use 'echo ... | base64 -d'."""
        commands_sent: list[str] = []

        def capture(cmd, **kwargs):
            commands_sent.append(cmd)
            if "hostname" in cmd:
                return "login01.iter.org"
            return ""

        with (
            patch("imas_codex.cli.llm_cli._run_llm_remote", side_effect=capture),
            patch("imas_codex.cli.llm_cli._llm_port", return_value=4000),
            patch("imas_codex.cli.llm_cli._SERVICES_DIR", "/services"),
            patch("imas_codex.cli.llm_cli._PROJECT", "/project"),
        ):
            try:
                from imas_codex.cli.llm_cli import _install_llm_service_remote

                _install_llm_service_remote()
            except Exception:
                pass

        for cmd in commands_sent:
            assert "base64" not in cmd, (
                f"base64 found in systemd install command: {cmd!r}"
            )

    def test_stale_cleanup_no_base64(self):
        """Stale instance cleanup must not use 'echo ... | base64 -d | bash'."""
        commands_sent: list[str] = []

        def capture(cmd, **kwargs):
            commands_sent.append(cmd)
            return ""

        with (
            patch("imas_codex.cli.llm_cli._run_llm_remote", side_effect=capture),
            patch("imas_codex.cli.llm_cli._llm_port", return_value=4000),
        ):
            from imas_codex.cli.llm_cli import _stop_stale_llm_instances

            _stop_stale_llm_instances()

        for cmd in commands_sent:
            assert "base64" not in cmd, (
                f"base64 found in stale cleanup command: {cmd!r}"
            )


# ===========================================================================
# TestLlmDbBase64
# ===========================================================================


class TestLlmDbBase64:
    """Anti-base64 gates for llm_db.py remote command construction."""

    def test_ensure_pg_bin_no_base64(self):
        """pg_bin discovery command must not use 'echo ... | base64 -d'."""
        commands_sent: list[str] = []

        def capture(cmd, **kwargs):
            commands_sent.append(cmd)
            # First call: cat bin_cache → CalledProcessError (not cached)
            if "cat" in cmd and ".pg_bin" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            # Second call: the pg_bin discovery command
            if "pgserver" in cmd or "pg_ctl" in cmd:
                return "/usr/lib/postgresql/15/bin"
            return ""

        with (
            patch("imas_codex.cli.llm_db._run", side_effect=capture),
            patch("imas_codex.cli.services._SERVICES_DIR", "/services"),
        ):
            try:
                from imas_codex.cli.llm_db import _ensure_pg_bin

                _ensure_pg_bin()
            except Exception:
                pass  # incomplete mocking is fine

        # The discovery command (if sent) must be base64-free
        discovery_cmds = [
            c
            for c in commands_sent
            if "pgserver" in c or "python" in c or "uv run" in c
        ]
        for cmd in discovery_cmds:
            assert "base64" not in cmd, f"base64 found in pg_bin discovery: {cmd!r}"
