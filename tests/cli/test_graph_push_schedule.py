"""The graph push schedule runs from the network-capable login node."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from click.testing import CliRunner

from imas_codex.cli.graph import graph, registry
from imas_codex.cli.graph.registry import graph_push


def _schedule_files(tmp_path: Path) -> tuple[Path, Path]:
    service_dir = tmp_path / "systemd" / "user"
    return (
        service_dir / "imas-codex-graph-push.service",
        service_dir / "imas-codex-graph-push.timer",
    )


def _configure_schedule(
    monkeypatch, tmp_path: Path
) -> tuple[Path, Path, list[list[str]]]:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='imas-codex'\n")
    service_file, timer_file = _schedule_files(tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        registry, "_push_schedule_paths", lambda: (service_file, timer_file)
    )
    monkeypatch.setattr(registry, "_push_schedule_project_dir", lambda: project_dir)
    monkeypatch.setattr(
        registry.shutil,
        "which",
        lambda command: (
            "/opt/oras/bin/oras" if command == "oras" else f"/usr/bin/{command}"
        ),
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("socket.getfqdn", lambda: "login.example.org")

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "active\n", "")

    monkeypatch.setattr(registry.subprocess, "run", run)
    return service_file, timer_file, calls


def test_push_help_documents_login_node_schedule() -> None:
    result = CliRunner().invoke(graph_push, ["--help"])

    assert result.exit_code == 0
    assert "--schedule" in result.output
    assert "Install the weekly login-node push timer." in result.output
    assert "--cycle" in result.output
    assert "Run one verified full-graph push cycle." in result.output


def test_generated_service_and_timer_run_weekly_on_login_node() -> None:
    service = registry._push_schedule_service_text(
        Path("/srv/imas-codex"),
        uv_path="/usr/bin/uv",
        oras_path="/opt/oras/bin/oras",
        hostname="login.example.org",
    )
    timer = registry._push_schedule_timer_text()

    assert "ConditionHost=login.example.org" in service
    assert "Type=oneshot" in service
    assert "WorkingDirectory=/srv/imas-codex" in service
    assert 'Environment="PATH=/opt/oras/bin:' in service
    assert (
        "ExecStart=/usr/bin/uv run --no-sync --project /srv/imas-codex "
        "imas-codex graph push --cycle"
    ) in service
    assert "OnCalendar=weekly" in timer
    assert "Persistent=true" in timer
    assert "Unit=imas-codex-graph-push.service" in timer
    assert "sbatch" not in service + timer


def test_schedule_dry_run_prints_units_without_writing_or_systemctl(
    monkeypatch, tmp_path
) -> None:
    service_file, timer_file, calls = _configure_schedule(monkeypatch, tmp_path)

    result = CliRunner().invoke(graph_push, ["--schedule", "--dry-run"])

    assert result.exit_code == 0
    assert "[DRY RUN] systemd user service:" in result.output
    assert "ConditionHost=login.example.org" in result.output
    assert "imas-codex graph push --cycle" in result.output
    assert "[DRY RUN] systemd user timer:" in result.output
    assert "OnCalendar=weekly" in result.output
    assert "systemctl was not called" in result.output
    assert not service_file.exists()
    assert not timer_file.exists()
    assert calls == []


def test_schedule_install_writes_and_enables_timer(monkeypatch, tmp_path) -> None:
    service_file, timer_file, calls = _configure_schedule(monkeypatch, tmp_path)

    result = CliRunner().invoke(graph_push, ["--schedule"])

    assert result.exit_code == 0
    assert service_file.exists()
    assert timer_file.exists()
    assert "imas-codex graph push --cycle" in service_file.read_text()
    assert "OnCalendar=weekly" in timer_file.read_text()
    assert calls == [
        ["/usr/bin/systemctl", "--user", "daemon-reload"],
        [
            "/usr/bin/systemctl",
            "--user",
            "enable",
            "--now",
            "imas-codex-graph-push.timer",
        ],
    ]
    assert "installed and enabled" in result.output


def test_schedule_install_refuses_without_oras(monkeypatch, tmp_path) -> None:
    service_file, timer_file, calls = _configure_schedule(monkeypatch, tmp_path)
    monkeypatch.setattr(
        registry.shutil,
        "which",
        lambda command: None if command == "oras" else f"/usr/bin/{command}",
    )

    result = CliRunner().invoke(graph_push, ["--schedule"])

    assert result.exit_code != 0
    assert (
        "oras not found; cannot install the weekly graph push service" in result.output
    )
    assert not service_file.exists()
    assert not timer_file.exists()
    assert calls == []


def test_schedule_status_uses_systemctl(monkeypatch, tmp_path) -> None:
    service_file, timer_file, calls = _configure_schedule(monkeypatch, tmp_path)
    service_file.parent.mkdir(parents=True)
    service_file.write_text("service")
    timer_file.write_text("timer")

    result = CliRunner().invoke(graph_push, ["--schedule-status"])

    assert result.exit_code == 0
    assert "active" in result.output
    assert calls == [
        [
            "/usr/bin/systemctl",
            "--user",
            "status",
            "imas-codex-graph-push.timer",
        ]
    ]


def test_remove_schedule_disables_and_deletes_units(monkeypatch, tmp_path) -> None:
    service_file, timer_file, calls = _configure_schedule(monkeypatch, tmp_path)
    service_file.parent.mkdir(parents=True)
    service_file.write_text("service")
    timer_file.write_text("timer")

    result = CliRunner().invoke(graph_push, ["--remove-schedule"])

    assert result.exit_code == 0
    assert not service_file.exists()
    assert not timer_file.exists()
    assert calls == [
        [
            "/usr/bin/systemctl",
            "--user",
            "disable",
            "--now",
            "imas-codex-graph-push.timer",
        ],
        [
            "/usr/bin/systemctl",
            "--user",
            "stop",
            "imas-codex-graph-push.service",
        ],
        ["/usr/bin/systemctl", "--user", "daemon-reload"],
    ]
    assert "timer removed" in result.output


def test_graph_has_no_separate_offsite_push_command() -> None:
    result = CliRunner().invoke(graph, ["offsite-push", "--help"])

    assert result.exit_code != 0
    assert "No such command 'offsite-push'" in result.output


def test_schedule_and_cycle_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(graph_push, ["--schedule", "--cycle"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
