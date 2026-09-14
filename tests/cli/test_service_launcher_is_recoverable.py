"""Regression tests for durable SLURM service launch scripts."""

from __future__ import annotations

from unittest.mock import patch


def _captured_submission(
    *, cpus: int, mem: str, gpus: int, job_name: str = "codex-embed"
) -> str:
    from imas_codex.cli import services

    commands: list[str] = []

    def run_remote(command: str, **_kwargs: object) -> str:
        commands.append(command)
        if command == "echo $HOME":
            return "/home/service-user"
        return "Submitted batch job 12345"

    with (
        patch.object(services, "_stop_login_services"),
        patch.object(services, "_run_remote", side_effect=run_remote),
        patch.object(
            services,
            "_embedding_target",
            return_value=({"name": "titan"}, {"location": "gpu-0002"}),
        ),
        patch.object(services, "_get_node_state", return_value=("idle", "none")),
    ):
        services._submit_service_job(
            job_name,
            "exec example-service",
            cpus=cpus,
            mem=mem,
            gpus=gpus,
        )

    return commands[-1]


def test_service_submission_preserves_its_script_beside_the_log() -> None:
    submit_command = _captured_submission(cpus=2, mem="16G", gpus=1)

    service_dir = "/home/service-user/.local/share/imas-codex/services"
    script_path = f"{service_dir}/codex-embed.sh"

    assert f"> {script_path}" in submit_command
    assert f"sbatch {script_path}" in submit_command
    assert "rm -f" not in submit_command
    assert f"#SBATCH --output={service_dir}/codex-embed.log" in submit_command
    assert "#SBATCH --cpus-per-task=2" in submit_command
    assert "#SBATCH --mem=16G" in submit_command
    assert "#SBATCH --gres=gpu:1" in submit_command


def test_embed_footprint_matches_the_submitted_directives() -> None:
    from imas_codex.cli import services

    requested: dict[str, object] = {}

    def ensure_service_job(*_args: object, **kwargs: object) -> dict[str, str]:
        requested.update(kwargs)
        return {"job_id": "12345"}

    with (
        patch.object(services, "_get_embed_job", return_value=None),
        patch.object(
            services,
            "_embedding_target",
            return_value=({"name": "titan"}, {"location": "gpu-0002"}),
        ),
        patch.object(services, "_kill_embed_orphans"),
        patch.object(services, "_embed_port", return_value=18765),
        patch.object(services, "_ensure_service_job", side_effect=ensure_service_job),
    ):
        services.deploy_embed(gpus=1, workers=1)

    footprint = services._embed_service_footprint(gpus=1, workers=1)
    assert requested == {
        "cpus": footprint.cpus,
        "mem": footprint.mem,
        "gpus": footprint.gpus,
        "health_cmd": "curl -sf http://gpu-0002:18765/health",
        "health_test": '"status"',
    }

    submit_command = _captured_submission(
        cpus=footprint.cpus,
        mem=footprint.mem,
        gpus=footprint.gpus,
    )
    assert f"#SBATCH --cpus-per-task={footprint.cpus}" in submit_command
    assert f"#SBATCH --mem={footprint.mem}" in submit_command
    assert f"#SBATCH --gres=gpu:{footprint.gpus}" in submit_command
