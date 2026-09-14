"""Resource requests for the SLURM embedding service."""

from __future__ import annotations

from unittest.mock import patch


def _deploy_request(**deploy_kwargs: int) -> dict[str, object]:
    from imas_codex.cli import services

    request: dict[str, object] = {}

    def ensure_service_job(*args: object, **kwargs: object) -> dict[str, str]:
        request.update(kwargs)
        request["service_command"] = args[1]
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
        services.deploy_embed(**deploy_kwargs)

    return request


def _submission_script(request: dict[str, object]) -> str:
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
            "codex-embed",
            str(request["service_command"]),
            cpus=int(request["cpus"]),
            mem=str(request["mem"]),
            gpus=int(request["gpus"]),
        )

    return commands[-1]


def test_default_embed_submission_requests_one_gpu_and_two_cpus() -> None:
    request = _deploy_request()

    assert request["cpus"] == 2
    assert request["mem"] == "16G"
    assert request["gpus"] == 1
    assert "--gpus 0 --workers 1" in str(request["service_command"])

    submission = _submission_script(request)
    assert "#SBATCH --gres=gpu:1" in submission
    assert "#SBATCH --cpus-per-task=2" in submission
    assert "#SBATCH --mem=16G" in submission
    assert "--gpus 0 --workers 1" in submission


def test_explicit_multi_gpu_embed_submission_retains_its_larger_footprint() -> None:
    request = _deploy_request(gpus=8, workers=8)

    assert request["cpus"] == 9
    assert request["mem"] == "32G"
    assert request["gpus"] == 8
    assert "--gpus 0,1,2,3,4,5,6,7 --workers 8" in str(request["service_command"])

    submission = _submission_script(request)
    assert "#SBATCH --gres=gpu:8" in submission
    assert "#SBATCH --cpus-per-task=9" in submission
    assert "#SBATCH --mem=32G" in submission
    assert "--gpus 0,1,2,3,4,5,6,7 --workers 8" in submission
