"""The image entrypoint hands the server the arguments its start command names.

The data-dictionary image starts with the Dockerfile's own ``CMD``; the plant
image is the same build and takes its serve mode from the compose service's
``command``. Neither mode is inserted by the entrypoint, so these tests run the
entrypoint's own argument handling with every process it launches stubbed and
compare the argument list the server receives against the list each start
command names.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO / "docker-entrypoint.sh"
DOCKERFILE = REPO / "Dockerfile"
COMPOSE = REPO / "docker-compose.yml"

DD_COMMAND = ["serve", "--dd-only", "--host", "0.0.0.0", "--port", "8000"]
PLANT_COMMAND = ["serve", "--plant-only", "--host", "0.0.0.0", "--port", "8000"]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _start_command(tmp_path: Path, argv: list[str], env_extra: dict | None = None):
    """Run the entrypoint with the server stubbed and return its arguments."""
    neo4j_home = tmp_path / "neo4j"
    (neo4j_home / "bin").mkdir(parents=True)
    _write_executable(neo4j_home / "bin" / "neo4j", "#!/bin/sh\nexec sleep 60\n")
    _write_executable(neo4j_home / "bin" / "cypher-shell", "#!/bin/sh\nexit 0\n")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_executable(stubs / "curl", "#!/bin/sh\nexit 0\n")
    arguments = tmp_path / "arguments"
    _write_executable(
        stubs / "imas-codex",
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_FILE"\n',
    )

    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "NEO4J_HOME": str(neo4j_home),
        "ARGS_FILE": str(arguments),
        "HOME": str(tmp_path),
    }
    env.update(env_extra or {})

    result = subprocess.run(
        ["bash", str(ENTRYPOINT), *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert arguments.exists(), result.stdout + result.stderr
    return arguments.read_text().splitlines()


def _dockerfile_cmd() -> list[str]:
    """The last ``CMD`` the Dockerfile declares, as the JSON list it is."""
    matches = re.findall(r"^CMD (\[.*\])$", DOCKERFILE.read_text(), re.MULTILINE)
    assert matches, "the Dockerfile declares no CMD"
    return json.loads(matches[-1])


def _compose_plant_command() -> list[str]:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    return services["imas-codex-plant"]["command"]


def test_the_data_dictionary_image_starts_the_mode_its_cmd_names(tmp_path):
    assert _dockerfile_cmd() == DD_COMMAND
    assert _start_command(tmp_path, _dockerfile_cmd()) == DD_COMMAND


def test_the_plant_image_starts_the_mode_its_service_command_names(tmp_path):
    assert _compose_plant_command() == PLANT_COMMAND
    assert _start_command(tmp_path, _compose_plant_command()) == PLANT_COMMAND


def test_the_entrypoint_inserts_no_mode_of_its_own(tmp_path):
    """An environment the plant image never sets cannot change the argument list."""
    assert "IMAS_CODEX_SERVE_MODE" not in ENTRYPOINT.read_text()
    assert "SERVE_MODE" not in (REPO / "Dockerfile").read_text()
    assert (
        _start_command(
            tmp_path,
            DD_COMMAND,
            {"IMAS_CODEX_SERVE_MODE": "--plant-only"},
        )
        == DD_COMMAND
    )


@pytest.mark.parametrize("dropped", ["--plant-only"])
def test_a_plant_command_that_lost_its_mode_is_refused(tmp_path, dropped):
    """Removing the plant mode leaves an argument list the plant image may not start with."""
    start = [argument for argument in PLANT_COMMAND if argument != dropped]
    assert _start_command(tmp_path, start) != PLANT_COMMAND
