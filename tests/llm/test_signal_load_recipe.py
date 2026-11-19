"""The load recipe the server returns is the skill's pattern, executed, not quoted."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from imas_codex.llm.server import AgentsServer
from imas_codex.remote import codac_uda as service
from imas_codex.remote.archive_access import (
    LOAD_RECIPE_ACCESS,
    LOAD_RECIPE_EXAMPLE,
    SIGNAL_LOAD_RECIPE,
    load_recipe_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / ".claude" / "skills" / "archive-signals" / "SKILL.md"

INVENTED_SELECTOR = "DEMO-PLANT:TEMP"
INVENTED_PULSE = "DEMO:TRIAL/7"


def _shared_block(text: str, name: str) -> str:
    start, end = f"<!-- recipe:{name}:start -->", f"<!-- recipe:{name}:end -->"
    assert text.count(start) == 1, f"{name} start marker must appear exactly once"
    assert text.count(end) == 1, f"{name} end marker must appear exactly once"
    body = text.split(start, 1)[1].split(end, 1)[0].strip("\n")
    if body.startswith("```python\n"):
        body = body.removeprefix("```python\n")
        assert body.endswith("\n```"), f"{name} example must keep its python fence"
        body = body.removesuffix("\n```") + "\n"
    return body


def test_tool_text_and_skill_carry_the_same_access_pattern():
    """The tool and the skill quote one source, so they cannot drift apart."""
    skill = SKILL.read_text()
    assert _shared_block(skill, "access-pattern").rstrip("\n") == (
        LOAD_RECIPE_ACCESS.rstrip("\n")
    )
    assert _shared_block(skill, "example").rstrip("\n") == (
        LOAD_RECIPE_EXAMPLE.rstrip("\n")
    )
    recipe = load_recipe_text()
    assert LOAD_RECIPE_ACCESS.rstrip("\n") in recipe
    assert LOAD_RECIPE_EXAMPLE.rstrip("\n") in recipe


def test_recipe_states_the_three_things_a_load_needs_beyond_the_server():
    first_paragraph = load_recipe_text().split("\n\n")[1]
    assert "imas-codex package" in first_paragraph
    assert "m-uda-client" in first_paragraph
    assert "private host configuration" in first_paragraph


def test_recipe_text_carries_only_invented_identities():
    """Zero real host, selector or pulse names; the invented ones are present."""
    recipe = load_recipe_text()
    assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", recipe), "no IP literals"
    assert "://" not in recipe, "no endpoint URLs"
    selectors = set(re.findall(r"\b[A-Z][A-Z0-9-]*:[A-Za-z0-9/_-]+", recipe))
    assert selectors == {INVENTED_SELECTOR, INVENTED_PULSE}

    # A recorded, genuinely private identity list is only available here; when
    # it is, it is a positive control that the scan sees real names at all.
    private = (
        Path.home()
        / ".cache"
        / "imas-codex-codac-uda"
        / "catalogue-private-config.yaml"
    )
    if not private.exists():
        pytest.skip("no private configuration on this host to compare against")
    private_text = private.read_text()
    hosts = set(re.findall(r"host:\s*[\"']?([A-Za-z0-9_.-]+)", private_text))
    assert hosts, "positive control: the private configuration names hosts"
    assert {host for host in hosts if host in recipe} == set()


def _receipts(epoch_ns: bool) -> dict:
    base = {
        "name": INVENTED_SELECTOR,
        "time_dtype": "int64" if epoch_ns else "float64",
        "value_dtype": "float32",
        "time_base": "epoch_ns" if epoch_ns else "pulse_seconds",
        "mode": "raw",
        "decimated": False,
        "sampling_verification": "cadence_checked",
        "unit": "K",
        "description": "Demonstration temperature",
    }
    if epoch_ns:
        return {
            **base,
            "time": [1_700_000_000_000_000_500, 1_700_000_000_000_001_500],
            "values": [1.5, 2.5],
        }
    return {**base, "time": [0.0, 0.5, 1.0], "values": [1.5, 2.5, 3.5]}


@pytest.fixture
def recorded_response(monkeypatch):
    """Stand a recorded receipt where the archive client would run."""
    calls: list[dict] = []

    def runner(script, **kwargs):
        assert script == "codac_uda.py"
        request = kwargs["input_data"]
        calls.append(request)
        absolute = request.get("time_format", "absolute") == "absolute"
        return json.dumps({"ok": True, "result": _receipts(absolute)})

    monkeypatch.setattr(service, "run_python_script", runner)
    return calls


@pytest.fixture
def private_config(tmp_path, monkeypatch):
    path = tmp_path / "private-config.yaml"
    path.write_text(
        json.dumps(
            {
                "data_systems": {
                    "codac_uda": {
                        "archives": [
                            {"role": "fast", "host": "fast.invalid"},
                            {"role": "slow", "host": "archive.invalid"},
                            {
                                "role": "operator",
                                "host": "operators.invalid",
                                "excluded": True,
                            },
                        ],
                        "readable_roles": ["slow"],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("ARCHIVE_CONFIG", str(path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return path


def test_recipe_example_runs_and_returns_arrays_with_units(
    recorded_response, private_config
):
    """The example is executed as written, not only printed."""
    namespace: dict = {"__name__": "__main__", "__builtins__": __builtins__}
    exec(compile(LOAD_RECIPE_EXAMPLE, "<recipe example>", "exec"), namespace)

    window, pulse = namespace["window"], namespace["pulse"]
    assert isinstance(window.time, np.ndarray)
    assert isinstance(window.values, np.ndarray)
    assert window.time.shape == window.values.shape == (2,)
    assert window.metadata["unit"] == "K"
    assert window.metadata["time_base"] == "epoch_ns"
    assert pulse.time.shape == pulse.values.shape == (3,)
    assert pulse.metadata["unit"] == "K"
    assert pulse.metadata["time_base"] == "pulse_seconds"
    assert [call.get("time_format", "absolute") for call in recorded_response] == [
        "absolute",
        "relative",
    ]
    assert recorded_response[1]["pulse"] == INVENTED_PULSE


def _tool_names(server: AgentsServer) -> set[str]:
    return {
        key.removeprefix("tool:").rstrip("@")
        for key in server.mcp._local_provider._components
        if key.startswith("tool:")
    }


def test_recipe_tool_is_read_only_facility_surface():
    read_only = AgentsServer(read_only=True, dd_only=False, no_embed=True)
    assert "get_signal_load_recipe" in _tool_names(read_only)
    assert not {"repl", "add_to_graph"} & _tool_names(read_only)
    dd_only = AgentsServer(dd_only=True, no_embed=True)
    assert "get_signal_load_recipe" not in _tool_names(dd_only)


@pytest.mark.timeout(180)
def test_read_only_server_serves_the_recipe_outside_the_checkout(tmp_path):
    """A client started away from the checkout fetches the recipe over stdio."""
    import anyio
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    import imas_codex

    package_root = Path(imas_codex.__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), *filter(None, [environment.get("PYTHONPATH")])]
    )
    environment["HOME"] = str(tmp_path)
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-c",
            "from imas_codex.cli import main; main()",
            "serve",
            "--read-only",
            "--no-embed",
        ],
        env=environment,
        cwd=str(tmp_path),
        keep_alive=False,
        log_file=tmp_path / "server.log",
    )

    async def fetch():
        async with Client(transport) as client:
            names = {tool.name for tool in await client.list_tools()}
            result = await client.call_tool("get_signal_load_recipe", {})
            return names, result

    names, result = anyio.run(fetch)
    assert "get_signal_load_recipe" in names
    text = "".join(
        item.text for item in result.content if getattr(item, "type", "") == "text"
    )
    assert text.strip() == load_recipe_text().strip()
    assert not (tmp_path / "imas_codex").exists(), "client ran outside the checkout"
    assert SIGNAL_LOAD_RECIPE.startswith(text[:40])
