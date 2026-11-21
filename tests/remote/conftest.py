"""Shared real-process controls for the script executor."""

import asyncio
import subprocess

import pytest

from iter_codex.remote import executor


@pytest.fixture(params=["sync", "async"])
def run_local(request):
    """Exercise the same request through each executor entry point."""

    def run(script_name, **kwargs):
        try:
            if request.param == "sync":
                return executor.run_python_script(script_name, **kwargs)
            return asyncio.run(executor.async_run_python_script(script_name, **kwargs))
        except subprocess.CalledProcessError as exc:
            exc.add_note(exc.stderr)
            raise

    return run
