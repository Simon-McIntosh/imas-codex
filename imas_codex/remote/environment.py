"""Resolve a facility's remote environment into executor parameters.

One home for the host-wide interpreter-and-shell recipe a facility needs
before any remote script runs. The facility config declares a
``remote_environment`` block; a data system may override ``python_command``
and ``setup_commands`` for its own scripts. This module merges the two so
every caller reaches the same answer without re-implementing the precedence.

The executor stays facility-agnostic: it receives ``python_command`` and
``setup_commands`` as parameters, and the facility-aware layer uses this
resolver to produce them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_PYTHON_COMMAND = "python3"


@dataclass(frozen=True)
class RequiredModule:
    """A module a facility's remote scripts must be able to import."""

    name: str
    path: str | None = None


@dataclass(frozen=True)
class RemoteEnvironment:
    """The interpreter and shell environment for one facility's remote scripts."""

    python_command: str
    setup_commands: tuple[str, ...]
    min_python_version: str | None
    required_modules: tuple[RequiredModule, ...]


def _as_module(entry: Any) -> RequiredModule | None:
    """Normalise a required-modules entry to a RequiredModule.

    Accepts a bare name, a mapping with ``name`` and optional ``path``, or a
    model instance exposing those attributes. Returns None for an entry with
    no name.
    """
    if isinstance(entry, str):
        name = entry.strip()
        return RequiredModule(name=name) if name else None
    if isinstance(entry, dict):
        name = str(entry.get("name") or "").strip()
        if not name:
            return None
        path = entry.get("path")
        return RequiredModule(name=name, path=path or None)
    name = str(getattr(entry, "name", "") or "").strip()
    if not name:
        return None
    return RequiredModule(name=name, path=getattr(entry, "path", None) or None)


def resolve_remote_environment(
    facility_config: dict[str, Any],
    data_system: str | None = None,
) -> RemoteEnvironment:
    """Merge a facility's remote environment with a data system's override.

    The facility ``remote_environment`` block supplies the default. When
    ``data_system`` is given and its configuration declares ``python_command``
    or ``setup_commands``, those win for that system's scripts.

    Args:
        facility_config: Loaded facility configuration mapping.
        data_system: Data system whose override applies (e.g., "edas").

    Returns:
        RemoteEnvironment with the resolved interpreter and setup commands.
    """
    config = facility_config or {}
    block = config.get("remote_environment") or {}

    python_command = block.get("python_command") or DEFAULT_PYTHON_COMMAND
    setup_commands = list(block.get("setup_commands") or [])
    min_python_version = block.get("min_python_version") or None
    required_modules = tuple(
        module
        for module in (
            _as_module(entry) for entry in (block.get("required_modules") or [])
        )
        if module is not None
    )

    if data_system:
        systems = config.get("data_systems") or {}
        system_config = systems.get(data_system) or {}
        if isinstance(system_config, dict):
            if system_config.get("python_command"):
                python_command = system_config["python_command"]
            if system_config.get("setup_commands"):
                setup_commands = list(system_config["setup_commands"])

    return RemoteEnvironment(
        python_command=python_command,
        setup_commands=tuple(setup_commands),
        min_python_version=min_python_version,
        required_modules=required_modules,
    )
