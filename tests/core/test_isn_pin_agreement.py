"""Keep the build and runtime ISN requirements on one exact release."""

import importlib.metadata
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ISN_DISTRIBUTION = "imas-standard-names"


def _isn_requirements(value: object) -> list[Requirement]:
    if isinstance(value, dict):
        return [
            requirement
            for item in value.values()
            for requirement in _isn_requirements(item)
        ]
    if isinstance(value, list):
        return [
            requirement for item in value for requirement in _isn_requirements(item)
        ]
    if isinstance(value, str):
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            return []
        if canonicalize_name(requirement.name) == ISN_DISTRIBUTION:
            return [requirement]
    return []


def pyproject_path() -> Path:
    """The pyproject.toml carrying the ISN pins, resolved from this file."""
    return Path(__file__).resolve().parents[2] / "pyproject.toml"


def pinned_isn_version(pyproject: Path | None = None) -> Version:
    """The single exact ISN version every pyproject dependency entry names."""
    path = pyproject if pyproject is not None else pyproject_path()
    with path.open("rb") as handle:
        requirements = _isn_requirements(tomllib.load(handle))

    assert len(requirements) >= 2, "both ISN dependency pins must be present"
    versions: set[Version] = set()
    for requirement in requirements:
        specifiers = list(requirement.specifier)
        assert requirement.url is None and len(specifiers) == 1
        specifier = specifiers[0]
        assert specifier.operator == "==" and not specifier.version.endswith(".*")
        versions.add(Version(specifier.version))
    assert len(versions) == 1, "ISN dependency pins must name the same version"
    return versions.pop()


def installed_isn_version() -> Version:
    """The ISN version the running interpreter imports."""
    return Version(importlib.metadata.version(ISN_DISTRIBUTION))


def test_isn_requirements_agree_on_one_exact_version() -> None:
    pinned_isn_version()


def test_isn_pin_matches_the_installed_package() -> None:
    """A pin that names a version other than the installed one is drift.

    The pin exists to hold the runtime, so it must agree with the package
    actually importable here: bumping the installed package without the pin,
    or the pin without the package, has to be visible rather than assumed.
    """
    pinned = pinned_isn_version()
    try:
        installed = installed_isn_version()
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - env fault
        pytest.fail(
            f"pyproject pins {ISN_DISTRIBUTION}=={pinned} but the package is "
            "not installed, so the pin cannot hold the runtime"
        )

    assert installed == pinned, (
        f"pyproject pins {ISN_DISTRIBUTION}=={pinned} but the installed "
        f"package is {installed}; bump the pin with the package"
    )
