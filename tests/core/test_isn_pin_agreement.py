"""Keep the build and runtime ISN requirements on one exact release."""

import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


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
        if canonicalize_name(requirement.name) == "imas-standard-names":
            return [requirement]
    return []


def test_isn_requirements_agree_on_one_exact_version() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
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
