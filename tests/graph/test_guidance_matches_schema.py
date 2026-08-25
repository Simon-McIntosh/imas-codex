"""Keep graph-identity guidance aligned with the LinkML declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from imas_codex.graph.schema import GraphSchema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "imas_codex" / "schemas" / "standard_name.yaml"
_GENERIC_STANDARD_NAME_KEY = re.compile(r"^(?:sn|standard_name)_ids?$")
_RETIRED_UNDECLARED_SLOTS = ("docs_review_resolution_method",)


@dataclass(frozen=True)
class _RetiredSpelling:
    spelling: str
    pattern: re.Pattern[str]


def _guidance_files() -> tuple[Path, ...]:
    """Return every checked agent-guidance file in deterministic order."""
    paths = {
        _REPO_ROOT / "AGENTS.md",
        *(_REPO_ROOT / "agents").rglob("*.md"),
        *(_REPO_ROOT / "imas_codex").rglob("AGENTS.md"),
    }
    assert all(path.is_file() for path in paths)
    return tuple(sorted(paths))


def _declared_standard_name_keys(schema: GraphSchema) -> set[str]:
    """Read generic StandardName foreign-key spellings from LinkML."""
    return {
        slot_name
        for label in schema.node_labels
        for slot_name, details in schema.get_all_slots(label).items()
        if _GENERIC_STANDARD_NAME_KEY.fullmatch(slot_name)
        and "standardname" in details.get("description", "").lower()
    }


def _review_axis_values(schema: GraphSchema) -> set[str]:
    """Read the enum values selected by the sole review-axis slot."""
    enums = schema.get_enums()
    enum_names = {
        details["type"]
        for label in schema.node_labels
        for slot_name, details in schema.get_all_slots(label).items()
        if slot_name == "review_axis" and details["type"] in enums
    }
    assert len(enum_names) == 1, (
        "LinkML must declare exactly one enum-backed review_axis slot; "
        f"found {sorted(enum_names)}"
    )
    return set(enums[enum_names.pop()])


def _retired_spellings(schema: GraphSchema) -> tuple[_RetiredSpelling, ...]:
    """Derive retired aliases from the currently declared slot and enum names."""
    declared_keys = _declared_standard_name_keys(schema)
    assert declared_keys, "LinkML declares no generic StandardName foreign keys"

    declared_slots = {
        slot_name
        for label in schema.node_labels
        for slot_name in schema.get_all_slots(label)
    }
    assert not declared_slots.intersection(_RETIRED_UNDECLARED_SLOTS)

    short_key_aliases = {
        f"sn{slot_name.removeprefix('standard_name')}"
        for slot_name in declared_keys
        if slot_name.startswith("standard_name_")
    }
    axis_values = _review_axis_values(schema)
    plural_axis_aliases = {
        f"{axis}s"
        for axis in axis_values
        if not axis.endswith("s") and f"{axis}s" not in axis_values
    }

    exact_spellings = short_key_aliases | set(_RETIRED_UNDECLARED_SLOTS)
    retired = [
        _RetiredSpelling(
            spelling=spelling,
            pattern=re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(spelling)}(?![A-Za-z0-9_])"
            ),
        )
        for spelling in sorted(exact_spellings)
    ]
    retired.extend(
        _RetiredSpelling(
            spelling=f"review_axis={axis}",
            pattern=re.compile(
                rf"review_axis.{{0,120}}[`'\"]{re.escape(axis)}[`'\"]",
                re.DOTALL,
            ),
        )
        for axis in sorted(plural_axis_aliases)
    )
    return tuple(retired)


def _retired_occurrences(
    guidance: dict[Path, str], retired: tuple[_RetiredSpelling, ...]
) -> list[str]:
    """Return path, line, and spelling for every retired guidance occurrence."""
    findings: list[str] = []
    for path, text in guidance.items():
        for item in retired:
            for match in item.pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path}:{line}: {item.spelling}")
    return findings


def _assert_guidance_matches_schema(
    guidance: dict[Path, str], retired: tuple[_RetiredSpelling, ...]
) -> None:
    """Fail with an itemized report when guidance carries a retired spelling."""
    findings = _retired_occurrences(guidance, retired)
    assert not findings, "Retired schema spellings remain in guidance:\n" + "\n".join(
        findings
    )


def test_guidance_contains_zero_retired_schema_spellings() -> None:
    """All agent guidance describes only slot and enum spellings LinkML declares."""
    schema = GraphSchema(_SCHEMA_PATH)
    retired = _retired_spellings(schema)
    guidance = {path: path.read_text() for path in _guidance_files()}

    findings = _retired_occurrences(guidance, retired)
    print(
        f"retired spelling occurrences: {len(findings)} "
        f"across {len(guidance)} guidance files"
    )
    _assert_guidance_matches_schema(guidance, retired)


def test_reintroduced_retired_spelling_fails_guidance_check() -> None:
    """The guard rejects a retired spelling deliberately restored in a fixture."""
    schema = GraphSchema(_SCHEMA_PATH)
    retired = _retired_spellings(schema)
    retired_key = next(item for item in retired if item.spelling.startswith("sn_"))
    fixture_path = Path("fixture-guidance.md")
    fixture = {fixture_path: f"Use `{retired_key.spelling}` for this graph join.\n"}

    with pytest.raises(AssertionError, match=re.escape(retired_key.spelling)):
        _assert_guidance_matches_schema(fixture, retired)
