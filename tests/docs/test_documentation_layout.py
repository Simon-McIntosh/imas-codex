from pathlib import Path

import pytest
from reckon.resources import TYPE_ROOTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _typed_resource_path_violations(docs_root: Path) -> list[Path]:
    typed_roots = frozenset(TYPE_ROOTS.values())
    violations = []
    for document in docs_root.rglob("*.html"):
        relative = document.relative_to(docs_root)
        if not relative.parts or relative.parts[0] not in typed_roots:
            continue
        tail = relative.parts[1:]
        if len(tail) == 1 or (len(tail) == 2 and tail[0] == "archive"):
            continue
        violations.append(relative)
    return sorted(violations)


def _assert_typed_resource_paths(docs_root: Path) -> None:
    violations = _typed_resource_path_violations(docs_root)
    rendered = "\n".join(f"  - {path.as_posix()}" for path in violations)
    assert not violations, f"Typed resource path violations:\n{rendered}"


def test_documentation_uses_typed_resource_paths() -> None:
    _assert_typed_resource_paths(PROJECT_ROOT / "docs")


def test_nested_typed_resource_is_reported(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    illegal = docs_root / "research" / "nested" / "result.html"
    legal_archive = docs_root / "evidence" / "archive" / "result.html"
    illegal.parent.mkdir(parents=True)
    legal_archive.parent.mkdir(parents=True)
    illegal.touch()
    legal_archive.touch()

    with pytest.raises(AssertionError, match="research/nested/result.html") as error:
        _assert_typed_resource_paths(docs_root)

    assert "evidence/archive/result.html" not in str(error.value)
