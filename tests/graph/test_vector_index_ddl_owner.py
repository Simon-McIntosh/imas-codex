"""Single-owner pin for the ``CREATE VECTOR INDEX`` DDL.

``GraphClient.ensure_vector_indexes`` is the one place that composes a
``CREATE VECTOR INDEX`` statement.  Every other module that needs a vector
index must delegate to it rather than repeat the statement, so the index shape
(dimensions, similarity function, quantization, registered filter properties)
has exactly one definition.

Two properties are pinned here:

1. No Python source outside ``imas_codex/graph/client.py`` carries a
   ``CREATE VECTOR INDEX`` string literal.  The scan walks every string
   constant in the package -- not just the DDL form that carries
   ``IF NOT EXISTS`` -- so a statement of any shape is caught.  Prose that
   merely names the grammar is excluded by *being a docstring*: a docstring is
   text describing the code, while a string built to be sent to the database is
   a composition of the DDL.  The exclusion keys on that structural fact, never
   on any wording inside the string.
2. The modules that used to hand-write the statement call the owner instead.
"""

import ast
from pathlib import Path

import pytest

MARKER = "CREATE VECTOR INDEX"
OWNER = Path("graph") / "client.py"

# Modules that previously composed the statement themselves and must now
# route through GraphClient.ensure_vector_indexes.
DELEGATING_MODULES = [
    "imas_codex.graph.build_dd",
    "imas_codex.graph.dd_identifier_enrichment",
    "imas_codex.graph.dd_ids_enrichment",
    "imas_codex.discovery.wiki.pipeline",
]

# A docstring is the first statement of a module, class or function; those
# string constants describe the code rather than compose a statement.
_DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def _package_root() -> Path:
    import imas_codex

    return Path(imas_codex.__file__).resolve().parent


def _docstring_constants(tree: ast.AST) -> set[int]:
    """Ids of the ``ast.Constant`` nodes that are docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def _ddl_bearing_files() -> list[Path]:
    """Package-relative paths holding a ``CREATE VECTOR INDEX`` string.

    A string constant carrying the marker anywhere but in a docstring counts;
    the shape of the statement (its ``IF NOT EXISTS`` clause, its options) is
    irrelevant to whether it is a hand-written copy.
    """
    root = _package_root()
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_constants(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and MARKER in node.value
                and id(node) not in docstrings
            ):
                files.append(path.relative_to(root))
                break
    return files


def test_only_the_owner_composes_vector_index_ddl():
    files = _ddl_bearing_files()
    assert files == [OWNER], (
        "CREATE VECTOR INDEX statements must be composed only by "
        f"GraphClient.ensure_vector_indexes in {OWNER}; found {files}"
    )


@pytest.mark.parametrize("module_name", DELEGATING_MODULES)
def test_module_delegates_to_the_owner(module_name):
    relative = Path(*module_name.split(".")[1:]).with_suffix(".py")
    source = (_package_root() / relative).read_text(encoding="utf-8")
    assert "ensure_vector_indexes(" in source, (
        f"{module_name} must delegate vector index DDL to "
        "GraphClient.ensure_vector_indexes"
    )
