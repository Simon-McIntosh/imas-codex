"""Single-owner pin for the ``CREATE VECTOR INDEX`` DDL.

``GraphClient.ensure_vector_indexes`` is the one place that composes a
``CREATE VECTOR INDEX`` statement.  Every other module that needs a vector
index must delegate to it rather than repeat the option block, so the index
shape (dimensions, similarity function, quantization, registered filter
properties) has exactly one definition.

Two properties are pinned here:

1. No Python source outside ``imas_codex/graph/client.py`` composes a
   ``CREATE VECTOR INDEX`` statement.  A statement is recognised by its DDL
   form -- ``CREATE VECTOR INDEX ... IF NOT EXISTS`` -- which prose that
   merely names the grammar does not carry.
2. The modules that used to hand-write the statement call the owner instead.
"""

import re
from pathlib import Path

import pytest

# A DDL statement always carries ``IF NOT EXISTS``; both the owner and every
# former hand-written copy placed it on the CREATE line.
STATEMENT = re.compile(r"CREATE\s+VECTOR\s+INDEX\b[^\n]*IF\s+NOT\s+EXISTS")

OWNER = Path("graph") / "client.py"

# Modules that previously composed the statement themselves and must now
# route through GraphClient.ensure_vector_indexes.
DELEGATING_MODULES = [
    "imas_codex.graph.build_dd",
    "imas_codex.graph.dd_identifier_enrichment",
    "imas_codex.graph.dd_ids_enrichment",
    "imas_codex.discovery.wiki.pipeline",
]


def _package_root() -> Path:
    import imas_codex

    return Path(imas_codex.__file__).resolve().parent


def _ddl_bearing_files() -> list[Path]:
    root = _package_root()
    return [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if STATEMENT.search(path.read_text(encoding="utf-8"))
    ]


def test_only_the_owner_composes_vector_index_ddl():
    files = sorted(_ddl_bearing_files())
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
