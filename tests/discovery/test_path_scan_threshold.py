"""Path gates use the scan threshold setting."""

import ast
from pathlib import Path

import imas_codex.discovery.paths as paths


def test_path_gates_do_not_read_retired_discovery_threshold():
    path_dir = Path(paths.__file__).parent
    modules = list(path_dir.glob("*.py"))
    assert modules
    for module in modules:
        tree = ast.parse(module.read_text())
        retired_imports = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "imas_codex.settings"
            for alias in node.names
            if alias.name == "get_discovery_threshold"
        }
        assert not retired_imports, f"{module} imports a retired path gate"
        assert not any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "get_discovery_threshold"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_discovery_threshold"
            )
            for node in ast.walk(tree)
        ), f"{module} calls a retired path gate"
