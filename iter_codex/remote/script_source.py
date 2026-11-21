"""Package declared pure-Python dependencies with isolated facility scripts."""

import importlib.resources

from iter_codex.remote.scripts import SCRIPT_DEPENDENCIES


def load_script_source(script_name: str) -> str:
    """Ship canonical helper source when the facility interpreter lacks the project.

    Dependency modules must import only the standard library at module scope.
    Bootstrap each module separately so future imports retain their semantics.
    The registry is package-owned and never supplied by a reader request.
    """
    source = (
        importlib.resources.files("iter_codex.remote.scripts")
        .joinpath(script_name)
        .read_text()
    )
    dependencies = SCRIPT_DEPENDENCIES.get(script_name, ())
    if not dependencies:
        return source
    bootstrap = ["import sys, types"]
    for name in dependencies:
        for count in range(1, len(name.split("."))):
            parent = ".".join(name.split(".")[:count])
            bootstrap.append(
                f"sys.modules.setdefault({parent!r}, types.ModuleType({parent!r}))"
            )
        path = "/".join(name.split(".")[1:]) + ".py"
        module_source = (
            importlib.resources.files("iter_codex").joinpath(path).read_text()
        )
        bootstrap.extend(
            [
                f"module = types.ModuleType({name!r})",
                f"sys.modules[{name!r}] = module",
                f"exec({module_source!r}, module.__dict__)",
            ]
        )
    bootstrap.append(f"exec({source!r})")
    return "\n".join(bootstrap)
