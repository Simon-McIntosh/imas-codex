"""Every ``sn`` command that can spend model calls writes a run log.

A command that spends places model calls whose record is the run log under
``~/.local/share/imas-codex/logs/``. When it calls no logging helper the run
leaves nothing behind, so its evidence can cite only the command line and its
exit status. These tests hold the whole class, not a list: a command qualifies
structurally, from what the command *is* rather than from a hard-coded roster,
so a newly added spending command is caught before it silently runs unlogged.

Qualification is computed from two signals, matching how a command reaches a
model call in this CLI: it takes a cost-limit option, or its callback reaches
one of the compose / review / documentation / inline-review entry points listed
below. A command that qualifies must call ``setup_logging``.

The behavioural tests run ``sn harmonize --report``, ``sn edit`` and
``sn rescore`` with the graph and the model paths stubbed and assert
``configure_cli_logging`` is reached with domain ``sn`` and the command's stem.
Patching that function is also what keeps every test off the home log
directory: it is the only writer.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click import Command, Group
from click.testing import CliRunner

from imas_codex.cli import sn as sn_module
from imas_codex.cli.sn import sn

# Option parameter means "this command spends": the user states a spend ceiling.
COST_LIMIT_PARAMS = frozenset({"cost_limit"})

# Callable names that reach a model call from an ``sn`` command: the pooled
# compose / documentation / review drain in this module, the inline review, the
# harmonize family apply and its cost probe, the model benchmark, the release
# review that writes LLM notes, and the fold-back summary that does the same.
SPENDING_ENTRY_POINTS = frozenset(
    {
        "_run_sn_cmd",
        "_docs_drain_factory",
        "run_inline_review",
        "rescore_name",
        "apply_families",
        "plan_family_applies",
        "measure_doc_call_cost",
        "run_benchmark",
        "rescore_descriptions_report",
        "run_role_bench",
        "judge_name_physics",
        "run_calibration_gate",
        "run_review_release",
        "run_sn_review_engine",
        "tag_fold_back",
        "drain_validation_backlog",
    }
)


# ---------------------------------------------------------------------------
# Structural walk of the ``sn`` command group
# ---------------------------------------------------------------------------


def _collect_imports(body: list[ast.stmt], target: dict[str, str]) -> None:
    for node in body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                target[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target[alias.asname or alias.name.split(".")[0]] = alias.name


def _module_call_graph(path: Path) -> dict[str, tuple[set[str], set[str]]]:
    """Map each function name in *path* to its called names and import aliases."""
    tree = ast.parse(path.read_text())
    module_aliases: dict[str, str] = {}
    _collect_imports(tree.body, module_aliases)
    graph: dict[str, tuple[set[str], set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        aliases = dict(module_aliases)
        _collect_imports(node.body, aliases)
        calls: set[str] = set()
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                calls.add(func.value.id)
                calls.add(func.attr)
        graph.setdefault(node.name, (calls, set(aliases)))
    return graph


def _reachable_names(start: str, graph: dict[str, tuple[set[str], set[str]]]) -> set[str]:
    """Every call name reachable from *start* through functions in this module."""
    seen = {start}
    stack = [start]
    names: set[str] = set()
    while stack:
        calls, _aliases = graph.get(stack.pop(), (set(), set()))
        for call in calls:
            tokens = {call, call.rsplit(".", 1)[-1]}
            names |= tokens
            for token in tokens:
                if token in graph and token not in seen:
                    seen.add(token)
                    stack.append(token)
    return names


def _iter_commands(group: Group, prefix: Group | None = None):
    for name, command in group.commands.items():
        qualified = f"{prefix} {name}" if prefix else name
        if isinstance(command, Group):
            yield from _iter_commands(command, qualified)
        else:
            yield qualified, command


def _qualification(
    command: Command, graph: dict[str, tuple[set[str], set[str]]]
) -> str | None:
    """Why *command* can spend, or ``None`` when it cannot."""
    params = {param.name for param in command.params}
    if params & COST_LIMIT_PARAMS:
        return "takes a cost-limit option"
    reached = _reachable_names(command.callback.__name__, graph)
    hit = sorted(reached & SPENDING_ENTRY_POINTS)
    if hit:
        return "reaches " + ", ".join(hit)
    return None


def test_structural_walk_finds_the_command_group():
    names = {name for name, _ in _iter_commands(sn)}
    # A walk that found nothing would pass the spending assertion vacuously.
    assert {"run", "review", "harmonize", "edit", "rescore"} <= names


def test_every_spending_sn_command_calls_setup_logging():
    graph = _module_call_graph(Path(inspect.getsourcefile(sn_module)))
    found: list[tuple[str, str, bool]] = []
    for name, command in _iter_commands(sn):
        why = _qualification(command, graph)
        if why is None:
            continue
        calls, _aliases = graph.get(command.callback.__name__, (set(), set()))
        found.append((name, why, "setup_logging" in calls))

    report = "\n".join(
        f"  {name}: {why} -> setup_logging={logs}" for name, why, logs in found
    )
    print(f"spending sn commands found:\n{report}")
    missing = [name for name, _why, logs in found if not logs]
    assert not missing, (
        "spending commands that call no logging helper "
        f"(a run leaves no log file): {missing}\n{report}"
    )
    # The three named in the plan brief must be among them, so the assertion
    # above is never trivially true because the walk stopped resolving.
    assert {"harmonize", "edit", "rescore"} <= {name for name, _w, _l in found}


# ---------------------------------------------------------------------------
# Behavioural: each stubbed spending run reaches configure_cli_logging
# ---------------------------------------------------------------------------


def _spy(monkeypatch: Any) -> Any:
    """Patch the log writer so no real file is created under the home dir."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _record(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("imas_codex.cli.logging.configure_cli_logging", _record)
    return calls


def _install_stub_graph(monkeypatch: Any) -> None:
    from imas_codex.graph import client as graph_client

    class _StubGraph:
        def __enter__(self) -> _StubGraph:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: _StubGraph())


def test_harmonize_report_writes_a_run_log(monkeypatch: Any) -> None:
    calls = _spy(monkeypatch)
    _install_stub_graph(monkeypatch)
    monkeypatch.setattr(
        "imas_codex.standard_names.harmonize.report_family_rows",
        lambda *a, **k: [],
    )

    result = CliRunner().invoke(sn, ["harmonize", "--all", "--report"])

    assert result.exit_code == 0, result.output
    assert calls, "sn harmonize --report configured no run log"
    (args, kwargs) = calls[0]
    assert args[0] == "sn"
    assert kwargs["facility"] == "sn-harmonize"


def test_edit_inline_review_writes_a_run_log(monkeypatch: Any) -> None:
    calls = _spy(monkeypatch)
    monkeypatch.setattr(sn_module, "_require_embed_ready", lambda *_a, **_k: None)
    plan = SimpleNamespace(
        target="electron_temperature",
        run_id="run-1",
        applied=True,
        blocked=None,
        actions=[],
        successor=None,
        cascade_deferred=[],
        mode="hint",
    )
    outcome = SimpleNamespace(ran=True, all_accepted=True)
    monkeypatch.setattr("imas_codex.standard_names.edit.apply_edit", lambda *a, **k: plan)
    monkeypatch.setattr(
        "imas_codex.standard_names.edit.run_inline_review", lambda *a, **k: outcome
    )
    monkeypatch.setattr(sn_module, "_render_edit_plan", lambda *a, **k: None)
    monkeypatch.setattr(sn_module, "_render_inline_review_outcome", lambda *a, **k: None)

    result = CliRunner().invoke(
        sn, ["edit", "electron_temperature", "--hint", "clarify", "--reason", "why"]
    )

    assert result.exit_code == 0, result.output
    assert calls, "sn edit configured no run log"
    (args, kwargs) = calls[0]
    assert args[0] == "sn"
    assert kwargs["facility"] == "sn-edit"


def test_rescore_inline_review_writes_a_run_log(monkeypatch: Any) -> None:
    calls = _spy(monkeypatch)
    monkeypatch.setattr(sn_module, "_require_embed_ready", lambda *_a, **_k: None)
    receipt = {
        "ok": True,
        "sn_id": "electron_temperature",
        "prior_stage": "reviewed",
        "run_id": "run-2",
        "reviewed": False,
    }
    monkeypatch.setattr("imas_codex.standard_names.edit.rescore_name", lambda *a, **k: receipt)

    result = CliRunner().invoke(sn, ["rescore", "electron_temperature"])

    assert result.exit_code == 0, result.output
    assert calls, "sn rescore configured no run log"
    (args, kwargs) = calls[0]
    assert args[0] == "sn"
    assert kwargs["facility"] == "sn-rescore"
