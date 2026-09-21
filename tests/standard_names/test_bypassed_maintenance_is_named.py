"""Scoped maintenance bypasses remain deliberate and leave a receipt."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from imas_codex.standard_names import loop
from imas_codex.standard_names.loop import summary_table
from tests.standard_names import test_scoped_global_maintenance as scoped_maintenance

_LOOP_LOGGER = "imas_codex.standard_names.loop"


def _contains_negated_maintenance_bypass(node: ast.expr) -> bool:
    return any(
        isinstance(candidate, ast.UnaryOp)
        and isinstance(candidate.op, ast.Not)
        and isinstance(candidate.operand, ast.Name)
        and candidate.operand.id == "skip_global_maintenance"
        for candidate in ast.walk(node)
    )


def _calls_any(nodes: list[ast.stmt], names: set[str]) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id in names
        for node in nodes
        for candidate in ast.walk(node)
    )


def _unreceipted_direct_maintenance_guards(path: Path) -> list[int]:
    """Return direct bypass guards that neither delegate nor record the skip."""
    tree = ast.parse(path.read_text(), filename=str(path))
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_sn_pools"
    )
    violations: list[int] = []
    for guard in (node for node in ast.walk(run) if isinstance(node, ast.If)):
        if not _contains_negated_maintenance_bypass(guard.test):
            continue
        delegated = _calls_any(guard.body, {"_global_maintenance_call"})
        recorded = _calls_any(guard.orelse, {"_record_bypassed_maintenance"})
        if not delegated and not recorded:
            violations.append(guard.lineno)
    return violations


def test_every_direct_maintenance_guard_records_its_bypass() -> None:
    path = Path(loop.__file__)
    violations = _unreceipted_direct_maintenance_guards(path)
    rendered = "\n".join(f"{path}:{line}" for line in violations)
    assert not violations, f"unreceipted maintenance bypass guard(s):\n{rendered}"


@pytest.mark.asyncio
async def test_scoped_run_names_every_bypassed_maintenance_pass(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO, logger=_LOOP_LOGGER)

    maintenance_mocks = scoped_maintenance._maintenance_mocks

    def named_maintenance_mocks(stack):
        mocks = maintenance_mocks(stack)
        for function_name, function_mock in mocks.items():
            function_mock.__name__ = function_name
        return mocks

    monkeypatch.setattr(
        scoped_maintenance,
        "_maintenance_mocks",
        named_maintenance_mocks,
    )
    result = await scoped_maintenance._run_loop(skip_global_maintenance=True)
    summary, maintenance, *_ = result

    receipt = summary.bypassed_maintenance_passes
    assert summary_table(summary)["bypassed_maintenance_passes"] == receipt

    skipped_writers = {
        function_name
        for function_name, function_mock in maintenance.items()
        if function_name != "run_orphan_sweep_loop" and not function_mock.called
    }
    assert skipped_writers <= set(receipt)

    messages = [record.getMessage() for record in caplog.records]
    prefix = "run_sn_pools: global maintenance bypassed — "
    logged_bypasses = [
        message.removeprefix(prefix)
        for message in messages
        if message.startswith(prefix)
    ]
    assert logged_bypasses == receipt

    summary_message = next(
        message
        for message in messages
        if message.startswith("run_sn_pools: bypassed global maintenance summary")
    )
    assert f"count={len(receipt)}" in summary_message
    assert ", ".join(receipt) in summary_message
