"""Lint: fan-out configuration is read only by the refine_name worker.

Structured fan-out is a refine-only mechanism — it splits one refine
attempt into competing arms and charges the extra calls to the refine
budget.  A read of ``fanout_settings`` (or a call to ``run_fanout`` /
``should_trigger_fanout``) from a sibling worker would silently enable
fan-out at that site too, multiplying its LLM spend and its telemetry
under a pool that was never budgeted or measured for it.  Confinement is
therefore enforced mechanically rather than by convention.

This lint AST-parses ``imas_codex/standard_names/workers.py`` and asserts
that every reference to fan-out configuration or fan-out entry points sits
inside ``process_refine_name_batch`` — never in compose, review or
refine_docs.  It is intentionally narrow: one AST walk over an allow-list
of fan-out attribute and function names.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

WORKERS_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "imas_codex"
    / "standard_names"
    / "workers.py"
)

# Names that hint a caller is consuming fan-out machinery.  Reading a
# ``FanoutSettings`` attribute outside the refine-name worker means fan-out
# has been plugged into a second site.
_FANOUT_ATTRS: frozenset[str] = frozenset(
    {
        "enabled",  # FanoutSettings.enabled — only consumed by the refine site
        "refine_trigger_keywords",
        "refine_trigger_comment_dims",
        "refine_trigger_comment_chars",
        "refine_fanout_arm_percent",
    }
)

# Names of fan-out functions whose presence outside refine_name would
# indicate a new plug-in site.
_FANOUT_FUNCTIONS: frozenset[str] = frozenset(
    {
        "run_fanout",
        "should_trigger_fanout",
        "assign_arm",
    }
)


def _load_module() -> ast.Module:
    src = WORKERS_PATH.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(WORKERS_PATH))


class _FanoutReferenceVisitor(ast.NodeVisitor):
    """Walk the module collecting ``(function_name, lineno)`` pairs for
    every reference to a fan-out function."""

    def __init__(self) -> None:
        self.fn_stack: list[str] = ["<module>"]
        self.references: list[
            tuple[str, str, int]
        ] = []  # (containing_fn, name, lineno)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.fn_stack.append(node.name)
        self.generic_visit(node)
        self.fn_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 — AST API
        if node.id in _FANOUT_FUNCTIONS:
            self.references.append((self.fn_stack[-1], node.id, node.lineno))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # FanoutSettings.<attr> reads.
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "fanout_settings"
            and node.attr in _FANOUT_ATTRS
        ):
            self.references.append((self.fn_stack[-1], node.attr, node.lineno))
        self.generic_visit(node)


def test_fanout_calls_confined_to_refine_name() -> None:
    """``run_fanout`` / ``should_trigger_fanout`` are referenced only
    inside ``process_refine_name_batch``."""
    tree = _load_module()
    visitor = _FanoutReferenceVisitor()
    visitor.visit(tree)

    # Filter to the function-call references.
    fn_refs = [
        (fn, name, line)
        for (fn, name, line) in visitor.references
        if name in _FANOUT_FUNCTIONS
    ]

    assert fn_refs, (
        "expected at least one reference to run_fanout / should_trigger_fanout "
        "in workers.py — the refine_name worker drives fan-out."
    )

    bad = [
        (fn, name, line)
        for (fn, name, line) in fn_refs
        if fn != "process_refine_name_batch"
    ]
    assert not bad, (
        f"fan-out entry points reached from a non-refine worker: {bad}.  "
        "Fan-out calls must live only inside process_refine_name_batch, "
        "otherwise a sibling pool silently fans out on an unbudgeted path."
    )


def test_fanout_settings_reads_confined_to_refine_name() -> None:
    """Attribute reads on ``fanout_settings`` are confined to refine_name."""
    tree = _load_module()
    visitor = _FanoutReferenceVisitor()
    visitor.visit(tree)

    attr_refs = [
        (fn, name, line)
        for (fn, name, line) in visitor.references
        if name in _FANOUT_ATTRS
    ]
    bad = [
        (fn, name, line)
        for (fn, name, line) in attr_refs
        if fn != "process_refine_name_batch"
    ]
    assert not bad, (
        f"FanoutSettings reads outside refine_name: {bad}.  Only "
        "process_refine_name_batch may consume fan-out configuration."
    )


def test_lint_runs_on_synthetic_violation() -> None:
    """Self-test: the visitor flags a fan-out call planted in a compose worker."""
    src = """
async def process_compose_batch(batch):
    settings = load_fanout_settings()
    if settings.enabled:
        await run_fanout(site="compose")
    return 0
"""
    tree = ast.parse(src)
    visitor = _FanoutReferenceVisitor()
    visitor.visit(tree)
    fns = [name for (fn, name, _) in visitor.references if name == "run_fanout"]
    assert fns, "self-test failed to detect the synthetic violation"
