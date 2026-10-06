"""AST guard that every model-seat reference names a live section.

A seat passed only as a default parameter is invisible to a call-shape search,
so the instrument is a syntax-tree walk over ``imas_codex``. It asserts that
every string literal bound to a model-seat position — the first argument of
``get_model`` / ``get_model_config``, the first argument of the health checks,
a ``section=`` or ``model_section=`` keyword, a ``section`` or ``model_section``
parameter default, and a ``section`` / ``model_section`` dataclass field value —
is a member of ``settings.MODEL_SECTIONS``.

``get_reasoning_effort`` reads ``reasoning-effort`` from a configuration section
that need not carry a single ``model`` (``sn-review`` owns a reviewer chain and
``sn-fanout`` a proposer model), so its section argument is validated against
``MODEL_SECTIONS`` or any live top-level ``[tool.imas-codex.*]`` section. A
retired seat (removed from ``MODEL_SECTIONS`` and ``pyproject.toml``) satisfies
neither and is still caught.

Retiring a seat therefore turns every stale reference into a collection-time
failure instead of a runtime ``Unknown model section`` at the first call.

The scan root is overridable with ``IMAS_CODEX_SEAT_SCAN_ROOT`` so the same
test can run against a scratch copy of the package to prove it fails on a
planted dead seat.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import imas_codex
from imas_codex import settings

# Functions whose first positional argument is a model seat.
_SEAT_FIRST_ARG = frozenset({"get_model", "get_model_config", "get_reasoning_effort"})
# Functions whose ``section`` argument is a model seat.
_SEAT_SECTION_ARG = frozenset({"llm_health_check", "llm_deep_health_check"})
# Keyword names that carry a model seat wherever they appear.
_SEAT_KEYWORDS = frozenset({"section", "model_section"})
# Parameter / field names that carry a model seat.
_SEAT_PARAM_NAMES = frozenset({"section", "model_section"})

_DEFAULT_SCAN_ROOT = Path(imas_codex.__file__).parent


def _scan_root() -> Path:
    override = os.environ.get("IMAS_CODEX_SEAT_SCAN_ROOT")
    return Path(override) if override else _DEFAULT_SCAN_ROOT


def _live_config_sections() -> frozenset[str]:
    """Top-level ``[tool.imas-codex.*]`` section names that actually exist."""
    settings._load_pyproject_settings.cache_clear()
    return frozenset(settings._load_pyproject_settings())


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _string_constants(node: ast.Call) -> list[tuple[str, str, str]]:
    """Return (label, literal, kind) for every seat-shaped literal in *node*.

    ``kind`` is ``"seat"`` for a reference that must name a member of
    ``MODEL_SECTIONS`` and ``"reasoning"`` for a ``get_reasoning_effort``
    section, which may name any live ``[tool.imas-codex.*]`` section.
    """
    found: list[tuple[str, str, str]] = []
    name = _called_name(node)

    if name in _SEAT_FIRST_ARG and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            kind = "reasoning" if name == "get_reasoning_effort" else "seat"
            found.append((f"{name}() first argument", first.value, kind))
    if name in _SEAT_SECTION_ARG and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((f"{name}() positional section", first.value, "seat"))

    for kw in node.keywords:
        if kw.arg in _SEAT_KEYWORDS and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                found.append((f"{name}({kw.arg}=...)", kw.value.value, "seat"))
    return found


def _default_strings(tree: ast.AST) -> list[tuple[str, str, str]]:
    """Return (label, literal, kind) for seat-shaped parameter and field
    defaults (all of kind ``"seat"``)."""
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            defaults = [
                *([None] * (len(positional) - len(args.defaults))),
                *args.defaults,
            ]
            for arg, default in zip(positional, defaults, strict=True):
                if (
                    arg.arg in _SEAT_PARAM_NAMES
                    and isinstance(default, ast.Constant)
                    and isinstance(default.value, str)
                ):
                    found.append(
                        (f"{node.name}(...) param {arg.arg}", default.value, "seat")
                    )
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if (
                    arg.arg in _SEAT_PARAM_NAMES
                    and isinstance(default, ast.Constant)
                    and isinstance(default.value, str)
                ):
                    found.append(
                        (f"{node.name}(...) param {arg.arg}", default.value, "seat")
                    )
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id in _SEAT_PARAM_NAMES
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                found.append((f"field {target.id}", node.value.value, "seat"))
    return found


def _collect_references() -> list[tuple[str, str, str, str]]:
    """Walk every module under the scan root; return (path, label, literal, kind)."""
    references: list[tuple[str, str, str, str]] = []
    root = _scan_root()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover — unparseable file is not a seat
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for label, literal, kind in _string_constants(node):
                    references.append((str(path), label, literal, kind))
        for label, literal, kind in _default_strings(tree):
            references.append((str(path), label, literal, kind))
    return references


def test_every_seat_name_reference_is_a_known_section() -> None:
    """No string bound to a model-seat position names a retired section."""
    references = _collect_references()
    # Positive control: the walk must see the seats that DO exist, so an empty
    # (or truncated) scan cannot masquerade as a clean one.
    assert any(
        label.startswith("get_model") and literal in settings.MODEL_SECTIONS
        for _, label, literal, _ in references
    ), "scan found no seat references at all — the walk is broken, not clean"

    live_config = _live_config_sections()
    unknown = [
        (path, label, literal)
        for path, label, literal, kind in references
        if not (
            literal in settings.MODEL_SECTIONS
            or (kind == "reasoning" and literal in live_config)
        )
    ]
    assert not unknown, (
        "model-seat reference(s) name a section outside MODEL_SECTIONS: "
        + "; ".join(
            f"{literal!r} from {label} in {path}" for path, label, literal in unknown
        )
    )
