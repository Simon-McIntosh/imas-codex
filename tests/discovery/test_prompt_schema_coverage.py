"""Every discovery prompt must carry its response model's schema in its text.

A local endpoint is served a request that no longer carries ``response_format``
(see ``base/llm.py``), so the schema a structured call needs must reach the
engine inside the prompt itself. This test derives the prompt↔response-model
pairs by walking the AST of ``imas_codex.discovery`` — no table of pairs is
maintained here — and asserts each response model's field set is present in the
rendered prompt.
"""

from __future__ import annotations

import ast
import importlib
from functools import lru_cache
from pathlib import Path

import pytest

import imas_codex.discovery as discovery_pkg
from imas_codex.llm import prompt_loader

DISCOVERY_ROOT = Path(discovery_pkg.__file__).parent
_RENDERERS = {"render_prompt", "render_prompt_strict"}

# The prompts whose call sites this walk must reach; a positive control that the
# analysis actually follows discovery's render→builder→call dataflow rather than
# silently returning nothing.
EXPECTED_PROMPTS = {
    "paths/triage",
    "paths/scorer",
    "code/triage",
    "code/scorer",
    "wiki/scorer",
    "wiki/document-scorer",
    "wiki/image-captioner",
    "signals/enrichment",
    "signals/source_unwind",
    "discovery/static-enricher",
}


def _module_key(path: Path) -> str:
    rel = path.relative_to(DISCOVERY_ROOT.parent.parent).with_suffix("")
    return ".".join(rel.parts)


def _iter_module_files() -> list[Path]:
    return sorted(DISCOVERY_ROOT.rglob("*.py"))


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _render_prompt_in(body: list[ast.stmt]) -> str | None:
    """Return the prompt name rendered directly inside *body*, if any."""
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(stmt, ast.Call):
            continue
        func = stmt.func
        name = func.id if isinstance(func, ast.Name) else None
        if name in _RENDERERS and stmt.args:
            literal = _literal_str(stmt.args[0])
            if literal:
                return literal
    return None


def _build_render_map() -> dict[tuple[str, str], str]:
    """(module, function) -> prompt name for every prompt-rendering function."""
    render_map: dict[tuple[str, str], str] = {}
    for path in _iter_module_files():
        tree = ast.parse(path.read_text())
        module = _module_key(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                prompt = _render_prompt_in(node.body)
                if prompt:
                    render_map[(module, node.name)] = prompt
    return render_map


def _imports(tree: ast.Module, module: str) -> dict[str, tuple[str, str]]:
    """local name -> (source module, original name) for one module's imports."""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = (node.module, alias.name)
    return out


def _builder_calls(body: list[ast.stmt]) -> dict[str, str]:
    """variable -> builder function name, for ``var = builder(...)`` in body."""
    out: dict[str, str] = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            builder = None
            if isinstance(func, ast.Name):
                builder = func.id
            elif isinstance(func, ast.Attribute):
                # A method call such as ``self._build_system_prompt(...)``
                # resolves in the module that defines the method.
                builder = func.attr
            if builder:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = builder
    return out


def _enclosing_function(tree: ast.Module, call: ast.Call) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(child is call for child in ast.walk(node)):
                return node
    return None


def _resolve_prompt(
    enclosing: ast.AST,
    imports: dict[str, tuple[str, str]],
    render_map: dict[tuple[str, str], str],
    current_module: str,
) -> str | None:
    direct = _render_prompt_in(enclosing.body)
    if direct:
        return direct
    for builder in _builder_calls(enclosing.body).values():
        if builder in imports:
            source_module, original = imports[builder]
        else:
            source_module, original = current_module, builder
        prompt = render_map.get((source_module, original))
        if prompt:
            return prompt
    return None


@lru_cache(maxsize=1)
def _collect_pairs() -> dict[str, str]:
    """prompt name -> response model name, discovered by AST walk."""
    render_map = _build_render_map()
    pairs: dict[str, str] = {}
    for path in _iter_module_files():
        tree = ast.parse(path.read_text())
        module = _module_key(path)
        imports = _imports(tree, module)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            model = next(
                (
                    kw.value.id
                    for kw in node.keywords
                    if kw.arg == "response_model" and isinstance(kw.value, ast.Name)
                ),
                None,
            )
            if not model:
                continue
            enclosing = _enclosing_function(tree, node)
            if enclosing is None:
                continue
            prompt = _resolve_prompt(enclosing, imports, render_map, module)
            if prompt:
                pairs[prompt] = model
    return pairs


@lru_cache(maxsize=1)
def _model_objects() -> dict[str, type]:
    """Response model name -> class, discovered by scanning the package."""
    out: dict[str, type] = {}
    for path in _iter_module_files():
        tree = ast.parse(path.read_text())
        module = _module_key(path)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Batch"):
                try:
                    obj = getattr(importlib.import_module(module), node.name)
                except (ImportError, AttributeError):
                    continue
                out[node.name] = obj
    return out


def _schema_field_names(model: type) -> set[str]:
    """Every property name anywhere in the model's JSON schema, nested included."""
    names: set[str] = set()
    schema = model.model_json_schema()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(props)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return names


def test_walk_reaches_expected_prompts():
    """Positive control: the AST walk must reach the named discovery prompts."""
    pairs = _collect_pairs()
    missing = EXPECTED_PROMPTS - set(pairs)
    assert not missing, f"AST walk did not reach: {sorted(missing)}"
    assert pairs["signals/source_unwind"] == "SignalSourceCodeUnwindBatch"


@pytest.mark.parametrize("prompt_name", sorted(EXPECTED_PROMPTS))
def test_prompt_text_carries_response_model_fields(prompt_name):
    pairs = _collect_pairs()
    model_name = pairs[prompt_name]
    model = _model_objects()[model_name]
    rendered = prompt_loader.render_prompt(prompt_name, {})
    missing = {f for f in _schema_field_names(model) if f not in rendered}
    assert not missing, (
        f"prompt {prompt_name!r} is missing fields of {model_name}: {sorted(missing)}"
    )
