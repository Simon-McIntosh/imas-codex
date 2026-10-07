"""Every discovery prompt must carry its response model's generated example.

A local endpoint is served a request that no longer carries ``response_format``
(see ``base/llm.py``), so the schema a structured call needs must reach the
engine inside the prompt itself. This test derives the prompt↔response-model
pairs by walking the AST of ``imas_codex.discovery`` — no table of pairs is
maintained here — and asserts the exact example ``get_pydantic_schema_json``
generates for each response model appears verbatim in the rendered prompt, so
naming a field in prose cannot mask a missing, truncated or swapped generated
block.
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

# Prompts whose rendered body is a decisions-questions mapping consumed by
# ``acall_decisions``, not a structured response model.  Such a prompt carries
# no generated example, so a function that makes a ``response_model`` call must
# not be paired with it even when it also builds the questions.
_DECISIONS_PROMPTS = {"code/triage"}

# The prompts whose call sites this walk must reach; a positive control that the
# analysis actually follows discovery's render→builder→call dataflow rather than
# silently returning nothing.
EXPECTED_PROMPTS = {
    "paths/triage",
    "paths/scorer",
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
    if direct and direct not in _DECISIONS_PROMPTS:
        return direct
    for builder in _builder_calls(enclosing.body).values():
        if builder in imports:
            source_module, original = imports[builder]
        else:
            source_module, original = current_module, builder
        prompt = render_map.get((source_module, original))
        if prompt and prompt not in _DECISIONS_PROMPTS:
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


def test_walk_reaches_expected_prompts():
    """Positive control: the AST walk must reach the named discovery prompts."""
    pairs = _collect_pairs()
    missing = EXPECTED_PROMPTS - set(pairs)
    assert not missing, f"AST walk did not reach: {sorted(missing)}"
    assert pairs["signals/source_unwind"] == "SignalSourceCodeUnwindBatch"


@pytest.mark.parametrize("prompt_name", sorted(_collect_pairs()))
def test_prompt_carries_generated_example_verbatim(prompt_name):
    model_name = _collect_pairs()[prompt_name]
    model = _model_objects()[model_name]
    generated = prompt_loader.get_pydantic_schema_json(model)
    rendered = prompt_loader.render_prompt(prompt_name, {})
    assert generated in rendered, (
        f"prompt {prompt_name!r} does not contain the generated example for "
        f"{model_name} verbatim"
    )


# The scope questions the code/triage decisions prompt must carry in both arms.
TRIAGE_SCOPE_NOULS = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
    "reads_or_writes_reconstruction_db",
)


@pytest.mark.parametrize("with_content", [False, True])
def test_code_triage_asks_every_scope_question_in_both_arms(with_content):
    """The triage prompt carries the five scope nouls in both arms.

    The content arm adds the graded and facet Scores; the scope questions are
    shared, so a noul dropped from the shared block is absent from both arms.
    Rendering with each arm and requiring every scope question pins that.
    """
    rendered = prompt_loader.render_prompt(
        "code/triage", {"with_content": with_content}
    )
    for question in TRIAGE_SCOPE_NOULS:
        assert f'"{question}"' in rendered, (
            f"code/triage (with_content={with_content}) omitted {question}"
        )
