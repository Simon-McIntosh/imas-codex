"""Move facade references to the module that owns each symbol.

The split inventory supplies candidate owners. Every candidate is checked against
the live module before a replacement is offered. Use ``--check`` to inspect the
tree and ``--apply --scope-manifest PATH`` to edit only granted paths.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DESIGN = Path(__file__).with_name("srtj-graph-ops-split-design.json")
BASE = "imas_codex.standard_names"
FACADES = {"graph_ops": "graph_store", "workers": "worker_tasks"}


def owners() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    inventory = json.loads(DESIGN.read_text())["inventories"]
    result: dict[str, dict[str, str]] = {}
    stale: dict[str, list[str]] = {}
    for facade, package in FACADES.items():
        candidates = inventory[f"{facade}.py"]["symbols"]
        result[facade] = {}
        stale[facade] = []
        for symbol, record in candidates.items():
            module = f"{BASE}.{package}.{record['module']}"
            if hasattr(importlib.import_module(module), symbol):
                result[facade][symbol] = module
            else:
                stale[facade].append(symbol)
    return result, stale


def offset(lines: list[str], line: int, column: int) -> int:
    return sum(len(part.encode()) for part in lines[: line - 1]) + column


def span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    return offset(lines, node.lineno, node.col_offset), offset(  # type: ignore[attr-defined]
        lines,
        node.end_lineno,
        node.end_col_offset,  # type: ignore[attr-defined]
    )


def module_path(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        prefix = module_path(expr.value)
        return f"{prefix}.{expr.attr}" if prefix else None
    return None


def rewrite(
    source: str, mapping: dict[str, dict[str, str]]
) -> tuple[str, Counter[str]]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    aliases: dict[str, str] = {}
    string_aliases: dict[str, str] = {}
    edits: list[tuple[int, int, str, str]] = []
    additions: set[str] = set()
    counts: Counter[str] = Counter()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                for facade in FACADES:
                    if item.name == f"{BASE}.{facade}" and item.asname:
                        aliases[item.asname] = facade
        elif isinstance(node, ast.ImportFrom):
            if node.module == BASE:
                for item in node.names:
                    if item.name in FACADES:
                        aliases[item.asname or item.name] = item.name
            elif node.module in (f"{BASE}.{name}" for name in FACADES):
                facade = node.module.rsplit(".", 1)[-1]
                groups: dict[str, list[str]] = {}
                retained: list[str] = []
                for item in node.names:
                    spelled = item.name + (f" as {item.asname}" if item.asname else "")
                    owner = mapping[facade].get(item.name)
                    if owner:
                        groups.setdefault(owner, []).append(spelled)
                        counts["imports"] += 1
                    else:
                        retained.append(spelled)
                if groups:
                    indentation = lines[node.lineno - 1][: node.col_offset]
                    statements = []
                    if retained:
                        statements.append(
                            f"from {node.module} import {', '.join(retained)}"
                        )
                    statements.extend(
                        f"from {owner} import {', '.join(names)}"
                        for owner, names in sorted(groups.items())
                    )
                    begin, end = span(lines, node)
                    edits.append(
                        (begin, end, ("\n" + indentation).join(statements), "imports")
                    )
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        for facade in FACADES:
                            if node.value.value == f"{BASE}.{facade}":
                                string_aliases[target.id] = facade

    def replacement_for(facade: str, symbol: str) -> str | None:
        owner = mapping[facade].get(symbol)
        if not owner:
            return None
        short = owner.rsplit(".", 1)[-1]
        alias = f"_owning_{short}"
        package = owner.rsplit(".", 1)[0]
        additions.add(f"from {package} import {short} as {alias}")
        return f"{alias}.{symbol}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            path = module_path(node.value)
            facade = aliases.get(path or "")
            if facade is None and path:
                for candidate in FACADES:
                    if path == f"{BASE}.{candidate}":
                        facade = candidate
            if facade:
                replacement = replacement_for(facade, node.attr)
                if replacement:
                    edits.append((*span(lines, node), replacement, "attributes"))
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            call = module_path(node.func)
            if call in {"patch.object", "mock.patch.object", "monkeypatch.setattr"}:
                target = module_path(node.args[0])
                facade = aliases.get(target or "")
                name = node.args[1]
                if (
                    facade
                    and isinstance(name, ast.Constant)
                    and isinstance(name.value, str)
                ):
                    owner = mapping[facade].get(name.value)
                    if owner:
                        replacement_for(facade, name.value)
                        alias = f"_owning_{owner.rsplit('.', 1)[-1]}"
                        edits.append(
                            (*span(lines, node.args[0]), alias, "object_patches")
                        )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for facade in FACADES:
                prefix = f"{BASE}.{facade}."
                if node.value.startswith(prefix):
                    symbol = node.value[len(prefix) :].split(".", 1)[0]
                    owner = mapping[facade].get(symbol)
                    if owner:
                        tail = node.value[len(prefix) + len(symbol) :]
                        edits.append(
                            (
                                *span(lines, node),
                                repr(f"{owner}.{symbol}{tail}"),
                                "patch_strings",
                            )
                        )
        elif isinstance(node, ast.JoinedStr):
            if len(node.values) >= 2 and isinstance(node.values[0], ast.FormattedValue):
                expression = node.values[0].value
                first = node.values[1]
                if isinstance(expression, ast.Name) and isinstance(first, ast.Constant):
                    facade = string_aliases.get(expression.id)
                    if (
                        facade
                        and isinstance(first.value, str)
                        and first.value.startswith(".")
                    ):
                        symbol = first.value[1:].split(".", 1)[0]
                        owner = mapping[facade].get(symbol)
                        if owner and len(node.values) == 2:
                            tail = first.value[1 + len(symbol) :]
                            edits.append(
                                (
                                    *span(lines, node),
                                    repr(f"{owner}.{symbol}{tail}"),
                                    "built_patch_strings",
                                )
                            )
                        elif (
                            facade == "graph_ops"
                            and first.value == "."
                            and len(node.values) == 3
                            and isinstance(node.values[2], ast.FormattedValue)
                            and isinstance(node.values[2].value, ast.Name)
                        ):
                            name = node.values[2].value.id
                            additions.add(
                                f"from {BASE}.graph_store import owner_module_path as _owner_module_path"
                            )
                            edits.append(
                                (
                                    *span(lines, node),
                                    f'f"{{_owner_module_path({name})}}.{{{name}}}"',
                                    "dynamic_patch_strings",
                                )
                            )

    # Nested AST nodes can both identify one patch site; the outermost replacement
    # wins, leaving a single exact source span to apply.
    chosen: list[tuple[int, int, str, str]] = []
    for item in sorted(edits, key=lambda value: (value[0], -value[1])):
        if chosen and item[0] < chosen[-1][1]:
            continue
        chosen.append(item)
    data = source.encode()
    for begin, end, replacement, kind in reversed(chosen):
        data = data[:begin] + replacement.encode() + data[end:]
        counts[kind] += kind != "imports"
    result = data.decode()
    if additions:
        parsed = ast.parse(result)
        insertion = 0
        for statement in parsed.body:
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ):
                insertion = statement.end_lineno
            elif (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
            ):
                insertion = statement.end_lineno
            else:
                break
        parts = result.splitlines(keepends=True)
        parts[insertion:insertion] = [line + "\n" for line in sorted(additions)]
        result = "".join(parts)
    ast.parse(result)
    return result, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--scope-manifest", type=Path)
    parser.add_argument("--only", choices=("product", "tests"))
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    mapping, stale = owners()
    allowed: set[str] = set()
    if args.scope_manifest:
        record = args.scope_manifest.read_text()
        line = next(
            line
            for line in record.splitlines()
            if line.startswith("orientation_write_paths: ")
        )
        allowed = set(json.loads(line.partition(": ")[2]))
    paths = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=ROOT, text=True
    ).splitlines()
    if args.scope_manifest:
        paths = [name for name in paths if name in allowed]
    if args.only == "product":
        paths = [name for name in paths if name.startswith("imas_codex/")]
    elif args.only == "tests":
        paths = [name for name in paths if name.startswith("tests/")]
    paths = [name for name in paths if name not in args.exclude]
    totals: Counter[str] = Counter()
    changed: list[str] = []
    forwarders: dict[str, list[str]] = {}
    value_aliases: dict[str, list[str]] = {}
    for facade in FACADES:
        module = ROOT / "imas_codex" / "standard_names" / f"{facade}.py"
        tree = ast.parse(module.read_text())
        forwarders[facade] = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (ast.get_docstring(node) or "").startswith("Forward to ")
        ]
        value_aliases[facade] = [
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id.startswith("_moved_")
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
    for name in paths:
        path = ROOT / name
        original = path.read_text()
        revised, counts = rewrite(original, mapping)
        if revised != original:
            changed.append(name)
            totals.update(counts)
            if args.apply:
                if name not in allowed:
                    raise SystemExit(f"outside write scope: {name}")
                path.write_text(revised)
    print(
        json.dumps(
            {
                "files": len(changed),
                "references": sum(totals.values()),
                "by_kind": totals,
                "forwarders": forwarders,
                "value_aliases": value_aliases,
                "stale_map": stale,
                "paths": changed,
            },
            indent=2,
        )
    )
    return int(
        args.check
        and (bool(changed) or any(forwarders.values()) or any(value_aliases.values()))
    )


if __name__ == "__main__":
    raise SystemExit(main())
