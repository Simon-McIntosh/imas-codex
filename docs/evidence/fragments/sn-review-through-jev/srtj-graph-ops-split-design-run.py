"""Measure a proposed domain split without importing product modules or using the graph."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("srtj-graph-ops-split-design.json")
ENCODER = tiktoken.get_encoding("cl100k_base")
RANGES = {
    "graph_ops.py": [
        (1, 597, "graph_foundation"),
        (597, 2143, "graph_claim_scope"),
        (2143, 5393, "graph_lineage"),
        (5393, 8020, "graph_catalog"),
        (8020, 9395, "graph_catalog_reset"),
        (9395, 9943, "graph_doc_links"),
        (9943, 12371, "graph_source_bindings"),
        (12371, 15828, "graph_audits"),
        (15828, 17072, "graph_cost_and_claims"),
        (17072, 18176, "graph_review_persistence"),
        (18176, 19805, "graph_refinement"),
        (19805, 23971, "graph_supersession"),
        (23971, 25394, "graph_claim_release"),
        (25394, 26340, "graph_docs_persistence"),
        (26340, 27718, "graph_parent_lifecycle"),
    ],
    "workers.py": [
        (1, 1582, "worker_common"),
        (1582, 3929, "worker_extraction_and_enrichment"),
        (3929, 5835, "worker_compose_and_validate"),
        (5835, 6960, "worker_compose_batches"),
        (6960, 7823, "worker_refine_names"),
        (7823, 9279, "worker_review_names"),
        (9279, 10183, "worker_generate_docs"),
        (10183, 10422, "worker_enrich_parents"),
        (10422, 11370, "worker_review_docs"),
        (11370, 11797, "worker_refine_docs"),
    ],
}


def module_for(source: str, line: int) -> str:
    for start, stop, module in RANGES[source]:
        if start <= line < stop:
            return module
    raise AssertionError((source, line))


def top_level_symbols(node: ast.stmt) -> list[str]:
    def assigned_names(target: ast.expr) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple | ast.List):
            return [name for item in target.elts for name in assigned_names(item)]
        return []

    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in assigned_names(target)]
    if isinstance(node, ast.AnnAssign):
        return assigned_names(node.target)
    if isinstance(node, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname or alias.name for alias in node.names]
    if (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ):
        return [
            name for statement in node.body for name in top_level_symbols(statement)
        ]
    return []


def source_inventory(source: str) -> dict:
    path = ROOT / "imas_codex" / "standard_names" / source
    content = path.read_text()
    lines = content.splitlines(keepends=True)
    tree = ast.parse(content)
    symbols = {}
    definitions = {}
    for node in tree.body:
        for name in top_level_symbols(node):
            # Assignments that repeat a name remain in its first module.
            symbols.setdefault(name, module_for(source, node.lineno))
            definitions.setdefault(
                name, {"line": node.lineno, "kind": type(node).__name__}
            )
    size = {
        module: len(ENCODER.encode("".join(lines[start - 1 : stop - 1])))
        for start, stop, module in RANGES[source]
    }
    refs = Counter()
    examples = defaultdict(list)
    for node in tree.body:
        owner = module_for(source, node.lineno)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Name | ast.Call):
                continue
            name = (
                child.id
                if isinstance(child, ast.Name)
                else child.func.id
                if isinstance(child.func, ast.Name)
                else None
            )
            if name is None or name not in symbols or symbols[name] == owner:
                continue
            if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
                continue
            pair = f"{owner} -> {symbols[name]}"
            refs[pair] += 1
            if len(examples[pair]) < 6 and name not in examples[pair]:
                examples[pair].append(name)
    return {
        "path": str(path.relative_to(ROOT)),
        "source_tokens": len(ENCODER.encode(content)),
        "modules": {
            module: {
                "source_tokens": tokens,
                "imports_from_new_modules": ["graph_foundation"]
                if module.startswith("graph_") and module != "graph_foundation"
                else ["worker_common"]
                if module.startswith("worker_") and module != "worker_common"
                else [],
            }
            for module, tokens in size.items()
        },
        "symbols": {
            name: {"module": module, **definitions[name]}
            for name, module in sorted(symbols.items())
        },
        "cross_module_references": dict(refs),
        "cross_module_reference_examples": dict(examples),
    }


def patch_targets(inventories: dict) -> dict:
    occurrences = Counter()
    by_module = Counter()
    examples = defaultdict(set)
    by_symbol = Counter()
    files_by_module = defaultdict(set)
    unknown = Counter()
    test_files = sorted((ROOT / "tests").rglob("*.py"))
    pattern = re.compile(
        r"^imas_codex\.standard_names\.(graph_ops|workers)\.([A-Za-z_]\w*)$"
    )
    for path in test_files:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            match = pattern.fullmatch(node.value)
            if not match:
                continue
            source = f"{match.group(1)}.py"
            symbol = match.group(2)
            occurrences[source] += 1
            record = inventories[source]["symbols"].get(symbol)
            if record is None:
                unknown[f"{source}:{symbol}"] += 1
                continue
            module = record["module"]
            by_module[module] += 1
            by_symbol[f"{module}:{symbol}"] += 1
            files_by_module[module].add(str(path.relative_to(ROOT)))
            examples[module].add(f"{path.relative_to(ROOT)}:{node.lineno}:{symbol}")
    return {
        "literal_occurrences": dict(occurrences),
        "moving_by_module": dict(sorted(by_module.items())),
        "moving_by_symbol": dict(sorted(by_symbol.items())),
        "test_files_by_module": {
            module: sorted(files) for module, files in sorted(files_by_module.items())
        },
        "test_file_tokens_by_module": {
            module: sum(
                len(ENCODER.encode((ROOT / path).read_text())) for path in files
            )
            for module, files in sorted(files_by_module.items())
        },
        "unresolved_import_or_dynamic_targets": dict(unknown),
        "examples": {
            module: sorted(values)[:8] for module, values in sorted(examples.items())
        },
        "test_files_scanned": len(test_files),
    }


def live_claims() -> tuple[list[dict], dict]:
    live = Path("/home/ITER/mcintos/.config/reckon/crew/live")
    results = []
    controls = []
    for path in sorted(live.glob("*.json")):
        data = json.loads(path.read_text())
        node = data.get("node") or {}
        paths = node.get("write_paths", []) if isinstance(node, dict) else []
        if any("srtj-graph-ops-split-design.html" in name for name in paths):
            controls.append(data.get("run_id"))
        matches = [
            name
            for name in paths
            if isinstance(name, str)
            and (name.endswith("/graph_ops.py") or name.endswith("/workers.py"))
        ]
        if matches:
            results.append(
                {
                    "run_id": data.get("run_id"),
                    "node": node.get("id") or node.get("name") or data.get("task"),
                    "phase": data.get("phase"),
                    "write_paths": matches,
                }
            )
    assert "r-20261023T130048028115-srtj-graph-ops-split-design" in controls
    return results, {
        "files_scanned": len(list(live.glob("*.json"))),
        "known_current_run_seen": True,
        "current_run_id": controls[0],
    }


def check_dag(inventories: dict) -> list[str]:
    imports = {
        name: item["imports_from_new_modules"]
        for inv in inventories.values()
        for name, item in inv["modules"].items()
    }
    imports["graph_ops"] = sorted(inventories["graph_ops.py"]["modules"])
    imports["workers"] = ["graph_ops", *sorted(inventories["workers.py"]["modules"])]
    visiting = set()
    visited = set()
    order = []

    def visit(name: str) -> None:
        if name in visiting:
            raise AssertionError(f"import cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in imports[name]:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in imports:
        visit(name)
    return order


def main() -> None:
    inventories = {source: source_inventory(source) for source in RANGES}
    order = check_dag(inventories)
    claims, live_control = live_claims()
    result = {
        "revision": subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                "imas_codex/standard_names/graph_ops.py",
                "imas_codex/standard_names/workers.py",
            ],
            cwd=ROOT,
            text=True,
        ).strip(),
        "tokenizer": "cl100k_base; original source ranges without repeated imports",
        "max_module_tokens": 60000,
        "inventories": inventories,
        "patch_targets": patch_targets(inventories),
        "import_dag_topological_order": order,
        "cross_module_calls_policy": "Pass an explicit service object; domain modules import only their foundation module, never one another. The composition root wires services after imports.",
        "live_claims": claims,
        "live_claims_positive_control": live_control,
    }
    assert all(
        item["source_tokens"] <= 60000
        for inv in inventories.values()
        for item in inv["modules"].values()
    )
    assert all(inv["symbols"] for inv in inventories.values())
    assert (
        result["patch_targets"]["moving_by_symbol"].get(
            "graph_foundation:GraphClient", 0
        )
        > 0
    ), "patch scanner missed a known graph client seam"
    fragment = OUT.with_suffix(".html").read_text()
    for inv in inventories.values():
        for module in inv["modules"]:
            assert f'id="srtj-graph-ops-split-design-{module}"' in fragment
    assert result["revision"] in fragment
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"graph={inventories['graph_ops.py']['source_tokens']} workers={inventories['workers.py']['source_tokens']}"
    )
    print(
        f"modules={sum(len(inv['modules']) for inv in inventories.values())} max={max(item['source_tokens'] for inv in inventories.values() for item in inv['modules'].values())}"
    )
    print(
        f"patch_occurrences={sum(result['patch_targets']['literal_occurrences'].values())} mapped={sum(result['patch_targets']['moving_by_module'].values())}"
    )
    print(f"live_claims={len(result['live_claims'])} cycle=none")


if __name__ == "__main__":
    main()
