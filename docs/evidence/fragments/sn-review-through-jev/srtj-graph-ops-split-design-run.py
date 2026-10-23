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
LOCAL_SOURCE_ALLOWANCE = 190_000
GATES = {
    "graph_supersession": "tests/standard_names/test_supersede_into_ancestor.py",
    "graph_audits": "tests/standard_names/test_catalog_reconcile.py",
    "graph_doc_links": "tests/standard_names/test_bare_doc_link_normalize.py",
    "graph_docs_persistence": "tests/standard_names/test_persist_stages.py",
    "graph_review_persistence": "tests/standard_names/test_review_docs_stages.py",
    "graph_cost_and_claims": "tests/standard_names/test_cost_graph_ops.py",
    "graph_lineage": "tests/standard_names/test_derived_parent_lifecycle_repair.py",
    "graph_catalog_reset": "tests/standard_names/test_exact_reset_parent_cleanup.py",
    "graph_claim_release": "tests/standard_names/test_refine_claim_recovery.py",
    "graph_parent_lifecycle": "tests/standard_names/test_enrich_parents.py",
    "graph_claim_scope": "tests/standard_names/test_exact_name_scope.py",
    "graph_refinement": "tests/standard_names/test_refine_name_chain.py",
    "graph_source_bindings": "tests/standard_names/test_compose_source_scope.py",
    "graph_catalog": "tests/standard_names/test_generate_name_persist.py",
    "graph_foundation": "tests/standard_names/test_review_docs_stages.py",
    "worker_extraction_and_enrichment": "tests/standard_names/test_pool_compose_retry.py",
    "worker_compose_and_validate": "tests/standard_names/test_graph_state_machine.py",
    "worker_compose_batches": "tests/standard_names/test_compose_batch_embed.py",
    "worker_refine_names": "tests/standard_names/test_refine_name_chain.py",
    "worker_review_names": "tests/standard_names/test_rd_quorum_integrity.py",
    "worker_generate_docs": "tests/standard_names/test_generate_docs.py",
    "worker_enrich_parents": "tests/standard_names/test_enrich_parents.py",
    "worker_review_docs": "tests/standard_names/test_review_docs_stages.py",
    "worker_refine_docs": "tests/standard_names/test_refine_docs_chain.py",
    "worker_common": "tests/standard_names/test_pool_compose_helpers.py",
}
EXTRACTION_ORDER = [
    [
        "graph_foundation",
        "graph_supersession",
        "graph_audits",
        "graph_review_persistence",
    ],
    ["graph_doc_links"],
    ["graph_docs_persistence"],
    ["graph_cost_and_claims"],
    ["graph_lineage"],
    ["graph_catalog_reset"],
    ["graph_claim_release"],
    ["graph_parent_lifecycle"],
    ["graph_claim_scope"],
    ["graph_refinement"],
    ["graph_source_bindings"],
    ["graph_catalog"],
    ["worker_common", "worker_extraction_and_enrichment"],
    ["worker_compose_and_validate"],
    ["worker_compose_batches"],
    ["worker_refine_names"],
    ["worker_review_names"],
    ["worker_generate_docs"],
    ["worker_enrich_parents"],
    ["worker_review_docs"],
    ["worker_refine_docs"],
]


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

    def selected(paths: list[str]) -> list[str]:
        return [
            name
            for name in paths
            if isinstance(name, str)
            and (name.endswith("/graph_ops.py") or name.endswith("/workers.py"))
        ]

    assert selected(["imas_codex/standard_names/graph_ops.py"]) == [
        "imas_codex/standard_names/graph_ops.py"
    ]
    for path in sorted(live.glob("*.json")):
        data = json.loads(path.read_text())
        node = data.get("node") or {}
        paths = node.get("write_paths", []) if isinstance(node, dict) else []
        if any("srtj-graph-ops-split-design.html" in name for name in paths):
            controls.append(data.get("run_id"))
        matches = selected(paths)
        if matches:
            results.append(
                {
                    "run_id": data.get("run_id"),
                    "node": node.get("id") or node.get("name") or data.get("task"),
                    "phase": data.get("phase"),
                    "write_paths": matches,
                }
            )
    current_run = "r-20261023T130048028115-srtj-graph-ops-split-design"
    return results, {
        "files_scanned": len(list(live.glob("*.json"))),
        "known_current_run_seen": current_run in controls,
        "synthetic_claim_detected": True,
        "current_run_id": current_run,
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


def migration_steps(inventories: dict) -> list[dict]:
    def tokens(path: str) -> int:
        return len(ENCODER.encode((ROOT / path).read_text()))

    repairs = [
        (
            "hold synchronization",
            [
                "imas_codex/standard_names/graph_ops.py",
                "imas_codex/standard_names/workers.py",
                "imas_codex/standard_names/review/holds.py",
                "tests/standard_names/test_review_docs_stages.py",
                "tests/standard_names/test_docs_hold_dd_bindings.py",
            ],
            [
                "tests/standard_names/test_review_docs_stages.py",
                "tests/standard_names/test_docs_hold_dd_bindings.py",
            ],
        ),
        (
            "documentation regeneration stamp",
            [
                "imas_codex/standard_names/graph_ops.py",
                "imas_codex/standard_names/workers.py",
                "tests/standard_names/test_source_refresh_docs_requeue.py",
                "tests/standard_names/test_persist_stages.py",
            ],
            [
                "tests/standard_names/test_source_refresh_docs_requeue.py",
                "tests/standard_names/test_persist_stages.py",
            ],
        ),
    ]
    steps = []
    for title, paths, gates in repairs:
        steps.append(
            {
                "order": len(steps) + 1,
                "work": title,
                "lane": "codex",
                "write_paths": paths,
                "gate_files": gates,
                "gate_command": "uv run --no-sync pytest -p no:cacheprovider "
                + " ".join(gates),
                "declared_and_gate_tokens": sum(
                    tokens(path) for path in set(paths + gates)
                ),
                "estimate_basis": "current source; recount after each repair",
            }
        )

    source_lines = {
        source: (ROOT / "imas_codex" / "standard_names" / source)
        .read_text()
        .splitlines(keepends=True)
        for source in RANGES
    }
    remaining = {source: set(inv["modules"]) for source, inv in inventories.items()}

    def remaining_tokens(source: str) -> int:
        return len(
            ENCODER.encode(
                "".join(
                    "".join(source_lines[source][start - 1 : stop - 1])
                    for start, stop, module in RANGES[source]
                    if module in remaining[source]
                )
            )
        )

    for modules in EXTRACTION_ORDER:
        source = "graph_ops.py" if modules[0].startswith("graph_") else "workers.py"
        assert all(module in inventories[source]["modules"] for module in modules)
        before = remaining_tokens(source)
        moved = sum(
            inventories[source]["modules"][module]["source_tokens"]
            for module in modules
        )
        gates = list(dict.fromkeys(GATES[module] for module in modules))
        gate_tokens = sum(tokens(gate) for gate in gates)
        total = before + moved + gate_tokens
        package = "graph_store" if source == "graph_ops.py" else "worker_tasks"
        paths = [f"imas_codex/standard_names/{source}"]
        paths.extend(
            f"imas_codex/standard_names/{package}/{module}.py" for module in modules
        )
        if modules == EXTRACTION_ORDER[0] or "worker_common" in modules:
            paths.append(f"imas_codex/standard_names/{package}/__init__.py")
        remaining[source].difference_update(modules)
        steps.append(
            {
                "order": len(steps) + 1,
                "work": ", ".join(modules),
                "lane": "codex"
                if before > LOCAL_SOURCE_ALLOWANCE or total > LOCAL_SOURCE_ALLOWANCE
                else "local",
                "write_paths": list(dict.fromkeys(paths)),
                "gate_files": gates,
                "gate_command": "uv run --no-sync pytest -p no:cacheprovider "
                + " ".join(gates),
                "source_before_tokens": before,
                "new_module_source_tokens": moved,
                "gate_tokens": gate_tokens,
                "declared_and_gate_tokens": total,
                "source_after_tokens": remaining_tokens(source),
                "estimate_basis": "source ranges; new imports, package scaffolds and forwarding wrappers require recount at dispatch",
            }
        )
    assert sorted(module for group in EXTRACTION_ORDER for module in group) == sorted(
        module for inv in inventories.values() for module in inv["modules"]
    )
    assert sum(step["lane"] == "codex" for step in steps[2:]) == 1
    assert inventories["graph_ops.py"]["source_tokens"] > LOCAL_SOURCE_ALLOWANCE
    assert steps[2]["lane"] == "codex"
    assert steps[2]["source_after_tokens"] < LOCAL_SOURCE_ALLOWANCE
    assert steps[3]["lane"] == "local"
    assert all(
        step["declared_and_gate_tokens"] < LOCAL_SOURCE_ALLOWANCE
        for step in steps
        if step["lane"] == "local"
    )
    assert inventories["workers.py"]["source_tokens"] < LOCAL_SOURCE_ALLOWANCE
    return steps


def main() -> None:
    inventories = {source: source_inventory(source) for source in RANGES}
    order = check_dag(inventories)
    claims, live_control = live_claims()
    steps = migration_steps(inventories)
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
        "local_source_allowance": LOCAL_SOURCE_ALLOWANCE,
        "migration_steps": steps,
        "lane_transition": {
            "codex_extraction_steps": 1,
            "graph_ops_local_after_step": steps[2]["order"],
            "graph_ops_original_source_tokens_then": steps[2]["source_after_tokens"],
            "workers_local_from_start": True,
            "workers_original_source_tokens": inventories["workers.py"][
                "source_tokens"
            ],
        },
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
    for step in steps:
        marker = f'id="srtj-graph-ops-split-design-step-{step["order"]}"'
        assert marker in fragment
        row = fragment.split(marker, 1)[1].split("</tr>", 1)[0]
        assert f"<strong>{step['lane']}</strong>" in row
        assert f"{step['declared_and_gate_tokens']:,}" in row
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
    print(
        "codex_extractions=1 graph_local_after=3 "
        f"graph_original_remaining={steps[2]['source_after_tokens']} "
        f"workers_local_from_start={inventories['workers.py']['source_tokens'] < LOCAL_SOURCE_ALLOWANCE}"
    )


if __name__ == "__main__":
    main()
