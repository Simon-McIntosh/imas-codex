---
name: sn-focus-debug
description: How to use sn run --focus for iterative standard name pipeline debugging
---

# SN Focus Debug Workflow

## When to Use

- Investigating why a specific DD path produces a poor standard name
- Testing prompt improvements on specific paths before a full rotation
- Debugging pipeline behavior for quarantined or vocab-gap sources
- Verifying that a prompt change produces the expected output without running a full pipeline cycle

## How It Works

`--focus` is a scoped full-pipeline debug feature. It routes the specified DD paths through all 6 SN worker pools (GENERATE_NAME → REVIEW_NAME → REFINE_NAME → GENERATE_DOCS → REVIEW_DOCS → REFINE_DOCS) filtered by a UUID `scope_run_id`. It uses the identical production pipeline — not a parallel implementation.

Sequence:
1. Clear stale `scope_run_id`s for the targeted paths
2. Seed `StandardNameSource` (SNS) nodes for the paths
3. Stamp the `scope_run_id` on those SNS nodes
4. Force-reset any existing StandardName nodes for those paths
5. Enter the full 6-pool completion loop (scoped to the `scope_run_id`)

`--focus` is mutually exclusive with `--paths`.

## Usage

```bash
# Focus on a single path
uv run imas-codex sn run --focus equilibrium/time_slice/profiles_1d/psi -c 5

# Focus on multiple paths (space-separated)
uv run imas-codex sn run --focus equilibrium/time_slice/profiles_1d/psi equilibrium/time_slice/profiles_1d/q -c 5

# Focus on multiple paths (repeated flag)
uv run imas-codex sn run --focus equilibrium/time_slice/profiles_1d/psi --focus core_profiles/profiles_1d/electrons/temperature -c 5
```

## Closed-Loop Workflow

1. **Identify** a problematic path (from `sn status`, quarantined names, low benchmark scores, or a specific path you want to improve)
2. **Run focus**: `uv run imas-codex sn run --focus <path> -c 5`
3. **Inspect** the result: check the graph for the generated name, reviewer scores, validation status, and documentation
4. **If quality is poor**: modify the relevant prompt template in `imas_codex/llm/prompts/sn/`
5. **Re-run**: `uv run imas-codex sn run --focus <path> -c 5` — the previous result is automatically reset and re-processed
6. **Repeat** steps 3–5 until name quality is acceptable
7. **Verify**: run a broader rotation (without `--focus`) to confirm the prompt change does not regress other names

## Key Properties

- **Re-entrant**: Re-running `--focus` on the same path always resets and re-processes — safe to iterate without manual cleanup
- **Scoped**: The UUID `scope_run_id` isolates the focused run from the rest of the pipeline state
- **Production-identical**: Uses the same pool logic, prompts, and scoring as a full `sn run` — results are representative
- **Cost-bounded**: Use `-c/--cost-limit` (e.g. `-c 5`) to cap spend during iteration

## Relevant Files

| File | Purpose |
|------|---------|
| `imas_codex/llm/prompts/sn/` | Prompt templates for all 6 pools |
| `imas_codex/standard_names/pool_adapter.py` | Pool routing logic; `--focus` entry point |
| `imas_codex/llm/config/sn_review_criteria.yaml` | Reviewer scoring criteria |
