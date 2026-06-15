# SN Composer Physics-Correctness Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure each candidate composer model's *physical correctness* on a fixed hard-case test set, judged by a gold-set-calibrated LLM judge, to decide keep / switch / tiered for the SN composer.

**Architecture:** A new `physics_judge.py` module scores composed names against their DD documentation on a measurement-principle rubric (structured LLM call). A calibration gate validates the judge against a human gold set before its scores count. The existing `benchmark.py` compose loop supplies the candidates for all 6 models; a thin CLI flag wires it together and renders a comparison table.

**Tech Stack:** Python 3.12, pydantic (structured output), `call_llm_structured`/`render_prompt` (imas_codex LLM layer), pytest, the existing `imas_codex/standard_names/benchmark.py`.

**Spec:** `docs/superpowers/specs/2026-06-15-sn-composer-physics-benchmark-design.md`

---

## File Structure

- **Create** `research/physics_bench_paths.json` — fixed ~30 DD source paths (the test set), version-controlled.
- **Create** `imas_codex/standard_names/physics_judge.py` — `PhysicsVerdict` model, `judge_name_physics()` (one name → verdict), `run_calibration_gate()` (gold set vs judge), `score_physics_batch()`.
- **Create** `imas_codex/llm/prompts/sn/judge_physics_correctness_system.md` + `judge_physics_correctness_user.md` — the measurement-principle rubric prompt.
- **Create** `research/physics_bench_gold.json` — human gold labels (authored during execution, Task 7).
- **Modify** `imas_codex/standard_names/benchmark.py` — add physics fields to `ModelResult` + a `physics_judge_model`/`gold_set_path` to `BenchmarkConfig`; invoke physics scoring after compose; add report aggregation.
- **Modify** `imas_codex/cli/sn.py` — add `--physics` flag (+ `--gold-set`, `--physics-judge-model`) to `sn bench`; render the physics comparison table.
- **Create** `tests/standard_names/test_physics_judge.py` — unit tests (stub judge, calibration gate, batch scoring) — all mock-only, no live LLM.

---

## Task 1: Physics verdict model + test-set loader

**Files:**
- Create: `imas_codex/standard_names/physics_judge.py`
- Create: `research/physics_bench_paths.json`
- Test: `tests/standard_names/test_physics_judge.py`

- [ ] **Step 1: Write the test set file**

Create `research/physics_bench_paths.json` — a list of objects `{path, category}`. `category` ∈ {`measurement_principle`, `locus_ordering`, `specificity_consistency`, `source_qualifier`, `control`}. Hard cases first, controls last:

```json
[
  {"path": "magnetics/rogowski_coil/current", "category": "measurement_principle"},
  {"path": "magnetics/ip", "category": "measurement_principle"},
  {"path": "iron_core/segment/b_field", "category": "locus_ordering"},
  {"path": "pf_active/coil/b_field_max", "category": "locus_ordering"},
  {"path": "pf_active/coil/b_field_max_timed", "category": "locus_ordering"},
  {"path": "ferritic/permeability_table/b_field", "category": "locus_ordering"},
  {"path": "equilibrium/time_slice/boundary_separatrix/x_point/r", "category": "specificity_consistency"},
  {"path": "summary/boundary/x_point_main/r", "category": "specificity_consistency"},
  {"path": "equilibrium/time_slice/constraints/x_point/position/r", "category": "specificity_consistency"},
  {"path": "breeding_blanket/module/cooling/time_slice/pressure_inlet", "category": "source_qualifier"},
  {"path": "breeding_blanket/module/time_slice/wall_flux_max", "category": "source_qualifier"},
  {"path": "core_profiles/profiles_1d/electrons/density", "category": "control"},
  {"path": "core_profiles/global_quantities/ip", "category": "control"},
  {"path": "equilibrium/time_slice/global_quantities/magnetic_axis/r", "category": "control"},
  {"path": "core_profiles/profiles_1d/e_field/toroidal", "category": "control"}
]
```

(Author may extend toward ~30; the structure is the contract. Keep ≥1 control per
non-control category for judge calibration balance.)

- [ ] **Step 2: Write the failing test for the verdict model + loader**

```python
# tests/standard_names/test_physics_judge.py
from __future__ import annotations
import json
from pathlib import Path
from imas_codex.standard_names.physics_judge import PhysicsVerdict, load_bench_paths


def test_physics_verdict_fields():
    v = PhysicsVerdict(
        name="current_of_rogowski_coil",
        faithful=False,
        base_correct=True,
        measurement_principle_correct=False,
        qualifiers_preserved=True,
        no_over_qualification=True,
        valid=True,
        reason="Rogowski coil measures enclosed current via induced voltage, not coil current.",
    )
    assert v.faithful is False
    assert v.measurement_principle_correct is False


def test_load_bench_paths():
    paths = load_bench_paths()
    assert any(p["path"] == "magnetics/rogowski_coil/current" for p in paths)
    assert all("category" in p for p in paths)
```

- [ ] **Step 3: Run it to verify failure**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: imas_codex.standard_names.physics_judge`.

- [ ] **Step 4: Implement the model + loader**

```python
# imas_codex/standard_names/physics_judge.py
"""Physics-correctness judging for SN composer benchmarking.

A measurement-principle rubric judge: scores a composed standard name against
its DD documentation for PHYSICAL faithfulness, not naming-form conventions.
Gated by a human gold set (run_calibration_gate) before its verdicts are
trusted — the judge must catch the known-hard errors (e.g. rogowski) or the
benchmark falls back to human scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

_BENCH_PATHS = Path(__file__).resolve().parents[2] / "research" / "physics_bench_paths.json"


class PhysicsVerdict(BaseModel):
    """Per-name physical-correctness verdict from the rubric judge."""

    name: str = Field(description="The composed standard name being judged.")
    faithful: bool = Field(description="Overall: is the name physically faithful to the source quantity?")
    base_correct: bool = Field(description="Is the physical_base the correct quantity?")
    measurement_principle_correct: bool = Field(
        description="Does the name reflect what is ACTUALLY measured (not the hardware)?"
    )
    qualifiers_preserved: bool = Field(
        description="Are loci/extrema/species/medium the source states present?"
    )
    no_over_qualification: bool = Field(
        description="Free of qualifiers the canonical quantity already implies?"
    )
    valid: bool = Field(description="Grammatically valid + canonical order (round-trips)?")
    reason: str = Field(description="One-sentence justification, citing the physics where relevant.")


def load_bench_paths(path: Path | None = None) -> list[dict]:
    """Load the fixed physics-benchmark test-set path list."""
    return json.loads((path or _BENCH_PATHS).read_text())
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
uv run --no-sync ruff check --fix imas_codex/standard_names/physics_judge.py tests/standard_names/test_physics_judge.py
uv run --no-sync ruff format imas_codex/standard_names/physics_judge.py tests/standard_names/test_physics_judge.py
git add imas_codex/standard_names/physics_judge.py tests/standard_names/test_physics_judge.py research/physics_bench_paths.json
git commit -m "feat(sn): physics-judge verdict model + benchmark test-set loader"
```

---

## Task 2: Physics-correctness judge prompt

**Files:**
- Create: `imas_codex/llm/prompts/sn/judge_physics_correctness_system.md`
- Create: `imas_codex/llm/prompts/sn/judge_physics_correctness_user.md`
- Test: `tests/standard_names/test_physics_judge.py` (add render test)

- [ ] **Step 1: Write the system prompt (the rubric)**

Create `judge_physics_correctness_system.md`. Static, cacheable. Content:

```markdown
You are a fusion-plasma physicist auditing whether a proposed IMAS standard
name is PHYSICALLY FAITHFUL to the quantity its source actually measures or
represents. You are NOT checking naming style or grammar conventions — you are
checking physics.

For each candidate you receive the source path, its unit, and its
documentation. Judge on these dimensions and return the structured verdict:

- **base_correct** — is the core physical quantity right? (e.g. a magnetic flux
  named as a current is wrong.)
- **measurement_principle_correct** — does the name reflect what is ACTUALLY
  measured, not the hardware? CRITICAL: a Rogowski coil measures the current
  ENCLOSED by the loop via induced voltage — a name implying "current of/in the
  coil" is WRONG. Diagnostic names must describe the physical observable, not
  the instrument's internal state.
- **qualifiers_preserved** — does the name keep physically-essential qualifiers
  the source states: locus (at_poloidal_field_coil), extremum (maximum),
  species (electron/neutron), medium (coolant)? Dropping a stated qualifier
  that changes WHAT is measured is unfaithful.
- **no_over_qualification** — does it avoid adding a component/qualifier the
  canonical quantity already implies (e.g. plasma current is inherently
  toroidal; `toroidal_plasma_current` over-qualifies)?
- **valid** — grammatically valid and in canonical order.

`faithful` is true ONLY if base_correct AND measurement_principle_correct AND
qualifiers_preserved AND no_over_qualification AND valid. Give a one-sentence
`reason` citing the physics for any failure.
```

- [ ] **Step 2: Write the user prompt (per-candidate, dynamic)**

Create `judge_physics_correctness_user.md`:

```markdown
Judge this candidate standard name for physical faithfulness.

- **Proposed name:** {{ name }}
- **Source path:** {{ path }}
- **Unit:** {{ unit | default('dimensionless', true) }}
- **Source documentation:** {{ documentation }}

Return the structured PhysicsVerdict.
```

- [ ] **Step 3: Write the failing render test**

```python
# add to tests/standard_names/test_physics_judge.py
from imas_codex.llm.prompt_loader import render_prompt


def test_judge_prompts_render():
    sysp = render_prompt("sn/judge_physics_correctness_system", {})
    assert "Rogowski" in sysp and "measurement_principle_correct" in sysp
    userp = render_prompt(
        "sn/judge_physics_correctness_user",
        {"name": "current_of_rogowski_coil", "path": "magnetics/rogowski_coil/current",
         "unit": "A", "documentation": "Measured plasma current"},
    )
    assert "current_of_rogowski_coil" in userp and "magnetics/rogowski_coil/current" in userp
```

- [ ] **Step 4: Run to verify it passes** (prompts now exist)

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py::test_judge_prompts_render -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add imas_codex/llm/prompts/sn/judge_physics_correctness_system.md imas_codex/llm/prompts/sn/judge_physics_correctness_user.md tests/standard_names/test_physics_judge.py
git commit -m "feat(sn): physics-correctness judge rubric prompt"
```

---

## Task 3: Judge call + batch scoring (with a stub-injectable LLM)

**Files:**
- Modify: `imas_codex/standard_names/physics_judge.py`
- Test: `tests/standard_names/test_physics_judge.py`

- [ ] **Step 1: Write the failing test (stub judge fn injected)**

```python
# add to tests/standard_names/test_physics_judge.py
import anyio
from imas_codex.standard_names.physics_judge import score_physics_batch


def _stub_judge(name, path, unit, documentation):
    # deterministic: rogowski is unfaithful, everything else faithful
    bad = "rogowski" in name
    return PhysicsVerdict(
        name=name, faithful=not bad, base_correct=True,
        measurement_principle_correct=not bad, qualifiers_preserved=True,
        no_over_qualification=True, valid=True,
        reason="enclosed current" if bad else "ok",
    )


def test_score_physics_batch():
    cands = [
        {"name": "current_of_rogowski_coil", "path": "magnetics/rogowski_coil/current", "unit": "A", "documentation": "d"},
        {"name": "electron_density", "path": "core_profiles/profiles_1d/electrons/density", "unit": "m^-3", "documentation": "d"},
    ]
    verdicts = anyio.run(score_physics_batch, cands, _stub_judge)
    assert verdicts[0].faithful is False
    assert verdicts[1].faithful is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py::test_score_physics_batch -q`
Expected: FAIL — `score_physics_batch` not defined.

- [ ] **Step 3: Implement the judge call + batch scorer**

Add to `physics_judge.py`:

```python
from collections.abc import Awaitable, Callable

from imas_codex.discovery.base.llm import acall_llm_structured
from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.settings import get_reasoning_effort


async def judge_name_physics(
    name: str, path: str, unit: str | None, documentation: str, *, model: str
) -> PhysicsVerdict:
    """Score one composed name for physical faithfulness via the rubric judge."""
    system = render_prompt("sn/judge_physics_correctness_system", {})
    user = render_prompt(
        "sn/judge_physics_correctness_user",
        {"name": name, "path": path, "unit": unit or "dimensionless", "documentation": documentation or ""},
    )
    result, _cost, _tokens = await acall_llm_structured(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_model=PhysicsVerdict,
        service="standard-names",
        reasoning_effort=get_reasoning_effort("sn-review"),
    )
    return result


JudgeFn = Callable[[str, str, str | None, str], Awaitable[PhysicsVerdict] | PhysicsVerdict]


async def score_physics_batch(candidates: list[dict], judge_fn: JudgeFn) -> list[PhysicsVerdict]:
    """Score a list of {name, path, unit, documentation} dicts.

    judge_fn may be sync (tests) or async (production judge_name_physics
    closed over a model). Runs sequentially — benchmark sets are small (~180).
    """
    out: list[PhysicsVerdict] = []
    for c in candidates:
        res = judge_fn(c["name"], c["path"], c.get("unit"), c.get("documentation", ""))
        if hasattr(res, "__await__"):
            res = await res
        out.append(res)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
uv run --no-sync ruff check --fix imas_codex/standard_names/physics_judge.py
git add imas_codex/standard_names/physics_judge.py tests/standard_names/test_physics_judge.py
git commit -m "feat(sn): physics judge call + batch scorer"
```

---

## Task 4: Calibration gate

**Files:**
- Modify: `imas_codex/standard_names/physics_judge.py`
- Test: `tests/standard_names/test_physics_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/standard_names/test_physics_judge.py
from imas_codex.standard_names.physics_judge import run_calibration_gate


def test_calibration_gate_trusts_when_hardcases_caught():
    gold = [
        {"name": "current_of_rogowski_coil", "path": "magnetics/rogowski_coil/current",
         "unit": "A", "documentation": "d", "faithful": False, "hard_case": True},
        {"name": "electron_density", "path": "core_profiles/profiles_1d/electrons/density",
         "unit": "m^-3", "documentation": "d", "faithful": True, "hard_case": False},
    ]
    rep = anyio.run(run_calibration_gate, gold, _stub_judge)
    assert rep["trusted"] is True
    assert rep["hardcase_errors_caught"] == 1
    assert rep["hardcase_errors_total"] == 1


def test_calibration_gate_distrusts_when_hardcase_missed():
    # judge that says everything faithful → misses the rogowski hard case
    def blind(name, path, unit, documentation):
        return PhysicsVerdict(name=name, faithful=True, base_correct=True,
            measurement_principle_correct=True, qualifiers_preserved=True,
            no_over_qualification=True, valid=True, reason="ok")
    gold = [{"name": "current_of_rogowski_coil", "path": "p", "unit": "A",
             "documentation": "d", "faithful": False, "hard_case": True}]
    rep = anyio.run(run_calibration_gate, gold, blind)
    assert rep["trusted"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py -k calibration -q`
Expected: FAIL — `run_calibration_gate` not defined.

- [ ] **Step 3: Implement the gate**

Add to `physics_judge.py`:

```python
async def run_calibration_gate(
    gold: list[dict], judge_fn: JudgeFn, *, min_overall_agreement: float = 0.90
) -> dict:
    """Validate the judge against human gold labels before trusting it.

    Each gold entry: {name, path, unit, documentation, faithful: bool,
    hard_case: bool}. The judge is TRUSTED iff it catches every hard-case error
    (gold faithful=False, hard_case=True) AND overall agreement on
    `faithful` >= min_overall_agreement. Returns a report dict.
    """
    verdicts = await score_physics_batch(gold, judge_fn)
    by_name = {v.name: v for v in verdicts}
    agree = 0
    hc_total = hc_caught = 0
    misses: list[str] = []
    for g in gold:
        v = by_name.get(g["name"])
        if v is None:
            continue
        if v.faithful == g["faithful"]:
            agree += 1
        if g.get("hard_case") and g["faithful"] is False:
            hc_total += 1
            if v.faithful is False:
                hc_caught += 1
            else:
                misses.append(g["name"])
    n = len(gold)
    overall = agree / n if n else 0.0
    trusted = (hc_total == hc_caught) and (overall >= min_overall_agreement)
    return {
        "trusted": trusted,
        "overall_agreement": round(overall, 3),
        "hardcase_errors_total": hc_total,
        "hardcase_errors_caught": hc_caught,
        "hardcase_misses": misses,
        "n_gold": n,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_judge.py -k calibration -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
uv run --no-sync ruff check --fix imas_codex/standard_names/physics_judge.py
git add imas_codex/standard_names/physics_judge.py tests/standard_names/test_physics_judge.py
git commit -m "feat(sn): physics-judge calibration gate (gold-set gated trust)"
```

---

## Task 5: Benchmark integration — physics fields + orchestration

**Files:**
- Modify: `imas_codex/standard_names/benchmark.py` (`ModelResult` ~74, `BenchmarkConfig` ~37, `run_benchmark` ~889)
- Test: `tests/standard_names/test_physics_bench_integration.py` (create)

- [ ] **Step 1: Add physics fields to `ModelResult` and config**

In `benchmark.py`, add to `ModelResult` (after the reference fields, ~100):

```python
    # Physics-correctness judging (Task 5)
    physics_verdicts: list[dict] = field(default_factory=list)  # serialized PhysicsVerdict
    physics_faithful_count: int = 0
    physics_total: int = 0
    physics_rate: float = 0.0
    physics_hardcase_faithful: int = 0
    physics_hardcase_total: int = 0
    physics_hardcase_rate: float = 0.0
```

Add to `BenchmarkConfig` (~44):

```python
    physics_judge: bool = False
    physics_judge_model: str | None = None  # default opus-4.8 at call site
    gold_set_path: str | None = None
```

- [ ] **Step 2: Write the failing aggregation test**

```python
# tests/standard_names/test_physics_bench_integration.py
from __future__ import annotations
from imas_codex.standard_names.benchmark import ModelResult, apply_physics_metrics
from imas_codex.standard_names.physics_judge import PhysicsVerdict


def test_apply_physics_metrics():
    r = ModelResult(model="m")
    verdicts = [
        PhysicsVerdict(name="a", faithful=True, base_correct=True, measurement_principle_correct=True,
                       qualifiers_preserved=True, no_over_qualification=True, valid=True, reason="ok"),
        PhysicsVerdict(name="b", faithful=False, base_correct=True, measurement_principle_correct=False,
                       qualifiers_preserved=True, no_over_qualification=True, valid=True, reason="bad"),
    ]
    hard = {"b"}  # b is a hard-case name
    apply_physics_metrics(r, verdicts, hard)
    assert r.physics_total == 2 and r.physics_faithful_count == 1
    assert r.physics_rate == 0.5
    assert r.physics_hardcase_total == 1 and r.physics_hardcase_faithful == 0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_bench_integration.py -q`
Expected: FAIL — `apply_physics_metrics` not defined.

- [ ] **Step 4: Implement `apply_physics_metrics`**

Add to `benchmark.py` (near `_apply_review_metrics` ~1241):

```python
def apply_physics_metrics(result, verdicts, hardcase_names: set[str]) -> None:
    """Aggregate PhysicsVerdict list onto a ModelResult."""
    result.physics_verdicts = [v.model_dump() for v in verdicts]
    result.physics_total = len(verdicts)
    result.physics_faithful_count = sum(1 for v in verdicts if v.faithful)
    result.physics_rate = (
        result.physics_faithful_count / result.physics_total if result.physics_total else 0.0
    )
    hc = [v for v in verdicts if v.name in hardcase_names]
    result.physics_hardcase_total = len(hc)
    result.physics_hardcase_faithful = sum(1 for v in hc if v.faithful)
    result.physics_hardcase_rate = (
        result.physics_hardcase_faithful / result.physics_hardcase_total
        if result.physics_hardcase_total else 0.0
    )
```

- [ ] **Step 5: Wire physics scoring into `run_benchmark`**

In `run_benchmark` (~889), after the per-model compose+review results are built and when `config.physics_judge` is set, add (use the candidate's resolved name + the test-set path/unit/doc, and the hard-case set = paths whose category != "control"):

```python
    if config.physics_judge:
        from functools import partial
        from imas_codex.standard_names.physics_judge import (
            judge_name_physics, score_physics_batch, load_bench_paths,
        )
        from imas_codex.standard_names.benchmark import _resolve_name  # local helper
        judge_model = config.physics_judge_model or "openrouter/anthropic/claude-opus-4.8"
        bench_paths = {p["path"]: p for p in load_bench_paths()}
        hardcase = {p["path"] for p in bench_paths.values() if p["category"] != "control"}
        jfn = partial(judge_name_physics, model=judge_model)
        for result in results:  # results: list[ModelResult]
            cands = []
            hard_names = set()
            for cand in result.candidates:  # each compose candidate dict
                nm = _resolve_name(cand)
                pth = cand.get("path") or cand.get("source_id", "")
                cands.append({"name": nm, "path": pth,
                              "unit": cand.get("unit"), "documentation": cand.get("documentation", "")})
                if pth in hardcase:
                    hard_names.add(nm)
            verdicts = await score_physics_batch(cands, jfn)
            apply_physics_metrics(result, verdicts, hard_names)
```

(If `ModelResult` does not expose `candidates`, use the existing field that
holds the composed candidate dicts — confirm the attribute name in `_run_model`
and use it here. Do NOT invent a new field.)

- [ ] **Step 6: Run tests**

Run: `uv run --no-sync pytest tests/standard_names/test_physics_bench_integration.py tests/standard_names/test_physics_judge.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
uv run --no-sync ruff check --fix imas_codex/standard_names/benchmark.py tests/standard_names/test_physics_bench_integration.py
git add imas_codex/standard_names/benchmark.py tests/standard_names/test_physics_bench_integration.py
git commit -m "feat(sn): wire physics judging + metrics into run_benchmark"
```

---

## Task 6: CLI flag + comparison table

**Files:**
- Modify: `imas_codex/cli/sn.py` (`sn bench` ~1805–1868)
- Modify: `imas_codex/standard_names/benchmark.py` (`render_comparison_table` ~1631)

- [ ] **Step 1: Add `--physics` / `--gold-set` / `--physics-judge-model` options to `sn bench`**

In `cli/sn.py`, on the `@sn.command("bench")` decorator block, add:

```python
@click.option("--physics", is_flag=True, help="Run the physics-correctness judge (gold-set gated).")
@click.option("--gold-set", "gold_set", type=click.Path(exists=True), default=None,
              help="Path to human gold labels (research/physics_bench_gold.json).")
@click.option("--physics-judge-model", default=None,
              help="Judge model (default openrouter/anthropic/claude-opus-4.8).")
```

In `sn_bench(...)` signature add `physics: bool, gold_set: str | None, physics_judge_model: str | None`, and when building `BenchmarkConfig` set `physics_judge=physics, gold_set_path=gold_set, physics_judge_model=physics_judge_model`. When `physics` and `gold_set` are provided, run the calibration gate first and print its trust report; if not trusted, print a loud warning that scores are advisory / fall back to human.

```python
    if physics and gold_set:
        import json as _json
        from imas_codex.standard_names.physics_judge import run_calibration_gate, judge_name_physics
        from functools import partial
        gold = _json.loads(open(gold_set).read())
        jmodel = physics_judge_model or "openrouter/anthropic/claude-opus-4.8"
        cal = safe_asyncio_run(run_calibration_gate(gold, partial(judge_name_physics, model=jmodel)))
        console.print(f"[bold]Judge calibration:[/] trusted={cal['trusted']} "
                      f"overall={cal['overall_agreement']} hardcase={cal['hardcase_errors_caught']}/{cal['hardcase_errors_total']}")
        if not cal["trusted"]:
            console.print("[red]Judge FAILED calibration — physics scores are advisory; "
                          f"hardcase misses: {cal['hardcase_misses']}. Fall back to human scoring.[/]")
```

(Use the same `safe_asyncio_run` the command already uses for `run_benchmark`.)

- [ ] **Step 2: Render the physics comparison table**

In `benchmark.py` `render_comparison_table` (~1631), add a physics block when any result has `physics_total > 0`:

```python
    if any(r.physics_total for r in report.results):
        from rich.table import Table
        t = Table(title="Physics Correctness", show_header=True, header_style="bold")
        t.add_column("model"); t.add_column("physics %", justify="right")
        t.add_column("hard-case %", justify="right"); t.add_column("valid %", justify="right")
        t.add_column("$/name", justify="right")
        for r in sorted(report.results, key=lambda x: x.physics_rate, reverse=True):
            valid_pct = (r.valid_count / r.composed_count * 100) if getattr(r, "composed_count", 0) else 0.0
            cost_per = (r.compose_cost / r.composed_count) if getattr(r, "composed_count", 0) else 0.0
            t.add_row(r.model, f"{r.physics_rate*100:.0f}", f"{r.physics_hardcase_rate*100:.0f}",
                      f"{valid_pct:.0f}", f"${cost_per:.4f}")
        console.print(t)
```

(Confirm the actual attribute names for composed/valid counts + compose cost on
`ModelResult` before using — match what `_apply_*` and `validate_candidate`
populate; do not invent.)

- [ ] **Step 3: Verify CLI help + table wiring (smoke)**

Run: `uv run --no-sync imas-codex sn bench --help`
Expected: shows `--physics`, `--gold-set`, `--physics-judge-model`.

Run: `uv run --no-sync python -c "import imas_codex.cli.sn, imas_codex.standard_names.benchmark; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 4: Commit**

```bash
uv run --no-sync ruff check --fix imas_codex/cli/sn.py imas_codex/standard_names/benchmark.py
git add imas_codex/cli/sn.py imas_codex/standard_names/benchmark.py
git commit -m "feat(sn): sn bench --physics flag + physics-correctness comparison table"
```

---

## Task 7: Author the gold set + run the benchmark (human-in-the-loop)

**Files:**
- Create: `research/physics_bench_gold.json`

- [ ] **Step 1: Author the gold labels (AGENT — using physics judgment)**

The lead delegated gold-set authoring to the agent. For ~12–15 of the test-set
paths (all hard cases + a few controls), the agent records the
physically-correct verdict using domain judgment, citing the measurement
principle / DD doc. Pull each path's current DD doc + unit to ground the label:

```bash
uv run --no-sync python -c "
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.physics_judge import load_bench_paths
with GraphClient() as gc:
    for p in load_bench_paths():
        r = gc.query('MATCH (n:IMASNode {id:\$id}) RETURN n.unit AS u, n.description AS d', id=p['path'])
        if r: print(p['category'], p['path'], '|', r[0]['u'], '|', (r[0]['d'] or '')[:90])
"
```

Write `research/physics_bench_gold.json`: a list of
`{name, path, unit, documentation, faithful, hard_case}` where `name` is the
*expected-correct* (or a representative wrong) name and `faithful` is the human
verdict. Include the rogowski/iron_core/x_point hard cases with `faithful:false`
+ the correct rationale, and clean controls with `faithful:true`.

- [ ] **Step 2: Run the benchmark**

```bash
IMAS_CODEX_SN_DEDUP_THRESHOLD=0.85 uv run --no-sync imas-codex sn bench \
  --physics --gold-set research/physics_bench_gold.json \
  --models "hosted_vllm/deepseek-v4-flash openrouter/anthropic/claude-opus-4.8 openrouter/openai/gpt-5.5 openrouter/google/gemini-3.1-pro-preview openrouter/anthropic/claude-sonnet-4.6 openrouter/moonshotai/kimi-k2.6"
```

(Use the `sn bench` model-list flag that exists; if the flag name differs,
match it — do not invent. DSv4-flash composes free/local; the 5 others via
OpenRouter.)

Expected: calibration report (trusted true/false), then the Physics Correctness
table ranking the 6 models, and a JSON report under `research/`.

- [ ] **Step 3: Commit the gold set + result artifact**

```bash
git add research/physics_bench_gold.json research/physics_bench-*.json
git commit -m "research(sn): physics-correctness benchmark gold set + results"
```

- [ ] **Step 4: Record the decision (composer + docs unification + failure-3 finding)**

Append to the spec's status:
- **Composer decision** — keep / switch / tiered, with per-model physics rates +
  judge trust level; update `[sn-compose].model` only if justified.
- **Docs unification** — the pre-local best for names was gpt-5.5; docs is
  currently sonnet-4.6 (chosen via a DD-grounded physics probe, so less
  blindsided, but LLM-judged so not immune). If a single model wins physics
  correctness on names AND is competitive for docs, unify `[sn-docs].model`
  with `[sn-compose].model`. Re-score the candidates' DOCS faithfulness with the
  SAME calibrated judge (reuse `judge_name_physics` adapted for docs prose) to
  confirm — do not unify on the old prose-definition probe alone.
- **Failure-3 treatability** — the calibration gate result IS the answer: if the
  calibrated judge caught the hard-case measurement-principle errors (rogowski
  etc.), record that failure 3 is treatable by adding the measurement-principle
  rubric dimension to the live name-review (`review_names.md` +
  `sn_review_criteria.yaml`) — queue that as the next robustness task. If the
  judge could NOT catch them, record that instrument/measurement-principle names
  require human physics review (not treatable with current models).

---

## Self-Review

**Spec coverage:**
- Test set (~30 hard-case paths) → Task 1 ✓
- 6-model compose matrix → Task 5 step 5 + Task 7 step 2 ✓
- Grammar-round-trip recorded as correctness signal → reuses `validate_candidate` (existing) + surfaced via `valid %` in Task 6 table; physics `valid` dim in rubric ✓
- Calibration gate (gold-set gated, fall back to human) → Task 4 + Task 6 step 1 ✓
- Physics judge rubric (5 dimensions) → Task 2 + Task 1 model ✓
- Report + comparison table + decision rule → Task 5 (metrics) + Task 6 (table) + Task 7 step 4 ✓

**Placeholder scan:** No TBD/TODO. Two "confirm the attribute name" notes (Task 5 step 5, Task 6 step 2) are explicit guards against inventing fields on the existing `ModelResult`/candidate dict — the implementer must read `_run_model`/`validate_candidate` to bind the real names. Flagged, not vague.

**Type consistency:** `PhysicsVerdict` fields used identically across Tasks 1, 3, 4, 5. `score_physics_batch(candidates, judge_fn)` and `run_calibration_gate(gold, judge_fn)` signatures consistent. `apply_physics_metrics(result, verdicts, hardcase_names)` consistent Task 5 ↔ test.

**Known integration risk:** Task 5 step 5 and Task 6 step 2 depend on the real attribute names for the composed-candidate list and per-model compose-cost/valid counts on `ModelResult`. The implementer MUST read `benchmark.py` (`_run_model`, `_apply_*`, `validate_candidate`) and bind to the existing fields rather than create new ones. This is the one place the plan cannot be fully literal without the live attribute names.
