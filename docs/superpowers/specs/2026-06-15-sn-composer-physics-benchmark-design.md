# SN Composer Physics-Correctness Benchmark — Design

**Date:** 2026-06-15
**Status:** approved (design); pending implementation plan
**Owner:** Simon McIntosh

## Problem

The standard-name composer (`hosted_vllm/deepseek-v4-flash`, "DSv4-flash") was
selected via a benchmark that judged names on grammar / semantic / convention /
completeness — the same rubric the review quorum uses. A recovery regeneration
surfaced three failures that rubric is **blind to**:

1. **Grammar-invalid name persisted** — `magnetic_field_of_iron_core_segment_magnitude`
   fails the ISN round-trip (`base=None`, unknown physical_base token) yet was
   stored as a candidate with `grammar_parse_fallback`.
2. **Specificity regression + cross-source inconsistency** —
   `major_radius_of_primary_x_point` (prior score 1.0) regenerated as **three
   divergent, weaker** names across sources: `major_radius_of_x_point` (lost
   *primary*), `radius_of_primary_x_point` (lost *major*),
   `major_radius_of_constraint_position`.
3. **Physically-wrong name accepted at 0.975** — `current_of_rogowski_coil`
   passed the full quorum (qwen3.7-max + minimax-m3 + gpt-5.5), but a Rogowski
   coil measures the current **enclosed by the loop** (via induced voltage),
   not current flowing in the coil.

Failures 1–2 implicate the composer; failure 3 implicates the composer **and**
the review (frontier reviewers passed a physics error). The open question:
**is DSv4-flash adequate for physically-correct naming, or is a stronger
composer worth the cost?** The current benchmark cannot answer this because it
never measured physical correctness — only a gameable proxy.

## Goal & success criterion

Produce a per-model **physics-correctness rate** (overall and on the known-hard
cases) plus cost/name, yielding a clear decision: **keep DSv4-flash** (free,
adequate) / **switch composer** (frontier materially better) / **tiered**
(DSv4 routine, escalate instrument/hard cases). The benchmark also quantifies
the residual error rate that downstream guards + review must still catch.

This benchmark settles the **composer-model question only**. Deterministic
guards (hard-quarantine invalid names; anchor-on-established-names for
consistency) and a physics-aware **review** check are a separate follow-on
design — informed by, but not part of, this benchmark.

## Model matrix (6)

| Model | id | Role |
|---|---|---|
| DeepSeek V4-flash | `hosted_vllm/deepseek-v4-flash` | incumbent, free (local GPU) |
| Claude Opus 4.8 | `openrouter/anthropic/claude-opus-4.8` | frontier |
| GPT-5.5 | `openrouter/openai/gpt-5.5` | frontier |
| Gemini 3.1 Pro | `openrouter/google/gemini-3.1-pro-preview` | frontier |
| Claude Sonnet 4.6 | `openrouter/anthropic/claude-sonnet-4.6` | mid-tier |
| Kimi K2.6 | `openrouter/moonshotai/kimi-k2.6` | mid-tier, vendor-diverse |

> **CONFIRM:** the only "2.7" Kimi on OpenRouter is `kimi-k2.7-code`
> (code-specialized — inappropriate for physics naming). `kimi-k2.6` is the
> latest *general* pinned model and is used here for reproducibility. Switch to
> the floating `~moonshotai/kimi-latest` alias only if a pinned general 2.7
> ships.

All composers run through the **same** rich-grounded compose path
(`generate_name_*` prompts, ISN grammar validation, dedup) so the comparison
isolates the model, not the prompt. Compose at the model's configured reasoning
effort (max for DSv4 per current config; high for the frontier/mid models).

## Test set (~30 fixed DD paths, reusable)

A fixed, version-controlled list (`research/physics_bench_paths.json`)
deliberately weighted toward the failure classes the recovery regen exposed,
plus clean canonicals as controls. Each entry carries its DD path, unit, and
rich `description` for grounding.

- **Measurement-principle traps:** `magnetics/rogowski_coil/current`,
  `magnetics/ip`, instrument-measured quantities where the name must reflect
  *what is measured*, not the hardware.
- **Locus / ordering:** `iron_core/segment/b_field`,
  `pf_active/coil/b_field_max`, `pf_active/coil/b_field_max_timed`,
  `ferritic/permeability_table/b_field`,
  `spectrometer_visible/.../b_field_modulus` (the over-merged b_field family).
- **Specificity / consistency:** `equilibrium/.../boundary_separatrix/x_point/r`,
  `summary/boundary/x_point_main/r`, `equilibrium/.../constraints/x_point/position/r`
  (same concept, multiple sources — tests cross-source consistency).
- **Source-stated qualifiers:** `breeding_blanket/.../pressure_inlet` (coolant),
  `breeding_blanket/.../wall_flux_max` (maximum neutron).
- **Clean controls:** `core_profiles/.../electrons/density`,
  `core_profiles/.../ip`, `equilibrium/.../magnetic_axis/r`,
  `core_profiles/.../e_field/toroidal`.

## Phases

### Phase 1 — Compose
Each of the 6 models composes the full test set through the shared pipeline.
Record per name: composed name, grammar-round-trip pass/fail (a name failing
the round-trip is recorded as a **hard correctness failure** — failure class 1),
segments, cost, tokens. Reuses `benchmark.py`'s model-iteration loop.

### Phase 2 — Calibration gate (trust mechanism)
The judge is an LLM and risks the *same* blind spot that passed rogowski, so it
must earn trust before its scores count:

1. A human-labeled **gold set** of ~12–15 names (the author labels them) spans
   hard + normal cases, each marked physically-faithful / unfaithful + reason
   (e.g. `current_of_rogowski_coil` → unfaithful: measures enclosed current).
   Stored at `research/physics_bench_gold.json`.
2. The opus-4.8 judge scores the gold-set names with the rubric below.
3. Compute agreement. **Gate:** the judge is trusted iff it (a) flags **every**
   hard-case error the human flagged, and (b) agrees on ≥ ~90% overall.
4. **Fail-open to human:** if the judge fails the gate, its scores are
   discarded and the benchmark is scored by the human directly — and that
   failure is itself a reported finding ("LLM judges unreliable for SN physics
   correctness"). Calibration stats are always reported so trust is explicit.

### Phase 3 — Physics-correctness judging (once calibrated)
The trusted opus-4.8 judge scores each composed name against its DD doc on a
**measurement-principle rubric**, returning faithful/unfaithful + reason per
name. Rubric dimensions:

- **Base correctness** — is the physical_base the right quantity?
- **Measurement principle** — does the name reflect *what is actually
  measured*? (the rogowski test)
- **Essential-qualifier preservation** — are loci / extrema / species /
  medium that the source states present? (coolant / neutron / maximum / major)
- **No over-qualification** — no component/qualifier the canonical quantity
  already implies (the `toroidal_plasma_current` test).
- **Validity** — grammatically valid + canonical order (round-trip passes).

### Phase 4 — Report & decision
`research/physics_bench-<stamp>.json` + a comparison table:

- Per model: physics-correctness rate (overall + hard-case subset),
  grammar-validity rate, cost/name, mean tokens.
- Head-to-head on each hard case (which models got rogowski / iron_core /
  x_point / b_field right).
- Calibration agreement stats (judge trust level).

**Decision rule:**
- DSv4-flash within noise of the frontier on physics correctness → **keep**
  (free wins).
- A frontier model materially better, esp. on hard cases → **cost-benefit a
  switch**, or **tiered** (DSv4 routine + escalate instrument/hard cases to the
  winner).
- Whichever composer wins, the report's residual error rate sizes the
  downstream guards + physics-aware review (separate design).

## Build / reuse

- **Extends** `imas_codex/standard_names/benchmark.py` — compose loop, model
  iteration, `BenchmarkReport` structure, `[sn-benchmark]` config
  (`compose-models` list → the 6-model matrix).
- **Adds:** the physics-correctness judge function + rubric prompt
  (`llm/prompts/sn/judge_physics_correctness_*.md`), the calibration gate, and
  the fixed hard-case test set + gold-set files under `research/`.
- Judge uses `call_llm_structured` with `service="standard-names"`; cost
  tracked via the usual LLMCost path.

## Cost

~180 composes (DSv4 free; 5 paid models × ~30 paths) + ~180 judge calls
(opus-4.8) + the ~15 gold-set judge calls. Estimated **$10–20 one-time**.

## Out of scope (follow-on design)

- Deterministic persist-time quarantine of grammar-invalid names.
- Cross-source consistency / anchor-on-established(catalog) names.
- Physics-aware addition to the live review quorum.
- The review tail-hang fix (`review_name` ghost-pending liveness bug).
