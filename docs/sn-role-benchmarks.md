# SN pipeline seat benchmarks — GPT-5.6 role evaluation

Measured evidence for the per-seat model decision. Each pipeline seat exercises
a different capability, so a compose-benchmark result must **not** be
extrapolated across seats — every seat gets its own measured row, its own
production prompt, its own real fixtures, and its own production
reasoning-effort. Reports live under
`~/.local/share/imas-codex/benchmarks/sn_rolebench_<role>_<ts>.json`.

Candidate slate: incumbent + the GPT-5.6 tiers the plan lists for each seat
(`gpt-5.6-luna`, `-terra`, `-sol`). Held-out judge where a seat needs one:
`openrouter/anthropic/claude-opus-4.8`. Deterministic sample seed: 0 (CLI
default). Measured spend so far (four completed roles): **$23.59** — refine
$3.55, breaker-names $2.08, breaker-docs $6.57, docs $11.39. Classifier and the
added sn-compose re-bench are pending; the $40 ceiling applies to the aggregate
and no sample-size reductions have been forced.

## Summary — verdict per seat

| Seat | Incumbent | Candidates | Verdict | One line |
|---|---|---|---|---|
| refine | gpt-5.5 | luna, terra | **HOLD** | Incumbent best defect-resolution (0.874); Terra ties quality at only ~11% less cost; Luna worse quality at half cost. |
| breaker-names | gpt-5.5 | luna, terra | **SWITCH (luna)** | Both candidates more independent from the blind pair and higher flip-quality than incumbent at ~1/3–1/4 cost; luna best flip-quality (0.71). Thin flip sample. |
| breaker-docs | gpt-5.5 | luna, terra | **HOLD** | Incumbent both most independent (rho 0.176) and most correct on overrides (0.286); candidates flip 2–3× more but are almost always wrong (vfq 0.07/0.16). |
| docs | sonnet-4.6 | luna, terra, sol | **SWITCH (luna)** | Rubric parity+ (0.813 vs 0.806), zero banned prose vs incumbent's 15%, 23% cheaper; terra/sol gain marginal rubric at 45–158% more cost. |
| classifier | (none) | luna | PENDING (running) | |
| sn-compose (added) | (see note) | dsv4-local, luna, terra, gpt-5.5, kimi-k3 | PENDING | Local DSv4 vs GPT-5.6; free-at-margin economics. |

## Methods & caveats

- **Fixtures are real graph history, read-only.** refine replays real
  reviewer-critique → refinement cases from `REFINED_FROM` history; the breakers
  re-score a stratified sample already scored by the blind pair
  (`qwen3.7-max` + `minimax-m3`); docs regenerates documentation for a
  stratified accepted-name sample; classifier runs the 209-path domain gold set.
- **Two refine bench-fidelity fixes landed before the refine numbers were
  trusted** (imas-codex `0a05c173`, `0baa04ee`). The first refine run produced a
  hollow all-zeros report because the bench under-fed the incumbent:
  1. It built the refine prompt without the compose grammar vocabulary, so the
     system prompt's grammar-reference include rendered an empty token map and
     candidates invented unregistered tokens.
  2. It fed no scored examples, so the model had no in-prompt signal for the
     entry `kind` field and set it to a semantic guess (`standard_name`,
     `physical`, …), failing validation on ~54/60 cases.
  Both are now loaded exactly as production's `process_refine_name_batch` does
  (`build_compose_context` + domain-scoped `load_compose_examples`). A tripwire
  now raises rather than saving a report when every model judges zero cases, and
  a `valid_refine_rate` metric is recorded per model.
- **Latent production fragility (flagged, not fixed here).** At the time of the
  first run, `RefinedName.kind` was a free `str` with no schema enum, and the
  refine prompt never enumerated the valid entry kinds — production conveyed the
  valid `kind` values *only* through the scored-examples include
  (`kind=<scalar|vector|metadata>`). Had example loading ever failed in
  production, refine would have degraded exactly as the bench did. The source
  was since hardened (imas-codex `6ef0ab2c`: `kind` is now a `Literal` enum
  mirroring the ISN catalog Kind, and the prompts enumerate the kinds). A
  hardening followup is queued by the lead.
- **Cost is OpenRouter `response_cost` summed from each report.** A refine case
  that fails structured-output validation is a single non-retried call whose
  cost is not added to the model's row (the failed compose raises before the
  cost is tallied), so a low-`valid_refine_rate` model's reported cost slightly
  undercounts its true spend.

---

## refine — defect resolution + collateral change

Report: `sn_rolebench_refine_20260717T101049.json` · sample 20 (seed 0) ·
judge opus-4.8 · reasoning-effort high (production `sn-refine` setting).
Incumbent: **gpt-5.5**.

| model | n | defect_resolution ↑ | collateral_change ↓ | valid_refine_rate ↑ | cost | cost/item |
|---|---|---|---|---|---|---|
| **gpt-5.5** (incumbent) | 19 | **0.874** | 0.147 | 0.95 | $1.54 | $0.081 |
| gpt-5.6-luna | 17 | 0.765 | 0.218 | 0.85 | $0.64 | $0.038 |
| gpt-5.6-terra | 19 | 0.837 | 0.145 | 0.95 | $1.36 | $0.072 |

**Verdict: HOLD (gpt-5.5).** The incumbent has the highest defect-resolution
(0.874) with low collateral (0.147). Luna is clearly worse on quality — lower
resolution (0.765), higher collateral (0.218), lower valid-refine rate (0.85) —
even at half the per-item cost. Terra matches the incumbent on collateral
(0.145) and valid rate (0.95) with slightly lower resolution (0.837) at only
~11% lower cost; the marginal saving does not justify switching off the
best-resolving seat. Spend: $3.55.

---

## breaker-names — independence + verdict-flip quality

Report: `sn_rolebench_breaker-names_20260717T103917.json` · sample 30 (seed 0) ·
axis names · reasoning-effort high · blind pair `qwen3.7-max` + `minimax-m3`.
Incumbent: **gpt-5.5**. A breaker adds value by being *decorrelated* from the
pair (low `independence_rho`) while its overrides are *correct*
(high `verdict_flip_quality` = fraction of pair-disagreements where the
breaker matched the final outcome).

| model | n | independence_rho ↓ | verdict_flip_quality ↑ | n_flips | cost | cost/item |
|---|---|---|---|---|---|---|
| gpt-5.5 (incumbent) | 30 | 0.319 | 0.545 | 11 | $1.29 | $0.0429 |
| **gpt-5.6-luna** | 30 | 0.252 | **0.714** | 7 | $0.35 | $0.0118 |
| gpt-5.6-terra | 30 | 0.197 | 0.667 | 6 | $0.44 | $0.0145 |

**Verdict: SWITCH (gpt-5.6-luna).** Both GPT-5.6 tiers are *more* independent
from the blind pair than the incumbent (rho 0.25 / 0.20 vs 0.32) and get their
overrides right more often (flip-quality 0.71 / 0.67 vs 0.55), at roughly a
third to a quarter of the incumbent's per-item cost. Luna has the best
flip-quality; Terra is marginally more independent but overrides slightly less
accurately and costs a touch more. **Caveat:** verdict-flip quality rests on a
small number of disagreements (7 flips for luna), so it is the noisier axis; the
independence and cost advantages are on the full n=30 and are robust. Spend:
$2.08.

## breaker-docs — independence + verdict-flip quality (docs axis)

Report: `sn_rolebench_breaker-docs_20260717T124539.json` · sample 30 (seed 0) ·
axis docs · reasoning-effort high · blind pair `qwen3.7-max` + `minimax-m3`.
Incumbent: **gpt-5.5**.

| model | n | independence_rho ↓ | verdict_flip_quality ↑ | n_flips | cost | cost/item |
|---|---|---|---|---|---|---|
| **gpt-5.5** (incumbent) | 30 | **0.176** | **0.286** | 7 | $3.92 | $0.1307 |
| gpt-5.6-luna | 30 | 0.247 | 0.071 | 14 | $1.04 | $0.0348 |
| gpt-5.6-terra | 30 | 0.227 | 0.158 | 19 | $1.61 | $0.0535 |

**Verdict: HOLD (gpt-5.5).** The opposite of the names axis: here the incumbent
is *both* the most independent from the blind pair (lowest rho, 0.176) *and*
the most correct when it overrides (flip-quality 0.286). The GPT-5.6 tiers
disagree with the pair far more often (14 / 19 flips vs 7) but are almost always
wrong when they do (flip-quality 0.07 / 0.16) — that is added noise, not added
signal, and a breaker whose overrides are usually wrong degrades the docs
review. They are ~2.5–3.7× cheaper, but a docs breaker's job is correct
independent judgement, which the incumbent alone delivers. (Absolute
flip-quality is low across the board on the docs axis; the incumbent is best in
relative terms on both axes that matter.) Spend: $6.57.

## docs — rubric score + banned-prose audit

Report: `sn_rolebench_docs_20260717T134200.json` · sample 20 (seed 0) ·
judge opus-4.8 · production docs rubric + grep audit for banned prose (typical
values / estimator recipes / procedural padding). Incumbent:
**claude-sonnet-4.6**.

| model | n | rubric_score ↑ | banned_prose_rate ↓ | banned_findings | cost | cost/item |
|---|---|---|---|---|---|---|
| claude-sonnet-4.6 (incumbent) | 20 | 0.806 | 0.15 | 3 | $2.33 | $0.117 |
| **gpt-5.6-luna** | 20 | 0.813 | 0.00 | 0 | $1.79 | $0.090 |
| gpt-5.6-terra | 20 | 0.820 | 0.00 | 0 | $2.63 | $0.131 |
| gpt-5.6-sol | 20 | 0.836 | 0.00 | 0 | $4.64 | $0.232 |

**Verdict: SWITCH (gpt-5.6-luna).** The incumbent scores the *lowest* rubric
(0.806) and is the *only* model that emitted banned prose (15% of docs, 3
findings) — a real quality miss given the accepted-name docs policy. All three
GPT-5.6 tiers clear the banned-prose gate cleanly (0%) and score at or above the
incumbent on the rubric. Luna is the value pick: rubric parity-plus (0.813),
zero banned prose, and 23% cheaper than the incumbent. Terra and Sol score
marginally higher on the rubric (0.820, 0.836) but cost 45% and 158% more than
Luna for a ≤0.024 rubric gain — not worth it. Spend: $11.39.

## classifier — PENDING (report incoming — run by the lead)

Domain classifier exact-match accuracy over the 209-path gold set; candidate
`gpt-5.6-luna`. Table and verdict fold in when the report lands.

## sn-compose (added run) — PENDING

Compose re-bench comparing the locally-served `deepseek-v4-flash` against the
GPT-5.6 tiers (luna, terra) and the `gpt-5.5` baseline on the full 54-path
reference set, standard physics-judge scoring, plus `kimi-k3` at a reduced
sample (15 paths) to measure day-one per-name cost. Rationale: local DSv4 makes
compose inference free at the margin, so a quality near-tie versus Luna flips
the compose-seat economics. Table and verdict fold in when the report lands.
