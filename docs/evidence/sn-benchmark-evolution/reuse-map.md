# DSv4 benchmark and graph-wide census reuse map

Source checkout: `/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/s9-20260824a/n-benchreusemap`  
Source commit: `5dbad32f60e9b1bc44ae1ade4301a10046e7b715`  
Scope: design evidence for a frozen-batch, advisory-only DeepSeek v4 benchmark followed by a zero-mutation graph-wide advisory census.

## Decision summary

The implementation should assemble a new read-only runner from existing pure or query-only components. It should not call the existing end-to-end review, audit, budget, or signed-manifest orchestrators: those paths persist review cycles, refresh embeddings, record cost nodes, or carry mutation authority. The benchmark should use the configured production three-seat hosted panel, preserve each seat's judgment, and add row-level median, spread, and contested status. The census should run only after the benchmark gates pass, score every live name without graph writes, and emit a frozen ranked artifact with exact identities and reasons.

The existing tree provides nearly all mechanisms below, but no single entry point currently satisfies both halves of that contract.

## Candidate map: panel-judged benchmark

| Candidate machinery | Citation | Verdict | Reuse decision |
|---|---|---|---|
| Production-context reviewer call | `imas_codex/standard_names/benchmark.py:449-468` | fit | Reuse `score_with_reviewer`: it supplies the production rubric, full graph context, scored examples, and lexical/semantic neighbours without writing review records. |
| Domain-stratified deterministic sampling | `imas_codex/standard_names/benchmark_roles.py:277-299` | fit | Reuse the round-robin domain sampler for a frozen, reproducible calibration batch; preserve the selected IDs and seed in the artifact. |
| Reviewer-discrimination corpus and metrics | `imas_codex/standard_names/benchmark_roles.py:950-1085` | partial | Reuse good/bad paired examples, AUC, bad-recall and good-pass calculations; expand beyond the three synthetic corruption families and retain exact row labels. |
| Independent reviewer-seat execution | `imas_codex/standard_names/benchmark.py:1002-1075` | fit | Reuse the parallel independent calls and per-reviewer records. Invoke DSv4 as the candidate and the hosted panel only as judges. |
| Production three-seat hosted panel configuration | `pyproject.toml:588-597` | fit | Resolve the production `sn-review.names` seats: Grok 4.5, GPT 5.6 Luna, and Claude Sonnet 5. Do not inherit the adjacent local-only profile. |
| Review quorum and escalation shape | `imas_codex/standard_names/review/pipeline.py:900-952` | partial | Reuse the two-blind-plus-escalator decision shape and configured disagreement threshold, but extract it into an artifact-only adapter because the full engine persists cycles. |
| Per-dimension disagreement detection | `imas_codex/standard_names/review/pipeline.py:2160-2214` | partial | Reuse score normalization and dimension parsing; replace the current two-seat threshold result with three-seat median, spread, and contested fields per benchmark row. |
| Review merge helper | `imas_codex/standard_names/review/pipeline.py:2217-2273` | partial | Reuse tier/comment normalization, but do not reuse its arithmetic mean as the panel result; retain all seat scores and compute the frozen-batch consensus statistics explicitly. |
| Provenance, dataset hash, and atomic report save | `imas_codex/standard_names/benchmark.py:160-248` | fit | Reuse source/DD/ISN provenance, deterministic dataset identity, and temp-file plus `fsync`/replace publication for the benchmark receipt. |
| Compose/reviewer/docs cost summary | `imas_codex/standard_names/benchmark.py:2171-2216` | fit | Reuse artifact-local cost aggregation and report authorized ceiling, actual candidate spend, judging spend, and total separately. |
| CLI physics-judge calibration guard | `imas_codex/cli/sn.py:3460-3486` | partial | Reuse the fail-closed trust gate and human-scoring fallback, but apply it to every panel-derived benchmark gate rather than a single optional physics judge. |
| Legacy single-model benchmark loop | `scripts/run_benchmark.py:410-516` | fail | Do not reuse: the same selected model composes and reviews examples, so it is self-judging rather than independently panel-adjudicated. |
| Legacy evaluation-set inventory | `tests/standard_names/eval_sets/benchmark.json:1-31` | fail | Do not treat it as the frozen batch: it declares 50 positive targets but contains only 20 populated positives and 30 placeholders. Its populated rows can seed, not define, the expanded balanced set. |

Benchmark candidate count: **13**.

## Candidate map: advisory graph-wide census

| Candidate machinery | Citation | Verdict | Reuse decision |
|---|---|---|---|
| Full live-name catalog projection | `imas_codex/standard_names/review/pipeline.py:246-310` | partial | Extract the query/projection into a public read-only loader; do not call the surrounding engine, which later writes review state. |
| Production-context DSv4 scoring call | `imas_codex/standard_names/benchmark.py:530-604` | fit | Reuse candidate construction and neighbour context for every live identity, with DSv4 as the advisory scorer and no review persistence callback. |
| Deterministic lexical lint | `imas_codex/standard_names/review/audits.py:248-349` | fit | Reuse grammar, prose, token and convention findings as explicit census flags and ranking reasons. |
| In-memory link-integrity audit | `imas_codex/standard_names/review/audits.py:357-422` | fit | Reuse source/link consistency checks over the frozen catalog projection; attach findings to exact identities. |
| Near-duplicate detector | `imas_codex/standard_names/review/audits.py:472-618` | partial | Reuse blocking and similarity candidates, but freeze neighbour inputs and record thresholds so one census artifact remains reproducible. |
| Corpus-wide attachment authority audit | `imas_codex/standard_names/attachment_audit.py:776-860` | fit | Reuse the explicitly read-only, all-name attachment verdicts as authority-boundary flags. Do not call reconciliation. |
| Identity-returning graph integrity ratchets | `tests/graph/test_sn_integrity_ratchets.py:32-80` | fit | Reuse the four read-only queries for multiple live targets, stale bindings, unsourced names, and explicit-axis generic parents; preserve offending IDs rather than counts alone. |
| Read-only graph status denominators | `imas_codex/cli/sn.py:3719-3803` | partial | Reuse stage, validation, source and documentation counts as census denominators and reconciliation checks, not as row-level evidence. |
| Exhaustion and collision priority fields | `imas_codex/schemas/standard_name.yaml:1270-1303` | fit | Rank parked identities using `refine_attempts`, `refine_stop_reason`, and `refine_collision_name`; distinguish vocabulary gaps, rescore candidates, and identities that should stay parked. |
| Exact cohort resolver and count fence | `imas_codex/standard_names/campaign_pricing.py:236-314` | fit | Reuse live selector resolution, missing/overlap checks, and before/after `StandardName` count assertion to freeze the complete census population. |
| Frozen campaign-manifest structure | `imas_codex/standard_names/config/campaign_manifest.schema.json:39-125` | partial | Reuse totals, per-domain/per-predicate counts, deterministic ordering and evidence fields, but emit every ranked row rather than a sample and omit approval-to-spend semantics. |
| Non-mutation proof pattern | `docs/evidence/sn-graph-wide-integrity/sprint-closing-census.md:98-119` | fit | Reuse before/after canonical payload hash, `StandardNameChange`, produced-name-edge and receipt counters; require exact equality and include the measurements in the census artifact. |
| All-audits orchestrator | `imas_codex/standard_names/review/audits.py:626-655` | fail | Do not call it unchanged: it invokes embedding preflight, whose implementation writes refreshed embeddings and hashes. Call only selected pure audits. |
| Graph-backed cost recorder | `imas_codex/standard_names/graph_ops.py:13781-13907` | fail | Do not reuse in the census: it creates graph cost records and updates per-name fields. Accumulate provider-returned costs in the artifact only. |
| Signed repair-manifest apply path | `imas_codex/standard_names/signed_manifest.py:4374-4445` | fail | Do not reuse: even preview/apply machinery belongs to governed repair authority. The census emits flags and exact ranked identities only, never an executable acceptance or mutation instruction. |

Census candidate count: **15**.

## Recommended assembly

### Frozen DSv4 benchmark

1. Resolve DSv4 explicitly as the candidate and resolve the production three-seat hosted review profile explicitly. Record concrete model identifiers, endpoint class, effort, pricing snapshot, source commit, DD/ISN versions, and authorized cost ceiling.
2. Build one domain-balanced frozen batch from accepted real names plus independently labelled corruptions. Seed examples should include source-path bindings such as `equilibrium/time_slice/profiles_1d/psi` → `poloidal_magnetic_flux` (`tests/standard_names/eval_sets/benchmark.json:33-40`), `core_profiles/profiles_1d/electrons/temperature` → `electron_temperature` (`tests/standard_names/eval_sets/benchmark.json:82-88`), and `magnetics/ip/data` → `plasma_current` (`tests/standard_names/eval_sets/benchmark.json:114-120`); do not fill the missing inventory with unlabelled placeholders.
3. Ask DSv4 for one advisory result per row. Send the same immutable row payload independently to all three hosted judges. Persist each seat's raw dimension scores and comments, then derive row-level median, spread, and contested status. Any escalator sees both blind critiques but cannot write graph state.
4. Compute the plan's grammar, description-identity, semantic-equivalence, convention, duplicate/collision, panel-consensus, regression, and cost gates within this single frozen batch. Keep qualified or failed gates visible; a low-cost local result is not a promotion decision.
5. Publish one atomic JSON result plus a concise Markdown receipt. Hash the ordered input rows and ordered result rows separately so a later census cannot silently redefine the benchmark population.

### Zero-mutation graph-wide census

1. Start only after every benchmark gate passes. Freeze every live `StandardName` identity and the relevant source, domain, stage, review, refinement-stop, attachment, and neighbour fields into a deterministic local payload.
2. Run the selected pure audits and DSv4 scoring over that payload. Give exhausted and collision-parked identities explicit ranking features; label each output as one of `rescore`, `vocabulary-gap`, or `stay-parked`, with reasons and source-path evidence.
3. Emit every identity, not a sample, in a stable ranked artifact containing advisory score, flags, confidence, matched predicates, domain, source paths, and evidence. The artifact is review input only and has no approval signature or apply command.
4. Measure and record canonical graph payload hash, total `StandardName` count, `StandardNameChange` count, produced-name edge count, cost-node count, and run-receipt count immediately before and after. Any delta invalidates the zero-mutation claim.
5. Account for DSv4 and any hosted adjudication from returned usage in the local artifact. Do not reserve or settle through the graph-backed budget manager during the census.

## Required implementation seams

- Add a read-only catalog loader rather than reusing the persistence-owning review engine.
- Add a non-persisting three-seat adjudication adapter that reports raw seat judgments plus median, spread, and contested status.
- Add a complete row schema for ranked census outputs; campaign samples and exact-name lists are insufficient because neither carries advisory scores and flags for every identity.
- Add an artifact-only cost ledger and a pre/post non-mutation guard.
- Keep all output advisory: no graph edits, no review-cycle nodes, no embedding refresh, no cost nodes, no signed repair manifests, and no automatic acceptance.

## Quantitative completion

This map contains **28 candidates**: **13 benchmark candidates** and **15 census candidates**. Every candidate row has one source `file:line` citation and one reuse verdict.
