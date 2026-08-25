# Benchmark corpus recovery assessment

Assessment date: 2026-08-25  
Source revision: `495b697e0ef7`  
Scope: recovery of the legacy Standard Names benchmark population and derivation of five advisory-review gates. No provider call, graph write, or benchmark run was made.

## Outcome

The legacy JSON is **not recoverable as a frozen population by filling its blank cells**. The 31 rows are allocation stubs rather than partially specified examples, and the 20 populated positives have drifted: only 9 still agree with the live graph on accepted state, validation, exact DD-path binding, and unit. The reuse-map rejection is therefore fatal for that file's identity as the gold corpus, but not fatal for the benchmark goal.

A replacement corpus is practical. The graph currently contains **1,894 accepted, valid, documented, unit-bearing, source-bound names across 18 current physics domains**. Extracting and freezing those rows is read-only and incurs **USD 0 provider cost**. Existing code already builds a deterministic good/bad paired corpus from that population, although one fail-closed repair is required before it can be frozen: the name-corruption helper can return an unchanged two-token name, and the advisory runner requires globally unique row names.

The five gate values can be derived rather than picked. The defensible rule is: **the candidate must be no worse than the weakest production hosted seat on the same independently labelled, frozen rows and the same two-pass protocol**. Critical-defect recall, clean false-positive rate, Brier calibration error, repeated-label agreement, and leave-one-seat-out quorum agreement then all get numeric bounds from the hosted panel's observed envelope. The live graph already proves the computation is possible, but its opportunistic history is not itself a release calibration set: labels are post-hoc lifecycle outcomes, repeats are unbalanced, and the third seat appears preferentially on disputed rows.

The cheapest runnable route is a new committed 20-good/20-bad paired population, two hosted-panel calibration passes, then one DSv4-plus-panel benchmark. Graph extraction and the local candidate are free. Using historical `review_name` charges, the **36 hosted batch calls** are estimated at **USD 1.402 at historical means** or **USD 2.588 using the per-seat p95 call costs**, comfortably inside the existing USD 25 ceiling. This is cheaper and more defensible than preserving the stale 51-positive layout.

| Question | Answer | Affirmative? |
|---|---|---:|
| What do the 31 slots lack? | They lack every row-defining field: DD path, expected identity, unit, and provenance; several carry only stale generic rationale. | Yes — determined exactly |
| Can graph data fill them, and at what cost? | The benchmark population can be rebuilt from 1,894 eligible graph names at USD 0 extraction cost; literal one-for-one filling under the old domain taxonomy is not defensible. | **Yes, by rebuild** |
| Why was reuse inadmissible; is that fixable? | Incomplete allocation stubs, self-judging legacy execution, stale taxonomy, and drift in populated bindings. Fatal for the file as gold authority; fixable for the benchmark by replacing it. | **Yes, benchmark goal is fixable** |
| Can all five thresholds be derived? | Yes, from a two-pass leave-one-seat-out hosted-panel envelope on a frozen independently labelled corpus. Existing graph history supplies provisional values only. | **Yes, after controlled calibration** |
| What is the cheapest runnable path? | Freeze 20 good + 20 bad rows, calibrate twice, benchmark once; estimated hosted spend USD 1.402 mean / USD 2.588 p95. | **Yes — path and cost identified** |

## 1. What the 31 unpopulated slots actually lack

The file metadata says 20 positives, 20 negatives, and 30 stubs against a target of 50 positives (`tests/standard_names/eval_sets/benchmark.json:5-13`). The actual array contains **51 positives: 20 populated and 31 stubs**. The metadata is off by one before any semantic audit begins.

Every stub has:

- `dd_path: "TODO"`;
- `expected_name: "TODO"`;
- `expected_unit: "TODO"`;
- `source: "todo"`;
- only a legacy `physics_domain` allocation and, usually, the word `Stub.` as rationale.

The first waves row says a missing object vocabulary must be resolved (`benchmark.json:195-200`), but the following four waves rows contain no candidate-specific information (`benchmark.json:202-232`). The same empty shape repeats through fast particles, turbulence, plasma-wall interactions, gyrokinetics, transport, and edge plasma physics (`benchmark.json:234-440`). These are not unapproved labels awaiting a reviewer; there is no quantity, path, unit, or proposed identity to review.

Exact slot distribution, computed from the committed JSON:

| Legacy domain allocation | Total positive rows | Empty rows |
|---|---:|---:|
| waves | 5 | 5 |
| fast_particles | 5 | 5 |
| turbulence | 5 | 5 |
| plasma_wall_interactions | 5 | 5 |
| gyrokinetics | 5 | 5 |
| transport | 5 | 3 |
| edge_plasma_physics | 5 | 3 |
| **Total** | **35** | **31** |

The populated part covers only equilibrium, core-plasma, magnetic-diagnostic, transport, and edge allocations. It therefore cannot establish the intended ten-domain balance even before checking whether those labels still describe the live graph.

## 2. Recoverability from the live graph

### 2.1 The 20 populated positives are seeds, not a current gold set

The following read-only audit joined each populated JSON row to `StandardName.id`, normalized the stored `dd:` prefix on `source_paths`, and checked stage, validation, unit, and exact binding:

```cypher
UNWIND $items AS item
OPTIONAL MATCH (sn:StandardName {id: item.expected_name})
WITH item, sn,
     CASE WHEN sn IS NULL THEN false
          ELSE any(p IN coalesce(sn.source_paths, [])
                   WHERE replace(p, 'dd:', '') = item.dd_path)
     END AS path_bound
RETURN item.dd_path, item.expected_name,
       sn.name_stage, sn.validation_status, sn.unit, sn.physics_domain,
       path_bound, sn.unit = item.expected_unit AS unit_matches
```

Live result:

| Check over 20 populated rows | Passing rows |
|---|---:|
| Expected identity exists | 20 |
| Current stage is `accepted` | 13 |
| Validation is `valid` | 20 |
| Expected DD path is still bound to that identity | 10 |
| Unit matches | 20 |
| Legacy domain equals current graph domain | 7 |
| **Accepted + valid + exact binding + matching unit** | **9** |

Representative drift is material rather than cosmetic:

- `equilibrium/time_slice/global_quantities/volume` still names `plasma_volume`, but that identity is now `superseded` and the path is no longer bound to it.
- `equilibrium/time_slice/profiles_1d/psi` still has the right unit and identity, but the identity is `reviewed` and the expected path is not bound.
- `core_profiles/profiles_1d/electrons/density` points at a `drafted` identity and is not bound.
- `core_profiles/global_quantities/beta_tor` points at a `reviewed` identity, is not bound, and its current domain is `general`, not the legacy `transport` allocation.

This live result independently strengthens the reuse-map's rejection. Even the supposedly populated rows cannot all be carried forward without re-adjudication.

### 2.2 The graph can supply a replacement population for free

Eligibility was measured with the same read-only shape already used by the repository's discrimination-corpus loader (`imas_codex/standard_names/benchmark_roles.py:950-987`): accepted and valid identity, real description and documentation, unit, and at least one source path.

```cypher
MATCH (sn:StandardName {name_stage: 'accepted', validation_status: 'valid'})
WHERE sn.description IS NOT NULL
  AND sn.documentation IS NOT NULL
  AND size(sn.documentation) > 40
  AND sn.unit IS NOT NULL
  AND size(coalesce(sn.source_paths, [])) > 0
RETURN sn.physics_domain AS domain, count(*) AS eligible
ORDER BY domain
```

Live result: **1,894 eligible names in 18 current domains**.

| Current domain | Eligible | Current domain | Eligible |
|---|---:|---|---:|
| auxiliary_heating | 127 | magnetohydrodynamics | 69 |
| divertor_physics | 53 | mechanical_measurement_diagnostics | 14 |
| edge_plasma_physics | 204 | particle_measurement_diagnostics | 63 |
| electromagnetic_wave_diagnostics | 51 | plant_systems | 117 |
| equilibrium | 139 | plasma_wall_interactions | 73 |
| general | 18 | radiation_measurement_diagnostics | 119 |
| machine_operations | 2 | structural_components | 38 |
| magnetic_field_diagnostics | 28 | transport | 653 |
| magnetic_field_systems | 86 | turbulence | 40 |

The three apparent zeroes under legacy labels are taxonomy drift, not an empty graph. Source-prefix traversal found **82 accepted names under `waves/`**, **46 under `gyrokinetics_local/`**, **130 under `distributions/`**, **24 under `distribution_sources/`**, and **14 under `nbi/`**; their current domain classifications are auxiliary heating, wave diagnostics, turbulence, transport, equilibrium, and related current categories. Literal balance against `waves`, `fast_particles`, and `gyrokinetics` would preserve a dead taxonomy.

The deterministic round-robin sampler is already implemented (`benchmark_roles.py:277-299`), and the good/bad loader reads this graph projection without writes (`benchmark_roles.py:950-1051`). Population extraction therefore costs **USD 0** and needs no LLM call.

### 2.3 One code repair is required before freezing

The existing bad-name generator uses the last two tokens for a `vacuous` corruption (`benchmark_roles.py:927-947`). For a two-token good name, that returns the original unchanged. A live dry construction at seed `20260825`, sample 20 produced **20 good and 20 bad rows across 18 domains, but `plasma_resistance` was unchanged in its bad twin**. Good and bad rows can also collide because semantic-mismatch rows deliberately borrow a foreign accepted name, while the advisory runner rejects duplicate resolved identities (`imas_codex/standard_names/advisory_benchmark.py:152-180`).

Before freezing, the builder must fail closed unless every bad twin differs from its good row and all resolved row identities are globally unique. This is a small implementation repair, but silently filtering the collision after sampling would change the domain population and its hash, so it belongs in the corpus builder rather than in an execution note.

## 3. Why the reuse map rejected the inventory

The reuse map made two separate judgments that should not be collapsed:

1. The legacy loop is inadmissible because one selected model composes and reviews the same examples, so it is self-judging (`docs/evidence/sn-benchmark-evolution/reuse-map.md:27-29`; implementation flow at `scripts/run_benchmark.py:410-516`).
2. The JSON is inadmissible as the frozen batch because it advertises a target rather than providing a complete labelled population (`reuse-map.md:29`).

The new implementation fixes the first defect. It invokes the candidate and the three-seat panel independently on deep copies of the same frozen rows (`advisory_benchmark.py:169-200`), retains raw seat judgments, and computes median, spread, and contested state without graph persistence (`advisory_panel.py:68-179`).

The new implementation does not fix the second defect. Its population is caller-supplied (`advisory_benchmark.py:133-180`), so passing the legacy JSON would faithfully hash and run an inadmissible input. The runner also contains no gate calculations or repeatability protocol; this is why the earlier refusal correctly produced no hashes and spent nothing.

Verdict:

- **Fatal as-is:** the legacy file cannot remain the gold-corpus identity. Completing 31 blank allocations would still leave 11 of 20 populated rows failing the combined live-authority check and would preserve obsolete domain labels.
- **Recoverable as evidence:** the 20 populated positives remain candidates; 9 are current-valid seeds. The 20 negatives remain candidate defect templates because their rejection reasons are explicit (`benchmark.json:443-585`), but the locked consensus rule still requires independent review before publication.
- **Fixable benchmark goal:** rebuild and freeze a replacement repo JSON from current graph authority, then run the already-separated candidate/panel path.

## 4. Deriving all five gates from hosted-quorum behaviour

### 4.1 Protocol

Use the production acceptance boundary of **0.85**, which is already the repository's minimum score for acceptance (`imas_codex/standard_names/defaults.py:12-14`); do not borrow the older role-benchmark value of 0.75 (`benchmark_roles.py:57-59`). Freeze paired known-good and known-bad rows, then run every hosted seat twice with identical row order and prompt provenance.

For seat `s`, derive:

| Measure | Seat measurement on frozen labels | Candidate gate derived from seats |
|---|---|---|
| Critical-defect recall | Fraction of known-bad rows with score `< 0.85` | Minimum recall observed across hosted seats |
| Clean-name false-positive rate | Fraction of known-good rows with score `< 0.85` | Maximum false-positive rate observed across hosted seats |
| Calibration | Brier mean `(score - label)^2`, with good = 1 and bad = 0 | Maximum Brier error observed across hosted seats |
| Repeatability | Agreement of each seat's pass/fail label between the two controlled passes | Minimum repeated-label agreement across hosted seats |
| Hosted-quorum agreement | For each seat, agreement with the other two seats where those two agree; candidate is compared with the full panel median | Minimum leave-one-seat-out agreement across hosted seats |

This locks a semantic policy—**non-inferior to the weakest production seat**—without selecting five convenient numbers after seeing DSv4. The calibration artifact supplies the numeric values before the candidate result is opened. Known-good/known-bad labels, both panel passes, per-seat values, derived bounds, and population/input hashes must be published together.

### 4.2 Existing graph history proves calculability but is not the frozen calibration

Review-property coverage was checked before aggregation: all **26,050** `StandardNameReview` nodes had `id`, `standard_name_id`, `review_axis`, `score`, and `reviewer_model`. The following query then selected accepted identities as provisional good rows and refined predecessors as provisional bad rows for the three current models:

```cypher
MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview {review_axis: 'name'})
WHERE r.reviewer_model IN $production_models
  AND (
    sn.name_stage = 'accepted'
    OR (sn.name_stage = 'superseded'
        AND EXISTS { MATCH (:StandardName)-[:REFINED_FROM]->(sn) })
  )
RETURN sn.id, sn.name_stage, r.reviewer_model, r.score,
       r.review_group_id, r.cycle_index, r.reviewed_at
```

The query returned **4,585 reviews**. Applying the protocol formulas gives this provisional hosted-seat envelope:

| Hosted seat | Good reviews | Bad reviews | Clean FPR | Critical recall | Brier | Repeat pairs | Repeat agreement | Leave-one-out rows | Quorum agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grok 4.5 | 1,544 | 509 | 0.0421 | 0.8114 | 0.1366 | 452 | 0.8695 | 327 | 0.7370 |
| GPT-5.6 Luna | 1,349 | 394 | 0.0586 | 0.8655 | 0.1209 | 70 | 0.7143 | 347 | 0.6945 |
| Claude Sonnet 5 | 360 | 429 | 0.2472 | 0.9441 | 0.2441 | 421 | 0.9240 | 287 | 0.8397 |

There were **479 exact same-name, same-group triples** containing all three current models. On this historical cohort, the mechanical envelope would be recall at least **0.8114**, false-positive rate at most **0.2472**, Brier at most **0.2441**, repeatability at least **0.7143**, and quorum agreement at least **0.6945**.

Those five values are **not recommended as committed gates**. They are evidence that every threshold can be computed from hosted behaviour, not evidence that this historical population is admissible:

- lifecycle labels were outcomes of the same review/refinement system, not independent gold labels;
- models saw different row counts and different historical prompt/grammar revisions;
- repeats are opportunistic rather than two controlled draws on one frozen population;
- the escalator is preferentially present on disputed rows, so the 479 triples are selection-biased;
- current lifecycle state can differ from the state at the time a review was written.

The controlled calibration must therefore replace the provisional values before any census authorization. No lead-chosen numeric threshold is necessary, but one calibrated measurement run is.

## 5. Cheapest path to a runnable benchmark

### Recommended minimal population

1. Repair the paired-corpus builder so every bad twin is changed and all 40 resolved names are unique.
2. Freeze **20 accepted good rows + 20 deterministic critical-defect twins**, selected with the existing fixed-seed stratifier. The present graph can span 18 current domains in 20 good rows.
3. Commit the ordered rows, labels, defect categories, DD/source bindings, units, descriptions, current domain values, graph extraction query/hash, DD version, ISN version, and seed as the locked repo JSON baseline.
4. Run the three hosted seats twice on the entire 40-row population. Derive the five numeric bounds from the seat envelope and publish both raw passes.
5. Run DSv4 twice for repeatability and once through the frozen-run report path, or extend the runner to publish the two candidate passes and gate matrix atomically. Compare the candidate to the already-derived bounds. Do not authorize the census unless all five pass and costs remain under the ceiling.

The current runner processes name reviews in batches of ten (`imas_codex/standard_names/benchmark.py:530-536`, `672-727`). A 40-row panel pass therefore makes four calls per seat. Two calibration passes plus one candidate-comparison panel make **12 calls per seat, 36 hosted calls total**; DSv4's local calls are free at point of use.

Historical live cost query:

```cypher
MATCH (c:LLMCost)
WHERE c.llm_model IN $production_models
  AND c.pool = 'review_name'
  AND c.llm_cost > 0
RETURN c.llm_model, count(*) AS calls,
       avg(c.llm_cost) AS mean_usd,
       percentileDisc(c.llm_cost, 0.95) AS p95_usd,
       max(c.llm_cost) AS max_usd
```

| Hosted seat | Historical calls | Mean USD/call | p95 USD/call |
|---|---:|---:|---:|
| Grok 4.5 | 2,539 | 0.039282 | 0.071350 |
| GPT-5.6 Luna | 2,314 | 0.005067 | 0.015453 |
| Claude Sonnet 5 | 1,090 | 0.072483 | 0.128830 |

Estimated hosted spend:

| Route | Rows | Hosted calls | Historical-mean estimate | Per-seat-p95 estimate |
|---|---:|---:|---:|---:|
| One hosted-panel pass | 40 | 12 | USD 0.467 | USD 0.863 |
| Two-pass calibration | 40 | 24 | USD 0.935 | USD 1.725 |
| Calibration + candidate-comparison panel | 40 | 36 | **USD 1.402** | **USD 2.588** |
| Preserve all 51 positive slots + 20 negatives | 71 | 72 | USD 2.804 | USD 5.175 |

The estimate assumes four or eight ten-row batches per seat, no provider retry, and historical prompt/cache behaviour. It is an estimate, not a reservation. Even the 71-row route is comfortably below USD 25, so the reason to choose 40 rows is population quality and balanced paired power, not budget pressure.

## Final disposition

- **Do not fill or rename the 31 `TODO` objects in place.** Replace the corpus as one versioned frozen population.
- **Do not promote the provisional five graph-history values to gates.** Use the same formulas on two controlled hosted-panel passes and lock the resulting values before revealing the candidate result.
- **Do reuse the graph loader, fixed-seed stratifier, panel adapter, and frozen runner.** They provide the read-only and provenance boundaries required by the plan.
- **Keep the census unauthorized** until the builder uniqueness repair, committed corpus, controlled calibration, repeatability output, and five derived gates all exist in one auditable benchmark receipt.
