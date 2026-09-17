<meta name="plan-slug" content="sn-lifecycle-integrity">
<meta name="plan-status" content="active">
<meta name="plan-evidence-for" content="sn-lifecycle-integrity">
<meta name="docs-project" content="imas-codex">

# Are these numbers still true?

Every count this plan asserts was re-measured against the live graph and against
the code at HEAD, so the sprint can be rescoped on current figures rather than on
figures taken on 2026-09-04.

Base revision `3f8c06fcef82b8a769ee5ec5d81ae83b4fe3c5c5`, graph re-queried
2026-09-17. Read-only: no node, property, relationship or index was written.

## Verdict rule

A verdict is a statement about the *claim*, not the exact digits.

- **current** — the figure still measures what the plan says it measures, at the
  same value. The claim stands as written.
- **drifted** — the figure has moved. The mechanism the row describes is still
  exactly what the plan says it is, so the row's *argument* survives; only its
  arithmetic is out of date.
- **stale** — the figure no longer describes the live state **and the claim it
  supported is no longer true**. A stale row is a row the sprint must rewrite or
  retire, because acting on it would repair something already repaired.

## Re-measured quantities

| # | Quantity, verbatim from the plan | Plan asserts | Re-measured | Verdict | Exact query or code read |
|---|---|---|---|---|---|
| 1 | "Every count here is from the live graph over **4852** `StandardName` nodes" (§1) | 4852 | 5130 | drifted | `MATCH (sn:StandardName) RETURN count(sn)` → 5130 (`logs/census.jsonl`, `sn_total`) |
| 2 | `name_stage` in use: "accepted 2356, superseded 1963, exhausted 278, drafted 136, reviewed 107, pending 12" (§1) | those six counts | accepted 2528, superseded 2163, exhausted 293, reviewed 113, drafted 22, pending 11 | drifted | `MATCH (sn:StandardName) RETURN coalesce(sn.name_stage,'<null>'), count(*)` (`logs/census.jsonl`, `census_name_stage`) |
| 3 | `docs_stage` in use: "accepted 2993, pending 1787, reviewed 31, drafted 26, exhausted 10, superseded 2", null 3 (§1) | those counts + null 3 | accepted 3343, pending 1734, reviewed 27, exhausted 12, drafted 10, superseded 2; null 2 | drifted | `... coalesce(sn.docs_stage,...)` (`logs/census.jsonl`, `census_docs_stage`) |
| 4 | `validation_status` in use: "valid 4522, quarantined 327", null 3 (§1) | valid 4522 / quarantined 327 | valid 4457, quarantined 660, pending 11; null 2 | drifted | `... coalesce(sn.validation_status,...)` (`logs/census.jsonl`, `census_validation_status`). 660 rows now sit behind the 327: a later quarantine campaign |
| 5 | "Terminal states are … `exhausted` when it hits the rotation cap" — `name_stage` null is **0** (§1) | null 0 | null 0 — six values returned, no `<null>` row | current | `census_name_stage`: the census returned exactly six keys, none of them `<null>` (`logs/census.jsonl`) |
| 6 | "No identity has reached `approved` yet, which is correct — no catalog pull request has ever merged" (§1) | `approved` absent | `approved` absent; `active` and `deprecated` also absent | current | `census_status` returned only `draft` and `superseded`; `census_name_stage` returned neither `approved` nor `contested` (`logs/census.jsonl`) |
| 7 | `status` in use: "draft 1996, superseded 323", null **2533 (52%)** (§1) | draft 1996 / superseded 323 / null 2533 | draft 2934, superseded 2196, **null 0** | stale | `... coalesce(sn.status,'<null>')` (`logs/census.jsonl`, `census_status`). The 52% hole is closed, so the null-status gate can no longer be demonstrated against a live null-status row |
| 8 | `origin` in use: "catalog_edit 2096, pipeline 1288, derived 246", null **1222 (25%)** (§1) | catalog_edit 2096 | catalog_edit **0**; pipeline 2803, derived 428, null 1899 | stale | `... coalesce(sn.origin,'<null>')` (`logs/census.jsonl`, `census_origin`) and `... WHERE sn.origin = 'catalog_edit'` → 0 (`logs/followup.jsonl`, `origin_writers_check`). The axis is retired by the `origin-axis-disposition` decision; the plan's own axis table still presents it as live |
| 9 | "`export.py:1242` writes `"status": "active"` as a literal" (§2) | a literal `"active"` at `:1242` | no `"status": "active"` literal anywhere in `export.py`; line 1488 is `"status": node["status"] if "status" in node else "active"`, a conditional fallback, and a null/absent-status gate refuses the whole cut at `:2545-2568` | stale | `grep -n '"status": "active"' imas_codex/standard_names/export.py` → 0 hits; fallback and gate read at `export.py:1488`, `:2545` |
| 10 | "the graph holds *zero* `active`" (§2) | active 0 | active 0 | current | `census_status` — the census returned exactly two keys, `draft` and `superseded` (`logs/census.jsonl`) |
| 11 | "**That decision is locked but the guard is not proven to exist.**" (§4) | guard unproven | the guard exists: a positive allow-list `record_catalog_import_provenance` (`imas_codex/standard_names/catalog_import.py:51-106`), behind tests that fail when it is removed | stale | code read of `catalog_import.py:51-106`; corroborated by the plan's own `s4` comment records, which name the closing commit |
| 12 | "59 of 283 failed sources carry no error text" (§5 row 1) | 59 of 283 | failed **111**, of which blank **59** | drifted | `MATCH (s:StandardNameSource) WHERE s.status='failed' RETURN count(s), sum(...last_error is null or '')` → 111 / 59 (`logs/census.jsonl`, `failed_sources_reasonless`) |
| 13 | "216 sources parked at `attempt_count = 5`" (§5 row 2) | 216 | **217** at exactly 5; 234 at ≥5, of which 70 produced no name and 41 carry no reason | drifted | `... WHERE s.attempt_count >= 5 RETURN count(s), size([(s)-[:PRODUCED_NAME]->()|1]) AS produced ...` (`logs/followup3.jsonl`, `attempt_cap_population`; `exactly_five` = 217 in `logs/followup4.jsonl`) |
| 14 | "54 leaf nodes labelled `quantity` that are `/weight` or `/reconstructed` pairs" (§5 row 3) | 54 | **14** | drifted | `MATCH (n:IMASNode) WHERE n.node_category='quantity' WITH n WHERE n.id CONTAINS '/weight' OR n.id CONTAINS '/reconstructed'` → 14 (`logs/followup.jsonl`, `quantity_contains_weight_or_reconstructed`), against 473 nodes carrying `node_category='fit_artifact'` (`logs/census.jsonl`, `fit_artifact_categories`) |
| 15 | "1945 superseded names with a null scalar; 513 with no successor on edge or scalar" (§5 row 4) | 1945 / 513 | **2145** null scalar of 2163 superseded; **517** with no successor by any route | drifted | `MATCH (sn:StandardName {name_stage:'superseded'}) RETURN count(sn), sum(CASE WHEN sn.superseded_by IS NULL THEN 1 ELSE 0 END)` → 2163 / 2145 (`logs/census.jsonl`, `superseded_scalar_rot`); `... size([(x)-[:REFINED_FROM]->(sn)|1]) ... WHERE sn.superseded_by IS NULL AND NOT (sn)-[:HAS_SUCCESSOR]->() AND inc = 0` → 517 (`logs/followup2.jsonl`, `superseded_no_successor_corrected`) |
| 16 | "the embedding gate writes a constant 0.30 with a null resolution method and no reviewer" (§5 row 5) | 0.30, null method | 52 rows at exactly 0.30 across 37 names; the resolution method is **no longer null** — `semantic_similarity_gate` on 40 rows | drifted | `MATCH (sn:StandardName)-[r:HAS_REVIEW]->(rv) WHERE rv.score = 0.30` → 52 rows / 37 names; `... WHERE rv.resolution_method IS NOT NULL RETURN rv.resolution_method, count(*)` → `semantic_similarity_gate` 40 (`logs/followup3.jsonl`) |
| 17 | "`StandardName.updated_at` is written at **3 of 218** property-write sites in `graph_ops.py` and read at none" (§5a) | 3 of 218 | **119** occurrences of the token in `graph_ops.py`; 3279 of 5130 names carry a stamp | stale | `grep -c updated_at imas_codex/standard_names/graph_ops.py` → 119 (the token count, not a stamp census); `MATCH (sn:StandardName) RETURN sum(CASE WHEN sn.updated_at IS NULL THEN 0 ELSE 1 END), count(sn)` → 3279 / 5130 (`logs/census.jsonl`, `updated_at_stamped`) |

### Verdict counts

| Verdict | Rows |
|---|---|
| current | 3 |
| drifted | 9 |
| stale | 5 |
| **total** | **17** |

3 + 9 + 5 = 17, and 17 is the number of rows in the table above.

### Unmeasured

One asserted quantity could not be measured inside the fence, and is recorded
here rather than estimated or given a verdict:

| Quantity, verbatim | Plan asserts | Why unmeasured |
|---|---|---|
| the halted candidate's `catalog.yml` sidecar carries "386 entries all reading `status: active`" (§2) | 386 | the halted candidate's export directory is not in this worktree and is not reachable from it. Measuring it needs a checkout that holds the halted candidate, which this node was not given |

### Instrument checks (why these zeros and non-zeros are believed)

Three readings came back as a zero or a near-zero, and a zero is a claim about
the instrument until the instrument is shown to see something known present.
Each was re-run with a detector proven on a live positive:

- The `/weight`-suffix detector returned **0** while the plan expects
  misclassified leaves. `ENDS WITH '/weight'` was the wrong predicate — the
  matching ids read `/ion/temperature_fit/weight` and `/.../density_fit/reconstructed`.
  Re-run with `CONTAINS` it returned **14**, and the `fit_artifact` category
  sample (`plasma_profiles/profiles_1d/ion/temperature_fit/weight`) confirms the
  shape. Row 14 reports the 14, not the 0.
- The "no successor" detector returned **14** through an outgoing
  `HAS_SUCCESSOR` edge, but a relationship-type census of `StandardName` shows
  `HAS_SUCCESSOR` carries only 14 edges in total while `REFINED_FROM` carries
  1850, and a successor points *into* the superseded node
  (`superseded_refined_into_reverse` = 1641). Re-run with an incoming
  `REFINED_FROM` comprehension when the scalar is null it returned **517**,
  which is the plan's 513 order of magnitude and not an artifact.
- The `status` census returned no `<null>` key. That is a real zero rather than a
  missed key because the same `coalesce(...,'<null>')` idiom produced a `<null>`
  row on two sibling axes in the identical query shape — `docs_stage` (2) and
  `validation_status` (2) — so the idiom is known to see a null when one exists.

## The first unstarted beat

Plan document order is §2 → §3 → §4 → §5 → §5a, with §5's five footgun rows in
table order. Against the code and the graph:

- §2 landed — the exporter reads the graph status and refuses a null (row 9).
- §3 landed — the two axes were kept, each with one writer.
- §4 landed — the import guard exists (row 11).
- §5 row 1 landed — the failure write path records a reason; the population is
  the residual (row 12).
- §5 row 2 — **unstarted**.
- §5 row 3 landed — `fit_artifact` is applied.
- §5 row 4 — decision locked, not landed (row 15).
- §5 row 5 landed — the gate writes a typed resolution method (row 16).
- §5a landed — `updated_at` is declared and stamped across write paths (row 17).

**First unstarted beat: §5 row 2, the compose attempt cap → `still-required`.**

The measurement that decides it: `MATCH (s:StandardNameSource) WHERE
s.attempt_count >= 5` returns 234, of which **217 sit at exactly 5** — the cap
value itself, which is the signature of a cap and not of a source that gave up
on its own — and 70 of those produced no name while 41 carry no recorded reason
(`logs/census.jsonl`, `logs/followup3.jsonl`, `logs/followup4.jsonl`). A repair
that had landed would have raised the cap, written a stop reason, or drained the
cohort; the population is intact at the cap and one row larger than the plan's
216. So the beat is not complete, and it is the first in document order that is
not.

## Remaining effort, restated in worker-hours

The plan declares `plan-effort-hours` 8.0 at `plan-impl` 0.9. Restated against
the re-measured state, the residual is the plan's own 10%:

**0.8 worker-hours remain of the declared 8.0**, covering the two beats that are
not landed — §5 row 2 (the attempt cap) and §5 row 4 (the `superseded_by`
scalar retirement). The basis is the plan's own residual rather than a fresh
estimate, and it should be read with two cautions the measurement makes
concrete: the attempt-cap beat is a repair pass over 234 sources, and the scalar
beat is a cross-module field removal the prior currency audit measured at six
modules plus a 2163-row migration. Either can exceed the declared residual, so
the figure is a floor for those two beats rather than a schedule.

## Validation

`uv run reckon audit-doc docs/evidence/sn-lifecycle-integrity/plan-figure-audit.md`
exits 0 (`logs/audit-doc.log`). No figure is embedded: every relationship in this
audit is a table row or a two-value comparison, and an image of what is
naturally a table is disallowed by the landing contract.