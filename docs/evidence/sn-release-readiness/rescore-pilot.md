NEEDS-HELP: no sanctioned accepted-but-unscored cohort transition exists in the current CLI, so the 50-name pilot was not opened.

# Catalog-import name rescore pilot preflight

Date: 2026-08-23  
Graph: `codex` at `iter`  
Requested pilot: 50 accepted catalog-import identities with no name-axis score  
Requested spend fence: USD 75  
Outcome: **BLOCKED BEFORE MUTATION OR MODEL SPEND**

## Material outcome

The sanctioned *downstream* mechanism is the ordinary `REVIEW_NAME` pool. It
claims only `name_stage='drafted'` and `validation_status='valid'` identities,
keeps the name unchanged, obtains a fresh RD-quorum draw, and lets
`persist_reviewed_name` determine whether the identity becomes accepted,
reviewed/refine-eligible, or exhausted. This is the only route that satisfies
the binding rule that acceptance is earned by the quorum rather than assigned
by an operator.

The current repository has no sanctioned *upstream* operator that moves an
accepted, valid, unscored identity into that pool while preserving its catalog
identity and bindings:

- `sn rescore` is explicitly limited to stranded **non-accepted** names
  (`imas_codex/cli/sn.py:7367-7404`). Its compare-and-set backend matches only
  `exhausted` or `reviewed` and explicitly refuses `accepted`
  (`imas_codex/standard_names/graph_ops.py:23220-23307`). It is also a
  single-name operator rather than a guarded cohort operator.
- Exact-name `sn run --name ...` only preflights and stamps `run_id`; it does
  not change lifecycle state (`imas_codex/cli/sn.py:2230-2284` and
  `imas_codex/standard_names/graph_ops.py:1787-1873`). An accepted name therefore
  remains ineligible for `REVIEW_NAME`, whose eligibility is documented and
  enforced as drafted plus valid
  (`imas_codex/standard_names/graph_ops.py:14574-14615`).
- `sn run --reset-to drafted --include-accepted` does not express the required
  transition. Without `--retry-quarantined`, the CLI calls
  `reset_standard_names(from_stage='drafted', to_stage=None)`, so accepted rows
  match zero (`imas_codex/cli/sn.py:2457-2466`). Calling the backend directly
  with accepted/drafted arguments is not a review-restage side-car: it clears
  generation and DD/COCOS projections and removes `HAS_STANDARD_NAME`,
  `HAS_UNIT`, and `HAS_COCOS` relationships
  (`imas_codex/standard_names/graph_ops.py:7807-7840,7965-7995`).
- `sn review --stage accepted --unreviewed` is not the lifecycle quorum pool.
  Its writer stores score fields on the already-accepted node without moving
  the lifecycle through drafted and without letting the score decide acceptance
  (`imas_codex/standard_names/graph_ops.py:6099-6179`). A below-threshold result
  would remain accepted, so it cannot measure the requested acceptance and
  refine outcomes.

The explicit `--include-accepted` data-safety guard is authorized for this
pilot because the node dispatch deliberately requires touching the 50 accepted
catalog-authoritative rows. It was **not used**, because no current exact-cohort
restage operator consumes it with the required semantics. Treating the flag as
authorization for raw Cypher or for the destructive generic reset would turn a
safety guard into a bypass.

## Live preflight

Read-only graph queries reproduced the intended release-eligible catalog cohort:

| Population | Live count |
|---|---:|
| `accepted`, `catalog_edit`, valid, docs accepted, null name score, null reviewer model | 1,181 |
| Exact archive-score recoveries already identified by the plan | 2 |
| Remaining identities requiring a fresh quorum draw | 1,179 |
| Foreign live `SNRun` rows with a heartbeat in the preceding five minutes | 0 |

The two archive identities are `normalized_collisionality` and
`thermal_ion_density`; they were excluded from the intended paid population in
the live-plan design. The run window was clear, but clearing the scheduler
window does not create lifecycle authority.

No graph write, lifecycle transition, reviewer call, or other provider call was
made by this node. In particular, no accepted identity was temporarily removed
from export eligibility.

## Requested quantitative receipt

These figures are deliberately reported as **not produced**, rather than
substituting a dry run or an unsafe lifecycle mutation for an actual campaign.

| Measure | Actual result |
|---|---:|
| Pilot identities selected and staged | 0 |
| Pilot identities actually reviewed | 0 |
| Total USD spent | USD 0.00 |
| USD per reviewed name | not measurable (0-name denominator) |
| LLM calls | 0 |
| LLM calls per reviewed name | not measurable (0-name denominator) |
| Accept rate at or above 0.85 | not measurable (0-name denominator) |
| Routed to refine | 0 (campaign did not open) |
| Terminal name stages after campaign | 0 of 0 |
| Mid-pipeline after campaign | 0 of 0 |
| Left stranded at `drafted` by this node | 0 |
| USD 75 cap accuracy to within one call | not exercised; cannot be measured |

There is no list of 50 before/after identities because there was no actual
50-identity campaign. Inventing such a receipt would falsely imply graph
mutation and quorum evidence that do not exist. All 1,181 live cohort rows
remained in their before state (`accepted`); this node selected and mutated none
of them.

The pilot therefore does **not** supersede any earlier estimate. The three
figures remain projections or historical context, not a measured answer:

| Prior figure | Basis | Implied USD/name |
|---|---|---:|
| USD 0.1038 | post-hoc ledger average | 0.1038 |
| USD 271.17 for 1,179 | controlled projection | 0.2300 |
| USD 1,276.78 for 1,179 | production-enriched projection | 1.0829 |

The requested measured actual, actual acceptance rate, and actual call fan-out
remain unknown.

## Tried

1. Read the live plan and confirmed that catalog imports must earn acceptance
   through the ordinary name quorum, with no rewording and no hand-written
   score.
2. Audited every repository hit for accepted-to-drafted, rescore, exact-name
   scope, and `--include-accepted` transitions across the CLI, Standard Names
   implementation, and tests.
3. Ran read-only graph preflight queries to reproduce the 1,181/1,179 cohort,
   classify the accepted-unscored rows, and verify the attributed run window was
   clear.
4. Stopped before mutation when every public path either refused accepted rows,
   left them accepted while writing a score, or destroyed bindings outside the
   intended review lifecycle.

## Options

1. **Add a guarded exact-cohort rescore operator (recommended).** Extend
   `stage_name_for_rescore` and the CLI to accept an explicit manifest of exact
   identities plus `--include-accepted`. Its compare-and-set must require
   `name_stage='accepted'`, `validation_status='valid'`, null name score, the
   catalog-import cohort predicates, and an otherwise claim-free row; preserve
   the identity, documentation, source bindings, units, COCOS, and lineage;
   clear only stale name-review/claim/refine fields; stamp one run scope; and
   refuse the entire cohort if any member changes between preflight and stamp.
   Then drain that exact run scope through ordinary `REVIEW_NAME` with
   `--names-only --skip-global-maintenance --cost-limit 75` and verify all 50
   returned to a terminal name stage before closing the run.
2. Add a separate accepted-import repair side-car with the same atomic
   predicates, explicit guard, rollback receipt, and ordinary pool handoff.
   This is clearer as a one-time migration but duplicates rescore lifecycle
   machinery.
3. Use raw Cypher or the generic reset backend. **Rejected:** neither is a
   sanctioned review transition; the reset removes authority-bearing bindings,
   and raw mutation would have no tested all-or-none restoration contract.

## Leaning

Option 1. It reuses the existing same-identity rescore validation, scoped pool,
and failure-restoration design while adding only the missing accepted cohort
eligibility and `--include-accepted` authorization. It needs source and test
write scope before any live row is touched.

## Cost if the wrong path is taken

An unsafe restage can de-list up to 50 catalog names from export while leaving
them at drafted, or strip their DD, unit, COCOS, and source relationships. The
standalone review command can produce the opposite false result: low-scoring
rows remain accepted, inflating the apparent accept rate and suppressing refine
routing. Either mistake invalidates the pilot and requires graph restoration,
fresh attribution, and a complete rerun; direct acceptance or hand-written
scores would invalidate the evidence outright.

## Exact blocker

The necessary guarded accepted-import cohort restage does not exist, and adding
it requires edits outside this node's exclusive evidence-only write fence.
Until that operator is implemented and tested, the requested actual campaign
cannot be run without violating the acceptance, data-safety, or receipt
requirements.
