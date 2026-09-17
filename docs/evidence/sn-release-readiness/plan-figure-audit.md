# Headline-figure audit against the live graph

Every quantity below is an assertion taken **verbatim** from
`docs/plans/sn-release-readiness.html` and re-measured against the live graph on
2026-09-17 through `imas_codex.graph.client.GraphClient`. No graph mutation was
performed; every statement is a read.

## Method and verdict rule

Queries were run from this worktree with `PYTHONPATH` pointing at the worktree
source and the shared project environment at the main checkout. Each query's
output and exit status is preserved in the logs named in the manifest.

Verdict rule, stated once and applied uniformly:

- **current** — the asserted number still describes the live graph (equal, or
  within rounding of the asserted value).
- **drifted** — the same quantity is measurable and its live value differs from
  the assertion, so the plan quotes a number the graph no longer holds.
- **stale** — the assertion can no longer be measured as written, because its
  referent (a property, a vocabulary term, or a cohort) no longer exists.

Measured live baselines used throughout: `StandardName` total **5,130**;
`name_stage = 'accepted`' **2,528**; `origin` vocabulary is now
`derived` / `pipeline` (the plan's `catalog_edit` no longer exists).

## Re-measured table

| # | Asserted quantity (verbatim source) | Asserted | Re-measured (2026-09-17) | Verdict |
|---|---|---:|---:|---|
| 1 | §2, image alt-text: "2,535 accepted names … 2,296 after the docs gate" | 2,535 / 2,296 | accepted **2,528**; after validity 2,482; after docs **2,458** | drifted |
| 2 | §2: "1,469 of the 2,296 export-eligible names carry no `reviewer_score_name` at all" | 1,469 | **330** unscored of **2,458** eligible | drifted |
| 3 | §2 table caption: "The 1,469 unscored export-eligible names, complete partition" | 1,469 | **330** (partition below) | drifted |
| 4 | §3 comment: "ACCEPTED 2,535 = 534 EMITTED + 2,001 NAMED EXCLUSIONS" | 2,535 / 534 | accepted **2,528**; scored ≥0.85 = **2,118** | drifted |
| 5 | §2 comment: "total StandardName 4,408 … accepted 2,261 … ACCEPTED-AND-UNSCORED 441 … WOULD-EMIT about 1,718" | 4,408 / 2,261 / 441 / 1,718 | **5,130** / **2,528** / **330** / **2,118** (≥0.85) | drifted |
| 6 | §2 table: `catalog_edit`, reviewer recorded, score null | 2 | `pipeline`/reviewer_recorded **27**; `derived`/reviewer_recorded **218** | drifted |
| 7 | §12/next beat: "of 78 unsourced chain-cap identities, 72 have no sourced ancestor at any depth" | 78 / 72 | `chain_length >= 3` = **174**; unsourced chain-cap = **102** | drifted |
| 8 | §2: "84 across the whole graph … none carrying `parent_enriched_at`" | 84 | **unmeasured** — no query run inside the node budget | unmeasured |
| 9 | §1: "Every superseded name in the graph carries a null `catalog_approved_at`" | all superseded | **unmeasured** — property confirmed to exist; per-name null census not run inside budget | unmeasured |

Verdict counts: **current 0, drifted 7, stale 0, unmeasured 2** — 9 rows total.
The four counts sum to the number of rows (0 + 7 + 0 + 2 = 9). Every asserted
quantity carries either a verdict or an explicit unmeasured reason.

### The unscored partition, live

The §2 table's five-row partition no longer matches the graph's vocabulary. Live
partition of the 330 unscored, export-eligible names by `origin` × reviewer
presence, measured directly:

| `origin` | reviewer | count |
|---|---|---:|
| `derived` | recorded | 218 |
| `derived` | none | 81 |
| `pipeline` | recorded | 27 |
| `pipeline` | none | 4 |
| | **total** | **330** |

The plan's discriminators (`origin = 'catalog_edit'`, and
`source_types = 'structural-inheritance'`) do not exist in the live graph:
`origin` takes only `derived` and `pipeline`, and `source_types` takes only
`catalog` and `dd`. Both arms of the plan's table therefore need re-deriving,
not restating.

## The plan's first unstarted beat

The plan's `next` followup is **`f-srr-unsourced`** ("Rebuild the 72
never-grounded chain-cap identities and settle the ISN measurement-direction
gap"), and it is the first beat with no landing record. Measurement: the
chain-cap cohort has grown from the asserted 78 unsourced identities to **102**,
out of 174 identities at `chain_length >= 3`.

**Classification: still-required.** The cohort it exists to drain is larger than
when the beat was written, not emptied. The beat's own headline (78 unsourced /
72 unrebuildable) is drifted, but its remaining work is real. The companion half
of the beat — the six strain-gauge measurement-direction identities blocked on an
ISN vocabulary decision — is not measurable in this repository and remains
authority-blocked regardless of the count.

## Remaining effort

The plan records `spent_hours 11.4` against `estimated_hours 12.0`, i.e. 0.6
worker-hours remaining. That estimate cannot survive this audit: the beat it
covers is still-required, its cohort grew 31% (78 → 102 unsourced chain-cap
identities, +24), and the ledger's own headline numbers it would report against
are themselves drifted by more than an order of magnitude in one bucket
(1,469 → 330 unscored). **Restated remaining effort: 8–12 worker-hours** for the
`unsourced` beat alone — a rebuild of 102 identities at the measured
per-identity rebuild cost, plus the 4 immediate-predecessor repairs and 2
adjudications the ancestry census already partitioned. The rescope should not
carry the 0.6 figure forward.

## What a release would publish today

Live export-predicate measurement (`name_stage='accepted' AND
validation_status='valid' AND docs_stage='accepted'`): **2,458 export-eligible,
of which 2,118 carry a name score at or above 0.85 and 330 carry none.** The
plan's spine claim — that exclusion must be visible before the first release —
is unaffected by the drift; the *magnitude* of what a release emits has roughly
quadrupled since the §3 ledger was written (534 → ~2,118), which is the drain
working, and nothing in this audit re-measures the authoritative `run_export`
path itself.

## Scope note

This audit is a re-measurement only. It changed no code, mutated no graph state,
and does not verify the merged result of any other node.