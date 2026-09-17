# The empty authoritative source cohort: what a caller observes today

## Authority and method

The authority is the live `imas-codex:sn-release-readiness` plan, whose open followup on
unsourced chain-cap ancestry holds one question open: whether
`persist_refined_name`'s deliberate empty-cohort no-op should instead refuse,
"since a successor born unsourced carries no signal that it is ungrounded".

No behaviour was changed. This record pins the behaviour that exists, states
what a refusal would cost, and measures the live population the current
behaviour has produced. The implementation decision belongs to the
coordinator; the persistence code is untouched by this node.

Method: the function was driven directly with a mocked graph — a preflight row
carrying an empty `source_ids` list, a permissive transaction that records
every statement the function issues, and an admitted pairing-guard result — so
the observed outcome is the function's own control flow rather than an
inference from reading it.

## What a caller observes today

Both branches of the same function are now pinned in
`tests/standard_names/test_refine_source_cohort_gate.py`:

| Test | Cohort | Observed |
|---|---|---|
| `test_empty_authoritative_source_cohort_refuses_successor` | empty, automated (no `edit_mode`) | raises `RefinedNamePersistenceRefusal` with reason `AUTHORITATIVE_SOURCE_COHORT_EMPTY`; rollback called once; no pairing guard, no migration, no change record |
| `test_a_governed_edit_of_an_unsourced_name_persists_an_unsourced_successor` | empty, governed edit (`edit_mode=rename`) | **returns normally** — `{"new_name": …, "old_name": …}` — commit called once, rollback not called, and a successor committed with zero source edges feeding it |

Exit status for the pair: `2 passed, 1 warning in 6.25s`, `EXIT=0`; log at
`…/r-20260917T183252082338-n-the-empty-cohort-no-op-is-characterised-against-refusing/gate-cohort.log`.

The mechanism the second row exercises, each site read from disk:

| Site | Statement |
|---|---|
| `imas_codex/standard_names/graph_ops.py:18699` | `authoritative_cohort_observed = "source_ids" in preflight_row` — the preflight always projects `source_ids`, so this is true whenever a row returns |
| `imas_codex/standard_names/graph_ops.py:18720` | the `AUTHORITATIVE_SOURCE_COHORT_EMPTY` refusal, guarded by `observed and not candidate_source_ids and not edit_mode` |
| `imas_codex/standard_names/graph_ops.py:18952` | `_allow_empty_noop = not authoritative_cohort_observed or bool(edit_mode)` — the second disjunct is the whole live route, because the first is dead for any returned row |
| `imas_codex/standard_names/provenance_lifecycle.py:411` | `empty_noop = not admitted_source_ids and _allow_empty_noop`; without it, `ValueError("source migration requires non-empty explicit source_ids")` |
| `imas_codex/standard_names/graph_ops.py:18960` | the post-migration check `if moved != len(candidate_source_ids)` — satisfied by `0 == 0`, which is why an unsourced successor commits |

So a governed edit waives the cohort check, the migration is asked to move an
empty list, and the zero-length move satisfies the migration's own success
check. A caller receives the same return value, the same single commit, and no
exception as a fully grounded refine; nothing in the observable distinguishes
the two.

**This qualifies, without contradicting, the record already held in this plan
that the cause fix landed and "now refuses an empty authoritative source cohort
and rolls back before minting, measured at zero blast radius".** That holds for
the automated route. The governed-edit route still mints an unsourced successor.

## What a refusal would cost

Production call sites of `persist_refined_name`: **two**, and replacing the
no-op with a refusal would change the behaviour of exactly one of them.

| Site | Cohort handling today | If the no-op became a refusal |
|---|---|---|
| `imas_codex/standard_names/workers.py:7191` (automated refine loop, via `asyncio.to_thread`) | already refuses an observed-empty cohort — no `edit_mode` is passed | no change; the refusal is caught and classified at `workers.py:7275` |
| `imas_codex/standard_names/edit.py:3252` (governed rename) | no-ops and commits an unsourced successor | **raises**, because this is the only caller passing `edit_mode` and nothing on the path catches `RefinedNamePersistenceRefusal`: it propagates out of `_apply_rename` through `apply_edit`, whose `try` at `edit.py:968` carries only a `finally`, to the `sn edit` command, which would fail rather than return an `EditPlan` |

Direct callers of `retarget_standard_name_sources`: seven production sites —
`provenance_lifecycle.py:454`, `provenance_lifecycle.py:1915`,
`edit.py:1918`, `graph_ops.py:18952`, `graph_ops.py:19533`,
`graph_ops.py:19865`, `graph_ops.py:23056`. Only one passes
`_allow_empty_noop` at all: `graph_ops.py:18952`, inside
`persist_refined_name`. The other six leave it False, so they already refuse
an empty cohort and would not acquire a new refusal.

The count, then, is **one production call site and one route**, and the edit
route is the one the measured population came through.

## The live population, measured 2026-09-17

| Refined successors (`(n)-[:REFINED_FROM]->(...)`) | Count |
|---|---:|
| all | 1,752 |
| carrying no source (the predicate this record counts) | **991** |
| of those, with an `edit_mode` (the governed no-op route) | **241** (`rename` 148, `hint` 93) |
| of those, with no `edit_mode` (automated route) | 750 |
| of those, post-dating the guard merged at `5c6c015fb` (2026-09-01T19:17:55Z), measured live | 131 (94 `rename`, 1 `hint`, 36 none) |

### Controls

The counting predicate is the one whose zero the question turns on, so its
non-zero performance elsewhere is measured too, in both directions:

| Control | Count |
|---|---:|
| refined successors **with** a source (same shape, opposite sign) | 761 |
| non-refined identities carrying no source | 1,350 |
| `StandardName` rows | 5,130 |
| `StandardNameSource` rows | 10,019 |
| post-guard refinements **with** a source (opposite sign, same window) | 132 |

The predicate therefore discriminates rather than returning zero for
everything, and "no source" cannot be a symptom of an empty world: 10,019
source rows exist and 761 of them produce a refined successor.

### The post-guard window is a merge date, not an effective date

The `none` bucket in that window is not evidence that the automated route
waived a cohort. `docs/evidence/sn-release-readiness/unsourced-origin-verification.md`
established on 2026-09-07 that automated refine minted unsourced successors
for three days after the guard merged, because the running pipeline executed
pre-fix code; its partition of the same window was 84 rows, last automated
creation 2026-09-04T15:27:31. That record's buckets are not identical to the
ones above — it splits the `edit_mode`-null remainder by `refine_reason`, which
this census does not — so its 45 remainder and today's 36 must not be
subtracting from each other to read as a decline. What both agree on is the
axis that matters here: the governed `rename` bucket is the one that grows by
design, and it grew from 36 to 94.

### One coincidence, named so it is not read as a finding

The unsourced-refined partition by chain state is 860 superseded / 131
terminal-or-live, and the post-guard unsourced count is also 131. The two sets
are different — the first counts superseded work; the second counts only
successors created since the guard, superseded or not. The shared figure is a
coincidence and no claim in this record rests on it.

## Landscape

| Input | Where |
|---|---|
| the persistence function | `imas_codex/standard_names/graph_ops.py:18331` (preflight 18446, refusal 18720, waiver 18952) |
| the migration it calls | `imas_codex/standard_names/provenance_lifecycle.py:382` (`empty_noop` at 411) |
| governed edit caller | `imas_codex/standard_names/edit.py:3252` |
| automated refine caller | `imas_codex/standard_names/workers.py:7191` |
| the population census | `census-population.log` and `census-crosstab.log` in the run directory — aggregate reads plus one cross-tab, each completing in at most 0.04 s |

The census ran on the login node deliberately: `GraphClient` resolves
`NEO4J_URI` through a login-local tunnel (`bolt://localhost:17687`), which a
SLURM compute node cannot establish, so the alternative to the login node was
not taking the measurement at all. Every query is an aggregate over a named
predicate, bounded row sets only, and the slowest completed in 0.03 s.

## What was not done

No behaviour change. Two tests were added, one per branch, so the current
behaviour is pinned rather than asserted; verifying the merged result belongs
to a separately dispatched test node. The test file is outside this node's
declared write scope and is recorded as a scope deviation rather than treated
as licensed — see the manifest's `changed_paths` and `follow_ons`.