# Atomic count ancestor-fold apply

## Completed apply — 2026-08-19

The hardened `supersede_into_ancestor` operator folded
`atomic_count_of_ion_state` into its existing `atomic_count` ancestor. The
fresh preview reproduced the reviewed manifest SHA-256 exactly:

`50f3c01c8dbc6ee6a1da2f58544d78a773b93c4a4a05fa9032f00865936198f8`

The apply accepted that same hash, not a substituted or stale value. Its signed
76-source cohort was unchanged from review: 36 sources retargeted, 39 existing
dual bindings deduplicated, one stale source detached, and zero stale sources
refused. The stale disposition was
`dd:waves/coherent_wave/beam_tracing/beam/ion/element/multiplicity`.

The transaction wrote two deterministic `StandardNameChange` receipts, one for
the 36-source migration manifest and one for the enclosing lifecycle fold:

- `sn-change:source-migration:696776fed6ad79d9fcf27727f46f19a12c8555c1fcb767c5ecef54e090f4fa51`
- `sn-change:ancestor-supersession:50f3c01c8dbc6ee6a1da2f58544d78a773b93c4a4a05fa9032f00865936198f8`

Replay returned `already_applied`, `changed=0`. Exact snapshots of all signed
names, sources, and incident relationships were byte-identical before and after
replay: 79 nodes, 286 relationships, 4,848,894 serialized bytes, SHA-256
`f56d1956db1265ce315117b112188220c01eabdfab431f013c17a131152f2212`.
Global graph counters were also identical across the replay: 1,589,061 nodes,
4,234,378 relationships, 7,151 `StandardNameChange` nodes, 27,467 `LLMCost`
nodes, and 489 `SNRun` nodes. Measured replay writes were therefore zero.

The post-apply census proved the intended lifecycle and provenance state:

| Measure | Result |
|---|---:|
| `atomic_count_of_ion_state.name_stage` | `superseded` |
| `atomic_count_of_ion_state.status` | `superseded` |
| Live sources still bound to `atomic_count_of_ion_state` | 0 |
| Total sources bound to `atomic_count` | 77 |
| Signed cohort sources bound to `atomic_count` | 75 |
| Signed stale sources left detached | 1 |
| Stale source status | `stale` |
| Stale source scalar / live bindings | `null` / 0 |
| Surviving `REFINED_FROM` edges | 6 |
| Apply `StandardNameChange` receipts | 2 |
| Replay persistent writes | 0 |
| `LLMCost` / `SNRun` deltas | 0 / 0 |
| Provider calls | 0 |

The ancestor's 77 total bindings comprise 75 surviving members of the signed
cohort plus two pre-existing sources outside it. The 76th signed source is the
preserved-stale row, intentionally detached rather than revived or migrated.
All six pre-existing lineage edges remain; the operation created no reverse
`REFINED_FROM` edge.

The canonical `.env` was copied only for graph access, removed after every
invocation, and never staged. Full outputs are retained as `live-apply.log`,
`post-apply-verification.log`, and `receipt-census.log` in the worker run
directory. The first driver committed the fold successfully, then exited 1 on
an over-strict assertion that expected one change row; the receipt census
proved the correct sanctioned delta is two because the retarget primitive
persists its own source-migration manifest receipt. The separate replay and
census driver exited 0.

## Earlier fail-closed refusal — 2026-08-18

Before the stale-source lifecycle contract was hardened, the exact signed
cohort could not be applied because one signed source was stale and the
sanctioned migration primitive refused stale sources.

- tried: Regenerated the live dry-run for `atomic_count_of_ion_state` into `atomic_count`, obtained the exact approved SHA-256 `e51c191e06fb4d8df8d8e32ff374304e0e02a94cdcdeef1238ddbb9a22019a57`, and invoked the hash-bound apply once. The manifest remained 76 sources: 37 retarget and 39 deduplicate, with six `REFINED_FROM` edges.
- observable result: The apply transaction raised `source migration compare-and-set failed` for `dd:waves/coherent_wave/beam_tracing/beam/ion/element/multiplicity`, whose current signed state is `status='stale'`, scalar `atomic_count_of_ion_state`, and sole `PRODUCED_NAME` binding `atomic_count_of_ion_state`.
- rollback proof: An independent read-only census after the exception found no fold `StandardNameChange`, `atomic_count_of_ion_state` still accepted with 76 producing bindings, and `atomic_count` still carrying 41 producing bindings. The copied `.env` was removed and was never staged.
- options: (1) correct the instrument so its preview refuses stale migration rows before signing; (2) explicitly authorize stale sources for this ancestor-fold migration if stale provenance is intended to move; or (3) repair the source lifecycle through an existing sanctioned operator, regenerate a new manifest, and review the resulting hash delta.
- leaning: Option 1, unless Standard Name source lifecycle policy explicitly says stale sources remain valid migration participants. Preview and apply currently disagree on the same exact signed state, so fail-closed preview admission is the safest contract repair.
- cost-if-wrong: Choosing option 1 when stale sources should migrate requires revising the admission rule and its regression; choosing option 2 or 3 incorrectly risks moving or reviving provenance that lifecycle policy intended to remain stale.

## Recorded evidence

| Measure | Result |
|---|---:|
| Expected manifest SHA-256 | `e51c191e06fb4d8df8d8e32ff374304e0e02a94cdcdeef1238ddbb9a22019a57` |
| Regenerated manifest SHA-256 | `e51c191e06fb4d8df8d8e32ff374304e0e02a94cdcdeef1238ddbb9a22019a57` |
| Manifest sources | 76 |
| Retarget disposition | 37 |
| Deduplicate disposition | 39 |
| Signed lineage edges | 6 |
| Apply writes committed | 0 |
| New fold change receipts | 0 |
| Post-rollback descendant bindings | 76 |
| Post-rollback ancestor bindings | 41 |

Full command output is retained in `live-apply.log`; the independent post-failure state is retained in `rollback-census.log` in the worker run directory.
