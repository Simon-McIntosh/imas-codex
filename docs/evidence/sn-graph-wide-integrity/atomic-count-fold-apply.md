# Atomic count ancestor-fold apply

NEEDS-HELP: the exact signed cohort cannot be applied because one signed source is stale and the sanctioned migration primitive refuses stale sources.

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
