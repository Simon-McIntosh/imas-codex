# Accepted StandardName embedding-gap closure

## Outcome

The accepted-population gate now passes at exact zero. The controlled census
changed from **2 accepted rows without an embedding and 2,316 with one** to
**0 without and 2,318 with one**. The firing control is the 2,318 accepted rows
that carry an embedding; `StandardName.id` coverage is independently complete at
2,318 of 2,318 accepted rows and 4,670 of 4,670 graph-wide rows.

The two affected identities were:

| StandardName id | Created at | Producing run | Verdict | Deciding evidence |
|---|---|---|---|---|
| `frequency_of_wave_diagnostic_channel` | `2026-09-01T08:21:03.65Z` | `e16307b7-b16d-4595-86c7-756b45ee851f` | **Never produced** | Before repair, `embedding`, `embedded_at`, `embed_text_hash`, and `embed_failed_at` were all null. The run log records scoped mode bypassing global maintenance, then name acceptance at 10:23:26, description persistence at 10:23:45, and documentation acceptance at 10:25:33. There are **0** `embed_worker` lines from the run's start through its 10:40:23 pool exit. |
| `intensity_at_spectral_line` | `2026-09-01T08:42:58.607Z` | `5632b8ac-a204-4259-812e-39cf0975f771` | **Never produced** | The same four embedding-state fields were null before repair. Its run likewise records scoped mode bypassing global maintenance, then name acceptance at 10:44:24, description persistence at 10:44:41, and documentation acceptance at 10:46:02. There are **0** `embed_worker` lines from start through the 10:46:55 pool exit. |

These are omissions, not lost vectors. Both rows were newly created by the two
named runs, retained real descriptions, had no failed-embedding timestamp, and
never had an embedding producer active during those runs. The source and run-log
evidence agree on that mechanism; there is no observed write-then-clear event for
either identity.

## Producer defect

A producer defect exists. In `run_sn_pools`, scoped mode sets
`skip_global_maintenance`; the only automatic StandardName embedding producer is
then created only under `if not skip_global_maintenance`. Scoped runs can still
generate a name, accept it, generate its description and accept its documentation,
but they never start `embed_description_worker(labels=["StandardName"])`.
Consequently a newly accepted row can exit the scoped run with a real description
and no vector. This is exactly the path both rows followed.

The defect is in `imas_codex/standard_names/loop.py` at the embedding-worker
construction guard, not in the embedding service and not in either row's content.
The evidence fence permits only this report, so the producer repair is not folded
into this commit. It needs a follow-on code node against `loop.py` and its
orchestration tests: scoped runs must retain a scoped or final embedding drain,
and completion must not return while newly produced StandardName descriptions are
unembedded.

## Deterministic repair

Both identities were repaired through the existing dedicated StandardName path:
`claim_embed_batch` -> `process_embed_batch` -> `persist_embed_batch`. Claims were
fenced by each row's producing `run_id`; the first claim also selected three
already-embedded rows whose hashes were absent, and those three claims were
released immediately without processing. The repair persisted exactly one row for
each target.

No description, review, lifecycle state, threshold or catalog field changed. The
two vectors have the configured dimension of **256**:

- `frequency_of_wave_diagnostic_channel`: `embed_text_hash=9929398fe47c80ce`,
  `embedded_at=2026-09-01T12:16:52.925Z`.
- `intensity_at_spectral_line`: `embed_text_hash=8859dd6d5aebb88a`,
  `embedded_at=2026-09-01T12:16:53.104Z`.

Embedding used the configured local embedding service and incurred no LLM cost.
Before and after, the graph held **35,065** `LLMCost` rows totaling
**USD 1,715.784847**; the target-linked subset stayed at **11** rows and
**USD 0.348719**. Deltas were therefore **0 calls and USD 0.000000** globally and
for the two targets.

## Re-measured gate

| Observation | Before | After |
|---|---:|---:|
| All StandardName rows | 4,670 | 4,670 |
| All rows carrying schema key `id` | 4,670 | 4,670 |
| Accepted rows | 2,318 | 2,318 |
| Accepted rows carrying an embedding | **2,316** | **2,318** |
| Accepted rows lacking an embedding | **2** | **0** |

The independent post-repair assertion required a positive accepted-population
control, complete `id` coverage, accepted-with-embedding equal to accepted, and
accepted-without-embedding equal to zero. It exited 0 with **2,318 present and 0
missing**.

## Evidence inputs

- Before/after censuses, exact target fields, scoped claims, write counts,
  dimensions, hashes and zero-cost deltas:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T121202639118-n-embedgap/logs/reembed.log`.
- Producer run excerpts, zero-worker counts and the live source guard:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T121202639118-n-embedgap/logs/producer-evidence.log`.
- Independent post-repair firing-control assertion:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T121202639118-n-embedgap/logs/accepted-embedding-gate.log`.
