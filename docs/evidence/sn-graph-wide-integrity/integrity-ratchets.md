# Standard Name integrity ratchets

Date measured: 2026-08-20

## Result

The new live-graph module records ceilings for four semantic-integrity classes
and fails with the complete measured identities whenever a ceiling is exceeded.
The current graph contains 2,538 accepted Standard Names, so the corpus gate is
well above the ten-name minimum and all four assertions ran rather than skipped.

| Integrity class | Measured | Ceiling | Result |
|---|---:|---:|---|
| Standard Name sources with more than one live target | 23 | 23 | pass |
| Stale sources retaining a live producing binding | 3 | 3 | pass |
| Live names with no producing source and no live structural child | 36 | 36 | pass |
| Explicit-axis DD sources selecting the matching generic parent identity | 1 | 1 | pass with named residue |

The axis ratchet derives its axis from the live projection `HAS_PARENT` edge and
requires a literal DD leaf of `<generic-parent>_<axis>` or
`<axis>_<generic-parent>`. This avoids treating a nested axis token belonging to
another quantity in a compound DD leaf as the outer projection. The one measured
residue is:

- source `dd:langmuir_probes/reciprocating/plunge/mach_number_parallel`;
- selected generic parent `mach_number`;
- live projection child `parallel_mach_number`;
- projection axis `parallel`.

An empty result remains valid after that binding is repaired. Any different
single row fails even while the numeric ceiling remains one, so the known
residue cannot be silently replaced by a new mismatch.

## Live-state definitions

- A live target or structural child has `name_stage` other than `superseded` or
  `exhausted`.
- A multi-target source has more than one distinct live `PRODUCED_NAME` target.
- A stale live binding starts at `StandardNameSource.status = stale` and follows
  `PRODUCED_NAME` to a live target.
- A genuine unsourced name is live, has no incoming `PRODUCED_NAME` edge, and
  has no incoming live-child `HAS_PARENT` edge.
- An axis mismatch is a composed or attached DD source whose
  `produced_sn_id` selects the generic parent of a live projection child while
  the DD leaf spells that exact parent identity plus the projection axis.

Every ceiling assertion formats the offending source or name IDs, target IDs,
and axis-child IDs rather than reporting only an aggregate count. The
module-scoped corpus fixture skips all four tests with an explicit population
message when fewer than ten accepted names exist.

## Verification

- Focused live graph:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env pytest -p no:cacheprovider -m graph tests/graph/test_sn_integrity_ratchets.py -vv`
  — **4 passed, 0 failed**, 1 configuration warning, 41.80 s.
- Full Standard Names suite:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`
  — **6,531 passed, 8 skipped, 2 failed, 245 deselected**, 34 warnings,
  1,258.30 s. Both failures were 30-second timeout failures:
  `test_review_writer_requires_successful_owned_transition[None-name]` took
  30.60 s and `test_pool_phase_import_orders_succeed_in_fresh_process[...]`
  took 30.09 s. The run therefore does not satisfy the required zero-failure
  evidence, despite 6,531 passing tests. No second full run was attempted
  because its measured 21-minute duration would exceed the worker time fence.

Durable read-only census and test logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T202243350344-sgwi-integrity-invariant-ratchets/preflight-census.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T202243350344-sgwi-integrity-invariant-ratchets/axis-census.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T202243350344-sgwi-integrity-invariant-ratchets/focused-test-second.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T202243350344-sgwi-integrity-invariant-ratchets/standard-names-suite.log`

The earlier failed focused log is retained as investigation evidence. Its broad
suffix predicate included nested `parallel` tokens belonging to magnetic-field
mechanisms; no code or graph state was changed by that read-only run.
