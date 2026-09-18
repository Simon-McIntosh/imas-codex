# The six red export tests start at one revision, and it is not before the work they were said to predate

Attribution for the six ids in followup `f-usb-six-export-tests-are-red-and-predate-this-work`.
Every one of them passes at the revision immediately preceding `b161235bb`, and fails at it —
measured on a login node and on a debug partition, over identical ids.

**The headline is a correction.** The followup records that the same six ids appear at three
revisions on one partition, including "the commit before the producer predicate landed at all",
and concludes they predate this session's export work. That third measurement does not
reproduce: at `617fdaca9` (the predicate itself) and at `b161235bb^` (the immediately preceding
revision, `b468bfb5c`) all fifteen tests in the two files pass. What the six are is the
fail-open exposure that closing a key-presence guard normally causes — the shape the followup
explicitly ruled out for this surface.

## The six, and the revision each starts at

All six start failing at the same revision, `b161235bb09ab939be067c7cd5ccf14e86cd5542`
("withhold an export candidate carrying no producer evidence", 2026-09-18T15:42:26+02:00),
and each was observed on both host classes measured.

| # | id | file:line | at `b161235bb^` | at `b161235bb` | host classes where red |
|---|---|---|---|---|---|
| 1 | `test_export_ledger_closes_over_fixture_population` | `test_export_exclusion_ledger.py:110` | pass | **fail** | login node, `all_debug` |
| 2 | `test_export_emits_generic_source_bindings_and_preserves_accounting` | `test_export_exclusion_ledger.py:142` | pass | **fail** | login node, `all_debug` |
| 3 | `test_export_validates_cross_links_against_full_catalog` | `test_export_exclusion_ledger.py:334` | pass | **fail** | login node, `all_debug` |
| 4 | `test_manifest_sources_reconcile_emitted_excluded_and_non_nameable` | `test_export_exclusion_ledger.py:501` | pass | **fail** | login node, `all_debug` |
| 5 | `test_explicitly_included_identities_remain_eligible` | `test_export_release_holds.py:122` | pass | **fail** | login node, `all_debug` |
| 6 | `test_release_holds_close_the_identity_ledger` | `test_export_release_holds.py:135` | pass | **fail** | login node, `all_debug` |

**They fail on every host class measured, not only some.** Two classes were used: the login
node and the `all_debug` partition. The file pair collects 15 tests (12 + 3) on both, and at
HEAD both classes report the same six ids, 6 failed and 9 passed. Nothing in this pair is
host-conditional.

## The command and host class behind every row

The login-node form, run from the worktree or from a revision tree:

```bash
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_export_exclusion_ledger.py \
  tests/standard_names/test_export_release_holds.py
```

The partition form, identical apart from the launcher, plus `export TMPDIR=/tmp`. The `uv`
shell function inserts `--no-sync` on SLURM, so passing it again exits 2 with
`the argument '--no-sync' cannot be used multiple times`; the first partition attempt failed
that way and is recorded rather than smoothed over.

```bash
srun --partition=all_debug --time=00:59:00 --cpus-per-task=4 --mem=32G bash -lc \
  'export TMPDIR=/tmp; cd <tree>; UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=<shared .venv> \
   PYTHONPATH=<tree> uv run pytest -p no:cacheprovider <the two files>'
```

| revision | tree | host class | collected | result |
|---|---|---|---|---|
| `4c9d1dabb4ab0afb679ccf15b85eb552e585169c` (the determinism record's) | archive | login node | 15 | 15 passed, 0 failed |
| `617fdaca9` (the producer predicate) | archive | login node | 15 | 15 passed, 0 failed |
| `b161235bb^` = `b468bfb5c1b86131ec16262149e778e9ea5e3dff` | archive | login node | 15 | 15 passed, 0 failed |
| `b161235bb` (the fail-closed repair) | archive | login node | 15 | **6 failed**, 9 passed |
| `617fdaca9` | tree on GPFS | `all_debug` | 15 | 15 passed, 0 failed |
| HEAD `af53d4671` | worktree | login node | 15 | **6 failed**, 9 passed |
| HEAD `af53d4671` | archive | login node | 15 | **6 failed**, 9 passed |

The last two rows are the instrument's calibration: the archived checkout at HEAD reproduces
the worktree run exactly, same six ids and same counts. The failure set is therefore a
property of the revision and not of the archive.

## Why they fail: one guard, one fixture route, six assertions

The commit removes the key-presence escape hatch from `_has_producing_source`
(`imas_codex/standard_names/export.py:777`), which now withholds a candidate that carries
none of `_has_derived_producer`, `_has_non_derived_producer` or `_has_live_child`. Every one
of the six drives the export through `_run_fixture_export`, which patches
`_fetch_export_population` to return rows built by `_candidate()`
(`test_export_exclusion_ledger.py:26`); that helper sets no producer flag, so the population
the guard sees is a caller-built projection carrying none of the three keys — precisely the
case the removed clause existed for, and precisely what the repair now refuses.

The per-id receipt is the assertion each one dies on, quoted from the calibrating run:

- **#1** `assert 0 == 1` on `ExportReport(...).exported_count`, whose exclusion ledger reads
  `{'invalid_validation_status': 1, 'no_producing_source': 3}`: three of the four fixture
  candidates are withheld for the absent producer evidence.
- **#2** `FileNotFoundError: .../test_export_emits_generic_sour0/standard_names/equilibrium.yml`
  — the emitted catalog file does not exist because nothing was emitted.
- **#3** `assert set() == {'electron_density', 'ion_density'}` — the emitted identity set is
  empty.
- **#4** `assert False` on `ExportReport(...).all_gates_passed`.
- **#5** and **#6** `assert set() == {'radial_neutral_state_momentum_flux', 'tendency_of_total_thermal_plasma_internal_energy'}`
  (the pair named by `_INCLUDED_IDENTITIES`, `test_export_release_holds.py:28`).

## Reconciliation with the suite-base determinism record

The record is not contradicted — it is confirmed, and the apparent disagreement is entirely a
revision difference.

```text
  4c9d1dabb 11:04  ──15 pass──  617fdaca9 15:05 ──15 pass──  b468bfb5c 15:42 ──6 fail──▶ HEAD 19:03
   record's revision             predicate landed             guard closed
   (its empty set is real)       (fail-open still open)       (the six start here)
```

- The record measured 38 passed, 2 skipped, 0 failed over six files on `all_debug` at
  `4c9d1dabb`, with the two files here contributing 12 and 3 passed. Re-measured now, at that
  revision, over those same two files: **15 passed, 0 failed.** Same figure, different host
  class, and the per-file split matches exactly.
- The record's revision is 4.6 hours and 154 commits older than HEAD, so the two measurements
  were never of the same tree. The six do not refute the empty set; the empty set does not
  license a count at HEAD either.
- The empty set's own caveats do not reach these six. The record's host properties are a
  collection-time graph probe in `test_seed_live_graph.py` (skip vs run), the cwd-relative
  credential load, and a heavyweight autouse import (setup timeouts under contention) — none
  establishing the six, and no side reaches the six. Nothing recorded there is needed to
  explain the result.

## What the attribution does and does not establish

- **It is a failure-set boundary, measured at one revision step.** `b161235bb^` and
  `b161235bb` are adjacent, so the revision at which they start is the commit itself rather
  than an interval. For each id the boundary is the same, and each id's own assertion is
  quoted above, so the attribution does not rest on the set boundary alone.
- **It is not a repair and it assigns no fault.** The fixtures build a population that never
  passes the projection at `export.py:699-700`; the guard refuses a candidate carrying no
  producer evidence. Which of the two should change — the fixture population, which would then
  be exercising a projection it currently bypasses, or the guard, whose whole point is the case
  it now refuses — is a decision this record does not take. What is established is that the
  green suite before 15:42 was resting on the guard's own fail-open.
- **The six ids sit in two files, not four.** The followup's summary line says "6 failures
  across four export files"; the ids it names are four in `test_export_exclusion_ledger.py`
  and two in `test_export_release_holds.py`, measured by collecting exactly those two files.
  No wider set was run here, so a count over a larger export surface is not made here.
- **No figure accompanies this record.** The content is a lookup table per id plus a
  revision axis that is naturally a timeline row; the two-file attribution has no spatial
  relationship to plot.

## Provenance

Worktree `.../ship-s10-20260918/n-usb-the-six-red-export-tests-are-attributed`, HEAD
`af53d46713f0e3e4ad81d512df6f6775de972979`. Logs and the run scripts, under
`~/.config/reckon/crew/runs/r-20260918T170428583755-n-usb-the-six-red-export-tests-are-attributed/`:
`login-export-tests.log` (worktree HEAD, login node), `archive-head.log` (calibration),
`record-rev-4c9d1dab.log`, `archive-617fdaca9.log`, `archive-b161235bb-parent.log`,
`archive-b161235bb.log`, `partition-head.log`, `partition-pre-boundary.log`, `run-at-rev.sh`,
`run-on-partition.sh`.

The archive instrument is `git archive <sha> | tar -x` into `/dev/shm`, plus the five
gitignored generated files from the worktree, plus a link to the main checkout's environment
file, run through the shared `.venv` with `PYTHONPATH` pointing at the archived tree. An
archive has no `.git`, so the export's provenance timestamp is unavailable and every export
test dies on `RuntimeError: export provenance timestamp unavailable` — the first attempt
produced five such failures (`test_export_publishes_derived_producer_parent`,
`test_export_withholds_hard_catalog_semantic_issue`, `test_export_keeps_catalog_semantic_advisories`,
`test_quantity_token_duals_pass_identity_collision_gate`,
`test_generated_roles_mark_quantity_parent_and_leaf`) and they are named here so a reader can
tell them from the six. Supplying `SOURCE_DATE_EPOCH` from the revision's own commit time
removes them; the calibrated HEAD run then agrees with the worktree exactly.