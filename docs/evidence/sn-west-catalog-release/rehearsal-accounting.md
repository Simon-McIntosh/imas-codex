# A rehearsal prints the accounting it exists to rehearse

## What was asked

One rehearsal invocation should report the candidate count, the published count
and every exclusion with its mechanism, while writing nothing: afterwards
`git status --porcelain` returns zero lines, no roster, staging or candidate
artifact is created, and no tag is written.

## Why the rehearsal could not report it before

The dry-run branch returned before the export leg. That return is what makes a
rehearsal inert, and it is also what left an operator unable to ask the rehearsal
what a cut would publish and what it would drop — the candidate count, the
published count and the per-mechanism exclusions are all produced by the export
leg. Measuring them required driving that call rather than reading the rehearsal's
own output.

## What changed

`imas_codex/standard_names/catalog_release.py`:

- `ReviewReleaseReport` carries `candidate_count`, `published_count`,
  `exclusion_counts`, `accounting_residue` and `accounting_error`, all surfaced
  through `to_dict()`.
- `_record_export_accounting` reads the export leg's own returned report —
  `total_candidates`, `exported_count`, `exclusion_counts`, `exclusion_records` —
  and logs the candidate count, the published count, every exclusion beside its
  mechanism, and the residue. It reads the report the leg returns rather than
  recomputing a tally, so the rehearsal and the cut cannot disagree.
- `_rehearse_export_accounting` drives the production export leg into a
  `tempfile.TemporaryDirectory` that is removed before the rehearsal returns, so
  the accounting appears while the release staging directory, the frozen roster,
  the review branch and the candidate counter all stay untouched.
- The real export leg calls the same recorder, so both paths share one source.

The leg is driven on a rehearsal only when no exporter was injected: a caller
that installs its own exporter owns what that leg reports, and there is nothing
to measure on its behalf.

## The recorded invocation

Run once, on 2026-09-18, from the node's worktree:

```
$ uv run --no-sync imas-codex sn release --batch west_production_dd_paths --dry-run

Standard-Name Review Batch
  ISNC: /home/ITER/mcintos/Code/imas-standard-names-catalog
  Batch: .../imas_codex/standard_names/manifests/west_production_dd_paths.yaml
  PR target: fork
  Mode: dry run

Errors: 1
  - ISNC not on main branch (current: rehearsal/fold-back-20260917). Switch
first: cd /home/ITER/mcintos/Code/imas-standard-names-catalog && git checkout
main
EXIT=1
```

The rehearsal refused at its pre-flight, before the export leg, so it produced no
accounting. The cause is outside this node's write scope and is not a defect in
the change: the catalog checkout at `/home/ITER/mcintos/Code/imas-standard-names-catalog`
sits on a peer's branch `rehearsal/fold-back-20260917`, and the release path
refuses to rehearse a cut from anywhere but `main`. Switching that checkout's
branch belongs to whoever is working there.

The figures the invocation would have printed are emitted by
`_record_export_accounting` as `logger.info` records — this repository's `sn` CLI
auto-logs full DEBUG output, which is its recorded-output surface — and the
machine-readable equivalents are the `candidate_count`, `published_count`,
`exclusion_counts` and `accounting_residue` keys on `ReviewReleaseReport.to_dict()`.
No figure is recorded here, because none was produced.

## The three tree checks, taken after that invocation

Every check below was taken after the run above. Each instrument is shown reading
a populated state, so an untouched reading means untouched rather than unreadable.

### Check 1 — the worktree is clean

```
$ git -C <worktree> status --porcelain
(lines: 0)
```

### Check 2 — no roster, staging or candidate artifact was created

```
$ ls -la /home/ITER/mcintos/.cache/imas-codex/staging
total 321
drwxr-xr-x. 2 mcintos mcintos   4096 Sep 15 21:22 .
drwxr-xr-x. 5 mcintos mcintos   4096 Sep 14 17:36 ..
-rw-r--r--. 1 mcintos mcintos 173564 Sep 11 10:56 catalog.yml
-rw-r--r--. 1 mcintos mcintos 146828 Sep 15 21:23 .export_report.json

$ find /home/ITER/mcintos/.cache/imas-codex/staging -name '*.sn_names.yaml' | wc -l
0

$ find /home/ITER/mcintos/Code/imas-standard-names-catalog \
      -name '*.sn_names.yaml' -newermt '2026-09-18 12:30'
(no output)
```

The staging directory is not empty — it holds a catalog written 2026-09-11 and an
export report written 2026-09-15 — so the emptiness check is reading a directory
that demonstrably carries artifacts. Neither carries anything written by this
run. The newest file in staging predates the invocation by three days, and no
frozen roster or candidate artifact was created anywhere in the catalog checkout.

### Check 3 — no tag was written

```
$ git -C /home/ITER/mcintos/Code/imas-standard-names-catalog \
      for-each-ref --sort=-creatordate \
      --format='%(refname:short) %(creatordate:iso-strict)' refs/tags | head -3
v0.4.0rc7+west-task-2e 2026-09-11T10:56:22+02:00
v0.4.0rc6+west-task-2e 2026-09-08T12:48:37+02:00
v0.4.0rc5+west-task-2e 2026-09-07T12:29:58+02:00
```

The catalog carries 30-plus release tags, so the tag check is reading a populated
ref namespace. The newest is `v0.4.0rc7+west-task-2e`, cut 2026-09-11 — a week
before this run — and the rehearsal added none.

## The test that holds both halves

`tests/standard_names/test_release_rehearsal_accounting.py` (new, five tests)
pins both halves. The accounting half is driven through the production exporter
seam, with the leg's own report supplying the counts; the inert half is asserted
against a checkout that already carries a roster, a review branch and a tag, so
each emptiness check has a populated reading it could be wrong about. Two further
tests hold the residue reporting (a candidate neither published nor excluded is
reported, not absorbed into the arithmetic) and the unreachable-leg path (a
rehearsal whose accounting cannot be measured still reports the roster and branch
it would cut, and still writes nothing).

### Run and result

Log file: `/tmp/s14-rehearsal/accounting.log`

```
$ UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD \
    uv run --no-sync pytest -p no:cacheprovider \
    tests/standard_names/test_release_rehearsal_accounting.py \
    tests/standard_names/test_release_rehearsal_writes_nothing.py

9 passed, 1 error in 138.09s
```

The single error is in **setup**, before any test body:

```
tests/standard_names/conftest.py:54  _bound_synthetic_model_exposure
  -> imas_codex.discovery.base.llm -> litellm -> openai -> pydantic model build
E   Failed: Timeout (>30.0s) from pytest-timeout
```

It is an environment cost, not the change, and it was confirmed as such in this
tree:

- `import imas_codex.discovery.base.facility` measured **84.2 s**, against 84
  seconds wall and 3.3 s user CPU — I/O wait on a loaded filesystem, not compute.
- The same fixture errors identically on
  `test_release_rehearsal_writes_nothing.py`, a module this node does not modify.
- Raising the per-test timeout for one run moved the count to 9 passed, with the
  same single import-bound setup failing, and the four tests that exercise the
  accounting recorder and the inert half passing either way.

So four of the five tests pass, and the fifth cannot reach its body while the
autouse fixture's import exceeds the repository's 30 s per-test timeout. That
fixture is outside this node's scope and is reported separately.

## What is not covered

- No real rehearsal produced the three figures, for the reason recorded above:
  the catalog checkout is on a peer's branch. The figures exist only as the
  programmed output of `_record_export_accounting`.
- A full end-to-end run of the release path against the live graph was therefore
  not exercised; the accounting recorder is covered by the four passing tests,
  which drive the production exporter seam rather than a live export.