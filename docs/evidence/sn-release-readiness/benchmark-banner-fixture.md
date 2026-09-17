# The benchmark banner names the fixture it reports

Plan `imas-codex:sn-release-readiness`, followup
`f-srr-the-bench-banner-reports-the-wrong-dataset`, which asked that the
displayed reference total be derived from the same branch the extractor takes
and that the fixture path be printed beside the count.

Two committed fixtures back `sn bench`. The curated reference dataset
(`imas_codex/standard_names/benchmark_reference.py`, 47 paths) is used in
default mode; the physics hard-case set (`research/physics_bench_paths.json`,
15 paths) is used under `--physics`. The banner reported the curated set's size
in both, so a physics run's header named a population the run did not use.

## The defect, reproduced before the change

`banner-before.log` in run `r-20260917T200845366757` drives the command in each
mode and reports, beside the banner line, the paths the extractor was handed
through a recording stand-in for `extract_dd_candidates`:

| Mode | Banner line | Paths the extractor read | Fixture actually used |
|---|---|---:|---|
| default | `Reference paths: 47/47` | 47 | `imas_codex/standard_names/benchmark_reference.py` |
| physics | `Reference paths: 47/47` | **15** | `research/physics_bench_paths.json` |

This is the recorded defect exactly as the followup stated it: physics mode
printed 47 against 15 extracted. No graph, no model call and no Data Dictionary
read was needed to see it — the two figures come from the fixture files and from
the paths the extractor is handed.

## The change

`imas_codex/cli/sn.py`, in one commit:

- `_benchmark_fixture(physics)` (line 3210) resolves the fixture the extractor
  will read and returns its repo-relative path beside its paths. It takes the
  same branch the extractor takes — `load_bench_paths()` under `--physics`,
  `REFERENCE_NAMES` otherwise — so the banner and the extractor cannot name
  different populations without the test below failing.
- The banner body (line 3510) calls it in place of the unconditional
  `total_ref = len(REFERENCE_NAMES)`, and the printed line (line 3549) carries
  the fixture path beside the count.
- Two stale figures that named a population in prose, and were wrong in the
  same way: the `--max-candidates` help said `default: all 54` (54 is
  `BenchmarkConfig.max_candidates`, a cap, never a set size) and the command
  docstring called the reference dataset "54 curated DD paths".

After the change, from `banner-after.log`:

| Mode | Banner line | Paths the extractor read |
|---|---|---|
| default | `Reference paths: 47/47  (imas_codex/standard_names/benchmark_reference.py)` | 47 |
| physics | `Reference paths: 15/15  (research/physics_bench_paths.json)` | 15 |

## The test

`tests/standard_names/test_benchmark_banner.py::test_banner_names_the_fixture_the_extractor_reads_in_each_mode`
drives the banner through the CLI in both modes and, in the same test, hands the
config the banner built to the real `_extract_candidates`, whose extractor is
replaced by a recording stand-in. The assertions are:

- the paths the extractor was handed are exactly the fixture file's contents,
  read from disk independently of the code under test, so a size match by
  coincidence cannot pass;
- the banner's total equals that fixture's size;
- the banner's shown figure equals its total with the cap lifted to 1000, so the
  comparison is of fixture sizes rather than of the cap;
- the banner line names the fixture's repo-relative path.

The fixture for each mode is named in the test's own table
(`FIXTURES`), not read back from the production helper, so the test states
which file backs each mode independently of the implementation it checks.

## Gate

Run on the `all_debug` partition (`/tmp` as `TMPDIR`), one log per run, in the
run directory:

| Run | Command | Result |
|---|---|---|
| this worktree | `pytest tests/standard_names/test_benchmark_banner.py tests/standard_names/test_benchmark.py` | **83 passed**, exit status 0 |
| baseline, main checkout at `86976450d` | `pytest tests/standard_names/test_benchmark.py` | 82 passed, exit status 0 |

Zero added failures against the baseline; the one added test is the node's own.
Before the change the same test file failed (`1 failed`, exit status 1) on the
before/after order's first assertion — the default-mode line names no fixture.
The physics-mode figure the test asserts is the 47-against-15 pair quoted in
`banner-before.log` above.

## Controls

- The extractor stand-in records what it was handed and the test asserts the
  record is non-empty, so a banner checked while the extractor was never
  reached cannot pass. The first version of this test failed on exactly that,
  which is how the arrangement was found: the extractor runs inside
  `run_benchmark`, not in the command, so driving the CLI alone cannot observe
  it.
- Default mode passes both before and after the change on the size assertions,
  which is what makes the physics-mode failure a difference between the two
  branches rather than a broken instrument.
- The paths of both fixtures are read from their files, so a renamed fixture
  fails the test rather than silently changing the label.

## What was not done

- **`benchmark.py` was not touched.** The report's `reference_names` field is
  still populated from `REFERENCE_NAMES` unconditionally
  (`imas_codex/standard_names/benchmark.py:999`, stored in the report at 1005),
  so a physics-mode report saved to disk names the curated set while the run
  extracted the hard-case set. That is the same mislabelled population one layer
  down, in the artifact rather than the banner, and it is outside this node's
  write scope. Recorded as a follow-on.
- **No behaviour change to extraction, scoring or the fixture contents.** Only
  the displayed total, the label beside it, and the two prose figures were
  changed.
- **Verification of the merged result belongs to a separately dispatched test
  node**, not to this one.