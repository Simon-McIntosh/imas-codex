# The six-file failure base is reproducible at one revision, and the couplings that would break that

The followup records three measurements of these six files — 5, 19 and 27 failures —
and asks whether they share live state, ordering or filesystem coupling. This is the
measurement that answers it for one revision, one tree and one host class.

**Result.** Three independent runs at `4c9d1dabb4ab0afb679ccf15b85eb552e585169c`
produced the **same failure-id set**, and that set is **empty**: 38 passed, 2 skipped,
0 failed, three times. A fourth run through the identical invocation form, carrying one
planted failing test, reported exactly one FAILED id and exit 1 — so the empty sets are
clean runs, not a blind instrument.

| Run | Form | Collected | Result | Failure ids | Exit |
|---|---|---|---|---|---|
| A | one invocation, forward argument order | 40 | 38 passed, 2 skipped (21.16 s) | `{}` | 0 |
| B | one invocation, reverse argument order (execution order verified reversed) | 40 | 38 passed, 2 skipped (20.99 s) | `{}` | 0 |
| C | one invocation per file, six processes, union taken afterwards | 40 | 38 passed, 2 skipped | `{}` | 0 |
| Control | form A plus one planted failing test outside the tree | 41 | 1 failed, 38 passed, 2 skipped (24.46 s) | `{test_planted_failure}` | 1 |

Run C in detail: `test_catalog_layout_hierarchy.py` 20 passed; `test_export_exclusion_ledger.py`
12 passed; `test_export_release_holds.py` 3 passed; `test_export_bound_adjacent.py` 2 passed;
`test_export_resolution_method.py` 1 passed; `test_seed_live_graph.py` 2 skipped
(`DD-loaded graph not available`, lines 37 and 64). 20+12+3+2+1 = 38; +2 skips = 40 collected.

## Why the empty set is a measurement and not a blind instrument

A zero needs the check shown to see something known present. Two receipts:

1. **Collection is seeing known-present items.** 40 items were collected in runs A/B/C,
   and the per-file counts in run C match the inventory of the six files exactly. A run
   that collects nothing cannot report `[100%]` or a totals line; these runs carry both,
   and the totals line is the completion marker. (The `-qq` trap — a second `-q` beyond
   the project's own suppressing the totals line, so a partial failure list looks whole —
   is deliberately avoided: no `-q` was passed anywhere.)
2. **The reporter surfaces failures.** The control run added one file with a single
   failing test, invoked through the same command, the same configuration file, the same
   environment and the same working directory. It reported the FAILED id with its file and
   assertion text, and the process exit status moved from 0 to 1. The harness that
   reported `{}` for runs A/B/C is therefore able to report a non-empty set.

## Where the base is a property of the run, not the revision

The three runs agree today. These are the couplings that decide whether they would still
agree tomorrow, or on another host, each with its mechanism and location.

### 1. Live-graph reachability, decided at collection time (measured on two hosts)

`tests/standard_names/test_seed_live_graph.py:20,23-34` runs a module-level probe,
`_has_dd_content()`, whose result is bound into the `skipif` at `:37` and `:64`. That
probe constructs a `GraphClient` and runs one query at **collection** time.

- On the debug partition: the probe returned **False** → both tests skipped, reason
  `DD-loaded graph not available` (runs A, B and C).
- On the login node, where the graph is reachable: the same probe returned **True**,
  with the underlying query returning `[{'c': 61366}]` — 61,366 IMASNode rows. The same
  two tests therefore do not skip there; they execute.

```text
                 collection of test_seed_live_graph.py
                              |
                   _has_dd_content()  (one query)
                    /                       \
          False (no route)                True (route up)
                |                             |
     2 tests SKIP, reason given        2 tests RUN against live content
     (debug partition, this run)      (login node: True, 61366 rows)
```

So a failure count for these six files is a **host property** before it is a revision
property. The message collapses three distinct causes — no route, no credential, and a
reachable-but-empty graph — into one string, so the receipt does not show which of them
applied; what the skip does show is that the probe's own answer was False.

Two further facts about that probe: it is **uncapped** (no timeout around a graph
connection attempted at import) and, on the login-node route, it is **slow** — 205.22 s
for the single query that returned True. A collection that reaches that path pays it
before the first test runs.

### 2. The graph credential and the working directory

`tests/conftest.py:15-24` loads the environment with `load_dotenv(dotenv_path=None, override=False)`,
which is **cwd-relative**; `:96-121` turns that into a credential verdict, and `:201-229`
turns the verdict into skip markers on `graph`/`integration`/`requires_graph` items. A
checkout or worktree without the `.env` link therefore changes skip/fail behaviour for
those items. This is the same class of coupling as (1): the run's inputs include a file
outside the tree.

### 3. An external editable checkout supplies the grammar context

`tests/standard_names/conftest.py:128-172` caches an ISN `get_grammar_context()` for the
session, reading an **editable checkout outside this repository**
(`/home/ITER/mcintos/Code/imas-standard-names`, receipt below). On exception it warns and
skips memoization — grammar-dependent tests then fail **individually** rather than the
session refusing. The base for these six files is therefore also a function of a path
that is not in the repository and not in the revision.

### 4. An ordering coupling exists in the package, outside these six files

`tests/standard_names/test_release_import_locality.py` deletes and reloads
`imas_codex.standard_names.*` entries in `sys.modules`; it is the documented cause of 100
of the 118 failures in the completed-suite measurement. Any file that holds a module-level
reference into those modules can be affected by running after it, and vice versa. Runs A
and B are the ordering control for the six files themselves — reversed execution order,
identical outcome — so this coupling is live in the package but does not reach these six
files today.

### 5. Regenerated models are a different instrument (candidate, not reproduced)

The build hook regenerates `imas_codex/graph/models.py`, `dd_models.py`,
`config/models.py` and `agents/schema-reference.md` at sync time. Two trees with
different generated files are two different measurement instruments, which is the one
structural difference recorded between the workers whose results disagreed. This node did
not reproduce it: doing so needs a second provisioned tree, which is outside its write
scope. It is reported as a candidate, and the number it would explain (the extra 14) is
not reproducible at this revision in this environment.

## A second host property, reported by a peer after this was written

This result establishes that the base is reproducible **at one revision, on one host
class, under the load condition these runs saw**. A peer coordinator measured a second
property of the same tier that the three runs above could not have seen, and it is
recorded here rather than left in a message.

`tests/standard_names/conftest.py:54`, the autouse fixture `_bound_synthetic_model_exposure`,
imports `imas_codex.discovery.base.llm` and through it litellm, openai and a pydantic model
build. Every test in the tier pays that import in **setup**. Under filesystem contention it
exceeds the repository's own 30 s per-test timeout and raises
`Failed: Timeout (>30.0s) from pytest-timeout` before any test body runs. Measured the same
day: `import imas_codex.discovery.base.facility` took **84.2 s wall against 3.3 s user CPU**
— I/O wait on a loaded GPFS, not compute. The attribution control is what makes it
environmental rather than a change: the same fixture errored identically on a module the
measuring node did not touch. At `--timeout=600` four of five tests passed and only the
expensive-import setup still failed, and in that same run an unmodified test failed on
`subprocess.TimeoutExpired` for `git branch --show-current`.

So there are now **two** host properties in this tier with different signatures. The one
measured above changes which tests *skip*; this one changes which tests *error in setup*.
Neither is flakiness and both are invisible in a bare failure count.

**The consequence is one line longer than this record originally carried.** A suite figure
from this tier needs its **load condition** stated alongside its host class, or the delta it
supports is not attributable. The three runs above were taken on `all_debug` in one window
and agreed; that agreement is evidence about that window, and a base taken during contention
is not the same base. The reproducibility established here is real and it is conditional,
and the condition is now named.

An autouse fixture doing a heavyweight import is a design question rather than a timeout to
raise, and it is not repaired here.

## Provenance

Revision `4c9d1dabb4ab0afb679ccf15b85eb552e585169c`, worktree
`.../ship-s10-20260918/n-sli-the-suite-base-is-reproducible-or-it-is-not`, all runs on
`all_debug` (jobs 1273462, 1273498), logs under
`~/.config/reckon/crew/runs/r-20260918T090616383064-n-sli-the-suite-base-is-reproducible-or-it-is-not/suite-base/`:
`run-a-joint-forward.log`, `run-b-joint-reverse.log`, `run-c-*.log`, `instrument-check.log 2>&1`,
`login-graph-coupling.log`, `orientation.log`.

The orientation receipt pins what "the revision under test" means here: `pytest 9.0.3`;
`imas_codex` served from the worktree; `python` = the main checkout's shared `.venv`;
`imas_standard_names` resolved to `/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/__init__.py`
— the external checkout of coupling 3.

Two driver defects are recorded rather than hidden, because they are visible in the log
directory: two redirects were written as `> "$LOG/status-before.txt 2>&1"` instead of
`> "$LOG/status-before.txt" 2>&1`, so the files are literally named `status-before.txt 2>&1`
and `instrument-check.log 2>&1`; the contents are correct and complete — only the names
were affected.

The login-node probe is a deliberate, bounded use of the login-node exception that work
needing a login-local endpoint carries: one read-only statement over a named label, run
once, after which the probe stopped; nothing further was issued to the graph. The single
query exceeded the ten-second ceiling (205.22 s) because establishing the route is part of
the cost — reported here rather than pressed further. The two tests that the probe gates
were **not** executed there: they seed live graph state, which is outside this node's
write scope, so what is measured is the gate, not their verdict.