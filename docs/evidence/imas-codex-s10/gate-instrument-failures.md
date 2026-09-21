# Gate instrument failures, and a retraction

Recorded 2026-09-21 during the S10 recovery wave.

## The retraction, first

Two ledger outcomes promoted during this wave carry a note asserting that the
standard-names base figure of **20 failed / 3 errors** was a shared-checkout
artifact and that the true isolated figure was **31 / 3**. **That note is wrong
and is retracted here.** Ledger rows are immutable, so the correction lives in
this record rather than in the rows.

**20 failed / 3 errors is correct.** The eleven-test difference is self-inflicted
by the measurement: a `git archive` extraction has no `.git`, and
`_manifest_iso_timestamp` in `imas_codex/standard_names/export.py` resolves a
commit date from the checkout, falls back to `SOURCE_DATE_EPOCH`, and refuses
with *export provenance timestamp unavailable* when it has neither. Every test
driving the public export path then fails for a reason unrelated to the code
under test.

Verified two ways rather than accepted on report:

| check | result |
|---|---|
| `grep -c 'provenance timestamp unavailable'` in the isolated base log | 11 |
| the same in the isolated head log | 11 |
| `.git` present in the gate tree | absent |
| 31 − 20 | 11 |

**The conclusions those rows support are unaffected.** Every comparison in this
wave measured a base and a head through the same setup, so both sides carried
the same eleven. Only the absolute number moved, and it moved back to the figure
the rows already quoted.

**How the error happened is the part worth keeping.** I declined to name a
mechanism for the eleven-test gap — correctly, since I did not have one — and
then drew a confident conclusion about which figure was real. Declining to guess
at a cause is not the same as declining to conclude, and the first does not
license the second.

## The remedy, with its measured cost

```bash
export SOURCE_DATE_EPOCH=$(git -C <repo> show -s --format=%ct <sha>)
```

This clears all eleven and pins provenance deterministically, which is more
correct than depending on a `.git` that should not be in an extraction at all.
It costs exactly one test:

| gate at `f2cb7b21b`, isolated | FAILED | ERROR | provenance refusals |
|---|---|---|---|
| without `SOURCE_DATE_EPOCH` | 31 | 3 | 11 |
| with `SOURCE_DATE_EPOCH` | 21 | 3 | 0 |

The remaining delta is
`test_export_determinism.py::TestManifestDeterminism::test_no_commit_uses_stable_unversioned_timestamp`,
which asserts the behaviour when no stamp is available and therefore fails
*because* the variable is set. So an archive gate reads **+1 against a real
checkout, deterministically and for a stated reason**. Deselect that one id or
record the offset. Never reconcile it by unsetting the variable: that trades one
explained failure for eleven unexplained ones.

## Six instrument failures in one day

<img src="/imas-codex/figures/gate-instrument-failures/gate-pipeline.svg" alt="The gate pipeline with each instrument failure located at the stage it occurred, and what it reported instead of failing" width="100%">

Found across two coordinator sessions working the same repository:

| # | failure | what it reported |
|---|---|---|
| 1 | archive built in `/dev/shm`, which is node-local, so `cd` failed inside the SLURM step | a complete run: 86 failures, then 62 |
| 2 | `PYTHONPATH` named the missing path, so the editable install served the main checkout | a complete run of a tree the gate was built to escape |
| 3 | a peer's prompt file truncated to 0 bytes; `git status` calls that *modified*, never *empty* | 66 added failures, clustered in the files that read it |
| 4 | archive has no `.git`, so export provenance refuses | 31 instead of 20 |
| 5 | the provenance probe called a function without its required argument | **exit 96 — the suite never ran** |
| 6 | log read by grepping counts and tailing; a failed `cd` printed on line 3, and a padded `=` broke a peer's comparison loop | nothing amiss, for two sessions at once |

**Five of the six reported a number. One reported an error, and it is the only
one that announced itself.** That asymmetry is the finding: an instrument that
fails by answering is indistinguishable from one that works, because the shape of
its output is unchanged.

## What the gate does now

Every precondition exits on its own code before the suite starts, so a setup
failure can never present as a test result:

```bash
srun ... bash -lc "cd $G || exit 97
export UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=$ROOT/.venv PYTHONPATH=$G SOURCE_DATE_EPOCH=$SDE
uv run python -c \"
import sys, os, imas_codex
iso = '$G' in imas_codex.__file__
sde = (os.environ.get('SOURCE_DATE_EPOCH') or '').strip()
print('ISOLATION', 'OK' if iso else 'FAILED', imas_codex.__file__)
print('PROVENANCE', 'OK' if sde else 'FAILED', sde or '<unset>')
sys.exit(0 if iso and sde else (98 if not iso else 96))\" || exit \$?
uv run pytest -p no:cacheprovider tests/standard_names/"
```

- **97** the tree is not there · **98** the tree is there but the import resolved
  elsewhere · **96** provenance unavailable · **1** real test failures.
- The probe asserts only that the variable is **set and non-empty**, so the probe
  cannot itself be wrong. The first version of it called
  `_manifest_iso_timestamp()` without `source_commit_sha` and exited 96 — failure
  5 above, caught on its first use.
- Extract to shared storage (`/home/ITER/mcintos/Code/.gate-trees/<sha>`), never
  `/dev/shm` or `/tmp`.
- **Measure the base in the same setup as the head.** A shared-tree base against
  an isolated head is two instruments, not a comparison.
- **Read the head of the log, not only the tail.** Every count below a failed
  `cd` is wrong and the line saying so is three lines from the top.

A cheap standing check for anyone gating from the shared checkout, which needs no
assumption about file type or directory:

```bash
git diff --name-only | while read -r f; do [ -f "$f" ] && [ ! -s "$f" ] && echo "EMPTY: $f"; done
```

## Corroboration

The corrected figure was reproduced independently at the same SHA through a
different extraction path on a different node allocation, giving
`21 / 3 / 7543 passed` against this session's `21 / 3 / 7544 passed` at a head one
commit further on. The same peer established byte-level determinism of the suite:
two pytest sessions over one fixed tree differ only in their tmpdir sequence
numbers — identical ordering, identical per-line output, identical duration. An
earlier claim that the suite's failure count was unstable is not supported; the
readings behind it had been taken through different setups.

## A seventh failure, found by a peer's audit of a different field

Every `--gate-log-path` this session passed to `crew complete` pointed into the
harness scratchpad under `/run/user/<uid>/…`, which is node-local and dies with
the session. The ledger rows therefore cite evidence that will not resolve for
anyone reading them later, including from another login node **now**.

This is the same defect a peer coordinator found from the worker side: a control
log written to `/tmp` that had already been cleaned when they went to verify it,
nearly producing a false report that the node had no evidence. Theirs was a
worker writing evidence to scratch; mine was the coordinator *citing* it. The
field accepted both without complaint.

The cited logs are copied to durable storage at
`~/.local/share/imas-codex/gate-logs/ship-s10-20260918/` — 900 KB, eleven files,
including both isolated base/head pairs, the two contaminated runs kept as
counter-evidence, and the hand-built negative control.

**The rule this earns:** a path field that accepts a location under `/tmp`,
`/dev/shm` or `/run/user` is accepting evidence with a few hours of shelf life.
Write gate and control logs where the run record lives, or copy them there before
citing them. A promote that cites a path nobody else can open has recorded the
*claim* without the evidence — the same shape as the git note that never leaves
one clone.
