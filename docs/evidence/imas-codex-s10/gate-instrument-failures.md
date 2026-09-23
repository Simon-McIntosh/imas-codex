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

## An eighth and a ninth, and two adopted improvements

**Eighth: a cold archive tree can blow the per-test timeout and report a class
that is neither pass nor fail.** A peer reproducing a control against a freshly
extracted tree got **exit 3** and nearly read it as a red. The cause was
`Failed: Timeout (>30.0s) from pytest-timeout` on a cold GPFS tree, which pytest
reported as `INTERNALERROR` **with no totals line at all**. Re-run at
`--timeout=180` the same command gave exit 1 and 2 failed, which was the real
control. So the first run against any newly extracted tree pays cold-cache cost
against a 30-second per-test ceiling, and the failure presents as an exit class
rather than as a test result.

Audited across this session's nine cited gate logs: **zero timeouts, zero
`INTERNALERROR`, and every pytest log carries its totals line**, so none of the
measurements here are affected. The hazard is real for the next cold tree.

**Ninth: a gate can finish without summarising, and counting failures cannot
tell.** Both of the checks this session used — `grep -cE '^FAILED '` and
`grep -cE '^ERROR tests/'` — return a perfectly good `0` from a log that has no
totals line because the run died. The standing assertion that catches it costs
one line and is now part of the recipe:

```bash
grep -cE "^=+ .*(passed|failed)" "$LOG"    # must be 1; 0 means the run never summarised
```

Anchor on `^=+` rather than a fixed run of `=`: pytest pads the banner to
terminal width, so a literal `=====` can miss a line that reads `= 31 failed, …`.
A peer's comparison loop silently matched nothing for twelve and thirty-four
minutes for exactly that reason.

**Adopted: reproduce a control against committed history, not by mutating.**
Where a predecessor commit already contains the behaviour a control is meant to
demonstrate, checking out that revision is strictly better evidence than either
a worker's self-report or a mutation applied by hand. It needs no weakening of
the thing under test — and a hand-applied narrowing of a live guard is the kind
of edit a safety classifier will decline, correctly.

This session has one datum on the other side, which is why "strictly better" is
worth taking seriously. The one hand-built control here — adding a seventh
classifier branch returning a string outside the declared vocabulary — returned
**exit 0**. The mutation was applied correctly and the test passed, because the
test never supplies the input that reaches a newly added branch. That was a true
finding about the test's aperture rather than a failed control, but it cost two
attempts, the first of which returned exit 2 from a malformed edit. A checkout
of committed history has neither failure mode.

## A third place evidence goes to die: relative paths inside a reclaimable worktree

A worktree audit over this session's 25 trees found **zero holding unintegrated
commits** — the fleet is clean — but two holding untracked directories, and one of
those is cited evidence.

`n-usb-the-producer-predicate-fails-closed` records its evidence in the manifest at
**relative paths inside its own worktree**: `logs/gate_fail_closed.log`,
`logs/gate_reverted_fails_closed.log`, `logs/surface_base.log` and four more. Two
properties make that unreadable later, and neither is visible from the manifest:

1. The paths resolve only from inside that worktree, so any reader elsewhere — a
   later session, another login node, the ledger — cannot open them.
2. The worktree is reclaimable. `crew gc` takes it as soon as its state reads
   integrated, which it now does, and the evidence goes with it.

`gate_reverted_fails_closed.log` is the **negative control for a fail-closed
guard** — precisely the artifact a later reader would want, because it is the only
thing showing the guard fires when reverted. The second tree held the script that
produced a published evidence document's figures.

Both preserved to `~/.local/share/imas-codex/gate-logs/<session>/<node>/`, 998 KB
in total for the session.

**Three variants of one defect now, all found in a single day:** a worker writing
evidence to `/tmp` that was cleaned before verification; a coordinator citing
`/run/user` scratch that dies with the session; and a worker citing a relative
path inside a tree that gc will reclaim. In every case the manifest field accepted
the path without complaint and the promote succeeded. **The field validates that a
string was supplied, not that anyone else can open it** — which is the same defect
as the git note that never leaves one clone, and the same shape as every other
instrument in this record: it answers rather than erring.

The check that would catch all three at promote time is a stat plus a location
test: reject a cited path that does not exist, that resolves under `/tmp`,
`/dev/shm` or `/run/user`, or that is relative to a worktree rather than to the
run record.

## How narrow the aperture actually was

The three iterations above argued about whether the guarded surface could shrink
without a test failing. A separate measurement, taken while the fourth attempt was
in flight, says the more important thing: **the surface was already about a tenth
of the tree.**

Counted over `imas_codex` with the project interpreter, matching the guard's own
predicate shape (a three-argument `getattr` whose attribute name is a string
literal):

| | modules | literal-name candidates |
|---|---|---|
| the guard's enumerated surface (its ratchet floors) | 11 | 30 |
| the package | **55** | **321** |

So the check was inspecting under 10% of the calls it exists to inspect, and the
ratchet floors — the thing that made a shrink fail — recorded that narrow aperture
as the standard to hold. A floor set to a measured value protects the measurement;
it says nothing about whether the measurement covered the right surface. **Both
facts were true at once: the aperture could not shrink, and it was already small.**

The declaration side was never the problem — `_declared_class_attributes` already
walks the whole package for fields and properties. Only the *candidate* side was
narrowed by the root enumeration.

**Deriving the surface from the tree therefore has an exposure, and it should have
been measured before the node was dispatched rather than discovered by it.** A
crude package-wide predicate reports on the order of 80 candidates whose attribute
is declared on no in-package class, concentrated in `standard_names/workers.py`
(20), `graph/schema.py` (11), `discovery/base/llm.py` (7), `standard_names/audits.py`
(6) and `standard_names/review/pipeline.py` (6). The sample's character is mostly
legitimate: attributes on third-party receivers — litellm usage fields
(`prompt_tokens_details`, `cached_tokens`, `_hidden_params`), rich live-display
internals (`_live`, `is_started`), logging record attributes (`worker_name`,
`batch`). The existing exemption mechanism,
`LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES`, holds **two** entries, each carrying a
reason.

That figure is an upper bound from a cruder predicate than the guard's and is
recorded as such. It is enough to establish the shape of the work: widening the
surface is a fail-open closure whose exposure is the node's scope, not a
tidying-up. The rule this repository already carries — measure the exposure in a
scratch copy before writing the change, and put the number in the brief — was not
followed here.

## Correction: for the defect class it exists to catch, coverage was 0 of 30

The section above reported the guarded surface as 11 modules and 30 candidates
against 55 and 321 in the package, and framed that as "under 10% of the calls it
exists to inspect". Directionally right, materially understated, and the sharper
measurement reverses part of it.

**The guard's defect class is not every literal-name three-argument `getattr`. It
is the one whose default is an empty container** — the shape that dropped every
Layer-1 audit finding in this sprint's worst defect, `getattr(audit_report,
"findings", [])` against a report with no such field. Counted package-wide with the
project interpreter, that class holds exactly **30 calls across 9 modules**. And:

| | |
|---|---|
| empty-container candidates **inside** the three declared roots | **0** |
| **outside** them | **30** |

So for the shape the guard was built for, coverage was **zero**. The 30 the ratchet
floor names are literal-name candidates of *any* default within the roots; the 30 in
the defect class are elsewhere entirely. **Two different thirties, matched by
coincidence** — a worker's census read the match as an identity and concluded the
aperture was already complete, which is the opposite of the truth. Caught by
recomputing coverage rather than by re-reading the claim.

Where they are, and why they are not 30 bugs:

- **24 of the 30** are `getattr(..., 'operators' | 'args' | 'qualifiers', <empty>)`
  in `standard_names/audits.py` (12), `workers.py` (6) and `graph_ops.py` (6) —
  reads on ISN grammar objects. All three names **are** declared, in the
  `imas_standard_names` package, verified by walking its classes. They are
  legitimate, and a package-wide widening flags them only because
  `_declared_class_attributes` walks `imas_codex` alone.
- **6 residual sites** in `discovery/base/engine.py`, `discovery/base/progress.py`,
  `graph/schema.py`, `services/response.py`, `standard_names/benchmark.py` and
  `standard_names/fanout/runners.py` need classifying one at a time.

**So the fix has two halves and the coordinator's first design had only one.**
Widening the *candidate* surface without widening the *declaration* surface
produces 49 findings that are nearly all false, which is exactly what the refusing
node measured. The revised design narrows the widening to the defect class and adds
the ISN package to the declaration collector.

**The refusal was the right outcome and is worth recording as such.** The node
reproduced the incompatibility end to end, restored the file byte-for-byte with a
verified digest, left the worktree clean, and reported a blocker rather than adding
exemptions to force a green gate. A worker that had forced green would have shipped
a guard reporting 49 false findings, and a reviewer would then have been arguing
about exemption lists instead of about coverage.

## Correction: the declaration surface never needed widening, and the residual is one site

The section above concluded that widening the candidate surface without widening the
declaration surface produces mostly false findings, and prescribed adding the
`imas_standard_names` package to the declaration collector. **The prescription was
wrong and is withdrawn.**

Measured by the worker's classifier and verified independently:

```
RESOLVED_WITH_CODEX_ALONE        29
RESOLVED_ONLY_BECAUSE_OF_ISN      0
TOTAL                            30
```

`operators`, `args` and `qualifiers` are declared by classes in **`imas_codex`
itself**, not only in the standard-names package. So the ISN change resolves nothing.

**The reasoning error is worth naming because it is the day's most common shape in a
new costume.** I checked whether ISN declares those attributes — it does, verified by
walking its classes — and concluded they resolved *only* through ISN. That does not
follow. I established a sufficient condition and read it as a necessary one, and never
ran the one check that would have separated them: whether `imas_codex` declares them
too. One extra query, never made, and a whole design half built on its absence.

**What the measurement actually leaves is a single site.** Of the thirty
empty-container candidates, twenty-nine resolve against the existing collector. The
one that does not is `imas_codex/graph/schema.py:406`:

```python
annotations = getattr(slot, "annotations", {}) or {}
```

`slot` is a LinkML slot definition — a third-party type — so this is a legitimate
foreign-object read and wants one exemption entry naming the owning library, not a
repair.

**So the node that looked like a fail-open closure with a 49-finding exposure is a
one-line exemption plus a tree-derived coverage requirement.** The three figures this
surface produced, in order — 49 findings, then about six residual sites, then one —
were each measured honestly and each rested on a differently-drawn population. Every
correction shrank it, which is the signature of a denominator nobody had stated.

## A resumed worker rewrote its manifest instead of appending to it

Recorded because the recovery was luck rather than design. The resumed turn replaced a
full manifest — commits, tests, `evidence_inputs`, a three-path blocker, `follow_ons` —
with a **two-line stub**. The blocker analysis that justified redesigning this node was
lost from the record.

It survived only because the worker had also written its probe scripts and logs into
the run directory: `census.py`, `probe_wide.py`, `probe_empties.py`, `probe_isn.py`,
`widened_probe.log`, `base_full.log`, and an `orig_blob.sha256` proving the byte-for-byte
restore. Re-running `probe_isn.py` recovered the figure the manifest had dropped — and
that figure is what produced the correction above.

The manifest rule the fleet already carries is *write it incrementally so a partial
result is recoverable*. This is its missing half: **append, never rewrite.** An
incrementally-written manifest that a later turn truncates is no safer than one composed
at the end.

## Two whole-package AST walks sit inside 2 s of the per-test timeout

**A gate on `tests/standard_names/` can report a phantom failure under cluster
contention, and the mechanism is a timeout rather than an assertion.** Measured
2026-09-23 while taking a base reading for an unrelated exposure measurement: the
surface came back at 22 failed against a merged-head reading of 21 taken hours
earlier, with nothing but documentation and plan state committed in between.

The extra failure was not a regression:

```
FAILED tests/standard_names/test_cost_ledger_augments_only.py::test_no_production_statement_deletes_an_llm_cost_node
E   Failed: Timeout (>30.0s) from pytest-timeout.
```

Its traceback ends inside `ast.parse(path.read_text())`. The durations table
explains the rest:

| test | duration | limit |
|---|---|---|
| `test_no_production_statement_deletes_an_llm_cost_node` | **30.05 s** | 30 s |
| `test_declared_attribute_access.py::test_three_argument_getattr_names_a_declared_attribute` | **28.32 s** | 30 s |
| next slowest on the surface | 7.88 s | 30 s |

Both walk the whole package with `rglob("*.py")` and `ast.parse` every file —
`_llm_cost_deletions(package)` in the first, and the tree-derived aperture in the
second. Nothing else on the surface is within 20 s of the limit, so the boundary
is occupied by exactly these two and the gap to third place is four-fold.

**Half of this is self-inflicted.** The declared-attribute test is one this
session authored, and its aperture is deliberately derived by walking the tree
rather than from a hand-maintained list — which is the right design and is also
what puts it 1.7 s from the timeout. A latent flake was introduced along with a
real improvement, and only a loaded node revealed it.

**Why it matters more than one flaky test.** Every conclusion in this document
rests on comparing failure-id sets between two revisions. A load-dependent
timeout injects an id into one set and not the other, and it presents exactly as
a regression: a `FAILED` line naming a real test, in a file the change plausibly
touched. The correct reading requires opening the traceback to see
`Failed: Timeout` rather than an assertion — a distinction a count cannot carry
and a `grep ^FAILED` deliberately discards.

**So a failure-set diff needs one more discrimination than it has been getting:**
separate timeout failures from assertion failures before attributing either.

```bash
grep -E "^FAILED " "$LOG" | sort -u > ids
grep -B1 "Failed: Timeout" "$LOG" | grep -oE "^_+ [^_]+ _+$"   # which ids are timeouts
```

The durable fix is to stop parsing the package twice per run — the two tests
could share one cached parse — or to raise the limit for these two explicitly via
`@pytest.mark.timeout`. Either is preferable to a gate whose verdict depends on
what else the cluster is doing, because a phantom failure spends a coordinator's
attention on code that never changed. This one cost a diff, a traceback read and
a retraction of the suspicion that documentation commits had moved a test.

### Correction to the section above: the second test was never near the limit

**The claim that `test_declared_attribute_access` sits 1.7 s from a 30 s timeout is
wrong, and so is the sentence attributing the flake half to it.** A peer session
checked the file and found what this session did not:

```python
tests/standard_names/test_declared_attribute_access.py:443
@pytest.mark.timeout(300)
def test_three_argument_getattr_names_a_declared_attribute() -> None:
```

The marker is on exactly the whole-package walk — the one call to
`assert_declared_attribute_access()` with no root. Its 28.32 s runs against a
**300 s** ceiling, not the 30 s default, so it is nowhere near timing out. The
other eight tests in that file take `tmp_path` and scan scratch trees.

The correction is appended rather than applied in place, because the wrong reading
was published and a reader deserves to see what the record said. Two sentences
above are retracted: *"Both walk the whole package … and both sit at 28–30 s
against a 30 s per-test timeout"* — only one does — and *"Half of this is
self-inflicted … the improvement and the flake arrived together"*, which
attributed a phantom failure to the only test on this surface that declares its
own cost. The self-blame was not humility, it was an unchecked assumption; the
file was never opened.

**The real exposure is larger than the one first recorded.** Censused on
2026-09-23 over `tests/standard_names/`:

| | files |
|---|---|
| use `ast.parse` | **16** |
| carry a `pytest.mark.timeout` | **1** |
| carry none, running under the 30 s default | **15** |

The single marked file is `test_declared_attribute_access.py`. The timeout that
prompted this whole section came from `test_cost_ledger_augments_only.py`, which is
in the unmarked fifteen. (A peer's independent census reported 15 and 14; this one
finds one more file, and the discrepancy is not resolved — it is likely a file
using `ast.parse` only inside a helper. The direction is the same either way.)

So the finding is not *two tests near a limit*. It is **one test over the limit and
a class of fifteen whole-package AST walkers running under a default none of them
declares**, any of which can emit a phantom `FAILED` naming a real test as soon as
the node is contended. The reading rule stands unchanged — separate
`Failed: Timeout` from assertion failures before attributing either — but the
durable fix is that a test which walks the package should say so with a marker,
which is what the one marked file already does and what the other fifteen could
copy. That is a mechanical, behaviour-preserving sweep and therefore its own
commit, sequenced when no session holds uncommitted work in the affected files.
