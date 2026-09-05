# Worker lane trial — measured throughput and reviewed-code receipts

A live trial inside a working sprint, comparing a locally-served model
(`deepseek-v4-flash`, free, 2xH200) against the two metered lanes on the same
nodes. The question is not whether nodes exit green. It is whether the code that
comes back survives a line-by-line read. Every figure below is taken from the
run's own stream or ledger record; none is recalled.

A parallel trial is running on the nova project and will contribute rows in the
same shape. This file is the imas-codex half.

## How a lane is named

A backend in `flight.yaml` is a model-and-effort preset, not an account.
`claude` and `claude-opus` are two presets over one account; `codex`,
`codex-luna`, `codex-spark` and `codex-terra` are four presets over another, and
all four went held together at 95%. Budget reasoning that treats presets as
independent lanes over-counts.

Member `--harness` does not select the backend. A member registered
`--harness claude-opus` ran on sonnet, because dispatch falls through to
`default_backend` and `roles.<role>.by_spec_level` then forces effort. The
selector that works is `--set default_backend=<lane>` on the dispatch, or
`--local`. Confirmed twice: once here by a dry run resolving to
`dsv4-flash / deepseek-v4-flash / worktree-full`, once independently on nova.

## Throughput, measured

| Lane | Node | Wall | Output tokens | tok/s (wall) | Cost |
|---|---|---|---|---|---|
| claude / sonnet-5 high | `n-sli-updated-at-is-maintained-everywhere` | 1857 s | 97,328 (52,187 thinking) | 52.4 | $6.77 |
| codex / gpt-5.6-sol high | `n-sgr-size-the-reviewer-findings` | 3040 s | 55,704 (16,671 reasoning) | 18.3 | not reported in stream |
| clive / deepseek-v4-flash high | `n-wcr-the-divergence-scan-names-its-subtrees` | 2004 s (partial) | ~44,950 estimated | ~22.4 | $0 |

Three cautions on that table, all of which matter more than the numbers.

The sonnet row is the only one with a first-party figure: its `result` record
carries `output_tokens` directly, along with `duration_api_ms` 1,122,117 against
`duration_ms` 1,857,212 — so 60% of its wall time was inside the API and 40% was
local tool execution. Its API-time rate is 86.7 tok/s.

The codex row is a single cumulative `turn.completed` usage block. Its input
figure of 20,124,632 tokens is 97.9% cache reads.

**The clive row above is a mid-flight estimate, and mid-flight is the only place
an estimate is needed.** Every assistant message from the local server reports
`output_tokens: 0`, so nothing can be read while a run is in progress; the figure
above is character count over four across text, tool-call inputs and thinking.
But the terminal `result` record does carry totals, and `crew complete` reads
them — corrected by the parallel project after I had written the gap up as
absolute. So the lane is measurable **at promotion** and blind **in flight**,
which is a smaller defect than I first reported, and still a real one: a
coordinator cannot see a local run's consumption while deciding whether to let it
continue.

Of the mid-flight estimate, 83% is thinking (150,095 characters) against sonnet's
54% — the local model spends far more of its output on reasoning per unit of
delivered work. Over 95% by characters was observed independently on two nodes in
the parallel trial.

### The local lane caches about a third of what the metered lanes cache

**Corrected 2026-09-05, and the first version of this section over-claimed.** I
wrote that the lane has *no* prompt caching, reading a client-side
`cache_read_input_tokens: 0` as a fact about the engine. The serving project then
sampled the live two-card engine's own metrics: `prefix_cache_hit_rate = 0.3389`.
vLLM automatic prefix caching **is** active and hitting about 34%.

So this is substantially a **reporting gap** rather than a compute gap: the
client-side usage record says zero while the engine says a third. Any cost or
cache analysis built on this lane's client usage records — including the table
below as first written — is wrong in the same direction.

What survives, because the comparison is still lopsided:



The input columns diverge by more than an order of magnitude, and in the
direction that explains everything else:

| Lane | Input tokens | Of which cached | Fresh input |
|---|---:|---:|---:|
| codex / gpt-5.6-sol | 20,124,632 | 19,695,744 (97.9%) | 428,888 |
| claude / sonnet-5 | 24,233,869 | 23,983,548 (99.0%) | 250,321 |
| clive / deepseek-v4-flash | 19,974,934 | **0** | 19,974,934 |

The `clive` row is what the **client** reports, and the engine contradicts it:
about 34% of prefix queries hit. The metered lanes report 97.9% and 99.0%
*cached* on the same axis. So the honest statement is that this lane re-reads
roughly **two thirds** of its context every turn against a couple of percent on
the metered lanes — a real and large gap, and not the total absence the zeros
suggested.

That still explains the throughput (~22–33 tok/s against sonnet's 52 and Opus's
78) and why a long node does not accelerate as context accumulates. And it
sharpens rather than weakens the recommendation: **34% cumulative is well short
of what a properly cached agentic conversation reaches**, so there is headroom at
the endpoint, and closing it is worth more than changing model or effort. What
changes is that the lever is *cache configuration*, not *cache introduction*.

**Two instrument lessons, and I am the cautionary half of the first.** A
client-side zero is a statement about the client. I read it as a statement about
the engine, wrote a section headline on it, and it drove a config recommendation.
The engine's own metrics were one query away and I did not think to ask for them
because the absence looked like data. That is the same shape as the four other
corrections in this record: an instrument's silence read as a measurement.

The second is the serving project's, and worth carrying if anyone scrapes that
endpoint: matching vLLM metric names **by substring** silently sums each
counter's `_created` timestamp gauge and the `external_` cross-instance counters
into the real totals. Match exact bare names. On this engine they are
`vllm:kv_cache_usage_perc` — not `gpu_cache_usage_perc` —
`vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`.

**Node shape follows from this.** A free lane with no caching is cheapest per
token and dearest per turn. It suits work with a small, stable context and a
clear finish line, and it is the wrong home for a node that must read widely
before it can act — which is exactly what the divergence-scan node was asked to
do, and exactly where it spent its first budget.

The clive stream also emits 35,404 `system` records against sonnet's 535 for a
node of comparable length, which is why its transcript is 7.7 MB.

## Reviewed-code receipts

The count that matters: **defects found by reading the diff that the node's own
gate could not see.** Split into two, because they have different owners — a gate
that structurally cannot reach the change is the orchestrator's error, and a gate
that could have reached it with a better assertion is the worker's.

### claude / sonnet-5 — `n-sli-updated-at-is-maintained-everywhere` — 4 defects

Gate verdict: passed, and honestly so. It was mutation-proven both directions
(stamp removed, gate exit 1; restored, exit 0), the manifest named its own
judgement calls, listed exemptions it found beyond the two it was given, and
flagged 13 out-of-fence files it could not reach. Nothing in the return looked
weak. The commit raised stamped write sites in a 22k-line module from 3 to 118 by
a scripted static-analysis pass.

| # | Defect | Could the gate see it? |
|---|---|---|
| 1 | The one statement in the file that provably modifies nothing — a compare-and-set lock written `SET old.claimed_at = old.claimed_at, ancestor.claimed_at = ancestor.claimed_at` — now stamps both aliases. That self-assignment idiom is exactly what the change's own stated rule exempts. | No: the checker's predicate asks whether a SET block mentions the stamp anywhere, never whether the alias it stamps is one the block writes. Worker-owned. |
| 2 | Two blocks assigning only `StandardNameSource` properties now stamp the `StandardName` alias, because the injector keyed on "a `:StandardName` binding exists somewhere in this statement". | No, same predicate. Worker-owned. |
| 3 | The mirror fault in the same function: a block assigning `ancestor.source_paths` stamps only `old`, so a genuinely modified identity carries no stamp. | No, same predicate. Worker-owned. |
| 4 | The schema description says the property records "the last substantive modification", but the pass injected into idempotent `coalesce(new, old)` batch writebacks that sweep the whole cohort every pipeline pass. The value it will hold is when the pipeline last touched the node. | No: no test can assert a description against an intention. Worker-owned. |

**Two of those four were misattributed, and the repair worker corrected me.**
Findings 2 in the table above — the two blocks I called source-only — do assign
`StandardName` properties: `sn.run_id = coalesce(sns.run_id, sn.run_id)` and
`sn.source_paths = CASE … END`, both several lines past the end of the diff hunk
I read. I judged the statements from a truncated view and reported a conclusion
the visible lines supported but the code did not.

The repair node held to the checkable contract over my prose, said so explicitly
rather than deviating quietly, explained that a rule broad enough to flag those
two overshoots to six or more, and put in their place a second self-assignment
lock I had missed entirely — `SET name.name_stage = name.name_stage` in
`supersede_exhausted_standard_name_orphans`. So the count of four held while its
composition changed: **two locks and one under-stamp, not three over-stamps and
one under-stamp.**

It also found why my under-stamp was invisible to the obvious fix. The
pre-existing block-boundary scanner ended a SET clause at the first `WHERE`,
including the one inside a Cypher list comprehension — which truncated the very
statement holding the under-stamp. A naive symmetric predicate would have found
three, reported itself clean, and left the real gap in place. That scanner bug is
the node's best work and nothing in the brief pointed at it.

Worth recording for the trial, because it cuts against the framing: **the
reviewer was wrong twice and the reviewed worker was right.** A control column
built from coordinator review inherits the coordinator's errors, which is the
argument for the independent read-only reviewer the parallel project is using.

One structural hole alongside them, and this one is mine. The gate ran
`pytest tests/graph/` under the repo's default markers, which exclude the marker
that needs a live database. **None of the 118 modified Cypher statements has ever
been sent to a real Neo4j.** The worker did run once without the filter, got 11
failures, and attributed them to a missing backend — which it had no way to
distinguish from a malformed statement of its own. The brief should have named a
gate that reaches the surface the change touches.

Repair dispatched to the same member for continuity, at spec-level `exact`, with
all four sites named and a symmetric per-alias predicate specified.

### clive / deepseek-v4-flash — `n-wcr-the-divergence-scan-names-its-subtrees` — first turn

The first turn ran to its 2400-second budget and past it, and produced **no edit
to the file the node exists to change.** At termination: 88 tool calls (68 Bash,
19 Read, 1 Write), zero commits, and the orientation stub manifest still as
written at minute two. `git status` in its worktree showed `export.py` untouched.

The whole budget went on baseline measurement. That is partly the brief's fault
and partly not, and the two halves should be scored separately.

**The brief's half.** The measure demanded two release dry runs plus two
`tests/standard_names` runs inside 40 minutes, against a comparable claude node
given 55 minutes that used 31. The node was under-budgeted for its own measure.
That is an orchestrator error and it must not be charged to the model.

**The model's half.** The last four minutes were spent recovering a pytest tally
line that a progress bar had overwritten, after the worker had written in its own
reasoning that it did not need the number: *"I don't strictly need the total — the
orchestrator's measure is 5 failures before, added failures 0."* It then ran the
same `--collect-only` command three more times with different greps. It does not
budget its own remaining time against the deliverable.

**What it got right**, and this is the field that matters most for a trial: it
identified the 5 `tests/standard_names` failures as pre-existing, correctly, and
did not attempt to fix or absorb them. Own-versus-inherited attribution passed.

**What the run found anyway.** By exercising the release path it exposed a real
defect the node was not looking for: `imas-codex sn release --dry-run` mints a
release-shaped roster into the repository and advances the release-candidate
counter, because `_freeze_review_artifact` is called at step three and `dry_run`
is not consulted until step five. Recorded on the release plan; it very likely
explains the rc2, rc3 and rc4 rosters already sitting in that directory. A node
that lands nothing can still be worth its wall clock.

**One repository rule broken.** It piped `imas-codex` CLI output through `tail`,
which the repository instructions ban explicitly and with a stated reason (the
CLI auto-logs, and piping blocks command auto-approval). The instruction was in a
file it had been told to read first.

Resumed with the measuring handed back as given, told to commit before measuring.

### codex / gpt-5.6-sol — `n-sgr-size-the-reviewer-findings` — 0 defects

The third control, and the strongest return of the three. An attribution-only
census sizing four reviewer-finding families against the live catalogue and the
frozen 431-name batch roster. Report-only: no source, graph, pipeline, release or
plan mutation, and its gate was a report-integrity check rather than a test suite.

Reading it found nothing to correct, and four things worth copying:

- It carries the **verbatim** Cypher and Python predicates it ran, not a
  description of them, so every figure is re-derivable without the author.
- It ran a **property-coverage positive control** before trusting any count —
  4,937 of 4,937 for `id`, `name_stage` and `kind` — which is the direct guard
  against this project's silent-zero class, where naming a property the schema
  does not declare returns zero rather than erroring.
- It named the two live identities that do not parse under the current grammar
  and stated the adjusted denominator (2,685 rather than 2,687) instead of
  quietly dropping them.
- It reported live, superseded and exhausted populations separately and never
  summed them, which is where family counts in this project usually go wrong.

Cost of the discipline: 3040 s and 18.3 tok/s, the slowest of the three, for a
node that wrote no code. Whether that trade is right depends on whether the
figures are load-bearing. Here they are: they size the remaining migration.

## Steering a live worker: the message waits for a human who is not there

`SendMessage` addressed to the worker's session returned `success: true`, and the
message never reached the worker: **zero plain-text user records in its stream**
across four subsequent tool rounds, and a grep for the message text over the 8 MB
transcript returns nothing.

The mechanism is not a lost message. Two delivery notices arrived afterwards: the
message was **held for the recipient user's approval**, and then **expired
unapproved**. A dispatched crew worker has no human watching its socket, so a
peer message to one is queued against an approval that can never come.

That distinction decides the fix. The channel is not missing and not broken — it
is gated on a human in a place where there is no human. Either exempt a
coordinator's message to a run it dispatched from that approval gate, or make the
send fail immediately with the reason rather than reporting success and expiring
silently. The current behaviour is the worst of the three: the sender is told it
worked.

Until then the orchestration guidance's "message the worker" row does not apply
to a CLI-launched run, and the only channel is `reckon crew resume --advice`,
which requires the run to be terminal. So correcting a worker that is burning its
budget on the wrong thing costs a `crew stop` first — the stop is what makes the
advice deliverable.

The recovery worked: stop, then resume with the measurements handed back as
established facts. But the sequence should not require killing a healthy process
to say one sentence to it.

Recorded as a correction rather than edited away, because the first reading of
this — "the message silently vanishes" — was wrong in a way that would have sent
the fix to the wrong layer.

### The class, once three instances were on the table

The messaging finding turned out not to be about one channel. Sent to the
repository that owns crew dispatch, it prompted a search that found two
first-party instances of the same shape, in the only two cases that repository
had ever exercised:

- Its worker-to-worker peer channel produced exactly two questions in one
  session — at 12:04:41Z and 12:48:16Z — and **both went unanswered forever.**
  Neither carries an `answered_at`. Both workers blocked with a NEEDS-HELP brief
  and were unblocked by hand. Two for two, so the premise that a worker can reach
  the worker it needs is false in every case it has been tried.
- Plus the coordinator-to-worker case measured here.

**One of the three fails well, and the difference is the whole lesson.** The peer
channel *names the unanswered question and blocks*. My send *returned
`success: true`* and expired silently. Both channels are equally broken; only one
is diagnosable. Both of the blocked manifests were understood in seconds, while
the undelivered message took a stream-wide grep, a record count and two delivery
notices arriving after the fact to establish — and my first conclusion from that
evidence was wrong.

So the principle worth carrying into whatever replaces any of them: **a channel
that cannot deliver should say so at the point of sending.** A success return on
a message that will expire is the one behaviour that guarantees the sender plans
around it, and it converts a cheap failure into an expensive investigation.

Ownership was settled the right way too, and it is worth recording as a pattern
for cross-project findings: the approval gate belongs to the host harness rather
than to crew dispatch, so the owning repository recorded it explicitly as **not
ours** — so that nobody spends an afternoon hunting it in the wrong codebase — and
kept the two halves that are theirs. A finding routed to a repository that cannot
fix it is worth less than the same finding routed there with the boundary drawn.

## The checkpoint is a stop-the-world operation, not a background one

Recorded here because it constrains every wave from now on. The graph backup runs
`neo4j-admin database dump`, which refuses while the database is in use — the
remote path stops Neo4j, waits 30 seconds for a GPFS lock reclaim, dumps, and
restarts. So a backup requires every graph-touching worker to be at rest, and a
node holding a read-only Cypher session is as blocking as one writing. The work
gap before a checkpoint is mechanical, not hygiene.

## Standing method

- Read every lane's diff to the same depth. Reviewing only the trial lane and
  taking the metered lanes on their gate verdicts would attribute
  review-detectable defects to the trial that are not model-specific.
- Record model turn time separately from job wait wherever a node carries a
  scheduler leg.
- Prefer a second detached checkout at the base revision for failure
  attribution, over the worker's own statement of its base.

## What the shared routing config should change

These are for the project that owns crew dispatch and the flight configuration.
Each is evidenced from this workstation's own records rather than proposed from
preference. The first is a safety gap and the rest are consistency.

### The claude lanes' budget check reads the wrong number, in the wrong unit

`budget_check: true` is set on both claude backends, and it provides no
protection. The preflight reports, for both:

```
utilisation_pct: 1.21   rate_limit_type: overage   resets_at: 2026-10-01T00:00:00Z
detail: "backend 'claude' exposes no account-limit surface to read"
```

The lane's own stream carries the gating signal, twenty-two times in one run:

```json
{"status": "allowed_warning", "rateLimitType": "overage", "utilization": 1.21,
 "unifiedWindows": {"five_hour": {"utilization": 0.48, "resetsAt": 2026-09-05T12:00Z},
                    "seven_day": {"utilization": 0.14, "resetsAt": 2026-09-10T18:00Z}}}
```

Two errors compound. The preflight reads the top-level **overage** figure — the
account's position against a horizon three weeks out, whose reset stamp is
exactly the `2026-10-01` the preflight repeats — rather than
`unifiedWindows.five_hour`, which is what actually gates the next request and
stood at 48% with three and a half hours to run. And `utilization` is a
**fraction**: 1.21 means 121%, and it is being rendered as `1.21%`, a hundredfold
optimistic.

**Confirmed at the source, and the conclusion is stronger than the observation.**
The reckon session read `_ClaudeDialect._budget` in `reckon/_backends.py`: it
takes `info.get("utilization")`, a fraction, and assigns it straight into
`utilisation_pct`, which is then compared against a 95 percent ceiling. So **the
fence cannot fire on that lane at any utilisation** — a five-hour window at 99%
reads `0.99%`. Not "reads the wrong number sometimes": never fires.

The detail that makes it a textbook case: `unifiedWindows` appears seven times in
that repository and **every one is inside a test fixture.** The correct field is
sitting in reckon's own test data, unread by the parser that fixture feeds. A
fail-open guard whose own tests contain the input that would have caught it.

The separate-records half was reproduced independently too: 47 budget rows for
`claude` against 16 for `claude-opus`, both landing on 1.21/overage nine minutes
apart.

The `detail` line is the tell — it says no account-limit surface can be read
while the number it prints comes from one. The signal is present and unparsed,
which is the same shape the shared guidance already records for spend refusals.

This is the reverse of the hazard that guidance warns about. A stale hold is
self-sustaining and visible: the fleet stops and someone asks why. A stale
**clearance** is silent, and it is what the claude lanes have now.

### Two backends over one account keep two budget records

`claude` and `claude-opus` are `command: claude` on one account. Their budget
states are tracked separately and were last observed 26 hours apart —
`2026-09-05T08:03Z` against `2026-09-04T06:17Z` — while carrying the identical
1.21 figure, because it is one account's number read twice. So a lane can be
cleared on a sibling's day-old reading of a shared account, and the four codex
presets hold together today only because they happened to be probed together.

Declaring the account a backend belongs to would make the sharing explicit and
let one observation serve every preset over it.

### The local lane is the least completely specified backend

`clive` declares `launch`, `command`, `model`, `effort` and `alias`. Its sibling
`clive-glm` additionally declares `sandbox`, `session_reuse` and `time_budget`,
as do all four codex presets and both claude presets. A dispatch to `clive`
resolves `sandbox: worktree-full` from a shipped default rather than from the
lane, and takes a default time budget.

Give it an explicit `time_budget` **and treat it as advisory, because for this
launch kind it is not a fence.** Measured: a node here ran past its 2400-second
budget and kept going until a `crew stop` ended it; two nodes in the parallel
trial ran 8128 s and 7733 s against 60-minute budgets with nothing stopping them.
All three delivered correct, gate-green work at the end, so the overrun cost wall
clock rather than quality — but a coordinator planning a wave around the
configured figure is planning around a number the harness does not apply.

**Do not read a wall-clock number as a property of the lane.** Corrected by the
parallel project's lead: those two-hour walls were measured on the 2×H200
deployment, a 4×H200 one is being stood up, and the figure is expected to move.
So the recommendation is the framing, not the number: declare a budget so the
ledger records an intent, and plan on the assumption that nothing enforces it.

What *is* hardware-independent, and worth stating because it compounds with the
steering finding above: **a clive node can be neither steered nor time-boxed, so
its only fence is a person watching.** More cores will change how long it runs,
not whether anything stops it.

The related lane property is real and separate: this backend's own configuration
comment records that its thinking is intrinsic and its budget parameter ignored,
and the measured thinking share is 83% by characters here and over 95% on two
nodes in the parallel trial. A lane that cannot be told to think less cannot be
made faster by asking; it can only be given work whose shape does not invite
exploration.

### The local lane reports no output-token usage

Every assistant message from `deepseek-v4-flash` carries `output_tokens: 0`, and
no usage total appears anywhere in the stream. Confirmed independently on two
further nodes in the parallel trial. Input tokens *are* reported, so the rework
metric that the `by_spec_level` rows were qualified on remains computable for
this lane — but throughput and cost-per-node are not, and any tok/s figure for it
is a character-derived estimate that must be labelled as one.

### Member `harness` does not route, and reads as though it does

A member registered `--harness claude-opus` ran on sonnet, because dispatch falls
through to `default_backend`. Hit independently by both coordinators in this
trial, in different projects, on the same day. Either make the member's declared
harness route its dispatches, or stop accepting the flag — a field that is
recorded, displayed, and ignored will keep costing this.

### A message to a dispatched worker waits for approval from nobody

A peer message to a CLI-launched worker's session returns `success: true`, is
then held for the recipient user's approval, and expires undelivered — because a
dispatched worker has no human watching its socket. The orchestration guidance
lists messaging the worker as a recovery step before redispatch; for this launch
kind it cannot work.

Exempt a coordinator's message to a run it dispatched from the approval gate, or
fail the send immediately with the reason. Reporting success on a message that
will expire is the one behaviour that guarantees the sender plans around it.

## What decides whether the local lane succeeds

Three nodes on `deepseek-v4-flash`, two of them from the parallel project, land on
the same answer: **the governing variable is how specified the brief is, not how
hard the task is.** The model does not decide well when a measurement is
finished, and it does not need to when the target is fixed.

| Node | Brief shape | First materialisation | Outcome |
|---|---|---|---|
| divergence scan (here) | "measure before and after, decide the file set" | never, in 40 min | no edit, whole budget on baseline |
| the same node, resumed | measurements handed back as given, commit before measuring | within minutes | committed change, honest measure, real cause found |
| stale-pin inventory (parallel) | read-only judgement over ~70 files | after a guard refused its fan-out and forced a sequential read | 46 pins classified, verdicts checkable and correct |
| class-margin repair (parallel) | field name, formula, sign rule, two fixtures, all numeric | **assistant turn 6, 17 seconds in** | one commit, +181/-1, gate 12 of 12, no defect on a hunk-by-hunk read |

The two extremes are the same model on the same lane hours apart. What separates
them is that one brief named the target and the other asked the worker to decide
when it had measured enough.

That is an orchestrator lever, not a model limitation, and it is cheap to pull.
It also matches the caching finding above: a lane that re-reads its whole context
every turn gets more expensive the longer it explores, so a brief that invites
exploration is the worst possible fit for it, and one that names a fixed target
is the best.

### The trial's sharpest result: the worker was right about the brief

The divergence node was aimed at the wrong function, by me. I wrote that the
scan lived in `detect_divergence` and named its line. The worker measured, found
its own gate could not discriminate — the release dry run never executes the
post-copy check at all, so virtual-environment rows were already zero at baseline
— and then found the real source: `check_catalog` in `catalog_import.py`
discovers entry files with `catalog_dir.rglob("*")` plus a suffix filter, which
is exactly the walk-a-checkout-and-filter shape the plan bans.

It proved that from the transcript of the earlier release cut rather than by
argument: `Post-copy check found 387 diverged entries` appears immediately after
a pydantic parse error on
`.venv/lib/python3.13/site-packages/markdown_it/port.yaml`. Then it stopped,
because that file is outside its write fence, and routed it to `follow_ons` with
the note that no concurrent node holds it.

So the same day produced two coordinator errors, each caught by the worker it
was given to: a claude node corrected my attribution of two defects, and a local
node corrected the target of a whole brief. **On the axis this trial exists to
measure — does the returned work survive a careful read — the local lane's
failure was mine.**

Its committed change (naming the entry-file set instead of globbing a directory)
is a genuine improvement to `detect_divergence` and does not fix the reported
defect. The real repair is a follow-on node against `catalog_import.py`.

I raised one review note on its diff and **it was wrong, which makes three
coordinator errors today.** I flagged that it widened the read from `*.yml` to
`*.yml` plus `*.yaml` on an unsourced docstring claim of a `.ya?ml` convention.
The claim is sourced: `catalog_import.py` carries
`_DOMAIN_PATH_RE = re.compile(r"standard_names/([^/]+)\.ya?ml$")`, which both
accepts either suffix and requires a direct child. The worker's read set matches
the repository's own regex exactly, and my objection came from checking two write
sites rather than the definition.

Its final return is the strongest of the four nodes on any lane:

- Gate: `tests/standard_names` at 5 failures, **the same five ids as baseline,
  zero added**, both logs on disk.
- It declared its own headline measure **vacuous** rather than banking it. The
  before and after dry runs both report 101 divergence entries and zero from a
  virtual environment, and it said plainly that the dry run executes only Gate D
  against a catalog checkout sitting on blank main, so neither run could have
  surfaced such a row. A worker with a weaker conscience reports "0 virtual
  environment entries, target met".
- Having lost its measure, it built a replacement: four fixture tests, one
  proving a genuine divergence is still reported and one proving a
  `.venv`-nested YAML can never become a comparison authority — plus a
  **counterfactual probe** confirming the first test fails if the scan is
  narrowed to nothing. That is the two-sided gate the brief asked for, obtained
  by a route the brief did not suggest.
- It cited the earlier release cut by run id when proving where the 387 rows come
  from, and left the out-of-fence file alone.

### A free lane priced like a metered one

The parallel project's ledger prices its two clive nodes at $41.15 and $52.13.
The lane is free and locally served, so those are notional — what the tokens
would have cost metered, inflated by the absence of caching. That is a defect in
any routing decision that reads cost from the ledger: **the cheapest lane on this
workstation records the highest per-node cost of the three.** A ledger that
prices a free lane should either record zero or mark the figure as imputed.

**Confirmed on a larger sample, with a consequence neither project had seen.**
Measured across 619 ledger rows rather than two nodes: `clive` at a median
$21.59 per node against `claude`'s $1.61, so the free lane ranks about thirteen
times dearer. A different magnitude from the figures above because the samples
differ; the direction is identical. The consequence is that reckon shipped a
routing view **whose routing quantity is cost per durable node** — so the surface
built to make routing legible currently inverts the comparison it exists to
support. That section's implementation figure has been reduced rather than left
reading as landed, which is the right disposition: a legibility surface that
misleads is worse than an absent one.

## The regime-bound check

Both projects independently found the same defect class today, and it is worth
naming because it is what a green gate cannot see. **A regime-bound check is one
whose inputs never leave the regime its correctness assumes.** It passes, it is
honestly written, and it is silent about everything outside that regime.

Three worked examples from today, at three severities:

| Severity | Defect | The regime it never left |
|---|---|---|
| High | A stationary-point census takes the first `2*maxsize` admitted origins via `jnp.where(size=)` and publishes `overflow False`, so a field admitting more than 60 origins silently drops the rest. | The gate's two analytic fields admit 2 origins each. A probe on the gate's own lattice admitted 147 and changed 31 of 39 published keys. |
| Low | A Cypher SET-clause boundary scanner ends the clause at the first `WHERE`, including the one inside a list comprehension, truncating the statement that held the fault being looked for. | Every statement the checker had been run against had no list comprehension in its SET body. |
| Latent | New code reads four keys off a live diagnostics dict; the tests monkeypatch that dict. | The gate never reads the real host, so it cannot say whether the keys exist there. Checked by hand; they do. |

The remedy is one question, and it found all three: **before trusting a green
gate, name one input that leaves the regime and ask whether the gate would
notice.** None of the three came from reading the diff harder. Each came from
asking what the check never sees.

This belongs in the shared guidance next to the fail-open guard rule, which is
its sibling: a fail-open guard reports a protection it does not provide, and a
regime-bound check reports a coverage it does not have.

### Implicit test contracts on incidental structure

A near relative of the regime-bound check, and it caught both projects on the
same day. A test can silently make something *incidental* into a contract, so a
change that is semantically free breaks it — or worse, a change that is
semantically real passes.

- **Query text.** Several standard-names tests mock `GraphClient` and dispatch on
  the literal leading token of a `SET` clause. Comma-separated assignments to
  independent properties are order-independent in Cypher, so *prepending* an
  assignment is free semantically and cost 22 test failures. The fix is to append,
  which is correct and one line — but query text is now a contract in this
  repository, and nothing says so at the call sites.
- **Field-name sets.** In the parallel project, corroboration tests monkeypatch
  the host diagnostics dict, so the *set of key names* a topology read expects is
  a contract there. A renamed key passes the gate and fails in production. Same
  class, opposite direction: here a free change breaks a test, there a breaking
  change passes one.

Both are invisible at the site of the change and only findable by reading what
the test actually asserts on. Worth one line in a repository's own guidance
wherever it holds: *these tests pin query text* / *these tests pin key names*.

### The cost of the specificity lever, stated honestly

The finding that a fixed-target brief transforms this lane's output is real, and
it is not free. A brief can only name the target when the coordinator already
knows the answer's shape, so the lever trades worker exploration for coordinator
work — and both of today's coordinator errors are that trade showing up on the
other side of the ledger. A brief specific enough to make the local lane succeed
is specific enough to be confidently wrong, and mine was: I named the function,
the line, and the gate, and the function was not where the defect lived.

So the recommendation is not "write more specific briefs" on its own. It is:
write the target-naming brief, and expect the worker to contradict it. The two
nodes that corrected me today both did so because their measures were checkable
and their fences held — not because the briefs were good.

## A dispatch validator false positive worth fixing

The fully-specified check refused a node on the string `'<domain>'`, reporting
that it "leaves the worker to infer intent". It was a path template —
`standard_names/<domain>.yml` — in a sentence that then named the concrete
example file. Angle brackets in a path are ordinary notation, not an unresolved
instruction, and the brief had to be reworded into prose to pass.

This is the sixth phrasing refusal this session, after `correctly`,
`a good outcome`, `IS CLEAN`, `is a readable`, and a trailing `DECIDE`. Most of
those were fair. This one cost a rewrite that made the brief longer and no
clearer, which is the failure mode a phrasing gate has to avoid: if it pushes
authors toward prose, it is trading precision for compliance.
