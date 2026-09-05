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

**The clive row is an estimate, because the local server reports
`output_tokens: 0` on every assistant message.** The figure above is character
count over four, taken across text, tool-call inputs and thinking blocks. That is
a measurement gap in the local lane worth closing before any spend decision rests
on it: a lane whose throughput cannot be read from its own stream cannot be
compared honestly with one that can. Of that estimate, 83% is thinking
(150,095 characters) against sonnet's 54% — the local model spends far more of
its output on reasoning per unit of delivered work.

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

## Steering a live worker does not work on this launch kind

`SendMessage` addressed to the worker's session returned `success: true`. The
message never arrived: **zero plain-text user records in the worker's stream**,
across four subsequent tool rounds, and a grep for the message text over the
8 MB transcript returns nothing.

This matters because the orchestration skill's continuity table lists "message
the worker" as a recovery step before redispatch. For a CLI-launched crew run
there is no live steering channel at all: the only channel is
`reckon crew resume --advice`, which requires the run to be terminal first. So
correcting a worker that is burning its budget on the wrong thing costs a
`crew stop` — and the stop is what makes the advice deliverable.

The recovery worked: stop, then resume with the measurements handed back as
established facts. But the sequence should not require killing a healthy process
to say one sentence to it.

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
optimistic. So the fence reads a lane at 1.21% when its gating window is at 48%,
and will keep reading 1.21% until October regardless of any five-hour exhaustion.

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

Give it an explicit `time_budget`, and make it **longer** than the metered lanes'
rather than equal. That is not generosity: this lane's own configuration comment
already records that its thinking is intrinsic and unbudgetable, and the measured
thinking share here is 83% by characters, with over 95% observed independently on
two nodes in the parallel trial. A lane that cannot be told to think less needs
to be given time to finish, or it will keep spending whole budgets on
preliminaries. The one node run here spent 40 minutes on baseline measurement and
never reached its edit.

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

### A live worker cannot be steered

A message to a CLI-launched worker's session returns success and never reaches
its turn. The orchestration guidance lists messaging the worker as a recovery
step before redispatch; for this launch kind the only channel is
`crew resume --advice`, which needs the run terminal first. Either deliver peer
messages into a CLI run's turn, or remove that row so nobody plans around it.
