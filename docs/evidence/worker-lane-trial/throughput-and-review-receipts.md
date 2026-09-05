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

### clive / deepseek-v4-flash — `n-wcr-the-divergence-scan-names-its-subtrees` — in flight

Observed at 33 minutes into a 40-minute budget: 88 tool calls (68 Bash, 19 Read,
1 Write), no commit, and only the orientation stub manifest written at
minute two. One untracked file had appeared outside the node's two declared write
paths, at `imas_codex/standard_names/manifests/reviews/`, shaped like a release
roster. The coordinator messaged the worker to land its commit and manifest
before budget rather than after, and to account for that file without staging it.
Verdict to follow.

## Standing method

- Read every lane's diff to the same depth. Reviewing only the trial lane and
  taking the metered lanes on their gate verdicts would attribute
  review-detectable defects to the trial that are not model-specific.
- Record model turn time separately from job wait wherever a node carries a
  scheduler leg.
- Prefer a second detached checkout at the base revision for failure
  attribution, over the worker's own statement of its base.
