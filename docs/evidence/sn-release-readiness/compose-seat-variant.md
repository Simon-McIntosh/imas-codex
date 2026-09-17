<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="sn-release-readiness">
<meta name="plan-status" content="active">
<meta name="plan-title" content="SN Release Readiness">
<meta name="plan-evidence-for" content="sn-release-readiness">

# The compose seat is measured against the served variant

Plan `imas-codex:sn-release-readiness` §1. The seated model and the served model
name different variants of the same family: the seats ask for
`local/deepseek-v4-flash`, the router serves `deepseek-v4.1-flash`. This node
was to measure the served variant on a committed fixture, compare the result
against the recorded figures for the previous variant, and move the four local
seats only if the measured figures meet or exceed the record.

## Result

**The seats are left unchanged, and the shortfall is not a number — the run
could not be taken.** The comparison the node is defined by requires a
benchmark run, every benchmark fixture path in this repository extracts its
candidate set through the live graph, and the graph is unreachable from this
workstation right now: there is no bolt listener on `127.0.0.1:17687`, and the
check that establishes it refuses the connection (`/dev/tcp/127.0.0.1/17687:
Connection refused`). No benchmark report was taken. **No verdict on the quality
bar: BLOCKED BEFORE MEASUREMENT**, the same status the 2026-09-14 attempt at this
measurement reached, and for the second of the two causes it recorded.

The same blocker killed the earlier attempt on a SLURM lane, where the tunnel
could not be established either, so this is not a placement choice that a
different lane would fix.

## What the endpoint itself answers, measured

Two facts about the served endpoint, taken directly rather than inferred from
the configuration:

| Request | Response |
|---|---|
| `GET /v1/models` on the router | one entry: `deepseek-v4.1-flash` (served by `sglang`, `max_model_len` 512000) |
| `POST /v1/chat/completions`, `model=deepseek-v4.1-flash` | **HTTP 200** |
| `POST /v1/chat/completions`, `model=deepseek-v4-flash` | **HTTP 404** |

The third row is the one the seats currently ask for, and it is the answer to
the question the plan left open rather than assumed: the prefix is stripped
before the request leaves the process
(`imas_codex/discovery/base/llm.py:128-133`, `_acompletion_local` drops the
`local/` prefix because "the local endpoint serves the bare model name"), so
the router receives the bare identifier — and it does not normalise the name,
and it does not accept an arbitrary identifier. **It rejects it.** The third of
the three possibilities the plan named is the one that happens: the mismatch
fails loudly. The seats carry `endpoint-class = "local-free"` and
`model-route = "ambix-local"`, so the failure costs a refused request rather
than a bill.

The served variant cannot be reached by a live A/B against the previous one
from here regardless of the graph: the router lists one model, so the previous
variant is not merely unrouted, it is absent from the served endpoint.

## The bar, from the record

The prior figures for the previous variant `hosted_vllm/deepseek-v4-flash` are
recorded in `research/physics_bench-20260616T100837Z.json`, dated
**2026-06-16T10:30:59Z** (companion run `...T061155Z.json`, same day, 07:24Z).
The comparison target is that artifact. `reference_total` is 52 reference names
with an extraction of 12 sources; the physics fixture committed today holds 15
paths, so the two populations are not the same size and any comparison carried
across them would be population-confounded as well as instrument-confounded.

| Figure | 100837Z (10:30Z) | 061155Z (07:24Z) |
|---|---:|---:|
| `reference_precision` | 0.1111 | 0.0769 |
| `reference_recall` | 0.0192 | 0.0192 |
| `reference_total` | 52 | 52 |
| `grammar_valid_count` | 8 | 12 |
| `grammar_invalid_count` | 1 | 1 |
| `batch_errors` | 1 | 0 |
| `avg_quality_score` | 0.876389 | 0.613462 |
| `physics_rate` (faithful/total) | 0.8889 (8/9) | 0.3077 (4/13) |
| `elapsed_seconds` | 635.62 | 388.97 |
| `total_cost` | 0.0 | 0.0 |

**A caveat that would make this bar invalid on its own columns, recorded before
the comparison is ever run.** The `reference_precision` and `reference_recall`
figures here were computed by a name resolver that was later found to drop the
locus and projection segments of the intermediate representation, which is why
they sit at 0.11 and 0.019 — one matched reference name out of 52 for a model
whose candidate names were faithful. The resolver was repaired afterwards. So
those two columns in the record are a reading from a broken instrument, not a
property of the model, and a served-variant run scoring above or below them
would be comparing against the defect. `avg_quality_score` and `physics_rate`
are reviewer-verdict figures and are not affected by that resolver.

## Concurrency, pinned by construction

`sn bench` exposes no concurrency option, so the pin is stated from the code
path the served endpoint actually sees. The compose stage walks its extraction
batches in a plain nested loop and awaits one batch at a time
(`imas_codex/standard_names/benchmark.py:1613-1615`), so the served endpoint
receives **one** compose request at a time. That is at or below 4, and it is a
property of the runner rather than of a knob:

| Stage | Concurrency against the served endpoint |
|---|---:|
| compose (name generation) | 1 in flight, serial batch loop |
| docs generation (skipped in names-only mode) | not reached |
| review quorum | remote reviewer, AIMD-governed at the default ceiling of 128 |

The local-call path bypasses the governor deliberately (`governor = None if
_use_local`, `llm.py:2384`), so an env-var ceiling would not have pinned
anything; the serial loop is what holds the pin, and it holds it without a
configuration edit.

## Why no run was attempted anyway

The bench banner prints its fixture and the extractor takes the graph's
candidate sources; a run started with the graph down fails inside extraction and
produces a report with zero candidates and `batch_errors` at zero as well, which
reads as "the model produced nothing" rather than as "the extraction never ran".
Recording the bar and the pin without a spurious run keeps the next attempt's
comparison honestly empty rather than falsely flat.

## Follow-ons this node does not own

- The four seats are provably refused by the served endpoint today (404 on the
  bare identifier). Repairing the seats to name `local/deepseek-v4.1-flash`
  would fix a measured break; this node leaves them unchanged because the bar
  it is fenced to test could not be measured.
- A benchmark report whose candidates were never extracted should not be
  written as a result, or should carry a distinct status; `benchmark.py` is
  outside this node's write fence.