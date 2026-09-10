# Tail drain to the authorised ceiling with the orphan sweep running

## Purpose

Resume the capped tail drain recorded in
[`stale-claim-release-and-resume.md`](stale-claim-release-and-resume.md) now
that the orphan sweep starts unconditionally (see
[`stale-claim-release-and-resume.md`](stale-claim-release-and-resume.md)).
That prior drain stalled because each slice left claims the next slice
refused: the sweep was bundled under the same gate as the reconcile writers,
so a scoped run lost the reaper exactly when a dead worker had left claims on
rows, and the age-blind exact-name preflight then refused those rows until a
sweep ran. The repair (this repo, HEAD `563a613d5`) moves the sweep outside
that gate so it starts on every run regardless of
`--skip-global-maintenance`.

This record covers the drained tail: the census and cumulative spend before
and after, the claim state before and after each slice (with timestamps), and
whether the pipeline still reports eligible work when stopped.

All counts come from bounded, read-only `GraphClient` queries against the
live graph on 2026-09-10, immediately before and after each slice.
Definitions mirror `stale-claim-release-and-resume.md`:

- **Live** = `name_stage <> 'superseded'`.
- **Fully accepted on both axes** = `name_stage='accepted'` **and**
  `docs_stage='accepted'`.
- **Exhausted cohort** = `name_stage='exhausted'`; `contested` is empty.
- **Run scope** = live, name-terminal states excluded (`exhausted`,
  `contested`), catalog-status-terminal rows excluded (`status` in
  `superseded`/`deprecated`), and not fully accepted on both axes.
- **Missing a name score** = `reviewer_score_name IS NULL`; **missing a docs
  score** = `reviewer_score_docs IS NULL`; **no documentation text** =
  `coalesce(trim(documentation),'') = ''`.
- **Budget is cumulative across the whole campaign**: every slice's
  `--cost-limit` is 150 minus the campaign total at that point, never a fresh
  allowance.

## Before-state (2026-09-10, ~11:47 local / 09:47Z)

Picked up exactly where `stale-claim-release-and-resume.md` left off: the
graph rents the 31 claims that had blocked the previous slice 2, and the
campaign ledger sat at the same figure that record reported.

| Population | Count |
|---|---|
| Total StandardName nodes | 5,104 |
| Live (non-superseded) | 2,952 |
| **Fully accepted on both axes** | **2,451** |
| **Run scope** (live, non-terminal, not fully accepted) | **215** |
| — name_stage: reviewed 111, drafted 52, accepted 38, pending 11, refining 3 | |
| — docs_stage: pending 136, accepted 36, reviewed 24, drafted 14, exhausted 3, (null) 2 | |
| Missing name score (full live) | 408 |
| Missing docs score (full live) | 422 |
| No documentation text (full live) | 287 |
| Missing in scope: name 66 / docs 175 / no docs 115 | |

### Prior blocker state (the stalled claims)

| Measure | Count |
|---|---|
| Live claims present on entry | **31** |
| oldest `claimed_at` | 2026-09-10T07:36:44Z |
| newest `claimed_at` | 2026-09-10T07:45:42Z |

These are the exact rows the previous drain's slice 1 left when it hit its
time limit at 07:45Z; no worker ever returned for them, and they were what
made the previous slice 2 refusal ("current worker claim", age-blind) — the
wedge that repair exists to remove. They were reaped at this node's start.

### Campaign spend at entry (LLMCost ledger, `llm_at >= 2026-09-10T04:00Z`)

| Measure | Value |
|---|---|
| Campaign rows | 1,449 |
| Campaign spend | **USD 84.126801** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 65.873199** |

## The sweep is running (repair confirmation)

Both invocations started the orphan sweep unconditionally under
`--skip-global-maintenance`. Log line, slice 1 (`sn_sn-compose.log`,
2026-09-10 11:48:40 local):
`run_sn_pools: Orphan sweep loop started (interval=30s, timeout=1800s)` —
the same line appears at slice 2 (12:09:14 local). The bundled mutating
reconcile writers stayed bypassed: `reconcile complete — 0 actions`.
Maintenance-only modes still return before the sweep block.

Stale claims were released via the existing orphan-sweep turn,
`_orphan_sweep_tick(timeout_s=600)` from `imas_codex.standard_names.orphan_sweep`,
the same synchronous one-pass mechanism the pipeline's background loop uses,
not by writing `claim_token`/`claimed_at` directly and not via the global
maintenance pass.

| Sweep (each slice boundary) | Released |
|---|---|
| Pre-slice-1: `name_refining` / `stale_token_sn` | 3 / 28 = **31** (the stalled block) |
| Pre-slice-2: `name_refining` / `stale_token_sn` | 4 / 15 = **19** (slice 1's in-flight stop) |

### Claim count before and after each invocation

| Moment (local) | Live claims | `claimed_at` range |
|---|---|---|
| Before node (stalled block) | 31 | 07:36:44Z–07:45:42Z (hours old, refused by preflight) |
| After slice 1 stopped | 19 | 09:48:40Z–09:57:13Z (slice 1's own window) |
| Before slice 2 (after age-out + sweep) | 0 | — |
| After slice 2 stopped | **12** | 10:09:15Z–10:15:56Z (slice 2's own window) |

Claims no longer accumulate as stranded inventory: every stop leaves only the
just-stopped slice's in-flight rows, all minutes old, and they age past the
600 s orphan threshold a few minutes after the stop. The next slice's
boundary sweep clears them exactly as this node cleared the 31 that had
blocked the previous drain. The repair took.

## Slice 1 (run started 2026-09-10 11:48:38 local / 09:48:38Z)

`--name <215 identities> --skip-global-maintenance --cost-limit 65.87 --time 8`

Run record (`SNRun`, started 09:48:37.955Z): `stop_reason=time_limit_reached`,
`cost_spent=USD 11.424299`, `cost_limit=65.87`, `elapsed_s=542.5`,
`names_reviewed=42`, `names_regenerated=13`.

Post-slice ledger (`10:03Z` read):

| Measure | Value |
|---|---|
| Campaign rows | 1,585 |
| Campaign spend | **USD 95.551100** |
| Remaining under the 150 USD ceiling | **USD 54.448900** |
| Fully accepted both axes | **2,469** (+18) |
| Run scope | 191 (−24) |
| Missing name (full live) | 394 (−14) |
| Missing docs (full live) | 410 (−12) |

The ledger delta (95.551100 − 84.126801 = 11.424299) matches the run record
exactly.

## Slice 2 (run started 2026-09-10 12:09:12 local / 10:09:12Z)

The 15 slice-1 stragglers were under the 600 s orphan age at the boundary
(claimed 09:56:00Z–09:57:13Z), so the pre-slice-1 sweep could not touch them;
they aged out minutes later and the boundary sweep cleared all 15, restoring
the full 191-identity scope.

`--name <191 identities> --skip-global-maintenance --cost-limit 54.45 --time 8`

Run record (`SNRun`, started 10:09:12.581Z): `stop_reason=time_limit_reached`,
`cost_spent=USD 4.275100`, `cost_limit=54.45`, `elapsed_s=542.3`,
`names_reviewed=14`, `names_regenerated=3`.

## After-state (2026-09-10, ~12:19 local / 10:19Z)

| Measure | Entry | Final | Delta |
|---|---|---|---|
| Fully accepted both axes | 2,451 | **2,478** | **+27** |
| Run scope | 215 | **179** | −36 |
| Missing name score (full live) | 408 | 391 | −17 |
| Missing docs score (full live) | 422 | 404 | −18 |
| No documentation text (full live) | 287 | 282 | −5 |
| `name_stage=exhausted` (full live) | 284 | 293 | +9 |
| Live claim tokens at stop | 31 | **12** | −19 |

Run scope composition at stop — name_stage: reviewed 109, drafted 31,
accepted 25, pending 11, refining 3. docs_stage: pending 126, reviewed 24,
accepted 23, exhausted 3, (null) 2, drafted 1. Missing in scope: name 45 /
docs 139 / no docs 105.

### Cumulative spend (final for this record)

| Measure | Value |
|---|---|
| Campaign rows (`llm_at >= 2026-09-10T04:00Z`) | 1,628 |
| Campaign spend | **USD 99.826166** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 50.173834** |
| Spend this node (2 slices) | USD 15.699365 |

### Does the pipeline still report eligible work when you stop?

Yes. At the stopped state the run scope is **179 identities**, of which the
12 slice-2 in-flight rows are the only ones carrying a claim; 167 are
unclaimed and immediately eligible. The pipeline processes real work to a
clean time-limit stop in both directions (2-3 slices would spend the
remaining ~USD 50 at the measured ~USD 4–12 per 9-minute slice of
review-driven burn). The previous wedge — hours-old claims with no worker
process to sweep them — is gone: the repair starts the reaper on every
scoped run, and the boundary sweep returns claims abandoned by a stopped or
dead slice within minutes of their 600 s age.

## Fences respected

- `--skip-global-maintenance` on every invocation (the graph-wide reconcile
  writers carry an unrelated open delete defect), `--name` scopes preflight
  the exact set atomically and never seeds DD sources, `--time 8` bounds each
  slice to eight minutes, `--cost-limit` is the remaining cumulative ceiling
  at each launch. No `--reseed`, no `--force`.
- The exhausted cohort (`name_stage='exhausted'`, refine cap spent) is
  excluded from scope by definition; the run never reached it.
- Live graph work ran on the login node (the Neo4j tunnel is login-node-local);
  every query above is a bounded indexed read. No CLI output was piped or
  redirected; the run's own log (`~/.local/share/imas-codex/logs/sn_sn-compose.log`)
  is the evidence. The CLI's non-zero exit on each slice is the expected
  `time_limit_reached` signal (`_require_terminal_drain` refuses exit 0 until
  `no_eligible_work`), not a crash: both `SNRun` records are
  `stop_reason=time_limit_reached` and the ledger sums match the run-reported
  spend exactly.
- No figure was produced and no image read: this lane is not multimodal.

---

# Second drain round (2026-09-10, 10:39Z–11:34Z)

A second dispatch on the same fence: spend the remaining authorised campaign
budget down from the 99.826166 USD the first round recorded toward the 150.00
USD ceiling through the same ordinary scoped pools. The tail was still
time-bound, not exhausted: **run scope 179 identities** at current HEAD, quoted
from the pre-round census below. The round ended on **this node's own wall
clock** after one slice: the ceiling was not reached (final cumulative campaign
spend 100.660059 USD) and no slice reported no eligible work (both slice 1's
stop and the round-ending decision happened with eligible work still pending).
The second planned slice never launched because the 55-minute fence expired
while this node's first slice was still completing.

## Pre-round census (2026-09-10, ~10:41Z, current HEAD `b0f254d0f`)

| Population | Count |
|---|---|
| Total StandardName nodes | 5,109 |
| Live (non-superseded) | 2,952 |
| **Fully accepted on both axes** | **2,478** |
| **Run scope** (live, non-terminal, not fully accepted) | **179** |
| — name_stage: reviewed 109, drafted 31, accepted 25, pending 11, refining 3 | |
| — docs_stage: pending 126, accepted 23, reviewed 24, drafted 1, exhausted 3, (null) 2 | |
| Missing name score (full live) | 391 |
| Missing docs score (full live) | 404 |
| No documentation text (full live) | 282 |
| Missing in scope: name 45 / docs 139 / no docs 105 | |
| Live claims on entry | **12** (10:09:15Z–10:15:56Z, prior round's in-flight) |

Campaign ledger at entry (`llm_at >= 2026-09-10T04:00Z`): rows 1,628,
spend **USD 99.826166**, remaining under the 150 USD ceiling **USD 50.173834**.

## The launch gate: claims must be reaped before preflight

As in the first round, the age-blind exact-name preflight refuses any row
carrying a worker claim at launch, and the run's background sweep starts after
preflight, so a stale block must be reaped first. The 12 claims found at entry
were the prior round's just-stopped slice's in-flight rows, ~26 minutes old
(past the 600 s orphan age). One `_orphan_sweep_tick(timeout_s=600)` turn
released them: `name_refining` 3 + `stale_token_sn` 9 = **12**; claims after 0.
The run's own sweep then started on every invocation regardless of
`--skip-global-maintenance`, as recorded in the first round.

## Slice 1 (run started 2026-09-10 11:05:42.104Z)

Exact invocation (identity list = the 179 in-scope `StandardName.id` values,
space-joined; retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260910T103903396923-n-srr-drain-the-remaining-ceiling/scope_ids.txt`):

```
imas-codex sn run --name <179 in-scope identities> --skip-global-maintenance --time 18 --cost-limit 50.17
```

`--cost-limit 50.17` = 150.00 − 99.826166 (cumulative campaign total measured
at launch, truncated to two decimals). No `--reseed`, no `--force`.

Run record (`SNRun`, started 11:05:42.103934Z): `stop_reason=stalled`,
`cost_spent=USD 0.833893`, `cost_limit=50.17`, `elapsed_s=764.976`,
`names_reviewed=5`, `names_composed=0`, `names_regenerated=0`.

The stop was not a clean time-limit stop: `run_pools` logged "no forward
progress for ~600s despite pending work … overdue_in_flight={'review_name':
711.1} — wedged residue; signalling graceful shutdown" (11:17:36 local). One
`review_name` LLM call stayed in flight 711 s and only completed during the
60 s stop grace (`persist_reviewed_name: particle_flux_normalized_due_to_e_cross_b_drift
→ name_stage=reviewed`, score 0.625, 11:18:27 local). A single wedged call,
not an empty work queue: the run stopped with 6 `review_name` items still
pending. The expected `time_limit_reached` exit was never the signal; the CLI
exited non-zero with the stalled summary, which is the pipeline's degraded-stop
contract, not a crash.

### Claim count before and after slice 1

| Moment | Live claims | `claimed_at` range |
|---|---|---|
| Pre-round entry (stalled prior round) | 12 | 10:09:15Z–10:15:56Z |
| Pre-slice-1 after boundary sweep | 0 | — |
| Pre-slice-1 launch (preflight passed) | 0 | — |
| After slice 1 stopped | 8 | 11:05:44Z–11:17:25Z |

### Post-slice-1 ledger (`11:20Z` read)

| Measure | Value |
|---|---|
| Campaign rows | 1,641 |
| Campaign spend | **USD 100.660059** |
| Remaining under the 150 USD ceiling | **USD 49.339941** |
| Fully accepted both axes | 2,480 (+2) |
| Run scope | 176 (−3) |

The ledger delta (100.660059 − 99.826166 = 0.833893) matches the slice-1 run
record exactly.

## Slice 2 was planned but the wall clock ended the round first

At the slice-2 boundary (`~11:20Z`) the sweep released a further 3
`name_refining`, leaving **5 live claims**, all stamped 11:17:25Z — the
grace-period completions of the just-stopped slice 1, ~3 minutes old and under
the 600 s orphan age, so the sweep correctly left them. They sit on the five
rows the previous four names above came from and were excluded from the planned
slice-2 launch set (171 identities; scope fell 179 → 176 in net after the
sweep's refining→reviewed return) rather than being directly written — the
reaper clears claims, code never does. The slice-2 invocation was staged at
`--cost-limit 49.33` (150.00 − 100.660059). It never launched: with the
preflight, boundary re-census and this staging, this node crossed its 55-minute
fence before a second slice could fit, and overrunning the fence is the one
thing the contract forbids.

## After-state (final for this round, 2026-09-10 ~11:31Z)

| Measure | Entry | Final | Delta |
|---|---|---|---|
| Fully accepted both axes | 2,478 | **2,480** | **+2** |
| Run scope | 179 | **176** | −3 |
| Missing name score (full live) | 391 | 391 | 0 |
| Missing docs score (full live) | 404 | 404 | 0 |
| No documentation text (full live) | 282 | 282 | 0 |
| Live claim tokens at stop | 12 | **5** | −7 |

Run scope composition at stop — name_stage: reviewed 113, drafted 27, accepted
25, pending 11. docs_stage: pending 126, reviewed 24, accepted 22, (null) 2,
exhausted 2. (The three `refining` rows returned to `reviewed` at the
boundary sweep; one `drafted` moved up; the accepted/docs-accepted count moved
+1 each axis.)

### Cumulative spend (final for this record)

| Measure | Value |
|---|---|
| Campaign rows (`llm_at >= 2026-09-10T04:00Z`) | 1,641 |
| Campaign spend | **USD 100.660059** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 49.339941** |
| Spend this round (1 launched slice) | USD 0.833893 |

The cumulative campaign total **100.660059 USD** is below the 150.00 USD
ceiling; no slice breached it.

### Which stop condition ended the round?

**This node's own wall clock.** The ceiling was not reached (49.34 USD still
unspent) and the pipeline was not out of work — slice 1 stalled on a single
wedged LLM call with 6 `review_name` items still pending, and the 5 live
claims at the final census are unreaped only because they were under the
orphan age at that moment (they cross 600 s and clear on the next run's
boundary sweep, exactly as the first round's blocks did). The round stopped
when the 55-minute fence expired with one launched slice spent and a second
sized and staged but not launched — overrunning the fence is a contract
violation, under-spending the authorised ceiling is not.

## Fences respected

- `--skip-global-maintenance` on the invocation, `--name` scoped preflight the
  exact 179-identity set atomically, `--time 18` bounds the slice, `--cost-limit`
  is the remaining cumulative ceiling at launch. No `--reseed`, no `--force`.
- Stale claims were released only via the orphan-sweep turn
  (`_orphan_sweep_tick(timeout_s=600)`, the pipeline's own precondition), never
  by writing `claim_token`/`claimed_at` directly; the sub-age in-flight claims
  from the stopped slice were excluded from the next launch set, not cleared.
- No signed manifest apply was attempted; a peer owns that path.
- Live graph work ran on the login node (the Neo4j tunnel is login-node-local);
  every query above is a bounded indexed read under ten seconds. No CLI output
  was piped or redirected; the run's own log
  (`~/.local/share/imas-codex/logs/sn_sn-compose.log`) is the evidence, and the
  ledger sums match the run-reported spend exactly.
- No figure was produced and no image read: this lane is not multimodal.
