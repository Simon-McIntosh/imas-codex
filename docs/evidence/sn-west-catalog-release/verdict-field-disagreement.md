# Which verdict field wins, and how many identities carry a disagreeing pair

**Answer in one line.** `name_stage` is the verdict and `refine_stop_reason` is a
cause label, not a second verdict; **62** live identities carry a disagreeing
pair, split **20 legitimate history / 42 legacy stage writes with no recorded
cause**, and the two shapes need different treatment.

Measured live on 2026-09-18 against the login-node graph (DD-pinned graph client,
`GraphClient()`), over a population of **5,130** `StandardName` nodes, corrected
the same day against a review of the first pass: which function performs the
clear at `graph_ops.py:10721`, how many read paths `refine_stop_reason` has, and
whether the 42-row arm is a live fault or legacy state. The **62** count and the
**42 / 11 / 7 / 2** breakdown are unchanged by the correction.

## The named predicate

`DISAGREEING_VERDICT_PAIR` — a row whose two name-axis fields cannot be read
together as one verdict for the identity:

```
MATCH (sn:StandardName)
WHERE (sn.name_stage = 'accepted'  AND sn.refine_stop_reason IS NOT NULL)
   OR (sn.name_stage = 'exhausted' AND sn.refine_stop_reason IS NULL)
RETURN count(sn) AS n
```

Why those two arms are the disagreement, and nothing else is:

- **`accepted` with a stop reason** — the row asserts the name is committed
  while simultaneously carrying a terminal cause for refinement having stopped.
  A reader taking the two fields together sees a committed name and a closed
  name axis in the same breath.
- **`exhausted` with no reason** — the row asserts the name axis is closed while
  the field that carries *why* is empty, so the obvious diagnostic question
  ("closed, on grammar? on collision? on budget?") has no answer in the graph at
  all. Exhaustion is the one stage whose justification lives only in
  `refine_stop_reason`.

The arms are not arbitrary: `superseded` and `reviewed` rows are **excluded**
because the writers produce reason-carrying rows in both stages legitimately (a
`reviewed` row means a rotation remains, so a recorded reason for the rotation
that just stopped is coherent). See the writer map below.

**Measured wall time: 0.062 s** (`universe_seconds`), against the 10 s ceiling.
Predicate driver: `predicate-test.py` in the node's run directory
(`~/.config/reckon/crew/runs/r-20260918T091042347055-n-the-two-verdict-fields-are-censused-for-disagreement/`).

## The count and the per-pair breakdown

**Integer count: 62.** Breakdown by `(name_stage, refine_stop_reason)` pair, with
one example identity per pair:

| `name_stage` | `refine_stop_reason` | count | example identity |
|---|---|---|---|
| `exhausted` | *(null)* | 42 | `toroidal_particle_current` |
| `accepted` | `successor_collision` | 11 | `power_of_lower_hybrid_antenna` |
| `accepted` | `grammar_invalid` | 7 | `magnetic_field_magnitude` |
| `accepted` | `attempts_exhausted` | 2 | `diamagnetic_current_density` |
| | **total** | **62** | |

`42 + 11 + 7 + 2 = 62`, and the breakdown query returns the same integer as the
predicate query (`sums_to_universe: true` in the run output). Breakdown wall
time 0.078 s.

**Instrument check.** The discovery census (all `(name_stage,
refine_stop_reason)` pairs over the whole label, 0.049 s) shows the predicate is
not trivially matching the population: 2,508 rows read `accepted` with a null
reason and 251 read `exhausted` with one, so both arms discriminate rather than
sweep. The plan's named example, `net_power_due_to_ion_cyclotron_heating`, was
re-read directly and confirmed present in the `accepted` +
`attempts_exhausted` bucket with `refine_stopped_at =
2026-09-08T11:53:51.758Z`.

## Which field the code treats as authoritative

**`name_stage` is authoritative for the verdict.** Every downstream consumer
gates on it; nothing gates on `refine_stop_reason`.

Writers of `name_stage` (all in `imas_codex/standard_names/`):

| Site | Function | What it writes |
|---|---|---|
| `graph_ops.py:17255` | `persist_reviewed_name` (`:16922`) | `sn.name_stage = $target_stage` — the name-review verdict, fenced on `WHERE sn.name_stage = 'drafted'` at `:17245` |
| `graph_ops.py:24598` | `stop_refine_name_attempt` (`:24560`) | `sn.name_stage = target_stage` where the stage is `'exhausted'` or `'reviewed'`, derived inside the write from `refine_attempts` and the terminal-reason set, fenced on claim token + stage `'refining'` |
| `graph_ops.py:12524` | `promote_stranded_reviewed` (`:12461`) | `sn.name_stage = 'accepted'` for a reviewed row whose score clears the bar |
| `promote.py:1021`, `:1584`, `:1604` | promotion path | `sn.name_stage = 'accepted'` on the catalog-promotion side |

Writers of `refine_stop_reason`:

| Site | Function | Behaviour |
|---|---|---|
| `graph_ops.py:24611` | `stop_refine_name_attempt` | **the only writer that records a cause**: `'attempts_exhausted'` when the charged budget is spent, else the caller's `reason` |
| `graph_ops.py:17273` | `persist_reviewed_name` | `CASE WHEN $target_stage <> 'exhausted' THEN sn.refine_stop_reason ...` — **carries the cause forward unchanged** on any non-exhausting outcome, including `accepted` |
| `edit.py:860` | name-steering edit stamp | **clears** it (a name hint refunds the refine budget and its diagnosis) |
| `graph_ops.py:24453` | `stage_name_for_rescore` (`:24401`) | **clears** it (a rescore buys a fresh quorum draw on the same name, not a fresh rewrite budget) |
| `graph_ops.py:10721` | `_lock_claimed_name_bindings` (`:10589`) | **clears** it when reviving a dead-end identity |

Both lines were opened and re-checked: `graph_ops.py:10589` reads
`def _lock_claimed_name_bindings(` — the clear sits in that body, inside the
`revive_dead_end_identity` `FOREACH` that re-stages the name — while
`graph_ops.py:14275` reads
`def reconcile_reviewable_name_stage(gc: Any | None = None) -> dict[str, int]:`
and never writes the field: the only `refine_stop_reason` sites of any kind in
`graph_ops.py` are `:10721`, `:17273`, `:18144`, `:24453`, `:24611`.

Consumers that gate on `name_stage`: `promote.py:1053`, `graph_ops.py:12555`,
`enrichment.py:419`, `campaign.py:299`. `refine_stop_reason` has exactly **two**
read paths, and neither gates on it: the projection at `graph_ops.py:18144`
(`", sn.refine_stop_reason AS refine_stop_reason"`, inside
`claim_refine_name_batch`, `:18109`) carries the field out with the claimed batch,
and the report at `workers.py:6924` prints the previous stop so the next attempt
can say what the last one hit. Both read a note about a past attempt; neither
branches a verdict on it.

**So the asymmetry is in the code, not just in the data.** `name_stage` decides
what may be published, enriched or exported; `refine_stop_reason` is a note left
for the next refinement attempt. A reader that treats them as two verdicts is
reading one verdict and one diagnostic.

## Legitimate history or legacy state — the split is measurable

The predicate's two arms are two different shapes and must not be reported as
one number. A second probe separates them:

| Shape | rows | `refine_stopped_at` present | `refine_attempts > 0` | `reviewed_name_at` present |
|---|---|---|---|---|
| `accepted` + reason | 20 | 20 | 20 | 20 |
| `exhausted` + null | 42 | **0** | 2 | — |

**`accepted` + reason — 20 rows, LEGITIMATE HISTORY.** All 20 carry
`refine_stopped_at`, a positive `refine_attempts` and a `reviewed_name_at`. The
sequence is real and in order: the identity refined, rotations were charged, the
loop stopped with a recorded cause, and the identity was later accepted, with the
reason left standing as a true statement about that earlier event
that was never cleared, because the accept path (`graph_ops.py:17273`) carries
it forward by design and only the *steering*, *rescore* and *revive* paths clear
it. Nothing is corrupt; the row is a stale diagnosis attached to a committed
name. The remedy, if wanted, is a clear on the accept path, not a repair of
these 20 rows.

**`exhausted` + null — 42 rows, LEGACY STATE.** Not one of the 42 carries a
`refine_stopped_at`, and 40 of 42 carry `refine_attempts` 0 or absent; the
remaining **2**, `flux_surface_normal_momentum_convection_velocity` and
`flux_surface_normal_neutral_energy_diffusion_coefficient`, carry
`refine_attempts = 3` with the reason and every stamp gone. The first pass read
this arm as a live write-ordering fault — a stage reached by a route that wrote
`name_stage` and skipped the diagnosis. A second measurement over the same arm
refutes that reading: none of the 42 carries any write stamp a current writer
leaves.

| stamp | in the 42-row arm | over the label (5,130 rows) |
|---|---|---|
| `updated_at` | **0 / 42** | 3,279 |
| `run_id` | **0 / 42** | 576 |
| `claim_token` | **0 / 42** | 0 — unused label-wide, so this column discriminates nothing |
| `claimed_at` | **0 / 42** | — |
| `refine_stopped_at` | **0 / 42** | — |

The `updated_at` zero is decisive, because **every current statement that writes
`name_stage` — including one that leaves the row `exhausted` — stamps
`updated_at` in the same statement**: `stop_refine_name_attempt`
(`graph_ops.py:24598`), `persist_reviewed_name` (`:17249`), the atomic supersede
fold (`edit.py:1533`–`:1534`), and `cancel_staged_rename`
(`provenance_lifecycle.py:300`–`:312`). A live write-ordering fault in the
current code would have committed the stage and its stamp together, so these 42
predate those writers: **legacy state, not a live defect** — the two
attempt-carrying rows included, an earlier charge whose stamp is likewise gone
rather than a partial write of the current path. **The remedy is a data cleanup
that re-derives their stage through a current writer, not a code fix**, because
no current path can produce the state.

## What a read should do

Gate on `name_stage`. `refine_stop_reason` is a cause label for the name axis
and must never be read as the identity's verdict; a report that renders the pair
as one status is wrong on all 62 rows and wrong in both directions. Anyone
enumerating "parked" or "exhausted" identities by `refine_stop_reason` alone
misses the 42 that are `exhausted` with no reason, and anyone enumerating
"accepted" identities by that field alone will wrongly report 20 committed names
as stopped.

## Login-node graph exception

This node used the standing login-node exception: `NEO4J_URI` is unset here, and
`GraphClient()` answers on the login node; inside a SLURM step the same client
resolves a loopback endpoint (`bolt://localhost:17687`) it never establishes, so
the call that answers here fails there. The census, and this correction pass, ran
on the login node. The census took three indexed reads and a whole-label grouped
count over 5,130 rows (longest 0.078 s); the correction pass took two indexed
reads over the 42-row arm and a whole-label grouped count (longest 0.049 s). Every
query stayed under the 10 s per-query ceiling, and no other login-node compute was
paired with either.
