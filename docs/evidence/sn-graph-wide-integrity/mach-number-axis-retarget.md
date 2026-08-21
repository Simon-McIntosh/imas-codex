# Parallel Mach number exact-source retarget

## Outcome

**Blocked without production mutation.** The exact source remains bound to
`mach_number`; no `StandardNameChange` receipt was written, no other source row
changed, and the required apply/replay/test evidence was therefore not produced.

The production driver was deliberately fail-closed. Its first invocation stopped
before the operator because the live DD description had been enriched from the
older short text, `Parallel Mach number`, to a longer description explicitly
covering the parallel Mach number measured by a reciprocating Langmuir probe.
The stable semantic check was corrected to require that explicit meaning instead
of byte-equality with stale prose.

The second invocation also stopped before the operator. Its freshly measured
closure found that
`dd:langmuir_probes/reciprocating/plunge/mach_number_parallel` is the final
direct live producer of `mach_number`: `other_live_old_producers=0`. The driver
required at least one other direct producer, so it refused rather than silently
turning the generic predecessor into a directly unsourced identity.

## Authority and measured topology

The intended direction remains supported by all semantic evidence:

- exact source:
  `dd:langmuir_probes/reciprocating/plunge/mach_number_parallel`;
- current scalar, live binding, and DD projection: `mach_number`;
- intended target: accepted, valid `parallel_mach_number`, unit `1`, reviewer
  score `0.99375`;
- DD authority: unit `1`; description explicitly identifies the parallel Mach
  number measured by a reciprocating Langmuir probe;
- topology: `parallel_mach_number` has projection axis `parallel` and
  `HAS_PARENT` `mach_number`;
- deterministic compatibility guard: the exact source-to-target pair was not
  reached on the second invocation because the stronger producer-closure check
  refused first;
- corroborating adjudication:
  `docs/evidence/sn-graph-wide-integrity/unsourced-name-adjudication.json`
  classifies the target as recoverable from this exact DD path while explicitly
  retaining `mutation_authority=classification_only`.

The generic predecessor therefore has a live accepted structural child after
the intended move, even though it would have no direct producing source. That is
the important distinction: the failed check proves absence of another direct
producer, not absence of structural justification.

## Required next action

Use a fresh execution node. Keep the exact one-row source compare-and-set,
accepted-valid target and unit checks, parallel-axis relation, shared attachment
guard, synchronized edge/scalar/DD-projection/cache mutation, one deterministic
receipt, collateral digest, and zero-write replay. Replace only the rejected
closure predicate:

- require either another live direct producer of `mach_number` **or** an
  accepted-valid live structural child whose `HAS_PARENT` edge survives the
  move;
- in this case require the latter witness to be exactly
  `parallel_mach_number`, projection axis `parallel`;
- after apply, require `mach_number` to have zero direct producers but the exact
  live structural child, while `parallel_mach_number` has the exact DD source;
- then require `changed=1`, `receipt_rows=1`, `untouched_changed=0`, a replay at
  `already_applied` with `changed=0`, and one passing explicit-axis graph test.

This is the recommended route because it preserves the plan’s explicit
source-axis decision and the repository’s established rule that a structural
parent may be legitimately unsourced. If that structural closure is not accepted
as sufficient, the alternative is to hold the migration until an authoritative
second direct producer for `mach_number` exists.

## Quantitative gate status

| Measure | Required | Observed |
|---|---:|---:|
| Applying change rows | 1 | 0 — preflight refusal |
| Migration receipt rows | 1 | 0 |
| Untouched source rows changed | 0 | 0 — operator never entered |
| Replay outcome | `already_applied`, `changed=0` | not run |
| Explicit-axis graph test | 1 passed | not run after mutation; mutation did not occur |
| Other live direct producers of `mach_number` | closure-dependent | 0 |

## Durable logs and driver

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T130959992588-machretarget/apply-driver-preflight-refusal.log`
  — first zero-write refusal on stale description equality;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T130959992588-machretarget/apply-driver.log`
  — second zero-write refusal on final-direct-producer closure;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T130959992588-machretarget/apply_mach_parallel_retarget.py`
  — retained driver with exact authority, mutation, collateral, receipt, and
  replay checks.

