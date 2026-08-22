# Scalar-mirror reconciliation evidence

## Outcome

**COMPLETE.** The signed semantic-mirror operator reconciled the one remaining
live scalar mismatch without changing either source-to-name edge. The graph-wide
scalar-mirror-mismatch census moved from **1 to 0**.

The exact source is
`dd:plasma_sources/source/ggd/neutral/state/momentum/phi`, the toroidal
component of a resolved-neutral-state momentum source. Its surviving live
identity is `toroidal_neutral_internal_state_torque_density`. The scalar mirror
changed from `neutral_internal_state_torque_density` to that surviving identity.
The older target remains present only as an exhausted historical binding; it is
not a live competitor and this authority did not remove it.

| Required measure | Observed result |
|---|---:|
| Signed rows | **1** |
| Admitted | **1** |
| Refused | **0** |
| Admitted + refused | **1 = 1 signed row** |
| Mutated rows | **1** |
| Receipt rows | **1** |
| `StandardNameChange` baseline | **7,786** |
| `StandardNameChange` after apply | **7,787** |
| `StandardNameChange` delta | **+1 = 1 receipt** |
| Scalar-mirror mismatches | **1 → 0** |
| Replay outcome | **`already_applied`** |
| Replay changed | **0** |
| Replay persistent writes | **0** |

No row was refused. The signed result retains an empty refusal array, so no
refusal reason is absent or summarized away.

## Signed source closure

Immediately before preview, the source was `status='composed'`, had no claim
timestamp or token, and carried these two relationships:

| Target identity | Lifecycle | Authority meaning |
|---|---|---|
| `toroidal_neutral_internal_state_torque_density` | accepted | **sole live target and survivor** |
| `neutral_internal_state_torque_density` | exhausted | historical terminal binding, not live authority |

The DD-side `HAS_STANDARD_NAME` projection already selected
`toroidal_neutral_internal_state_torque_density`. The apply therefore changed
only `StandardNameSource.produced_sn_id`; it added no projection and removed no
binding. This preserves the adjudicated physics: the `phi` leaf requires the
toroidal component, while the path, description, and unit
`kg.m^-1.s^-2` identify a resolved-neutral-state torque-density source.

The operator was `repair_scalar_projection_mismatches`, whose authority is the
sole non-terminal `PRODUCED_NAME` target plus the exact upstream projection. It
derived and locked that closure afresh inside the applying transaction. The
authorized manifest digest was:

`1be2c0525ae774f9c1753a6e3bfc35afb05acaac98bd6c0d6169233d86d33876`

## Exact receipt attribution

Mutation was proved by querying the durable receipt with both of its own keys,
not by guessing an operation name and not by relying on a bare counter total:

- `run_id=r-20260822T010423293355-scalarfix`
- `manifest_sha256=1be2c0525ae774f9c1753a6e3bfc35afb05acaac98bd6c0d6169233d86d33876`

That exact query returned one receipt:

`sn-change:semantic-mirror-repair:1be2c0525ae774f9c1753a6e3bfc35afb05acaac98bd6c0d6169233d86d33876`

The receipt's `from_name` records
`neutral_internal_state_torque_density`; its `to_name` records
`toroidal_neutral_internal_state_torque_density`. A second query by run ID alone
returned the same one receipt and the same one manifest digest, excluding a
hidden receipt for this run under another digest.

## Replay and absence-of-write proof

Replay used the same source set, reason, run ID, and manifest digest. It returned
`already_applied` with `changed=0`. Before and after replay, the exact
run-and-manifest receipt query returned the same single receipt and the exact
source snapshot was byte-equivalent at the structured-value level.

The following live counters were also identical across replay:

| Persistent graph measure | Before replay | After replay | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,787 | 7,787 | **0** |
| `PRODUCED_NAME` relationships | 5,780 | 5,780 | **0** |
| `HAS_STANDARD_NAME` relationships | 5,390 | 5,390 | **0** |
| `HAS_INTERNAL_CHANGE` relationships | 4,471 | 4,471 | **0** |
| `LLMCost` rows | 27,631 | 27,631 | **0** |

Together, the unchanged exact receipt closure, exact source snapshot, and graph
counters establish `persistent_writes=0` for replay. The post-apply graph-wide
query found no composed or attached source whose scalar disagreed with its sole
live target.

## Invocation record

- Machine-readable apply, receipt, replay, and census result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T010423293355-scalarfix/scalar-mirror-reconcile-result.json`
  (SHA-256
  `b383fb10817bb59fb21be3190389197ccf09fc154fcb42e30c04b8236158cce6`).
- Successful invocation diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T010423293355-scalarfix/scalar-mirror-reconcile-attempt2.log`
  (SHA-256
  `ae0ddca4bb511661c5ff1513dcb7bc42ebb0f5be10b89a7fb785d3dec6436dd2`).
- Applying source commit: `6993c57e3d59fb45dd82b5ed459ed54f5bb4bb4a`.

The first harness launch stopped before preview or mutation because its
evidence-only preflight counted the exhausted historical edge as live. The
corrected invocation used the same live-versus-terminal predicate as the signed
operator. Its observed `StandardNameChange` baseline remained exactly 7,786,
which also proves that the stopped preflight wrote nothing.
