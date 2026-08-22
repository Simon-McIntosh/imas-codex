# Ordinary-source bootstrap and governed attachment

## Outcome

**COMPLETE — two exact DD sources materialized and attached.** The ordinary
exact-path extractor qualified both requested Data Dictionary paths, and
`merge_standard_name_sources` captured immutable DD 4.1.1 snapshots and created
their `StandardNameSource -> FROM_DD_PATH -> IMASNode` provenance. The closed
signed unbound-source program then attached both rows to their independently
accepted targets.

The signed preview covered **2 authority rows = 2 admitted + 0 refused** and
returned `outcome=would_apply`. Its refusal list is exactly `[]`. The signed
apply returned `outcome=applied`, `changed=2`, and `receipt_rows=2` under
manifest SHA-256
`19bcb8d9f5a6a1cfc2128f1823bda1a9e6df7cbe6215c01e2fb9a300a09b90da`.
The same-hash replay returned `outcome=already_applied`, `changed=0`, and
`persistent_writes=0`.

The canonical live unsourced-name census fell from **6 before to 4 after**, a
difference of **2**, exactly equal to the number of applied rows. The complete
operation used **0 LLM events** and cost **$0.000000**.

## Exact source materialization

The invocation used `extract_specific_paths` for the two named paths, then fed
the returned ordinary extraction items to `merge_standard_name_sources`. That
is the same immutable-snapshot and `FROM_DD_PATH` writer used by focused pool
seeding; no source node was synthesized with direct Cypher.

| Exact DD path | Source after extraction | DD snapshot | Materialization refusal |
|---|---|---|---|
| `plasma_transport/model/ggd/neutral/momentum/d_parallel` | `dd:plasma_transport/model/ggd/neutral/momentum/d_parallel`; **`status=extracted`** | DD **4.1.1**, snapshot pinned, one exact `FROM_DD_PATH` backing | none |
| `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol`; **`status=extracted`** | DD **4.1.1**, snapshot pinned, one exact `FROM_DD_PATH` backing | none |

The `extracted` intermediate status is additionally enforced by the closed
signed program: a source whose live status is anything else is refused with
`signed ordinary source lifecycle does not match attachment authority`. Both
rows have durable apply receipts, so both passed this exact precondition before
the atomic attachment changed their status to `attached`. Requested **2**,
materialized **2**, refused **0**; there is therefore no unreported path-level
materialization failure.

Had exact extraction returned no eligible batch, the driver would have recorded
`ordinary extraction returned no eligible batch for the exact DD path`. Had
immutable snapshot persistence failed, it would have retained the exception
class and message after `ordinary extraction source materialization failed:`.
Neither mechanism fired for this cohort.

## Accepted targets and semantic bindings

| DD source-path binding | Accepted target and review evidence | Semantic distinction |
|---|---|---|
| `plasma_transport/model/ggd/neutral/momentum/d_parallel` → `parallel_neutral_momentum_diffusion_coefficient` | `name_stage=accepted`, `validation_status=valid`, name score **0.98125**. Description: “Effective parallel diffusivity for transporting momentum by the aggregate neutral population along magnetic-field lines.” | A diffusion coefficient describes parallel spreading of aggregate neutral momentum; it is not an advective velocity. |
| `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` → `poloidal_neutral_internal_state_momentum_convection_velocity` | `name_stage=accepted`, `validation_status=valid`, name score **0.9625**. Description identifies the poloidal effective advective velocity transporting momentum for one neutral-species internal state, distinct from bulk particle flow. | The accepted `…convection_velocity` successor is the quorum-earned replacement for superseded `poloidal_neutral_internal_state_momentum_convected_velocity`; the attachment does not revive the rejected spelling. |

## Signed preview, apply receipt, and replay

The retained authority contains exactly the two source ids above. Its file
SHA-256 is
`1202c229321fb1350bc8b91c59685c8afbb32ecc03b89d463c20dd1f543d4ec8`,
and its canonical signed-payload SHA-256 is
`615d8b7064b0d8fc15a578b2147d5066ba3631309d1065981ea7aa5e958b6c56`.

| Governed measure | Result |
|---|---|
| Preview | `outcome=would_apply`; authority rows **2**; admitted **2**; refused **0**; exact refusals `[]` |
| Apply | `outcome=applied`; `changed=2`; `receipt_rows=2`; manifest `19bcb8d9f5a6a1cfc2128f1823bda1a9e6df7cbe6215c01e2fb9a300a09b90da` |
| Exact receipt query | **2 rows**, selected by `run_id=r-20260822T220555777674-n-sourcebootstrap` plus operation and manifest; both rows pin the retained file and payload hashes |
| Replay | `outcome=already_applied`; `changed=0`; `receipt_rows=2`; `persistent_writes=0` |
| Replay counter proof | `StandardNameChange=7,900`, `PRODUCED_NAME=5,779`, and `LLMCost=27,909` before and after; all deltas **0** |

The preview evidence survived through the apply invariants even though its
buffered JSON print did not. The applying driver asserted `would_apply` before
it could enter apply. The durable result then contains one receipt for every
signed authority row, with both receipts carrying the complete two-id
`cohort_admitted_ids`. Therefore the only possible preview partition was the
recorded **2 admitted + 0 refused**. This is stronger than inferring success
from a bare global counter.

## Four-mirror post-apply reread

| Target identity | `PRODUCED_NAME` | Backing `HAS_STANDARD_NAME` | Source scalar/lifecycle | `source_paths` occurrence |
|---|---:|---:|---|---:|
| `parallel_neutral_momentum_diffusion_coefficient` | exactly **1**, from `dd:…/d_parallel` | exactly **1**, same target | exact target id; `attached` | exactly **1** |
| `poloidal_neutral_internal_state_momentum_convection_velocity` | exactly **1**, from `dd:…/v_pol` | exactly **1**, same target | exact target id; `attached` | exactly **1** |

Both sources reread with `dd_snapshot_pinned=true`, `dd_version=4.1.1`, one
exact backing id, and no additional bound or projected identity. Thus each
accepted target gained one producing source beside the matching DD projection,
scalar mirror, and source-path mirror without an identity fold.

## Unsourced-name and cost accounting

The before count is the canonical ledger census recorded immediately before
this node by the accepted neutral-velocity disposition. The post-apply query
used `find_provenance_orphans`, which includes every materialized live name
without an incoming `PRODUCED_NAME` and excludes terminal names, structural
scaffolds, and deterministic error siblings.

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Live unsourced Standard Names | **6** | **4** | **−2** |
| Requested target identities without producers | **2** | **0** | **−2** |

The four remaining identities are
`fast_ion_charge_state_power_at_inside_flux_surface`,
`neutron_flux_due_to_fusion`,
`tendency_of_total_thermal_plasma_internal_energy`, and
`toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions`.
They retain their separately adjudicated collision, predecessor, distinct-path,
or no-candidate conditions; this invocation did not broaden into them.

| LLM accounting | Result |
|---|---:|
| `LLMCost` rows with this run id | **0** |
| Exact run cost | **$0.000000** |

This zero is expected rather than missing accounting: exact DD qualification,
immutable source snapshotting, signed preview/apply, graph rereads, and replay
use no model seat.

## Interruption and durable recovery

The first attempt stopped before extraction because a read-only Cypher query
used implicit aggregation grouping rejected by Neo4j. The corrected applying
invocation committed both source attachments at
`2026-08-22T22:15:49.479Z`, then remained silent in a network poll and was
terminated at the workstation's five-minute process limit, exiting 143 before
its buffered result print.

No mutation was guessed or repeated. Recovery first queried the exact run id
and found the two same-manifest receipts, matched their authority hashes to the
retained bytes, reread all four mirrors, and invoked only the same-hash replay.
That replay returned `already_applied` with zero persistent writes. The
interrupted process therefore affects presentation, not transaction validity or
evidence completeness.

## Durable artifacts

The machine-readable recovered result is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T220555777674-n-sourcebootstrap/source-bootstrap-attach-result.json`
(SHA-256
`d93404f4e5572c8d599acb9de57bbd420fe2cc4d4e2021ca554f6eb8792d86c7`).
The signed authority, applying driver, exact state inspector, recovery driver,
and complete logs are retained beside it. The successful recovery log is
`receipt-recovery.log` (SHA-256
`bc4184baa0055215e6190c4810c0ab6cbaa7db29053b7d03f84e4f9226cec14b`)
and terminates with `EXIT=0`.
