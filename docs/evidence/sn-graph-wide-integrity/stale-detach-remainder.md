NEEDS-HELP: The live StandardNameChange baseline is already 7,570, so the fenced 7,517-plus-receipt proof is impossible before this node performs any mutation.

tried: Read the live plan at version 196, including the complete remaining-work scope and latest execution evidence; verified the committed 58-row stale-source authority; connected to the production graph read-only; classified every signed row against its current source, binding, target-producer, and direct-child topology; and re-read the global counters after preflight.

options: First, audit the 53 StandardNameChange rows already written after 7,517 and refresh the node baseline to 7,570 if all 53 belong to authorized preceding nodes, then redispatch against a quiet graph. Second, retain 7,517 as immutable evidence authority and recover the graph to that exact ledger state under separate rollback authority before redispatch. Third, expand the implementation scope if the refreshed execution still lacks a tested operator that can consume all signed DD and derived rows, partition already-applied and refused rows, and apply the admitted remainder atomically.

leaning: Audit the intervening 53 rows and, if authorized, refresh the fence to 7,570. This preserves already-landed work and keeps the stale detach fail-closed; accepting the new baseline without that audit would conceal possible collateral.

cost-if-wrong: If any of the 53 rows is unauthorized, treating 7,570 as the new baseline would bless unrelated mutation and invalidate the no-more-than-declared receipt proof. If 7,517 remains mandatory, recovery would have to identify and reverse all later authorized changes before this detach can run, then repeat every affected downstream verification.

# Signed stale-source remainder preflight hold

Date: 2026-08-20

No production mutation was attempted. The stop occurred before an apply
manifest was generated or any transaction was opened for writes.

## Signed authority

- Authority: `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json`
- File SHA-256: `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`
- Declared row signature: `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Independently recomputed row signature: `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Signed rows: 58

The focused committed-authority test passed. A read-only topology pre-partition
accounted for all 58 rows as 3 already detached, 52 not refused by the
last-producing-binding check, and 3 refused. These are preflight classifications,
not an apply receipt: the 52-row class was not mutated and must be re-derived and
locked inside the eventual apply invocation.

## Last-producing-binding refusals

| Signed source | Refused target | Reason |
|---|---|---|
| `dd:ece/channel/t_e_voltage` | `voltage_of_diagnostic_antenna` | Detach would strip the target's final live producing source; it has no live direct child. |
| `dd:equilibrium/time_slice/profiles_1d/b_average` | `flux_surface_average_magnetic_field_magnitude` | Detach would strip the target's final live producing source; it has no live direct child. |
| `derived:neutral_state_energy_convection_velocity` | `neutral_state_energy_convection_velocity` | The signed source currently has two bindings; removing it would strip this non-structural target's final live producing source. |

The three already-detached rows retain their exact stale-detach change receipts
and have null scalars, zero live bindings, and zero matching projections.

## Hard counter conflict

| Counter | Required baseline | Live before preflight | Live after preflight | Preflight delta |
|---|---:|---:|---:|---:|
| `StandardNameChange` | 7,517 | 7,570 | 7,570 | 0 |
| `LLMCost` | 27,489 | 27,489 | 27,489 | 0 |

The live ledger is already 53 rows above the required baseline. Therefore no
receipt produced now can prove that `StandardNameChange` rose from 7,517 by
exactly this node's declared receipt-row count and no more. The read-only
preflight itself changed neither counter and made zero provider calls.

## Evidence and validation

- Preflight classification log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T150833249115-sgwi-stale-detach-remainder/live-preflight.json`
- Focused validation:
  `pytest -p no:cacheprovider tests/standard_names/test_stale_source_detach.py::test_committed_authority_signature_selects_exact_blocking_rows`
  — 1 passed, 0 failed.

An immediate replay, out-of-allowlist before/after digests, declared apply
receipt count, and post-apply dual-bound/unsourced census do not exist because
the apply was correctly withheld before mutation. They remain mandatory after
the baseline conflict is resolved.
