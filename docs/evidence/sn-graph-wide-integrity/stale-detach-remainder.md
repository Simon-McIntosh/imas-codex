NEEDS-HELP: The signed stale-detach operator can execute only 36 of the 52 live admissions; five DD rows and all eleven admitted derived rows require an operator repair outside this node's write fence.

tried: Re-read live plan version 197 and its authorized attribution of the 53 intervening `remove_derived_parent` rows; verified the committed 58-row lifecycle signature; reused the preceding read-only production classification; then invoked the public signed operator write-free against all 52 admissions, each DD admission, and finally the 36-row subset the operator accepts. The complete cohort and the 16 unsupported admissions failed before mutation; the 36-row subset returned `would_apply` but was deliberately not applied because it would not satisfy the node's complete partition receipt.

options: First, expand the write fence to repair `detach_signed_stale_source_bindings` so its signed closure supports derived sources, multi-binding DD sources, and signed scalar/binding mismatches while retaining the last-producing-binding guard, disposable-graph tests, atomic apply, and replay. Second, authorize a new composite operator that consumes the same committed signature and joins the existing DD detach and structural-source release mechanisms in one atomic manifest-bound transaction. Third, narrow the semantic authority and re-sign separate executable cohorts, but that weakens the requested single 58-row partition and is not recommended.

leaning: Repair the existing signed stale-source operator. The lifecycle authority already declares all 58 dispositions and the live classifier already identifies the exact three last-producer refusals; widening the existing operator's closure model preserves one signature, one partition, one receipt vocabulary, and one replay contract.

cost-if-wrong: A DD-only partial apply would add 36 ledger rows and change the closure seen by the eventual composite repair, forcing a new baseline, new manifest, and repeated collateral proof while still leaving 16 admitted stale bindings live. A generic detach that mishandles derived sources or scalar mismatches could orphan a non-structural target, delete the wrong projection, or record an incomplete receipt.

# Signed stale-source remainder operator hold

Date: 2026-08-20

No production mutation was attempted. The node stopped before apply because the
only tested signed operator cannot express the complete admitted subset required
by the committed lifecycle authority and the node's quantitative exit measure.

## Signed authority and refreshed baseline

- Authority: `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json`
- File SHA-256: `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`
- Declared and independently verified row signature: `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Signed rows: **58** = **46 DD + 12 derived**
- Node-owned pre-transaction baseline: `StandardNameChange=7570`, `LLMCost=27489`
- Write-free probe result: both counters remained byte-identical at **7570 / 27489**

The plan now authorizes the refreshed 7,570 ledger baseline. All 53 rows above
the earlier 7,517 fence are `remove_derived_parent` records written during the
preceding authorized ancestor-rescore reconcile; the live plan records that
attribution explicitly. The earlier baseline conflict is therefore resolved and
is not the present blocker.

## Complete live partition

The preceding production preflight classified every signed row from its current
source, binding, projection, target-producer, and direct-child topology:

| Partition | Rows | Detail |
|---|---:|---|
| Already detached | 3 | Exact deterministic stale-detach receipt exists; scalar null; no live binding or matching projection |
| Live topology admissions | 52 | Removing the signed source does not strip a non-structural target's final live producer |
| Last-producer refusals | 3 | Each names the target whose final live producing source would be stripped |
| **Total** | **58** | **3 + 52 + 3 = 58** |

The three fail-closed refusals are:

| Signed source | Target whose final producer would be stripped | Reason |
|---|---|---|
| `dd:ece/channel/t_e_voltage` | `voltage_of_diagnostic_antenna` | Zero other live producers and zero live direct children |
| `dd:equilibrium/time_slice/profiles_1d/b_average` | `flux_surface_average_magnetic_field_magnitude` | Zero other live producers and zero live direct children |
| `derived:neutral_state_energy_convection_velocity` | `neutral_state_energy_convection_velocity` | Removing the second signed binding would leave this non-structural target with zero live producers and zero live direct children |

## Tested operator boundary

`detach_signed_stale_source_bindings` validates a selected signed row only when
all of these are true: it is a DD source, it has exactly one signed target, and
its signed scalar equals that target. The committed authority is intentionally
broader: it includes stale derived producers, dual bindings, and scalar/binding
mismatches that the detach is meant to close.

The 52 live admissions therefore split as follows:

| Operator result | Rows | Evidence |
|---|---:|---|
| Public operator accepted and produced a live write-free manifest | 36 DD | `would_apply`, 36 receipt rows, manifest SHA-256 `3def413ce18b6b19878937977bdeeaa6519cfc53eb487ef0f52716184787422a` |
| Rejected by operator shape validation | 5 DD | Signed scalar differs from the sole target, or the signed source has multiple live targets |
| Rejected by operator source-type validation | 11 derived | Public operator requires `source_type=dd` and one DD backing/projection |
| **Live admissions** | **52** | **36 + 5 + 11 = 52** |

The five admitted DD rows rejected by the current operator are:

| Source | Signed shape outside the operator contract |
|---|---|
| `dd:bolometer/channel/aperture/surface` | Scalar `surface_area_of_diagnostic_aperture` differs from live target `area_of_diagnostic_aperture` |
| `dd:bolometer/channel/detector/surface` | Scalar `surface_area_of_diagnostic_aperture` differs from live target `area_of_diagnostic_aperture` |
| `dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z` | Two signed live targets, including the stale scalar selection |
| `dd:magnetics/method/diamagnetic_flux` | Scalar `diamagnetic_magnetic_flux` differs from live target `toroidal_magnetic_flux_due_to_diamagnetic_drift` |
| `dd:neutron_diagnostic/detectors/aperture/surface` | Scalar `surface_area_of_diagnostic_aperture` differs from live target `area_of_diagnostic_aperture` |

The eleven admitted derived rows are `derived:electron_density`,
`derived:electron_diffusivity`, `derived:electron_energy_flux`,
`derived:ion_diffusivity`, `derived:ion_energy_flux`, `derived:ion_pressure`,
`derived:ion_state_pressure`, `derived:neutral_energy_flux`,
`derived:neutral_particle_flux`, `derived:neutral_pressure`, and
`derived:neutral_state_pressure`.

The executable 36-row preview proved its own write-free boundary:

- out-of-allowlist source count: **9,563**;
- out-of-allowlist SHA-256: `cb96bee5440a0892cca18826027a96bba8abed4917d74cbca3a6fce5a2842cbb`;
- `StandardNameChange`: **7570 -> 7570**;
- `LLMCost`: **27489 -> 27489**.

It was not applied. A 36-row partial mutation would fail the requested
`already-detached + admitted-and-applied + refused = 58` receipt because 16
topology-admitted rows would remain neither applied nor valid last-producer
refusals.

## Missing completion evidence

Because no apply occurred, there is intentionally no replay receipt, no positive
ledger delta, and no post-apply dual-bound/unsourced census. The node therefore
does **not** claim the requested done-when. The last durable census recorded by
the live plan before this node is 23 dual-bound sources and 36 unsourced names,
all 36 genuine non-structural orphans with zero unsourced structural names after
reconcile; those are pre-apply context, not post-apply evidence.

## Durable logs

- Preceding complete classifier:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T150833249115-sgwi-stale-detach-remainder/live-preflight.json`
- First operator probe, preserving the grouped DD refusal:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T152326931473-sgwi-stale-detach-remainder-apply/operator-probe.log`
- Corrected per-row operator probe and 36-row live preview:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T152326931473-sgwi-stale-detach-remainder-apply/operator-probe-second.log`
