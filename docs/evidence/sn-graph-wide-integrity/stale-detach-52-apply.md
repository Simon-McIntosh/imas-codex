# Signed stale-source maximal-subset production apply

Date: 2026-08-20

## Outcome

The maximal refusal-free subset of the signed stale-source authority was
derived from the complete parent authority while its production participants
were locked, then applied in the same transaction. The signed 58 rows partition
exactly as:

| Disposition | Rows | Result |
|---|---:|---|
| Already detached | 3 | Exact prior receipts and detached postconditions re-verified |
| Applied | 52 | 54 `PRODUCED_NAME` bindings and 41 matching projections removed |
| Excluded | 3 | Each retained with its live refusal reason below |
| **Total** | **58** | **Complete signed-authority accounting** |

The transaction created exactly 52 `StandardNameChange` rows. It did not call a
provider or change `LLMCost`. Immediate exact-manifest replay returned
`already_applied` with `changed=0`.

## Authority and transaction boundary

- Authority:
  `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json`
- File SHA-256:
  `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`
- Canonical rows SHA-256:
  `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Signed rows: **58**
- Executable manifest SHA-256:
  `81b4fa8e38c1a626c26b5fbc154e8a0877a146eb24713461ed9d6854658e39fc`

The driver accepted no declared apply count or caller-authored subset. It loaded
all 58 signed rows, locked every existing signed source, DD backing, signed
target and incoming producer participant, and re-read their closures. It then:

1. recognized only rows carrying the exact deterministic prior event whose
   detached scalar, binding and projection postcondition still held;
2. ran the public signed-closure and last-producer validation over every other
   row against the locked live graph;
3. retained every refusal with its exact reason;
4. jointly revalidated the resulting subset, generated its manifest and applied
   it before committing the same transaction.

This ordering makes the subset a property of the signed authority plus locked
live state. A committed authorized change observed before participant locking
can narrow or expand the safe subset without escaping the 58-row authority;
participant changes after locking cannot interleave before commit. A containment
assertion proved all 52 applied source ids were members of the signed authority.
The machine receipt contains the complete applied-source list.

## Complete non-action accounting

The three previously detached rows retained their exact receipt and
postcondition:

- `dd:neutron_diagnostic/detectors/aperture/centre/phi`
- `dd:neutron_diagnostic/detectors/detector/centre/phi`
- `dd:refractometer/channel/frequencies`

The three excluded rows are:

| Source | Signed target | Live exclusion reason |
|---|---|---|
| `dd:ece/channel/t_e_voltage` | `voltage_of_diagnostic_antenna` | Detach would orphan the target; this stale source remains its final producer |
| `dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance` | `gap_at_plasma_boundary` | Signed source closure changed; the row no longer matches its signed binding/projection authority |
| `dd:equilibrium/time_slice/profiles_1d/b_average` | `flux_surface_average_magnetic_field_magnitude` | Detach would orphan the target; this stale source remains its final producer |

`derived:neutral_state_energy_convection_velocity` is not an exclusion. Its
locked live state now has reviewed structural children, so the last-producer
guard admitted it without weakening the invariant. This is the intended effect
of deriving the subset from live state rather than preserving a historic
three-row refusal count.

Representative applied source bindings include the multi-target
`dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z`, the signed
scalar-mismatch row `dd:bolometer/channel/aperture/surface`, ordinary DD rows
such as `dd:core_profiles/profiles_1d/j_tor`, and derived provenance rows such as
`derived:electron_density`, `derived:electron_diffusivity` and
`derived:neutral_state_pressure`. The full 52-row identity list is in the
machine receipt.

## Ledger, replay and collateral proof

| Measure | Before transaction | After apply | Immediate replay |
|---|---:|---:|---|
| `StandardNameChange` | 7,598 | 7,650 | unchanged at 7,650 |
| Declared receipt rows | 0 | 52 | 52 existing |
| `LLMCost` | 27,489 | 27,489 | unchanged |
| Operator outcome | — | `applied` | `already_applied` |
| `changed` | — | 52 | 0 |

The node read 7,598 itself from the locked transaction immediately before
preview/apply. The operator independently returned the same baseline and a
post-apply count of 7,650, so the ledger rose by exactly the declared 52 receipt
rows and no more. `LLMCost` stayed byte-count stable at 27,489.

Every source closure outside the derived 52-row allowlist was normalized and
hashed individually before and after mutation:

| Out-of-allowlist proof | Before | After |
|---|---:|---:|
| Source closures | 9,547 | 9,547 |
| Aggregate SHA-256 | `69d25bc4f8a4c4aa79d0bcb798bbab8f550e07c4134d095fbb307c79ffebb2ab` | identical |
| Changed individual row digests | 0 | 0 |

The public operator independently computed the same count and aggregate digest
inside its own apply checks.

## Independent postflight

A separate read-only query after commit and replay reports **23 live dual-bound
sources**, carrying 50 live bindings; none of the 23 sources is stale.
Representative surviving pairs include:

- `dd:core_sources/source/profiles_1d/ion/momentum/radial` →
  `momentum_source` + `radial_ion_momentum_source`;
- `dd:edge_profiles/ggd/mass_density/values` → `mass_density` +
  `total_plasma_mass_density`;
- `dd:edge_profiles/ggd/neutral/velocity/phi` →
  `toroidal_neutral_velocity` +
  `toroidal_neutral_momentum_convection_velocity`.

The independent relationship-unsourced census reports **38 live names**,
partitioned exactly as:

| Unsourced partition | Names | Detail |
|---|---:|---|
| Holding at least one live `HAS_PARENT` child | 2 | `electron_diffusivity`, `ion_diffusivity` |
| Holding no live child | 36 | 27 accepted, 4 reviewed, 4 drafted, 1 pending |
| **Total** | **38** | **2 + 36** |

The two structural rows are the immediate post-detach backlog anticipated by
the structural-provenance measurement rule: both are accepted derived parents
with live poloidal, parallel and effective children. The reconcile may mint derived
provenance for them later; this receipt reports the immediate graph without
smoothing that transient state into the 36 genuine-orphan count.

## Durable evidence

- Machine receipt, including all applied ids and exclusions:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T165334445184-sgwi-detach-derived-subset-apply/production-receipt.json`
- Machine receipt SHA-256:
  `97248c105518706920113bfb56b108d2567602ebbf4bb9b785568f1d6d01631f`
- Production driver log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T165334445184-sgwi-detach-derived-subset-apply/production-apply.log`
- Focused validation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T165334445184-sgwi-detach-derived-subset-apply/test-stale-source-detach.log`
