# Stale-source lifecycle disposition

## Disposition

The live, read-only census on 2026-08-20 found **58 stale
`StandardNameSource` nodes with 59 live `PRODUCED_NAME` bindings**. The
configured Data Dictionary is uniquely current at **4.1.1**. All **46 DD
sources** are pinned to 4.1.0 and their exact `source_id` resolves only to an
`IMASNode` whose `lifecycle_status` is `removed`; none is present under the
configured-version presence predicate. The other **12 sources are derived** and
have no DD backing from which a versioned migration could be authorized.

The signed disposition is therefore **58 detach / 0 versioned migration**.
This is disposition authority, not mutation evidence. A later operator must
sign and compare-and-set the complete source, binding, projection, scalar and
affected-target closure, write the applicable lifecycle ledger, and then prove
replay and collateral immutability. No row in this record authorizes a rename,
replacement binding, or name-lifecycle promotion.

The machine record is
[`stale-source-lifecycle.json`](stale-source-lifecycle.json). Its 58 ordered
rows hash to
`316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
under `jq -cS '.rows' | sha256sum`.

## Decision rule

The disposition uses the same upstream-presence boundary as production source
reconciliation:

- A DD source is present only if an `IMASNode` with `id = source_id` exists and
  `coalesce(lifecycle_status, '') != 'removed'`.
- A signal source is present only if its exact `FacilitySignal` exists.
- Other source types have no DD snapshot-migration authority.
- Present exact DD identity permits a governed versioned snapshot migration;
  absent or removed identity requires detach. A migration never retargets a
  Standard Name.

No source in this census passes the first branch. The zero-migration result is
therefore measured, not a policy preference.

## Named blocking rows

| Source | 4.1.1 backing state | Live target | Choice | Repair unblocked |
|---|---|---|---|---|
| `dd:refractometer/channel/frequencies` | exact path absent; backing node `removed` | `frequency_of_diagnostic_antenna` | detach | Removes the stale compare-and-set blocker from the ordinary reviewed rename to `frequency_of_wave_diagnostic_channel`; the removed refractometer path itself is not migrated or rebound. |
| `dd:neutron_diagnostic/detectors/aperture/centre/phi` | exact path absent; plural-detectors backing node `removed` | `toroidal_angle_of_measurement_position` | detach | Closes stale owner/geometry row 21 as an upstream-absent row. It supplies no authority for an aperture-qualified replacement. |
| `dd:neutron_diagnostic/detectors/detector/centre/phi` | exact path absent; plural-detectors backing node `removed` | `toroidal_angle_of_measurement_position` | detach | Closes stale owner/geometry row 22 as an upstream-absent row. It supplies no authority for a detector-qualified replacement. |

The two neutron rows reproduce the stale refusals at ordinals 21 and 22 in
`owner-geometry-authority-mapping.json`. Their disposition is deliberately
detach, not semantic rewiring: the configured dictionary contains no exact
source entity capable of supporting a new owner identity.

## Other live-bound stale rows

The JSON enumerates every other row with its target IDs, scalar, backing state,
choice and repair consequence. The cohort is complete at **55 additional
sources / 56 additional live bindings**:

- **43 other DD sources / 44 bindings**: every backing node is `removed`, so
  every row is detached. This closes obsolete line-of-sight endpoints,
  separatrix-to-iron-core claims, old `j_tor` and `w_mhd` paths, obsolete
  gyrokinetic normalized-flux paths, legacy magnetics and x-ray paths, and the
  numerical-grid area claim without inventing a current replacement.
- **12 derived sources / 12 bindings**: every row lacks a DD version and
  `FROM_DD_PATH` backing. Each synthetic producer is detached; the target name
  may remain live only through independently signed structural or current-source
  authority. Detach is not authority to retire the target.
- **One dual-bound stale source**,
  `dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z`, carries
  both `vertical_coordinate_of_geometric_axis` and `vertical_outline`; detach
  removes both non-authoritative bindings and clears that dual-bound residue.
- **Five rows have a scalar outside their live target set**: the two bolometer
  surface rows, neutron aperture surface, `magnetics/method/diamagnetic_flux`,
  and `derived:neutral_state_energy_convection_velocity`. Detachment resolves
  the stale scalar/binding residue without treating either spelling as
  authority.

One additional carried row is materially settled by the same evidence:
`dd:ece/channel/t_e_voltage` is absent in 4.1.1, so its disposition is detach.
It must **not** be migrated or renamed onto `voltage_of_ece_channel`; after the
detach, the old identity can be adjudicated under the ordinary orphan lifecycle
if it has no other authority.

## Write-free proof

`StandardNameChange` measured **7,492 before** the authority read and **7,492
after** it: delta **0**. The census performed no graph writes and made no
provider calls. The committed JSON records the exact before/after counters,
58-row and 59-binding identities, and the per-row detach-or-migrate decision.

## Apply boundary

Before any later mutation, the executor must re-read the live cohort and refuse
if any source, target, scalar, binding, DD backing, projection, configured DD
version, or target closure differs from this signed record. The apply must also
partition target consequences explicitly: a detached stale binding is not
counted as a live producer today, but deleting it can still change structural
and replay closures. A target that becomes relationship-unsourced is handled by
the locked individual orphan policy; it is never silently retired or
hand-accepted.
