# Source-path mirror repair evidence

## Outcome

**COMPLETE.** A fresh graph-wide census derived the complete repair cohort in
the same invocation that previewed, applied, and replayed the signed
transaction. The number of live Standard Name identities whose
`source_paths` scalar was not the canonical mirror of their live incoming
ordinary `PRODUCED_NAME` edge closure moved from **49 to 0**.

The 49 identities comprised **54 scalar-only ordinary source entries**, **0
edge-only entries**, and **5 identities whose members were already equal but
whose scalar list order was non-canonical**. Structural `derived:` provenance
is intentionally excluded from the ordinary-source comparison and preserved,
matching the repository's source-path invariant. A source lifecycle of `stale`
does not make an extant `PRODUCED_NAME` relationship disappear: the three
recorded stale-source refusal bindings therefore remained correctly mirrored.

This is a newly measured defect class, not a reinterpretation of the closing
census. The closing census measured the source-side `produced_sn_id` scalar;
it did **not** compare the name-side `source_paths` scalar against incoming
`PRODUCED_NAME` relationships. The source-side count could therefore be zero
while 49 name identities still carried this separate desynchronization.

| Required measure | Observed result |
|---|---:|
| Graph-wide mismatched live identities | **49 → 0** |
| Scalar-only ordinary entries | **54 → 0** |
| Edge-only ordinary entries | **0 → 0** |
| Order-only non-canonical identities | **5 → 0** |
| Preview authority rows | **49** |
| Preview admitted / refused | **49 / 0** |
| Mutated rows | **49** |
| Receipt rows | **49** |
| `StandardNameChange` | **7,787 → 7,836; delta +49** |
| `PRODUCED_NAME` relationships | **5,780 → 5,780; delta 0** |
| `LLMCost` rows | **27,631 → 27,631; delta 0** |
| Replay | **`already_applied`; changed=0; persistent_writes=0** |
| Fresh-connection residual rows | **0** |

There were no refused rows, so no residual identity or refusal reason remains
to report.

## Exact closure definition

The applying invocation selected every non-terminal, non-deprecated
`StandardName`. For each identity it recomputed the ordinary closure as the
sorted, deduplicated set of `StandardNameSource.id` values on incoming
`PRODUCED_NAME` relationships where `source_type != 'derived'`. The target
scalar was that set plus any existing `derived:` structural-provenance entries,
again sorted and deduplicated.

The invocation compared the raw stored list with that canonical target, so it
caught both membership errors and non-canonical ordering. It then emitted a
typed 49-row `set_properties` repair authority, previewed the whole cohort,
and applied it through the generic signed transaction envelope with the
`out-of-allowlist-immutability` guard. The graph closure, participant property
fingerprints, and collateral hash were re-derived after locking and before the
transaction committed.

Authority identifiers:

- Authority file SHA-256:
  `a9c19de7f7382a79d2caae535bbca7ba27af31eda214dbed13b2fa4670157de2`
- Authority payload SHA-256:
  `aa6cab965574b9814332ec6e2322f21229a76b38e644ee79331c1a5f3cc8e834`
- Closure-sensitive manifest SHA-256:
  `ce4fb622c8a7c2c96f871015780f227e7b7efff5f6a59435dc91f5514fa72f5e`
- Receipt lookup keys:
  `run_id=r-20260822T060302659370-mirrorrepair` and the manifest SHA-256 above
- Receipt operation: `reconcile_source_path_mirror`

The exact run-id-plus-manifest query recovered all 49 immutable
`StandardNameChange` rows. Their count equals both the mutated-row count and
the global `StandardNameChange` delta. A run-independent edge census proved
that no `PRODUCED_NAME` relationship was added or removed.

## Mass-density correction

Before the transaction, `total_plasma_mass_density.source_paths` listed four
DD sources even though only one incoming `PRODUCED_NAME` edge remained. The
three scalar-only entries removed were:

- `dd:edge_profiles/ggd/mass_density/values`
- `dd:equilibrium/time_slice/profiles_1d/mass_density`
- `dd:mhd/ggd/mass_density/values`

After apply and again from the fresh postflight connection,
`total_plasma_mass_density.source_paths` is exactly:

```text
dd:plasma_profiles/ggd/mass_density/values
```

That one entry exactly equals its one incoming ordinary `PRODUCED_NAME`
source. The three removed scalar entries were not deleted from the graph and
no edge moved: all three remain present in both the incoming edge closure and
the `source_paths` mirror of `mass_density`. This repair therefore resolves the
read-view discrepancy without prejudging the reopened identity adjudication.

## Complete affected identity census

The table lists the complete 49-identity before cohort. “Scalar-only” is the
number of ordinary entries removed. The five zero rows were membership-equal
but stored in a non-canonical order.

| Identity | Scalar-only entries | Edge-only entries |
|---|---:|---:|
| `area_of_langmuir_probe` | 2 | 0 |
| `beryllium_density_at_plasma_boundary` | 1 | 0 |
| `boron_density_at_plasma_boundary` | 1 | 0 |
| `carbon_density_at_plasma_boundary` | 1 | 0 |
| `deuterium_density_at_plasma_boundary` | 1 | 0 |
| `deuterium_tritium_density_at_plasma_boundary` | 1 | 0 |
| `diamagnetic_momentum_diffusivity` | 1 | 0 |
| `electron_temperature` | 3 | 0 |
| `electrostatic_potential_imaginary_part` | 1 | 0 |
| `helium_3_density_at_plasma_boundary` | 1 | 0 |
| `helium_4_density_at_plasma_boundary` | 1 | 0 |
| `hydrogen_density_at_plasma_boundary` | 1 | 0 |
| `ion_temperature` | 1 | 0 |
| `lithium_density_at_plasma_boundary` | 1 | 0 |
| `momentum_source` | 1 | 0 |
| `neon_density_at_plasma_boundary` | 1 | 0 |
| `neutral_species_energy_convection_velocity` | 1 | 0 |
| `neutral_state_particle_diffusivity` | 1 | 0 |
| `normalized_perturbed_current_density` | 1 | 0 |
| `outer_squareness_of_flux_surface` | 1 | 0 |
| `parallel_normalized_perturbed_current_density` | 1 | 0 |
| `parallel_runaway_electron_current_density` | 1 | 0 |
| `perturbed_current_density` | 1 | 0 |
| `poloidal_current_weighted_average_external_magnetic_flux` | 1 | 0 |
| `poloidal_linear_neutral_internal_state_momentum_flux` | 2 | 0 |
| `radial_ion_momentum` | 2 | 0 |
| `radial_momentum` | 1 | 0 |
| `radial_neutral_internal_state_momentum_source` | 1 | 0 |
| `squareness_of_flux_surface` | 1 | 0 |
| `toroidal_angle_of_measurement_position` | 0 | 0 |
| `toroidal_coordinate_of_active_spatial_resolution_zone` | 0 | 0 |
| `toroidal_coordinate_of_detector` | 0 | 0 |
| `toroidal_coordinate_of_polarizer` | 0 | 0 |
| `toroidal_electric_field` | 2 | 0 |
| `toroidal_flux_surface_averaged_current_density` | 1 | 0 |
| `toroidal_helium_3_velocity_at_plasma_boundary` | 1 | 0 |
| `toroidal_helium_4_velocity_at_plasma_boundary` | 1 | 0 |
| `toroidal_hydrogen_velocity_at_plasma_boundary` | 1 | 0 |
| `toroidal_krypton_velocity_at_plasma_boundary` | 1 | 0 |
| `toroidal_neutral_momentum_convection_velocity` | 2 | 0 |
| `toroidal_nitrogen_velocity_at_plasma_boundary` | 1 | 0 |
| `toroidal_thermal_electron_torque_density_due_to_collisions` | 0 | 0 |
| `toroidal_torque_density_due_to_diamagnetic_drift` | 1 | 0 |
| `total_plasma_mass_density` | 3 | 0 |
| `tungsten_density_at_plasma_boundary` | 1 | 0 |
| `vertical_momentum_convection_velocity` | 1 | 0 |
| `vertical_outline` | 2 | 0 |
| `volume_averaged_runaway_electron_current_density` | 1 | 0 |
| `xenon_density_at_plasma_boundary` | 1 | 0 |

## Replay and independent verification

The exact replay used the same authority bytes, file and payload digests,
reason, run ID, and manifest digest. It returned `already_applied` with
`changed=0` and `persistent_writes=0`. The exact 49-row receipt closure and all
persistent counters were byte-equivalent before and after replay.

A separate process opened a fresh `GraphClient` connection after the applying
process exited. It independently re-derived the graph-wide closure and found
**0 residual mismatched identities**, recovered exactly **49** receipts by the
receipt's own run and manifest keys, re-read `PRODUCED_NAME=5,780`, and proved
the mass-density scalar and edge sets above.

Focused validation passed **5 tests**:

```text
tests/standard_names/test_source_paths_reconcile.py: 4 passed
tests/graph/test_sn_edge_integrity.py::TestStandardNameEdgeIntegrity::test_source_paths_scalar_consistent_with_edges: 1 passed
total: 5 passed, 0 failed, 0 skipped in 5.93 s
```

## Durable artifacts

- Applying authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-authority.json`
- Machine-readable preview, apply, receipts, replay, and census result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-result.json`
  (SHA-256
  `057e6ad6e1e9289d27a27ea3bc1dda6ff9fe7098ca43f9b010e4b1a4b2fcd265`)
- Complete apply diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-apply.log`
  (SHA-256
  `06ac8da39ee0d9a3ac0a38a20bb77e3b801fb2ec3f9630716941fb731d613f49`)
- Fresh-connection postflight:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-postflight.json`
  (SHA-256
  `61ab7917f663107962efc494af036c12b2410f62472a2100e57ff71db0591eaf`)
- Postflight diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-postflight.log`
  (SHA-256
  `b848afb627b1f6bd3b3f185231bafcbe6092b050a793a0d4758f17105fdd0c9b`)
- Focused test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T060302659370-mirrorrepair/source-paths-repair-tests.log`
  (SHA-256
  `e12cfc9b04d01916fd86fa95da2f160d805a7cd7536aeefb74b412ad0bbaab07`)
