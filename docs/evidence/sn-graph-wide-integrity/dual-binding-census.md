# Dual-bound StandardNameSource census

## Result

The read-only live-graph census completed at 2026-08-18 22:44:58 UTC. It found **279 `StandardNameSource` nodes with more than one live `PRODUCED_NAME` target**. A live target is one whose `name_stage` is neither `superseded` nor `exhausted`.

The five exclusive disposition classes account for every source exactly once:

| Disposition | Sources | Operational meaning |
| --- | ---: | --- |
| Scalar-selected dedup | 51 | `produced_sn_id` already selects one live target, no catalog-edited target is involved, and no stronger stale or exact-lineage disposition applies. A reviewed repair may remove the non-selected live edge. |
| Fold into ancestor candidate pair | 10 | Exactly two live targets have exactly one directed `REFINED_FROM` connection. These are candidates for the established descendant-to-ancestor fold, subject to an exact reviewed mutation manifest. |
| Stale detach candidate | 1 | The source is stale and its DD or signal upstream is absent under the production reconciliation predicate. All live bindings are candidates for detachment. |
| Catalog-edit involved | 216 | At least one target has `origin=catalog_edit`, with no earlier stale or exact-lineage disposition. These rows remain protected from automatic scalar dedup and require catalog-aware adjudication. |
| Genuinely ambiguous | 1 | No scalar target, catalog-edited target, exact two-target lineage, or missing upstream resolves the binding. Individual semantic adjudication is required. |
| **Total** | **279** | **The class counts sum exactly to the fresh dual-bound total.** |

This is 80 fewer sources than the 359-source snapshot recorded on 2026-08-18. The completed atomic-count fold explicitly accounts for 39 of those removals; this census does not attribute the other 41 to a particular concurrent graph change without a matching mutation receipt.

The classification precedence is stale detach, exact two-target lineage fold, catalog-edit protection, scalar-selected dedup, then genuine ambiguity. The order deliberately prevents a scalar mirror from overriding upstream absence, lineage evidence, or catalog-edit authority. The JSON receipt preserves the orthogonal facts as well: 275 sources have a scalar selecting one live target, 226 involve a catalog edit, 11 contain at least one lineage-connected target pair, and one has absent upstream data. One source has three live targets and a lineage-connected sub-pair; it is not labeled a safe two-target fold.

## Recurring target-pair families

Each source with more than two live targets contributes each unique pair to this frequency table. Exact source lists for all pairs are in the receipt.

| Sources | Live target pair |
| ---: | --- |
| 55 | `atomic_mass` + `neutral_species_atomic_mass` |
| 9 | `mode_number` + `toroidal_mhd_mode_number` |
| 6 | `momentum_source_due_to_diamagnetic_drift` + `toroidal_torque_density_due_to_diamagnetic_drift` |
| 6 | `radial_ion_momentum_flux_over_edge_region` + `radial_momentum` |
| 5 | `energy_density` + `ion_kinetic_energy_density` |
| 5 | `toroidal_angle_of_coil_conductor_element` + `toroidal_angle_of_measurement_position` |
| 4 | `mass_density` + `total_plasma_mass_density` |
| 4 | `poloidal_magnetic_flux` + `poloidal_magnetic_flux_at_constraint_position` |
| 4 | `toroidal_angle_of_measurement_position` + `toroidal_coordinate_of_aperture` |
| 3 | `energy_flux` + `radial_total_thermal_electron_energy_flux` |
| 3 | `radial_ion_momentum` + `radial_ion_momentum_source` |
| 3 | `roughness_of_optical_element` + `surface_roughness_of_optical_element` |
| 3 | `spectral_bremsstrahlung_radiance` + `spectral_radiance` |
| 3 | `vertical_coordinate_of_geometric_axis` + `vertical_outline` |

## Representative bindings

These examples expose the semantic distinction, source-path binding, origin, and review evidence behind each class. The complete per-class source IDs and target metadata are machine-readable in the accompanying receipt.

- Scalar-selected dedup: `dd:camera_visible/channel/fibre_bundle/geometry/outline/x2` is bound to `horizontal_coordinate_of_diagnostic_aperture` (name review score 0.95) and `horizontal_coordinate_of_optical_element` (0.975). Its scalar selects the optical-element identity, the upstream DD path exists, neither target is catalog-edited, and there is no `REFINED_FROM` connection.
- Fold candidate: `dd:camera_ir/channel/camera/direction/x` is bound to `x_image_up_unit_vector_of_camera` (pipeline, 0.9875) and `x_direction_unit_vector_of_camera` (catalog edit, 0.98125). The image-up name is a one-hop `REFINED_FROM` descendant of the direction name, and the scalar selects the ancestor.
- Stale detach candidate: `dd:equilibrium/time_slice/boundary_secondary_separatrix/outline/z` is bound to `vertical_coordinate_of_geometric_axis` (catalog edit, accepted, 0.98125) and `vertical_outline` (reviewed, 0.625). The source is stale and the DD upstream path is absent; its scalar currently selects the geometric-axis target, but lifecycle absence takes precedence.
- Catalog-edit involved: `dd:amns_data/a` is bound to derived `atomic_mass` (structurally accepted, no numeric name-review score) and catalog-edited `neutral_species_atomic_mass` (0.875). The scalar selects `atomic_mass`, but catalog-edit protection prevents automatic edge removal.
- Genuinely ambiguous: `dd:waves/coherent_wave/global_quantities/ion/state/power_fast` is bound to pipeline `fast_ion_charge_state_absorbed_wave_power` (accepted, 0.8625) and derived `ion_charge_state_power` (structurally accepted, no numeric name-review score). The source has no scalar selection, both targets are live and valid, its upstream exists, neither target is catalog-edited, and no lineage connects them.

## Read-only proof and artifacts

The census used the project's `GraphClient` resolution path with the canonical ignored `.env` temporarily copied into the worktree. The file was removed immediately after the graph read; it is absent, ignored, and was never staged. The query consisted only of graph reads. It made no graph mutation and no LLM call.

| Counter | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `StandardNameChange` | 7,151 | 7,151 | 0 |
| `LLMCost` | 27,467 | 27,467 | 0 |

- Machine-readable receipt: `docs/evidence/sn-graph-wide-integrity/dual-binding-census.json`
- Receipt SHA-256: `3e83656f18bf2094ceff95f4bf9f66f8e4832679f2f93377848af19c2a998809`
- Raw read-only query artifact: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T224011873903-dual-binding-census/census-raw.json`
- Raw artifact SHA-256: `f453b8473f985ac24c4f0527b001c6402c30af2bf9c20a413587b9f6115b06c5`
- Query log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T224011873903-dual-binding-census/census-query.log`
- Classification log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T224011873903-dual-binding-census/classification.log`
- Receipt invariant validation log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T224011873903-dual-binding-census/receipt-validation.log`
- Source code commit at query time: `5e11de522a628345f1d737d42d3e25e2b069c8e0`

The receipt is disposition evidence, not mutation authority. Applying scalar dedup, lineage folds, or stale detachment still requires an exact reviewed mutation manifest and its prescribed rollback evidence. Catalog-edit and genuinely ambiguous rows require individual adjudication before any edge is changed.
