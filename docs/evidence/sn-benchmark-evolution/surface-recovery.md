# Legacy benchmark surface recovery

Assessment date: 2026-08-25

Source revision: `1725e7fca6a2e733d9d6acbfc7918265eaac1131`

Live DD authority: `4.1.1` (`DDVersion.is_current = true`)

Scope: read-only recovery of the 51 legacy positive slots and 20 legacy negatives in `tests/standard_names/eval_sets/benchmark.json`. No graph write, provider call, benchmark run, or corpus mutation was made.

## Outcome

The current graph recovers **32 of 51 positive slots** without inventing a cross-domain substitute: 15 resolve exactly from the original DD node or its nearest unit-matching quantity owner, one has an accepted successor in its `REFINED_FROM` lineage, and 16 formerly empty slots can be filled with distinct accepted DD-bound candidates in the exact legacy domain they name. The remaining **19 positive slots are unrecoverable under the stated constraints**: four populated paths have no accepted current resolution, while all 15 `waves`, `fast_particles`, and `gyrokinetics` stubs target legacy domain labels that currently contain zero accepted DD-bound names.

The stricter own-path measure is **13 of 20 populated positives**. Those 13 have an accepted, valid name projected from the exact original `IMASNode`; nine retain the legacy expected identity and four resolve to a different, more specific current identity. The two nearest-owner recoveries (`magnetics/ip/data` and `magnetics/flux_loop/flux/data`) are deliberately excluded from that 13 because their accepted name is projected one `HAS_PARENT` hop above the structural `data` node.

For the 31 allocation stubs, **16 of 31 now have at least one accepted DD-bound candidate in the exact physics domain named by the stub**: turbulence 5/5, plasma-wall interactions 5/5, transport 3/3, and edge-plasma physics 3/3. The capacity check is much larger than the requested surface—40, 27, 498, and 67 distinct accepted names respectively—so the rows below choose a deterministic, alphabetically first set of distinct identities, one source path per identity. These are recovery candidates, not gold labels; independent benchmark curation remains required before freezing them.

All **20 of 20 original negatives are recovered exactly as negative examples**. Every DD path still exists. Nineteen candidate identities do not exist as `StandardName` nodes; the twentieth, `electron_temperature`, is accepted globally but is not bound to the ion-temperature DD path, preserving the intended wrong-species negative. No negative candidate is accepted for its cited DD path.

| Disposition | Positives | Negatives | Total |
|---|---:|---:|---:|
| `recovered-exact` | 15 | 20 | **35** |
| `recovered-through-supersession-lineage` | 1 | 0 | **1** |
| `fillable-within-its-stated-domain` | 16 | 0 | **16** |
| `unrecoverable-with-reason` | 19 | 0 | **19** |
| **All original rows** | **51** | **20** | **71** |

### Disposition semantics

- `recovered-exact` means the row's essential measurement survives current authority. A positive has an accepted, valid, unit-matching name on the exact DD node or the nearest unit-matching quantity owner. A negative's DD node exists and its candidate is not accepted for that DD path. Owner-hop positives are explicit and are not counted in the 13-row own-path metric.
- `recovered-through-supersession-lineage` means the legacy positive identity has an accepted, valid successor reachable through `REFINED_FROM`. It does not silently claim an exact current path binding; the per-row note states the live path state.
- `fillable-within-its-stated-domain` applies only to a blank legacy slot and names one accepted, valid, DD-bound candidate whose `StandardName.physics_domain` and backing `IMASNode.physics_domain` both equal the stub's legacy domain.
- `unrecoverable-with-reason` is fail-closed: no accepted resolution satisfies the applicable exact-path, owner, lineage, or exact-domain test. Adjacent modern domains and mere reviewed/exhausted candidates are not promoted into recovery.

## Live Cypher evidence

Every row below cites one of these queries and its returned slot or domain. Before trusting a zero, the key-coverage query established that all 9,773 `StandardNameSource` nodes have `id`, all 5,351 observed `PRODUCED_NAME` targets have `StandardName.id` and `name_stage`, and 5,347/5,351 carry `validation_status`. The four missing validation values remain visible and no query treats null as valid.

### Q-K0 — identity and lifecycle key coverage

```cypher
MATCH (src:StandardNameSource)
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName)
RETURN count(src) AS source_candidates,
       count(src.id) AS sources_with_id,
       count(src.dd_path) AS sources_with_dd_path,
       count(sn) AS produced_edges,
       count(sn.id) AS produced_names_with_id,
       count(sn.name_stage) AS produced_names_with_stage,
       count(sn.validation_status) AS produced_names_with_validation,
       count(sn.physics_domain) AS produced_names_with_domain
```

Live result: `9773, 9773, 8830, 5351, 5351, 5351, 5347, 5346` in the selected column order.

### Q-P1 — populated positive resolution

`$items` is the ordered set of the 20 non-`TODO` positives with `slot`, `dd_path`, `expected_name`, `expected_unit`, and legacy `physics_domain` copied verbatim from the committed JSON.

```cypher
UNWIND $items AS item
MATCH (dd:IMASNode {id: item.dd_path})
CALL (dd) {
  OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(exact:StandardName)
  WHERE exact.name_stage = 'accepted'
    AND exact.validation_status = 'valid'
  RETURN [x IN collect(DISTINCT CASE WHEN exact IS NULL THEN null ELSE {
    id: exact.id, unit: exact.unit, domain: exact.physics_domain
  } END) WHERE x IS NOT NULL] AS exact_names
}
CALL (dd) {
  OPTIONAL MATCH p=(dd)-[:HAS_PARENT*1..8]->(owner:IMASNode)
  OPTIONAL MATCH (owner)-[:HAS_STANDARD_NAME]->(owned:StandardName)
  WHERE owner.node_category = 'quantity'
    AND owned.name_stage = 'accepted'
    AND owned.validation_status = 'valid'
    AND owned.unit = dd.unit
  WITH owned, owner, p ORDER BY length(p), owned.id
  RETURN [x IN collect(CASE WHEN owned IS NULL THEN null ELSE {
    id: owned.id, unit: owned.unit, domain: owned.physics_domain,
    owner_path: owner.id, ancestor_depth: length(p)
  } END) WHERE x IS NOT NULL][0..5] AS owner_names
}
CALL (item) {
  OPTIONAL MATCH (orig:StandardName {id: item.expected_name})
  OPTIONAL MATCH (desc:StandardName)-[:REFINED_FROM*1..20]->(orig)
  WHERE desc.name_stage = 'accepted'
    AND desc.validation_status = 'valid'
  RETURN [x IN collect(DISTINCT CASE WHEN desc IS NULL THEN null ELSE {
    id: desc.id, unit: desc.unit, domain: desc.physics_domain
  } END) WHERE x IS NOT NULL] AS accepted_descendants
}
OPTIONAL MATCH (src:StandardNameSource {id: 'dd:' + item.dd_path})
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(current:StandardName)
RETURN item.slot, item.dd_path, item.physics_domain AS legacy_domain,
       item.expected_name, item.expected_unit,
       dd.node_category, dd.physics_domain AS dd_domain,
       exact_names, owner_names, accepted_descendants,
       src.status AS source_status,
       current.id AS current_id, current.name_stage AS current_stage,
       current.validation_status AS current_validation
ORDER BY item.slot
```

### Q-S1 — exact-domain stub capacity and deterministic candidates

`$requests` is the seven legacy stub domains with their slot counts. The count query uses mandatory matches inside a subquery, so a zero remains a real zero rather than an `OPTIONAL MATCH` null artefact.

```cypher
UNWIND $requests AS req
CALL (req) {
  MATCH (src:StandardNameSource)-[:FROM_DD_PATH]->(dd:IMASNode)
  MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName)
  WHERE sn.name_stage = 'accepted'
    AND sn.validation_status = 'valid'
    AND sn.physics_domain = req.domain
    AND dd.physics_domain = req.domain
  WITH sn, min(dd.id) AS dd_path
  ORDER BY sn.id
  RETURN count(*) AS distinct_names,
         collect({name: sn.id, dd_path: dd_path, unit: sn.unit,
                  description: sn.description})[0..req.slots] AS candidates
}
RETURN req.domain, req.slots, distinct_names, candidates
ORDER BY req.domain
```

Capacity result: `edge_plasma_physics=67`, `fast_particles=0`, `gyrokinetics=0`, `plasma_wall_interactions=27`, `transport=498`, `turbulence=40`, `waves=0` distinct accepted DD-bound names with agreeing name and DD domains.

### Q-N1 — negative path and candidate disposition

`$items` is the ordered set of 20 negatives with `slot`, `dd_path`, `candidate_name`, and `anti_pattern_category` copied verbatim from the committed JSON.

```cypher
UNWIND $items AS item
OPTIONAL MATCH (dd:IMASNode {id: item.dd_path})
OPTIONAL MATCH (candidate:StandardName {id: item.candidate_name})
OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(accepted:StandardName)
WHERE accepted.name_stage = 'accepted'
  AND accepted.validation_status = 'valid'
WITH item, dd, candidate,
     [x IN collect(DISTINCT accepted.id) WHERE x IS NOT NULL] AS exact_names
OPTIONAL MATCH p=(dd)-[:HAS_PARENT*1..8]->(owner:IMASNode)
OPTIONAL MATCH (owner)-[:HAS_STANDARD_NAME]->(owned:StandardName)
WHERE owner.node_category = 'quantity'
  AND owned.name_stage = 'accepted'
  AND owned.validation_status = 'valid'
  AND owned.unit = dd.unit
RETURN item.slot, item.dd_path, item.candidate_name,
       item.anti_pattern_category,
       dd IS NOT NULL AS dd_exists, dd.node_category, dd.physics_domain,
       candidate.name_stage AS candidate_stage,
       candidate.validation_status AS candidate_validation,
       EXISTS { (dd)-[:HAS_STANDARD_NAME]->(candidate) } AS candidate_bound_exact,
       exact_names,
       [x IN collect(DISTINCT CASE WHEN owned IS NULL THEN null ELSE {
         id: owned.id, owner_path: owner.id, depth: length(p)
       } END) WHERE x IS NOT NULL] AS owner_names
ORDER BY item.slot
```

Live aggregate: all 20 DD nodes exist; 19 candidate nodes are absent; `electron_temperature` is accepted globally but `candidate_bound_exact=false` for `core_profiles/profiles_1d/ion/temperature`; zero negative candidates are accepted for their cited path.

## Positive dispositions

### Populated positives P01–P20

| Slot | Original DD path | Legacy expected → current accepted recovery | Disposition | Current domain | Live evidence and qualification |
|---|---|---|---|---|---|
| P01 | `equilibrium/time_slice/profiles_1d/psi` | `poloidal_magnetic_flux` → `poloidal_magnetic_flux_at_flux_surface` | `recovered-exact` | equilibrium | Q-P1 slot 1: exact-node accepted/valid projection, Wb. Current identity is preferred over the stale generic spelling. |
| P02 | `equilibrium/time_slice/profiles_1d/q` | `safety_factor` → same | `recovered-exact` | transport | Q-P1 slot 2: exact-node accepted/valid projection, unit 1. |
| P03 | `equilibrium/time_slice/boundary/elongation` | `elongation_of_plasma_boundary` → same | `recovered-exact` | equilibrium | Q-P1 slot 3: exact-node accepted/valid projection, unit 1. |
| P04 | `equilibrium/time_slice/boundary/minor_radius` | `minor_radius_of_plasma_boundary` → same | `recovered-exact` | equilibrium | Q-P1 slot 4: exact-node accepted/valid projection, m. |
| P05 | `equilibrium/time_slice/global_quantities/volume` | `plasma_volume` → `volume_of_plasma_boundary` | `recovered-through-supersession-lineage` | equilibrium | Q-P1 slot 5: accepted/valid m^3 successor reaches the superseded legacy identity through `REFINED_FROM`. The exact path currently points to reviewed `volume_of_flux_surface`, so this is not counted as an own-path recovery and needs curation before reuse. |
| P06 | `equilibrium/time_slice/global_quantities/q_axis` | `safety_factor_at_magnetic_axis` → same | `recovered-exact` | equilibrium | Q-P1 slot 6: exact-node accepted/valid projection, unit 1. |
| P07 | `core_profiles/profiles_1d/electrons/temperature` | `electron_temperature` → same | `recovered-exact` | magnetohydrodynamics | Q-P1 slot 7: exact-node accepted/valid projection, eV. Legacy domain `core_plasma_physics` has drifted. |
| P08 | `core_profiles/profiles_1d/electrons/density` | `electron_density` → `total_electron_density` | `recovered-exact` | transport | Q-P1 slot 8: exact-node accepted/valid projection, m^-3. Current total-population identity is preferred. |
| P09 | `core_profiles/profiles_1d/electrons/pressure` | `electron_pressure` → `total_electron_pressure` | `recovered-exact` | transport | Q-P1 slot 9: exact-node accepted/valid projection, Pa. Current total-population identity is preferred. |
| P10 | `core_profiles/profiles_1d/ion/temperature` | `ion_temperature` → none accepted | `unrecoverable-with-reason` | — | Q-P1 slot 10: exact source currently produces `ion_temperature` only at `reviewed`; no accepted exact, owner, or successor resolution. |
| P11 | `magnetics/ip/data` | `plasma_current` → same at `magnetics/ip` | `recovered-exact` | equilibrium | Q-P1 slot 11: the original `data` node is structural; its nearest unit-matching quantity owner is one `HAS_PARENT` hop away and projects accepted/valid `plasma_current`, A. Excluded from the 13 own-path count. |
| P12 | `magnetics/flux_loop/flux/data` | `magnetic_flux` → `poloidal_magnetic_flux_of_flux_loop` at `magnetics/flux_loop/flux` | `recovered-exact` | equilibrium | Q-P1 slot 12: structural `data` node resolves one owner hop to the accepted/valid, Wb, flux-loop-specific identity. Excluded from the 13 own-path count. |
| P13 | `tf/field_map/b_field_tor/values` | `toroidal_magnetic_field` → same | `recovered-exact` | equilibrium | Q-P1 slot 13: exact-node accepted/valid projection, T. The exact node wins over the different accepted owner projection `toroidal_vacuum_magnetic_field`. |
| P14 | `tf/field_map/b_field_r/values` | `radial_magnetic_field` → same | `recovered-exact` | equilibrium | Q-P1 slot 14: exact-node accepted/valid projection, T. |
| P15 | `tf/field_map/b_field_z/values` | `vertical_magnetic_field` → same | `recovered-exact` | equilibrium | Q-P1 slot 15: exact-node accepted/valid projection, T. |
| P16 | `summary/global_quantities/tau_energy/value` | `energy_confinement_time` → none accepted | `unrecoverable-with-reason` | — | Q-P1 slot 16: source is `extracted` with no name. The only refinement descendant, `thermal_energy_confinement_time`, is exhausted rather than accepted. |
| P17 | `core_profiles/global_quantities/beta_tor` | `toroidal_beta` → none accepted | `unrecoverable-with-reason` | — | Q-P1 slot 17: exact source produces reviewed `beta`; no accepted exact, owner, or successor resolution. |
| P18 | `equilibrium/time_slice/profiles_2d/phi` | `toroidal_magnetic_flux` → none accepted | `unrecoverable-with-reason` | — | Q-P1 slot 18: source is `extracted` with no produced name; globally accepted `toroidal_magnetic_flux` is not bound to this DD path and has no accepted path-specific successor. |
| P19 | `edge_profiles/profiles_1d/electrons/temperature` | `electron_temperature` → same | `recovered-exact` | magnetohydrodynamics | Q-P1 slot 19: exact-node accepted/valid projection, eV. |
| P20 | `edge_profiles/profiles_1d/electrons/density` | `electron_density` → `total_electron_density` | `recovered-exact` | transport | Q-P1 slot 20: exact-node accepted/valid projection, m^-3. |

### Stub positives P21–P51

Each filled row below is distinct by both selected `StandardName.id` and selected DD path. Descriptions are included because the slot rationales themselves contain no quantity-level semantics.

| Slot | Legacy domain | Candidate name, unit, and source-path binding | Disposition | Live evidence / reason |
|---|---|---|---|---|
| P21 | waves | — | `unrecoverable-with-reason` | Q-S1 `waves`: 0 accepted DD-bound candidates in the exact stated domain. |
| P22 | waves | — | `unrecoverable-with-reason` | Q-S1 `waves`: 0; no cross-domain wave substitute admitted. |
| P23 | waves | — | `unrecoverable-with-reason` | Q-S1 `waves`: 0; no cross-domain wave substitute admitted. |
| P24 | waves | — | `unrecoverable-with-reason` | Q-S1 `waves`: 0; no cross-domain wave substitute admitted. |
| P25 | waves | — | `unrecoverable-with-reason` | Q-S1 `waves`: 0; the legacy label is absent from current accepted DD-bound authority. |
| P26 | fast_particles | — | `unrecoverable-with-reason` | Q-S1 `fast_particles`: 0 accepted DD-bound candidates in the exact stated domain. |
| P27 | fast_particles | — | `unrecoverable-with-reason` | Q-S1 `fast_particles`: 0; adjacent transport or heating names are not substitutes. |
| P28 | fast_particles | — | `unrecoverable-with-reason` | Q-S1 `fast_particles`: 0; adjacent transport or heating names are not substitutes. |
| P29 | fast_particles | — | `unrecoverable-with-reason` | Q-S1 `fast_particles`: 0; adjacent transport or heating names are not substitutes. |
| P30 | fast_particles | — | `unrecoverable-with-reason` | Q-S1 `fast_particles`: 0; the legacy label is absent from current accepted DD-bound authority. |
| P31 | turbulence | `derivative_with_respect_to_poloidal_angle_of_normalized_effective_particle_energy` (1) ← `gyrokinetics_local/species/potential_energy_gradient_norm` | `fillable-within-its-stated-domain` | Q-S1 `turbulence`; 40 distinct candidates. Poloidal-angle derivative of normalized effective particle energy on a flux surface. |
| P32 | turbulence | `electron_density_at_outboard_midplane` (m^-3) ← `gyrokinetics_local/normalizing_quantities/n_e` | `fillable-within-its-stated-domain` | Q-S1 `turbulence`; electron density at the low-field-side midplane. |
| P33 | turbulence | `electron_temperature_at_outboard_midplane` (eV) ← `gyrokinetics_local/normalizing_quantities/t_e` | `fillable-within-its-stated-domain` | Q-S1 `turbulence`; bulk-electron temperature at the low-field-side midplane. |
| P34 | turbulence | `normalized_collisionality` (1) ← `gyrokinetics_local/collisions/collisionality_norm` | `fillable-within-its-stated-domain` | Q-S1 `turbulence`; collision-frequency ratio to a characteristic streaming, transit, or bounce frequency. |
| P35 | turbulence | `normalized_debye_length` (1) ← `gyrokinetics_local/species_all/debye_length_norm` | `fillable-within-its-stated-domain` | Q-S1 `turbulence`; Debye length normalized to the gyrokinetic reference length. |
| P36 | plasma_wall_interactions | `accumulated_neutral_count_at_wall` (1) ← `wall/global_quantities/neutral/wall_inventory` | `fillable-within-its-stated-domain` | Q-S1 `plasma_wall_interactions`; 27 distinct candidates. Cumulative neutral inventory deposited at the wall. |
| P37 | plasma_wall_interactions | `electron_kinetic_energy_flux_at_wall_due_to_surface_emission` (W.m^-2) ← `wall/description_ggd/ggd/energy_fluxes/kinetic/electrons/emitted/values` | `fillable-within-its-stated-domain` | Q-S1 `plasma_wall_interactions`; emitted-electron kinetic-energy flux at a plasma-facing surface. |
| P38 | plasma_wall_interactions | `electron_particle_flux_at_wall_due_to_surface_emission` (m^-2.s^-1) ← `wall/description_ggd/ggd/particle_fluxes/electrons/emitted/values` | `fillable-within-its-stated-domain` | Q-S1 `plasma_wall_interactions`; emitted electron number flux density at the wall. |
| P39 | plasma_wall_interactions | `electron_source_rate_due_to_surface_emission` (s^-1) ← `wall/global_quantities/electrons/particle_flux_from_wall` | `fillable-within-its-stated-domain` | Q-S1 `plasma_wall_interactions`; electron-equivalent source rate supplied by surface emission. |
| P40 | plasma_wall_interactions | `incident_energy_flux_at_wall_due_to_eddy_current` (W.m^-2) ← `wall/description_ggd/ggd/energy_fluxes/current/incident` | `fillable-within-its-stated-domain` | Q-S1 `plasma_wall_interactions`; electromagnetic surface energy flux entering the wall through the eddy-current channel. |
| P41 | gyrokinetics | — | `unrecoverable-with-reason` | Q-S1 `gyrokinetics`: 0 accepted DD-bound candidates in the exact stated domain. |
| P42 | gyrokinetics | — | `unrecoverable-with-reason` | Q-S1 `gyrokinetics`: 0; current `gyrokinetics_local` paths classify mainly as turbulence, not this legacy label. |
| P43 | gyrokinetics | — | `unrecoverable-with-reason` | Q-S1 `gyrokinetics`: 0; adjacent turbulence names are not substitutes. |
| P44 | gyrokinetics | — | `unrecoverable-with-reason` | Q-S1 `gyrokinetics`: 0; adjacent turbulence names are not substitutes. |
| P45 | gyrokinetics | — | `unrecoverable-with-reason` | Q-S1 `gyrokinetics`: 0; the legacy label is absent from current accepted DD-bound authority. |
| P46 | transport | `atomic_count` (1) ← `core_profiles/profiles_1d/ion/element/atoms_n` | `fillable-within-its-stated-domain` | Q-S1 `transport`; 498 distinct candidates. Stoichiometric atom multiplicity for a specified species element. |
| P47 | transport | `atomic_mass` (u) ← `core_profiles/profiles_1d/ion/element/a` | `fillable-within-its-stated-domain` | Q-S1 `transport`; species mass parameter used for inertial and transport calculations. |
| P48 | transport | `center_of_mass_velocity_due_to_diamagnetic_drift` (m.s^-1) ← `plasma_profiles/ggd/velocity_mass_centre/diamagnetic` | `fillable-within-its-stated-domain` | Q-S1 `transport`; mass-weighted center-of-mass velocity contribution from diamagnetic drifts. |
| P49 | edge_plasma_physics | `diamagnetic_current_density_due_to_heat_viscosity` (A.m^-2) ← `edge_profiles/ggd/j_heat_viscosity/diamagnetic` | `fillable-within-its-stated-domain` | Q-S1 `edge_plasma_physics`; 67 distinct candidates. Diamagnetic current-density contribution from heat viscosity. |
| P50 | edge_plasma_physics | `diamagnetic_current_density_due_to_ion_neutral_friction` (A.m^-2) ← `edge_profiles/ggd/j_ion_neutral_friction/diamagnetic` | `fillable-within-its-stated-domain` | Q-S1 `edge_plasma_physics`; diamagnetic current-density contribution from ion-neutral friction. |
| P51 | edge_plasma_physics | `diamagnetic_momentum_flux` (kg.m^-1.s^-2) ← `edge_sources/source/ggd/neutral/momentum/diamagnetic` | `fillable-within-its-stated-domain` | Q-S1 `edge_plasma_physics`; momentum transport per unit area restricted to the diamagnetic contribution. |

## Negative dispositions N01–N20

The rejection rationale remains the committed semantic evidence; Q-N1 supplies the live graph half—path existence, candidate lifecycle, and exact binding. `absent` means no `StandardName` node with the candidate identity. A globally accepted identity remains a valid negative when it is demonstrably unbound to the cited, semantically different DD path.

| Slot | Candidate at DD path | Anti-pattern and live finding | Disposition | Evidence |
|---|---|---|---|---|
| N01 | `poloidal_magnetic_magnetic_field_probe_voltage` at `equilibrium/time_slice/constraints/b_field_pol_probe/measured` | `base_duplication`; path exists, candidate absent | `recovered-exact` | Q-N1 slot 1 |
| N02 | `electron_diffusivity_poloidal` at `edge_transport/model/ggd/ion/particles/d_pol/values` | `component_prefix_order`; path exists, candidate absent | `recovered-exact` | Q-N1 slot 2 |
| N03 | `ion_rotation_frequency_toroidal` at `core_profiles/profiles_1d/ion/rotation_frequency_tor` | `component_prefix_order`; candidate absent; exact current name is `toroidal_ion_rotation_frequency` | `recovered-exact` | Q-N1 slot 3 |
| N04 | `ion_cumulative_ionization_potential_at_ion_state` at `core_profiles/profiles_2d/ion/state/ionisation_potential` | `wrong_transformation`; path exists, candidate absent | `recovered-exact` | Q-N1 slot 4 |
| N05 | `diamagnetic_ion_e_cross_b_drift_velocity` at `plasma_profiles/ggd/ion/state/velocity_exb/diamagnetic` | `wrong_transformation`; candidate absent; exact current name is `perpendicular_ion_charge_state_velocity_due_to_diamagnetic_drift` | `recovered-exact` | Q-N1 slot 5 |
| N06 | `electron_temperature_ev` at `core_profiles/profiles_1d/electrons/temperature` | `llm_supplied_unit`; candidate absent; exact current name is `electron_temperature` | `recovered-exact` | Q-N1 slot 6 |
| N07 | `t_e` at `core_profiles/profiles_1d/electrons/temperature` | `symbol_leakage`; candidate absent; exact current name is `electron_temperature` | `recovered-exact` | Q-N1 slot 7 |
| N08 | `ip` at `magnetics/ip/data` | `symbol_leakage`; candidate absent; one-hop quantity owner resolves `plasma_current` | `recovered-exact` | Q-N1 slot 8 |
| N09 | `te` at `core_profiles/profiles_1d/electrons/temperature` | `symbol_leakage`; candidate absent; exact current name is `electron_temperature` | `recovered-exact` | Q-N1 slot 9 |
| N10 | `ne` at `core_profiles/profiles_1d/electrons/density` | `symbol_leakage`; candidate absent; exact current name is `total_electron_density` | `recovered-exact` | Q-N1 slot 10 |
| N11 | `electron_density_per_cubic_metre` at `core_profiles/profiles_1d/electrons/density` | `llm_supplied_unit`; candidate absent; exact current name is `total_electron_density` | `recovered-exact` | Q-N1 slot 11 |
| N12 | `toroidal_magnetic_field_tesla` at `tf/field_map/b_field_tor/values` | `llm_supplied_unit`; candidate absent; exact current name is `toroidal_magnetic_field` | `recovered-exact` | Q-N1 slot 12 |
| N13 | `safety_factor_squared` at `equilibrium/time_slice/profiles_1d/q` | `wrong_transformation`; candidate absent; exact current name is `safety_factor` | `recovered-exact` | Q-N1 slot 13 |
| N14 | `magnetic_field_toroidal` at `tf/field_map/b_field_tor/values` | `component_prefix_order`; candidate absent; exact current name is `toroidal_magnetic_field` | `recovered-exact` | Q-N1 slot 14 |
| N15 | `plasma_current_current` at `magnetics/ip/data` | `base_duplication`; candidate absent; one-hop quantity owner resolves `plasma_current` | `recovered-exact` | Q-N1 slot 15 |
| N16 | `elongation` at `equilibrium/time_slice/boundary/elongation` | `missing_qualifier`; candidate absent; exact current name is `elongation_of_plasma_boundary` | `recovered-exact` | Q-N1 slot 16 |
| N17 | `magnetics_ip_data` at `magnetics/ip/data` | `ids_leakage`; candidate absent; one-hop quantity owner resolves `plasma_current` | `recovered-exact` | Q-N1 slot 17 |
| N18 | `electron_temperature` at `core_profiles/profiles_1d/ion/temperature` | `wrong_identity`; candidate is accepted globally but is not bound to this ion path (`candidate_bound_exact=false`) | `recovered-exact` | Q-N1 slot 18 |
| N19 | `reconstructed_poloidal_magnetic_flux` at `equilibrium/time_slice/profiles_1d/psi` | `processing_adjective`; candidate absent; exact current name is `poloidal_magnetic_flux_at_flux_surface` | `recovered-exact` | Q-N1 slot 19 |
| N20 | `thomson_scattering_electron_temperature` at `core_profiles/profiles_1d/electrons/temperature` | `measurement_method`; candidate absent; exact current name is `electron_temperature` | `recovered-exact` | Q-N1 slot 20 |

## Domain distribution against the original

The first table preserves the original allocation labels and asks how much of each requested surface is now recoverable without changing that label. This is the appropriate completeness view for the 51 slots.

| Original allocation | Original slots | Exact | Through lineage | Domain-fillable | Unrecoverable | Recoverable surface |
|---|---:|---:|---:|---:|---:|---:|
| equilibrium | 7 | 5 | 1 | 0 | 1 | **6/7** |
| core_plasma_physics | 4 | 3 | 0 | 0 | 1 | **3/4** |
| magnetic_field_diagnostics | 5 | 5 | 0 | 0 | 0 | **5/5** |
| transport | 5 | 0 | 0 | 3 | 2 | **3/5** |
| edge_plasma_physics | 5 | 2 | 0 | 3 | 0 | **5/5** |
| waves | 5 | 0 | 0 | 0 | 5 | **0/5** |
| fast_particles | 5 | 0 | 0 | 0 | 5 | **0/5** |
| turbulence | 5 | 0 | 0 | 5 | 0 | **5/5** |
| plasma_wall_interactions | 5 | 0 | 0 | 5 | 0 | **5/5** |
| gyrokinetics | 5 | 0 | 0 | 0 | 5 | **0/5** |
| **Total** | **51** | **15** | **1** | **16** | **19** | **32/51** |

The second table reports the graph's current `StandardName.physics_domain` for those 32 recoverable rows. It demonstrates why blindly preserving the old domain counts would misstate the present graph: several exact DD-path recoveries have moved from legacy core, edge, or magnetic allocations into equilibrium, transport, or magnetohydrodynamics.

| Current recovered-name domain | Rows |
|---|---:|
| equilibrium | 10 |
| transport | 7 |
| magnetohydrodynamics | 2 |
| edge_plasma_physics | 3 |
| turbulence | 5 |
| plasma_wall_interactions | 5 |
| **Total recovered positives** | **32** |

## Final boundary

This record recovers evidence, not benchmark authority. It does not amend the legacy JSON and does not replace the already committed 40-row controlled corpus. The 13 exact-node positives and two quantity-owner positives are the strongest reusable legacy seeds; the lineage row needs path-level curation, and all 16 domain-fillable stubs need independent labelling before they can enter any frozen gold population. The 19 unrecoverable slots must remain explicit gaps unless the graph later gains an accepted exact-path resolution or accepted DD-bound names under the exact legacy domain.
