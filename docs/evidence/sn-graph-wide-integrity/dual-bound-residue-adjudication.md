# Residual dual-bound source adjudication

## Result

A single read-only production-graph invocation at
**2026-08-21 19:50:56 UTC** derived **23** `StandardNameSource` nodes with
more than one live `PRODUCED_NAME` target. This artifact classifies
**23 of 23 sources**, leaves **0 unclassified**, selects exactly **one surviving
identity per source**, and dispositions all **27 losing bindings** from the
50-binding live cohort. The one remaining scalar/edge mismatch is recorded
separately below.

This is semantic disposition evidence, not mutation authority. Every losing
binding is dispositioned for removal from this source only, together with its
matching DD-side `HAS_STANDARD_NAME` projection. No losing `StandardName` node
is authorized for retirement here. A later apply still owes an exact, freshly
derived signed manifest, complete last-producer closure, compare-and-set drift
checks, and replay proof.

| Measure | Result |
|---|---:|
| Cohort derived in the live invocation | 23 sources / 50 live bindings |
| Classified | **23** |
| Unclassified | **0** |
| Surviving bindings | 23 |
| Losing bindings dispositioned | 27 |
| Scalar retained on selected survivor | 14 |
| Scalar must move to selected survivor | 9 |
| Separate sole-live scalar mismatch | 1 |

## Authority and evidence rule

The DD path and documentation decide what the source measures. The source's
pinned `dd_unit`, current `IMASNode.unit`, and current `HAS_UNIT` edge were read
together; they agree on every row where a DD unit edge exists. The stored
`produced_sn_id`, target origin, and review score are corroborating state, not
semantic authority. The survivor is the shortest live identity that retains
every distinction actually present in the DD path or prose: owner, population,
internal-state resolution, direction, normalization, perturbation, moment,
representation, process, and locus.

Three older family-level dispositions are superseded by stronger evidence now
visible in the same live rows:

- The four unqualified `mass_density` leaves retain `mass_density`. Its current
  description already defines charged-plasma mass summed over charged species,
  while neither the paths nor their prose add a distinct `total` qualifier.
- Both Langmuir-probe descriptions say the surface is exposed to plasma. That
  is the semantic distinction carried by `wetted_area_of_langmuir_probe`, not
  unconstrained total geometric area.
- `neutral/state/momentum/flux_limiter/z` is a dimensionless coefficient. Its
  DD scalar and source cache are unit `1`; the rejected coordinate target is
  unit `m`. The coefficient survives even though the DD `HAS_UNIT` edge is
  absent on this one row.

## Per-source dispositions

In the unit column, `cache / DD property / DD edge` records the three independent
read surfaces. `—` means the current DD node has no `HAS_UNIT` edge.

| Source and DD evidence | Unit | Multiple live targets | Surviving identity | Losing binding disposition and rationale |
|---|---|---|---|---|
| `dd:core_sources/source/profiles_1d/ion/momentum/radial` — “Radial component” under an ion momentum **source** | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `momentum_source`; `radial_ion_momentum_source` | **`radial_ion_momentum_source`** | Remove `momentum_source`; the generic parent drops both ion population and radial component. Scalar already selects the survivor. |
| `dd:edge_profiles/ggd/mass_density/values` — one mass-density scalar per grid element | `kg.m^-3 / kg.m^-3 / kg.m^-3` | `mass_density`; `total_plasma_mass_density` | **`mass_density`** | Remove `total_plasma_mass_density`; the DD does not state a distinct total qualifier, and the shorter identity's description already defines summed charged-plasma mass. Scalar already selects the survivor. |
| `dd:edge_profiles/ggd/neutral/velocity/phi` — “Toroidal component” of neutral velocity | `m.s^-1 / m.s^-1 / m.s^-1` | `toroidal_neutral_velocity`; `toroidal_neutral_momentum_convection_velocity` | **`toroidal_neutral_velocity`** | Remove `toroidal_neutral_momentum_convection_velocity`; the path is mean neutral velocity, not an effective momentum transport coefficient. Scalar already selects the survivor. |
| `dd:edge_sources/source/ggd/ion/momentum/r` — major-radius component under an ion momentum **source** | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `radial_ion_momentum`; `radial_ion_momentum_source` | **`radial_ion_momentum_source`** | Remove `radial_ion_momentum` and move the scalar to the survivor; that reviewed identity describes momentum flux, while the IDS/path and force-per-area unit identify a source term. |
| `dd:edge_transport/model/ggd/neutral/state/particles/d_pol/values` — poloidal diffusivity for a resolved neutral state | `m^2.s^-1 / m^2.s^-1 / m^2.s^-1` | `neutral_state_particle_diffusivity`; `poloidal_neutral_state_particle_diffusivity` | **`poloidal_neutral_state_particle_diffusivity`** | Remove `neutral_state_particle_diffusivity`; `d_pol` supplies the poloidal distinction. Scalar already selects the survivor. |
| `dd:equilibrium/time_slice/profiles_1d/mass_density` — “Mass density” | `kg.m^-3 / kg.m^-3 / kg.m^-3` | `mass_density`; `total_plasma_mass_density` | **`mass_density`** | Remove `total_plasma_mass_density`; neither leaf nor prose states a separate total operation. Scalar already selects the survivor. |
| `dd:equilibrium/time_slice/profiles_1d/squareness_upper_outer` — upper-outer Luce squareness | `1 / 1 / 1` | `outer_squareness_of_flux_surface`; `squareness_of_flux_surface`; `upper_outer_squareness_of_flux_surface` | **`upper_outer_squareness_of_flux_surface`** | Remove `outer_squareness_of_flux_surface` and `squareness_of_flux_surface`; the first merges upper and lower outer quadrants and the second erases both quadrant qualifiers. Scalar already selects the survivor. |
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/j_parallel` — normalized gyrocenter first-Bessel parallel-current moment | `1 / 1 / 1` | `normalized_perturbed_current_density`; `parallel_normalized_perturbed_current_density`; `parallel_normalized_perturbed_current_density_bessel_1`; `perturbed_current_density` | **`parallel_normalized_perturbed_current_density_bessel_1`** | Remove `parallel_normalized_perturbed_current_density`, `normalized_perturbed_current_density`, and `perturbed_current_density`, then move the scalar to the survivor; only the survivor preserves parallel projection, normalization, perturbation, and first-Bessel weighting together. |
| `dd:langmuir_probes/embedded/surface_area` — probe surface “exposed to the plasma” | `m^2 / m^2 / m^2` | `area_of_langmuir_probe`; `wetted_area_of_langmuir_probe` | **`wetted_area_of_langmuir_probe`** | Remove `area_of_langmuir_probe` and move the scalar; “exposed to plasma” is the wetted-area distinction and excludes shielded area. |
| `dd:langmuir_probes/reciprocating/surface_area` — collector surface exposed to plasma | `m^2 / m^2 / m^2` | `area_of_langmuir_probe`; `wetted_area_of_langmuir_probe` | **`wetted_area_of_langmuir_probe`** | Remove `area_of_langmuir_probe` and move the scalar; the prose again explicitly limits the quantity to exposed collector surface. |
| `dd:mhd/ggd/mass_density/values` — one mass-density scalar per grid element | `kg.m^-3 / kg.m^-3 / kg.m^-3` | `mass_density`; `total_plasma_mass_density` | **`mass_density`** | Remove `total_plasma_mass_density`; the path is unqualified and `mass_density` already defines the MHD inertial charged-plasma density. Scalar already selects the survivor. |
| `dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary` — imaginary part of perturbed electrostatic potential | `V / V / V` | `electrostatic_potential_imaginary_part`; `perturbed_electrostatic_potential_imaginary_part` | **`perturbed_electrostatic_potential_imaginary_part`** | Remove `electrostatic_potential_imaginary_part`; `phi_potential_perturbed` makes perturbation identity-bearing. Scalar already selects the survivor. |
| `dd:plasma_profiles/ggd/mass_density/values` — one mass-density scalar per grid element | `kg.m^-3 / kg.m^-3 / kg.m^-3` | `mass_density`; `total_plasma_mass_density` | **`mass_density`** | Remove `total_plasma_mass_density`; no separate total qualifier occurs in path or prose. Scalar already selects the survivor. |
| `dd:plasma_profiles/ggd/neutral/velocity/phi` — toroidal component of neutral velocity | `m.s^-1 / m.s^-1 / m.s^-1` | `toroidal_neutral_velocity`; `toroidal_neutral_momentum_convection_velocity` | **`toroidal_neutral_velocity`** | Remove `toroidal_neutral_momentum_convection_velocity`; the path measures particle flow velocity. Scalar already selects the survivor. |
| `dd:plasma_sources/source/ggd/ion/momentum/radial` — radial component under an ion momentum **source** | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `radial_ion_momentum`; `radial_ion_momentum_source` | **`radial_ion_momentum_source`** | Remove `radial_ion_momentum` and move the scalar; the source IDS and unit identify force-per-area input rather than transported momentum flux. |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` — radial component under an ion momentum **source** | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `radial_ion_momentum`; `radial_ion_momentum_source` | **`radial_ion_momentum_source`** | Remove `radial_ion_momentum` and move the scalar for the same source-versus-flux distinction. |
| `dd:plasma_transport/model/ggd/momentum/flux/radial` — radial component of momentum flux | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `radial_momentum`; `radial_momentum_flux` | **`radial_momentum_flux`** | Remove `radial_momentum`; the path explicitly names `flux`, and the survivor retains that transport quantity. Scalar already selects the survivor. |
| `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` — parallel neutral-species energy convection velocity | `m.s^-1 / m.s^-1 / m.s^-1` | `neutral_species_energy_convection_velocity`; `parallel_neutral_species_energy_convection_velocity` | **`parallel_neutral_species_energy_convection_velocity`** | Remove `neutral_species_energy_convection_velocity`; `v_parallel` is identity-bearing. Scalar already selects the survivor. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/poloidal` — poloidal component of resolved-neutral-state momentum flux | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `poloidal_linear_neutral_internal_state_momentum_flux`; `poloidal_neutral_state_momentum_flux` | **`poloidal_neutral_state_momentum_flux`** | Remove `poloidal_linear_neutral_internal_state_momentum_flux`; `state` and `momentum` already carry internal-state and linear-momentum semantics, so the extra tokens add no DD distinction. The survivor passed the corrected-grammar quorum at 0.8875, and the scalar already selects it. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` — radial component of resolved-neutral-state momentum flux | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `radial_neutral_internal_state_momentum_flux`; `radial_neutral_state_momentum_flux` | **`radial_neutral_state_momentum_flux`** | Remove `radial_neutral_internal_state_momentum_flux` and move the scalar; this is the radial member of the same canonical family, and the DD adds no semantic distinction beyond neutral state, momentum flux, and radial component. |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux/poloidal` — poloidal component of resolved-neutral-state momentum flux | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `poloidal_linear_neutral_internal_state_momentum_flux`; `poloidal_neutral_state_momentum_flux` | **`poloidal_neutral_state_momentum_flux`** | Remove `poloidal_linear_neutral_internal_state_momentum_flux` for the same canonical-family reason. Scalar already selects the survivor. |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux_limiter/z` — dimensionless vertical neutral-state momentum flux-limiter coefficient | `1 / 1 / —` | `vertical_coordinate_of_active_limiter_point`; `vertical_neutral_state_momentum_flux_limiter_coefficient` | **`vertical_neutral_state_momentum_flux_limiter_coefficient`** | Remove `vertical_coordinate_of_active_limiter_point` and move the scalar; path ownership, unit `1`, and the survivor's unit all identify a coefficient, while the losing target is unit `m`. |
| `dd:runaway_electrons/global_quantities/volume_average/current_density` — “Runaways parallel current density = average(j.B) / B0” | `A.m^-2 / A.m^-2 / A.m^-2` | `parallel_runaway_electron_current_density`; `parallel_volume_averaged_runaway_electron_current_density`; `volume_averaged_runaway_electron_current_density` | **`parallel_volume_averaged_runaway_electron_current_density`** | Remove `parallel_runaway_electron_current_density` and `volume_averaged_runaway_electron_current_density`, then move the scalar; the documentation requires both field-parallel projection and volume averaging. |

## Remaining scalar mismatch

The same invocation derived exactly one sole-live-target scalar mismatch:

| Source and DD evidence | Unit | Stored scalar | Sole live target and disposition |
|---|---|---|---|
| `dd:plasma_sources/source/ggd/neutral/state/momentum/phi` — toroidal component of a resolved-neutral-state momentum source | `kg.m^-1.s^-2 / kg.m^-1.s^-2 / kg.m^-1.s^-2` | `neutral_internal_state_torque_density` | **Keep the sole live binding `toroidal_neutral_internal_state_torque_density` and reconcile only the scalar mirror to it.** The `phi` leaf requires the toroidal component; no `PRODUCED_NAME` edge removal is authorized for this row. |

This mismatch is classified **1 of 1**, with **0 unclassified**. It requires the
existing exact signed scalar-mirror reconcile after asserting the sole live edge
and matching DD projection; it must not be repaired by a raw property write.

## Read-only proof

The cohort, row evidence, scalar mismatch, and both counter samples were read
through `GraphClient` in one process. The invocation contained only `MATCH`,
`OPTIONAL MATCH`, `WITH`, `UNWIND`, and `RETURN`; it made no LLM or provider
call.

| Production counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | 7,756 | 7,756 | **0** |
| `PRODUCED_NAME` relationships | 5,791 | 5,791 | **0** |

The identical counters prove **zero production graph mutation by this
adjudication invocation**.

- Source commit: `c81913747f78ead2c3bf57ab0ea0208543105d94`
- Live-plan version read before execution: `235`
- Raw live receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194813624595-dualresidue/live-cohort.json`
- Raw receipt SHA-256:
  `6ce6f91ad89d8ffb7452fc279fadcf5b75cb52626c3fa23ea0a3a2ea9bc9c9d2`
- Query diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194813624595-dualresidue/live-cohort.log`

## Apply boundary

This artifact closes the semantic judgment queue only. The apply owner must
derive the live cohort again inside its own invocation, fail closed unless its
source and complete target sets agree with these dispositions, sign the exact
27 losing bindings and 9 scalar moves, protect every losing target's global
last-producer closure, and prove replay plus out-of-allowlist immutability.
