# WEST accepted-document refresh: first bounded tranche

## Outcome

The first accepted-document refresh tranche completed over exactly **40 of the 175** documentation-accepted WEST identities. Selection used `StandardName.id`, took the first 40 clean accepted/accepted identities in lexical order after live revalidation, and retained **zero overlap** with the 20-identity concurrent name-refinement exclusion. The executed command was a docs-only, scope-fenced `sn run` with global maintenance skipped. It completed with `status=completed` and `stop_reason=no_eligible_work`.

| Measure | Result |
|---|---:|
| WEST docs-accepted source cohort | **175** |
| Refreshed in this tranche | **40** |
| Zero-call exact-name preflight | **40/40** |
| Concurrent-refine overlap | **0** |
| Refreshed documents accepted after the loop | **40/40** |
| Accepted on first review | **39** |
| Landed below the quorum bar on first review | **1** |
| Entered documentation refinement | **1** |
| Refined and then accepted | **1/1** |
| Documentation texts changed | **40/40** |
| Changed texts with exact prior DocsRevision snapshot | **40/40** |
| Final scores below prior scores | **6/40** |

The only below-bar document was `effective_turn_count_of_passive_loop`. Its first review routed it to the ordinary refine pool; the stored pre-refine snapshot carries score **0.84375**, below the 0.85 threshold. The refined document then cleared quorum at **0.91875**. No document was directly accepted or edited around the quorum.

## Admission and spend

Admission rendered and priced all 40 production-enriched generation requests and all 40 reviewer requests before any provider call. All generation requests carried at least one enrichment candidate. The prior-smoothed flow reserved 40 generation operations, 52 review cycles, and 12 conditional refinement cycles.

| Admission component | Projected work | Projected USD |
|---|---:|---:|
| Generate documentation | 40 | $0.474484 |
| Review documentation | 52 | $135.975920 |
| Refine documentation | 12 | $1.249669 |
| **Main scoped run** | — | **$137.700073** |

The zero-call projection was **$137.700073** against the authorized **$150.000000** ceiling, so the tranche was admitted. The executed `sn run` made **124 provider calls** and spent **$5.120219**: 40 generation calls for $0.206372, 83 review calls for $4.772107, and one refinement call for $0.141740.

One accepted legacy document, `flux_surface_averaged_major_radius`, had no historical aggregate documentation score even though its docs stage was accepted. To avoid reporting a missing comparator, the exact prior DocsRevision text was submitted read-only to the current three-seat quorum after the run. That supplementary comparison projected $2.619193 exposure, made **3 calls**, spent **$0.303280**, and scored the prior prose **0.81250** by authoritative escalation. It made no graph mutation. Including this explicitly separated comparison, total measured work was **127 calls** and **$5.423499**; the executed refresh run itself remains the 124-call, $5.120219 figure above.

## Snapshot integrity

The reset attempted a DocsRevision snapshot for every selected identity before generation. Exact post-run comparison initially found **22/40** prior texts. Eighteen snapshots had collided with already-existing deterministic `#rev-N` identifiers because live `docs_chain_length` values no longer identified an unused revision number; `MERGE ... ON CREATE` therefore retained an older snapshot while the reset continued.

The repair was fail-closed and exact. A manifest pinned all 18 identities, prior and current text SHA-256 values, current lifecycle, expected chain value, and a collision-free next revision identifier. One transaction required accepted/accepted lifecycle, byte-identical current text, the exact expected chain, and absence of every proposed revision id; a partial match would roll back. It appended **18/18** missing prior snapshots and advanced their chain counters. Final independent graph reconciliation confirms an exact prior-text DocsRevision for **40/40** changed identities. The collision is a source defect in the snapshot primitive and should be repaired outside this evidence-only file before the remaining 135 identities run.

## Reviewer scores: prior versus refreshed

Thirty-nine prior scores are the historical graph aggregates captured before reset. The one marked with an asterisk is the current-quorum score over the exact prior snapshot described above. Delta is `refreshed - prior`.

| Standard Name identity | Prior | Refreshed | Signed delta | Direction |
|---|---:|---:|---:|---|
| `accumulated_deposited_energy_of_plasma_facing_component` | 0.86250 | 0.90625 | +0.04375 | not down |
| `accumulated_total_particle_count_due_to_gas_injection` | 0.93125 | 0.93125 | +0.00000 | not down |
| `area_of_poloidal_magnetic_field_probe` | 0.90000 | 0.92500 | +0.02500 | not down |
| `area_of_toroidal_magnetic_field_probe` | 0.95000 | 0.94375 | -0.00625 | **down** |
| `atomic_mass` | 0.98125 | 0.99375 | +0.01250 | not down |
| `breakdown_initial_time` | 0.91250 | 0.97500 | +0.06250 | not down |
| `cold_neutral_fraction` | 0.85000 | 0.96875 | +0.11875 | not down |
| `cold_neutral_temperature` | 0.90000 | 0.89375 | -0.00625 | **down** |
| `coolant_temperature_at_inlet` | 0.88750 | 0.93750 | +0.05000 | not down |
| `coolant_temperature_at_outlet` | 0.91250 | 0.88125 | -0.03125 | **down** |
| `coolant_transit_time_of_plant_component_port` | 0.92500 | 0.93125 | +0.00625 | not down |
| `current_of_passive_loop` | 0.88750 | 0.93125 | +0.04375 | not down |
| `derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume_of_flux_surface` | 0.86250 | 0.93750 | +0.07500 | not down |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | 0.91250 | 0.95625 | +0.04375 | not down |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | 0.87500 | 0.96875 | +0.09375 | not down |
| `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy` | 0.92500 | 0.92500 | +0.00000 | not down |
| `difference_of_vacuum_poloidal_current_function_and_initial_vacuum_poloidal_current_function` | 0.93125 | 0.96250 | +0.03125 | not down |
| `effective_charge` | 0.87500 | 0.95625 | +0.08125 | not down |
| `effective_turn_count_of_coil_conductor_element` | 0.88750 | 0.90000 | +0.01250 | not down |
| `effective_turn_count_of_passive_loop` | 0.92500 | 0.91875 | -0.00625 | **down** |
| `electron_density_at_divertor_target` | 0.87500 | 0.95000 | +0.07500 | not down |
| `electron_density_at_magnetic_axis` | 0.90000 | 0.91250 | +0.01250 | not down |
| `electron_density_at_plasma_boundary` | 0.94375 | 1.00000 | +0.05625 | not down |
| `electron_temperature` | 0.85625 | 0.95625 | +0.10000 | not down |
| `electron_temperature_at_divertor_target` | 0.93125 | 0.98750 | +0.05625 | not down |
| `electron_temperature_at_magnetic_axis` | 0.92500 | 0.97500 | +0.05000 | not down |
| `elongation_of_flux_surface` | 0.86250 | 0.96875 | +0.10625 | not down |
| `elongation_of_plasma_boundary` | 0.93750 | 0.96875 | +0.03125 | not down |
| `energy_confinement_enhancement_factor` | 0.94375 | 0.96250 | +0.01875 | not down |
| `equilibrium_weight_of_flux_loop` | 0.90000 | 0.88125 | -0.01875 | **down** |
| `equilibrium_weight_of_poloidal_magnetic_field_probe` | 0.90625 | 0.95000 | +0.04375 | not down |
| `etendue_of_hard_xray_detector` | 0.91250 | 0.93125 | +0.01875 | not down |
| `etendue_of_spectrometer_channel` | 0.98125 | 0.96875 | -0.01250 | **down** |
| `faraday_angle` | 0.91250 | 0.95625 | +0.04375 | not down |
| `flux_surface_averaged_inverse_of_major_radius` | 0.86250 | 0.94375 | +0.08125 | not down |
| `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude` | 0.87500 | 0.95000 | +0.07500 | not down |
| `flux_surface_averaged_inverse_of_square_of_major_radius` | 0.86250 | 0.97500 | +0.11250 | not down |
| `flux_surface_averaged_magnetic_field_magnitude` | 0.87500 | 0.91250 | +0.03750 | not down |
| `flux_surface_averaged_major_radius`* | 0.81250 | 0.99375 | +0.18125 | not down |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | 0.87500 | 0.94375 | +0.06875 | not down |

The six downward moves are small but retained rather than averaged away: -0.00625 for `area_of_toroidal_magnetic_field_probe`, -0.00625 for `cold_neutral_temperature`, -0.03125 for `coolant_temperature_at_outlet`, -0.00625 for `effective_turn_count_of_passive_loop`, -0.01875 for `equilibrium_weight_of_flux_loop`, and -0.01250 for `etendue_of_spectrometer_channel`.

## Verbatim prior/new documentation pairs

The following eight pairs are byte-for-byte graph text, HTML-escaped only for display.

<h3><code>breakdown_initial_time</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>The initial breakdown time is the event time marking plasma initiation and the onset of discharge-current flow.

It identifies the beginning of the breakdown phase rather than an elapsed interval or the duration of the discharge. The [pulse duration](name:pulse_duration) uses this breakdown event as its starting point; the time-origin convention is external to the physical event definition.</pre></td>
<td><pre>This quantity identifies the instant at which plasma breakdown begins: plasma is initiated and discharge current starts to flow.

It applies specifically to the onset of the plasma-breakdown phase in a discharge. It is an event timestamp, not an elapsed duration, period, or time assigned to a later phase of the discharge.

The [pulse duration](name:pulse_duration) uses this breakdown instant as its starting point and measures the subsequent confined-plasma interval; therefore, this quantity and pulse duration represent distinct timing concepts.</pre></td>
</tr></tbody></table>

<h3><code>coolant_temperature_at_outlet</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>This quantity is the absolute thermodynamic temperature of the coolant stream at a cooling component or loop outlet. It defines the coolant’s downstream thermal state and is the coolant-specific form of [temperature at outlet].

The outlet value refers to the coolant at the outlet boundary and excludes the inlet state, which is represented by [coolant temperature at inlet](name:coolant_temperature_at_inlet). It applies to the coolant or working fluid identified as the cooling medium and does not represent the temperature of the cooled component or the temperature rise across the circuit.</pre></td>
<td><pre>This quantity is the absolute thermodynamic temperature of the coolant fluid at the outlet of a cooling component, cooling loop, or comparable coolant circuit. It describes the bulk fluid state at the exit boundary.

The scope is limited to coolant at an outlet and does not represent the inlet temperature, a wall temperature, or an aggregate temperature over the entire cooling system. It is an outlet state value for the selected cooling path, not a mass-flow rate or a deposited-power quantity.

Together with the [coolant temperature at the inlet](name:coolant_temperature_at_inlet) and the [coolant mass flow rate](name:ratio_of_coolant_mass_to_time), this temperature determines the coolant enthalpy change used to assess heat transferred into the fluid. The enthalpy balance distinguishes thermal power deposited in the coolant from the outlet temperature itself.</pre></td>
</tr></tbody></table>

<h3><code>etendue_of_hard_xray_detector</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>Etendue is the aggregate geometric acceptance of the optical system associated with a hard X-ray detector. It characterizes how much radiance can be coupled to the detector, independently of detector quantum efficiency or spectral response.

$$
G = A\Omega
$$

where $G$ is the etendue, $A$ is the detector collecting area, and $\Omega$ is the solid angle accepted by the optical system. Etendue is an optical-throughput quantity, not a linear detector dimension such as the [width of a hard X-ray detector](name:width_of_hard_xray_detector) or its [height](name:height_of_hard_xray_detector).</pre></td>
<td><pre>Étendue is the geometric throughput of the optical system associated with a hard X-ray detector. It characterizes the angular and spatial acceptance that determines how incident radiation is coupled into the detector.

The defining relation is

$$
G = A\Omega
$$

where $G$ is the detector étendue, $A$ is the effective collecting area, and $\Omega$ is the accepted solid angle.

This quantity applies to the detector-level optical system and uses effective area and angular acceptance for the associated detector; it does not specify whether those properties arise from one or multiple optical elements. It is distinct from the detector’s linear dimensions, such as its [height](name:height_of_hard_xray_detector) and width, which do not determine optical throughput alone.

Étendue provides the geometric factor relating incident [hard X-ray brightness](name:hard_xray_brightness) to radiation coupled into the detector and is the same physical optical-throughput concept represented by [etendue_of_spectrometer](name:etendue_of_spectrometer), with the owning optical system changed from a spectrometer to a hard X-ray detector.</pre></td>
</tr></tbody></table>

<h3><code>elongation_of_flux_surface</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>Poloidal elongation is the aspect ratio of a nested magnetic flux-surface cross-section, comparing its vertical extent with its radial extent. It describes plasma shaping independently of the surface&#x27;s absolute size.

In the right-handed cylindrical $(R, \phi, Z)$ frame, where $R$ is major radius and $Z$ is vertical height, the elongation of the surface labeled by $\psi$ is defined by

$$
\kappa(\psi) = \frac{a_Z(\psi)}{a_R(\psi)}
$$

where $a_Z$ and $a_R$ are the vertical and radial semi-extents of that flux surface, and $\psi$ is its magnetic-flux label.

This parent quantity applies to every nested magnetic flux surface; the boundary-specific case is [elongation of the plasma boundary](name:elongation_of_plasma_boundary). Its radial variation is described by [radial derivative of elongation of the flux surface](name:radial_derivative_of_elongation_of_flux_surface).</pre></td>
<td><pre>Elongation is the shape ratio of an individual nested magnetic flux surface, describing how extended its poloidal cross-section is vertically relative to horizontally.

For a given flux surface, the elongation is defined by

$$
\kappa = \frac{a_v}{a_h}
$$

where $\kappa$ is the elongation, $a_v$ is the vertical semi-axis of the poloidal cross-section, and $a_h$ is its horizontal semi-axis.

The quantity applies separately to each nested magnetic flux surface and is a shape ratio rather than an integrated, summed, or averaged quantity. It is not restricted to the outermost surface; the distinct [elongation of the plasma boundary](name:elongation_of_plasma_boundary) refers specifically to the last closed plasma boundary.

The [radial derivative of elongation of a flux surface](name:radial_derivative_of_elongation_of_flux_surface) describes the change of $\kappa$ across nested surfaces, whereas this quantity gives the elongation at one surface.</pre></td>
</tr></tbody></table>

<h3><code>energy_confinement_enhancement_factor</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>The energy confinement enhancement factor expresses the plasma&#x27;s energy confinement time relative to the IPB98(y,2) reference scaling prediction. Values above unity indicate confinement exceeding the reference prediction, while values below unity indicate weaker confinement.

$$
H_{98} = \frac{\tau_E}{\tau_{E,\mathrm{IPB98(y,2)}}}
$$

where $H_{98}$ is the enhancement factor, $\tau_E$ is the plasma [energy confinement time](name:energy_confinement_time), and $\tau_{E,\mathrm{IPB98(y,2)}}$ is the energy confinement time predicted by the IPB98(y,2) scaling relation.

This quantity is a comparative factor, not the confinement time or stored plasma energy itself; its interpretation depends on the stated reference scaling relation.</pre></td>
<td><pre>The energy confinement enhancement factor is the multiplicative departure of a plasma’s global energy confinement time from the IPB98(y,2) reference scaling prediction. It indicates whether energy is retained longer or shorter than the reference predicts for the corresponding scaling inputs.

The factor is defined by:

$$
h_{98} = \frac{\tau_E}{\tau_{E,\mathrm{IPB98(y,2)}}}
$$

where $h_{98}$ is the energy confinement enhancement factor, $\tau_E$ is the plasma energy confinement time, and $\tau_{E,\mathrm{IPB98(y,2)}}$ is the energy confinement time predicted by the IPB98(y,2) reference scaling.

This is a global plasma quantity: the numerator represents the confinement of the total stored plasma energy relative to the corresponding net energy-loss power, aggregated over the plasma rather than assigned to a local region or individual transport channel. It is not itself an energy confinement time, stored energy, power, or local transport coefficient.

A value of one denotes agreement with the reference scaling; a value greater than one denotes longer confinement, and a value less than one denotes shorter confinement than the reference prediction.</pre></td>
</tr></tbody></table>

<h3><code>equilibrium_weight_of_poloidal_magnetic_field_probe</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>An equilibrium weight assigns the relative importance of a poloidal magnetic-field probe constraint when fitting an equilibrium. It applies to the discrepancy between the probe&#x27;s measured and reconstructed poloidal magnetic-field values.

The contribution of the constraint to the objective function is represented by

$$
J = \frac{1}{2}\sum_i \frac{w_i^2\left(B_{\mathrm{rec},i}-B_{\mathrm{meas},i}\right)^2}{\sigma_i^2}
$$

where $J$ is the equilibrium objective function, $w_i$ is the equilibrium weight, $B_{\mathrm{rec},i}$ is the reconstructed poloidal magnetic-field value, $B_{\mathrm{meas},i}$ is the measured value, and $\sigma_i$ is the corresponding measurement-error scale. The weight modifies the constraint&#x27;s relative influence and is distinct from the field value, residual, and measurement uncertainty. It is dimensionless and conventionally nonnegative.</pre></td>
<td><pre>The equilibrium weight of a poloidal magnetic-field probe is a dimensionless factor assigned to the residual of one probe constraint in an equilibrium reconstruction objective.

For constraint index $i$, the objective contribution is defined through

$$
J = \frac{1}{2}\sum_i w_i^2\frac{(q_i^{\mathrm{rec}}-q_i^{\mathrm{meas}})^2}{\sigma_i^2}
$$

where $J$ is the equilibrium objective function, $w_i$ is the weight of constraint $i$, $q_i^{\mathrm{rec}}$ is its reconstructed poloidal magnetic-field value, $q_i^{\mathrm{meas}}$ is its measured value, and $\sigma_i$ is the standard deviation of its measurement error.

This quantity applies to a single poloidal magnetic-field probe constraint and does not represent the probe field, its residual, or the measurement uncertainty. It is not an aggregate over multiple probes; the objective function aggregates the individually weighted constraint contributions. The analogous [equilibrium weight of an interferometer beam](name:equilibrium_weight_of_interferometer_beam) applies the same weighting role to a different constraint observable.</pre></td>
</tr></tbody></table>

<h3><code>electron_density_at_divertor_target</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>Electron density at the divertor target is the local number density of free electrons in the plasma immediately adjacent to a divertor target surface, evaluated at the entrance to the electrostatic sheath. It is the electron-specific member of [density_at_divertor_target](name:density_at_divertor_target).

The local electron number density is defined by the differential particle count:

$$
n_e = \frac{dN_e}{dV}
$$

where $N_e$ is the number of free electrons and $V$ is the enclosing plasma volume. The volume approaches the sheath entrance from the plasma side.

This quantity includes free electrons in the plasma and excludes electrons bound in neutral atoms or molecules and electrons in the target material. It complements [electron_temperature_at_divertor_target](name:electron_temperature_at_divertor_target), which characterizes the electron energy distribution at the same sheath entrance.</pre></td>
<td><pre>Number density of the electron species at the divertor target is the local particle count per volume at the sheath entrance immediately upstream of the target-facing plasma boundary. It is the electron-specific member of [density_at_divertor_target](name:density_at_divertor_target).

The local electron number density is defined by the differential particle count:

$$
n_e = \frac{dN_e}{dV}
$$

where $N_e$ is the number of free electrons in a small plasma volume $V$ whose center approaches the sheath entrance from the plasma side at the divertor target.

This quantity applies to the free-electron population immediately adjacent to the divertor target and counts each free electron once, without charge-state weighting. Electrons bound in atoms or molecules, ions, and other neutral particles are excluded.

Unlike [total_ion_density_at_divertor_target](name:total_ion_density_at_divertor_target), which sums positive-ion densities over ion species and charge states, this quantity describes the electron population and is not an ionic density or a charge density.</pre></td>
</tr></tbody></table>

<h3><code>area_of_poloidal_magnetic_field_probe</code></h3>
<table>
<thead><tr><th>Prior documentation</th><th>Refreshed documentation</th></tr></thead>
<tbody><tr>
<td><pre>The poloidal magnetic-field probe area is the geometric surface area enclosed by one sensing turn of the coil. It determines the magnetic flux coupled to that turn when the field is approximately uniform and normal to the enclosed surface.

The area is defined by

$$
A_p = \int_{S_p} dS
$$

where $S_p$ is the surface bounded by one coil turn and $dS$ is its differential area element. For a uniform normal poloidal field, the linked flux is proportional to $N A_p B_p$, where $N$ is the dimensionless turn count and $B_p$ is the poloidal magnetic-field component normal to the turn surface.

This is a one-turn geometric area and does not include the turn-count factor or other calibration factors. It is the area parameter underlying the induced [voltage of a poloidal magnetic field probe](name:voltage_of_poloidal_magnetic_field_probe) and is analogous to the one-turn area of a [toroidal magnetic field probe](name:area_of_toroidal_magnetic_field_probe).</pre></td>
<td><pre>The quantity is the geometric cross-sectional area enclosed by one winding turn of a poloidal magnetic-field probe coil. It represents the magnetic-flux coupling area of an individual pickup-coil turn and is nonnegative.

For a winding composed of identical turns, the ideal effective flux-sensing area is

$$
A_{\mathrm{eff}} = N A_{\mathrm{turn}}
$$

where $A_{\mathrm{eff}}$ is the signed effective geometric flux-sensing area of the complete winding, $N$ is the signed number of winding turns, and $A_{\mathrm{turn}}$ is the area of one turn.

This quantity is defined per turn and applies only to the probe-coil geometry; it does not include the number of turns, winding orientation, or calibration factors. Consequently, it is distinct from the complete effective area of a flux-loop sensor, such as [area of a flux loop](name:area_of_flux_loop). The corresponding [turn count of a poloidal magnetic-field probe](name:turn_count_of_poloidal_magnetic_field_probe) combines with this area to determine the winding&#x27;s ideal effective flux-sensing area.</pre></td>
</tr></tbody></table>

## Durable evidence inputs

- `zero-call-priced-admission.json`: exact 40-name selection, live pre-run rows, prior prose and scores, rendered pricing, and zero-call admission.
- `reset-receipt.json`: exact scope id and 40/40 reset result.
- `sn-run-live.log`: full executed `sn run` transcript.
- `post-run-snapshot.json`: SNRun, LLMCost, per-pool operations, final prose, final scores, signed deltas, and lifecycle outcomes.
- `prior-score-supplement.json`: current-quorum score over the one prior snapshot lacking historical aggregate authority.
- `snapshot-repair-manifest.json` and `snapshot-repair-receipt.json`: exact 18-row snapshot collision repair and post-transaction verification.
- `final-evidence-check.log`: independent quantitative reconciliation against the live graph.

All run-local evidence is under `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T202047542522-n-westrefresh/`.
