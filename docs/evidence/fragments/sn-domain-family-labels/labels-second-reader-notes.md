# Second reader — domains that were hard to tell apart

**transport vs core_plasma_physics.** Rule: a coefficient, flux, damping rate or
source that relates a gradient to a flux is `transport`; a value of a profile, a
flux-surface average, a volume or chord average, or a local pressure or density is
`core_plasma_physics`. Diffusivities, conductivities, viscosities and momentum
fluxes went to transport; densities and pressures at a named locus, and abundance
ratios, went to core.

**edge_plasma_physics vs plasma_wall_interactions vs divertor_physics.** Rule:
the quantity belongs to the plasma or neutral population at the boundary
(neutral velocities and fluxes, edge-fluid closure coefficients, pedestal and
separatrix values) → `edge_plasma_physics`; the quantity belongs to the material
surface or its interaction, such as component thickness, incident ion flux at a
wall, emission from a surface, or the wall point nearest the plasma →
`plasma_wall_interactions`; anything on a divertor target structure, such as
neutral power there or nuclear heating, → `divertor_physics`. Neutral
populations in the edge stayed with edge.

**equilibrium vs magnetohydrodynamics.** Static flux-surface geometry, force
balance and the diamagnetic current → `equilibrium`; eigenfunction perturbations
(plasma and vacuum real/imaginary parts) and sawtooth loci →
`magnetohydrodynamics`.

**equilibrium vs plasma_control.** A coil field named on a control surface, used
for vertical stabilisation → `plasma_control`; geometry derivatives and the
three-dimensional vacuum field → `equilibrium`.

**magnetic_field_systems vs magnetic_field_diagnostics.** A property of a coil,
correction coil, passive loop, annulus or ferritic element → systems; a signal or
location of a measuring sensor (flux loop, probe, fibre-optic current sensor) →
diagnostics.

**auxiliary_heating vs waves.** Launcher hardware geometry or RF phase →
`auxiliary_heating`; field phasors, polarisation and absorbed wave power →
`waves`.

**radiation_measurement_diagnostics vs spectroscopy.** Imaging or band-integrated
radiance and X-ray detector band edges → radiation diagnostics; mass-resolved
channels and impurity emission → `spectroscopy`.

**fueling.** Any injection rate, accumulated injected inventory, or a locus
defined by injection (pellet path, prefill) → `fueling`, even when the measured
quantity itself is a plasma density or temperature.

`general` was used for nothing: every item had a specific home.