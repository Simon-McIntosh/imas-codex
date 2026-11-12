# Domain distinctions I found hard, and the rules I applied

**Physics domain vs diagnostic domain.** The hardest recurring call. Rule: if a
plasma model predicts or consumes the quantity as part of the plasma state, I
used the physics domain; a diagnostics domain only where the quantity is an
instrument signal or a hardware setting (detector power, camera brightness,
probe coordinate, pixel bandpass, channel bias voltage).

**`core_plasma_physics` vs `edge_plasma_physics`.** A species density or
velocity at a confined-plasma profile locus — magnetic axis, last closed flux
surface ("plasma boundary"), line or volume average, internal transport
barrier — I filed as core, because it is a profile value. Where the
description itself said "H-mode edge pedestal", "edge transport barrier" or
"in the edge plasma", or the quantity concerns neutrals, ion-neutral friction,
flux-limiter closures or separatrix fields, I filed edge. The boundary and
pedestal-top items are the ones I am least sure of.

**`transport` vs `core_plasma_physics`.** Coefficients, diffusivities and
fluxes went to transport; local scalar state variables (density, temperature,
pressure, charge number, abundance ratio) went to core.

**`equilibrium` vs `magnetohydrodynamics`.** Applied and vacuum field
components, flux-surface geometry, safety factor, and total/diamagnetic
current profiles in force balance went to equilibrium; perturbed
eigenfunctions and instability loci (sawtooth) went to MHD.

**`plasma_wall_interactions` vs `edge_plasma_physics`.** Anything incident on
or emitted from a material surface went to plasma-wall interactions; the
surrounding plasma state went to edge.

**`waves` vs `auxiliary_heating`.** Wave-field, polarization and absorption
quantities went to waves; antenna/strap and coil geometry and delivered system
power went to auxiliary heating.

I used `general` for no item. Lowest-confidence: annulus-centre coordinate,
passive-loop current, plasma-filament angle, post-sawtooth electron pressure.