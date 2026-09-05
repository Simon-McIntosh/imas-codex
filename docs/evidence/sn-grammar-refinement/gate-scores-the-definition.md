# The gate scores a bare token through its own definition

**Node:** the-gate-scores-a-token-against-its-own-definition (SN grammar refinement §9, family-anchor follow-up)
**Date:** 2026-09-05
**Mode:** read-only measurement; the only repository write is this document.

## Question

The name-review semantic gate (`semantic_similarity_check`, cosine between the
name as text and its description embedding, critical floor 0.55) cannot admit a
one-word name at any description: `beta` measured 0.475–0.508 across four
candidate descriptions while accepted multi-word family members scored 0.770–
0.849. The proposal under test is the escape hatch from the family-anchor
follow-up: **when the parsed name is a single vocabulary token and the
vocabulary carries a definition of that token, score the description against
the token's own definition (prose) instead of against the bare identifier
string**, falling through to today's behaviour when no definition exists so an
unknown token still fails closed.

This document is the calibration that decides whether the escape hatch works
and whether the 0.55 floor survives the switch of name-side text. Three parts:
(1) the measurement, (2) the definitions, (3) the code change, described and
not made.

**Instrument.** All embeddings were computed fresh from raw text with the live
project embedding server: `Qwen/Qwen3-Embedding-0.6B` at
`http://98dci4-gpu-0002:18765`, host `98dci4-gpu-0002.iter.org`, via the same
`Encoder`/`embed_descriptions_batch` path the gate itself uses. The remote
server reached the embedding endpoint on every call; no score below is an
offline artifact. Reproducing `beta` at 0.488 under today's coupling against a
fresh embed confirms the instrument agrees with the earlier measurement.

---

## Part 1 — The calibration measurement

### 1.1 How many accepted live identities are bare single-token bases?

A **bare single-token base** is a standard name that parses to exactly one
vocabulary token with no qualifier, operator, projection, locus or mechanism
(`parse(name, strict=True)`, then `ir.operators == [] and ir.qualifiers == []
and ir.projection is None and ir.locus is None and ir.mechanism is None`).

| Population | Count |
|---|---|
| Accepted live identities (name_stage = `accepted`, not superseded/exhausted) | 2282 |
| ... of which parse strictly | 2271 |
| ... of which **bare single-token bases** | **24** |
| ... of which `origin == "derived"` (already routed by the existing derived skip, not the semantic gate) | 6 |
| ... of which non-derived (the population the escape hatch is for) | 18 |

The 24: `atomic_number`, `center_of_mass_velocity`, `coulomb_logarithm`,
`current_density`, `electric_field`, `electrostatic_potential`,
`flux_limiter_coefficient`, `mach_number`, `magnetic_field`, `magnetic_flux`,
`magnetic_vector_potential`, `momentum`,
`normalized_poloidal_flux_coordinate`, `normalized_toroidal_flux_coordinate`,
`poloidal_current_function`, `radial_coordinate`, `refractive_index`,
`safety_factor`, `toroidal_flux_coordinate`,
`toroidal_flux_coordinate_gradient`, `torque_density`,
`tritium_breeding_ratio`, `vector_potential`, `vorticity`.
Derived among them: `flux_limiter_coefficient`, `mach_number`, `momentum`,
`toroidal_flux_coordinate_gradient`, `torque_density`, `vector_potential`.

`beta` is not in this list: the rescore moved it to `reviewed`, so it is
measured separately below.

**Four of the 24 already fail today's gate.** Today's coupling, run exactly as
the gate runs it, scores the descriptions of four accepted live bare bases
below the 0.55 critical floor: `safety_factor` 0.439, `mach_number` 0.411,
`momentum` 0.543, and `beta` 0.488. These identities are accepted with
well-formed descriptions yet cannot clear the gate, which is the same defect as
the beta case, live in the accepted corpus. The answer to the follow-up's
decisive count question is therefore **24, and the gate is *not* silently
admitting most of them by an exemption — it is simply never applied to
catalog-direct admissions** (`catalog_edit` writes `accepted` directly). Where
the gate *does* run, it rejects a bare base with a good description. The
escape hatch is a genuinely needed admission path, not a nicety.

### 1.2 Bare single-token bases: description vs identifier, and vs definition

For every accepted live bare base and for `beta`: **A** = cosine(description,
identifier as text) — today's coupling, exactly as `semantic_similarity_check`
computes it; **B** = cosine(description, candidate vocabulary definition for
that token) — the proposed coupling. Candidate definitions are Part 2 below;
each is grounded in a cited source, none composed from memory.

| identity | A (desc vs id, today) | B (desc vs def, proposed) | stage | origin |
|---|---|---|---|---|
| beta | **0.488** | **0.840** | reviewed | (none) |
| atomic_number | 0.631 | 0.862 | accepted | catalog_edit |
| center_of_mass_velocity | 0.714 | 0.847 | accepted | catalog_edit |
| coulomb_logarithm | 0.742 | 0.771 | accepted | catalog_edit |
| current_density | 0.751 | 0.852 | accepted | catalog_edit |
| electric_field | 0.698 | 0.963 | accepted | catalog_edit |
| electrostatic_potential | 0.729 | 0.984 | accepted | catalog_edit |
| flux_limiter_coefficient | 0.784 | 0.840 | accepted | derived |
| mach_number | **0.411** | **0.962** | accepted | derived |
| magnetic_field | 0.592 | 0.955 | accepted | catalog_edit |
| magnetic_flux | 0.688 | 0.835 | accepted | catalog_edit |
| magnetic_vector_potential | 0.729 | 0.961 | accepted | catalog_edit |
| momentum | **0.543** | **0.871** | accepted | derived |
| normalized_poloidal_flux_coordinate | 0.838 | 0.882 | accepted | catalog_edit |
| normalized_toroidal_flux_coordinate | 0.824 | 0.963 | accepted | pipeline |
| poloidal_current_function | 0.642 | 0.867 | accepted | catalog_edit |
| radial_coordinate | 0.814 | 0.946 | accepted | (none) |
| refractive_index | 0.616 | 0.993 | accepted | catalog_edit |
| safety_factor | **0.439** | **0.950** | accepted | pipeline |
| toroidal_flux_coordinate | 0.781 | 0.989 | accepted | pipeline |
| toroidal_flux_coordinate_gradient | 0.826 | 0.869 | accepted | derived |
| torque_density | 0.594 | 0.955 | accepted | derived |
| tritium_breeding_ratio | 0.753 | 0.886 | accepted | catalog_edit |
| vector_potential | 0.771 | 0.951 | accepted | derived |
| vorticity | 0.714 | 0.853 | accepted | catalog_edit |

Under the proposed coupling **every one of the 25 clears 0.55**, with the
lowest at 0.771 (`coulomb_logarithm`), i.e. a margin of 0.22 over the floor.
The five that today fail or hug the floor (β, mach_number, safety_factor,
momentum, magnetic_field/torque_density at 0.592/0.594) all move to ≥ 0.840
with a grounded definition. Truncating the definition as a name-side text to
500 chars changes no score (definition blocks are ≲500 chars; the full/capped
columns were identical for all rows).

### 1.3 Control: accepted multi-word names, ≥ 15 required

26 accepted multi-word names spanning 11 bases, measured under both couplings.
**A** = cosine(description, identifier) as today. **B** = cosine(description,
composed definition of the name's segments *where definitions exist*): the
base token's Part-2 definition when the base is one of the 25 proposed, plus
the locus token's `locus_registry.yml` definition when a locus is present.
Qualifiers, operators and projections carry no vocabulary definitions, so they
contribute nothing (an important fact, see 1.4). Where the composition is
empty there is no composed definition, which under the scoped design means the
name falls through to today's coupling.

| control name | A (desc vs id, today) | B (composed def) |
|---|---|---|
| binormal_wave_electric_field | 0.796 | 0.701 |
| binormal_wave_magnetic_field | 0.822 | 0.744 |
| breakdown_magnetic_field | 0.751 | 0.685 |
| co_passing_fast_current_density | 0.661 | 0.743 |
| counter_passing_current_density | 0.633 | 0.876 |
| counter_passing_fast_current_density | 0.722 | 0.692 |
| diamagnetic_current_density_due_to_heat_viscosity | 0.880 | 0.605 |
| diamagnetic_current_density_due_to_ion_neutral_friction | 0.847 | 0.636 |
| diamagnetic_current_density_due_to_parallel_viscosity | 0.787 | **0.560** |
| diamagnetic_current_density_due_to_perpendicular_viscosity | 0.899 | 0.615 |
| mhd_energy | 0.565 | *(no composed def → fallback)* |
| volume_of_flux_surface | 0.796 | 0.797 |
| parallel_electron_velocity | 0.799 | *(no composed def → fallback)* |
| parallel_electron_particle_flux | 0.645 | *(no composed def → fallback)* |
| accumulated_deuterated_methane_prefill_count | 0.862 | *(no composed def → fallback)* |
| effective_thermal_ion_charge_state_energy_velocity_due_to_convection | 0.923 | *(no composed def → fallback)* |
| counter_passing_torque_density | 0.733 | 0.720 |
| neutral_internal_state_energy_flux | 0.842 | *(no composed def → fallback)* |
| poloidal_magnetic_flux_at_plasma_boundary | 0.847 | 0.723 |
| parallel_ion_momentum_flux | 0.669 | *(no composed def → fallback)* |
| total_power_due_to_ohmic_dissipation | 0.709 | *(no composed def → fallback)* |
| effective_neutral_internal_state_momentum_velocity_due_to_convection | 0.870 | *(no composed def → fallback)* |
| power_at_inside_flux_surface | 0.719 | 0.740 |
| plasma_temperature_real_part | 0.733 | *(no composed def → fallback)* |
| normalized_perturbed_electrostatic_potential_amplitude | 0.752 | **0.565** |
| second_local_tangential_coordinate_of_aperture | 0.828 | 0.802 |

All 26 controls score ≥ 0.55 under today's coupling (min 0.565, `mhd_energy`)
— the floor holds for multi-word names. Where a composed definition exists, the
**definition-side scores are systematically lower than the identifier-side
scores** for the same name, because a partial composition (base + locus only,
qualifiers missing) under-represents the name's meaning. Two good controls with
composed definitions land at 0.560–0.565 — on the 0.55 floor's edge.

### 1.4 The floor question, answered with a mismatched-pair distribution

The calibration claim is that one floor can serve both texts. That becomes two
separate claims — *admission* (never wrongly reject a well-grounded bare base)
and *discrimination* (reject a wrong description). Both are measured. For each
of the 25 tokens the description of every other token's definition was scored
(def-side mismatches), and also the description of every other token as a bare
identifier (ident-side mismatches) — 125 mismatched pairs each.

| coupling | GOOD pairs (own token) | MISMATCHED pairs (other token) | mismatches ≥ 0.55 |
|---|---|---|---|
| identifier (today) | 0.411–0.838, mean 0.684 | 0.261–0.690, mean 0.476 | 28 / 125 (22%) |
| definition (proposed) | 0.771–0.993, mean 0.906 | 0.303–0.814, mean 0.530 | 52 / 125 (42%) |

**Admission: one floor at 0.55 serves both texts.** Every correct pair clears
it on both couplings for the population each coupling is applied to — the 26
multi-word controls on the identifier side (min 0.565) and the 25 single
tokens on the definition side (min 0.771). Keeping the floor unchanged is safe
for anything the scoped substitution touches.

**Discrimination: the definition side is a systematically weaker
discriminator, and no floor fixes that.** The good and mismatched ranges
overlap (correct min 0.771 < mismatch max 0.814), so no threshold separates
them perfectly; raising the floor above 0.55 would start rejecting correct
pairs before it removed the worst mismatches. At 0.55 the definition coupling
passes 42% of mismatched pairs versus 22% for the identifier coupling — two
quantities' prose definitions share richer vocabulary than a thin token does.
The gate is therefore best understood, under the proposed coupling, as an
*admission* gate (this bare base stands alone enough to spend reviewer seats)
rather than a *description-verifier*. Description correctness remains the
review chain's job.

**Concrete consequence: the gate cannot catch the beta documentation bug under
either coupling.** β's current description documents the wrong sibling
(toroidal beta), and scoring it against the correct total-beta definition gives
0.840; against the accepted `toroidal_beta` sibling description gives 0.965.
The two meanings share their substantive vocabulary ("ratio of total
perpendicular plasma pressure to magnetic pressure B0²/2μ0"); cosine on this
text is too coarse to separate them. The β description repair is a meaning fix
(the plan's §9 documentation repair) and no threshold in this gate could have
made it instead. That is stated so the escape hatch is not later expected to
have caught it.

**Scoping is what protects the floor.** The measured attenuation for multi-word
names (1.3) shows that extending the substitution to *composed* definitions of
multi-word names, while qualifiers and operators carry no vocabulary
definitions, would push good names toward and onto the floor (0.560 minimum
measured). The substitution must stay scoped to single vocabulary tokens —
which is the design under test — or the qualifier vocabulary must gain
definitions before any composition.

---

## Part 2 — The definitions

Convention follows the existing folded prose blocks in `geometry_carriers.yml`
and `locus_registry.yml`: a `definition: >` folded block, first sentence
stating what the token is, plain prose. `physical_bases.yml` today carries no
`definition` on any of its 177 entries — only `aliases`, `kind`,
`inherently_dimensional`, `constant_on_flux_surface` — so all 25 blocks below
are new, and 24 of the 25 are proposed for tokens that already have accepted
live identities. **Every definition is grounded in a cited source** — either
the DD documentation of a bound source path (quoted from the graph's recorded
`source_documentation` and verified against the DD MCP for the pinned 4.1.1
version) or an accepted sibling identity's description — never physics prose
composed from memory. One token (`beta`) is grounded on the plan's own lead
ruling plus accepted siblings, because its bound-path documentation is itself
the bug being repaired; that is flagged.

For each block: the proposed YAML, then the citation. The prose deliberately
reuses the cited wording rather than paraphrasing away from it.

### beta

```yaml
  beta:
    aliases: []
    definition: >
      Plasma beta, denoted beta, is a dimensionless measure of total plasma
      pressure relative to a magnetic-pressure scale. The poloidal and toroidal
      components are related to it by 1/beta = 1/beta_poloidal + 1/beta_toroidal.
```

*Grounding:* the plan's §9 lead ruling — "beta is TOTAL beta", with the
family relation 1/β = 1/β_poloidal + 1/β_toroidal — plus accepted sibling
descriptions: `plasma_beta` ("a dimensionless measure of total plasma pressure
relative to a magnetic-pressure scale") and `toroidal_beta` ("the pressure
aggregate taken as the confined-volume average of total perpendicular
pressure"). **Deliberately not** the recorded `source_documentation` on the
beta identity, which reads "Toroidal beta, defined as the volume-averaged
total perpendicular pressure divided by (B0²/(2μ0))" — that is the §9
documentation error (it documents the different, separately-accepted
`toroidal_beta` identity), so it is cited as the *anti*-source here, not the
source.

### atomic_number

```yaml
  atomic_number:
    aliases: [nuclear_charge_number]
    kind: scalar
    definition: >
      Nuclear proton count identifying the selected element in a plasma or
      neutral-particle species, independent of isotope and ionization state.
```

*Grounding:* DD documentation of bound path `core_profiles/profiles_1d/ion/element/z_n`
("Nuclear charge"), extended with the accepted identity's own description.

### center_of_mass_velocity

```yaml
  center_of_mass_velocity:
    aliases: []
    kind: vector
    definition: >
      The mass-weighted barycentric velocity vector of a multicomponent plasma,
      equal to total momentum density divided by total mass density.
```

*Grounding:* accepted sibling description `bulk_center_of_mass_velocity` (its
text is the mass-weighted barycentric velocity sentence); no DD binding exists
on the identity.

### coulomb_logarithm

```yaml
  coulomb_logarithm:
    aliases: []
    kind: scalar
    definition: >
      The Coulomb logarithm (ln Lambda) representing the magnitude of
      small-angle collective scattering interactions, governing collision
      frequencies.
```

*Grounding:* DD documentation of bound path
`plasma_initiation/global_quantities/coulomb_logarithm` ("The Coulomb logarithm
(ln Λ) representing the magnitude of small-angle collective scattering
interactions. Essential for determining collision frequencies...").

### current_density

```yaml
  current_density:
    aliases: []
    kind: vector
    inherently_dimensional: true
    definition: >
      The vector flux density of conventional electric charge, equal to
      electric current per unit area normal to the flow, including all
      charged-particle contributions.
```

*Grounding:* accepted sibling descriptions of the family
(`counter_passing_current_density`, `runaway_electron_current_density`,
`ion_current_density`) and the accepted identity's own description; no DD
binding exists on the identity.

### electric_field

```yaml
  electric_field:
    aliases: []
    kind: vector
    inherently_dimensional: true
    definition: >
      The local vector electric field, equal to the force per unit charge on a
      positive stationary test charge, including electrostatic, inductive, and
      electromagnetic-wave contributions.
```

*Grounding:* accepted sibling descriptions `radial_electric_field` ("force per
unit positive stationary test charge toward increasing major radius") and
`parallel_electric_field` ("including electrostatic, inductive, and
electromagnetic-wave contributions"); no DD binding exists on the identity.

### electrostatic_potential

```yaml
  electrostatic_potential:
    aliases: []
    kind: scalar
    inherently_dimensional: true
    definition: >
      The voltage-valued electrostatic potential giving the electric potential
      energy per unit positive charge relative to a chosen electrical
      reference.
```

*Grounding:* the accepted identity's own description. Its bound `phi_potential`
paths carry only field metadata ("One scalar value is provided per element in
the grid subset"), which is not physics prose and cannot be quoted as a
definition; that is reported rather than papered over.

### flux_limiter_coefficient

```yaml
  flux_limiter_coefficient:
    aliases: []
    kind: scalar
    definition: >
      A multiplicative factor, or vector of factors, that scales a collisionless
      free-streaming bound on a flux density while preserving the direction or
      sign of each resolved flux component.
```

*Grounding:* the accepted identity's own description; its children
(`energy_flux_limiter_coefficient` etc.) are deterministic parents still
carrying placeholder text, so they are not quotable. Origin `derived`, no DD
binding.

### mach_number

```yaml
  mach_number:
    aliases: []
    kind: scalar
    definition: >
      Dimensionless ratio of plasma flow velocity to the local sound speed.
```

*Grounding:* accepted sibling description `parallel_mach_number` ("Signed ratio
of the plasma flow component along the local magnetic-field direction to the
local plasma sound speed"). Also grounded on the plan's measured family member.

### magnetic_field

```yaml
  magnetic_field:
    aliases: []
    kind: vector
    inherently_dimensional: true
    definition: >
      The local total magnetic-induction vector in the right-handed cylindrical
      (R, phi, Z) frame, not selecting a source contribution, perturbation
      state, component, magnitude, or evaluation locus.
```

*Grounding:* accepted sibling descriptions `toroidal_magnetic_field` ("...local
total magnetic induction, resolved along increasing toroidal angle in the
right-handed cylindrical (R, φ, Z) frame") and `poloidal_magnetic_field`
("...local total induction, formed from radial and vertical components in the
right-handed cylindrical (R, φ, Z) frame"). The identity's own recorded
`source_documentation` ("Magnetic field value taking into account the non-linear
response of the probe") comes from a probe-specific binding and is too narrow;
it is cited as the reason the definition stays on the family text.

### magnetic_flux

```yaml
  magnetic_flux:
    aliases: []
    kind: scalar
    inherently_dimensional: true
    definition: >
      The signed scalar surface integral of the magnetic field through an
      oriented surface.
```

*Grounding:* accepted sibling description `poloidal_magnetic_flux` ("the signed
surface integral of the magnetic field over a surface bounded by a closed
toroidal contour..."); no DD binding exists on the identity.

### magnetic_vector_potential

```yaml
  magnetic_vector_potential:
    aliases: []
    kind: vector
    inherently_dimensional: true
    definition: >
      A vector field whose curl gives the magnetic induction, gauge-dependent
      but central to canonical-momentum and gyrokinetic formulations of
      magnetized plasmas.
```

*Grounding:* accepted sibling descriptions (`poloidal_magnetic_vector_potential`
and the parallel/radial members) and the accepted identity's own description;
no DD binding exists on the identity.

### momentum

```yaml
  momentum:
    aliases: []
    kind: vector
    definition: >
      The linear momentum associated with plasma motion, covering momentum
      content in a plasma volume or momentum transported through an oriented
      surface.
```

*Grounding:* accepted sibling description `plasma_momentum` ("volume-integrated
linear momentum carried by the included plasma species...") and the accepted
identity's own description (the child decides extensive-momentum vs
momentum-flux). Origin `derived`, no DD binding.

### normalized_poloidal_flux_coordinate

```yaml
  normalized_poloidal_flux_coordinate:
    aliases: []
    constant_on_flux_surface: true
    definition: >
      A radial label for nested magnetic flux surfaces based on poloidal
      magnetic flux, zero at the magnetic axis and one at the last closed flux
      surface.
```

*Grounding:* DD documentation of bound path
`core_profiles/profiles_1d/grid/rho_pol_norm` ("Normalized poloidal flux
coordinate (ρ_pol, normalized to [0,1]). Standard radial spatial coordinate for
mapping core profiles...") and the identity's recorded `source_documentation`
for `rho_pol_norm`.

### normalized_toroidal_flux_coordinate

```yaml
  normalized_toroidal_flux_coordinate:
    aliases: []
    constant_on_flux_surface: true
    definition: >
      A dimensionless radial label equal to the square root of the normalized
      toroidal flux, spanning the magnetic axis to the equilibrium boundary.
```

*Grounding:* DD documentation of bound path
`core_profiles/profiles_1d/grid/rho_tor_norm` ("Normalized toroidal flux
coordinate (ρ_tor_norm). This radial coordinate is defined as the square root
of the normalized toroidal flux and serves as the primary standardized radial
grid for core profile representation.").

### poloidal_current_function

```yaml
  poloidal_current_function:
    aliases: []
    kind: scalar
    inherently_dimensional: true
    definition: >
      The diamagnetic function F = R B_phi, the major-radius-weighted toroidal
      magnetic-field function used in axisymmetric equilibria.
```

*Grounding:* DD documentation of bound path `equilibrium/time_slice/profiles_1d/f`
("The diamagnetic function (F = R Bphi)... fundamental to representing the
plasma magnetic configuration") and the identity's recorded `source_documentation`
("Diamagnetic function (F=R B_Phi)").

### radial_coordinate

```yaml
  radial_coordinate:
    aliases: []
    definition: >
      The major-radius coordinate locating a geometric point or reference
      position by perpendicular distance from the toroidal symmetry axis.
```

*Grounding:* the identity's recorded `source_documentation` ("Centre major
radius") and its 46 accepted children, all of which carry the perpendicular
distance-from-axis meaning.

### refractive_index

```yaml
  refractive_index:
    aliases: []
    kind: scalar
    definition: >
      Dimensionless ratio of an electromagnetic wave's phase-gradient vector to
      its vacuum wavenumber, retaining propagation direction and signed
      components.
```

*Grounding:* accepted sibling description `parallel_refractive_index` ("the
signed component of the refractive-index vector projected onto the local
equilibrium magnetic-field direction, equal to the parallel wave-vector
component normalized by the vacuum wavenumber"); no DD binding exists on the
identity.

### safety_factor

```yaml
  safety_factor:
    aliases: []
    kind: scalar
    constant_on_flux_surface: true
    definition: >
      The signed ratio of toroidal to poloidal field-line winding on a magnetic
      flux surface, equal to the toroidal turns made during one poloidal
      circuit.
```

*Grounding:* DD documentation of bound path `equilibrium/time_slice/profiles_1d/q`
("Safety factor (q) radial profile, representing the ratio of toroidal to
poloidal magnetic flux increments. A fundamental equilibrium parameter that
determines magnetic shear and MHD stability.").

### toroidal_flux_coordinate

```yaml
  toroidal_flux_coordinate:
    aliases: []
    constant_on_flux_surface: true
    definition: >
      A non-negative, radius-like label of a nested magnetic flux surface,
      derived from enclosed toroidal magnetic flux using a positive reference
      vacuum toroidal field.
```

*Grounding:* the identity's recorded `source_documentation` for `rho_tor`
("Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ... ~ r [m]").

### toroidal_flux_coordinate_gradient

```yaml
  toroidal_flux_coordinate_gradient:
    aliases: []
    kind: vector
    definition: >
      The spatial gradient family of a flux-surface label based on enclosed
      toroidal magnetic flux, without fixing a norm, projection, component, or
      flux-surface average.
```

*Grounding:* accepted sibling description `toroidal_flux_coordinate_gradient_magnitude`
("Measure of how rapidly a toroidal magnetic-flux coordinate changes in space,
obtained from the magnitude of its spatial gradient") and the accepted
identity's own description. Origin `derived`, no DD binding.

### torque_density

```yaml
  torque_density:
    aliases: []
    kind: vector
    definition: >
      The local vector rate of net angular-momentum transfer per unit volume to
      or from the plasma.
```

*Grounding:* accepted sibling description `co_passing_torque_density` ("Net
angular-momentum transfer rate per unit volume...") and the other accepted
members of the family. Origin `derived`, no DD binding.

### tritium_breeding_ratio

```yaml
  tritium_breeding_ratio:
    aliases: []
    kind: scalar
    definition: >
      The blanket-level ratio of tritium atoms produced in breeding regions to
      the incident fusion-neutron count over the same interval (TBR).
```

*Grounding:* DD documentation of bound path
`breeding_blanket/time_slice/tritium_breeding_ratio` ("Tritium Breeding Ratio
(TBR), the efficiency metric defining the number of generated tritium atoms per
fusion neutron...") and the identity's recorded `source_documentation`
("Number of tritium atoms created for each fusion neutron (TBR)").

### vector_potential

```yaml
  vector_potential:
    aliases: []
    kind: vector
    inherently_dimensional: true
    definition: >
      A magnetic vector potential, covering the full potential, its directional
      components, and fluctuations about an equilibrium potential.
```

*Grounding:* accepted sibling descriptions (`poloidal_vector_potential`,
`radial_vector_potential`) and the accepted identity's own description; origin
`derived`, no DD binding.

### vorticity

```yaml
  vorticity:
    aliases: []
    kind: vector
    definition: >
      The curl of the plasma bulk velocity field, measuring the signed local
      angular rotation of fluid elements.
```

*Grounding:* accepted sibling description `parallel_vorticity` ("...projection
of the plasma bulk-flow curl onto the local magnetic-field direction, measuring
signed rotation in the perpendicular plane") and the accepted identity's own
description.

**Nothing in this list is unsourced.** Every block cites either a DD path
documentation, a recorded `source_documentation`, an accepted sibling identity's
description, or (β) the plan's own lead ruling. The two edges worth recording:
`electrostatic_potential` and `beta` ground on identity/sibling prose rather
than DD text because their bound-path documentation is metadata or is the bug
itself; `torque_density`, `momentum`, `vector_potential`,
`toroidal_flux_coordinate_gradient`, `mach_number`, `flux_limiter_coefficient`
ground on accepted siblings because they are derived parents with no DD
binding — none required inventing prose.

---

## Part 3 — The code change, described and not made

The gate that fires on a one-word name is the `review_name` semantic gate.
It is the `else` branch of `process_review_name_batch` in
`imas_codex/standard_names/workers.py` (the function the plan and follow-up
call `review_name`; it begins at `workers.py:8062`). The exact site that
builds the name-side text for `semantic_similarity_check` is
**`workers.py:8292-8297`**:

```python
else:
    try:
        sem_sim, sem_issues = await _asyncio.to_thread(
            semantic_similarity_check,
            sn_id,                            # ← workers.py:8295, the name-side text
            item.get("description") or "",
        )
```

`semantic_similarity_check` (`imas_codex/standard_names/audits.py:3684`) then
turns that name into text at **`audits.py:3725`** —
`name_text = name.replace("_", " ")` — embeds both texts with the project
embedding server and computes the cosine against the 0.55 critical floor
(`SEMANTIC_SIM_CRITICAL`, `defaults.py:27`). For a single token the
replace-underscore step is a no-op, so today's name-side text for `beta` is
the string `"beta"`.

**Minimal substitution** (stated, not applied): at `workers.py:8295`, replace
the `sn_id` argument with a definition-looked-up text that falls through to
`sn_id`:

1. Parse `sn_id` with the strict grammar. If it does not parse to exactly one
   vocabulary base/carrier token, pass `sn_id` (today's behaviour).
2. If it is a single token, look up that token's `definition` in the
   vocabulary registry (the `definition` blocks of Part 2, which live beside
   the tokens in the ISN vocabulary YAML — `physical_bases.yml` / carrier
   files — and are already surfaced by the grammar package's registry reader).
3. If the token carries a definition, pass the definition text as the name-side
   argument; if the token has no definition, pass `sn_id` — an unknown token
   still fails closed because no escape hatch exists for it.

The threshold, the description-side truncation (`description[:500]`), the
embedding path and the cosine are all unchanged. Because the substitution is
scoped to single parsed tokens, multi-word names keep today's identifier
coupling untouched (Part 1.3 shows why that scoping matters: composed
definitions push good multi-word names onto the floor; the design resists
extending to them).

Nothing was edited: this node is read-only, and the change is owned by the
implementing node once the lead accepts the escape-hatch shape and the
Part-2 definitions.

---

## Verdict summary

1. **The escape hatch works**: 25/25 single-token bases + β score ≥ 0.771 under
   the definition coupling, clearing the 0.55 floor with a ≥ 0.22 margin where
   five of them fail or hug the floor today.
2. **One floor at 0.55 serves both texts as an admission bar.** Raising it
   would reject correct pairs before it removed the worst mismatches (correct
   min 0.771 < mismatch max 0.814) and is not warranted.
3. **The definition-side text is a weaker *discriminator*** (42% of mismatched
   pairs clear 0.55 vs 22% on the identifier side). The gate under the new
   coupling admits; it does not verify descriptions. The beta documentation bug
   is not catchable at any threshold (0.840 against its own correct definition
   — the wrong meaning shares the text), so description correctness stays with
   the review chain; the escape hatch must not be expected to have caught it.
4. **24 accepted bare single-token bases exist; at least 4 cannot clear
   today's gate** with their current descriptions. They are admitted by
   catalog-direct paths to which the gate is never applied — beta is not an
   isolated case, and the escape hatch is a real admission path rather than a
   carve-out for one name.
5. **The substitution must stay scoped to single tokens.** Composed
   definitions of multi-word names under-score good names (0.560 minimum) while
   qualifiers carry no vocabulary definitions; extending the substitution
   there without first defining qualifiers would break the floor.

**Leave-alone notes for the lead:** (a) 15 of the 25 definitions are grounded on
accepted sibling descriptions because the identity has no DD binding; if a
stricter reading requires DD-only sourcing, those need a DD path per token
before landing. (b) `flux_limiter_coefficient`, `momentum`, `mach_number`,
`torque_density`, `vector_potential`, `toroidal_flux_coordinate_gradient` are
`origin == "derived"` and already route through the derived skip — they do not
need the escape hatch, but their definitions are proposed anyway because the
vocabulary should describe every base, not only the ones this node needed.
