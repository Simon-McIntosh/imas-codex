# Rendered documentation prompts

Read-only reconstruction at graph state after the forced apply. The render uses the production `sn/generate_docs_user` template, `build_compose_context`, `conventions_context_for`, `locus_context_for`, `_enrich_for_docs_gen` (including `fetch_sibling_family`), and `_nearby_names_for_docs_gen`; no model was called. Historical pre-reset item fields and the apply-time review example set are not recoverable from this render.

## `flux_surface_averaged_electron_density_at_plasma_boundary`

**[TEMPLATE]** `Number density of {subject}` (selected density opening).
**[EQUATION]** `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`; `restates_equation=False`.
**[ANCHOR]** `flux_surface_averaged_xenon_density_at_plasma_boundary`; its accepted documentation is present in the sibling block (1438 characters).

~~~~text
# Generate documentation for: flux_surface_averaged_electron_density_at_plasma_boundary

This standard name has passed name review and is now accepted. Your task is to write
clear, complete `description` and `documentation` fields. You must NOT change the name,
kind, unit, or any other identity field.

**Standard name:** `flux_surface_averaged_electron_density_at_plasma_boundary`
**Unit:** m^-3
**Kind:** scalar
**Physics domain:** edge_plasma_physics


## Selected documentation conventions for this name

Use these entries from the conventions registry for this item.

### Concept: flux-surface average (`flux_surface_averaged`)
- Definition: The average of a quantity over a closed magnetic flux surface, weighted by the inverse magnitude of the poloidal flux gradient.
- Canonical equation: `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `f`: the averaged quantity
- `S`: the flux surface
- `\psi`: poloidal magnetic flux
- Forbidden variants: \$\$[^$]*(dl_p/B_p|dl/B)[^$]*\$\$


### Concept: plasma boundary (`plasma_boundary`)
- Definition: The surface used to delimit the plasma domain for the quantity being described.
- Canonical equation: `\psi = \psi_{\mathrm{LCFS}}`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `\psi`: poloidal magnetic flux
- `\psi_{\mathrm{LCFS}}`: its value on the last closed flux surface
- Forbidden variants: the separatrix


### Notation: `density`
- Canonical symbol: `n_s`
- Parameter `s`: species label, substituted with the species named in the entry (n_e for electrons)
- Forbidden spellings: N_s


### Notation: `electron`
- Canonical symbol: `e`


### Documentation opening: `density` with subject
- Description opening: Number density of {subject}
- Documentation opening: Number density of {subject}






## Sibling Family — parallel structure REQUIRED

`flux_surface_averaged_electron_density_at_plasma_boundary` belongs to a family of sibling standard names that share a common
pattern. Family members must read as a
**matched set**: the same documentation section structure, differing where the physics genuinely
differs (axis, species, zone, locus — and the symbols that go with them).


### Accepted sibling anchor: `flux_surface_averaged_xenon_density_at_plasma_boundary`

Compare this member's section ordering and physical scope.

- **Anchor description:** Flux-surface-averaged xenon ion number density over the last closed magnetic flux surface, summed over all xenon ionization states and excluding neutral xenon.
- **Anchor documentation:**

The flux-surface-averaged xenon density is the average of the local xenon ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_{\mathrm{Xe}} \rangle_S = \frac{\oint_S n_{\mathrm{Xe}}\,dS/|\nabla\psi|}{\oint_S dS/|\nabla\psi|}
$$

where $\langle n_{\mathrm{Xe}} \rangle_S$ is the averaged xenon ion number density, $n_{\mathrm{Xe}}$ is the local number density of charged xenon ions, $S$ is the last closed magnetic flux surface, $dS$ is its surface element, and $\psi$ is the poloidal magnetic flux; $|\nabla\psi|$ is the magnitude of its gradient.

This quantity applies to the xenon ion population on the plasma boundary and sums contributions from all xenon ionic charge states. Neutral xenon is excluded. The quantity is not a local point value, volume average, or density aggregated over different chemical elements.

It is the flux-surface-averaged counterpart of the local [xenon ion density at the plasma boundary](name:xenon_density_at_plasma_boundary): the parent quantity supplies the local boundary density, while this quantity applies the inverse-magnitude poloidal-flux-gradient average over the complete boundary surface. The same species-resolved construction is used for [flux-surface-averaged neon density at the plasma boundary](name:flux_surface_averaged_neon_density_at_plasma_boundary), with neon replacing xenon.



### Family members

- `flux_surface_averaged_argon_density_at_plasma_boundary` (axis: argon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of argon ions over the last closed magnetic flux surface, summed over all ionization states and excluding neutral argon.
  - documentation opens: "The flux-surface-averaged argon density is the average of the local argon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_beryllium_density_at_plasma_boundary` (axis: beryllium+plasma_boundary) [subject+position] — docs accepted
  - description: Charge-state-summed beryllium ion number density averaged over the last closed magnetic flux surface that bounds the confined plasma.
  - documentation opens: "The flux-surface-averaged beryllium density is the mean beryllium ion number density over the entire last closed magnetic flux surface forming the plasma boundary.

The flux-surface average is defined by

$$
\left\langle n_{\mathrm{Be}} …"

- `flux_surface_averaged_carbon_density_at_plasma_boundary` (axis: carbon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged carbon-ion number density over the last closed magnetic flux surface, summed across C+ through C6+ ionization states.
  - documentation opens: "The flux-surface-averaged carbon density is the average of the local carbon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_C …"

- `flux_surface_averaged_deuterium_density_at_plasma_boundary` (axis: deuterium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged deuterium ion number density on the last closed magnetic flux surface, summed over all ionic charge states and excluding neutral deuterium.
  - documentation opens: "The flux-surface-averaged deuterium density is the average local number density of deuterium ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_D …"

- `flux_surface_averaged_helium_3_density_at_plasma_boundary` (axis: helium_3+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged helium-3 ion number density on the last closed magnetic flux surface, summed over all helium-3 ionization states.
  - documentation opens: "The flux-surface-averaged helium-3 density is the average of the local helium-3 ion number density over the entire last closed magnetic flux surface that defines the plasma boundary.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_helium_4_density_at_plasma_boundary` (axis: helium_4+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of helium-4 ions on the last closed magnetic flux surface, summed over all ionic charge states.
  - documentation opens: "The flux-surface-averaged helium-4 density is the average number density of helium-4 ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_hydrogen_density_at_plasma_boundary` (axis: hydrogen+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged hydrogen ion number density on the last closed magnetic flux surface, summed over all hydrogen ion charge states.
  - documentation opens: "The flux-surface-averaged hydrogen density is the hydrogen ion number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_H \rangle_S = …"

- `flux_surface_averaged_ion_density_at_plasma_boundary` (axis: ion+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged charged-ion number density on the last closed magnetic flux surface, with charge states combined for each represented ion species.
  - documentation opens: "The flux-surface-averaged ion density is the charged-ion particle number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma. It represents the core–scrape-off-layer boundary population rather …"

- `flux_surface_averaged_iron_density_at_plasma_boundary` (axis: iron+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged iron-ion number density on the last closed magnetic flux surface, summed over all ionic charge states from Fe+ through fully stripped Fe26+.
  - documentation opens: "The flux-surface-averaged iron density at the plasma boundary is the average of the local iron impurity-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is …"

- `flux_surface_averaged_lithium_density_at_plasma_boundary` (axis: lithium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged lithium-ion number density on the last closed magnetic flux surface, summed over all charged lithium ion states rather than evaluated at one point.
  - documentation opens: "The flux-surface-averaged lithium density is the average lithium-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma and separates it from the open-field-line scrape-off layer.

The …"

- `flux_surface_averaged_neon_density_at_plasma_boundary` (axis: neon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of neon ions on the last closed magnetic flux surface, summed over all neon ionization stages.
  - documentation opens: "The flux-surface-averaged neon density is the average of the local neon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_nitrogen_density_at_plasma_boundary` (axis: nitrogen+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of nitrogen ions over the last closed magnetic flux surface, summed over all positive ionization states and excluding neutral nitrogen.
  - documentation opens: "Number density of nitrogen ions at the plasma boundary is represented here by its flux-surface average over the entire last closed magnetic flux surface enclosing the confined plasma.

The flux-surface average is defined by

$$
\langle n_N …"


**Family rules (enforced at review):**

1. **Vary only the physics that varies** — the axis/species/zone-specific
   token and genuinely member-specific physical claims.
2. **Never flatten real physics differences.** If this member's quantity is
   genuinely different in kind (e.g. a radial branch index vs an angular
   Fourier harmonic), state the distinction explicitly in the body.
3. **Cross-link the family.** Link at least one adjacent sibling inline where the prose supports it.


## Why this name was accepted (reviewer feedback)


- **Reviewer score:** 1.00


- **Reviewer comments:** [Primary] Grammar: clean round-trip with registered bare-prefix operator flux_surface_averaged, subject=electron, physical_base=density, position=plasma_boundary; no absorbed tokens or unregistered segments. Semantic: fully self-describing flux-surface-averaged electron number density at the plasma boundary (separatrix/LCFS); unit m^-3 and DD path summary/local/separatrix_average/n_e/value match exactly; consistent with accepted peers flux_surface_averaged_ion_density_at_plasma_boundary and electron_density_at_plasma_boundary. Convention: canonical operator_subject_base_at_position ordering, snake_case, no redundancy or model-specific tokens; mirrors the established same-family pattern. Completeness: averaging operator, species, quantity, and plasma-boundary locus are all present as required by the separatrix_average parent and sibling pattern.
[Secondary] Grammar is canonical and round-trips exactly, with the registered flux_surface_averaged operator, electron subject, density base, and plasma_boundary position. Semantically, the name precisely matches the authoritative electron-density source and its flux-surface-averaged separatrix locus; the m^-3 unit is correct. It is consistent with accepted siblings electron_density_at_plasma_boundary, flux_surface_averaged_ion_density_at_plasma_boundary, and ion_density_at_plasma_boundary while preserving the meaningful electron and averaging distinctions. Naming order, snake_case, and terminology are canonical, and all required quantity, subject, averaging, and location segments are present.






## Draft description (untrusted scaffolding — do not inherit)

Rewrite the description from the Physics Reference Material and the name itself; do not carry over any claim you cannot verify from the reference material.
A deterministic-parent placeholder is a lifecycle marker, not physics content;
never paraphrase it into the result.

Flux-surface-averaged electron number density on the last closed magnetic flux surface, representing the electron population at the confined-plasma boundary.










## Parent Standard Name

This is a component of `electron_density_at_plasma_boundary`.
- Parent description: Number density of electrons at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface.
- Parent documentation excerpt: Number density of electrons at the plasma boundary is the local, intensive particle count per physical volume for the electron population evaluated on the last closed magnetic flux surface. The plasma boundary is the limiting closed magnetic flux surface separating confined plasma from the open-fiel

Focus on what distinguishes this component specifically.
Cross-reference the parent: `[electron_density_at_plasma_boundary](name:electron_density_at_plasma_boundary)`.





## Base Quantity

This quantity is derived from `electron_density_at_plasma_boundary`.

Number density of electrons at the plasma boundary is the local, intensive particle count per physical volume for the electron population evaluated on the last closed magnetic flux surface. The plasma boundary is the limiting closed magnetic flux surface separating confined plasma from the open-fiel

Describe how this derivative/transformation relates to the base quantity.





## Source Context (PRIVATE — do NOT cite in output)

These source paths are provided for physics context ONLY. They help you understand
what this quantity represents. NEVER mention these paths, IDS names, or DD references
in the description or documentation — source provenance is tracked externally.
A generic leaf or parent label does not make the accepted identity generic; use
the enriched source meaning and the accepted name's explicit segments.

- `summary/local/separatrix_average/n_e/value`



## Enriched Source Descriptions (PRIMARY GROUNDING — PRIVATE, do NOT cite)

These entries are rich-first: each `documentation` value is the enriched source
description, with terse DD documentation used only when no enriched description
exists. Use this material to ground your documentation in correct physics; never
replace it with a generic interpretation of the path.
Extract the PHYSICS MEANING, not the source identity. NEVER copy path identifiers,
IDS names, or DD-specific language into the output text.

- `summary/local/separatrix_average/n_e/value` [m^-3]: Electron density (ne) at the separatrix, flux-surface averaged over the entire core-SOL boundary. Represents the local electron particle density at the plasma-edge interface.



## DD Path Lineage (PRIVATE — do NOT cite in output)

The concrete source leaf is often terse; the quantity's meaning and its
evaluation locus live on parent nodes. Use this lineage to ground the physics
and the correct evaluation location. NEVER cite these paths in the output.

- `summary/local/separatrix_average/n_e`: Electron density (ne) flux-surface averaged at the separatrix, representing the local electron particle density at the plasma-edge interface. Serves as a key boundary condition linking core and scrape-off layer transport.
- `summary/local/separatrix_average`: Flux surface averaged parameters at separatrix (flux-surface average over the entire core-SOL boundary separatrix)
- `summary/local`: Plasma parameter values at different locations



## Aliases (PRIVATE — do NOT cite in output)

value



## Nearest Peer Standard Names

Concept-similar names already in the catalog.
Use these for inline cross-references `[label](name:bare_id)` where naturally relevant.

- `name:electron_density_at_plasma_boundary` [m^-3, ]: Number density of electrons at the plasma boundary, evaluated as the local, intensive particle count per physical volume
- `name:density_at_plasma_boundary` [m^-3, ]: Particle number density of a selected plasma population evaluated at the plasma boundary, the last closed magnetic flux 
- `name:volume_averaged_electron_temperature` [eV, ]: Thermal energy of the bulk electron population expressed as an energy-equivalent temperature, averaged over the confined
- `name:boron_density_flux_surface_averaged_at_plasma_boundary` [m^-3, ]: Flux-surface-averaged number density of boron, including neutral atoms and all ionization states, evaluated on the plasm
- `name:neon_density_at_plasma_boundary` [m^-3, ]: Number density of neon ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summe



## Related Physics Quantities

Cross-domain related quantities sharing cluster membership, coordinates, or units.

- `core_instant_changes/change/profiles_1d/neutral/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total neutral density (thermal plus non-thermal) for a specific neutral charge state in core 1D profiles, representing the signed perturbation caused by a rapid event such as pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/electrons/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total electron density (ne, thermal plus non-thermal) as a 1D radial profile, representing the signed perturbation caused by a rapid plasma event such as a pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/ion/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in total ion state density (thermal + non-thermal) as a 1D radial profile, for a given ion charge state. Captures rapid density perturbations from events such as pellet injection or sawtooth crashes.
- `core_instant_changes/change/profiles_1d/ion/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in the fast (non-thermal) particle density (n_fast) for a specific ion charge state as a 1D radial profile, representing the signed perturbation before and after a rapid plasma event such as a pellet injection or MHD instability.
- `core_instant_changes/change/profiles_1d/neutral/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in fast (non-thermal) neutral particle density (n_fast) as a 1D radial profile, representing the signed difference caused by a rapid transient event such as pellet injection or MHD instability.





## DD Version History

Notable changes to this path across Data Dictionary versions:

- path_added (v3.37.0)



## Keywords

value, n_e



## Parent Structure

Electron density (ne) flux-surface averaged at the separatrix, representing the local electron particle density at the plasma-edge interface. Serves as a key boundary condition linking core and scrape-off layer transport.



## Nearby Existing Names (same physics domain)

For consistency, compare your documentation style and cross-references against these
accepted names in the same physics domain.

- **average_temperature_over_scrape_off_layer**: Averaged kinetic temperature of a specified plasma species over the open-field-line scrape-off layer, with the averaging domain and measure defined by its specialization. (scalar, eV)
- **beryllium_density_at_plasma_boundary**: Number density of beryllium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **boron_density**: Elemental boron number density, summed over neutral atoms and all ionic charge states through fully stripped B5+. (scalar, m^-3)
- **boron_density_at_plasma_boundary**: Number density of boron at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface, summed over neutral atoms and all ionization states. (scalar, m^-3)
- **boron_density_flux_surface_averaged_at_plasma_boundary**: Flux-surface-averaged number density of boron, including neutral atoms and all ionization states, evaluated on the plasma boundary. (scalar, m^-3)
- **carbon_density**: Carbon density is the local number density of elemental carbon in the plasma, summed over neutral carbon atoms and all ionic charge states through fully stripped C6+. (scalar, m^-3)
- **carbon_density_at_plasma_boundary**: Number density of carbon impurity ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **coulomb_logarithm_at_pedestal_top**: Dimensionless Coulomb logarithm at the H-mode pressure pedestal top, expressing the range of impact parameters contributing to cumulative small-angle Coulomb scattering. (scalar, 1)
- **decay_length_over_scrape_off_layer**: Characteristic outward distance over which a decaying plasma quantity decreases by one e-folding in the scrape-off layer. (scalar, m)
- **density_at_pedestal_maximum**: Density at pedestal maximum is the selected plasma-particle number density evaluated at the pedestal maximum, the steep mid-pedestal locus of maximum pedestal-gradient magnitude. It is a density value at that locus, not a density gradient. (scalar, m^-3)
- **density_at_plasma_boundary**: Particle number density of a selected plasma population evaluated at the plasma boundary, the last closed magnetic flux surface separating confined plasma from the scrape-off layer. (scalar, m^-3)
- **derivative_of_electron_pressure_at_pedestal_top_with_respect_to_normalized_poloidal_flux_coordinate**: Signed local derivative of fitted electron pressure with respect to the normalized poloidal flux coordinate at the H-mode pedestal top. It describes the core-side pedestal-top variation of the fitted electron-pressure function, not the pressure value or a maximum over a pedestal interval. (scalar, Pa)
- **deuterium_density_at_plasma_boundary**: Number density of deuterium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface, summed over all ionic charge states, and excluding neutral deuterium. (scalar, m^-3)
- **deuterium_tritium_density_at_plasma_boundary**: Number density of the effective deuterium-tritium fuel pseudo-species at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface. (scalar, m^-3)
- **diamagnetic_current_density_due_to_heat_viscosity**: Diamagnetic component of the electric current density generated by heat-viscosity terms in a collisional magnetized-plasma fluid closure, projected perpendicular to both the magnetic field and poloidal-flux gradient. (vector, A.m^-2)
- **diamagnetic_current_density_due_to_ion_neutral_friction**: Diamagnetic component of the electric current density driven by frictional momentum exchange between plasma ions and neutral particles in a partially ionized plasma, projected along the local unit vector parallel to the magnetic field crossed with the ion-pressure gradient. Positive values correspond to current in that direction. (vector, A.m^-2)
- **diamagnetic_momentum_flux**: Rate of transport of linear momentum per unit area across an oriented surface in the plasma, restricted to the diamagnetic contribution. (scalar, kg.m^-1.s^-2)
- **diamagnetic_momentum_flux_limiter_coefficient**: Dimensionless coefficient that scales the collisionless free-streaming upper bound on diamagnetic momentum-flux density in multi-fluid edge plasma transport closures, capping the flux magnitude while leaving its sign direction unchanged. (scalar, 1)
- **edge_localized_mode_frequency**: Rate of recurrent edge-localized mode events during an ELMing H-mode phase, expressing the cadence of pedestal-relaxation bursts rather than their amplitude or exhaust. (scalar, Hz)
- **effective_charge_at_separatrix**: Local effective ionic charge of the plasma mixture at the separatrix, reflecting charge-squared weighting across all contributing ion species and charge states. (scalar, 1)


**Authoritative cross-reference list:** The names listed above (plus those in "Nearest Peer
Standard Names" if present) are the **only** standard names you may link to using
`[label](name:bare_id)` inline links or `name:bare_id` entries in the `links` array.
Do NOT invent or guess other name IDs — links to non-existent names are rejected by
the validation pipeline and cause the entire batch item to fail.


## Output schema

Return a JSON object with exactly these two fields:

```json
{
  "description": "1 sentence preferred, 2 maximum, ≤250 chars, no LaTeX, American spelling",
  "documentation": "Strict normative markdown with defining $LaTeX$, scope, exclusions, and essential cross-references"
}
```

### description requirements
- 1 sentence strongly preferred, 2 max — ≤ 250 characters (NOT 500)
- First sentence = self-contained definition
- Add ONLY information beyond what the name tokens encode
- NO trailing clauses starting with "Representing", "Characterizing", "Quantifying"
- American spelling (ionization, behavior, center, etc.)
- No LaTeX; no inline units (unit is shown separately)
- No trailing "See also:" blocks
- **No storage-shape tags**: NEVER write "1D", "2D", "3D", "scalar", "array",
  "profile", or "time-dependent" — describe the *physics*, not the data layout

### documentation requirements
- Use only as many sentences as the rigorous definition requires
- Cover: physical meaning, the defining equation and symbols when applicable,
  scope/exclusions, essential semantic relationships, and necessary sign conventions
- Do not include generic diagnostics, estimator recipes, simulation workflows,
  typical machine values, experiment ranges, or padding
- Mention measurement/computation only when constitutive of the quantity or
  necessary to distinguish it from another quantity
- Cross-references to related standard names use `[label](name:bare_id)` inline links only
- Sign convention (if COCOS-dependent): use exactly `Sign convention: Positive when …` as a standalone paragraph (blank line before and after, plain text — no markdown headings, no bold); omit if sign-invariant
- American spelling throughout
- Minimum 20 characters
~~~~

## `flux_surface_averaged_neon_density_at_plasma_boundary`

**[TEMPLATE]** `Number density of {subject}` (selected density opening).
**[EQUATION]** `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`; `restates_equation=False`.
**[ANCHOR]** `flux_surface_averaged_xenon_density_at_plasma_boundary`; its accepted documentation is present in the sibling block (1438 characters).

~~~~text
# Generate documentation for: flux_surface_averaged_neon_density_at_plasma_boundary

This standard name has passed name review and is now accepted. Your task is to write
clear, complete `description` and `documentation` fields. You must NOT change the name,
kind, unit, or any other identity field.

**Standard name:** `flux_surface_averaged_neon_density_at_plasma_boundary`
**Unit:** m^-3
**Kind:** scalar
**Physics domain:** edge_plasma_physics


## Selected documentation conventions for this name

Use these entries from the conventions registry for this item.

### Concept: flux-surface average (`flux_surface_averaged`)
- Definition: The average of a quantity over a closed magnetic flux surface, weighted by the inverse magnitude of the poloidal flux gradient.
- Canonical equation: `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `f`: the averaged quantity
- `S`: the flux surface
- `\psi`: poloidal magnetic flux
- Forbidden variants: \$\$[^$]*(dl_p/B_p|dl/B)[^$]*\$\$


### Concept: plasma boundary (`plasma_boundary`)
- Definition: The surface used to delimit the plasma domain for the quantity being described.
- Canonical equation: `\psi = \psi_{\mathrm{LCFS}}`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `\psi`: poloidal magnetic flux
- `\psi_{\mathrm{LCFS}}`: its value on the last closed flux surface
- Forbidden variants: the separatrix


### Notation: `density`
- Canonical symbol: `n_s`
- Parameter `s`: species label, substituted with the species named in the entry (n_e for electrons)
- Forbidden spellings: N_s


### Documentation opening: `density` with subject
- Description opening: Number density of {subject}
- Documentation opening: Number density of {subject}






## Sibling Family — parallel structure REQUIRED

`flux_surface_averaged_neon_density_at_plasma_boundary` belongs to a family of sibling standard names that share a common
pattern. Family members must read as a
**matched set**: the same documentation section structure, differing where the physics genuinely
differs (axis, species, zone, locus — and the symbols that go with them).


### Accepted sibling anchor: `flux_surface_averaged_xenon_density_at_plasma_boundary`

Compare this member's section ordering and physical scope.

- **Anchor description:** Flux-surface-averaged xenon ion number density over the last closed magnetic flux surface, summed over all xenon ionization states and excluding neutral xenon.
- **Anchor documentation:**

The flux-surface-averaged xenon density is the average of the local xenon ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_{\mathrm{Xe}} \rangle_S = \frac{\oint_S n_{\mathrm{Xe}}\,dS/|\nabla\psi|}{\oint_S dS/|\nabla\psi|}
$$

where $\langle n_{\mathrm{Xe}} \rangle_S$ is the averaged xenon ion number density, $n_{\mathrm{Xe}}$ is the local number density of charged xenon ions, $S$ is the last closed magnetic flux surface, $dS$ is its surface element, and $\psi$ is the poloidal magnetic flux; $|\nabla\psi|$ is the magnitude of its gradient.

This quantity applies to the xenon ion population on the plasma boundary and sums contributions from all xenon ionic charge states. Neutral xenon is excluded. The quantity is not a local point value, volume average, or density aggregated over different chemical elements.

It is the flux-surface-averaged counterpart of the local [xenon ion density at the plasma boundary](name:xenon_density_at_plasma_boundary): the parent quantity supplies the local boundary density, while this quantity applies the inverse-magnitude poloidal-flux-gradient average over the complete boundary surface. The same species-resolved construction is used for [flux-surface-averaged neon density at the plasma boundary](name:flux_surface_averaged_neon_density_at_plasma_boundary), with neon replacing xenon.



### Family members

- `flux_surface_averaged_argon_density_at_plasma_boundary` (axis: argon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of argon ions over the last closed magnetic flux surface, summed over all ionization states and excluding neutral argon.
  - documentation opens: "The flux-surface-averaged argon density is the average of the local argon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_beryllium_density_at_plasma_boundary` (axis: beryllium+plasma_boundary) [subject+position] — docs accepted
  - description: Charge-state-summed beryllium ion number density averaged over the last closed magnetic flux surface that bounds the confined plasma.
  - documentation opens: "The flux-surface-averaged beryllium density is the mean beryllium ion number density over the entire last closed magnetic flux surface forming the plasma boundary.

The flux-surface average is defined by

$$
\left\langle n_{\mathrm{Be}} …"

- `flux_surface_averaged_carbon_density_at_plasma_boundary` (axis: carbon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged carbon-ion number density over the last closed magnetic flux surface, summed across C+ through C6+ ionization states.
  - documentation opens: "The flux-surface-averaged carbon density is the average of the local carbon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_C …"

- `flux_surface_averaged_deuterium_density_at_plasma_boundary` (axis: deuterium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged deuterium ion number density on the last closed magnetic flux surface, summed over all ionic charge states and excluding neutral deuterium.
  - documentation opens: "The flux-surface-averaged deuterium density is the average local number density of deuterium ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_D …"

- `flux_surface_averaged_electron_density_at_plasma_boundary` (axis: electron+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged electron number density on the last closed magnetic flux surface, representing the electron population at the confined-plasma boundary.
  - documentation opens: "The flux-surface-averaged electron density is the average of the local electron number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_e …"

- `flux_surface_averaged_helium_3_density_at_plasma_boundary` (axis: helium_3+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged helium-3 ion number density on the last closed magnetic flux surface, summed over all helium-3 ionization states.
  - documentation opens: "The flux-surface-averaged helium-3 density is the average of the local helium-3 ion number density over the entire last closed magnetic flux surface that defines the plasma boundary.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_helium_4_density_at_plasma_boundary` (axis: helium_4+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of helium-4 ions on the last closed magnetic flux surface, summed over all ionic charge states.
  - documentation opens: "The flux-surface-averaged helium-4 density is the average number density of helium-4 ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_hydrogen_density_at_plasma_boundary` (axis: hydrogen+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged hydrogen ion number density on the last closed magnetic flux surface, summed over all hydrogen ion charge states.
  - documentation opens: "The flux-surface-averaged hydrogen density is the hydrogen ion number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_H \rangle_S = …"

- `flux_surface_averaged_ion_density_at_plasma_boundary` (axis: ion+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged charged-ion number density on the last closed magnetic flux surface, with charge states combined for each represented ion species.
  - documentation opens: "The flux-surface-averaged ion density is the charged-ion particle number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma. It represents the core–scrape-off-layer boundary population rather …"

- `flux_surface_averaged_iron_density_at_plasma_boundary` (axis: iron+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged iron-ion number density on the last closed magnetic flux surface, summed over all ionic charge states from Fe+ through fully stripped Fe26+.
  - documentation opens: "The flux-surface-averaged iron density at the plasma boundary is the average of the local iron impurity-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is …"

- `flux_surface_averaged_lithium_density_at_plasma_boundary` (axis: lithium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged lithium-ion number density on the last closed magnetic flux surface, summed over all charged lithium ion states rather than evaluated at one point.
  - documentation opens: "The flux-surface-averaged lithium density is the average lithium-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma and separates it from the open-field-line scrape-off layer.

The …"

- `flux_surface_averaged_nitrogen_density_at_plasma_boundary` (axis: nitrogen+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of nitrogen ions over the last closed magnetic flux surface, summed over all positive ionization states and excluding neutral nitrogen.
  - documentation opens: "Number density of nitrogen ions at the plasma boundary is represented here by its flux-surface average over the entire last closed magnetic flux surface enclosing the confined plasma.

The flux-surface average is defined by

$$
\langle n_N …"


**Family rules (enforced at review):**

1. **Vary only the physics that varies** — the axis/species/zone-specific
   token and genuinely member-specific physical claims.
2. **Never flatten real physics differences.** If this member's quantity is
   genuinely different in kind (e.g. a radial branch index vs an angular
   Fourier harmonic), state the distinction explicitly in the body.
3. **Cross-link the family.** Link at least one adjacent sibling inline where the prose supports it.


## Why this name was accepted (reviewer feedback)


- **Reviewer score:** 0.99


- **Reviewer comments:** [Primary] Grammar: All tokens are registered — `flux_surface_averaged` (transformation), `neon` (subject), `density` (physical_base), `plasma_boundary` (position), with correct `_at_` locus preposition. Round-trip parse/compose is clean. No decomposition defects. Semantic: The name is fully self-descriptive — a domain expert immediately understands this as the flux-surface-averaged number density of neon ions evaluated at the plasma boundary (separatrix). Unit m^-3 matches density. The vector neighbours establish an exact pattern: `flux_surface_averaged_deuterium_density_at_plasma_boundary`, `flux_surface_averaged_nitrogen_density_at_plasma_boundary`, `flux_surface_averaged_helium_4_density_at_plasma_boundary`, `flux_surface_averaged_xenon_density_at_plasma_boundary` all follow the identical decomposition with only the subject varying. The sibling `neon_density_at_plasma_boundary` (sim=0.96) exists without the flux-surface-averaged transformation, confirming this name fills the averaged variant without redundancy. The same_base neighbours (`argon_density`, `argon_density_at_limiter`, `argon_density_at_magnetic_axis`) confirm `density` as the correct physical_base for number density of impurity species. Convention: Perfect snake_case, canonical segment ordering (transformation → subject → physical_base → _at_ → position), matching the established sibling pattern exactly. No abbreviations, no model identifiers, no clashes. Completeness: All physically relevant segments present — transformation (flux_surface_averaged), subject (neon), physical_base (density), position (plasma_boundary). Scalar kind and m^-3 unit are consistent. Physics domain edge_plasma_physics is appropriate for a boundary quantity.
[Secondary] **Grammar (20/20):** The name parses cleanly using registered tokens. Decomposition: `flux_surface_averaged` = transformation token ✓, `neon` = subject token ✓, `density` = physical_base ✓, `_at_` = locus preposition ✓, `plasma_boundary` = position token ✓. All tokens are in their closed registries. Round-trip parse/compose works. No decomposition defects — no closed-vocab tokens absorbed into `physical_base`. The `flux_surface_averaged_<species>_density_at_plasma_boundary` pattern is canonical and directly mirrors the accepted siblings `flux_surface_averaged_deuterium_density_at_plasma_boundary`, `flux_surface_averaged_nitrogen_density_at_plasma_boundary`, `flux_surface_averaged_helium_4_density_at_plasma_boundary`, and `flux_surface_averaged_xenon_density_at_plasma_boundary`.

**Semantic (19/20):** The name is fully self-descriptive. A domain expert reading only the name understands it as the flux-surface-averaged number density of neon ions at the plasma boundary (separatrix). The unit m^-3 is consistent with number density. The vector neighbours (sim 0.95-0.96) confirm this exact pattern is well-represented in the catalogue for neon, deuterium, nitrogen, helium-4, and xenon. The `neon_density_at_plasma_boundary` sibling (sim=0.96) lacks the flux_surface_averaged transformation, so this candidate is a distinct and more specific quantity. Minor docking because the description mentions 'summed over all ionization stages' which isn't explicit in the name — but this is standard catalog convention for impurity species (matches all listed siblings).

**Convention (20/20):** Perfect snake_case, canonical segment ordering (transformation → subject → physical_base → _at_ → position). Matches the sibling pattern established by the flux_surface_averaged_*_density_at_plasma_boundary family exactly. No abbreviations, no model-specific identifiers, no redundancy, no clashes.

**Completeness (20/20):** All physically relevant segments present: transformation (flux_surface_averaged), subject (neon), physical_base (density), and locus (plasma_boundary). Scalar kind is appropriate. Matches the sibling pattern perfectly.






## Draft description (untrusted scaffolding — do not inherit)

Rewrite the description from the Physics Reference Material and the name itself; do not carry over any claim you cannot verify from the reference material.
A deterministic-parent placeholder is a lifecycle marker, not physics content;
never paraphrase it into the result.

Flux-surface-averaged number density of neon ions on the last closed magnetic flux surface, summed over all neon ionization stages.








## Parent Standard Name

This is a component of `neon_density_at_plasma_boundary`.
- Parent description: Number density of neon ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization stages.
- Parent documentation excerpt: Number density of neon ions at the plasma boundary is the local particle count per physical volume for the neon-ion population evaluated on the last closed magnetic flux surface. The population is summed over all ionization stages from singly ionized neon, Ne$^+$, through fully stripped neon, Ne$^{1

Focus on what distinguishes this component specifically.
Cross-reference the parent: `[neon_density_at_plasma_boundary](name:neon_density_at_plasma_boundary)`.





## Base Quantity

This quantity is derived from `neon_density_at_plasma_boundary`.

Number density of neon ions at the plasma boundary is the local particle count per physical volume for the neon-ion population evaluated on the last closed magnetic flux surface. The population is summed over all ionization stages from singly ionized neon, Ne$^+$, through fully stripped neon, Ne$^{1

Describe how this derivative/transformation relates to the base quantity.





## Source Context (PRIVATE — do NOT cite in output)

These source paths are provided for physics context ONLY. They help you understand
what this quantity represents. NEVER mention these paths, IDS names, or DD references
in the description or documentation — source provenance is tracked externally.
A generic leaf or parent label does not make the accepted identity generic; use
the enriched source meaning and the accepted name's explicit segments.

- `summary/local/separatrix_average/n_i/neon/value`



## Enriched Source Descriptions (PRIMARY GROUNDING — PRIVATE, do NOT cite)

These entries are rich-first: each `documentation` value is the enriched source
description, with terse DD documentation used only when no enriched description
exists. Use this material to ground your documentation in correct physics; never
replace it with a generic interpretation of the path.
Extract the PHYSICS MEANING, not the source identity. NEVER copy path identifiers,
IDS names, or DD-specific language into the output text.

- `summary/local/separatrix_average/n_i/neon/value` [m^-3]: Neon (Ne).



## DD Path Lineage (PRIVATE — do NOT cite in output)

The concrete source leaf is often terse; the quantity's meaning and its
evaluation locus live on parent nodes. Use this lineage to ground the physics
and the correct evaluation location. NEVER cite these paths in the output.

- `summary/local/separatrix_average/n_i/neon`: Neon (Ne)
- `summary/local/separatrix_average/n_i`: Ion density (ni) per species, flux-surface averaged at the separatrix, with entries for individual ion species. Represents the particle density at the core-SOL boundary and serves as a key input for edge plasma and divertor modelling.
- `summary/local/separatrix_average`: Flux surface averaged parameters at separatrix (flux-surface average over the entire core-SOL boundary separatrix)
- `summary/local`: Plasma parameter values at different locations



## Aliases (PRIVATE — do NOT cite in output)

value



## Nearest Peer Standard Names

Concept-similar names already in the catalog.
Use these for inline cross-references `[label](name:bare_id)` where naturally relevant.

- `name:neon_density_at_plasma_boundary` [m^-3, ]: Number density of neon ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summe
- `name:flux_surface_averaged_toroidal_neon_velocity_at_plasma_boundary` [m.s^-1, ]: Flux-surface average of the toroidal neon-ion bulk velocity on the last closed magnetic flux surface, with all ionizatio
- `name:tungsten_density_flux_surface_averaged_at_plasma_boundary` [m^-3, ]: Flux-surface-averaged number density of positively charged tungsten ions on the last closed magnetic flux surface, summe
- `name:tungsten_density_at_separatrix` [m^-3, ]: Tungsten-ion number density on the last closed magnetic flux surface, summed over all positive ionization states and dis
- `name:neon_density_at_internal_transport_barrier` [m^-3, ]: Particle number density of neon ions evaluated on the flux surface assigned to an internal transport barrier, summed ove



## Related Physics Quantities

Cross-domain related quantities sharing cluster membership, coordinates, or units.

- `core_instant_changes/change/profiles_1d/neutral/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total neutral density (thermal plus non-thermal) for a specific neutral charge state in core 1D profiles, representing the signed perturbation caused by a rapid event such as pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/electrons/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total electron density (ne, thermal plus non-thermal) as a 1D radial profile, representing the signed perturbation caused by a rapid plasma event such as a pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/ion/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in total ion state density (thermal + non-thermal) as a 1D radial profile, for a given ion charge state. Captures rapid density perturbations from events such as pellet injection or sawtooth crashes.
- `core_instant_changes/change/profiles_1d/ion/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in the fast (non-thermal) particle density (n_fast) for a specific ion charge state as a 1D radial profile, representing the signed perturbation before and after a rapid plasma event such as a pellet injection or MHD instability.
- `core_instant_changes/change/profiles_1d/neutral/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in fast (non-thermal) neutral particle density (n_fast) as a 1D radial profile, representing the signed difference caused by a rapid transient event such as pellet injection or MHD instability.





## DD Version History

Notable changes to this path across Data Dictionary versions:

- units (v4.1.0)
- path_added (v3.37.0)



## Keywords

value, neon





## Nearby Existing Names (same physics domain)

For consistency, compare your documentation style and cross-references against these
accepted names in the same physics domain.

- **average_temperature_over_scrape_off_layer**: Averaged kinetic temperature of a specified plasma species over the open-field-line scrape-off layer, with the averaging domain and measure defined by its specialization. (scalar, eV)
- **beryllium_density_at_plasma_boundary**: Number density of beryllium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **boron_density**: Elemental boron number density, summed over neutral atoms and all ionic charge states through fully stripped B5+. (scalar, m^-3)
- **boron_density_at_plasma_boundary**: Number density of boron at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface, summed over neutral atoms and all ionization states. (scalar, m^-3)
- **boron_density_flux_surface_averaged_at_plasma_boundary**: Flux-surface-averaged number density of boron, including neutral atoms and all ionization states, evaluated on the plasma boundary. (scalar, m^-3)
- **carbon_density**: Carbon density is the local number density of elemental carbon in the plasma, summed over neutral carbon atoms and all ionic charge states through fully stripped C6+. (scalar, m^-3)
- **carbon_density_at_plasma_boundary**: Number density of carbon impurity ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **coulomb_logarithm_at_pedestal_top**: Dimensionless Coulomb logarithm at the H-mode pressure pedestal top, expressing the range of impact parameters contributing to cumulative small-angle Coulomb scattering. (scalar, 1)
- **decay_length_over_scrape_off_layer**: Characteristic outward distance over which a decaying plasma quantity decreases by one e-folding in the scrape-off layer. (scalar, m)
- **density_at_pedestal_maximum**: Density at pedestal maximum is the selected plasma-particle number density evaluated at the pedestal maximum, the steep mid-pedestal locus of maximum pedestal-gradient magnitude. It is a density value at that locus, not a density gradient. (scalar, m^-3)
- **density_at_plasma_boundary**: Particle number density of a selected plasma population evaluated at the plasma boundary, the last closed magnetic flux surface separating confined plasma from the scrape-off layer. (scalar, m^-3)
- **derivative_of_electron_pressure_at_pedestal_top_with_respect_to_normalized_poloidal_flux_coordinate**: Signed local derivative of fitted electron pressure with respect to the normalized poloidal flux coordinate at the H-mode pedestal top. It describes the core-side pedestal-top variation of the fitted electron-pressure function, not the pressure value or a maximum over a pedestal interval. (scalar, Pa)
- **deuterium_density_at_plasma_boundary**: Number density of deuterium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface, summed over all ionic charge states, and excluding neutral deuterium. (scalar, m^-3)
- **deuterium_tritium_density_at_plasma_boundary**: Number density of the effective deuterium-tritium fuel pseudo-species at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface. (scalar, m^-3)
- **diamagnetic_current_density_due_to_heat_viscosity**: Diamagnetic component of the electric current density generated by heat-viscosity terms in a collisional magnetized-plasma fluid closure, projected perpendicular to both the magnetic field and poloidal-flux gradient. (vector, A.m^-2)
- **diamagnetic_current_density_due_to_ion_neutral_friction**: Diamagnetic component of the electric current density driven by frictional momentum exchange between plasma ions and neutral particles in a partially ionized plasma, projected along the local unit vector parallel to the magnetic field crossed with the ion-pressure gradient. Positive values correspond to current in that direction. (vector, A.m^-2)
- **diamagnetic_momentum_flux**: Rate of transport of linear momentum per unit area across an oriented surface in the plasma, restricted to the diamagnetic contribution. (scalar, kg.m^-1.s^-2)
- **diamagnetic_momentum_flux_limiter_coefficient**: Dimensionless coefficient that scales the collisionless free-streaming upper bound on diamagnetic momentum-flux density in multi-fluid edge plasma transport closures, capping the flux magnitude while leaving its sign direction unchanged. (scalar, 1)
- **edge_localized_mode_frequency**: Rate of recurrent edge-localized mode events during an ELMing H-mode phase, expressing the cadence of pedestal-relaxation bursts rather than their amplitude or exhaust. (scalar, Hz)
- **effective_charge_at_separatrix**: Local effective ionic charge of the plasma mixture at the separatrix, reflecting charge-squared weighting across all contributing ion species and charge states. (scalar, 1)


**Authoritative cross-reference list:** The names listed above (plus those in "Nearest Peer
Standard Names" if present) are the **only** standard names you may link to using
`[label](name:bare_id)` inline links or `name:bare_id` entries in the `links` array.
Do NOT invent or guess other name IDs — links to non-existent names are rejected by
the validation pipeline and cause the entire batch item to fail.


## Output schema

Return a JSON object with exactly these two fields:

```json
{
  "description": "1 sentence preferred, 2 maximum, ≤250 chars, no LaTeX, American spelling",
  "documentation": "Strict normative markdown with defining $LaTeX$, scope, exclusions, and essential cross-references"
}
```

### description requirements
- 1 sentence strongly preferred, 2 max — ≤ 250 characters (NOT 500)
- First sentence = self-contained definition
- Add ONLY information beyond what the name tokens encode
- NO trailing clauses starting with "Representing", "Characterizing", "Quantifying"
- American spelling (ionization, behavior, center, etc.)
- No LaTeX; no inline units (unit is shown separately)
- No trailing "See also:" blocks
- **No storage-shape tags**: NEVER write "1D", "2D", "3D", "scalar", "array",
  "profile", or "time-dependent" — describe the *physics*, not the data layout

### documentation requirements
- Use only as many sentences as the rigorous definition requires
- Cover: physical meaning, the defining equation and symbols when applicable,
  scope/exclusions, essential semantic relationships, and necessary sign conventions
- Do not include generic diagnostics, estimator recipes, simulation workflows,
  typical machine values, experiment ranges, or padding
- Mention measurement/computation only when constitutive of the quantity or
  necessary to distinguish it from another quantity
- Cross-references to related standard names use `[label](name:bare_id)` inline links only
- Sign convention (if COCOS-dependent): use exactly `Sign convention: Positive when …` as a standalone paragraph (blank line before and after, plain text — no markdown headings, no bold); omit if sign-invariant
- American spelling throughout
- Minimum 20 characters
~~~~

## `flux_surface_averaged_nitrogen_density_at_plasma_boundary`

**[TEMPLATE]** `Number density of {subject}` (selected density opening).
**[EQUATION]** `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`; `restates_equation=False`.
**[ANCHOR]** `flux_surface_averaged_xenon_density_at_plasma_boundary`; its accepted documentation is present in the sibling block (1438 characters).

~~~~text
# Generate documentation for: flux_surface_averaged_nitrogen_density_at_plasma_boundary

This standard name has passed name review and is now accepted. Your task is to write
clear, complete `description` and `documentation` fields. You must NOT change the name,
kind, unit, or any other identity field.

**Standard name:** `flux_surface_averaged_nitrogen_density_at_plasma_boundary`
**Unit:** m^-3
**Kind:** scalar
**Physics domain:** edge_plasma_physics


## Selected documentation conventions for this name

Use these entries from the conventions registry for this item.

### Concept: flux-surface average (`flux_surface_averaged`)
- Definition: The average of a quantity over a closed magnetic flux surface, weighted by the inverse magnitude of the poloidal flux gradient.
- Canonical equation: `\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `f`: the averaged quantity
- `S`: the flux surface
- `\psi`: poloidal magnetic flux
- Forbidden variants: \$\$[^$]*(dl_p/B_p|dl/B)[^$]*\$\$


### Concept: plasma boundary (`plasma_boundary`)
- Definition: The surface used to delimit the plasma domain for the quantity being described.
- Canonical equation: `\psi = \psi_{\mathrm{LCFS}}`
- Equation placement: name the concept; do not restate its equation unless needed to define this quantity
- `\psi`: poloidal magnetic flux
- `\psi_{\mathrm{LCFS}}`: its value on the last closed flux surface
- Forbidden variants: the separatrix


### Notation: `density`
- Canonical symbol: `n_s`
- Parameter `s`: species label, substituted with the species named in the entry (n_e for electrons)
- Forbidden spellings: N_s


### Documentation opening: `density` with subject
- Description opening: Number density of {subject}
- Documentation opening: Number density of {subject}






## Sibling Family — parallel structure REQUIRED

`flux_surface_averaged_nitrogen_density_at_plasma_boundary` belongs to a family of sibling standard names that share a common
pattern. Family members must read as a
**matched set**: the same documentation section structure, differing where the physics genuinely
differs (axis, species, zone, locus — and the symbols that go with them).


### Accepted sibling anchor: `flux_surface_averaged_xenon_density_at_plasma_boundary`

Compare this member's section ordering and physical scope.

- **Anchor description:** Flux-surface-averaged xenon ion number density over the last closed magnetic flux surface, summed over all xenon ionization states and excluding neutral xenon.
- **Anchor documentation:**

The flux-surface-averaged xenon density is the average of the local xenon ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_{\mathrm{Xe}} \rangle_S = \frac{\oint_S n_{\mathrm{Xe}}\,dS/|\nabla\psi|}{\oint_S dS/|\nabla\psi|}
$$

where $\langle n_{\mathrm{Xe}} \rangle_S$ is the averaged xenon ion number density, $n_{\mathrm{Xe}}$ is the local number density of charged xenon ions, $S$ is the last closed magnetic flux surface, $dS$ is its surface element, and $\psi$ is the poloidal magnetic flux; $|\nabla\psi|$ is the magnitude of its gradient.

This quantity applies to the xenon ion population on the plasma boundary and sums contributions from all xenon ionic charge states. Neutral xenon is excluded. The quantity is not a local point value, volume average, or density aggregated over different chemical elements.

It is the flux-surface-averaged counterpart of the local [xenon ion density at the plasma boundary](name:xenon_density_at_plasma_boundary): the parent quantity supplies the local boundary density, while this quantity applies the inverse-magnitude poloidal-flux-gradient average over the complete boundary surface. The same species-resolved construction is used for [flux-surface-averaged neon density at the plasma boundary](name:flux_surface_averaged_neon_density_at_plasma_boundary), with neon replacing xenon.



### Family members

- `flux_surface_averaged_argon_density_at_plasma_boundary` (axis: argon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of argon ions over the last closed magnetic flux surface, summed over all ionization states and excluding neutral argon.
  - documentation opens: "The flux-surface-averaged argon density is the average of the local argon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_beryllium_density_at_plasma_boundary` (axis: beryllium+plasma_boundary) [subject+position] — docs accepted
  - description: Charge-state-summed beryllium ion number density averaged over the last closed magnetic flux surface that bounds the confined plasma.
  - documentation opens: "The flux-surface-averaged beryllium density is the mean beryllium ion number density over the entire last closed magnetic flux surface forming the plasma boundary.

The flux-surface average is defined by

$$
\left\langle n_{\mathrm{Be}} …"

- `flux_surface_averaged_carbon_density_at_plasma_boundary` (axis: carbon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged carbon-ion number density over the last closed magnetic flux surface, summed across C+ through C6+ ionization states.
  - documentation opens: "The flux-surface-averaged carbon density is the average of the local carbon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_C …"

- `flux_surface_averaged_deuterium_density_at_plasma_boundary` (axis: deuterium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged deuterium ion number density on the last closed magnetic flux surface, summed over all ionic charge states and excluding neutral deuterium.
  - documentation opens: "The flux-surface-averaged deuterium density is the average local number density of deuterium ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_D …"

- `flux_surface_averaged_electron_density_at_plasma_boundary` (axis: electron+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged electron number density on the last closed magnetic flux surface, representing the electron population at the confined-plasma boundary.
  - documentation opens: "The flux-surface-averaged electron density is the average of the local electron number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_e …"

- `flux_surface_averaged_helium_3_density_at_plasma_boundary` (axis: helium_3+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged helium-3 ion number density on the last closed magnetic flux surface, summed over all helium-3 ionization states.
  - documentation opens: "The flux-surface-averaged helium-3 density is the average of the local helium-3 ion number density over the entire last closed magnetic flux surface that defines the plasma boundary.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_helium_4_density_at_plasma_boundary` (axis: helium_4+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of helium-4 ions on the last closed magnetic flux surface, summed over all ionic charge states.
  - documentation opens: "The flux-surface-averaged helium-4 density is the average number density of helium-4 ions over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"

- `flux_surface_averaged_hydrogen_density_at_plasma_boundary` (axis: hydrogen+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged hydrogen ion number density on the last closed magnetic flux surface, summed over all hydrogen ion charge states.
  - documentation opens: "The flux-surface-averaged hydrogen density is the hydrogen ion number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle n_H \rangle_S = …"

- `flux_surface_averaged_ion_density_at_plasma_boundary` (axis: ion+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged charged-ion number density on the last closed magnetic flux surface, with charge states combined for each represented ion species.
  - documentation opens: "The flux-surface-averaged ion density is the charged-ion particle number density averaged over the entire last closed magnetic flux surface that bounds the confined plasma. It represents the core–scrape-off-layer boundary population rather …"

- `flux_surface_averaged_iron_density_at_plasma_boundary` (axis: iron+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged iron-ion number density on the last closed magnetic flux surface, summed over all ionic charge states from Fe+ through fully stripped Fe26+.
  - documentation opens: "The flux-surface-averaged iron density at the plasma boundary is the average of the local iron impurity-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is …"

- `flux_surface_averaged_lithium_density_at_plasma_boundary` (axis: lithium+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged lithium-ion number density on the last closed magnetic flux surface, summed over all charged lithium ion states rather than evaluated at one point.
  - documentation opens: "The flux-surface-averaged lithium density is the average lithium-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma and separates it from the open-field-line scrape-off layer.

The …"

- `flux_surface_averaged_neon_density_at_plasma_boundary` (axis: neon+plasma_boundary) [subject+position] — docs accepted
  - description: Flux-surface-averaged number density of neon ions on the last closed magnetic flux surface, summed over all neon ionization stages.
  - documentation opens: "The flux-surface-averaged neon density is the average of the local neon-ion number density over the entire last closed magnetic flux surface that bounds the confined plasma.

The flux-surface average is defined by

$$
\langle …"


**Family rules (enforced at review):**

1. **Vary only the physics that varies** — the axis/species/zone-specific
   token and genuinely member-specific physical claims.
2. **Never flatten real physics differences.** If this member's quantity is
   genuinely different in kind (e.g. a radial branch index vs an angular
   Fourier harmonic), state the distinction explicitly in the body.
3. **Cross-link the family.** Link at least one adjacent sibling inline where the prose supports it.


## Why this name was accepted (reviewer feedback)


- **Reviewer score:** 0.97


- **Reviewer comments:** [Primary] Grammar: The name parses cleanly using registered tokens. Decomposition: `flux_surface_averaged` = transformation ✓, `nitrogen` = subject ✓, `density` = physical_base ✓, `_at_` = locus preposition ✓, `plasma_boundary` = position ✓. All tokens are in their closed registries. The round-trip parse/compose works with no decomposition defects. The `flux_surface_averaged_<subject>_density_at_<position>` pattern is canonical and matches the established sibling pattern exactly. Semantic: Fully self-descriptive — a domain expert reading only the name immediately understands this is the flux-surface-averaged number density of nitrogen ions at the plasma boundary (separatrix). The unit m^-3 is consistent with number density. Physics domain `edge_plasma_physics` fits perfectly — the plasma boundary/separatrix is a canonical edge feature. The vector neighbours establish this exact pattern: `flux_surface_averaged_neon_density_at_plasma_boundary` (sim=0.98), `flux_surface_averaged_deuterium_density_at_plasma_boundary` (sim=0.96), `flux_surface_averaged_helium_4_density_at_plasma_boundary` (sim=0.95), `flux_surface_averaged_iron_density_at_plasma_boundary` (sim=0.95) — the candidate simply substitutes `nitrogen` for the species, which is the correct and expected variation. No near-duplicate issues. The same_base_neighbours (`argon_density`, `argon_density_at_limiter`, `argon_density_at_magnetic_axis`) confirm that `density` is the correct physical_base for species number density with the `_at_<position>` locus pattern. Convention: Perfect snake_case, canonical segment ordering (transformation → subject → physical_base → locus → position). No abbreviations, no model-specific identifiers, no redundancy. Matches the sibling pattern exactly. Completeness: All physically relevant segments present — transformation (flux_surface_averaged), subject (nitrogen), physical_base (density), locus (plasma_boundary). Scalar kind is appropriate. The name is complete and unambiguous.
[Secondary] **Grammar (20/20):** All tokens are registered: transformation=`flux_surface_averaged`, subject=`nitrogen`, physical_base=`density`, position=`plasma_boundary`. Round-trip is clean. No decomposition defects — no closed-vocab tokens absorbed into `physical_base`. The canonical transformation-prefix → subject → base → locus-at ordering is preserved.

**Semantic (19/20):** Fully self-descriptive: flux-surface-averaged nitrogen number density at the plasma boundary (separatrix). Unit m^-3 matches density. The DD provenance and physics domain (edge_plasma_physics) are consistent. The vector neighbours `flux_surface_averaged_neon_density_at_plasma_boundary`, `flux_surface_averaged_deuterium_density_at_plasma_boundary`, `flux_surface_averaged_helium_4_density_at_plasma_boundary`, and `flux_surface_averaged_iron_density_at_plasma_boundary` establish the exact pattern — this candidate perfectly mirrors them, substituting nitrogen. Near-duplicate is not a defect here; it's consistency with established convention. Minor dock only because nitrogen (without explicit charge-state or isotope qualifier) implicitly sums over ionization stages, as documented in the `argon_density` siblings — this is the convention but not entirely self-evident from the name alone.

**Convention (19/20):** Perfect snake_case, canonical segment ordering matching the sibling pattern. No model-specific identifiers, no abbreviations. Matches `flux_surface_averaged_neon_density_at_plasma_boundary` (sim=0.98) and `flux_surface_averaged_iron_density_at_plasma_boundary` (sim=0.95) exactly. No clashes.

**Completeness (18/20):** All relevant segments present: transformation, subject, physical_base, locus. Consistent with `same_base_neighbours` (argon_density_at_magnetic_axis etc.) which similarly don't specify charge states. The candidate is fully specified.






## Draft description (untrusted scaffolding — do not inherit)

Rewrite the description from the Physics Reference Material and the name itself; do not carry over any claim you cannot verify from the reference material.
A deterministic-parent placeholder is a lifecycle marker, not physics content;
never paraphrase it into the result.

Flux-surface-averaged number density of nitrogen ions over the last closed magnetic flux surface, summed over all positive ionization states and excluding neutral nitrogen.








## Parent Standard Name

This is a component of `nitrogen_density_at_plasma_boundary`.
- Parent description: Number density of nitrogen ions at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface.
- Parent documentation excerpt: Number density of nitrogen ions at the plasma boundary is the local, intensive particle count per physical volume for the nitrogen ion population evaluated on the last closed magnetic flux surface. The plasma boundary is the limiting closed magnetic flux surface separating confined plasma from the o

Focus on what distinguishes this component specifically.
Cross-reference the parent: `[nitrogen_density_at_plasma_boundary](name:nitrogen_density_at_plasma_boundary)`.





## Base Quantity

This quantity is derived from `nitrogen_density_at_plasma_boundary`.

Number density of nitrogen ions at the plasma boundary is the local, intensive particle count per physical volume for the nitrogen ion population evaluated on the last closed magnetic flux surface. The plasma boundary is the limiting closed magnetic flux surface separating confined plasma from the o

Describe how this derivative/transformation relates to the base quantity.





## Source Context (PRIVATE — do NOT cite in output)

These source paths are provided for physics context ONLY. They help you understand
what this quantity represents. NEVER mention these paths, IDS names, or DD references
in the description or documentation — source provenance is tracked externally.
A generic leaf or parent label does not make the accepted identity generic; use
the enriched source meaning and the accepted name's explicit segments.

- `summary/local/separatrix_average/n_i/nitrogen/value`



## Enriched Source Descriptions (PRIMARY GROUNDING — PRIVATE, do NOT cite)

These entries are rich-first: each `documentation` value is the enriched source
description, with terse DD documentation used only when no enriched description
exists. Use this material to ground your documentation in correct physics; never
replace it with a generic interpretation of the path.
Extract the PHYSICS MEANING, not the source identity. NEVER copy path identifiers,
IDS names, or DD-specific language into the output text.

- `summary/local/separatrix_average/n_i/nitrogen/value` [m^-3]: Nitrogen (N).



## DD Path Lineage (PRIVATE — do NOT cite in output)

The concrete source leaf is often terse; the quantity's meaning and its
evaluation locus live on parent nodes. Use this lineage to ground the physics
and the correct evaluation location. NEVER cite these paths in the output.

- `summary/local/separatrix_average/n_i/nitrogen`: Nitrogen (N)
- `summary/local/separatrix_average/n_i`: Ion density (ni) per species, flux-surface averaged at the separatrix, with entries for individual ion species. Represents the particle density at the core-SOL boundary and serves as a key input for edge plasma and divertor modelling.
- `summary/local/separatrix_average`: Flux surface averaged parameters at separatrix (flux-surface average over the entire core-SOL boundary separatrix)
- `summary/local`: Plasma parameter values at different locations



## Aliases (PRIVATE — do NOT cite in output)

value



## Nearest Peer Standard Names

Concept-similar names already in the catalog.
Use these for inline cross-references `[label](name:bare_id)` where naturally relevant.

- `name:nitrogen_density_at_plasma_boundary` [m^-3, ]: Number density of nitrogen ions at the plasma boundary, evaluated as the local, intensive particle count per physical vo
- `name:tungsten_density_flux_surface_averaged_at_plasma_boundary` [m^-3, ]: Flux-surface-averaged number density of positively charged tungsten ions on the last closed magnetic flux surface, summe
- `name:neon_density_at_plasma_boundary` [m^-3, ]: Number density of neon ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summe
- `name:tungsten_density_at_separatrix` [m^-3, ]: Tungsten-ion number density on the last closed magnetic flux surface, summed over all positive ionization states and dis
- `name:flux_surface_averaged_toroidal_nitrogen_velocity_at_plasma_boundary` [m.s^-1, ]: Flux-surface-averaged toroidal velocity of nitrogen ions at the plasma boundary (separatrix), with charge states combine



## Related Physics Quantities

Cross-domain related quantities sharing cluster membership, coordinates, or units.

- `core_instant_changes/change/profiles_1d/neutral/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total neutral density (thermal plus non-thermal) for a specific neutral charge state in core 1D profiles, representing the signed perturbation caused by a rapid event such as pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/electrons/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in total electron density (ne, thermal plus non-thermal) as a 1D radial profile, representing the signed perturbation caused by a rapid plasma event such as a pellet injection or MHD crash.
- `core_instant_changes/change/profiles_1d/ion/state/density` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in total ion state density (thermal + non-thermal) as a 1D radial profile, for a given ion charge state. Captures rapid density perturbations from events such as pellet injection or sawtooth crashes.
- `core_instant_changes/change/profiles_1d/ion/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change (delta) in the fast (non-thermal) particle density (n_fast) for a specific ion charge state as a 1D radial profile, representing the signed perturbation before and after a rapid plasma event such as a pellet injection or MHD instability.
- `core_instant_changes/change/profiles_1d/neutral/state/density_fast` (core_instant_changes) — unit via m^-3 [transport]: Instantaneous change in fast (non-thermal) neutral particle density (n_fast) as a 1D radial profile, representing the signed difference caused by a rapid transient event such as pellet injection or MHD instability.





## DD Version History

Notable changes to this path across Data Dictionary versions:

- units (v4.1.0)
- path_added (v3.37.0)



## Keywords

value, nitrogen





## Nearby Existing Names (same physics domain)

For consistency, compare your documentation style and cross-references against these
accepted names in the same physics domain.

- **average_temperature_over_scrape_off_layer**: Averaged kinetic temperature of a specified plasma species over the open-field-line scrape-off layer, with the averaging domain and measure defined by its specialization. (scalar, eV)
- **beryllium_density_at_plasma_boundary**: Number density of beryllium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **boron_density**: Elemental boron number density, summed over neutral atoms and all ionic charge states through fully stripped B5+. (scalar, m^-3)
- **boron_density_at_plasma_boundary**: Number density of boron at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface, summed over neutral atoms and all ionization states. (scalar, m^-3)
- **boron_density_flux_surface_averaged_at_plasma_boundary**: Flux-surface-averaged number density of boron, including neutral atoms and all ionization states, evaluated on the plasma boundary. (scalar, m^-3)
- **carbon_density**: Carbon density is the local number density of elemental carbon in the plasma, summed over neutral carbon atoms and all ionic charge states through fully stripped C6+. (scalar, m^-3)
- **carbon_density_at_plasma_boundary**: Number density of carbon impurity ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface and summed over all ionization states. (scalar, m^-3)
- **coulomb_logarithm_at_pedestal_top**: Dimensionless Coulomb logarithm at the H-mode pressure pedestal top, expressing the range of impact parameters contributing to cumulative small-angle Coulomb scattering. (scalar, 1)
- **decay_length_over_scrape_off_layer**: Characteristic outward distance over which a decaying plasma quantity decreases by one e-folding in the scrape-off layer. (scalar, m)
- **density_at_pedestal_maximum**: Density at pedestal maximum is the selected plasma-particle number density evaluated at the pedestal maximum, the steep mid-pedestal locus of maximum pedestal-gradient magnitude. It is a density value at that locus, not a density gradient. (scalar, m^-3)
- **density_at_plasma_boundary**: Particle number density of a selected plasma population evaluated at the plasma boundary, the last closed magnetic flux surface separating confined plasma from the scrape-off layer. (scalar, m^-3)
- **derivative_of_electron_pressure_at_pedestal_top_with_respect_to_normalized_poloidal_flux_coordinate**: Signed local derivative of fitted electron pressure with respect to the normalized poloidal flux coordinate at the H-mode pedestal top. It describes the core-side pedestal-top variation of the fitted electron-pressure function, not the pressure value or a maximum over a pedestal interval. (scalar, Pa)
- **deuterium_density_at_plasma_boundary**: Number density of deuterium ions at the plasma boundary, evaluated locally on the last closed magnetic flux surface, summed over all ionic charge states, and excluding neutral deuterium. (scalar, m^-3)
- **deuterium_tritium_density_at_plasma_boundary**: Number density of the effective deuterium-tritium fuel pseudo-species at the plasma boundary, evaluated as the local, intensive particle count per physical volume on the last closed magnetic flux surface. (scalar, m^-3)
- **diamagnetic_current_density_due_to_heat_viscosity**: Diamagnetic component of the electric current density generated by heat-viscosity terms in a collisional magnetized-plasma fluid closure, projected perpendicular to both the magnetic field and poloidal-flux gradient. (vector, A.m^-2)
- **diamagnetic_current_density_due_to_ion_neutral_friction**: Diamagnetic component of the electric current density driven by frictional momentum exchange between plasma ions and neutral particles in a partially ionized plasma, projected along the local unit vector parallel to the magnetic field crossed with the ion-pressure gradient. Positive values correspond to current in that direction. (vector, A.m^-2)
- **diamagnetic_momentum_flux**: Rate of transport of linear momentum per unit area across an oriented surface in the plasma, restricted to the diamagnetic contribution. (scalar, kg.m^-1.s^-2)
- **diamagnetic_momentum_flux_limiter_coefficient**: Dimensionless coefficient that scales the collisionless free-streaming upper bound on diamagnetic momentum-flux density in multi-fluid edge plasma transport closures, capping the flux magnitude while leaving its sign direction unchanged. (scalar, 1)
- **edge_localized_mode_frequency**: Rate of recurrent edge-localized mode events during an ELMing H-mode phase, expressing the cadence of pedestal-relaxation bursts rather than their amplitude or exhaust. (scalar, Hz)
- **effective_charge_at_separatrix**: Local effective ionic charge of the plasma mixture at the separatrix, reflecting charge-squared weighting across all contributing ion species and charge states. (scalar, 1)


**Authoritative cross-reference list:** The names listed above (plus those in "Nearest Peer
Standard Names" if present) are the **only** standard names you may link to using
`[label](name:bare_id)` inline links or `name:bare_id` entries in the `links` array.
Do NOT invent or guess other name IDs — links to non-existent names are rejected by
the validation pipeline and cause the entire batch item to fail.


## Output schema

Return a JSON object with exactly these two fields:

```json
{
  "description": "1 sentence preferred, 2 maximum, ≤250 chars, no LaTeX, American spelling",
  "documentation": "Strict normative markdown with defining $LaTeX$, scope, exclusions, and essential cross-references"
}
```

### description requirements
- 1 sentence strongly preferred, 2 max — ≤ 250 characters (NOT 500)
- First sentence = self-contained definition
- Add ONLY information beyond what the name tokens encode
- NO trailing clauses starting with "Representing", "Characterizing", "Quantifying"
- American spelling (ionization, behavior, center, etc.)
- No LaTeX; no inline units (unit is shown separately)
- No trailing "See also:" blocks
- **No storage-shape tags**: NEVER write "1D", "2D", "3D", "scalar", "array",
  "profile", or "time-dependent" — describe the *physics*, not the data layout

### documentation requirements
- Use only as many sentences as the rigorous definition requires
- Cover: physical meaning, the defining equation and symbols when applicable,
  scope/exclusions, essential semantic relationships, and necessary sign conventions
- Do not include generic diagnostics, estimator recipes, simulation workflows,
  typical machine values, experiment ranges, or padding
- Mention measurement/computation only when constitutive of the quantity or
  necessary to distinguish it from another quantity
- Cross-references to related standard names use `[label](name:bare_id)` inline links only
- Sign convention (if COCOS-dependent): use exactly `Sign convention: Positive when …` as a standalone paragraph (blank line before and after, plain text — no markdown headings, no bold); omit if sign-invariant
- American spelling throughout
- Minimum 20 characters
~~~~
