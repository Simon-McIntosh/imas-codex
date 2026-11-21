## Curated Enrichment Exemplars

These paired exemplars teach by contrast. Study the *reasons* — the weak
variants look plausible until you see the canonical form side by side. All
exemplars use American spelling (NC-17) in every prose field.

### Positive exemplars — imitate these patterns

#### E1. Physical-base scalar with governing equation

- **Name:** `electron_temperature` (unit `eV`)
- ✅ *Description:* `Thermal energy of the bulk electron population expressed as a temperature via $T_e = 2 \langle E_e \rangle / (3 k_B)$.`
- ✅ *Documentation excerpt:* `The bulk electron temperature quantifies the second
  moment of the electron velocity distribution under the Maxwellian assumption.
  For an isotropic population it is defined by $3 k_B T_e / 2 =
  \langle E_e\rangle$, where $k_B$ is Boltzmann's constant and
  $\langle E_e\rangle$ is the mean random kinetic energy per electron. It
  relates to [electron pressure](name:electron_pressure) through the equation
  of state.`
- *Why good:* one-sentence description states the physical meaning, not a
  tokenization of the name. Documentation gives the defining physics, defines
  every symbol, links an essential related quantity, and contains no practical
  method or typical-value appendix.

#### E2. Vector component with COCOS sign convention

- **Name:** `toroidal_magnetic_field` (unit `T`)
- ✅ *Description:* `Projection of the plasma magnetic-field vector onto the geometric toroidal direction $\hat{\phi}$.`
- ✅ *Documentation:* `The toroidal component $B_\phi = \mathbf{B} \cdot \hat{\phi}$
  is the projection of magnetic field $\mathbf{B}$ onto the toroidal unit
  vector $\hat{\phi}$. Together with the
  [poloidal magnetic field](name:poloidal_magnetic_field), it determines the
  magnetic-field vector.

  Sign convention: Positive when $B_\phi$ points in the direction of increasing toroidal angle $\phi$.`
- *Why good:* the definition is exact, symbols are defined, the relationship is
  essential, and the sign convention is stated without measurement or device context.

#### E3. Constraint / measurement-weight path

- **Name:** `flux_loop_radial_magnetization_constraint_weight` (unit `1`)
- ✅ *Description:* `Relative weight applied to the radial magnetization signal from flux loops in the least-squares equilibrium cost functional.`
- ✅ *Documentation:* `Standard $\chi^2$ weight controlling the relative
  importance of the radial-magnetization constraint in its objective function.
  A larger value increases that constraint's relative contribution; the exact
  normalization is part of the objective-function convention.`
- *Why good:* uses the **one-line χ² reference** for the generic inverse-problem
  role (never re-derives it); focuses on WHAT the weight does, not the
  underlying physics of flux loops (already captured in the base name).

#### E4. Spectrum quantity — closure integral explicit

- **Name:** `toroidal_mode_spectrum_of_electron_temperature_perturbation`
  (unit `eV`)
- ✅ *Description:* `Fourier-mode amplitude of the electron-temperature perturbation along the toroidal coordinate.`
- ✅ *Documentation:* `Amplitude $|\tilde{T}_e(n_\phi)|$ of the complex Fourier
  decomposition of $\delta T_e$ with respect to the toroidal angle $\phi$.
  Integrating $|\tilde{T}_e|^2$ over toroidal mode number $n_\phi$ recovers the
  total perturbation variance under the stated Fourier normalization.`
- *Why good:* spelled-out closure integral (SR-spectrum rule — required when
  name ends in `_spectrum`); mode-number symbol defined; unit `eV` matches
  the amplitude (not a density — note: `_density_spectrum` would require a
  `eV/n_\phi` unit with the Parseval integral over $n_\phi$).

#### E5. Flux-surface quantity with a defining relation

- **Name:** `safety_factor` (unit `1`)
- ✅ *Description:* `Number of toroidal transits a field line makes per single poloidal circuit of a given flux surface.`
- ✅ *Documentation:* `The safety factor $q(\psi) = d\Phi_{\rm tor} / d\Phi_{\rm pol}$
  is the differential ratio of toroidal magnetic flux $\Phi_{\rm tor}$ to
  poloidal magnetic flux $\Phi_{\rm pol}$ and quantifies field-line helicity on
  a flux surface. Its radial variation determines
  [magnetic shear](name:magnetic_shear).`
- *Why good:* defines $q$ symbolically in terms of defined quantities; gives
  its scope and an essential relationship without adding operational or scenario context.

#### E6. Geometric property — concise, no DD leak

- **Name:** `radial_coordinate_of_magnetic_axis` (unit `m`)
- ✅ *Description:* `Radial (R) coordinate of the magnetic axis, i.e. the extremum of the poloidal magnetic flux surface.`
- ✅ *Documentation:* `The major radius $R_{\rm axis}$ of the magnetic axis is
  the $R$ coordinate at which $\nabla \psi = 0$ inside the plasma, identifying
  the innermost nested flux surface. Together with the
  [vertical coordinate of the magnetic axis](name:vertical_coordinate_of_magnetic_axis),
  it locates the magnetic axis in the cylindrical $R$-$Z$ plane.`
- *Why good:* Definition anchored to a mathematical condition ($\nabla \psi = 0$);
  pairs the coordinate with its Z-counterpart and contains no machine-specific
  values, code names, or reconstruction recipe.

### Anti-patterns — never emit these

#### AE1. Tautological description

- **Name:** `electron_temperature`
- ❌ `The temperature of the electrons.`
- *Why bad:* the description does not add information beyond the name tokens.
- *Fix:* describe the **physical meaning** (see E1 above — moment of the
  velocity distribution, units expressed via Boltzmann's constant).

#### AE2. DD-path leakage

- ❌ `Quantity stored at core_profiles/profiles_1d/electrons/temperature.`
- *Why bad:* standard names are IDS-agnostic. Referring to DD structure couples
  the reader to the source rather than the physics.
- *Fix:* describe the physics and its measurement; never mention IDS or DD
  paths anywhere in description or documentation.

#### AE3. Undefined LaTeX symbols

- ❌ `The safety factor $q = B_\phi R / (r B_\theta)$ ...` (no definition of
  $r$ or $B_\theta$)
- *Why bad:* introducing symbols without definition turns the documentation
  into shorthand; readers must consult other sources to parse it.
- *Fix:* `... where $r$ is the minor radius and $B_\theta$ is the poloidal
  magnetic-field magnitude on the given flux surface.`

#### AE4. Bracketed placeholder sign-convention

- ❌ `Positive when [condition].` or `Positive when the quantity is in the
  [direction] direction under COCOS-[N].`
- *Why bad:* the sanitizer will strip the entire sentence; the documentation
  loses the sign-convention payload required for vector and flux quantities.
- *Fix:* write the concrete condition — e.g. `Positive when $B_\phi$ is in
  the $+\phi$ direction under COCOS-11.` If the quantity is sign-invariant,
  omit the sentence entirely.

#### AE5. British spelling in prose

- ❌ `The normalised poloidal flux coordinate, used to parametrise flux
  surfaces ...`
- *Why bad:* the catalog is American-English only. British spellings become
  silent synonyms that break grep-based search and downstream consumers.
- *Fix:* `The normalized poloidal flux coordinate, used to parameterize flux
  surfaces ...`. Apply the same rule to `ionization`, `polarized`, `center`,
  `behavior`, `modeled`, `analyzed`, `fueling`, `color`, `fiber`, `meter`.
  SI unit symbols (`m`, `kg`, `V`) are unaffected.

#### AE6. Boilerplate inverse-problem derivation

- **Name:** `iron_core_segment_radial_magnetization_constraint_weight`
- ❌ `The iron core segment radial magnetization constraint weight is a
  parameter used in an inverse problem, where the forward model is ... and
  the cost function is $\chi^2 = \sum_i (y_i - F_i(x))^2 / \sigma_i^2 / w_i$.
  By adjusting this weight, the reconstruction emphasises the measurement
  more or less relative to other constraints ...`
- *Why bad:* every constraint-weight name gets the same two-paragraph
  derivation; the text is boilerplate, not physics, and bloats the catalog.
- *Fix:* single-line reference: `Standard $\chi^2$ weight controlling the
  relative importance of the iron-core segment radial magnetization
  measurement in the equilibrium reconstruction.` Link once to a canonical
  inverse-problem documentation target.

#### AE7. Documentation contradicts the name

- **Name:** `normal_magnetic_field` (claims scalar, unit `T`)
- ❌ *Doc says:* `The Fourier coefficients of the normal component of the
  magnetic field, expanded in poloidal mode number $m$ ...`
- *Why bad:* the name promises a scalar field-magnitude; the documentation
  describes a spectral coefficient. They disagree — either the name or the
  description is wrong.
- *Fix:* rename to `fourier_coefficient_of_normal_magnetic_field`
  (with unit `T` per-harmonic) **or** rewrite the documentation to describe
  the scalar field proper without Fourier language.

#### AE8. Practical-method appendix attached to a rigorous definition

- **Name:** `thermal_plasma_energy`
- ❌ `In practice, the quantity is inferred by integrating profiles derived
  from Thomson scattering, charge-exchange spectroscopy, and equilibrium
  reconstruction. Diamagnetic loops provide an independent estimate; typical
  values range from ...`
- *Why bad:* the diagnostic list, estimator comparison, and device range do
  not define thermal plasma energy and can introduce false equivalences.
- *Fix:* define the thermal-population pressure integral, its included
  populations, and exclusions only.

- **Name:** `total_power_due_to_ion_cyclotron_heating`
- ❌ `In practice, this quantity is inferred from forward, reflected, and
  coupled power together with transmission-line, matching-network, and antenna
  loss estimates; it may also be constrained by simulations or modulation.`
- *Why bad:* this is an estimator recipe, not a canonical definition, and it
  risks conflating plasma-absorbed power with generator or launched power.
- *Fix:* define total power absorbed by the plasma through ion-cyclotron-range
  waves and state the exclusion of generator and launched power.

#### AE9. Excessive / wrong cross-links

- ❌ `links: ["name:plasma_current", "name:plasma", "name:current",
  "https://en.wikipedia.org/wiki/Safety_factor_(plasma_physics)"]`
- *Why bad:* bare nouns like `plasma` and `current` are not valid SN ids;
  Wikipedia URLs are external noise; `plasma_current` is only weakly
  related to `safety_factor`.
- *Fix:* 2–6 bare-ID links to genuinely related **existing** SNs:
  `["poloidal_magnetic_flux", "toroidal_magnetic_flux", "magnetic_shear",
   "radial_coordinate_of_magnetic_axis"]` (verify each exists in the catalog).

#### AE10. Generic filler prose

- ❌ `This is an important quantity in fusion research. It plays a
  significant role in plasma behaviour and is widely used ...`
- *Why bad:* zero information content; any three sentences must carry
  normative physics or relational substance.
- *Fix:* every sentence earns its place with one of (definition, defining
  equation/symbols, scope/exclusion, essential relationship, sign convention).

### Pre-emit checklist for every enriched entry

1. Does the description add information beyond the name tokens?
2. Are all LaTeX symbols defined on first use, with units?
3. American spelling throughout every prose field?
4. For COCOS-dependent quantities: does the documentation contain a
   `Sign convention: Positive when ...` statement as a standalone paragraph
   (no markdown headings, no bold, blank lines before/after)?
5. For spectrum quantities: is the closure integral / Parseval relation
   spelled out with the correct integration variable?
6. For constraint weights / measurement times: is the generic inverse-problem
   role referenced in one line rather than re-derived?
7. Are `links` using the `name:` prefix (e.g. `name:electron_temperature`)
   and pointing to existing names that genuinely enrich understanding
   (typically 2–6)?
8. Is the `description` ≤ 2 concise sentences, and is the documentation no
   longer than its rigorous definition requires?
9. Does the documentation avoid any mention of DD paths, IDS names, or
    structural database prefixes?
10. Does it exclude generic diagnostics, estimator/simulation recipes,
    typical machine or experiment values, practical advice, and padding?
