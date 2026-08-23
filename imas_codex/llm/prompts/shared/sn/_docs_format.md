## Documentation Format — Canonical Structure

Every `documentation` field MUST be a **strict normative definition** and follow
this paragraph structure, separated by **blank lines** (literal `\n\n`):

**Independent content obligations (HARD).** The structure below is not a menu
or a priority order. Every applicable content element is mandatory on its own,
and satisfying one never compensates for omitting another. In particular,
symbol definitions, supported scope, supported exclusions or distinctions, and
supported essential relationships are four separate pass/fail obligations. If
several obligations are closely related, combine them in one clear paragraph;
never remove an obligation to shorten the documentation.

1. **Definition paragraph** — 1-3 sentences stating what the quantity *is* in physics terms. No equations. No diagnostic context. Just the concept.

2. **Governing equation paragraph** (when applicable) — introduce the principal defining equation as **display math** on its own line:

   ```
   The poloidal magnetic flux $\psi$ at a point on the symmetry axis is defined as the integral of the vertical magnetic field over a horizontal disk of radius $R$:

   $$
   \psi(R, Z) = \int_0^R B_Z(R', Z) \, 2\pi R' \, dR'
   $$

   where $B_Z$ is the vertical component of the magnetic field and the disk is centred on the toroidal symmetry axis at height $Z$.
   ```

   - Use `$$...$$` (display math) for the principal equation — MkDocs Material renders it centred on its own line.
   - The `$$` opening and closing delimiters must each be on their own line, with a blank line before and after the equation block.
   - Use `$...$` (inline math) only for variable names and small expressions in flowing prose.
   - **At most one display equation per documentation entry** — the *defining* one. Secondary relations stay inline.

3. **Symbol-definitions obligation** (whenever any symbol appears) — Define every symbol by its
   **identity** — the physical quantity it denotes —
   preferring an inline `[label](name:bare_id)` link when the symbol is itself a
   catalog quantity (e.g. `where $B_p$ is the [poloidal magnetic-field
   magnitude](name:poloidal_magnetic_field)`). An equation or relation with an
   undefined symbol fails this obligation even if every other content element
   is present. **Never state a unit in the documentation.** A unit describes a
   symbol's dimension, not which quantity it is; the quantity's unit is the
   authoritative structured `unit` field and a linked quantity carries its own
   unit via its link. Write the identity, never `where $B_p$ is in T` or `with
   unit $\mathrm{T}$`.

4. **Scope / distinction paragraph — scope obligation** — explicitly state
   each scope fact that the supplied name or context establishes: **(a)** the
   semantic boundary separating this quantity from a broader or narrower one,
   **(b)** the physical population or region to which it applies, and **(c)**
   its aggregation convention. Treat the name and injected context as positive
   evidence for these facts even when the source prose leaves them implicit.
   The scope obligation fails if any established boundary, population or
   region, or aggregation convention is left for the reader to infer. This
   content is semantic, not practical guidance; never invent an unsupported
   restriction.

5. **Exclusions / distinctions obligation** — explicitly state every exclusion
   or distinction supported by the supplied name and context that prevents the
   quantity being confused with a nearby quantity. A scope statement does not
   satisfy this obligation, and an exclusion must not be dropped to make room
   for a relationship or equation. It may share the scope paragraph when that
   is the clearest concise form.

6. **Essential-relationships paragraph** — explicitly state every essential
   semantic or mathematical relationship supported by the supplied context,
   such as a parent/component distinction, total, normalization, average,
   derivative, or integral. When the related standard name is available, link
   it inline and explain the relationship in prose; a bare link does not satisfy
   this requirement. This is not a measurement or computation recipe.

7. **Sign convention paragraph** (only for COCOS-dependent / signed quantities) — the LAST paragraph, in this exact format:

   ```
   Sign convention: Positive when <condition>.
   ```

   See PR-3 below for the strict format rules. **One paragraph**, blank line before, no blank line after (it ends the documentation).

### Sign-convention OMIT rule (STRICT)

A sign-convention paragraph is REQUIRED for quantities whose physical sign depends on a coordinate-system, orientation, or COCOS convention — currents, magnetic fluxes, voltages, electric fields, magnetic fields, velocities, torques, signed angles, etc.

A sign-convention paragraph is **FORBIDDEN** for sign-invariant quantities. The following inputs MUST produce documentation with **NO sign convention paragraph at all** — neither trivial nor non-trivial:

- `unit = "1"` AND no `cocos_label` → dimensionless shape/ratio/count → **omit**
- The quantity is intrinsically non-negative by physical definition: density, temperature, pressure magnitude, energy, power, count, area, volume, length, radius, distance, elongation, triangularity, squareness, frequency magnitude, lifetime, time interval, mass, opacity, transmissivity, emissivity, probability, fraction → **omit**
- The quantity is a magnitude of a vector or modulus of a complex number (e.g. `velocity_magnitude`, `|B|`) → **omit**

If you find yourself writing any of the following sign-convention phrasings, DELETE the paragraph entirely:

- ❌ "Positive when the [length / radius / height / width / depth / extent / value / population / count / number ...] is positive"
- ❌ "Positive when the quantity has a nonzero ..."
- ❌ "Positive when the [shape parameter / dimensionless ratio / fraction] is positive"
- ❌ "Positive by definition" / "Positive by construction" / "Inherently positive"

These are all tautological — they convey no information. The reader already knows magnitudes are positive. Adding a sign-convention paragraph for sign-invariant quantities pollutes the catalog with noise and is a documentation defect.

**Decision rule before emitting the sign-convention paragraph:**

> Ask: "Could a competent physicist set up a coordinate frame in which this quantity comes out NEGATIVE under the same physical situation?" If yes (e.g. you can reverse current direction or flip the Z axis), include the sign convention. If no (e.g. it is a length, a count, a density), OMIT it.

### Strict normative boundary

Canonical documentation defines the quantity. It is not a measurement guide,
simulation recipe, operational handbook, literature review, or collection of
representative machine values.

- Do **not** list diagnostics merely because they can estimate the quantity.
- Do **not** describe inference, reconstruction, power-balance, or simulation
  workflows merely because they are commonly used.
- Do **not** include typical device values, experiment ranges, machine names,
  performance records, or scenario estimates.
- Do **not** pad a rigorous definition with applications, significance, or
  loosely related physics.
- Measurement or computation may be stated **only when constitutive of the
  quantity's definition, or necessary to distinguish it from another physical
  quantity**. In that exceptional case state only the distinguishing semantic
  fact, not a diagnostic list or estimator recipe.

For example, documentation for `thermal_plasma_energy` may define the pressure
integral and its population scope, but must not list Thomson scattering, CXRS,
equilibrium reconstruction, diamagnetic loops, device ranges, or confinement
times. Documentation for `total_power_due_to_ion_cyclotron_heating` may define
the power absorbed by the plasma through ion-cyclotron-range waves and exclude
generator or launched power, but must not prescribe RF power-balance
measurements, transmission-loss estimates, wave simulations, modulation
experiments, or global energy-balance checks.

### Layout example (poloidal flux — corrected)

```
Poloidal magnetic flux at the plasma boundary is the magnetic flux through a horizontal disk centred on the toroidal symmetry axis at the height where the disk's edge lies on the last-closed flux surface. It labels the level sets of the equilibrium [poloidal_magnetic_flux](name:poloidal_magnetic_flux) function and defines the outer normalization reference for radial flux coordinates.

The normalized poloidal flux uses this boundary value as the outer endpoint:

$$
\psi_N = \frac{\psi - \psi_0}{\psi_b - \psi_0}
$$

where $\psi_0$ is the [magnetic axis](name:poloidal_magnetic_flux_at_magnetic_axis) value and $\psi_b$ is the boundary value.

Sign convention: Positive when $B_Z$ on the integration disk points in the $+Z$ direction.
```

### Anti-patterns

- ❌ One wall of prose with inline `$...$` equations crammed mid-sentence
- ❌ Display equations without blank lines around the `$$` delimiters
- ❌ Multiple display equations — only the *defining* one warrants display math
- ❌ Sign convention buried mid-paragraph or with extra header markup (`### Sign Convention`, `**Sign convention:**`)
- ❌ Diagnostic lists, estimator recipes, simulation workflows, or typical-value ranges
- ❌ Any unit anywhere in the prose — `(in Wb)`, `in T`, `with unit $\mathrm{m^{-3}}$`, or a `\mathrm{A\,m^{-2}}` unit expression in a where-clause. Units live in the structured `unit` field, never in the documentation. Define each symbol by identity (prefer a `name:` link) with no unit.

### When equations are not applicable

For quantities without a single defining equation (e.g. shape parameters, count fields, identifiers, integer indices), omit the governing-equation paragraph entirely. Definition → scope → exclusions/distinctions → essential relationships → sign convention (if applicable) is valid. Definition alone is acceptable for indices and metadata SNs only when the supplied context establishes no symbol, scope, exclusion, or essential relationship to carry.
