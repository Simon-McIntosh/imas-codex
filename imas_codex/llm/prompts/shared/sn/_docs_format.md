## Documentation Format — Canonical Structure

Every `documentation` field MUST follow this paragraph structure, separated by **blank lines** (literal `\n\n`):

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
   - Define every symbol introduced. Define units explicitly only when they are not already on the SN's `unit` field (e.g. inside an equation context: `where $T_e$ is in eV`).
   - **At most one display equation per documentation entry** — the *defining* one. Secondary relations stay inline.

3. **Measurement / computation paragraph** (when applicable) — how the quantity is measured or computed in practice (diagnostics, reconstruction methods). Keep it general — do NOT name specific codes (no "EFIT", "LIUQE", "JINTRAC") unless the SN itself encodes that context.

4. **Typical values paragraph** (when applicable) — representative ranges with units, distinguishing regimes if relevant. Use the SN's canonical unit. Format ranges as `1-10 keV`, `0.1-1 m^-3 \times 10^{20}`, etc.

5. **Sign convention paragraph** (only for COCOS-dependent / signed quantities) — the LAST paragraph, in this exact format:

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

### Layout example (poloidal flux — corrected)

```
Poloidal magnetic flux at the plasma boundary is the magnetic flux through a horizontal disk centred on the toroidal symmetry axis at the height where the disk's edge lies on the last-closed flux surface. It labels the level sets of the equilibrium [poloidal_magnetic_flux](name:poloidal_magnetic_flux) function and defines the outer normalization reference for radial flux coordinates.

The normalized poloidal flux uses this boundary value as the outer endpoint:

$$
\psi_N = \frac{\psi - \psi_0}{\psi_b - \psi_0}
$$

where $\psi_0$ is the [magnetic axis](name:poloidal_magnetic_flux_at_magnetic_axis) value and $\psi_b$ is the boundary value.

The boundary value is recovered from equilibrium reconstruction by integrating $B_Z$ over a horizontal disk whose edge is tracked to the outermost closed flux contour.

Typical absolute magnitudes range from 1-20 Wb in medium-sized tokamaks, with the exact value depending on plasma current, toroidal field, and shape.

Sign convention: Positive when $B_Z$ on the integration disk points in the $+Z$ direction.
```

### Anti-patterns

- ❌ One wall of prose with inline `$...$` equations crammed mid-sentence
- ❌ Display equations without blank lines around the `$$` delimiters
- ❌ Multiple display equations — only the *defining* one warrants display math
- ❌ Sign convention buried mid-paragraph or with extra header markup (`### Sign Convention`, `**Sign convention:**`)
- ❌ Inline unit decorations like `(in Wb)` outside the three narrow exceptions (typical values, equation variable defs, conversions — see PR-3)

### When equations are not applicable

For quantities without a single defining equation (e.g. shape parameters, count fields, identifiers, integer indices), omit the governing-equation paragraph entirely. Definition → Measurement → Typical values → Sign convention (if applicable) is a valid 4-paragraph form. Definition alone is acceptable for indices and metadata SNs.
