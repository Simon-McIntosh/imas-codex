# Dimensional defining-equation gate calibration

## Verdict

**SYMBOL-BINDING-GAP dominates.** Of the 38 curated-catalog rows marked `fail`, 32 (84.2%) contain a dimensionally valid relation whose symbols the gate does not bind completely or correctly, 5 (13.2%) are parser/selection limitations, and 1 (2.6%) contains a genuine dimensional physics error. Thus 37/38 catalog failures are properties of the gate apparatus on this corpus, not demonstrated physics defects in the documentation.

This is a calibration verdict, not an acceptance threshold. The holdout is path-level: its 85 rows contain repeated documentation for 13 catalog identities. The 38 failures collapse to eight distinct documentation families, so row counts measure the effect on the immutable evaluation population rather than 38 independent prose judgments.

## Recorded measure

The catalog arm scored all 85 immutable `catalog_documentation` strings through the merged dimensional `defining_equation` gate, with each row's pinned `dd_path` and `declared_unit`. It made no model calls. The named catalog-arm harness check passed.

| Arm | pass | fail | not_evaluable | total |
|---|---:|---:|---:|---:|
| Curated catalog | 47 | 38 | 0 | **85** |
| Generated arm, already measured | 47 | 38 | 0 | **85** |

The equal marginals do **not** establish parity: catalog failures are concentrated in eight repeated prose families and are overwhelmingly symbol-binding limitations. A pairwise generated-versus-catalog row comparison was not part of this node's evidence contract.

Recorded sources:

- Catalog distribution and one JSON record per row, including the exact display relation(s), declared unit, gate reason, extracted subject, symbol bindings and computed sides: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T200950056752-n-dimcalibrate/catalog-dimensional-outcomes.log`.
- Verbatim catalog documentation grouped over all eight failed families: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T200950056752-n-dimcalibrate/failed-family-documents.log`.
- Generated-arm distribution: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T163849987100-n-dimgate/holdout-dimension-outcomes.log`.
- Named harness check: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T200950056752-n-dimcalibrate/catalog-arm-check.log` (`1 passed`, exit 0).

## Failure-family census

The classification is of why the new gate reports `fail`. `SYMBOL-BINDING-GAP` means the relation is dimensionally sound under the units stated or semantically identified in the same document, but the binding layer does not recover those units. `PARSER-LIMITATION` means the one-display-equality selector or conservative expression grammar cannot select the target relation. `GENUINE-PHYSICS-ERROR` is reserved for an actual dimensional contradiction in the curated prose.

| Catalog identity | Failed path rows | Classification | Offending relation quoted verbatim | Declared unit quoted verbatim | Adjudication |
|---|---:|---|---|---|---|
| `area_of_flux_loop` | 1 | `GENUINE-PHYSICS-ERROR` | `A_{\text{eff}} = N \times A_{\text{turn}}`; `\Delta \Psi = \frac{1}{A_{\text{eff}}} \int V \, dt` | `m^2` | The first relation correctly defines area. The second states `\Delta \Psi` in Wb but divides the voltage-time integral (already Wb) by area, yielding T. The document therefore contains a real dimensional defect, although the current gate stops earlier because it rejects two display equations. |
| `electron_density` | 9 | `SYMBOL-BINDING-GAP` | `p_e = n_e k_B T_e` | `m^-3` | The ideal-gas relation is dimensionally valid. The prose states `p_e` in Pa, `k_B` in J/K and `T_e` in K, but the binder misses `k_B` and later binds `T_e` to the alternate eV representation. |
| `electron_temperature` | 13 | `SYMBOL-BINDING-GAP` | `p_e = n_e k_B T_e` | `eV` | The prose explicitly supplies `n_e` in `m^-3`, `k_B` in J/K, `T_e` in K and `p_e` in Pa, plus the alternate eV conversion. The gate overwrites the relation-local temperature representation with the DD unit and misses `k_B`. |
| `electron_temperature_at_magnetic_axis` | 2 | `PARSER-LIMITATION` | `n T \tau_E` | `eV` | The localized quantity is defined in prose and the only formula-like relation is an inline fusion-performance product, not a display equality defining the target. The selector reports zero equations; that is not a dimensional contradiction. |
| `elongation_of_plasma_boundary` | 2 | `SYMBOL-BINDING-GAP` | `\kappa_{\text{boundary}} = \frac{b}{a}` | `1` | The ratio of half-height to minor radius is dimensionless. The prose identifies both geometric lengths but does not repeat a unit token after each symbol, so the lexical binder produces no operands. |
| `minor_radius_of_plasma_boundary` | 2 | `SYMBOL-BINDING-GAP` | `a_{\text{boundary}} = \frac{R_{\text{out}} - R_{\text{in}}}{2}` | `m` | The prose states that both radii are in m. The binder captures only `R_{\text{in}}`, while subject selection chooses `R_{\text{out}}` instead of the equation's left-hand target. |
| `poloidal_magnetic_field` | 3 | `PARSER-LIMITATION` | `\mathbf{B}_p = B_R \hat{e}_R + B_Z \hat{e}_Z`; `\nabla \times \mathbf{B}_p = \mu_0 j_{\phi} \hat{e}_{\phi}` | `T` | Both vector decomposition and Ampere relation are dimensionally sound. The gate rejects the document before algebra because it admits exactly one display equation and its scalar grammar does not cover vector hats/curl. |
| `safety_factor` | 6 | `SYMBOL-BINDING-GAP` | `q = \frac{1}{2\pi} \oint \frac{B_{\phi}}{B_p R} dl_p` | `1` | The field ratio and length ratio make the integral dimensionless. The prose identifies the physical operands but states no adjacent literal unit tokens, leaving all four operand bindings empty. |
| **Total** | **38** | **32 binding / 5 parser / 1 genuine** |  |  | **37/38 are gate-apparatus failures on curated prose.** |

## Hand-adjudicated catalog rows

These 13 exact holdout rows cover every failed documentation family. Repeated identities are retained because the evaluation unit is the DD path, and each path carries its own authoritative declared unit.

| Row | DD source-path binding | Catalog identity | Offending relation quoted verbatim | Declared unit quoted verbatim | Classification |
|---:|---|---|---|---|---|
| 1 | `magnetics/flux_loop/area` | `area_of_flux_loop` | `\Delta \Psi = \frac{1}{A_{\text{eff}}} \int V \, dt` | `m^2` | `GENUINE-PHYSICS-ERROR` |
| 2 | `equilibrium/time_slice/constraints/n_e/measured` | `electron_density` | `p_e = n_e k_B T_e` | `m^-3` | `SYMBOL-BINDING-GAP` |
| 4 | `langmuir_probes/embedded/n_e` | `electron_density` | `p_e = n_e k_B T_e` | `m^-3` | `SYMBOL-BINDING-GAP` |
| 10 | `thomson_scattering/channel/n_e` | `electron_density` | `p_e = n_e k_B T_e` | `m^-3` | `SYMBOL-BINDING-GAP` |
| 11 | `core_profiles/profiles_1d/electrons/temperature` | `electron_temperature` | `p_e = n_e k_B T_e` | `eV` | `SYMBOL-BINDING-GAP` |
| 16 | `langmuir_probes/reciprocating/plunge/collector/t_e` | `electron_temperature` | `p_e = n_e k_B T_e` | `eV` | `SYMBOL-BINDING-GAP` |
| 23 | `turbulence/profiles_2d/electrons/temperature` | `electron_temperature` | `p_e = n_e k_B T_e` | `eV` | `SYMBOL-BINDING-GAP` |
| 24 | `camera_x_rays/t_e_magnetic_axis` | `electron_temperature_at_magnetic_axis` | `n T \tau_E` | `eV` | `PARSER-LIMITATION` |
| 25 | `summary/local/magnetic_axis/t_e/value` | `electron_temperature_at_magnetic_axis` | `n T \tau_E` | `eV` | `PARSER-LIMITATION` |
| 26 | `equilibrium/time_slice/boundary/elongation` | `elongation_of_plasma_boundary` | `\kappa_{\text{boundary}} = \frac{b}{a}` | `1` | `SYMBOL-BINDING-GAP` |
| 30 | `equilibrium/time_slice/boundary/minor_radius` | `minor_radius_of_plasma_boundary` | `a_{\text{boundary}} = \frac{R_{\text{out}} - R_{\text{in}}}{2}` | `m` | `SYMBOL-BINDING-GAP` |
| 32 | `edge_profiles/ggd/b_field/poloidal` | `poloidal_magnetic_field` | `\mathbf{B}_p = B_R \hat{e}_R + B_Z \hat{e}_Z` | `T` | `PARSER-LIMITATION` |
| 64 | `core_profiles/profiles_1d/q` | `safety_factor` | `q = \frac{1}{2\pi} \oint \frac{B_{\phi}}{B_p R} dl_p` | `1` | `SYMBOL-BINDING-GAP` |

## Consequence

The 38 failures do not justify labeling the curated catalog as dimensionally defective. The gate has demonstrated one valuable genuine catch, but on this calibration population it is primarily measuring whether prose happens to fit its symbol-unit binding and single-equation grammar. The correct next engineering target is the binding layer first—especially constants, grouped unit declarations, equation-left-hand subject selection and relation-local unit representations—followed by multi-equation target selection and vector/integral grammar. Until those limitations are closed and the catalog arm is remeasured, dimensional `fail` must remain diagnostic evidence rather than an automatic documentation-defect verdict.
