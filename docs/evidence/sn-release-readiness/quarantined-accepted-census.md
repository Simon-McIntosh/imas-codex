# Accepted-name validation contradiction census

Measured against the production `codex` graph on 2026-08-23. This was a
read-only census: every Cypher statement used `MATCH`/`RETURN`; no model was
called and no graph property, relationship, or node was changed.

## Headline result

| Measure | Live result |
|---|---:|
| All `StandardName` nodes, before | 4,395 |
| Accepted names | 2,535 |
| Accepted and `validation_status='quarantined'` | **47** |
| Accepted and `validation_status IS NULL` | **0** |
| Null validation status, graph-wide | 4 |
| All `StandardName` nodes, after | 4,395 |
| Node-count delta | **0** |

The live graph agrees with the reported figure of 47 accepted-and-quarantined
names. It disagrees with reading the four reported null-status names as
accepted rows: all four exist, but all four are `name_stage='pending'` derived
structural nodes. The accepted/null intersection is empty.

The 47 quarantines were stamped in one 39-second interval on 2026-08-08,
from `21:57:48.829Z` through `21:58:27.026Z`. This is a deterministic bulk
revalidation signature, not 47 independent acceptance events.

## Why accepted and quarantined can coexist

The ordinary review and export gates do exclude quarantine, but a later
validation pass can change the validation axis without changing the accepted
lifecycle axis:

1. `claim_names_for_validation` deliberately has no `name_stage` predicate; it
   claims every described node whose `validated_at` is null
   (`imas_codex/standard_names/graph_ops.py:7560-7585`).
2. `validate_name_candidate` classifies critical ISN, Pydantic, and audit
   findings as quarantined
   (`imas_codex/standard_names/workers.py:4601-4654`).
3. `mark_names_validated` overwrites `validation_status` but neither demotes nor
   even reads `name_stage`
   (`imas_codex/standard_names/graph_ops.py:7609-7648`).

That three-step path explains 44 rows: each has either `imported_at` or
`reviewed_name_at` before the August validation stamp (15 have both, 15 import
only, 12 review only), plus two legacy derived parents whose accepted event
predates the available acceptance timestamps. Their acceptance was already in
place; revalidation then quarantined them without retracting it.

Three rows show the reverse ordering and expose a second present-day path:
`cumulative_inside_flux_surface_torque`,
`magnetic_field_at_pedestal_top_low_field_side`, and
`total_thermal_plasma_internal_energy` have `parent_enriched_at` after
`validated_at`. `persist_enriched_parent` sets `name_stage='accepted'` while
using `coalesce(validation_status, 'valid')`, so an existing quarantine is
preserved (`imas_codex/standard_names/graph_ops.py:25095-25118`). The broader
structural repair path also promotes derived parents without a validation
predicate (`imas_codex/standard_names/graph_ops.py:25192-25206`). Thus the
combination is reachable in both orders.

Recommended invariant repair, not applied here: a validation transition to
`quarantined` must retract an accepted/approved name to a non-publishable
terminal or repairable lifecycle state in the same transaction, and every
structural-accept query must require `validation_status='valid'`. A graph test
should assert that the accepted/quarantined and accepted/null intersections are
both zero. The row repairs below must follow the ordinary `sn edit`/review or
deterministic revalidation paths; none should be hand-accepted.

## Dispositions by failed check

Every accepted-and-quarantined identity appears exactly once below. The
disposition names are recommendations, not applied mutations.

### `name_description_consistency_check` — 20 rows

All 20 failed the same specific predicate: their description mentions a
Fourier/spectral mode while `_SPECTRAL_NAME_MARKERS` does not recognize the
already-present perturbation/decomposition semantics. The leaf identities
carry `real_part` or `imaginary_part`; the remaining identities are the
structural parents of those leaves. Representative bindings are
`dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary`
and
`dd:mhd_linear/time_slice/toroidal_mode/vacuum/b_field_perturbed/coordinate3/real`.

Disposition for every row: **repair the audit, then deterministically
revalidate the same identity**. Extend the check through the public ISN parse so
real/imaginary decomposition and a structural parent of decomposed children are
recognized. Do not rewrite correct physics prose merely to silence a lexical
check.

| Name | Specific failed check |
|---|---|
| `perturbed_electrostatic_potential_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized as a decomposition marker. |
| `perturbed_electrostatic_potential_real_part` | Spectral description; existing `real_part` was not recognized. |
| `perturbed_plasma_mass_density` | Complex-eigenfunction description on the structural parent of real/imaginary members was not recognized. |
| `perturbed_plasma_mass_density_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `perturbed_plasma_pressure` | Complex-eigenfunction description on the structural parent was not recognized. |
| `perturbed_plasma_pressure_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `perturbed_plasma_pressure_real_part` | Spectral description; existing `real_part` was not recognized. |
| `perturbed_plasma_velocity` | Complex-eigenfunction description on the three-component structural parent was not recognized. |
| `poloidal_perturbed_plasma_magnetic_field_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `poloidal_perturbed_plasma_magnetic_field_real_part` | Spectral description; existing `real_part` was not recognized. |
| `poloidal_perturbed_plasma_velocity` | Complex-eigenfunction description on the real/imaginary structural parent was not recognized. |
| `poloidal_perturbed_vacuum_magnetic_field` | Complex-eigenfunction description on the real/imaginary structural parent was not recognized. |
| `poloidal_perturbed_vacuum_magnetic_field_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `radial_perturbed_plasma_velocity` | Complex-eigenfunction description on the real/imaginary structural parent was not recognized. |
| `radial_perturbed_plasma_velocity_real_part` | Spectral description; existing `real_part` was not recognized. |
| `radial_perturbed_vacuum_magnetic_field_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `toroidal_perturbed_plasma_magnetic_field_imaginary_part` | Spectral description; existing `imaginary_part` was not recognized. |
| `toroidal_perturbed_plasma_velocity` | Complex-eigenfunction description on the real/imaginary structural parent was not recognized. |
| `toroidal_perturbed_vacuum_magnetic_field` | Complex-eigenfunction description on the real/imaginary structural parent was not recognized. |
| `toroidal_perturbed_vacuum_magnetic_field_real_part` | Spectral description; existing `real_part` was not recognized. |

### `name_unit_consistency_check` — 11 rows

This lexical check conflates a token anywhere in the name with the dimensional
head quantity. The dispositions therefore split rather than treating all 11 as
unit corrections.

| Name | Specific failed check | Named disposition |
|---|---|---|
| `energy_confinement_enhancement_factor` | Token `energy` expected an energy unit, but the bound H98 enhancement factor is correctly dimensionless (`dd:summary/global_quantities/h_98/value`). | **Audit repair then same-identity revalidation**: recognize ratio/factor head semantics. |
| `gyrocenter_pressure` | `pressure` expected Pa-like units; node is a dimensionless normalized family parent. | **Source/family-guided recompose**: retain normalization in identity or retire the invalid peeled parent; do not change its DD-authoritative unit. |
| `perturbed_gyrocenter_pressure` | `pressure` expected Pa-like units; description explicitly says normalized dimensionless moment. | **Source-guided recompose** with explicit normalization, then ordinary review. |
| `perturbed_particle_pressure` | `pressure` expected Pa-like units; description explicitly says normalized dimensionless moment. | **Source-guided recompose** with explicit normalization, then ordinary review. |
| `perturbed_pressure` | `pressure` expected Pa-like units; it is an invalid peel from `normalized_perturbed_pressure`. | **Retire the invalid structural parent** while retaining the normalized child; prevent unit-erasing normalization peels. |
| `plasma_internal_energy` | Token `energy` expected J-like units, but unit is W and its description/source (`dd:summary/global_quantities/denergy_thermal_dt/value`) say rate of change. | **Family/source-guided recompose** to a tendency identity; retire the rate-as-stored-energy parent. |
| `thermal_plasma_internal_energy` | Token `energy` expected J-like units, but unit is W and description says an energy rate. | **Family/source-guided recompose** to a tendency identity; do not relabel W as stored energy. |
| `total_thermal_plasma_internal_energy` | Token `energy` expected J-like units; unit is W while prose claims stored energy. | **Resolve the derived-family semantic conflict**: the W-valued tendency child cannot authorize a stored-energy parent; retire or rebuild from a J-valued authority. |
| `tendency_of_rotation_frequency_of_neoclassical_tearing_mode` | Token `frequency` expected s^-1, but the leading tendency correctly makes the unit s^-2 (`dd:ntms/time_slice/mode/dfrequency_dt`). | **Audit repair then same-identity revalidation**: make unit checking operator-aware. |
| `tendency_of_total_thermal_plasma_internal_energy` | Token `energy` expected J-like units, but the leading tendency correctly makes the unit W. | **Audit repair then same-identity revalidation**: a tendency of energy has power dimensions. |
| `toroidal_angle_of_active_limiter_point` | Token `angle` expected rad/deg/sr; the DD-bound source declares `1` (`dd:edge_transport/model/ggd/ion/momentum/flux_limiter/phi`). | **DD-unit adjudication**: verify the source declaration and existing unit-exception policy upstream; never hand-edit the graph unit. Revalidate only after that authority is settled. |

### `implicit_field_check` — 4 rows

These are lexical false positives. `field_line` is a registered compound and
`high_field_side`/`low_field_side` are spatial side qualifiers; none is a bare
unnamed physics field.

Disposition for all four: **repair compound parsing in the audit, then
revalidate the same identity**.

| Name | Specific failed check |
|---|---|
| `ion_field_line_average_temperature_over_scrape_off_layer` | Mistook `field_line` for a bare field after `ion`; bound to `dd:langmuir_probes/reciprocating/plunge/t_i_average`. |
| `magnetic_field_at_pedestal_top_low_field_side` | Mistook the `low_field_side` locus for a bare field. |
| `poloidal_electron_beta_at_pedestal_top_high_field_side` | Mistook the `high_field_side` locus for a bare field. |
| `poloidal_electron_beta_at_pedestal_top_low_field_side` | Mistook the `low_field_side` locus for a bare field. |

### `cumulative_prefix_check` — 3 rows

The check rejected the literal `cumulative_`/`integrated_` spelling and advised
the canonical `_inside_flux_surface` construction. Each row therefore needs
name work rather than a status flip.

| Name | Specific failed check | Named disposition |
|---|---|---|
| `cumulative_inside_flux_surface_torque` | Redundant/non-canonical `cumulative_` before an already explicit `inside_flux_surface` locus. | **Family-guided recompose** toward the canonical torque-inside-flux-surface form, then review. |
| `volume_integrated_runaway_electron_density` | Non-vocabulary `integrated_` prefix; prose/source mean an extensive runaway-electron count. | **Source-guided recompose** to an extensive count identity, preserving runaway-electron semantics; do not retain `density` for a dimensionless total. |
| `volume_integrated_total_electron_density` | Non-vocabulary `integrated_` prefix; `dd:interferometer/electrons_n` and prose mean an extensive electron count. | **Source-guided recompose** to an extensive count identity, preserving total-electron semantics. |

### ISN semantic error plus `canonical_locus_check` — 3 rows

Each row failed the ISN hard semantic rule that `outline` must name the entity
whose boundary/path is represented, and the codex check independently rejected
intrinsic `_of_` for a coordinate evaluated at a locus.

Disposition for all three: **source-guided rename through ordinary review**,
using the audit's `radial_outline_at_…` direction as a hint rather than a hand
write; the proposed identity must pass the current ISN grammar and semantic
checks.

| Name | Specific failed check | Representative source |
|---|---|---|
| `radial_outline_of_flux_surface` | ISN: outline entity/locus incomplete; codex: use `at_flux_surface`, not `of_flux_surface`. | `dd:equilibrium/time_slice/contour_tree/node/levelset/r` |
| `radial_outline_of_plasma_boundary` | ISN: outline entity/locus incomplete; codex: use `at_plasma_boundary`. | `dd:equilibrium/time_slice/boundary/outline/r` |
| `radial_outline_of_wall` | ISN: outline entity/locus incomplete; codex: use `at_wall`. | `dd:wall/description_2d/mobile/unit/outline/r` |

### `canonical_locus_check` only — 2 rows

Disposition: **source-guided rename through review** from intrinsic `_of_` to
the evaluated-coordinate `_at_` relation. Preserve the ordered-sample rule:
the many first/second/third point paths bound to the line-of-sight row remain
provenance, not identity.

| Name | Specific failed check | Intended alternative |
|---|---|---|
| `toroidal_coordinate_of_detector_pixel` | Field-evaluated coordinate used intrinsic `_of_`. | `toroidal_coordinate_at_detector_pixel` |
| `vertical_coordinate_of_line_of_sight` | Field-evaluated coordinate used intrinsic `_of_`. | `vertical_coordinate_at_line_of_sight` |

### One-row failure groups

| Failed check | Name | Specific failure | Named disposition |
|---|---|---|---|
| `amplitude_of_prefix_check` | `phase_of_ion_cyclotron_heating_antenna` | Prefix form `phase_of_<X>` is non-canonical; the check gives `ion_cyclotron_heating_antenna_phase`. | **Source-guided rename through review** to the noun-suffix form, with the bound IC-heating phase source retained. |
| `latex_def_check` | `rotation_frequency_due_to_e_cross_b_drift` | Documentation uses `$(R, \\phi, Z)$` without a definition sentence. | **Documentation repair then same-identity revalidation** through the docs edit/review path; the name itself is not the failed object. |
| `multi_subject_check` | `electron_density_at_pellet_path` | Lexical check classified both `electron` and locus-owner `pellet` as subjects. | **Audit repair then same-identity revalidation**: use parsed segment roles so `pellet_path` is a locus, preserving `dd:pellets/time_slice/pellet/path_profiles/n_e`. |
| Pydantic grammar validation | `poloidal_perturbed_magnetic_flux_at_measurement_position_due_to_wave_particle_interaction` | Parser treated `poloidal_perturbed_magnetic_flux_at_measurement` as an unregistered coordinate-axis prefix. | **Source-guided recompose**, not rescore: steer the exact ECE `delta_position_suprathermal/psi` source to a shorter grammar-valid identity, and separately adjudicate its surprising W unit through DD authority. |

The partition closes: 20 + 11 + 4 + 3 + 3 + 2 + 4 one-row groups = **47**.

## The four graph-wide null-status rows

These are not accepted contradictions and cannot be silently exported under the
current accepted-name gate. All are pending, descriptionless derived nodes. The
null status is nevertheless ambiguous state and should be dispositioned rather
than left as an accidental default.

| Name | Structural evidence | Named disposition |
|---|---|---|
| `coolant_mass` | Binary ratio operand of `ratio_of_coolant_mass_to_time`; child unit kg.s^-1 cannot authorize a kg parent. | **Internal structural stub**: keep non-publishable or replace with an explicit internal-node lifecycle; do not infer a unit or promote. |
| `particle_temperature` | Binary ratio operand of `ratio_of_particle_temperature_to_particle_reference_temperature`; dimensionless child cannot authorize a temperature unit. | **Internal structural stub**: keep non-publishable or mark explicitly internal; do not infer or promote. |
| `flux_at_wall_due_to_recombination` | Qualifier parent of energy flux (W.m^-2) and particle flux (m^-2.s^-1); children are dimensionally heterogeneous. | **Non-materializable family abstraction**: keep internal/explicitly non-release or retire the parent edge; no single quantitative StandardName unit exists. |
| `outline` | Projection parent of `toroidal_outline` (rad) and `vertical_outline` (m); children are dimensionally heterogeneous and `outline` lacks the entity. | **Retire as a publishable-name candidate** while retaining any grammar-structure provenance; do not validate or promote the bare identity. |

The code already limits parent seeding to selected operator kinds and refuses
unitless/heterogeneous materialization
(`imas_codex/standard_names/graph_ops.py:4339-4399` and
`imas_codex/standard_names/graph_ops.py:3819-3849`). The follow-on is to make
that non-publishable outcome explicit so a null validation scalar no longer has
to carry the distinction.

## Recommended repair order

1. Close the lifecycle invariant so no new accepted/quarantined combination can
   be written in either validation order.
2. Correct the lexical/parse-aware audits, then revalidate the 28 same-identity
   audit false positives and the one documentation-only failure.
3. Route the 18 identity/unit-conflict rows through source- or family-guided
   `sn edit`/review work, with DD unit authority settled first where called out.
4. Give the four pending structural stubs an explicit internal/non-release
   disposition rather than manufacturing validation status for quantities that
   cannot be materialized.

No repair in this list was applied by this census.
