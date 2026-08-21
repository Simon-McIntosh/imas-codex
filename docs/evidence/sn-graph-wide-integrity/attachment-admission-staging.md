# Attachment admission repair staging

## Outcome

The exact nine-target attachment cohort is now **9 of 9 deterministically
admissible**, compared with the recorded live preflight baseline of **4 of 9**.
The repair applies the five Codex-owned dispositions without taking a name-review
draw: three attachment targets redirect to existing canonical identities and two
strict-valid identities retain their spellings with governed descriptions.

The shared admission function `validate_name_candidate` evaluated every staged
target with its exact DD path, live unit and identity metadata, live structural
children, and the description that the attachment cohort will carry. All nine
returned `validation_status='valid'`; all nine had a non-empty description; and
all nine returned zero critical findings. No target was accepted, rescored,
reviewed, attached, renamed, or hand-edited by this node.

| Measure | Recorded preflight | Repaired staging |
|---|---:|---:|
| Exact attachment rows | 9 | 9 |
| Deterministically admissible | **4** | **9** |
| Blocked | 5 | 0 |
| Target redirects | 0 | **3** |
| Governed description additions | 0 | **2** |
| Ordinary-review draws | 0 | **0** |
| Provider calls | 0 | **0** |
| Attributable spend | USD 0.000000 | **USD 0.000000** of USD 25 |

This is admission staging, not attachment authority. The later attachment apply
still owes its fresh exact manifest, live closure and unit guards, compare-and-set
checks, collateral proof, and replay. The three redirected rows must use their
staged targets when that manifest is regenerated. The two descriptions must ride
the governed composition boundary; this artifact does not authorize a direct
graph-text edit.

## Three canonical target redirects

Each redirect is already settled by the live plan and points to an existing
identity whose meaning and unit match the exact DD source. None creates a new
identity or requires an ISN grammar change.

| Attachment row | Exact DD source | Before target | After target | Why the after target is authoritative | Sanctioned route |
|---:|---|---|---|---|---|
| 02 | `core_profiles/profiles_1d/grid/area` | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | The value is the area enclosed in a poloidal plane, distinct from swept toroidal `surface_area_of_flux_surface`; the replacement is accepted at name score 1.0 and strictly round-trips. | Regenerate the exact signed attachment manifest with the accepted replacement target. |
| 05 | `interferometer/channel/n_e_line` | `line_integrated_electron_density` | `line_integrated_electron_number_density` | The replacement explicitly identifies the counted electron quantity, is accepted at name score 1.0, and owns the established interferometer/refractometer line-integral family. | Revalidate with the corrected compound-aware cumulative-prefix audit, then regenerate the exact signed attachment manifest against the accepted replacement target. |
| 15 | `distributions/distribution/profiles_2d/grid/theta_straight` | `poloidal_straight_field_line_angle` | `straight_field_line_angle` | The accepted identity describes the poloidal angular coordinate of a straight-field-line magnetic coordinate system without encoding the DD storage rank. | Revalidate with the corrected compound-aware field audit, then regenerate the exact signed attachment manifest against the accepted replacement target. |

The redirections preserve the original DD paths. They change only which reviewed
identity the future attachment manifest names.

## Two governed descriptions

Both identities strictly round-trip under ISN `0.8.0rc66`; their remaining
admission defect was absent description text. The descriptions below implement
the plan-recorded semantic outlines and pass the full deterministic gate with
the exact source path and unit.

The read-only source-hint preflight was also exercised. Both exact DD sources
currently have `status='attached'`, so `sn source-hint --dry-run` correctly
refused them as `source_not_extracted`: row 21 currently feeds
`toroidal_ion_velocity`, and row 29 currently feeds
`magnetic_field_at_pedestal_top_low_field_side`. This staging artifact therefore
does not pretend that a hint was applied. Persisting the descriptions must occur
in the later graph-mutating lane through a sanctioned vehicle that preserves
those existing source bindings and their last-producer closure; direct graph
text mutation and an unsafe reset remain forbidden.

### Attachment row 21

- Identity: `toroidal_line_integrated_impurity_ion_velocity`
- Exact DD source: `charge_exchange/channel/ion/velocity_phi`
- Unit: `m.s^-1`

Before:

```text
null
```

After:

```text
Toroidal component of the impurity-ion velocity inferred from a charge-exchange diagnostic channel's line-of-sight-integrated signal, expressed in m.s^-1. Here line_integrated describes integration along the diagnostic observation path, not accumulation inside a flux surface.
```

This retains `line_integrated` as diagnostic-observation semantics and explicitly
distinguishes it from cumulative integration inside a flux surface. The corrected
compound-aware cumulative-prefix audit returns no finding.

### Attachment row 29

- Identity: `magnetic_field_at_pedestal_top_low_field_side_magnitude`
- Exact DD source: `summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value`
- Unit: `T`

Before:

```text
null
```

After:

```text
Magnitude of the total magnetic field, expressed in tesla, evaluated at the pressure-pedestal-top position determined by the fit on the low-field (outboard) side.
```

This distinguishes the measured `magnetic_field` quantity from the registered
`low_field_side` spatial locus. The corrected compound-aware implicit-field and
repeated-token audits both return no finding.

## Exact repaired cohort

The four already-valid rows remain byte-for-byte identity choices. The three
redirects and two description additions are the complete delta; every source
path is the reviewed path from the attachment-candidate authority.

| Row | Exact DD path | Recorded target | Staged target | Staged description source | Deterministic result | Critical findings |
|---:|---|---|---|---|---|---:|
| 02 | `core_profiles/profiles_1d/grid/area` | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | Existing governed description | valid | 0 |
| 05 | `interferometer/channel/n_e_line` | `line_integrated_electron_density` | `line_integrated_electron_number_density` | Existing governed description | valid | 0 |
| 07 | `equilibrium/time_slice/global_quantities/q_min/value` | `minimum_of_safety_factor` | `minimum_of_safety_factor` | Existing description, unchanged | valid | 0 |
| 08 | `plasma_sources/source/profiles_1d/neutral/state/energy` | `neutral_state_power_density` | `neutral_state_power_density` | Existing description, unchanged | valid | 0 |
| 13 | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `poloidal_neutral_internal_state_momentum_convected_velocity` | `poloidal_neutral_internal_state_momentum_convected_velocity` | Existing description, unchanged | valid | 0 |
| 15 | `distributions/distribution/profiles_2d/grid/theta_straight` | `poloidal_straight_field_line_angle` | `straight_field_line_angle` | Existing governed description | valid | 0 |
| 20 | `distributions/distribution/profiles_2d/co_passing/collisions/electrons/torque_thermal_phi` | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | Existing description, unchanged | valid | 0 |
| 21 | `charge_exchange/channel/ion/velocity_phi` | `toroidal_line_integrated_impurity_ion_velocity` | `toroidal_line_integrated_impurity_ion_velocity` | Governed addition above | valid | 0 |
| 29 | `summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | Governed addition above | valid | 0 |

## Machine-readable staging input

The validation log parses this block and does not maintain a second hand-written
mapping. A null `description` means use the existing live target description.

```json
{
  "baseline_admissible": 4,
  "rows": [
    {"attachment_row": 2, "before_id": "cross_section_of_flux_surface", "target_id": "poloidal_plane_cross_sectional_area_of_flux_surface", "source_path": "core_profiles/profiles_1d/grid/area", "description": null},
    {"attachment_row": 5, "before_id": "line_integrated_electron_density", "target_id": "line_integrated_electron_number_density", "source_path": "interferometer/channel/n_e_line", "description": null},
    {"attachment_row": 7, "before_id": "minimum_of_safety_factor", "target_id": "minimum_of_safety_factor", "source_path": "equilibrium/time_slice/global_quantities/q_min/value", "description": null},
    {"attachment_row": 8, "before_id": "neutral_state_power_density", "target_id": "neutral_state_power_density", "source_path": "plasma_sources/source/profiles_1d/neutral/state/energy", "description": null},
    {"attachment_row": 13, "before_id": "poloidal_neutral_internal_state_momentum_convected_velocity", "target_id": "poloidal_neutral_internal_state_momentum_convected_velocity", "source_path": "plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol", "description": null},
    {"attachment_row": 15, "before_id": "poloidal_straight_field_line_angle", "target_id": "straight_field_line_angle", "source_path": "distributions/distribution/profiles_2d/grid/theta_straight", "description": null},
    {"attachment_row": 20, "before_id": "toroidal_co_passing_thermal_electron_torque_density_due_to_collisions", "target_id": "toroidal_co_passing_thermal_electron_torque_density_due_to_collisions", "source_path": "distributions/distribution/profiles_2d/co_passing/collisions/electrons/torque_thermal_phi", "description": null},
    {"attachment_row": 21, "before_id": "toroidal_line_integrated_impurity_ion_velocity", "target_id": "toroidal_line_integrated_impurity_ion_velocity", "source_path": "charge_exchange/channel/ion/velocity_phi", "description": "Toroidal component of the impurity-ion velocity inferred from a charge-exchange diagnostic channel's line-of-sight-integrated signal, expressed in m.s^-1. Here line_integrated describes integration along the diagnostic observation path, not accumulation inside a flux surface."},
    {"attachment_row": 29, "before_id": "magnetic_field_at_pedestal_top_low_field_side_magnitude", "target_id": "magnetic_field_at_pedestal_top_low_field_side_magnitude", "source_path": "summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value", "description": "Magnitude of the total magnetic field, expressed in tesla, evaluated at the pressure-pedestal-top position determined by the fit on the low-field (outboard) side."}
  ]
}
```

## Validation and mutation boundary

Validation uses the current merged compound-aware audit implementation and the
public ISN parser. The four focused precision regressions pass, covering
`line_integrated`, `straight_field_line`, `low_field_side`, and storage-rank
prose. The artifact-driven live-metadata validation then reports:

```text
baseline_admissible=4/9
staged_admissible=9/9
staged_quarantined=0
critical_findings=0
target_redirects=3
governed_descriptions=2
ordinary_review_draws=0
provider_calls=0
attributable_spend_usd=0.000000
PASS
```

The result spends none of the USD 25 ceiling. It deliberately stops before any
ordinary name review so the one-draw sequencing rule remains intact.
