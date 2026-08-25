# Graph-wide conditional sign-convention gate

Snapshot: `2026-08-25T05:54:35.712826+00:00`, live `codex` graph. The
documentation drain had recorded `stop_reason=no_eligible_work` before this
measurement, so this is the deferred post-drain census rather than an interim
observation while documents were still being promoted. Worktree HEAD at
measurement: `1725e7fc`.

## Result

The shipped conditional check was applied once to every node whose
`docs_stage` is `accepted`:

```python
score_documentation(
    documentation,
    physics_context=DocumentationPhysicsContext(
        cocos_transformation_type=transformation_type,
    ),
).gate_vector["sign_convention"]
```

Candidate and property coverage was retained before trusting any outcome:
**2,952 accepted documents, 2,952 with `id`, 2,952 with `documentation`, and
351 with `cocos_transformation_type`**. The three outcomes sum to the full
accepted-document population:

| Outcome | Count |
|---|---:|
| `pass` | **333** |
| `fail` | **18** |
| `not_evaluable` | **2,601** |
| **Accepted-document total** | **2,952** |

This is a **failed residual gate**, not graph-wide closure. Every current
failure has stored transformation class `one_like`; under the gate contract
that class is COCOS-invariant, so a sign-convention paragraph is forbidden.
There are **18 invariant failures and 0 sensitive failures**.

## Remaining failures

| Accepted identity | Stored transformation class | Gate classification |
|---|---|---|
| `change_in_ion_state_mean_ionisation_potential` | `one_like` | invariant |
| `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `one_like` | invariant |
| `ion_average_temperature` | `one_like` | invariant |
| `magnetic_field` | `one_like` | invariant by stored metadata; prior structural diagnosis resolved `b0_like`, so this row also exposes an unresolved authority mismatch |
| `parallel_current_density_due_to_ohmic_current_drive` | `one_like` | invariant |
| `poloidal_ion_velocity_at_measurement_position` | `one_like` | invariant |
| `toroidal_helium_3_velocity_at_plasma_boundary` | `one_like` | invariant |
| `x_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | `one_like` | invariant |
| `x_minor_axis_unit_vector_of_shatter_cone` | `one_like` | invariant |
| `x_unit_vector_of_pellet_injector` | `one_like` | invariant |
| `y_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | `one_like` | invariant |
| `y_direction_unit_vector_of_shatter_cone` | `one_like` | invariant |
| `y_minor_axis_unit_vector_of_shatter_cone` | `one_like` | invariant |
| `z_direction_unit_vector_of_camera` | `one_like` | invariant |
| `z_direction_unit_vector_of_electron_cyclotron_launcher_mirror` | `one_like` | invariant |
| `z_direction_unit_vector_of_pellet_injector` | `one_like` | invariant |
| `z_major_axis_unit_vector_of_shatter_cone` | `one_like` | invariant |
| `z_minor_axis_unit_vector_of_shatter_cone` | `one_like` | invariant |

All 18 return the same deterministic reason:
`COCOS-invariant quantity states a sign convention`. No failure has a
sensitive transformation class, so this census found no sensitive document
missing or misformatting its required sign paragraph.

## Comparison with the frozen repair

The frozen-repair census was **325 pass / 4 fail / 2,411 not_evaluable across
2,740 accepted documents**. The post-drain census is **333 / 18 / 2,601 across
2,952**, respectively:

| Measure | Frozen repair | Post-drain | Change |
|---|---:|---:|---:|
| `pass` | 325 | **333** | **+8** |
| `fail` | 4 | **18** | **+14** |
| `not_evaluable` | 2,411 | **2,601** | **+190** |
| Accepted-document total | 2,740 | **2,952** | **+212** |

The repair therefore remains visible in the larger population, but the drain
promoted more invariant documents carrying forbidden sign prose than the
frozen repair had left behind. The failing set is not merely the original four:
it now contains 18 accepted identities. This measurement is read-only; it does
not authorize or perform another stock repair.

## Reproducible record

The exact machine-readable result, including coverage, every failure, the
transformation distribution and the frozen baseline, is stored at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T055106499263-n-signgraphwiderescore/graphwide-sign-gate.json`.
