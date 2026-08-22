# Remaining governed source-attachment receipt

## Outcome

**COMPLETE.** One governed invocation regenerated the exact residual attachment
authority from the pinned candidate, canonical-target, persistence, review, and
score-transition evidence, then re-read the complete live source/name closure
inside its applying transaction. The signed cohort partitioned as **3 admitted +
6 refused = 9 signed rows**. Every refusal retains the guard's verbatim reason.

The maximal refusal-free subset applied exactly three source attachments:

- row 08 attached the existing unbound source
  `dd:plasma_sources/source/profiles_1d/neutral/state/energy` directly to
  `neutral_state_power_density`;
- row 20 retargeted
  `dd:distributions/distribution/profiles_2d/co_passing/collisions/electrons/torque_thermal_phi`
  from `toroidal_thermal_electron_torque_density_due_to_collisions` to
  `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions`; and
- row 29 retargeted
  `dd:summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` from
  `magnetic_field_at_pedestal_top_low_field_side` to
  `magnetic_field_at_pedestal_top_low_field_side_magnitude`.

The apply returned `outcome=applied`, `changed=3`, and `receipt_rows=3`.
`StandardNameChange` moved from **7,783 to 7,786**, so the live baseline read
inside the applying invocation increased by exactly the receipt count. The
genuine-orphan census fell from the required baseline **20 to 17**, exactly the
three newly sourced identities. `LLMCost` remained **27,631**.

The untouched source closure contained **9,628 rows**. It reported
`untouched_changed=0`, with the identical aggregate SHA-256 before and after:

```text
a60cede1ed552fdbfd0e28b155b5d9b2ee7c6c3f966f6a65c5b41598be0c62d4
```

Replay returned `outcome=already_applied`, `changed=0`, and
`persistent_writes=0`. It retained the same three exact receipts, the same
17-orphan/7,786-change/27,631-cost counters, and the same untouched-source
digest.

## Signed nine-row routes

The signed payload keeps all nine authority rows. It does not narrow the
authority to a convenient declared count. Rows 02, 05, and 15 point directly at
their canonical identities; no identity transition is inferred from any route.

| Row | Exact DD source | Recorded candidate | Signed target | Live disposition |
|---:|---|---|---|---|
| 02 | `core_profiles/profiles_1d/grid/area` | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | Refused because this exact source is already attached to the signed canonical target. |
| 05 | `interferometer/channel/n_e_line` | `line_integrated_electron_density` | `line_integrated_electron_number_density` | Refused because this exact source is already attached to the signed canonical target. |
| 07 | `equilibrium/time_slice/global_quantities/q_min/value` | `minimum_of_safety_factor` | `minimum_of_safety_factor` | Refused by the accepted-lifecycle guard. |
| 08 | `plasma_sources/source/profiles_1d/neutral/state/energy` | `neutral_state_power_density` | `neutral_state_power_density` | Applied as an exact unbound-source attachment. |
| 13 | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `poloidal_neutral_internal_state_momentum_convected_velocity` | `poloidal_neutral_internal_state_momentum_convected_velocity` | Refused by the accepted-lifecycle guard. |
| 15 | `distributions/distribution/profiles_2d/grid/theta_straight` | `poloidal_straight_field_line_angle` | `straight_field_line_angle` | Refused because the canonical target already has two live producers and one live child. |
| 20 | `distributions/distribution/profiles_2d/co_passing/collisions/electrons/torque_thermal_phi` | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | Applied as an exact retarget. |
| 21 | `charge_exchange/channel/ion/velocity_phi` | `toroidal_line_integrated_impurity_ion_velocity` | `toroidal_line_integrated_impurity_ion_velocity` | Refused by the accepted-lifecycle guard. |
| 29 | `summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | Applied as an exact retarget. |

The signed authority-payload SHA-256 is
`a99d1ee2bb14b73ff6e086d3cc61a82498e00c3fbfbb2c48ec3f96f245a56b68`.
The live participant closure SHA-256 was
`2eb3f0e48bb2acd34fed847d5b7ef116785dff31f93ccc1a5142e2fc61e80990`
both before and after participant locking. The resulting authorized manifest
SHA-256 is
`9f90e09523b874ee8c1bfdc7248d716d3ecd169380a05866c88ad62f26b72d46`.
It pins DD **4.1.1** and DD-resolution-manifest digest
`sha256:65a7ad8b1f1af0be59891f9dd84e506f292dfaa66930fe29601914882cdf9838`.

## Verbatim refusals

The six refusals divide into three observed classes: two already-canonical
attachments, three lifecycle refusals, and one target that is no longer a
genuine orphan. The strings below are the artifact's verbatim guard output.

| Row | Verbatim reason |
|---:|---|
| 02 | `source is already attached to the signed target; prior_receipts=[{'id': 'sn-change:b4acbe47-d0ff-4f65-9649-a902de519044', 'operation': 'human_edit'}, {'id': 'sn-change:source-migration:4f6b07868b528c27807599b5164465008b04d2a6984771a6963053404e4cff15', 'operation': 'source_migration_manifest'}]` |
| 05 | `source is already attached to the signed target; prior_receipts=[{'id': 'sn-change:bc834fea-a64b-4be7-9d70-51102d81b34e', 'operation': 'repair_semantic_source_binding'}, {'id': 'sn-change:36742cea-ac53-4d93-bb26-5f610dcc3a23', 'operation': 'repair_semantic_source_binding'}, {'id': 'sn-change:a0a8597d-a428-4b8e-b3df-e28fc9afc1c0', 'operation': 'repair_semantic_source_binding'}, {'id': 'sn-change:8a00298e-7ac6-4c90-88d8-8c7b16695033', 'operation': 'refine'}]` |
| 07 | `target lifecycle is not accepted: name_stage='reviewed'` |
| 13 | `target lifecycle is not accepted: name_stage='reviewed'` |
| 15 | `target is no longer a genuine orphan: live_producers=2, live_children=1` |
| 21 | `target lifecycle is not accepted: name_stage='drafted'` |

The lifecycle refusals are qualified results, not missing work from this
transaction. Rows 07 and 13 remain `reviewed` after fresh scores 0.6125 and
0.3625; row 21 remains `drafted` after fresh score 0.45. None cleared the locked
0.85 ordinary-review threshold, and this apply did not redraw, refine, or
direct-accept them.

## Direct canonical routing and zero-fold proof

Rows 02, 05, and 15 were signed directly against, respectively:

- `poloidal_plane_cross_sectional_area_of_flux_surface`;
- `line_integrated_electron_number_density`; and
- `straight_field_line_angle`.

Identity folds derived: **0**. An exact query for change rows under applying
run id `remaining-attachment-apply` whose `from_name` was any of the three
recorded predecessors and whose `to_name` differed returned **0 rows**.

The stronger structural proof snapshots every property on each predecessor and
every incident StandardName-to-StandardName relationship in both directions.
The complete three-predecessor snapshot was byte-equivalent before and after:

```text
4fafcc368a19bdb972fe5eba3b638407369cb9c266c7205cdc9ae7cf6bcfa1e5
```

Thus the `cross_section_of_flux_surface`,
`line_integrated_electron_density`, and
`poloidal_straight_field_line_angle` predecessor lineages are all untouched.
The canonical attachment routes did not become successor folds, parent folds,
or graph-text edits.

## Mutated rows and exact receipts

All three postconditions converge across the source scalar, `PRODUCED_NAME`
relationship, backing `HAS_STANDARD_NAME` projection, accepted target lifecycle,
and live-producer count:

| Row | Action | Post-apply source status | Scalar, relationship, and projection target | Live target producers |
|---:|---|---|---|---:|
| 08 | `attach_unbound` | `attached` | `neutral_state_power_density` | 1 |
| 20 | `retarget` | `composed` | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | 1 |
| 29 | `retarget` | `attached` | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | 1 |

The immutable receipt identities are:

| Row | Receipt id |
|---:|---|
| 08 | `sn-change:signed-source-attachment:9f90e09523b874ee8c1bfdc7248d716d3ecd169380a05866c88ad62f26b72d46:eb2cf60e6510e5ecc363fe9d` |
| 20 | `sn-change:source-migration:52be5498218c270ea0920389d8934450c16926192f3fe8187dd1fbdc7cce8502` |
| 29 | `sn-change:source-migration:d51f9c8d4f158b351d7b1ada7c6f19a7f54b5ef7a4f404f91478d34871ad1ab1` |

The proof query used both durable receipt keys, not an operation-name guess or
a bare global count:

```cypher
MATCH (change:StandardNameChange {
  run_id: 'remaining-attachment-apply',
  manifest_sha256: '9f90e09523b874ee8c1bfdc7248d716d3ecd169380a05866c88ad62f26b72d46'
})
RETURN change.id, change.row_id, change.source_id, change.target_id
ORDER BY change.row_id
```

It returned exactly the three rows above after apply and the same three rows on
replay. This exact run-plus-manifest read is the durable proof that replay added
no hidden write.

## Replay and counter proof

| Measure | Before apply | After apply | Replay after | Apply delta |
|---|---:|---:|---:|---:|
| Signed authority rows | 9 | 9 | 9 | 0 |
| Admitted / refused | 3 / 6 | 3 / 6 | 3 / 6 | 0 |
| `StandardNameChange` | 7,783 | 7,786 | 7,786 | **+3** |
| Exact run-plus-manifest receipts | 0 | 3 | 3 | **+3** |
| Genuine orphans | 20 | 17 | 17 | **-3** |
| `LLMCost` | 27,631 | 27,631 | 27,631 | 0 |
| Untouched source rows | 9,628 | 9,628 | 9,628 | 0 |
| Untouched source digest | `a60cede1…` | `a60cede1…` | `a60cede1…` | unchanged |

Replay outcome: `already_applied`; `changed=0`; `persistent_writes=0`;
`untouched_changed=0`.

## Durable artifacts and verification

- Complete machine-readable result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/remaining-attachment-result.json`.
- Apply receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/apply-receipt.json`.
- Replay receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/replay-receipt.json`.
- Exact invocation driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/apply_remaining_attachments.py`.
- Complete invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/remaining-attachment-apply.log`.
- Driver lint verification:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T003720161475-attachapply2/driver-lint-verified.log`;
  ruff check and ruff format-check both exited 0.

The invocation used the configured graph DD version and a pinned live
DD-resolution manifest for the DD unit axis. It made no provider call, created
no new source node, performed no identity fold, and made no mutation outside
the three admitted source closures plus their three immutable receipts.
