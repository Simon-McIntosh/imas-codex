# Ineligible DD source retirement

## Outcome

The three DD sources excluded from scalar-selected deduplication were not
missing legitimate backing projections. Their backing DD nodes are container
categories outside `SN_SOURCE_CATEGORIES`: two are `structural` and one is
`representation`. Such paths cannot provide quantity-name authority, so every
`PRODUCED_NAME` relationship from these sources was spurious regardless of the
source scalar.

The exact signed retirement applied successfully. It retired all three sources
to `not_physical_quantity`, cleared their `produced_sn_id` scalars, deleted all
32 signed `PRODUCED_NAME` relationships, and deleted zero backing projections.
It did not change any StandardName lifecycle. Instead, its receipt surfaced the
32 names left with no live producing source for the separate orphan workflow.
The exact replay returned `already_applied`, `changed=0`, and measured zero
persistent writes.

## Instrument contract

`retire_ineligible_standard_name_sources` accepts an exact source-id cohort and
projects the complete signed source, DD-backing, binding, target, and matching
backing-projection closure. Admission requires one DD backing node whose
`node_category` is present and outside `SN_SOURCE_CATEGORIES`; an eligible
category is a signed refusal. Apply locks and re-reads every signed participant,
requires the manifest SHA-256 to match, exact-CAS checks the source identity,
backing identity and category, status, scalar, claim state, and binding element
ids, then performs all retirement mutations in one transaction.

The dedicated class is the only path that overrides the ordinary would-orphan
guard. Its output is the canonical list of affected nonterminal names whose
live producing-source count becomes zero. Those names are surfaced only: the
operator neither supersedes nor otherwise changes them. One deterministic
`StandardNameChange` records the signed source cohort, removed bindings, retired
status, and orphan list. Replay verifies the applied state and rolls back its
read transaction without writes.

## Transactional regressions

The red proof failed during collection because the public retirement entry
point did not yet exist (`red-proof.log`, SHA-256
`d92c6cf0635c87048ffb6793e8ed634fb142bb72ca8051eed2c65b7fb7765737`).

The first full disposable-Neo4j execution exposed only an assertion ordering
error: the receipt correctly canonicalized the two orphan identities, while the
test expected seed order (`full-tests-disposable.log`, SHA-256
`6cdb4eaddac569c85be443ecc978c7fd0b8ca4d422cb9faf63242e4049656994`).
The focused correction passed 1/1 (`failed-test-rerun.log`, SHA-256
`55d6e091857f1ec69d16135149ad51e6b8b5ebed5ae3b794bf80ffa1158c8fc5`).
The final full execution against isolated Neo4j 2026.01.4 passed 9/9 with zero
skips (`full-tests-disposable-final.log`, SHA-256
`0c6d5ad8644eb9e0adf4b2d3dfbf5d90ffb753c9a8d2f82f69847f2530f2676e`).
The new cases prove:

- an ineligible `structural` source loses all bindings and its scalar, enters
  the durable retired state, and returns every newly source-less name;
- a `quantity` backing is refused with both bindings and scalar intact; and
- an exact replay preserves a byte-identical participant snapshot and leaves
  every Neo4j update counter at zero.

## Live exact-manifest apply

The fresh preview and applied receipt share manifest SHA-256
`6c750a85d74de1f875ded8e854e2cad43418a2275e6a9beaeb71c2dfdc9644ed`.
Counts were `requested=3`, `admitted=3`, `refused=0`,
`bindings_to_detach=32`, and `projections_to_detach=0`.

The first apply attempt failed closed because a whole-property-map comparison
did not preserve Neo4j temporal types across the driver boundary. The
transaction rolled back with no durable mutation. The final exact CAS retained
all signed semantic and identity predicates without comparing transported
temporal maps. The second fresh apply succeeded. Its full log is
`live-apply-2.log`, SHA-256
`2e2d61ca0b8664d864bb3fb5486360b9fa3b22ef3a6774d6acb6a04c7ee7f97e`.

Receipt:

`sn-change:ineligible-source-retirement:6c750a85d74de1f875ded8e854e2cad43418a2275e6a9beaeb71c2dfdc9644ed`

| Source | Backing category | Bindings before | Scalar before | Status after | Bindings after | Scalar after |
|---|---:|---:|---|---|---:|---|
| `dd:distributions/distribution` | `structural` | 26 | `trapped_torque_density_due_to_j_cross_b_force` | `not_physical_quantity` | 0 | null |
| `dd:edge_transport/model/ggd` | `representation` | 4 | `total_momentum_convection_velocity` | `not_physical_quantity` | 0 | null |
| `dd:gyrokinetics_local/non_linear` | `structural` | 2 | `normalized_perturbed_magnetic_field` | `not_physical_quantity` | 0 | null |

All three rows also carry `skip_reason=dd_node_category_ineligible`. Their
backing node categories were unchanged. The replay returned
`outcome=already_applied`, `changed=0`, `persistent_writes=0`; its complete
participant snapshot remained byte-identical at SHA-256
`5bca1462476a815b4729475737e4b66a7142180f3b8a87e7f2d086de5e52af3a`.
`StandardNameChange`, `LLMCost`, and `SNRun` counts were flat across replay at
7155, 27467, and 489 respectively. Provider calls were zero.

## Surfaced orphan cohort

The receipt lists these 32 names with zero live producing sources. They remain
available for the governed orphan workflow and were not auto-superseded:

1. `co_passing_fast_current_density`
2. `co_passing_fast_ion_charge_state_torque_density_due_to_collisions`
3. `co_passing_fast_ion_torque_density_due_to_collisions`
4. `co_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
5. `co_passing_thermal_ion_torque_density_due_to_collisions`
6. `co_passing_torque_density_due_to_j_cross_b_force`
7. `counter_passing_current_density`
8. `counter_passing_fast_current_density`
9. `counter_passing_fast_electron_torque_density_due_to_collisions`
10. `counter_passing_fast_ion_charge_state_torque_density_due_to_collisions`
11. `counter_passing_fast_ion_torque_density_due_to_collisions`
12. `counter_passing_thermal_electron_torque_density_due_to_collisions`
13. `counter_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
14. `counter_passing_torque_density_due_to_j_cross_b_force`
15. `current_density_due_to_fast_ion`
16. `electron_particle_convection_velocity`
17. `fast_electron_torque_density_due_to_collisions`
18. `fast_ion_charge_state_power_density_due_to_collisions`
19. `fast_ion_torque_density_due_to_collisions`
20. `ion_heat_convection_velocity`
21. `neutral_species_particle_diffusivity`
22. `normalized_momentum_flux_due_to_e_cross_b_drift`
23. `normalized_perturbed_magnetic_field`
24. `thermal_electron_torque_density_due_to_collisions`
25. `thermal_ion_charge_state_power_density_due_to_collisions`
26. `thermal_ion_torque_density_due_to_collisions`
27. `total_momentum_convection_velocity`
28. `trapped_current_density`
29. `trapped_fast_current_density`
30. `trapped_fast_electron_torque_density_due_to_collisions`
31. `trapped_thermal_ion_torque_density_due_to_collisions`
32. `trapped_torque_density_due_to_j_cross_b_force`

No provider call occurred.
