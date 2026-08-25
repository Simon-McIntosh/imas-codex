# Structural authority replay for described derived parents

Measured 2026-08-25 against the live production `codex` graph. The operation
gave already-accepted, valid derived parents a signed structural authority only
when they carried a real description, had no name score or prior structural
authority, and had at least one accepted child. It did not synthesize a score,
change lifecycle state, or treat the historical `structural-inheritance` marker
as authority.

## Live result

The closure-audit accepted-name partition changed by exactly the replayed
cohort size:

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Accepted names | 2,295 | 2,295 | 0 |
| Names carrying their own score | 1,968 | 1,968 | 0 |
| Names carrying structural authority | 248 | 318 | +70 |
| Score/structural overlap | 0 | 0 | 0 |
| Accepted names carrying neither authority | 79 | 9 | **-70** |

The exact replay cohort contained 74 described derived parents. Seventy had at
least one accepted child and received a content-addressed
`StructuralNameAuthority`; four were refused because none of their live
children were accepted. An immediate second invocation selected only those
four refusals and wrote **0** authorities.

Every one of the 70 newly written authorities has a non-empty `child_ids`
record exactly equal to its `ENTAILED_FROM_CHILD` edges, and every one still
has at least one accepted child. Thus no parent acquired authority without a
record naming the child closure from which it was entailed.

Representative records:

| Parent | Recorded child closure | Accepted grounding children |
|---|---|---|
| `average_external_magnetic_flux` | `current_weighted_average_external_magnetic_flux` | same |
| `vector_potential` | `perturbed_vector_potential`, `poloidal_vector_potential`, `radial_vector_potential` | `poloidal_vector_potential`, `radial_vector_potential` |

The mixed `vector_potential` family is intentional evidence of the contract:
the immutable record names the exact live topology, while acceptance grounding
is explicit rather than inferred from every child being terminal.

## Refusals

| Parent | Live child | Reason |
|---|---|---|
| `current_density_due_to_collisions` | `poloidal_current_density_due_to_collisions` | no accepted children |
| `effective_thermal_ion_charge_state_energy_velocity_due_to_convection` | `radial_effective_thermal_ion_charge_state_energy_velocity_due_to_convection` | no accepted children |
| `neutral_particle_convection_velocity` | `parallel_neutral_particle_convection_velocity` | no accepted children |
| `tritium_velocity` | `toroidal_tritium_velocity` | no accepted children |

These four remain inside the accepted-authority residual. The other five rows
in the residual are the already-measured quarantined names; this replay does
not convert invalid or unreadable state into permission.

## Integrity qualification

The graph now contains 331 structural-authority records, all 331 carrying
`id`, `accepted_name_id`, and `child_ids`. A graph-wide edge/property comparison
also found four **older** authorities whose recorded child closure no longer
equals their surviving `ENTAILED_FROM_CHILD` edges:

- `momentum_diffusion_coefficient`
- `energy_convection_velocity`
- `current_density_due_to_wave_driven_current_drive`
- `current_density`

All four were created on 2026-08-23, before this replay; none is among the 70
new records, whose exact-child result is 70 of 70. Repairing immutable historic
authority after a child identity disappears requires a separate governed
disposition and lies outside this node's fenced paths.

## Verification

The focused regression asserts that the replay persists exactly one signed
record naming its entailing child, returns that identity-bearing receipt, and
does not call persistence for a parent whose only child is non-accepted. It ran
with the existing atomic structural-authority persistence regressions.

Verbatim pytest summary:

```text
========================= 5 passed, 1 warning in 6.17s =========================
```

Full apply receipt and immediate replay:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T041225705936-n-structuralreplay/live-apply.log`.
