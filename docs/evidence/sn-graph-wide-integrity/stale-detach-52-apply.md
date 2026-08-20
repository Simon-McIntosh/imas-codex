NEEDS-HELP: The authorized 52-row stale-source cohort no longer matches its signed live closure, so both permitted production attempts failed closed before preview or mutation.

tried: Read live plan version 199 and the widened operator at commit `2d000336`; verified the committed 58-row authority file and canonical rows digest; derived the live signed partition; probed all named non-actions; then made two production-driver attempts. Attempt one stopped before preview because live topology now classifies the signed cohort as 3 already detached, 53 currently admissible and 2 current last-producer refusals: `neutral_state_energy_convection_velocity` acquired two reviewed live `HAS_PARENT` children after the prior admission snapshot. Attempt two conservatively retained the user-authorized 3/52/3 mutation fence, proved the other two exact operator refusals by target name, and then stopped when the widened operator raised `StaleSourceDetachConflict: signed source closure changed for dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance`. Both operator transactions rolled back before a 52-row preview or apply.

options: First, independently adjudicate the changed `closest_wall_point/distance` source closure and the two new children of `neutral_state_energy_convection_velocity`, then issue a refreshed signed cohort and redispatch one serialized apply. Second, retain the original 52-row membership but refresh and sign its exact current source/binding/projection closure, explicitly recording the now-conservative third non-action. Third, amend the operator only if independent review proves its closure comparison is mis-normalizing unchanged graph state; do not bypass or relax the current compare-and-set.

leaning: Re-adjudicate and re-sign against the current graph before any new apply. Two independent pieces of live topology drift are now visible, and the whole purpose of the signed closure is to make that drift a refusal rather than silently reuse stale authority.

cost-if-wrong: Reusing the stale manifest or weakening the closure check could detach a source whose binding, projection or structural authority changed after adjudication. Re-signing the wrong 52/3 partition would require another baseline, preview, collateral digest and receipt; widening to 53 without explicit authority would mutate a row the node was not authorized to apply.

# Signed stale-source production apply hold

Date: 2026-08-20

## Outcome

No production mutation occurred. Both attempts failed before the required
52-row preview was returned, so neither reached the apply transaction,
`StandardNameChange` mutation, replay, or independent postflight. The operator
uses a transaction wrapper and rolled back each raised conflict.

The requested quantitative completion measure is therefore **not met**:

| Measure | Required | Observed |
|---|---:|---:|
| Signed partition | 3 already detached + 52 applied + 3 refused = 58 | 3 already detached + 52 fixed cohort + 3 named non-actions derived, but no apply |
| Apply receipt rows | 52 | 0 |
| Immediate replay | `already_applied`, changed=0 | not reached |
| `StandardNameChange` delta | exactly +52 | 0 attributable to this node |
| `LLMCost` delta | 0 | 0 attributable to this node |
| Out-of-allowlist closure proof | every row digest identical | not reached for an apply |
| Independent dual-bound/unsourced postflight | required | not reached |

## Authority verified

- Authority: `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json`
- File SHA-256:
  `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`
- Canonical rows SHA-256:
  `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Signed rows: **58**
- Previously applied exact receipts: **3**, each re-read with null scalar,
  zero live binding, zero matching projection and the authority rows digest.

## First refusal: the current graph no longer reproduces 52/3

The first driver derived the partition from current live producers and direct
children. It found **3 already detached + 53 currently admissible + 2 current
last-producer refusals = 58** and stopped before preview.

The changed row is
`derived:neutral_state_energy_convection_velocity`. Its stale source still
binds both `neutral_state_energy_convection_velocity` and
`neutral_internal_state_energy_convection_velocity`, but the scalar target now
has two live reviewed children:

- `poloidal_neutral_state_energy_convection_velocity`
- `radial_neutral_state_energy_convection_velocity`

The public operator's current single-row preview consequently returns
`would_apply`, with 2 bindings and 0 projections, rather than the earlier
last-producer refusal. The node did not widen its authority to 53. The second
attempt retained `neutral_state_energy_convection_velocity` as a conservative
named non-action so the requested 52-row cohort remained fixed.

The other two rows still refuse by exact target name through the widened public
operator:

- `dd:ece/channel/t_e_voltage` —
  `detach would orphan target voltage_of_diagnostic_antenna`
- `dd:equilibrium/time_slice/profiles_1d/b_average` —
  `detach would orphan target flux_surface_average_magnetic_field_magnitude`

## Second refusal: admitted source closure drift

With the production mutation fence held at exactly 52 rows, the operator
re-read the complete source/binding/projection closure before creating a
manifest and raised:

```text
StaleSourceDetachConflict: signed source closure changed for dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance
```

This is an authority conflict, not an ordinary retry condition. The invocation
did not obtain a manifest digest and therefore could not legally enter apply.
No third attempt was made, because the node contract requires a stop after the
same production command fails twice with different fixes attempted.

## Durable evidence

- First fail-closed production attempt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T163749529305-sgwi-detach-52-apply/production-apply.log`
- Exact neutral scalar/binding probe:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T163749529305-sgwi-detach-52-apply/neutral-scalar-probe-second.log`
- Fixed 3/52/3 partition and current-topology annotation:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T163749529305-sgwi-detach-52-apply/live-partition.json`
- Exact refusal probes:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T163749529305-sgwi-detach-52-apply/refusal-probes.json`
- Second fail-closed production attempt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T163749529305-sgwi-detach-52-apply/production-apply-second.log`
