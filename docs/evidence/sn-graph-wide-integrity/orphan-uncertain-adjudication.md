# Physics adjudication of the uncertain unsourced identities

## Outcome

The eight identities classified as `UNCERTAIN` by the reverse Data Dictionary
search have been adjudicated individually: **3 ATTACH + 4 HOLD + 1 RETIRE = 8**.
The dispositions are exclusive, and all eight identities are unique.

`ATTACH` means that the physics, DD hierarchy, and unit now identify a specific
candidate path. It is scientific disposition evidence, not graph-mutation
authority: the candidate must still pass the separately authorized exact
attachment workflow and its closure guards. `HOLD` means preserve the identity
without attachment or removal until the named DD ambiguity is resolved.
`RETIRE` is likewise an adjudicated recommendation rather than apply authority;
any removal still owes the plan's exact signed-manifest and release-provenance
guards.

Authority inputs:

- live `sn-graph-wide-integrity` plan version 219, SHA-256
  `10c8d1c2acc35fed33d5733c00fcf7c431d5d190713422858a1e1fbfcbb6490b`;
- `orphan-resourcing-search.md`, SHA-256
  `4dea85aacb5d1e786711bfca68d578644f44964ddf8eb947a328b626a1f99759`;
- read-only DD 4 path detail, hierarchy, sibling, and search results from
  `fetch_dd_paths`, `list_dd_paths`, `check_dd_paths`, and `search_dd_paths`.

## Per-identity disposition

| Row | Standard name | Disposition | DD path or exact negative query | Unit evidence | Physics adjudication |
|---|---|---|---|---|---|
| U1 | `fast_ion_charge_state_power_at_inside_flux_surface` | HOLD | Candidate: `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` | name `W`; DD `W` — agreement | The hierarchy is charge-state resolved and the field suffix says `fast`, but the DD prose says power deposited into the **thermal** ion population. A distinct sibling, `power_inside_thermal`, exists and carries the same thermal wording. Unit and spelling cannot resolve this contradiction. Preserve the useful fast-ion quantity, but do not attach until DD authority confirms which recipient population `power_inside_fast` stores and corrects one side of the contradiction. |
| U2 | `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` | ATTACH | `equilibrium/time_slice/profiles_1d/gm7` | name `1`; DD-derived `1` — agreement. DD 4.1 documentation defines `gm7` as the flux-surface average of `\|grad rho_tor\|`; `rho_tor` is stored in `m`, so `grad rho_tor` has `m/m = 1`. | `gm7`, not `gm3`, is the unsquared metric: `gm3` averages `\|grad rho_tor\|^2`. The accepted name states the unsquared flux-surface-averaged magnitude and therefore matches `gm7`. The absent literal unit on `gm7` is a DD metadata omission, not a dimensional disagreement. |
| U3 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | ATTACH | `edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel` | name `m.s^-1`; DD `m.s^-1` — agreement | The earlier search stopped at the vector structure. Its explicit `parallel` child resolves the missing projection and is nested beneath neutral `state` and `velocity_diamagnetic`, so owner, internal-state resolution, mechanism, and field-aligned component all agree. The parallel sibling in `plasma_profiles` is semantically equivalent, but the listed edge-profile path is the adjudicated candidate for this row. |
| U4 | `poloidal_neutral_internal_state_convection_velocity` | ATTACH | `edge_transport/model/ggd/neutral/state/particles/v_pol` | name `m.s^-1`; DD `m.s^-1` — agreement | The accepted documentation defines this unadorned convection velocity as the coefficient in the **particle-number** transport equation. DD 4.1 explicitly changed the candidate description to “Particle effective convection” and separately provides `momentum/v_pol`. That separation resolves the earlier owner ambiguity: the `particles/v_pol` leaf is exact, while `momentum/v_pol` belongs to the distinct momentum-convected identity. |
| U5 | `toroidal_ion_charge_state_torque_density` | HOLD | Candidate: `plasma_sources/source/ggd/ion/momentum/phi`; competing candidate: `distributions/distribution/profiles_1d/collisions/ion/state/torque_thermal_phi` | name `kg.m^-1.s^-2`; both DD candidates are algebraically `N.m^-2 = kg.m^-1.s^-2` — agreement | No path carries the whole identity. The plasma-source `phi` leaf is a net toroidal momentum source but is ion-species resolved, not charge-state resolved. The distributions leaf is charge-state resolved but only the collisional transfer to the thermal population, not the process-total torque. The generic total charge-state torque is physically meaningful as an aggregate, so preserve it, but neither incomplete candidate may be attached. |
| U6 | `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | RETIRE | Exact query A: `collisional toroidal torque density delivered to trapped thermal ions in a specified charge state`; exact query B: `toroidal collision torque on trapped thermal ion population resolved by charge state` | All returned torque-density candidates use `m^-2.N`, algebraically agreeing with name `N.m^-2`; the failure is semantic, not dimensional. | The nearest thermal leaf, `distributions/distribution/profiles_2d/trapped/collisions/ion/state/torque_thermal_phi`, is torque delivered to the **background thermal ion population by a trapped non-Maxwellian distribution**. Here `trapped` classifies the source distribution, whereas `thermal` classifies the recipient. The accepted identity and documentation instead make both modifiers describe the recipient (“trapped thermal ions”). The high-scoring `torque_fast_tor` sibling targets trapped fast ions and is also wrong. The two exact searches found no path for the accepted source/recipient meaning. Retire this conflated identity; any replacement must express the two roles without collapsing them. |
| U7 | `x_direction_unit_vector_of_sensor` | HOLD | Exact query A: `dimensionless x component direction cosine of sensor orientation unit vector`; exact query B: `sensor orientation x direction cosine with unit one` | name `1`; only DD scalar candidate `operational_instrumentation/sensor/direction/x` is `m` — disagreement | The name is physically correct: the x projection of a normalized direction vector is a direction cosine in `[-1, 1]` and is dimensionless. The DD parent explicitly says **unit vector**, while its x child says “X coordinate of direction” with unit `m`; the two DD levels contradict each other. Both exact searches returned the same metre-valued child and no dimensionless alternative. Preserve the name and hold attachment until the DD child is corrected to unit `1` or the parent is redefined as a metric displacement. |
| U8 | `z_direction_unit_vector_of_sensor` | HOLD | Exact query A: `dimensionless z component direction cosine of sensor orientation unit vector`; exact query B: `sensor orientation vertical direction cosine with unit one` | name `1`; only DD scalar candidate `operational_instrumentation/sensor/direction/z` is `m` — disagreement | The same resolution applies on z: a component of the documented sensor direction **unit vector** is dimensionless, not a height. Both exact searches returned the metre-valued `direction/z` child and no unit-agreeing alternative. Preserve the name and hold attachment until DD corrects the scalar to unit `1` or explicitly changes the structure's physical meaning. |

## Sensor-direction unit resolution

The `1` versus `m` disagreement is resolved in favor of the Standard Names'
dimensionless unit **for the physics**, but not by overriding DD authority at
attachment time. A vector described as a unit vector satisfies

\[
\hat{\mathbf d}\cdot\hat{\mathbf d}=1,
\]

so each Cartesian projection is a dimensionless direction cosine. Metres would
instead describe a displacement or a point coordinate. DD 4 currently combines
the unit-vector parent definition with metre-valued, coordinate-worded scalar
children; that is an internal DD defect or unresolved modeling change. Therefore:

1. do not change either Standard Name's unit from `1` to `m`;
2. do not attach either name while the ordinary unit guard sees `1 != m`;
3. do not retire either identity, because the mismatch is exactly the kind of
   unit disagreement the plan forbids using as removal authority; and
4. after DD corrects the children to `1` (or otherwise clarifies the structure),
   re-run the exact attachment adjudication for `direction/x` and `direction/z`.

## Quantitative closure

The validation command below parses only the eight `U1`–`U8` adjudication rows,
checks identity uniqueness, and sums the allowed dispositions. Its recorded
output is:

```text
rows=8
unique=8
attach=3
hold=4
retire=1
disposition_sum=8
sensor_unit_resolution=present
PASS
```

No graph mutation, provider call, name acceptance, attachment, retirement, or
deletion was performed.
