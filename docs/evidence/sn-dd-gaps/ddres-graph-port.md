# DD resolution graph port evidence

Date: 2026-08-18

The packaged active authority was ported atomically into the live graph. Each
of its 37 records now has one `DDResolution` node, one incoming `BRIDGED_BY`
edge from the exact corrected `IMASNode`, one outgoing `EVIDENCED_BY` edge to
the exact `DDGap`, and one `FOR_DD_VERSION` edge. The transaction corrected
the `unit` scalar and sole `HAS_UNIT` edge for 30 nodes whose live value still
equaled the published value. Seven nodes already carried the effective value
and received provenance edges only.

The mutation compared both the live scalar and sole unit edge with the
published value at write time. Any missing row, extra edge, or divergent value
causes the entire transaction to roll back and reports every mismatching path.

## Receipts

| Run | Writes | Corrected | Edge-only | Unchanged | Nodes | Bridges | Evidence edges | Version edges | Receipt hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Apply | 37 | 30 | 7 | 0 | 37 | 37 | 37 | 37 | `sha256:de770461e0b497a8cb656b7867cac26accdb6a9b5d9223126b2eedcba9fd7c22` |
| Replay | 0 | 0 | 0 | 37 | 37 | 37 | 37 | 37 | `sha256:29f4b1b5ba2f3cb862ad6945109c3633fc7233703f423fafbd98d7ba5d87bf1b` |

The apply receipt contains the resolution identity, exact path, canonical
published value, canonical effective value, and action for every row. Its
per-path dispositions are reproduced below.

| Action | Exact DD path | Published | Effective |
| --- | --- | --- | --- |
| corrected | `operational_instrumentation/sensor/direction/z` | `m` | `1` |
| corrected | `wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/incident/values` | `m^-2.s^-1` | `W.m^-2` |
| edge-only | `plasma_profiles/ggd/ion/state/ionisation_potential` | `e` | `eV` |
| corrected | `spi/injector/injection_direction/x` | `m` | `1` |
| corrected | `ec_launchers/mirror/movement/direction/z` | `m` | `1` |
| corrected | `operational_instrumentation/sensor/direction_second/y` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/unit_vector_minor/x` | `m` | `1` |
| corrected | `ec_launchers/mirror/movement/direction/y` | `m` | `1` |
| corrected | `ec_launchers/mirror/movement/direction/x` | `m` | `1` |
| corrected | `operational_instrumentation/sensor/direction/y` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/unit_vector_major/x` | `m` | `1` |
| corrected | `camera_ir/channel/camera/direction/z` | `m` | `1` |
| corrected | `camera_ir/channel/camera/direction/y` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/unit_vector_minor/y` | `m` | `1` |
| corrected | `operational_instrumentation/sensor/direction_second/z` | `m` | `1` |
| corrected | `camera_ir/channel/camera/up/x` | `m` | `1` |
| corrected | `wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/state/incident/values` | `m^-2.s^-1` | `W.m^-2` |
| corrected | `spi/injector/shatter_cone/direction/z` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/unit_vector_major/y` | `m` | `1` |
| corrected | `spi/injector/injection_direction/z` | `m` | `1` |
| edge-only | `edge_profiles/ggd/ion/state/ionisation_potential` | `e` | `eV` |
| edge-only | `equilibrium/time_slice/constraints/pressure_rotational/reconstructed` | `1` | `Pa` |
| edge-only | `equilibrium/time_slice/constraints/pressure/reconstructed` | `1` | `Pa` |
| edge-only | `equilibrium/time_slice/constraints/n_e/reconstructed` | `1` | `m^-3` |
| edge-only | `equilibrium/time_slice/constraints/j_phi/reconstructed` | `1` | `A.m^-2` |
| corrected | `spi/injector/shatter_cone/unit_vector_minor/z` | `m` | `1` |
| corrected | `camera_ir/channel/camera/up/z` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/unit_vector_major/z` | `m` | `1` |
| corrected | `camera_ir/channel/camera/up/y` | `m` | `1` |
| corrected | `operational_instrumentation/sensor/direction/x` | `m` | `1` |
| corrected | `spi/injector/injection_direction/y` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/direction/x` | `m` | `1` |
| corrected | `operational_instrumentation/sensor/direction_second/x` | `m` | `1` |
| corrected | `spi/injector/shatter_cone/direction/y` | `m` | `1` |
| edge-only | `equilibrium/time_slice/constraints/j_parallel/reconstructed` | `1` | `A.m^-2` |
| corrected | `camera_ir/channel/camera/direction/x` | `m` | `1` |
| corrected | `wall/description_ggd/ggd/energy_fluxes/recombination/neutral/incident/values` | `m^-2.s^-1` | `W.m^-2` |

## Verification

- The credentialed schema-compliance suite ran all 9 tests: 9 passed, 0
  skipped, 0 failed.
- The focused resolution suite includes an observed-value mismatch and an
  incomplete write-time compare-and-set refusal; both expose the exact path.
- The active manifest remained
  `8b124a8ff7a040e639e5b773a07fae63d8b4135063f4af4ce82f7ce2d5286851`.
- The candidate resource remained
  `c6ee52aedd65cad1fa42c539661a127fffaa6bb2d25e87808f5fda9db35cd4b1`.
