# Scalar-selected source deduplication evidence

Date: 2026-08-19

## Contract

`deduplicate_scalar_selected_sources` accepts an exact set of
`StandardNameSource` identities and reads their complete live binding and
backing-projection closure inside one caller-owned transaction. A row is
admitted only when `produced_sn_id` selects exactly one live binding and every
other live binding has exactly one matching signed `HAS_STANDARD_NAME`
projection. The preview signs all source, target, backing, origin,
`PRODUCED_NAME`, and projection properties and identities.

Apply requires that exact digest. It locks the full signed closure, recomputes
the manifest under those locks, and removes the non-selected bindings and
their projections through element-identity compare-and-set predicates. The
selected name and scalar are preserved, affected `source_paths` mirrors are
recomputed, and one deterministic `StandardNameChange` records the cohort.
Replay verifies the postcondition before returning `already_applied`; it rolls
back without acquiring write locks.

## Transactional regression evidence

The initial red proof exited during collection because the public instrument
did not yet exist. The final run used an isolated, auth-disabled Neo4j
2026.01.4 instance on `bolt://127.0.0.1:28687`, never the project graph:

```text
tests/standard_names/test_dual_binding_dedup.py ...                      [100%]
3 passed in 8.75s
```

The tests prove exact deletion of a redundant source binding and matching
backing projection, signed refusal when the scalar selects no live binding,
and byte-identical participant state plus an unchanged change-row census on
replay.

## Live preview

The input was exactly the 51 source identities in
`dual-binding-census.json` under `scalar_selected_dedup`. The call was
preview-only; its transaction rolled back and no apply entry point was
invoked.

| Field | Value |
|---|---:|
| Requested census sources | 51 |
| Admitted sources | 48 |
| Refused sources | 3 |
| Redundant bindings signed for removal | 57 |
| Matching projections signed for removal | 57 |
| Manifest SHA-256 | `dc2bfa35580cdc1449f16e1b49c8b75a82cb4e0863135b11879493ac735c1d2f` |
| Persistent writes | 0 |
| Provider calls | 0 |

The three refusals are named rather than skipped:

- `dd:distributions/distribution` selects
  `trapped_torque_density_due_to_j_cross_b_force`, but its current backing has
  no matching projections for the 25 non-selected live bindings.
- `dd:edge_transport/model/ggd` selects
  `total_momentum_convection_velocity`, but its current backing has no matching
  projections for the three non-selected live bindings.
- `dd:gyrokinetics_local/non_linear` selects
  `normalized_perturbed_magnetic_field`, but its current backing has no
  matching projection for
  `normalized_momentum_flux_due_to_e_cross_b_drift`.

Before and after the preview, the graph carried exactly 7,151
`StandardNameChange` nodes, 27,467 `LLMCost` nodes, and 489 `SNRun` nodes. All
three counters were byte-for-byte equal. The complete 51-row disposition and
signed hash are retained in `live-dry-run.log` in the worker run envelope.
