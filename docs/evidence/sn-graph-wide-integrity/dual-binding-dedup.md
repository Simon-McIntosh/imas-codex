# Scalar-selected source deduplication evidence

Date: 2026-08-19

## Contract

`deduplicate_scalar_selected_sources` accepts an exact set of
`StandardNameSource` identities and reads their complete live binding and
backing-projection closure inside one caller-owned transaction. A row is
admitted only when `produced_sn_id` selects exactly one live binding and every
other live binding has exactly one matching signed `HAS_STANDARD_NAME`
projection. A caller may retain a known zero-projection row only through a
non-empty per-source exclusion reason signed into the same manifest; duplicate
projection authority and every other refusal remain blocking. The preview signs
all source, target, backing, origin,
`PRODUCED_NAME`, and projection properties and identities.

Apply requires that exact digest. It locks the full signed closure, recomputes
the manifest under those locks, and removes the non-selected bindings and
their projections through element-identity compare-and-set predicates. The
selected name and scalar are preserved, affected `source_paths` mirrors are
recomputed, and one deterministic `StandardNameChange` records the cohort.
Replay verifies the postcondition before returning `already_applied`; it rolls
back without acquiring write locks.

## Transactional regression evidence

The initial instrument red proof exited during collection because the public
instrument did not yet exist. The apply-gate red proof then executed the two
new projection refusal regressions and failed both because the refusal did not
distinguish absent authority from duplicate authority. The final run used an
isolated, auth-disabled Neo4j
2026.01.4 instance on `bolt://127.0.0.1:28687`, never the project graph:

```text
tests/standard_names/test_dual_binding_dedup.py ......                   [100%]
6 passed in 6.62s
```

The six tests prove exact deletion of a redundant source binding and matching
backing projection, scalar-disagreement refusal, distinct zero-projection and
duplicate-projection refusals, signed exclusion of a zero-projection row while
another cohort row applies, and byte-identical participant state plus an
unchanged change-row census on replay.

## Live apply and replay

The input was exactly the 51 source identities in
`dual-binding-census.json` under `scalar_selected_dedup`. The preview was
regenerated after the gate changes, and apply used that exact hash.

| Field | Value |
|---|---:|
| Requested census sources | 51 |
| Admitted sources | 48 |
| Refused sources | 3 |
| Redundant bindings signed for removal | 57 |
| Matching projections signed for removal | 57 |
| Manifest SHA-256 | `0c9fe0fdb92ed49a43b81526e4f005b2dd66053595b70948b3dc6c437808a030` |
| Applied SHA-256 | `0c9fe0fdb92ed49a43b81526e4f005b2dd66053595b70948b3dc6c437808a030` |
| Sources deduplicated | 48 |
| Change receipt | `sn-change:scalar-selected-dedup:0c9fe0fdb92ed49a43b81526e4f005b2dd66053595b70948b3dc6c437808a030` |
| Admitted sources with one live binding after apply | 48 / 48 |
| Replay outcome | `already_applied`, `changed=0` |
| Replay persistent writes | 0 |
| Provider calls | 0 |

The three exclusions are signed into the 51-row manifest rather than silently
skipped. Their backing nodes have no projection identity through which the
operator could exact-CAS remove the competing source edge, so fabricating or
guessing projection authority was rejected:

- `dd:distributions/distribution` — excluded with reason: backing carries no
  projection identities for 25 non-selected live bindings. It selects
  `trapped_torque_density_due_to_j_cross_b_force`, but its current backing has
  no matching projections for the 25 non-selected live bindings.
- `dd:edge_transport/model/ggd` — excluded with reason: backing carries no
  projection identities for three non-selected live bindings. It selects
  `total_momentum_convection_velocity`, but its current backing has no matching
  projections for the three non-selected live bindings.
- `dd:gyrokinetics_local/non_linear` — excluded with reason: backing carries no
  projection identity for the non-selected live binding. It selects
  `normalized_perturbed_magnetic_field`, but its current backing has no
  matching projection for
  `normalized_momentum_flux_due_to_e_cross_b_drift`.

Apply increased `StandardNameChange` from 7,151 to 7,152 and left `LLMCost` and
`SNRun` flat at 27,467 and 489. Immediately before and after replay, all three
counters were identical at 7,152 / 27,467 / 489 and the exact participant
snapshot hash was identical at
`d5c2d6c88b92be0c1c13bd7d11cd70739fd280672274c4cbfb65fa1077b440b6`.
The complete preview, apply receipt, replay proof, and 51-row post-apply census
are retained in `live-apply.log` in the worker run envelope.
