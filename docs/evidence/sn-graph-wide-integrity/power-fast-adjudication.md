# Power-fast dual-binding adjudication

## Disposition

Keep the DD source
`dd:waves/coherent_wave/global_quantities/ion/state/power_fast` bound only to
`fast_ion_charge_state_absorbed_wave_power`.

Remove its direct `PRODUCED_NAME` and backing `HAS_STANDARD_NAME` projections to
`ion_charge_state_power`, and set the source's `produced_sn_id` scalar to
`fast_ion_charge_state_absorbed_wave_power`. Retain
`ion_charge_state_power` as the derived structural parent in the Standard Name
family; it must not remain a second direct realization of this DD leaf. No new
identity is required.

This is a semantic deduplication outcome, but it should **not** be executed by
first hand-setting the scalar and then calling
`deduplicate_scalar_selected_sources`. The sanctioned existing operation for
this initially scalar-null row is the explicit-authority path of
`repair_semantic_source_invariants`, which selects the scalar, removes the
competing live edge and backing projection, rebuilds affected `source_paths`,
and writes its change record in one compare-checked transaction.

## Live evidence supplied by the coordinator

| Object | Stage / origin / score | Unit | Live semantics |
|---|---|---|---|
| `dd:waves/coherent_wave/global_quantities/ion/state/power_fast` | Source status `composed`; `produced_sn_id = NULL` | `W` | DD documentation: “Wave power absorbed by the fast particle population.” |
| `fast_ion_charge_state_absorbed_wave_power` | `accepted`; origin `pipeline`; name-review score `0.8625` | `W` | “Volume-integrated electromagnetic-wave power transferred directly to fast ions in one specified ion charge state, excluding collisional energy exchange.” |
| `ion_charge_state_power` | `accepted`; origin `derived`; name-review score `None` | `W` | “Volume-integrated power transferred from a coherent electromagnetic wave to an ion population, with population resolution and mode aggregation defined by each…” |

The earlier machine census adds that both targets are valid and live, the DD
4.1.1 upstream node is present, there is no scalar-selected target, and there is
no `REFINED_FROM` connection between the two target identities.

## Physics argument

The DD wording makes `fast` load-bearing. “Fast particle population” selects a
non-thermal energetic sub-population; it is not interchangeable with the
broader “ion population” in `ion_charge_state_power`. Losing that selector
would combine power absorbed by fast ions with power assigned to an otherwise
unspecified ion population.

The rest of the specific identity is also supported by the path and unit:

- `waves/coherent_wave` supplies the electromagnetic-wave transfer mechanism.
- `global_quantities` and unit `W` support a volume-integrated power rather than
  a local power density.
- `ion/state` supplies resolution by ion charge state.
- `power_fast` and the DD documentation supply the fast-ion population
  qualifier.
- Placement under the coherent-wave IDS branch supports excluding collisional
  energy exchange: the leaf records direct wave absorption, not subsequent
  ion-ion or ion-electron redistribution.

`fast_ion_charge_state_absorbed_wave_power` preserves all of those axes.
`ion_charge_state_power` deliberately omits the fast-population qualifier and
is therefore useful as a family abstraction, not as the leaf's direct live
identity. Its `derived` origin and absent review score are consistent with that
structural role, although the decision rests on DD physics rather than origin
or score. Keeping the parent relationship preserves discoverability and
hierarchy without violating the locked one-live-name-per-source invariant.

## Sanctioned execution contract

### Why scalar-selected dedup cannot be the first operation

`deduplicate_scalar_selected_sources` admits a row only when
`produced_sn_id` selects exactly one of at least two live bindings. The current
source has two live bindings and `produced_sn_id = NULL`, so its dry-run must
refuse with `produced_sn_id does not select exactly one live binding`.

A raw Cypher `SET source.produced_sn_id = ...` is not sanctioned. There is also
no general scalar-only governed instrument for an ordinary DD source that
would create a safe intermediate state for this row. Using
`repair_semantic_source_invariants` merely as a scalar preparer is impossible:
its apply transaction already performs the complete deduplication. A subsequent
`deduplicate_scalar_selected_sources` call would find fewer than two live
bindings and correctly refuse because there is nothing left to deduplicate.

### Exact preparatory dry-run

The preparatory step is the non-mutating explicit-authority dry-run below. The
coordinator-supplied adjudication is expressed as the one-entry
`authority_overrides` map; the instrument must re-read the exact current
source, both live targets, target stages and validation states, DD backing,
exclusive backing ownership, and backing projections before planning anything.

```python
source_id = "dd:waves/coherent_wave/global_quantities/ion/state/power_fast"
target_id = "fast_ion_charge_state_absorbed_wave_power"
reason = (
    "DD 4.1.1 documents wave power absorbed by the fast particle population; "
    "waves/coherent_wave supplies direct electromagnetic-wave transfer, "
    "global_quantities with unit W supplies volume integration, and ion/state "
    "supplies charge-state resolution. Keep the accepted specific fast-ion "
    "identity; retain ion_charge_state_power only as its derived structural parent."
)

preview = repair_semantic_source_invariants(
    gc,
    [source_id],
    reason=reason,
    dry_run=True,
    authority_overrides={source_id: target_id},
    origin="semantic_source_repair",
    run_id=None,
)
```

Required dry-run result:

- exactly one `planned` row and zero `ambiguous` rows;
- `authoritative_target = fast_ion_charge_state_absorbed_wave_power`;
- `authority_basis = explicit_authority_override`;
- `removed_targets = ["ion_charge_state_power"]`;
- after-state live and produced targets contain only the specific identity;
- after-state `produced_sn_id` and backing projection both select the specific
  identity;
- no graph, cost, review, or change counter moves during the dry-run.

Any different target set, stage, validation state, backing owner, DD backing,
or projection closure is drift and must refuse the future apply.

### Exact future apply under separate mutation authority

Only a graph-mutation node may execute the same inputs with the sole change
`dry_run=False`:

```python
applied = repair_semantic_source_invariants(
    gc,
    [source_id],
    reason=reason,
    dry_run=False,
    authority_overrides={source_id: target_id},
    origin="semantic_source_repair",
    run_id=None,
)
```

The operator's transaction must atomically:

1. delete only the source-to-`ion_charge_state_power` live binding;
2. retain/merge only the source-to-specific live binding;
3. set `produced_sn_id` to the specific identity;
4. replace the DD backing projection with the specific identity only;
5. rebuild both affected names' `source_paths` from surviving graph authority;
6. create one `repair_semantic_source_binding` change record containing the
   before/after closure and the exact reason above.

Postflight must show one live target, one matching scalar, one matching DD
projection, the parent identity still live as a derived structural parent, and
no unrelated graph/cost/review changes. Repeating the exact dry-run must return
the row as `already_clean`, with no planned mutation.

## Read-only proof and quantitative completion

- DD sources adjudicated: **1 of 1**.
- Live targets enumerated: **2 of 2**.
- Target stage/origin/score/unit sets recorded: **2 of 2**.
- Chosen direct identity: **1** specific accepted target.
- Direct bindings to remove: **1** derived-parent binding.
- New identities required: **0**.
- Graph write calls issued by this node: **0**.
- Repository paths changed by this node: **0**.
- Assigned-repository `git status --porcelain`: **no output** before writing
  this run-directory record.

The earlier sandbox read failure is superseded by the coordinator-supplied live
evidence above. This node performed only repository/code inspection and wrote
only its run-directory evidence and manifest.
