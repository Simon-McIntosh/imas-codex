NEEDS-HELP: all three sanctioned name-axis `sn edit --hint` dry runs refuse because every target has zero producing `StandardNameSource` edges, so no candidate can be regenerated or drawn.

tried: Queried the live graph for the three exact identities, their attached review history, producer closure, edit state, and the three nominated DD paths. Then invoked `sn edit <identity> --hint <DD-grounded text> --reason <DD-grounded reason> --axis name --scope self --dry-run` once for each identity. Every invocation exited 2 before mutation with the same observable refusal: `<identity> has no producing StandardNameSource ... a name-axis hint cannot regenerate it`. A postflight query confirmed zero counter or target-state change.

options: (1) add a separately sanctioned source-less name-hint transition that can compose a candidate from explicit, governed DD context without fabricating a producer; (2) authorize complete-replacement `sn edit --rename` proposals, which can enter ordinary review without a producer but changes the requested hint-mode mechanism; or (3) attach authoritative producers first through a new pre-acceptance admission route, which needs an explicit circularity-breaking contract because the existing attachment guard correctly requires an accepted target.

leaning: Option 1. The operator asked for steering rather than preselecting a spelling, and the graph deliberately refuses to attach these below-threshold targets. A source-less, DD-context-bearing hint transition preserves both facts while keeping grammar validation and the fresh quorum authoritative.

cost-if-wrong: Choosing option 2 preselects three spellings and may force collision/fold adjudication instead of testing semantic steering. Choosing option 3 weakens the accepted-lifecycle attachment boundary or requires later removal of provisional provenance. Either wrong route must be unwound and the three identities still need their first hint-driven quorum draw.

# Below-threshold hint-draw refusal

## Outcome

The requested measure is **blocked, with zero mutation and zero provider
spend**. All three identities were inspected live and each has exactly zero
incoming `PRODUCED_NAME` relationships from `StandardNameSource`. The public
`sn edit --hint` engine requires at least one such producing source for a
name-axis hint because its sanctioned action is to reset that source to
`extracted`, carry the edit `run_id`, regenerate a candidate, and send that
candidate to ordinary review. With no source, the CLI fails closed before it
stamps an edit.

The three dry-run invocations carried the mandatory reason and a hint grounded
in the nominated DD source semantics. They do not count as applied edits or
quorum draws. No non-dry-run invocation followed because the dry run proved the
same precondition would refuse before composition.

Quantitatively:

- applied hint edits: **0 of 3**;
- fresh quorum groups: **0 of 3**;
- identities drawn more than once: **0**;
- threshold: **0.85 before and after, unchanged**;
- `LLMCost`: **27,631 to 27,631** rows;
- `LLMCost.llm_cost`: **USD 1,366.843569 to USD 1,366.843569**;
- `StandardNameReview`: **20,777 to 20,777** rows;
- attributable spend: **USD 0.000000 of the USD 10.000000 cap**;
- name promotions: **0** by fresh quorum and **0** by every other route.

## Per-identity evidence

| Identity | Prior score | Producing sources | DD-grounded hint carried by the dry run | Resulting spelling | New score | Signed delta | Verdict at 0.85 |
|---|---:|---:|---|---|---:|---:|---|
| `minimum_of_safety_factor` | 0.61250 | 0 | Regenerate the shortest grammar-valid identity for the minimum value of the dimensionless safety-factor profile. Preserve the minimum aggregation and safety-factor quantity; do not introduce the position of the minimum or a rational-surface interpretation. | unchanged; no candidate generated | none | not defined; no draw | **blocked before draw; prior score remains 0.23750 below threshold** |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | 0.36250 | 0 | Regenerate the shortest grammar-valid identity for the effective poloidal convection velocity in the neutral internal-state momentum equation. Preserve poloidal direction, neutral state resolution, momentum transport, and convection velocity; do not substitute particle convection, momentum flux, or diffusivity. | unchanged; no candidate generated | none | not defined; no draw | **blocked before draw; prior score remains 0.48750 below threshold** |
| `toroidal_line_integrated_impurity_ion_velocity` | 0.45000 | 0 | Regenerate the shortest grammar-valid identity for the toroidal rotation velocity of an ion species measured at a charge-exchange recombination spectroscopy channel from the Doppler shift of charge-exchange emission. Preserve toroidal direction and ion velocity; do not retain `line_integrated` unless the DD source itself states an integration. | unchanged; no candidate generated | none | not defined; no draw | **blocked before draw; prior score remains 0.40000 below threshold** |

The signed score delta requested by the success measure cannot be calculated:
no new score exists. Reporting zero would be false because zero is a numeric
draw result; the honest value is **not defined because the gate refused before
review**.

## DD semantic inputs

The live DD records used to form the hints were:

| DD path | Live description | Unit and storage |
|---|---|---|
| `equilibrium/time_slice/global_quantities/q_min/value` | “Minimum q value and position.” | `1`; scalar and `HAS_UNIT` agree; `FLT_0D` |
| `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | Effective poloidal convection velocity for neutral-species momentum transport in radial profiles; the poloidal component of convective velocity in the neutral momentum equation. | `m.s^-1`; scalar and `HAS_UNIT` agree; `STRUCT_ARRAY` |
| `charge_exchange/channel/ion/velocity_phi` | Toroidal rotation velocity of an ion species at a charge-exchange recombination spectroscopy channel position, derived from the Doppler shift of charge-exchange emission. | `m.s^-1`; scalar and `HAS_UNIT` agree; `STRUCTURE` |

The third hint deliberately does **not** preserve `line_integrated`: the exact
selected DD source does not state an integration. This is an evidence-bearing
semantic correction, not reviewer persuasion toward the existing spelling.

## Live target and review state

Before and after the dry runs:

| Identity | Stage / validation | Stored score | Historical name-review rows | Open edit state |
|---|---|---:|---:|---|
| `minimum_of_safety_factor` | reviewed / valid | 0.61250 | 5 | none |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | reviewed / valid | 0.36250 | 5 | none |
| `toroidal_line_integrated_impurity_ion_velocity` | drafted / valid | 0.45000 | 2 | stale open hint requested 2026-07-30; unchanged |

The stale toroidal edit predates this node and has no `run_id`, producer,
claim, or fresh review. Its hint is grounded in a different X-ray spectroscopy
path. This node neither overwrote nor retired it because the dry-run source
precondition refused first and the write fence does not authorize graph-state
repair beyond the requested sanctioned invocations.

## Exact refusal

Each CLI invocation returned the identity-specific version of:

```text
<identity> has no producing StandardNameSource (it is a derived/structural
name) — a name-axis hint cannot regenerate it. Use --rename to propose a
replacement name, or --axis docs to steer only its documentation.
```

`--axis docs` cannot satisfy this node: it would draw the documentation axis,
not produce a resulting spelling or a name score against 0.85. `--rename`
also cannot be substituted silently because the requested experiment is
whether DD-grounded steering produces a better identity, not whether an
operator-selected complete replacement can pass review.

## Completion boundary

The node stops at the first exact unmet condition, as required. It did not
alter a threshold, attach provisional sources, invoke raw Cypher mutation,
promote a name, overwrite the stale edit, call a provider, or draw any identity.
Completion requires a new authority/capability decision selecting one of the
three options in the `NEEDS-HELP` preamble.
