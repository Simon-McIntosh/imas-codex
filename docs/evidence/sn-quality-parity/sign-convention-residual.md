# Conditional sign-convention residual closure

Snapshot: 2026-08-25, live `codex` graph. Status: **blocked after a partial,
governed repair**. No model call was made by this node.

## Outcome

The three identities that had become accepted outside the frozen 280-row
repair were re-diagnosed against fresh live DD authority and repaired under
their own signed deterministic manifest:

| Identity | Fresh authority | Exact change | Model spend |
|---|---|---|---:|
| `length_of_poloidal_magnetic_field_probe` | `magnetics/b_field_pol_probe/length`, whose DD `cocos_transformation_type` is null | Removed only the unsupported final sign-convention paragraph; retained `one_like`, scalar COCOS 17, and the single `HAS_COCOS` edge | USD 0.00 |
| `radial_coordinate_at_inboard_midplane` | `equilibrium/time_slice/profiles_1d/r_inboard`, whose DD `cocos_transformation_type` is null | Same bounded paragraph removal and metadata preservation | USD 0.00 |
| `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | `spectrometer_visible/channel/isotope_ratios/isotope/density_ratio`, whose DD `cocos_transformation_type` is null | Same bounded paragraph removal and metadata preservation | USD 0.00 |

The manifest SHA-256 is
`dff5809f306f2f3662d21376cb9e645d6ed4e38e0ff4a5b8b4686370b6eed0ea`.
It admitted and changed 3/3 rows with zero refusals, removed 463 characters in
total with a maximum per-document delta of 198 characters, and wrote durable
change id
`sn-change:sign-convention-document-repair:dff5809f306f2f3662d21376cb9e645d6ed4e38e0ff4a5b8b4686370b6eed0ea`.
All three documents remain accepted at their pre-repair documentation scores:
0.875, 0.91875, and 0.88125 respectively. The node-scoped `LLMCost` census is
0 calls / USD 0.00.

Representative bounded change: the probe-length document previously ended
with “Sign convention: Positive when the probe extent is measured along the
assigned local normal sensing-axis direction.” The repair removed that final
paragraph and no preceding character. The accepted identity and its
DD-authoritative source binding are unchanged.

## `magnetic_field` authority blocker

Fresh structural evaluation reproduces the diagnosis exactly: `magnetic_field`
has 16 children, 15 survive the production eligibility rule, and their sole
non-null transformation class is `b0_like`, supplied by
`vacuum_magnetic_field`. The accepted node, however, is `origin=catalog_edit`
and still carries `one_like`, scalar COCOS 17, one `HAS_COCOS` edge, and the
generic component-direction paragraph. Its existing documentation score is
0.91875, but that score does not repair the physics mismatch.

The sanctioned edit path was previewed successfully with self-only scope. It
would regenerate the documentation and route it through the ordinary
generate-docs, review-docs, and refine-docs quorum using explicit `b0_like`
guidance. Mutation did not begin because the sanctioned structural
materializer's authority query returned zero rows: that query admits only
parents whose origin is null or `derived`, while this identity is
`catalog_edit`. Hand-written Cypher would bypass the required authority path,
so it was not used. Model spend for this half is therefore 0 calls / USD 0.00,
and no quorum verdict exists yet.

## Graph-wide gate

The documented post-frozen-repair baseline was **325 pass / 4 fail / 2,411
not-evaluable** across 2,740 accepted documents. This node closed the three
named invariant rows, but the live accepted population changed concurrently.
The final rescore available to this node is **331 pass / 9 fail / 2,528
not-evaluable** across 2,868 accepted documents. The gate therefore remains
**failed**, not closed.

The nine current failures are:

- `magnetic_field` — the expected `b0_like` regeneration row;
- `change_in_ion_state_mean_ionisation_potential`;
- `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift`;
- `parallel_current_density_due_to_ohmic_current_drive`;
- `x_direction_unit_vector_of_electron_cyclotron_launcher_mirror`;
- `y_direction_unit_vector_of_electron_cyclotron_launcher_mirror`;
- `z_direction_unit_vector_of_camera`;
- `z_direction_unit_vector_of_electron_cyclotron_launcher_mirror`;
- `z_direction_unit_vector_of_pellet_injector`.

The last eight were not in the four-identity node specification. They entered
the accepted population while the concurrent documentation-backlog drain was
still moving the graph: accepted-document coverage rose from 2,800 at this
node's first live census to 2,868 at its final census. Each new failure needs a
fresh authority diagnosis and signed disposition after that producer is
quiescent; silently expanding the frozen cohort during an active drain would
not provide stable closure authority.

## Required continuation

NEEDS-HELP: the exact four-row closure is no longer a stable or fully
authorized mutation set.

- tried: repaired the three invariant rows under a fresh signed manifest, then
  invoked the sanctioned structural materializer for `magnetic_field`; after
  correcting a missing runtime import, its exact authority query returned zero
  rows because the node is `origin=catalog_edit`, and the live gate concurrently
  expanded to nine failures.
- options: (1) wait for the documentation drain to quiesce, re-diagnose every
  remaining failure, authorize the resulting signed deterministic cohort, and
  add a scoped signed metadata transition for `magnetic_field` before its
  ordinary quorum; (2) extend the sanctioned structural materializer to cover
  this catalog-edited structural-parent case, which requires source and test
  changes outside this node's fence; or (3) authorize a one-off governed
  metadata operator for `magnetic_field` while retaining ordinary quorum for
  its prose, then separately dispatch the new invariant cohort.
- leaning: option 1, using a governed one-row metadata transition rather than
  broadening the parent materializer. It closes against a quiescent population,
  preserves catalog-edit protection, and keeps prose acceptance on the ordinary
  quorum.
- cost-if-wrong: running before the concurrent drain stops can produce another
  non-zero graph-wide rescore immediately after closure; broadening the parent
  materializer can mutate protected catalog-edit state beyond this identity;
  hand-setting `b0_like` would require undoing an unaudited authority bypass and
  rerunning the documentation quorum.
