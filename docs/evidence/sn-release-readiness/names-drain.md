NEEDS-HELP: The exact 1,179-name review-only campaign reached 1,165 terminal identities, but the 55-minute node fence required interruption with 14 identities still drafted.

tried: Dry-ran and applied `sn restage-accepted` to the exact 1,179-row cohort, then ran only `REVIEW_NAME` under a USD 220 cap until the node time fence.
options: Resume the same deterministic scope with the same `--only review_name` command after the 14 interrupted claims expire; or explicitly release only those interrupted claims through the sanctioned claim-recovery path and resume immediately; do not restage or re-review the 1,165 terminal rows.
leaning: Resume the same scope after ordinary claim expiry, because the run id is durable, the exact-pool selector cannot enter refine, and waiting avoids an out-of-band claim mutation.
cost-if-wrong: Widening the scope or restaging the cohort would rebill up to 1,165 already-decided identities and could overwrite the clean terminal accounting; releasing claims by an unsanctioned write would bypass worker ownership.

# Catalog-import names drain

## Outcome

This node did **not** meet the completion condition. The campaign restaged the exact
1,179-name catalog-import cohort and completed ordinary three-seat name review for
1,165 identities before the time fence. It left 14 identities at `drafted`; those
rows are therefore outside export eligibility and are a reported regression, not a
completed drain.

| Measure | Result |
|---|---:|
| Exact identities selected | 1,179 |
| Identities restaged | 1,179 |
| Identities reviewed to a terminal stage | 1,165 |
| Accepted | 896 |
| Reviewed below 0.85 | 268 |
| Reviewed at or above 0.85 but blocked from acceptance | 1 |
| Within the inclusive bound-adjacent band 0.79375 to 0.90625 | 108 |
| Remaining mid-pipeline | **14 drafted** |
| Actual spend | **USD 60.452013 / USD 220 cap** |
| Spend against the USD 181.23 projection | 33.36% |
| Headroom to cap | USD 159.547987 |
| Calls | 2,596 |
| USD per selected cohort name | USD 0.051274 |
| Calls per selected cohort name | 2.201866 |
| USD per terminally reviewed identity | USD 0.051890 |
| Calls per terminally reviewed identity | 2.228326 |
| Largest recorded call | USD 0.158196 |
| StandardName population | 4,395 before, 4,395 after; delta 0 |

The cap held with zero overshoot, hence within one call. The run stopped because of
the node time fence, not the cost cap. The graph-backed `SNRun` finalized as
`degraded`; its measured cost equals the `LLMCost` sum, USD 60.452013.

## Sanctioned review-only mechanism

The exact selector is `sn run --only review_name`. It is sanctioned in two places:

- `imas_codex/standard_names/turn.py:34-39` maps the `review_name` action directly
  to the single canonical pool of the same name.
- `imas_codex/standard_names/loop.py:681-690` applies that exact selector after the
  broad filters and requires exactly one surviving pool, so `refine_name` cannot
  run as an adjacent action.
- `imas_codex/cli/sn.py:1555-1563` documents that `review_name` selects exactly one
  name-axis action, unlike the broader `review` and `review_names` selectors.

The executed command was:

```text
uv run --no-sync imas-codex sn run --only review_name --names-only --scope-run-id sn-review-restage-1b8c6c0ef31d2e5b831c --skip-global-maintenance --reviewer-profile default --cost-limit 220
```

The configured default profile supplied the three-seat chain
`grok-4.5 -> gpt-5.6-luna -> claude-sonnet-5`. All three appear in the cost
ledger; the third seat was used only for disagreement escalation. No
`refine_name` call or successor identity was created by this campaign.

## Restage receipt

The dry run resolved exactly 1,179 claim-free, accepted, valid, name-unscored
identities and printed deterministic scope
`sn-review-restage-1b8c6c0ef31d2e5b831c`. It predicted and the apply receipt
confirmed:

- 1,179 requested and 1,179 staged;
- zero reviewer scores written;
- 15,394 identity relationships before and after;
- bindings conserved at 2,002 `HAS_STANDARD_NAME`, 1,179 `HAS_UNIT`, and 7
  `HAS_COCOS` relationships.

The full dry-run and apply receipts are retained at:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T132802001348-n-drain/restage-dry-run.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T132802001348-n-drain/restage-apply.json`

## Terminal distribution and bound-adjacent measure

The interrupted terminal-stage distribution is:

| Stage | Count |
|---|---:|
| `accepted` | 896 |
| `reviewed` | 269 |
| `drafted` | **14** |
| Total | 1,179 |

Of the 269 reviewed rows, 268 have a final score below 0.85. One row,
`electron_deposited_power`, scored at or above the bar but remained reviewed
because its descendant cascade reported acceptance conflicts. This is kept
separate from the below-bar population.

The bound-adjacent band is the locked inclusive interval
`0.85 +/- 0.05625 = [0.79375, 0.90625]`. Exactly **108** identities have final
scores in that interval. The count includes either terminal stage because the
measurement is about bound exposure, not lifecycle disposition.

## Interrupted identities

These 14 rows were accepted before the campaign and remained drafted when the time
fence forced interruption. Their final score is null. They still carry interrupted
claim metadata and must not be manually scored, accepted, or reworded.

| Identity | Before | After | Final score |
|---|---|---|---:|
| `extent_of_detector_pixel` | accepted | drafted | null |
| `ion_diffusion_coefficient` | accepted | drafted | null |
| `parallel_neutral_momentum_diffusivity` | accepted | drafted | null |
| `poloidal_electron_energy_diffusion_coefficient` | accepted | drafted | null |
| `radial_coordinate_of_filter_window` | accepted | drafted | null |
| `radial_electron_energy_diffusion_coefficient` | accepted | drafted | null |
| `radial_ion_charge_state_momentum_diffusivity` | accepted | drafted | null |
| `radial_neutral_diffusion_coefficient` | accepted | drafted | null |
| `thickness_of_filter` | accepted | drafted | null |
| `time_derivative_of_flux_surface_averaged_metric` | accepted | drafted | null |
| `toroidal_counter_passing_torque_density` | accepted | drafted | null |
| `toroidal_fast_ion_torque_density_due_to_collisions` | accepted | drafted | null |
| `toroidal_trapped_torque_density` | accepted | drafted | null |
| `vertical_front_surface_curvature_of_optical_element` | accepted | drafted | null |

## Complete identity receipt

The machine-readable receipt names all 1,179 identities with `before_stage`,
`after_stage`, `final_score`, final reviewer, resolution method, and claim state:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T132802001348-n-drain/final-receipt.json`

SHA-256:
`1dea5f5309413790844ff10f27ffeb05eea1e81df0006636be44cb4ffa4a7aa4`.

The full execution log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T132802001348-n-drain/review-campaign.log`.

## Additional fenced finding

Review outcomes completed even when DD-gap evidence persistence rejected model prose
that was not a valid `DDGapEvidenceRule` enum. This did not alter the terminal stage
or score, but the repeated warning indicates an evidence-write contract defect in
`imas_codex/standard_names/dd_gaps.py`, outside this node's write fence.
