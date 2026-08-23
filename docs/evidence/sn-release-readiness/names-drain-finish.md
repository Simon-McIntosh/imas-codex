# Catalog-import names drain completion

## Outcome

The deterministic catalog-import cohort is fully drained on the names axis.
All **1,179** identities now have a terminal name stage: **905 accepted** and
**274 reviewed**, with **zero drafted**, **zero null name scores**, and **zero
active claims**. Of the reviewed population, **273 are below the 0.85 bar** and
one, `electron_deposited_power`, is at or above the bar but remains reviewed
because its descendant cascade reported acceptance conflicts.

The continuation spent **USD 1.104296** against its USD 20 fence. Together with
the interrupted run's USD 60.452013, the complete campaign cost is **USD
61.556309**. The additional run finalized `completed`, reported exact cost, and
stopped with `no_eligible_work`; the `LLMCost` ledger independently sums to the
same USD 1.104296 across 34 calls.

| Measure | Final result |
|---|---:|
| Exact restaged identities | 1,179 |
| Accepted | 905 |
| Reviewed below 0.85 | 273 |
| Reviewed at or above 0.85 but blocked from acceptance | 1 |
| Remaining at `drafted` | **0** |
| Null final name scores | **0** |
| Active claims | **0** |
| Within inclusive bound-adjacent band `[0.79375, 0.90625]` | **109** |
| Additional calls | 34 |
| Additional spend | **USD 1.104296 / USD 20** |
| Complete campaign spend | **USD 61.556309** |
| StandardName population | **4,395 before and after; delta 0** |

The locked bound-adjacent interval is `0.85 +/- 0.05625`, or
`[0.79375, 0.90625]`. The final count is **109 of 1,179** identities. This is a
cohort-wide score-exposure measure and therefore includes both terminal stages.

## Claim-expiry gate and exact continuation

The continuation did not release claims by hand. Immediately before dispatch,
all 14 interrupted identities were still `drafted` and carried their interrupted
claim metadata, but every claim had exceeded the required 600-second stale
window: the youngest was 605 seconds old and the oldest 686 seconds old. The
scope still resolved to exactly 1,179 identities and the graph still contained
4,395 StandardName nodes.

No restage command was run. The exact continuation was:

```text
uv run --no-sync imas-codex sn run --only review_name --names-only --scope-run-id sn-review-restage-1b8c6c0ef31d2e5b831c --skip-global-maintenance --reviewer-profile default --cost-limit 20
```

The run began with exactly 14 eligible `review_name` items and processed exactly
14. Its invocation scope records `only_pool=review_name`, and every one of its 34
cost rows records phase, pool, and event type `review_name`. Thus the run used the
locked three-seat review chain only: 14 calls to `grok-4.5`, 14 to
`gpt-5.6-luna`, and 6 disagreement escalations to `claude-sonnet-5`. It made no
refine call, created no successor identity, reworded no identity, wrote no hand
score, and performed no direct acceptance. Names that accepted did so through
the ordinary RD-quorum review persistence path.

## The fourteen interrupted identities

All fourteen reached a terminal stage. Nine accepted and five remained reviewed
below the bar; the continuation left none silently drafted.

| Identity | Terminal stage | Final score | Resolution |
|---|---|---:|---|
| `extent_of_detector_pixel` | reviewed | 0.77500 | authoritative escalation |
| `ion_diffusion_coefficient` | accepted | 0.98750 | quorum consensus |
| `parallel_neutral_momentum_diffusivity` | reviewed | 0.83750 | authoritative escalation |
| `poloidal_electron_energy_diffusion_coefficient` | reviewed | 0.70000 | authoritative escalation |
| `radial_coordinate_of_filter_window` | accepted | 1.00000 | quorum consensus |
| `radial_electron_energy_diffusion_coefficient` | accepted | 0.92500 | authoritative escalation |
| `radial_ion_charge_state_momentum_diffusivity` | accepted | 0.91875 | quorum consensus |
| `radial_neutral_diffusion_coefficient` | accepted | 0.92500 | quorum consensus |
| `thickness_of_filter` | accepted | 1.00000 | quorum consensus |
| `time_derivative_of_flux_surface_averaged_metric` | reviewed | 0.50000 | authoritative escalation |
| `toroidal_counter_passing_torque_density` | accepted | 1.00000 | quorum consensus |
| `toroidal_fast_ion_torque_density_due_to_collisions` | accepted | 1.00000 | quorum consensus |
| `toroidal_trapped_torque_density` | accepted | 0.92500 | quorum consensus |
| `vertical_front_surface_curvature_of_optical_element` | reviewed | 0.71250 | authoritative escalation |

`parallel_neutral_momentum_diffusivity` is the only one of these fourteen inside
the bound-adjacent band. Its final score of 0.83750 increases the campaign-wide
band count from 108 to 109.

## Complete identity receipt

The machine-readable receipt reports every one of the 1,179 identities with its
terminal stage, final score, reviewer, resolution method, quorum-shortfall field,
and final claim state. It also carries the cohort totals, the continuation run
row, the three-model cost breakdown, and the combined spend:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T142524040287-n-drainfinish/final-receipt.json`

SHA-256:
`e54b31cf9c581839c4898a7503a1e05889518ce98ba8240f2d29cafc1d7238a2`.

The complete execution log is:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T142524040287-n-drainfinish/review-campaign.log`

The graph-backed continuation run is
`4508ae42-61b2-45d6-b50d-efc0a4af99bb`.
