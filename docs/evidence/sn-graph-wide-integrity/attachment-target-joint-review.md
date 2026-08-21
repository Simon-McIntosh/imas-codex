# Attachment target joint ordinary review

## Outcome

The exact attachment cohort received one joint ordinary name-review invocation.
The invocation derived **9 live reviewable identities from 9 authority rows**;
it did not trust a declared cohort count. Every live identity was present,
unique, `validation_status='valid'`, described, unclaimed, and in an ordinary
reviewable lifecycle (`accepted`, `reviewed`, or `drafted`) before any provider
call.

All **9 of 9 identities received exactly one fresh quorum group**. The groups
contain 23 reviewer rows: nine primary, nine secondary, and five authoritative
escalations. Against the locked **0.85** acceptance threshold, **5 of 9 scores
are accept-level and 4 of 9 are below threshold**. No identity was redrawn,
zero identities have more than one new group, and zero claims survive.

This evidence records the review verdict separately from stored lifecycle. The
ordinary batched review writer refreshes reviewer scores and review records but
does not change `name_stage`: the three pre-existing accepted identities remain
accepted, while reviewed and drafted identities retain those stages. No
below-threshold identity was refined or retried, and no accept-level result was
hand-promoted.

## Live-derived cohort and results

The invocation parsed the committed machine-readable attachment authority,
then resolved and filtered the identities against the live graph. The
authority artifact SHA-256 was
`66b34ac9759bb40c0b07e9bc46229847f60b3b881ece019bb168d675a1245475`.
The resulting 9 unique live identities formed 9 context-sensitive review
batches inside the same invocation.

| Attachment row | Canonical target identity | Pre-draw stored lifecycle | Previous stored score | Fresh quorum group and cycles | Fresh score | Verdict at 0.85 |
|---:|---|---|---:|---|---:|---|
| 02 | `poloidal_plane_cross_sectional_area_of_flux_surface` | accepted / valid | 1.00000 | `f83e674e-6e78-4dca-ace9-8aaa012b50ee`; c0, c1 | **0.96875** | **accept-level** |
| 05 | `line_integrated_electron_number_density` | accepted / valid | 1.00000 | `3791870d-41d7-4bd4-8017-9b85240db597`; c0, c1, c2 | **0.81250** | below threshold |
| 07 | `minimum_of_safety_factor` | reviewed / valid | 0.72500 | `dca22f12-d58b-4e17-9b56-19e384fc7e89`; c0, c1, c2 | **0.61250** | below threshold |
| 08 | `neutral_state_power_density` | reviewed / valid | 0.83125 | `1e31a0b4-50ac-49e5-a7f9-f23aa56ff0eb`; c0, c1 | **0.97500** | **accept-level** |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | reviewed / valid | 0.56875 | `79df6ac7-0b25-439f-a2de-499e71572ab6`; c0, c1, c2 | **0.36250** | below threshold |
| 15 | `straight_field_line_angle` | accepted / valid | null | `22f64174-5055-40b8-8d3d-774ee2a25a1c`; c0, c1, c2 | **0.93750** | **accept-level** |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | reviewed / valid | 0.83125 | `3da2933e-fc8e-41a1-9110-be22a477b0ce`; c0, c1 | **1.00000** | **accept-level** |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | drafted / valid | null | `1d0c6799-7dc6-435b-bc6b-b8b128372e8c`; c0, c1 | **0.45000** | below threshold |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | drafted / valid | null | `eb8145e5-dbca-43a0-8eee-ee99f79005ad`; c0, c1, c2 | **0.90000** | **accept-level** |

The descriptions and exact DD source paths supplied to the reviewers came from
the live target records and the attachment authority. Representative semantic
distinctions remained explicit: row 02 is poloidal-plane cross-sectional area,
not swept toroidal surface area; row 05 is electron number density integrated
along a line of sight; and row 21 is a line-of-sight-inferred impurity-ion
velocity rather than a cumulative flux-surface integral.

## Counters, cost, and mutation boundary

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| `LLMCost` rows | 27,631 | 27,631 | **0** |
| `LLMCost.llm_cost` total | USD 1,366.843569 | USD 1,366.843569 | **USD 0.000000** |
| `StandardNameChange` rows | 7,756 | 7,756 | **0** |
| `StandardNameReview` rows | 20,754 | 20,777 | **+23** |

The 23 fresh `StandardNameReview.llm_cost` values sum to **USD 1.56922802**,
matching the review engine's measured cost, or **6.28% of the USD 25 cap**.
The legacy ordinary-review writer records its billable cost on the review rows
and target aggregates rather than adding `LLMCost` nodes; therefore the zero
`LLMCost` delta is reported alongside, not substituted for, actual spend.

No `StandardNameChange` row was created because the invocation neither renamed,
refined, accepted by hand, nor changed documentation. It wrote only fresh
review records and refreshed name-axis review projections. The pre-draw
catalog stages were 3 accepted, 4 reviewed, and 2 drafted; those stored stages
remain 3 accepted, 4 reviewed, and 2 drafted after the score-only review.

## One-draw and claim proof

- Live-derived identities: **9**.
- Identities with exactly one new quorum group: **9**.
- Identities with zero new groups: **0**.
- Identities with more than one new group: **0**.
- Maximum new draw count for any identity: **1**.
- Incomplete quorums: **0**; every identity has at least c0 and c1.
- Surviving `claimed_at` or `claim_token` values: **0**.
- Provider-bearing invocations: **1**.
- Identity retries after the draw: **0**.

An earlier bounded preflight was stopped during its deterministic catalog audit
before any provider call: `LLMCost` remained 27,631 and no fresh review row
existed. The successful invocation narrowed that audit to the exact live cohort;
it is the sole invocation that reached reviewers, so it does not constitute a
second draw or retry for any identity.

## Durable records

- Complete machine-readable result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194808978688-attachdraw/joint-review.json`
- Complete successful invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194808978688-attachdraw/joint-review.log`
- Pre-provider stopped preflight log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194808978688-attachdraw/preflight-aborted.log`
- Exact invocation driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194808978688-attachdraw/run_joint_review.py`

The JSON record contains every pre-draw lifecycle field, review group and cycle,
review-row model and cost, score, threshold verdict, global counter snapshot,
and final claim check used above.
