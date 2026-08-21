NEEDS-HELP: the score-consuming transition was stopped before mutation after the signed authority failed closed validation twice.

# Score-consuming attachment-target transition

## Outcome

The live graph remains unchanged. The exact nine-target cohort still contains
**3 accepted, 4 reviewed, and 2 drafted identities**; the accepted count is
therefore **3/9 before and 3/9 after** this node. No lifecycle transition was
written.

Both execution attempts failed while loading the signed scope authority,
before preview construction, participant locking, or a graph transaction. The
first authority used a non-canonical selection predicate. After correcting
that field to the required `artifact-rows`, the second authority retained an
unsupported top-level `all_or_nothing` field and was rejected by the generated
`RepairAuthorityArtifact` model. The worker stop rule forbids a third attempt
after the same operation fails twice with different corrections, so the
otherwise mechanical removal of that field was not executed.

The post-failure readback proves the safety boundary quantitatively:

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameReview` rows | 20,777 | 20,777 | **0** |
| `StandardNameChange` rows | 7,756 | 7,756 | **0** |
| `LLMCost` rows | 27,631 | 27,631 | **0** |
| `LLMCost.llm_cost` total | USD 1,366.843569 | USD 1,366.843569 | **USD 0.00** |
| surviving claims or drain scopes in the cohort | 0 | 0 | **0** |

Thus this attempt made **0 fresh quorum draws**, spent **USD 0.00**, and
created **0 transition receipts**. The requested successful measure remains
unmet: the `StandardNameChange` delta should equal the **3** score-consuming
transitions, not zero.

## Sanctioned score-consuming path

The existing path is sufficient and requires no provider call:

1. `apply_signed_manifest` at
   `imas_codex/standard_names/signed_manifest.py:2469` supplies exact signed
   three-row scope, lock/re-hash, receipt, counter, collateral, and replay
   protection.
2. `REVIEW_NAME_ELIGIBILITY_WHERE` at
   `imas_codex/standard_names/graph_ops.py:14554` admits an explicitly
   drain-scoped `reviewed` name only when its stored score is at least the
   threshold. `claim_review_name_batch` begins at line 14574 and selects its
   atomic name-axis restage at line 14657. The shared claim implementation at
   lines 13890-13900 changes only a passing `reviewed` stage to `drafted`.
3. `persist_reviewed_name` at
   `imas_codex/standard_names/graph_ops.py:14740` is the sanctioned
   score-consuming lifecycle writer. It selects `accepted` from the supplied
   score at lines 14885-14890, enforces the recorded quorum resolution at
   lines 14909-14913, and performs the claim/sequence-guarded lifecycle write
   at lines 15025-15055. Supplying `skip_review_node=True` consumes the already
   persisted fresh group without creating another `StandardNameReview`.

The prepared invocation limits the signed scope and score consumer to rows 08,
20, and 29, reuses each exact fresh group and stored score, and calls no review
worker or provider. The two `reviewed` targets carry stale shortfall text from
an older draw, but their named fresh groups are quorate; the normal persistence
writer clears that stale marker only after validating the fresh resolution.

## Complete cohort state

The post-failure state equals the pre-attempt state for all nine identities.
Scores and groups are the fresh joint-review authority; `after` is a live
readback after both authority-validation failures.

| Row | Canonical identity | Fresh group | Score | Verdict at 0.85 | Before | After | Transition disposition |
|---:|---|---|---:|---|---|---|---|
| 02 | `poloidal_plane_cross_sectional_area_of_flux_surface` | `f83e674e-6e78-4dca-ace9-8aaa012b50ee` | 0.96875 | accept-level | accepted / valid | accepted / valid | Already accepted; no transition needed. |
| 05 | `line_integrated_electron_number_density` | `3791870d-41d7-4bd4-8017-9b85240db597` | 0.81250 | below threshold | accepted / valid | accepted / valid | Untouched: ordinary review never demotes a pre-existing accepted identity. |
| 07 | `minimum_of_safety_factor` | `dca22f12-d58b-4e17-9b56-19e384fc7e89` | 0.61250 | below threshold | reviewed / valid | reviewed / valid | Untouched: score is below 0.85. |
| 08 | `neutral_state_power_density` | `1e31a0b4-50ac-49e5-a7f9-f23aa56ff0eb` | **0.97500** | **accept-level** | reviewed / valid | reviewed / valid | Eligible, but not transitioned because signed scope validation stopped the invocation. |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | `79df6ac7-0b25-439f-a2de-499e71572ab6` | 0.36250 | below threshold | reviewed / valid | reviewed / valid | Untouched: score is below 0.85. |
| 15 | `straight_field_line_angle` | `22f64174-5055-40b8-8d3d-774ee2a25a1c` | 0.93750 | accept-level | accepted / valid | accepted / valid | Already accepted; no transition needed. |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `3da2933e-fc8e-41a1-9110-be22a477b0ce` | **1.00000** | **accept-level** | reviewed / valid | reviewed / valid | Eligible, but not transitioned because signed scope validation stopped the invocation. |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | `1d0c6799-7dc6-435b-bc6b-b8b128372e8c` | 0.45000 | below threshold | drafted / valid | drafted / valid | Untouched: score is below 0.85. |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `eb8145e5-dbca-43a0-8eee-ee99f79005ad` | **0.90000** | **accept-level** | drafted / valid | drafted / valid | Eligible, but not transitioned because signed scope validation stopped the invocation. |

Only rows 08, 20, and 29 are in the prepared transition set. All four
below-threshold identities are excluded, and the two already-accepted
accept-level identities are deliberately outside the mutation set.

## Blocker and resumption

tried: Built an exact three-row signed `set_properties` authority, a write-free
preview, replay gate, scoped claim loop, and calls to `persist_reviewed_name`
using the existing fresh group data. Attempt one failed on the closed selection
predicate; attempt two failed on the extra `all_or_nothing` field. Both failed
inside `_load_authority`, before any graph write.

options: (1) resume with the unsupported field removed and run the existing
driver once; (2) regenerate the authority from the known-good two-description
authority template, then use the same claim/persist sequence; or (3) add a
dedicated batch score-consumer in `graph_ops.py`, which requires a scope change
and coordination with the current writer owner.

leaning: Option 1. The remaining failure is schema-shape validation, not a
semantic uncertainty. The score threshold, exact cohort, fresh group IDs,
quorum resolutions, lifecycle preconditions, expected three receipts, and
zero-provider path are already explicit and live-verified.

cost-if-wrong: If generic signed scoping is not accepted as sufficient
transition provenance, its three scope receipts must be discarded in favor of
a dedicated atomic batch lifecycle operator and this evidence rerun. No graph
rollback is needed from the present attempt because it wrote nothing.

## Durable diagnostics

- prepared driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/run_transition.py`
- first validation failure:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/transition.log`
- second validation failure:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/transition-second.log`
- post-failure live counter and nine-row readback:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/blocked-postcheck.json`
- rejected authority bytes:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/score-transition-authority.json`

