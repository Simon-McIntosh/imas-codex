# Score-consuming attachment-target transition

## Outcome

The exact attachment cohort's stored lifecycle now agrees with its fresh
ordinary-review authority. Exactly **3 of 9 identities transitioned to
`accepted`** by consuming their already-persisted, quorate score at or above
the locked **0.85** threshold:

- row 08, `neutral_state_power_density`: `reviewed → accepted` at **0.975**;
- row 20,
  `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions`:
  `reviewed → accepted` at **1.000**;
- row 29, `magnetic_field_at_pedestal_top_low_field_side_magnitude`:
  `drafted → accepted` at **0.900**.

The live accepted count among the nine moved from **3/9 to 6/9**. All three
transitions remained `validation_status='valid'`. The normal persistence path
cleared the stale `review_quorum_shortfall` markers on rows 08 and 20 after
checking their fresh `quorum_consensus` resolution; row 29 consumed its fresh
`authoritative_escalation` resolution.

No identity was reviewed, refined, retried, renamed, or redrawn. The global
`StandardNameReview` count remained exactly **20,777**, the `LLMCost` row count
remained **27,631**, and attributable spend was **USD 0.00**.

## Sanctioned score-consuming path

The transition used the existing generic and ordinary-review lifecycle
surfaces; no source or schema change was needed:

1. `apply_signed_manifest` at
   `imas_codex/standard_names/signed_manifest.py:2469` supplied exact signed
   three-row scoping, fresh participant and collateral closure, locking,
   counter checks, one receipt per admitted target, transaction commit, and
   write-free replay.
2. `REVIEW_NAME_ELIGIBILITY_WHERE` at
   `imas_codex/standard_names/graph_ops.py:14554` admitted a drain-scoped
   `reviewed` name only when its stored score cleared 0.85.
   `claim_review_name_batch` begins at line 14574 and selects the atomic
   name-axis restage at line 14657. The shared claim implementation at lines
   13890-13900 changes a passing `reviewed` stage to `drafted`; a pre-existing
   `drafted` name remains drafted.
3. `persist_reviewed_name` at
   `imas_codex/standard_names/graph_ops.py:14740` consumed the stored score and
   fresh resolution. It selects `accepted` from the score at lines
   14885-14890, enforces the quorum method at lines 14909-14913, and performs
   the claim/sequence-guarded lifecycle write at lines 15025-15055.
   `skip_review_node=True` prevented creation of a second review record.

The signed authority uses only schema-declared top-level fields. Its three
rows each carry the existing collateral-immutability guard and `orphan_policy`
and rely on `apply_signed_manifest`'s established one-transaction behavior.
No new `RepairAuthorityArtifact` field was introduced.

## Complete cohort lifecycle

| Row | Canonical target identity | Fresh group | Score | Verdict at 0.85 | Before | After | Disposition |
|---:|---|---|---:|---|---|---|---|
| 02 | `poloidal_plane_cross_sectional_area_of_flux_surface` | `f83e674e-6e78-4dca-ace9-8aaa012b50ee` | 0.96875 | accept-level | accepted / valid | accepted / valid | Already accepted; no transition needed. |
| 05 | `line_integrated_electron_number_density` | `3791870d-41d7-4bd4-8017-9b85240db597` | 0.81250 | below threshold | accepted / valid | accepted / valid | Untouched: ordinary review never demotes a pre-existing accepted identity. |
| 07 | `minimum_of_safety_factor` | `dca22f12-d58b-4e17-9b56-19e384fc7e89` | 0.61250 | below threshold | reviewed / valid | reviewed / valid | Untouched: score is below 0.85. |
| 08 | `neutral_state_power_density` | `1e31a0b4-50ac-49e5-a7f9-f23aa56ff0eb` | **0.97500** | **accept-level** | reviewed / valid | **accepted / valid** | Fresh `quorum_consensus` score consumed. |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | `79df6ac7-0b25-439f-a2de-499e71572ab6` | 0.36250 | below threshold | reviewed / valid | reviewed / valid | Untouched: score is below 0.85. |
| 15 | `straight_field_line_angle` | `22f64174-5055-40b8-8d3d-774ee2a25a1c` | 0.93750 | accept-level | accepted / valid | accepted / valid | Already accepted; no transition needed. |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `3da2933e-fc8e-41a1-9110-be22a477b0ce` | **1.00000** | **accept-level** | reviewed / valid | **accepted / valid** | Fresh `quorum_consensus` score consumed. |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | `1d0c6799-7dc6-435b-bc6b-b8b128372e8c` | 0.45000 | below threshold | drafted / valid | drafted / valid | Untouched: score is below 0.85. |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `eb8145e5-dbca-43a0-8eee-ee99f79005ad` | **0.90000** | **accept-level** | drafted / valid | **accepted / valid** | Fresh `authoritative_escalation` score consumed. |

Only rows 08, 20, and 29 entered the score-consuming scope. Rows 02 and 15
were already accepted and were not rewritten. All four below-threshold
identities were excluded by construction: row 05 retained its pre-existing
accepted lifecycle because review is non-demoting; rows 07 and 13 remained
reviewed; row 21 remained drafted.

## Counters, receipts, cost, and replay

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| accepted targets among the 9 | 3 | 6 | **+3** |
| `StandardNameChange` rows | 7,756 | 7,759 | **+3** |
| `StandardNameReview` rows | 20,777 | 20,777 | **0** |
| `LLMCost` rows | 27,631 | 27,631 | **0** |
| `LLMCost.llm_cost` total | USD 1,366.843569 | USD 1,366.843569 | **USD 0.00** |
| surviving claims or drain scopes | 0 | 0 | **0** |

The signed apply admitted **3/3 rows**, refused **0**, performed **3 scoped
mutations**, created **3 `StandardNameChange` receipts**, and reported **6
persistent writes**. The `StandardNameChange` delta therefore equals the exact
number of lifecycle transitions written. Immediate replay returned
`already_applied`, `changed=0`, and `persistent_writes=0` while retaining the
same three receipt identities.

The execution imported no review worker and made no provider call. Its score
inputs were the nine fresh groups already recorded by the joint draw. The
unchanged review count of 20,777 is the direct proof of **0 fresh quorum
draws**; the unchanged LLM counters and total are the independent proof of
**USD 0.00** transition spend.

## Authority and durable records

- authority file SHA-256:
  `b2ed85a5ae6dcb99592c2c1437bc7235b05a21be05be968d9d4fd524ced93121`;
- authority canonical payload SHA-256:
  `4be10726673a91167efd48cf64898289b224e60f78ed5d5226a684e5aa3544d1`;
- apply manifest SHA-256:
  `7b5f7a8f4067f22536bd11f8a987f5c50388bb2940531b3fa9cf657c8471f9ae`.

Complete machine-readable records:

- transition result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/score-transition-result.json`;
- independent completed-state readback:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/completed-postcheck.json`;
- exact signed authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/score-transition-authority.json`;
- exact invocation driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/run_transition.py`;
- complete resumed invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T201039371873-scoreconsume/transition-resumed.log`.

The two earlier schema-shape failures remain in their named logs as zero-write
prelude evidence. Both stopped in authority loading; neither reached the graph.
The successful resumed invocation changed only the three governed lifecycle
targets and their one-per-row internal receipts.
