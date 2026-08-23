# Name-review quorum self-agreement at the acceptance bound

Measured read-only against the production `codex` graph on 2026-08-23. No
model was called and no LLM budget was spent. Every review query selected
`review_axis = 'names'` explicitly; the graph contains zero rows under the
incorrect singular value `name`. The measurement never takes an unfiltered
maximum across the names and documentation axes.

## Result

| Quantity | Population | Median | 90th percentile | Maximum | What it measures |
|---|---:|---:|---:|---:|---|
| **Between-cycle absolute score delta** | **415 names; 718 adjacent cycle pairs** | **0.05625** | **0.225** | **0.9875** | The same Standard Name identity passing through the review instrument again. This is the genuine run-to-run swing. |
| **Within-cycle reviewer spread** | **2,167 names; 2,845 review groups with at least two scores** | **0.1000** | **0.2875** | **1.0000** | The highest minus lowest individual reviewer score inside one review group. This is reviewer-to-reviewer disagreement, not the instrument reversing itself. |

The between-cycle population is large enough to report: 415 independently
named quantities and 718 observed transitions are not a handful. Of those 718
transitions, **149 crossed the 0.85 acceptance bound** (20.75%), involving 134
distinct names. This is the direct retrospective reversal rate. It must still
be read as a selected operational cohort: names were re-reviewed for reasons,
so it is not a random sample of every catalog entry.

The within-cycle result is deliberately separate. A 0.10 median spread inside
one quorum draw says that the reviewer models disagree with one another. It
does not show the quorum giving a different answer on another run. Conflating
the 0.2875 within-cycle p90 with the 0.225 between-cycle p90 would overstate the
run-to-run uncertainty.

## How a review-cycle score was selected

The unit of a review cycle is one `(StandardName.id, review_group_id)` pair,
not an individual `cycle_index` row. Individual rows are reviewer seats inside
that group. For each group, the score follows the repository's canonical
projection semantics:

1. use the cycle-2 score when it is an `authoritative_escalation`;
2. otherwise use the mean of cycle 0 and cycle 1 for `quorum_consensus`;
3. otherwise use the cycle-0 row marked canonical;
4. never substitute the maximum reviewer score.

Groups were ordered by `reviewed_at`, and only adjacent groups were differenced.
This avoids the combinatorial overweighting that would result from comparing
every group with every other group for a frequently redrawn identity. Seven
unattached `lc_test__*` name-review rows have no `HAS_REVIEW` link to a
`StandardName`; they were included in the unchanged graph sentinel count but
excluded from every identity-level statistic.

As a sensitivity check, 362 adjacent transitions kept the same recorded
canonical model on both sides. Their absolute delta was 0.03125 median,
0.17375 p90 and 0.4125 maximum, and 52 crossed the bound (14.36%). This lower
rate shows that historical model-seat changes contribute to the operational
all-history result. The primary figure remains the whole recorded quorum
instrument because that is what actually routed names; the sensitivity result
prevents interpreting all 20.75% as irreducible sampling noise from a fixed
model configuration.

Representative same-identity histories show both the decision-boundary effect
and the long tail:

| Standard Name | Physical meaning | Consecutive projected cycle scores | Observation |
|---|---|---:|---|
| `radial_ion_momentum_convection_velocity` | Effective radial ion-momentum convection velocity representing advective radial momentum transport. | 0.8625 → 0.84375 | A 0.01875 swing reversed pass to refine. |
| `poloidal_bulk_ion_velocity` | Poloidal component of bulk-ion flow velocity averaged over charge states. | 0.8375 → 0.8625 → 0.8125 | The same identity crossed upward and then back downward in consecutive groups. |
| `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy` | Net confinement-loss power after subtracting stored-energy change from total plasma heating. | 0.0000 → 0.9875 | The maximum 0.9875 transition; it also changed from a single canonical result to a quorum mean and therefore belongs to the operational tail, not a fixed-seat noise estimate. |

`StandardNameReview` preserves the reviewed identity, scores, models, group and
time, but not a digest of the complete rendered review input. The primary
measurement is therefore a genuine same-identity operational redraw, as the
pipeline routes it, rather than a claim of laboratory repeatability under a
byte-identical prompt and configuration.

## Bound-crossing census

This census asks whether **any individual names-axis reviewer** occurred on the
other side of 0.85. It intentionally uses all reviewer seats after filtering
to the names axis; it is not the between-cycle canonical projection above.

| Direction | Count | Population | Interpretation |
|---|---:|---:|---|
| At least one reviewer scored at or above 0.85, current stage is not `accepted` | **566** | 1,860 currently non-accepted names | 466 are `superseded`; 100 are in the 217-name live tail. A passing reviewer did not by itself carry the instrument decision. |
| Current stage is `accepted`, at least one reviewer scored below 0.85 | **399** | 2,535 accepted names | 292 currently carry a passing scalar, 14 carry a below-bound scalar, and 93 have a null scalar. This is historical or within-cycle disagreement, not proof of a later self-reversal. |

The earlier tail census's **32 of 144 redraw-eligible** is reproduced as the
narrower, ordered-disposition subset; it was never the graph-wide count. The
100 live-tail identities with at least one passing reviewer partition exactly
as follows:

| Tail reason | Names |
|---|---:|
| Non-shortfall, valid identity with a passing reviewer — the previously reported redraw criterion | **32** |
| Recorded quorum shortfall despite at least one passing reviewer | 61 |
| Steering takes precedence: validation quarantined | 3 |
| Steering takes precedence: attempts exhausted | 2 |
| Steering takes precedence: successor collision | 2 |
| **Total live-tail bound crossings** | **100** |

Thus the reconciliation is **566 graph-wide = 466 superseded + 100 live tail**,
and **100 live tail = 32 census criterion + 61 quorum shortfall + 7 steering**.
The 32 remains correct at its declared scope.

## Decision number for the scored catalog

There are **2,293 currently scored Standard Names**: 1,029 at or above 0.85
and 1,264 below it. The decision-relevant point estimate is **476 of 2,293
(20.8%) flipping accept-versus-refine on a redraw**. It is obtained directly
from the observed between-cycle crossing rate:

`2,293 × (149 / 718) = 475.85`, rounded to 476.

The score-distance census gives a useful cross-check without pretending that
every swing moves in the adverse direction. At the measured median absolute
delta of **0.05625**, 538 current scores lie within one typical swing of the
bound: 298 below it and 240 at or above it. Those 538 are *susceptible* to a
flip; the observed-direction estimate of 476 is the number projected to
actually flip. At the p90 delta of 0.225, 1,944 scores are susceptible, but
that is a conservative exposure band, not a forecast that 1,944 names will
reverse.

This projection is intentionally qualified. The repeated-cycle cohort is
selected toward names that were redrawn, and historical seat changes are part
of the all-history instrument. The same-model sensitivity would project 329
flips (`2,293 × 52 / 362`) instead. The evidence therefore supports planning
for a material draw-dependent fraction, not claiming an unbiased campaign
acceptance probability. For the proposed 1,179-name campaign, it means the
difference between a three-call decision and a nine-call escalation cannot be
treated as negligible, but this retrospective does not turn the selected
history into an exact campaign cost forecast.

The **0.0755** value from another project's single-judge content gate was not
used anywhere in this calculation. It is neither a bound nor a proxy for the
name-review quorum.

## Read-only integrity sentinels

| Node label | Before | After | Delta |
|---|---:|---:|---:|
| `StandardName` | 4,395 | 4,395 | **0** |
| `StandardNameReview` | 20,928 | 20,928 | **0** |

The after-count was taken only after all evidence queries completed. Both
deltas are zero.
