# Accepted-documentation quorum campaign

## Outcome

Status: **blocked at the authorized cost fence after a real campaign**. The
exact 106-identity cohort was staged through the sanctioned docs-only recovery
operator and entered the ordinary documentation review/refine pools. The run
spent **USD 39.980907 of USD 40.00** and stopped with
`stop_reason=budget_saturated`; **16 identities remain drafted**, so the required
zero-mid-pipeline completion condition is not true and no completion claim is
made.

The run nevertheless produced authoritative partial outcomes: **97 distinct
identities acquired at least one docs-axis review**, **85 reached
`docs_stage=accepted` with a non-null `docs_review_resolution_method`**, and
**5 landed in `reviewed` below the 0.85 bar**. Nine drafted identities acquired
no completed review before budget saturation. No identity is currently claimed.

## Sanctioned route and safety preconditions

The sanctioned transition is `stage_docs_for_rescore()` in
`imas_codex/standard_names/graph_ops.py:15899-15956`, introduced by commit
`b8d69e4e989d63d61528a6552c62b6832278b3b5`. Its compare-and-set admits an
accepted name whose docs are in an operator-recoverable terminal stage and
which has no live or drain claim. It moves only the docs stage and aggregate
docs-review decision fields, stamps an exact `run_id`, and preserves the
description, documentation, refinement depth, and complete
`StandardNameReview` history. A scoped `sn run` then feeds those unchanged docs
to ordinary `REVIEW_DOCS`; no score, resolution method, or acceptance stage was
hand-written.

The focused operator check passed **3/3**:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_review_docs_stages.py::TestExactDocsRescoreStaging -q
```

Before trusting any zero-returning predicate, the live coverage check proved
that every filtered property exists in the graph: `StandardName.id` 4,666,
`name_stage` 4,666, `docs_stage` 4,662, `origin` 3,575,
`docs_review_resolution_method` 2,053, and `StandardNameReview.review_axis`
24,410. The observed review-axis vocabulary includes the required plural values
**`names` and `docs`**. This rules out the silent-zero failure mode caused by a
missing property or the invalid singular axis spelling.

The exact live predicate was:

```text
name_stage = accepted
AND docs_stage = accepted
AND origin = catalog_edit
AND no attached StandardNameReview where review_axis = docs
```

It returned **106**, matching the corrected contract. It returned 0 rows with a
docs score, 0 with a docs resolution method, and 0 with a live claim. The prior
107th row, `krypton_density_at_magnetic_axis`, is correctly absent because it
already carries real docs-axis review authority; a null scalar is not the same
condition as no review.

## Executed campaign

All 106 dry-run eligibility checks passed, then all 106 rows were staged under
the deterministic scope `docs-rescore-20260824T0505Z`. The production campaign
was:

```text
imas-codex sn run --docs-only \
  --scope-run-id docs-rescore-20260824T0505Z \
  --min-score 0.85 --cost-limit 40 --skip-global-maintenance
```

The durable `SNRun` id is `16a1ee3e-d6be-4d29-94e4-4fb1467238e3`.
It executed **460 LLM calls**: 165 completed docs-review actions and 75 docs
refinements, for **USD 30.3033** and **USD 9.6776** respectively. The maximum
single call cost was **USD 0.322688**. The budget stopped at
**USD 39.980907 / USD 40.00**, under the cap by **USD 0.019093**, so the fence
held to better than one call.

Actual spend was **USD 0.377178 per staged identity**. That is materially above
the sibling-measured USD 0.123-0.136 sizing band because the ordinary drain did
not stop after the first quorum outcome: 75 below-bar docs proceeded through
paid refinement and re-review. The sibling band was therefore not an admission-
safe bound for a complete review/refine drain.

| Measure | Actual outcome |
|---|---:|
| Exact identities staged | **106** |
| Distinct identities with a completed docs review | **97** |
| Completed docs-review actions | **165** |
| Completed docs refinements | **75** |
| Accepted with a docs resolution method | **85** |
| Final below-bar `reviewed` identities | **5** |
| Drafted identities still mid-pipeline | **16** |
| Drafted identities with no completed docs review | **9** |
| Live claims after exit | **0** |

Final cohort stage distribution is **accepted 85, reviewed 5, drafted 16**.
The five final below-bar outcomes are:

| Identity | Final score | Resolution |
|---|---:|---|
| `coulomb_logarithm` | 0.65000 | `authoritative_escalation` |
| `counter_passing_fast_particle_density` | 0.76875 | `quorum_consensus` |
| `inner_radius_of_ferritic_element` | 0.84900 | `quorum_consensus`; also demoted by a mismatched documentation link |
| `line_averaged_xenon_density` | 0.63750 | `quorum_consensus` |
| `neutral_energy_diffusion_coefficient` | 0.55000 | `quorum_consensus` |

Representative accepted results include `accumulated_krypton_count` at
0.94375, `alfven_frequency_imaginary_part` at 0.96250, and
`current_of_divertor_tile` at 0.99375, all resolved by `quorum_consensus`.

## Binding and population preservation

The stage operation and campaign preserved every fenced authority binding:

| Relationship / population | Before | After | Delta |
|---|---:|---:|---:|
| Incoming `HAS_STANDARD_NAME` | 162 | 162 | **0** |
| Outgoing `HAS_UNIT` | 106 | 106 | **0** |
| Outgoing `HAS_COCOS` | 0 | 0 | **0** |
| `StandardName` nodes | 4,666 | 4,666 | **0** |

Unit and domain authority therefore stayed on the DD-owned bindings; the
campaign did not rebuild or infer them.

## Production export impact

Both measurements used the real production path, not a reconstructed
predicate:

```text
run_export(staging_dir, min_score=0.85, skip_gate=True,
           force=True, include_sources=False)
```

The supplied baseline was **1,947 emitted**. A fresh pre-write export measured
**1,948**, showing one identity of concurrent drift before this node staged
anything. The post-campaign export emitted **2,033** identities. The attributable
gain against the fresh pre-write measurement is therefore **+85**, exactly the
85 newly docs-accepted/resolved cohort identities; the change against the
recorded 1,947 baseline is **+86**.

The accepted/approved export population remained 2,335. Both export reports
closed their exclusion accounting. Three unrelated catalog entries still fail
ISN entry validation in both measurements; they do not affect the attributable
identity difference.

## WEST intersection

The 106-row cohort intersects the 231-identity WEST gate manifest in **6
identities**:

| WEST identity | Final docs state | Score / disposition |
|---|---|---|
| `thickness_of_filter` | accepted | 0.89375, `quorum_consensus` |
| `upper_photon_energy` | accepted | 0.95625, `quorum_consensus` |
| `volume_averaged_electron_density` | accepted | 0.96250, `quorum_consensus` |
| `wavelength_of_spectral_line` | accepted | 0.95625, `quorum_consensus` |
| `voltage_of_poloidal_magnetic_field_probe` | drafted | no completed review |
| `width_of_poloidal_field_coil` | drafted | no completed review |

Thus four WEST identities moved from never-reviewed to docs-accepted in this
campaign, while two remain in the blocked continuation population.

## Blocker and durable evidence

The unmet condition is exact: **16 of 106 identities remain in
`docs_stage=drafted` after the authorized USD 40 was exhausted**. Eighteen
review attempts were explicitly deferred because a complete quorum could not be
funded, and the run refused to accept on a single review. Completing the drain
requires a new cost authorization; no additional run was started.

Durable receipts and logs:

- preflight and identity set:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/preflight.json`;
- stage receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/stage-receipt.json`;
- campaign log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/campaign.log`;
- post-campaign identity census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/post-campaign.json`;
- production export reports:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/before-export/.export_report.json`
  and
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T044334897102-n-docs106/after-export/.export_report.json`.
