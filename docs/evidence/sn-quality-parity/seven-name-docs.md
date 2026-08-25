# Seven WEST identities through the documentation quorum

Snapshot window: 2026-08-25 12:35:14–12:50:03 UTC, default live `codex` graph. Every lookup and lifecycle ledger is keyed on `StandardName.id`; live coverage was 4,658/4,658 for `id` and 0/4,658 for the undeclared `name` property.

## Outcome

All **seven** identities that had just cleared the name axis advanced from `docs_stage=pending` to `docs_stage=accepted` through the ordinary `generate_docs → review_docs → refine_docs` pools. Each promotion has **at least two fresh `StandardNameReview` rows on the documentation axis dated after this node started**, and each final aggregate is at or above the configured **0.85** minimum. Six passed their first documentation quorum; `toroidal_beta` scored 0.6500 on its first quorum, followed the ordinary refinement path once, and then accepted at 0.9500. No identity remains withheld, so there is no residual lifecycle reason to assign within this seven-row scope.

**Hand-accepted identities: 0. Documentation texts hand-edited: 0.** The zero-hand-accept check is fail-closed: every accepted row must carry at least two fresh attached documentation-review rows in this run window. All seven do. The only graph mutation command was the exact docs-only ordinary pool invocation; no direct acceptance or text-edit command was issued.

## Exact lifecycle ledger

| `StandardName.id` | Representative meaning and source binding | Documentation lifecycle | Fresh docs reviews | Final aggregate | Ordinary route |
|---|---|---|---:|---:|---|
| `electron_density` | Free-electron number density; 9 direct bindings, including `dd:sawteeth/profiles_1d/n_e` | pending → accepted | 2 | 0.9375 | generate → quorum consensus |
| `normalized_toroidal_flux_coordinate_of_line_of_sight` | Signed closest-approach toroidal-flux label; `dd:spectrometer_x_ray_crystal/channel/profiles_line_integrated/lines_of_sight_rho_tor_norm` | pending → accepted | 2 | 0.8875 | generate → quorum consensus |
| `normalized_toroidal_flux_coordinate_of_measurement_position` | Doppler-reflectometer position label; `dd:reflectometer_fluctuation/channel/doppler/position/rho_tor_norm` | pending → accepted | 2 | 0.90625 | generate → quorum consensus |
| `radial_coordinate_of_electron_cyclotron_launcher_mirror` | Launcher-mirror sphere-centre major radius; `dd:ec_launchers/mirror/geometry/sphere_centre/r` | pending → accepted | 2 | 0.9250 | generate → quorum consensus |
| `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` | Signed toroidal ion velocity divided by field magnitude; 3 direct bindings, including `dd:plasma_profiles/ggd/ion/velocity_over_b_field/r` | pending → accepted | 3 | 0.8875 | generate → authoritative escalation |
| `toroidal_beta` | Volume-averaged total perpendicular pressure relative to toroidal-field magnetic pressure; `dd:summary/global_quantities/beta_tor_mhd/value` | pending → reviewed (0.6500) → refining → drafted → accepted | 5 | 0.9500 | generate → escalation below bar → one ordinary refinement → quorum consensus |
| `turn_count_of_correction_coil` | Winding-turn count of one non-axisymmetric correction coil; `dd:coils_non_axisymmetric/coil/turns` | pending → accepted | 2 | 0.9250 | generate → quorum consensus |

The fresh-review counts are 2, 2, 2, 2, 3, 5 and 2 respectively. `toroidal_beta` has five because its below-bar three-seat first review and passing two-seat post-refinement review are both retained. `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` required an authoritative third seat but no refinement.

## Family documentation census on both bases

The two requested bases remain different by construction and are reported side by side. The fixed baseline preserves the 28 identities that were non-accepted on the documentation axis before the preceding name-gate node. The fresh production re-mint reruns `load_sources_file()` plus `mint_sn_list()` over `west_production_dd_paths.yaml`, excluding terminal identities and rebuilding the one-hop family closure from current graph state.

| Census basis | Population | Non-accepted before | Non-accepted after | What the change means |
|---|---:|---:|---:|---|
| Fixed 28-identity baseline | 28 fixed ids | 28 | **22** | Six baseline members gained accepted documentation. The fixed set deliberately retains five exhausted predecessors and one superseded predecessor. |
| Fresh production re-mint | 410 total = 218 direct + 192 family-only | 22 family-only | **16 family-only** | Six current family-only members gained accepted documentation. Cardinality remained 410 for this comparison; 27 DD paths remain unmatched exactly as the production mint reports. |

The seven-row scope produces a six-row improvement on either family census because `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` is not a member of either family-only set: its superseded predecessor is retained only by the fixed baseline, while the accepted successor does not enter the current immediate-family closure. The after figures still disagree materially—22 on the fixed baseline versus 16 on the fresh re-mint—so quoting only 16 would hide the six terminal predecessors that disappeared from the production closure before this node and would overstate repair through attrition.

The fixed-baseline residual 22 comprise: 11 `name_stage=reviewed/docs_stage=pending`, 5 exhausted/pending, 3 drafted/pending, 1 pending/null, 1 superseded/pending, and 1 accepted/reviewed. The fresh-re-mint residual 16 comprise: 11 reviewed/pending, 3 drafted/pending, 1 pending/null, and 1 accepted/reviewed. Those rows are outside this exact seven-identity documentation scope; the lifecycle labels are reported rather than converted into implicit acceptance.

## Run, spend, and independent checks

The exact command scoped seven `--name` values with `--docs-only --skip-global-maintenance --cost-limit 25 --time 25 --min-score 0.85 --rotation-cap 3`. Run `ab8759e4-556b-433a-b648-9accbe14e42d` stopped at `no_eligible_work` after **7 generation operations, 8 review operations, and 1 documentation refinement**. It made **26 provider calls**.

Actual run spend was **USD 2.094188 / USD 25.000000**, leaving USD 22.905812 of this node ceiling. `SNRun.cost_spent` exactly equals the sum of its 26 run-scoped `LLMCost.llm_cost` rows. Adding this run to the previously authorized running spend of USD 73.675270 brings the campaign to **USD 75.769458 / USD 150.000000**, leaving USD 74.230542. More authority was not used because the exact scope reached `no_eligible_work`, not because of a cost or time stop.

The review loop ignored three reviewer-proposed DD-gap paths outside the claimed batch for the ratio identity. That is the scope fence working: none was persisted by this exact documentation run.

Durable proof:

- Preflight lifecycle, claims, active-run and graph-key coverage: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123506736412-n-sevendocs/preflight.json`
- Full ordinary-pool transcript and exit marker: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123506736412-n-sevendocs/logs/sn-run-live.log`
- Independent final lifecycle, fresh review rows, hand-accept check, cost reconciliation, fixed baseline and fresh production re-mint: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123506736412-n-sevendocs/post-run.json`
- Current descriptions and representative source bindings: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123506736412-n-sevendocs/identity-context.json`
