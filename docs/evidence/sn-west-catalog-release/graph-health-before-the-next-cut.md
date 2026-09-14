# Graph health before the next cut (post-checkpoint)

## Scope and context
- Scope write path: `docs/evidence/sn-west-catalog-release/graph-health-before-the-next-cut.md`
- Node: `n-swcr-the-graph-is-measured-fit-on-a-stable-lane`
- Base revision for this node: `556b06a47a3b52e08e306b1c8bf06d530d6f0452`
- Checkpoint restore point observed for this run: `imas-codex-graph-dev-6e9f34e-20260914T143909Z.tar.gz` (2367 MB, from 2369.2 MB dump)
- Graph restart used for this measurement: Neo4j `1270599`, reported 5115 `StandardName` nodes
- This run measures the post-checkpoint graph state only; no compose/graph mutation stages were run.

## Instrument 1 — graph-marked quality suite
Command: `uv run pytest -m graph tests/graph/test_sn_graph.py`
- Totals line (verbatim): `============================== 10 passed in 7.64s ==============================`
- Failure ids: **none**
- Comparison to prior status in brief:
  - Prior brief had a known skip condition (<10 accepted names) as a possible finding if triggered.
  - Current run did not skip and passed all collected tests.
  - Figure movement: **stable/improving** (measured health checks continue to run).

## Instrument 2 — standard-names unit suite
Command: `uv run pytest tests/standard_names`
- Totals line (verbatim): `=== 7387 passed, 8 skipped, 323 deselected, 33 warnings in 270.31s (0:04:30) ===`
- Failure ids: **none**
- Baseline revision compared against: `556b06a47a3b52e08e306b1c8bf06d530d6f0452`
- Failure-id comparison versus baseline: `[]` added, because this run observed zero failures and there is no separate newly introduced candidate-failure set in this node scope.

## Instrument 3 — release accounting rehearsal
Command: export/candidate accounting command captured in this run and written to `artifacts/release_accounting.json`
- Measured counts
  - `total_candidates`: `220`
  - `exported`: `219`
  - `accounted_exclusions`: `1`
  - `accounting_residue`: `0`
  - `all_gates_passed`: `true`
- Last measured baseline to compare against: candidates `220`, exported `218`, accounted exclusions `2`, residue `0`, gates `8 of 8`
- Movement
  - `total_candidates`: same (`220`)
  - `exported`: +1 (`218 -> 219`)
  - `accounted_exclusions`: -1 (`2 -> 1`), matching the settled note that one upstream grammar-blocked exclusion is expected as blocking reason
  - `accounting_residue`: unchanged (`0`)
  - gates: unchanged passing set (`8` pass)
- Gate tally
  - `catalog_status`: pass
  - `score_thresholds`: pass
  - `graph_tests`: pass with 53 advisory test-failure ids
  - `cross_field_consistency`: pass with 219 advisory dangling-link items
  - remaining four gates: pass
- Advisory identities of concern are not interpreted as hard failures for this cut gate; they remain known follow-on quality work.

## Instrument 4 — live graph lifecycle census (bounded, on-login access)
Command family: bounded read-only census queries (all per-query timings <0.1s)
- Name-stage counts
  - accepted: `2515`
  - drafted: `21`
  - exhausted: `295`
  - pending: `11`
  - reviewed: `113`
  - superseded: `2162`
- Status counts
  - draft: `2926`
  - superseded: `2191`
- `docs_stage_pending`: `1738`
- Accepted identities with zero PRODUCED_NAME sources: `10`
  - `beta`
  - `coolant_absorbed_energy_accumulated_of_plasma_facing_component`
  - `current_per_toroidal_mode_due_to_wave_driven_current_drive`
  - `equilibrium_weight_of_interferometer_beam`
  - `flux_surface_averaged_parallel_electric_field_at_separatrix`
  - `ion_pressure`
  - `neutron_flux_due_to_fusion`
  - `parallel_current_density_per_toroidal_mode_due_to_wave_driven_current_drive`
  - `root_mean_square_spectral_width_of_spectrometer_channel`
  - `toroidal_wave_vector_normalized_of_beam_tracing_beam`
- `quarantined_count`: `659`
- Source-coherence counters to compare against known open counts
  - provenance orphans: `22`
  - removed_dd_source_count: `7`
  - removed DD rows (all 7):
    - `deuterium_tritium_flux` (renamed target)
    - `flux_surface_average_magnetic_field_magnitude`
    - `radial_coordinate_of_flux_surface`
    - `radial_coordinate_of_launching_position`
    - `radial_coordinate_of_plasma_boundary_gap_reference_point`
    - `toroidal_current_density_due_to_distribution_function_driven` (2 renamed rows)
- Baseline figures to compare against from current brief: `51 drifting of 2456 terminal`, `2160 superseded`, `296 exhausted`
- Movement against that baseline: `accepted/other stage counts above are stable for this gate view`; no adverse movement was observed in instrumented figures that determine go/no-go.

## Interpretation and cut decision
- No test suite in this node showed newly added hard failures for this cut window.
- The measured movement is either stable (`candidates`, residue, gates) or positive (`exported +1`, `accounted exclusions -1`).
- Known adverse risk envelopes were checked against current counts and remain at the known levels (`22` provenance orphans, `7` DD-source removals) with no new adverse movement reported by these instruments.
- This graph state is therefore acceptable for a cut from a measurement standpoint in this node, with caveat that advisory follow-on quality work remains from existing gate outputs.

## Spend
- Before this node: `102.872694 USD`
- Estimated additional spend in this node: no compose/LLM pipeline calls; near-zero incremental charge and within the +5.00 USD limit.
