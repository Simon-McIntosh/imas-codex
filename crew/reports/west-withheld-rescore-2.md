NEEDS-HELP: the fixed-code rerun did not recover the ten reviewed names, and two governed edits remain blocked by source compare-and-set conflicts.

tried: Staged all ten authorized identities with `sn rescore --stage-only`, drained an explicit sixteen-name scope, completed both requested docs drains, staged and reviewed the grammar-valid governed successors that the CLI admitted, and drove the existing opacity hint through its exact `sn-edit` scope. The ten fresh name reviews again scored 0.96875–1.000 but the accept path still invoked rename-cascade protection and refused every acceptance over an accepted descendant. The opacity compose path failed twice on competing live source bindings and was stopped. No direct acceptance and no graph-text Cypher were used.

options: (1) repair the identity-preserving rescore accept path so it bypasses rename cascade planning when the identity is unchanged, then rescore the same ten names again; (2) repair or adjudicate the multiply-bound sources through the governed provenance lifecycle before retrying `vertical_coordinate_at_line_of_sight` and the opacity hint; (3) separately adjudicate the two grammar-valid but semantically low-scoring pinned successors instead of forcing acceptance.

leaning: Option 1 first, followed by option 2. The ten names already have repeated fresh quorum evidence at or above 0.96875, so the observable blocker is the still-active rename-cascade guard. The multiply-bound source failures are independent provenance defects and must not be bypassed.

cost-if-wrong: A wrong rescore-path repair spends another quorum draw on ten names without changing their lifecycle. A wrong source repair can move DD provenance to the wrong identity and requires a governed ledger reconstruction before any further review.

# WEST withheld rescore recovery

## Outcome

The node is **blocked with partial recovery**.

- Rescore: **0/10** target names accepted. All ten were staged successfully and freshly reviewed; their final scores are 0.96875–1.000, but every acceptance was refused by an accepted-descendant cascade conflict.
- Scoped docs drains: **2/2** completed. `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` is now `accepted / accepted / valid` at docs score 0.86875. `spectral_wavelength_of_optical_element` is now `accepted / accepted / valid` at docs score 0.8625.
- Governed corrections: one successor is fully accepted (`toroidal_coordinate_at_detector_pixel`); two valid pinned successors exhausted below threshold (`ion_cyclotron_heating_antenna_phase` at 0.6875 and `cumulative_inside_flux_surface_total_electron_density` at 0.6000); `vertical_coordinate_at_line_of_sight` was not staged because source migration refused a double binding; the opacity hint remains open after the exact scoped compose failed twice on competing bindings.
- Original cohort fully accepted: **6/28** (`breakdown_initial_time`, `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`, `normalized_poloidal_flux_coordinate`, `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`, `radial_coordinate_of_conductor_cross_section`, and `spectral_wavelength_of_optical_element`).
- Total measured spend attributable to this node: **$2.025263 / $20.000000**, leaving **$17.974737** unused. The three completed receipts are exact. The interrupted opacity receipt records $0.000000 from two local-model calls but is explicitly inexact; its `LLMCost` sum is exactly $0.000000.
- Final census: **4,691** `StandardName` rows, **4,691** with `id`, **4,691** with `name_stage`, **2,335** accepted, **0 approved**, **0 contested**.

## Run receipts and command ledger

Every Python/CLI command used the shared project environment with `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, `PYTHONPATH=$PWD`, and `uv run --no-sync`. Read-only inspection commands are grouped where they shared one shell exit.

| Command | Exit | SNRun or audit id | Exact spend | Result |
|---|---:|---|---:|---|
| Read live plan `imas-codex:sn-west-catalog-release` and the plan HTML | 0 | none | $0.000000 | Version 35 authorized this rerun and named the three merged fixes. |
| `git status`, fixed-code inspection, shared `.venv`/`.env` checks | 0 | none | $0.000000 | Detached worktree clean; fixed commits present. |
| `sn run --help`; `sn rescore --help`; `sn edit --help` | 0 | none | $0.000000 | Confirmed `--name`, `--scope-run-id`, `--docs-only`, `--stage-only`, and cost flags. |
| Baseline GraphClient lifecycle and census read | 0 | none | $0.000000 | 4,690 candidates; approved 0; contested 0. |
| `sn rescore beryllium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154229Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore carbon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154239Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore deuterium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154248Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore helium_4_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154256Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore hydrogen_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154306Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore lithium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154315Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore neon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154323Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore poloidal_magnetic_flux --stage-only` | 0 | `sn-rescore-20260902T154331Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore tungsten_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154339Z` | $0.000000 | reviewed -> drafted. |
| `sn rescore xenon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T154347Z` | $0.000000 | reviewed -> drafted. |
| `sn edit vertical_coordinate_of_line_of_sight --rename vertical_coordinate_at_line_of_sight ... --dry-run` | 0 | none | $0.000000 | Governed rename plan valid. |
| `sn edit volume_integrated_total_electron_density --hint ... --dry-run` | 2 | none | $0.000000 | Accepted name cannot re-enter name generation from a hint; CLI required a full replacement. |
| Public ISN parser/semantic probes for the complete replacement | 0 | none | $0.000000 | `cumulative_inside_flux_surface_total_electron_density` strict round-trip and semantic checks passed. One initial diagnostic used the wrong public import and exited 1 before the corrected import exited 0. |
| `sn edit volume_integrated_total_electron_density --rename cumulative_inside_flux_surface_total_electron_density ... --dry-run` | 0 | none | $0.000000 | Governed rename plan valid. |
| Same vertical rename, live `--stage-only` | 1 | none | $0.000000 | CAS refused `dd:reflectometer_profile/channel/line_of_sight_emission/first_point/z`, bound to both the old name and `vertical_coordinate_of_diagnostic_component_centre`. |
| Same volume rename, live `--stage-only` | 0 | `sn-edit-20260902T154628Z` | $0.000000 | Successor drafted; predecessor superseded. |
| Explicit sixteen-name `sn run ... --skip-global-maintenance --cost-limit 20 --time 30` | 0 | `6a892963-cadd-412a-9375-2c72628f5424` | **$1.854569 exact** | 45 calls; 19 reviews, 2 docs generations, 1 docs refinement. Ten rescores were all held by the descendant cascade. |
| Post-run GraphClient lifecycle, run, cost, and source reads | 0 | none | $0.000000 | Verified outcomes and exact receipt. |
| Two-name `sn run --docs-only ... --cost-limit 4 --time 15` | 0 | `bc23fc43-4a2c-43bb-9cdb-3d5608d7c05a` | **$0.000000 exact** | No eligible work; exposed the stale reviewed/open spectral docs state. |
| `sn edit spectral_wavelength_of_optical_element --hint ... --axis docs --stage-only --dry-run` | 0 | none | $0.000000 | Governed docs reset plan valid. |
| Same spectral docs edit, live | 0 | `sn-edit-20260902T160222Z` | $0.000000 | Docs reset for one eligible name. |
| One-name `sn run --docs-only ... --cost-limit 4 --time 15` | 0 | `86acec34-45b0-466a-8627-d7fc9ada4932` | **$0.170694 exact** | Three calls; spectral docs accepted at 0.8625. |
| Opacity `sn run --scope-run-id sn-edit-20260902T142708Z ... --cost-limit 14.145431 --time 15` | 1 | `49d749d1-4a3f-49bd-b143-c8dfd972c660` | **$0.000000 inexact receipt; $0.000000 exact LLMCost** | Two zero-cost local compose calls each hit a source CAS conflict; stopped after the second failure. |
| Final GraphClient lifecycle, source, run, LLMCost, and census reads | 0 | none | $0.000000 | Final evidence below. |

The completed exact receipts sum to `$1.854569 + $0.000000 + $0.170694 = $2.025263`. The degraded receipt adds two calls and no billed spend. The sum remains below the single authorized ceiling even if the inexact receipt is conservatively treated as zero only because its two persisted `LLMCost.llm_cost` values sum to exactly zero.

## Per-name lifecycle evidence

Triplets are `name_stage / docs_stage / validation_status`. “Before” is the fresh baseline taken immediately before this node's mutations; “after” is the final GraphClient census.

| Original withheld name | Before | After | Route and result |
|---|---|---|---|
| `beryllium_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `beta` | reviewed / pending / valid | reviewed / pending / valid | Out of the authorized 0.97–1.00 rescore cohort; unchanged. |
| `breakdown_initial_time` | accepted / accepted / valid | accepted / accepted / valid | Export-ledger condition only; unchanged. |
| `carbon_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | accepted / accepted / valid | accepted / accepted / valid | Export-ledger condition only; unchanged. |
| `deuterium_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `deuterium_deuterium_neutron_flux` | drafted / pending / quarantined | drafted / pending / quarantined | Legitimately withheld; unchanged. |
| `helium_4_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `hot_neutral_temperature` | reviewed / accepted / valid | reviewed / accepted / valid | Score 0.300 and outside authorized rescore cohort; unchanged. |
| `hydrogen_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | drafted / pending / quarantined | drafted / pending / quarantined | Grammar gap; unchanged. |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | drafted / pending / quarantined | drafted / pending / quarantined | Exact hint scope composed twice; both source migrations failed CAS, edit remains open. |
| `lithium_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `neon_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `normalized_poloidal_flux_coordinate` | accepted / accepted / valid | accepted / accepted / valid | Already recovered before this rerun; unchanged. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | accepted / pending / valid | **accepted / accepted / valid** | Scoped docs generate/review accepted at 0.86875. |
| `phase_of_ion_cyclotron_heating_antenna` | superseded / accepted / quarantined | superseded / accepted / quarantined | Governed successor reviewed to exhaustion at 0.6875. |
| `poloidal_magnetic_flux` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `radial_coordinate_of_conductor_cross_section` | accepted / accepted / valid | accepted / accepted / valid | Export-ledger condition only; unchanged. |
| `radial_outline_of_plasma_boundary` | accepted / accepted / quarantined | accepted / accepted / quarantined | Legitimately withheld; unchanged. |
| `radial_outline_of_wall` | accepted / accepted / quarantined | accepted / accepted / quarantined | Legitimately withheld; unchanged. |
| `spectral_wavelength_of_optical_element` | accepted / reviewed / valid | **accepted / accepted / valid** | Scoped docs edit reset followed by docs drain; accepted at 0.8625. |
| `toroidal_coordinate_of_detector_pixel` | superseded / accepted / quarantined | superseded / accepted / quarantined | Governed successor fully accepted. |
| `tritium_tritium_neutron_flux` | drafted / pending / quarantined | drafted / pending / quarantined | Legitimately withheld; unchanged. |
| `tungsten_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 1.000 acceptance refused by accepted descendant. |
| `vertical_coordinate_of_line_of_sight` | accepted / accepted / quarantined | accepted / accepted / quarantined | Governed rename refused source CAS; no successor staged. |
| `volume_integrated_total_electron_density` | accepted / accepted / quarantined | superseded / accepted / quarantined | Grammar-valid successor staged but quorum exhausted at 0.6000. |
| `xenon_density_at_plasma_boundary` | reviewed / pending / valid | reviewed / pending / valid | Rescore staged; fresh 0.96875 acceptance refused by accepted descendant. |

### Governed successor outcomes

| Intended correction | Final successor lifecycle | Measured disposition |
|---|---|---|
| Invalid integrated opacity prefix -> grammar-steered opacity identity | no persisted successor; original drafted / pending / quarantined, edit open | Two candidates reached persistence but CAS refused competing bindings: first `spectral_wave_opacity_at_ece_channel_emission_position`, then `line_integrated_opacity`. |
| `phase_of_ion_cyclotron_heating_antenna` -> `ion_cyclotron_heating_antenna_phase` | exhausted / pending / valid | Grammar round-trip valid; final score 0.6875, edit exhausted. |
| `toroidal_coordinate_of_detector_pixel` -> `toroidal_coordinate_at_detector_pixel` | **accepted / accepted / valid** | Name 1.000; docs 0.93125; edit applied. |
| `vertical_coordinate_of_line_of_sight` -> `vertical_coordinate_at_line_of_sight` | no successor | Source CAS refusal on a multiply-bound DD source. |
| `volume_integrated_total_electron_density` -> `cumulative_inside_flux_surface_total_electron_density` | exhausted / pending / valid | Strict ISN and semantic validation passed before staging; final reviewer score 0.6000, edit exhausted. |

## Blocking evidence

### Identity-preserving rescore still enters rename cascade

All ten target identities earned fresh scores at or above the threshold. The persistence path nevertheless emitted the same conflict shape, for example:

```text
persist_reviewed_name: poloidal_magnetic_flux scored acceptance but its descendant cascade has 1 conflict(s) — refusing acceptance
'radial_derivative_of_poloidal_magnetic_flux' has name_stage='accepted'; pass include_accepted=True to rename anyway
```

Equivalent warnings were emitted for the nine boundary-density identities. This is not quorum variance: across the complete ten-name cohort, nine scores are 1.000 and xenon is 0.96875, with all ten still `reviewed / pending / valid`.

### Source compare-and-set refusals

The vertical governed rename refused this binding:

```text
dd:reflectometer_profile/channel/line_of_sight_emission/first_point/z
bindings=['vertical_coordinate_of_diagnostic_component_centre', 'vertical_coordinate_of_line_of_sight']
```

The opacity hint failed twice on `dd:ece/channel/optical_depth`; the first attempt observed bindings to the original plus `spectral_wave_opacity_at_ece_channel_emission_position`, and the second observed the original plus `line_integrated_opacity`. The run was interrupted after the second failure as required. No direct source rewiring was attempted.

## Final graph proof

The final read used the schema-owned `StandardName.id` key and included key-coverage positive controls in the same query:

```text
candidates=4691
with_id=4691
with_name_stage=4691
accepted=2335
approved=0
contested=0
```

Thus the graph contains **zero approved** and **zero contested** names. The instrument was aimed at `StandardName.name_stage`, the lifecycle property used by the release contract, and proved all candidates carried that property before trusting either zero.

## Evidence locations

- This report: `crew/reports/west-withheld-rescore-2.md`
- Auto-rotating CLI log: `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`
- Durable graph receipts: `6a892963-cadd-412a-9375-2c72628f5424`, `bc23fc43-4a2c-43bb-9cdb-3d5608d7c05a`, `86acec34-45b0-466a-8627-d7fc9ada4932`, and degraded `49d749d1-4a3f-49bd-b143-c8dfd972c660`.

No source code, plan/index state, catalog text, acceptance property, or graph text was edited by this node.
