# WEST family name-gate disposition

Snapshot window: 2026-08-25 09:30:49–09:53:11 UTC, default live `codex` graph. The cohort was minted from `west_production_dd_paths.yaml` with the production `load_sources_file()` plus `mint_sn_list()` path. Every census and ledger join uses `StandardName.id`.

## Outcome

All **28 baseline family-only identities with a non-accepted name lifecycle** are dispositioned. **Seven recovered to an accepted name** through ordinary quorum; **21 remain withheld**, each with a lifecycle reason in the exact ledger. The seven promotions carry **2–3 fresh `StandardNameReview` rows each**, all dated after the run start, with final aggregate scores from **0.8875 to 1.0000**, above the configured **0.85** minimum. **Hand-accepted identities: 0.**

The exact split that prevents two plausible counts being conflated is:

- Name-gate cohort: **28** family-only identities whose `name_stage != accepted`. It includes `iron_density_at_plasma_boundary`, whose documentation was already accepted but whose name was not.
- Documentation census: **28** family-only identities whose `docs_stage != accepted`. It instead includes accepted-name `square_of_magnetic_field_magnitude` and excludes already-documented `iron_density_at_plasma_boundary`.
- Intersection: **27** identities. These sets were measured separately and never joined on the undeclared `StandardName.name` property.

Graph key coverage was **4,656/4,656** `StandardName.id` and **0/4,656** `StandardName.name` before, then **4,658/4,658** and **0/4,658** after. There were no active Standard Name runs or live claims at admission.

## Sanctioned routing and run

Measured before-state routing was:

- **23 reviewed + valid:** dry-run, then `sn rescore --stage-only`, which restaged the same identity to `drafted` without rewording, accepting, or refunding refinement attempts. One exact names-only run then obtained fresh quorum. A fresh below-bar decision entered ordinary `refine_name` while attempts remained.
- **2 drafted + valid:** entered the same exact ordinary review scope without a rescore transition.
- **2 drafted + quarantined:** withheld; deterministic validation is not waived by review.
- **1 pending derived identity:** withheld; it has no reviewable name-axis artifact.

The exact 25-identity pool invocation used `--names-only --skip-global-maintenance --cost-limit 40 --min-score 0.85 --rotation-cap 3`. `SNRun e6a632f0-2bb1-49c0-978c-9733f5df9b19` stopped at `no_eligible_work` after **26 review operations** and **7 claimed ordinary-refine attempts**, two of which persisted successive identities in the one successful refinement lineage. No global maintenance ran.

The recovered names demonstrate each ordinary route:

| Accepted identity | Description and production binding | Final score | Fresh reviews | Route |
|---|---|---:|---:|---|
| `electron_density` | Local free-electron number density; nine direct DD bindings including `thomson_scattering/channel/n_e`. | 1.0000 | 2 | drafted → ordinary quorum |
| `normalized_toroidal_flux_coordinate_of_line_of_sight` | Signed closest-approach normalized toroidal-flux coordinate; `spectrometer_x_ray_crystal/channel/profiles_line_integrated/lines_of_sight_rho_tor_norm`. | 0.8875 | 3 | same-name rescore → authoritative escalation |
| `normalized_toroidal_flux_coordinate_of_measurement_position` | Doppler-reflectometer measurement-position coordinate; `reflectometer_fluctuation/channel/doppler/position/rho_tor_norm`. | 0.9125 | 2 | same-name rescore → quorum consensus |
| `radial_coordinate_of_electron_cyclotron_launcher_mirror` | Major-radius coordinate of the launcher-mirror sphere center; `ec_launchers/mirror/geometry/sphere_centre/r`. | 0.99375 | 2 | same-name rescore → quorum consensus |
| `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` | Toroidal ion velocity divided by local magnetic-field magnitude; three GGD component bindings. | 0.95625 | 2 | two ordinary refinements → quorum consensus |
| `toroidal_beta` | Plasma pressure divided by toroidal magnetic-field pressure; `summary/global_quantities/beta_tor_mhd/value`. | 0.9125 | 3 | same-name rescore → authoritative escalation |
| `turn_count_of_correction_coil` | Turns in a non-axisymmetric correction coil; `coils_non_axisymmetric/coil/turns`. | 0.99375 | 2 | same-name rescore → quorum consensus |

## Exact before/after ledger

Null attempt counters are displayed as zero because the claim predicates use `coalesce(refine_attempts, 0)`. “Reviews” is the count of name-axis review rows dated after 09:30:49 UTC on the final identity.

| Identity before | Identity after | Name before | Docs before | Validation before | Attempts before | Score before | Name after | Docs after | Validation after | Attempts after | Score after | Reviews | Disposition |
|---|---|---|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| `beryllium_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: fresh quorum cleared the score bar, but the sanctioned accept path refused because accepted descendant `flux_surface_averaged_beryllium_density_at_plasma_boundary` would require an accepted-descendant cascade. |
| `beta` | same | reviewed | pending | valid | 0 | 0.3000 | reviewed | pending | valid | 0 | 0.3000 | 1 | Withhold: same-name rescore hit the deterministic semantic-similarity gate, produced only one fresh review and no quorum resolution; ordinary refine remains ineligible while that shortfall persists. |
| `carbon_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_carbon_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `coolant_mass` | same | pending | null | null | 0 | — | pending | null | null | 0 | — | 0 | Withhold: pending derived identity has no reviewable name-axis artifact. |
| `deuterium_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_deuterium_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `deuterium_deuterium_neutron_flux` | same | drafted | pending | quarantined | 0 | — | drafted | pending | quarantined | 0 | — | 0 | Withhold: deterministic validation remains quarantined. |
| `electron_density` | same | drafted | pending | valid | 0 | — | accepted | pending | valid | 0 | 1.0000 | 2 | Accepted through fresh ordinary quorum. |
| `helium_4_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_helium_4_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `hydrogen_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 0.99375 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_hydrogen_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `iron_density_at_plasma_boundary` | same | reviewed | accepted | valid | 0 | 1.0000 | reviewed | accepted | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_iron_density_at_plasma_boundary` blocks a non-authorized cascade; its already-accepted docs do not waive the name axis. |
| `lithium_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 0.9625 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: three cascade conflicts—accepted flux-surface-averaged descendant plus unreachable line- and volume-averaged descendants—were refused atomically. |
| `neon_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_neon_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `normalized_toroidal_flux_coordinate_of_line_of_sight` | same | reviewed | pending | valid | 0 | 0.8750 | accepted | pending | valid | 0 | 0.8875 | 3 | Accepted through fresh ordinary quorum. |
| `normalized_toroidal_flux_coordinate_of_measurement_position` | same | reviewed | pending | valid | 0 | 0.83125 | accepted | pending | valid | 0 | 0.9125 | 2 | Accepted through fresh ordinary quorum. |
| `poloidal_magnetic_flux` | same | reviewed | pending | valid | 0 | 0.9875 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `radial_derivative_of_poloidal_magnetic_flux` blocks a non-authorized cascade. |
| `poloidal_turn_count` | same | reviewed | pending | valid | 0 | 0.7875 | exhausted | pending | valid | 1 | 0.6125 | 3 | Withhold: ordinary refine exhausted on successor collision with occupied `poloidal_field_line_turn_count`. |
| `radial_coordinate_of_arc_of_circle_center` | same | reviewed | pending | valid | 0 | 0.64375 | exhausted | pending | valid | 2 | 0.5750 | 3 | Withhold: ordinary refine exhausted when the proposed successor failed strict grammar validation. |
| `radial_coordinate_of_electron_cyclotron_launcher_mirror` | same | reviewed | pending | valid | 0 | 0.7000 | accepted | pending | valid | 0 | 0.99375 | 2 | Accepted through fresh ordinary quorum. |
| `radial_coordinate_of_pellet_path` | same | reviewed | pending | valid | 0 | 0.9000 | exhausted | pending | valid | 1 | 0.8125 | 3 | Withhold: ordinary refine exhausted on successor collision with occupied `radial_coordinate_of_pellet_path_point`. |
| `radial_coordinate_of_reflector` | same | drafted | pending | valid | 0 | — | drafted | pending | valid | 0 | — | 0 | Withhold: lifecycleless stub has neither description nor embedding, so the ordinary review predicate correctly excludes it. |
| `ratio_of_ion_velocity_to_magnetic_field` | `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude` | reviewed | pending | valid | 0 | 0.6250 | accepted | pending | valid | 2 | 0.95625 | 2 | Accepted after two ordinary refinements and fresh quorum; the predecessor is superseded. |
| `safety_factor_at_pedestal` | same | reviewed | pending | valid | 0 | 0.66875 | exhausted | pending | valid | 1 | 0.5750 | 3 | Withhold: ordinary refine exhausted on successor collision with occupied `safety_factor_at_pedestal_top`. |
| `time_derivative_of_electron_density` | same | reviewed | pending | valid | 0 | 0.83125 | exhausted | pending | valid | 1 | 0.6500 | 3 | Withhold: ordinary refine exhausted when the proposed operator chain failed strict grammar validation. |
| `toroidal_beta` | same | reviewed | pending | valid | 0 | 0.90625 | accepted | pending | valid | 0 | 0.9125 | 3 | Accepted through fresh ordinary quorum. |
| `tritium_tritium_neutron_flux` | same | drafted | pending | quarantined | 0 | — | drafted | pending | quarantined | 0 | — | 0 | Withhold: deterministic validation remains quarantined. |
| `tungsten_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 1.0000 | 2 | Withhold: accepted descendant `flux_surface_averaged_tungsten_density_at_plasma_boundary` blocks a non-authorized cascade. |
| `turn_count_of_correction_coil` | same | reviewed | pending | valid | 0 | 0.84375 | accepted | pending | valid | 0 | 0.99375 | 2 | Accepted through fresh ordinary quorum. |
| `xenon_density_at_plasma_boundary` | same | reviewed | pending | valid | 0 | 1.0000 | reviewed | pending | valid | 0 | 0.9750 | 2 | Withhold: accepted descendant `flux_surface_averaged_xenon_density_at_plasma_boundary` blocks a non-authorized cascade. |

## Documentation census and packet remint

On the fixed 28-row family documentation baseline, the id-keyed count remains **28 → 28**: this node changes only the name axis, so newly accepted names still have `docs_stage=pending`. A fresh production remint reads **28 → 22** family-only non-accepted documents because six now-terminal predecessor identities leave the immediate-family mint: five exhausted names and superseded `ratio_of_ion_velocity_to_magnetic_field`. The accepted ratio successor does not enter the immediate-family closure, so this remint is **410 identities: 218 direct + 192 family-only**, down from 416 with no added identities. The removed names are:

- `poloidal_turn_count`
- `radial_coordinate_of_arc_of_circle_center`
- `radial_coordinate_of_pellet_path`
- `ratio_of_ion_velocity_to_magnetic_field`
- `safety_factor_at_pedestal`
- `time_derivative_of_electron_density`

The fixed-baseline number is the lifecycle comparison; the fresh-remint number is the packet state. Reporting both prevents terminal rows disappearing from the evidence merely because minting correctly omits them.

## Spend, proofs, and residual authority

Actual model spend was **USD 2.455551 / USD 40.000000**, leaving **USD 37.544449** of this node ceiling. The `SNRun.cost_spent` value agrees with **65** run-scoped `LLMCost` rows summing to **USD 2.4555510000000007**. The running authorized total is **USD 73.675270 / USD 150.000000**, leaving **USD 76.324730**. More authority was not consumed because the exact run reached `no_eligible_work`; every remaining row is lifecycle-withheld rather than capacity-deferred.

Durable machine-readable proofs:

- Before census, key coverage, reviews, parent/child context, and 28-row baseline: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T092543930216-n-westnamegate/preflight.json`
- Exact route partition: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T092543930216-n-westnamegate/route.json`
- Full live pool log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T092543930216-n-westnamegate/logs/sn-run-live.log`
- Independent after census, exact lineage ledger, fresh review rows, LLMCost reconciliation, and zero-hand-accept proof: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T092543930216-n-westnamegate/post-run.json`
