# Accepted-name physics-domain projection

## Outcome

The live gap is closed. Re-measurement found **21** accepted Standard Names
without a `physics_domain`, exactly matching the preceding metadata census. A
signed deterministic projection set all 21 from authority already present in
the graph: **20** from pinned DD-source authority and **1** from an accepted
child. The post-apply accepted-name gap is **0**. Classification-required,
unresolvable, and LLM-call counts are all **0**.

The wider graph advanced between the earlier census and this operation: the
metadata census recorded 4,395 `StandardName` nodes, while this operation began
at 4,666. That is intervening graph activity, not a disagreement in the domain
cohort: the missing-domain count re-measured as 21 in both records. During this
operation the count was stable at **4,666 -> 4,666**, so the projection neither
created nor retired an identity. Accepted identities were likewise stable at
**2,335 -> 2,335**.

| Measure | Before | After | Delta / verdict |
|---|---:|---:|---|
| Accepted names missing `physics_domain` | **21** | **0** | **-21; closed** |
| All `StandardName` nodes | 4,666 | 4,666 | **0; identity count conserved** |
| Accepted `StandardName` nodes | 2,335 | 2,335 | **0** |
| Production `_fetch_candidates()` export-eligible rows | 540 | 540 | **0; measured no eligibility-count change** |
| `StandardNameChange` receipt rows | 8,442 | 8,463 | **+21; one per projected identity** |
| `LLMCost` nodes | 31,714 | 31,714 | **0; no LLM call** |

The export-eligible measure is the production query, not a reconstructed
predicate. It was evaluated immediately before and after the write through
`imas_codex.standard_names.export._fetch_candidates()`. The count remained 540:
adding the domain does not alter that query's lifecycle, validation, quorum, or
documentation predicates. The result measures rather than assumes the absence
of a publishability-count effect; the repaired rows now carry scoped catalog
metadata instead of an empty domain wherever they otherwise qualify.

## Sanctioned write and replay proof

The write used the repository's generic signed repair envelope,
`apply_signed_manifest`, at
`imas_codex/standard_names/signed_manifest.py:4374`. The authority was emitted
by `build_repair_authority` at
`imas_codex/standard_names/repair_authority.py:135`; it contained no raw Cypher
and only the closed `set_properties` mutation for `physics_domain`. Each row
fingerprinted its target plus the exact `PRODUCED_NAME`/DD-source participants,
or the exact accepted child plus its `HAS_PARENT` relationship. The operator
previewed the current closure, authorized its digest, re-read and locked the
same closure inside the applying transaction, checked collateral state, wrote
one receipt per logical row, and verified postconditions before commit.

Only `StandardName.physics_domain` was changed. No name text, lifecycle state,
source binding, domain source, or identity was edited. In particular, the path
made no classifier or LLM call; the projected values came from
`StandardNameSource.physics_domain`, the pinned backing `IMASNode.physics_domain`
when the source projection was empty, or the accepted child's
`StandardName.physics_domain`.

| Signed-envelope measure | Result |
|---|---:|
| Authority rows | 21 |
| Preview admitted / refused | **21 / 0** |
| Preview outcome / would change | `would_apply` / 21 |
| Apply outcome / logical rows changed | `applied` / **21** |
| Property mutations / receipt rows | 21 / 21 |
| Apply persistent writes | 42 (21 property updates + 21 receipts) |
| Exact replay outcome | **`already_applied`** |
| Exact replay changed / persistent writes | **0 / 0** |

- Authority file SHA-256:
  `220f60a6b1a1064af37560e683ae2c2388afefa3f7329f981656fe898abaf3f2`
- Canonical authority payload SHA-256:
  `a5088933b50a639e4a92790ad19bbf8fb4105e5b3ca05bd249b2fdc990cb1b89`
- Authorized manifest SHA-256:
  `43916c8b97404ab646a11c72e0f096e807e8b45efba7bf6b728515e74a15d1cc`
- Receipt run:
  `r-20260823T162157748842-n-domainproject`

Running the exact signed apply a second time therefore wrote nothing. This is
the idempotency proof: it is a receipt-backed replay of the same authority and
manifest digest, not merely a second null-count query.

## Per-identity receipt

Every row read back as `name_stage='accepted'` with the listed after value. All
before values were null. For names with several pinned DD sources, the primary
is the existing promote-on-higher-rank result; every contributing authority is
listed so the selection is auditable.

| Identity | Before | After | Projection authority |
|---|---|---|---|
| `current_of_antenna_strap` | null | `auxiliary_heating` | Accepted child `wave_current_of_antenna_strap` -> `StandardName.physics_domain=auxiliary_heating` |
| `flux_surface_averaged_effective_charge_at_plasma_boundary` | null | `edge_plasma_physics` | Pinned `dd:summary/local/separatrix_average/zeff/value` -> `StandardNameSource.physics_domain=edge_plasma_physics` |
| `radial_outline_of_limiter_tile` | null | `plasma_wall_interactions` | Pinned `dd:wall/description_2d/limiter/unit/outline/r` -> backing `IMASNode.physics_domain=plasma_wall_interactions` |
| `radial_outline_of_wall` | null | `plasma_wall_interactions` | Pinned `dd:wall/description_2d/mobile/unit/outline/r` -> backing `IMASNode.physics_domain=plasma_wall_interactions` |
| `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` | null | `transport` | Pinned `dd:summary/line_average/isotope_fraction_hydrogen/value` -> `StandardNameSource.physics_domain=transport` |
| `ratio_of_volume_averaged_hydrogen_density_to_volume_averaged_total_hydrogenic_density` | null | `transport` | Pinned `dd:summary/volume_average/isotope_fraction_hydrogen/value` -> `StandardNameSource.physics_domain=transport` |
| `toroidal_coordinate_at_beam_tracing_point` | null | `electromagnetic_wave_diagnostics` | Pinned `dd:ece/channel/beam_tracing/beam/position/phi` -> `StandardNameSource.physics_domain=electromagnetic_wave_diagnostics` |
| `toroidal_coordinate_at_pellet_path_point` | null | `plant_systems` | Pinned `dd:pellets/time_slice/pellet/path_geometry/first_point/phi` -> `StandardNameSource.physics_domain=plant_systems` |
| `toroidal_coordinate_at_shattering_position` | null | `plant_systems` | Pinned `dd:spi/injector/shattering_position/phi` -> `StandardNameSource.physics_domain=plant_systems` |
| `toroidal_coordinate_of_active_spatial_resolution_zone` | null | `particle_measurement_diagnostics` | Pinned `dd:mse/channel/active_spatial_resolution/centre/phi` -> `particle_measurement_diagnostics`; pinned `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` -> `radiation_measurement_diagnostics`; higher-ranked primary projected |
| `toroidal_coordinate_of_bragg_crystal` | null | `radiation_measurement_diagnostics` | Pinned `dd:spectrometer_x_ray_crystal/channel/crystal/centre/phi` -> `StandardNameSource.physics_domain=radiation_measurement_diagnostics` |
| `toroidal_coordinate_of_camera` | null | `radiation_measurement_diagnostics` | Pinned `dd:camera_x_rays/camera/centre/phi` and `dd:spectrometer_x_ray_crystal/channel/camera/centre/phi` -> `radiation_measurement_diagnostics` |
| `toroidal_coordinate_of_detector` | null | `particle_measurement_diagnostics` | Pinned `dd:mse/channel/detector/centre/phi` -> `particle_measurement_diagnostics`; pinned detector-centre paths under `bolometer`, `spectrometer_uv`, and `spectrometer_visible` -> `radiation_measurement_diagnostics`; higher-ranked primary projected |
| `toroidal_coordinate_of_neutron_detector` | null | `particle_measurement_diagnostics` | Pinned `dd:neutron_diagnostic/detector/geometry/centre/phi` -> `StandardNameSource.physics_domain=particle_measurement_diagnostics` |
| `toroidal_coordinate_of_pellet` | null | `plant_systems` | Pinned `dd:spi/injector/pellet/position/phi` -> `StandardNameSource.physics_domain=plant_systems` |
| `toroidal_coordinate_of_pellet_fragment` | null | `plant_systems` | Pinned `dd:spi/injector/fragment/position/phi` -> `StandardNameSource.physics_domain=plant_systems` |
| `toroidal_coordinate_of_polarizer` | null | `electromagnetic_wave_diagnostics` | Pinned `dd:ece/polarizer/centre/phi` -> `electromagnetic_wave_diagnostics`; pinned `dd:spectrometer_visible/channel/polarizer/centre/phi` -> `radiation_measurement_diagnostics`; higher-ranked primary projected |
| `toroidal_coordinate_of_reciprocating_probe` | null | `mechanical_measurement_diagnostics` | Pinned `dd:langmuir_probes/reciprocating/plunge/position_average/phi` -> `StandardNameSource.physics_domain=mechanical_measurement_diagnostics` |
| `toroidal_coordinate_of_reflectometer_antenna` | null | `electromagnetic_wave_diagnostics` | Pinned `dd:reflectometer_fluctuation/channel/antenna_detection_static/centre/phi` -> `StandardNameSource.physics_domain=electromagnetic_wave_diagnostics` |
| `toroidal_coordinate_of_soft_xray_detector` | null | `radiation_measurement_diagnostics` | Pinned `dd:soft_x_rays/channel/detector/centre/phi` -> `StandardNameSource.physics_domain=radiation_measurement_diagnostics` |
| `toroidal_coordinate_of_thomson_scattering_laser` | null | `particle_measurement_diagnostics` | Pinned `dd:thomson_scattering/laser/end_point/phi` -> `StandardNameSource.physics_domain=particle_measurement_diagnostics` |

Partition arithmetic closes exactly:
**20 pinned DD-source projections + 1 accepted-child inheritance + 0
classification-required + 0 unresolvable = 21**.

## Durable operational record

- Full signed preview, apply receipt, replay receipt, counts, and row readback:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T162157748842-n-domainproject/domain-projection-run.log`
- Exact builder-emitted signed authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T162157748842-n-domainproject/domain-projection-authority.json`

No source file or plan state was edited by this node.
