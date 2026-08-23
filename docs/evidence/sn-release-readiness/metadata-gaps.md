# Accepted-name metadata-gap dispositions

Measured against the production `codex` graph on 2026-08-23. This was a
read-only census: every graph statement used `MATCH`/`RETURN`; no model was
called and no graph node, relationship, or property was changed. The source
investigation was also read-only. None of the recommended repairs below was
applied.

## Headline result

| Accepted-name class | Previously reported | Live result | Named disposition |
|---|---:|---:|---|
| No `physics_domain` | 21 | **21** | **Deterministic DD inheritance/projection:** 20 have bound, pinned DD sources with domains and one derived parent has a domain-bearing accepted child. All 21 are recoverable without classification; classification-required count is **0**. |
| No documentation string | 1 | **1** | **Ordinary documentation generation and review:** keep excluded from publication until its pending docs axis reaches accepted; never hand-write or hand-accept it. |
| Catalog `status IS NULL` / `status='draft'` | 855 / 1,680 | **855 / 1,680** | **Publish-safe exporter projection:** neither graph value reaches output; every candidate is serialized as `active`. Normalize the graph-side lifecycle separately for clarity, but this scalar does not block or corrupt publication. |
| Bare single-bracket references in accepted docs | 2 docs | **0 actual name-reference docs**; the broader docs-quality detector flags **1 doc / 2 mathematical occurrences** | **Repair the detector contract:** preserve mathematical `C[f_q]`; make quality-gate and accept-normalizer definitions identical and math-aware. Then re-run the read-only census. |

The first three live populations agree exactly with the recorded figures. The
fourth does not: the previously reported two accepted documentation strings are
not reproduced. The live graph has no unlinked bracket token that is also a
`StandardName.id`. Details and the source-level reason are below.

## Physics domain: complete 21-row disposition

Every row is recoverable from authority already present in the graph. Twenty
names have direct DD-producing sources. Every such source has
`dd_snapshot_pinned=true` and a non-empty source-domain projection; where a
name has several source domains, the proposed primary below is the result of
the existing promote-on-higher-rank rule. The remaining derived parent,
`current_of_antenna_strap`, inherits `auxiliary_heating` from the accepted
`wave_current_of_antenna_strap` child. The derived-parent materializer already
specifies that child domains are inherited and the full set retained
(`imas_codex/standard_names/graph_ops.py:3782-3810`). Direct candidate
persistence likewise promotes a supplied DD domain while retaining all source
domains (`imas_codex/standard_names/graph_ops.py:5537-5622`).

Recommended repair, not applied: run a governed deterministic projection from
the pinned `StandardNameSource.physics_domain` values (or the accepted children
for a derived parent), recording the complete source set and selected primary.
Do not call the classifier and do not supply a domain from an LLM. Add a
read-only preflight plus an invariant that an accepted name with an
authoritative domain-bearing producer or child cannot retain a null primary.

| Name | Recovery authority | Proposed primary | Authority binding(s) |
|---|---|---|---|
| `current_of_antenna_strap` | Derived-child inheritance | `auxiliary_heating` | child `wave_current_of_antenna_strap` |
| `flux_surface_averaged_effective_charge_at_plasma_boundary` | Pinned DD source | `edge_plasma_physics` | `summary/local/separatrix_average/zeff/value` |
| `radial_outline_of_limiter_tile` | Pinned DD source | `plasma_wall_interactions` | `wall/description_2d/limiter/unit/outline/r` |
| `radial_outline_of_wall` | Pinned DD source | `plasma_wall_interactions` | `wall/description_2d/mobile/unit/outline/r` |
| `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` | Pinned DD source | `transport` | `summary/line_average/isotope_fraction_hydrogen/value` |
| `ratio_of_volume_averaged_hydrogen_density_to_volume_averaged_total_hydrogenic_density` | Pinned DD source | `transport` | `summary/volume_average/isotope_fraction_hydrogen/value` |
| `toroidal_coordinate_at_beam_tracing_point` | Pinned DD source | `electromagnetic_wave_diagnostics` | `ece/channel/beam_tracing/beam/position/phi` |
| `toroidal_coordinate_at_pellet_path_point` | Pinned DD source | `plant_systems` | `pellets/time_slice/pellet/path_geometry/first_point/phi` |
| `toroidal_coordinate_at_shattering_position` | Pinned DD source | `plant_systems` | `spi/injector/shattering_position/phi` |
| `toroidal_coordinate_of_active_spatial_resolution_zone` | Two pinned DD sources | `particle_measurement_diagnostics` | `spectrometer_visible/channel/active_spatial_resolution/centre/phi`; `mse/channel/active_spatial_resolution/centre/phi` |
| `toroidal_coordinate_of_bragg_crystal` | Pinned DD source | `radiation_measurement_diagnostics` | `spectrometer_x_ray_crystal/channel/crystal/centre/phi` |
| `toroidal_coordinate_of_camera` | Two pinned DD sources | `radiation_measurement_diagnostics` | `spectrometer_x_ray_crystal/channel/camera/centre/phi`; `camera_x_rays/camera/centre/phi` |
| `toroidal_coordinate_of_detector` | Four pinned DD sources | `particle_measurement_diagnostics` | detector-centre `phi` paths under `spectrometer_visible`, `bolometer`, `spectrometer_uv`, and `mse` |
| `toroidal_coordinate_of_neutron_detector` | Pinned DD source | `particle_measurement_diagnostics` | `neutron_diagnostic/detector/geometry/centre/phi` |
| `toroidal_coordinate_of_pellet` | Pinned DD source | `plant_systems` | `spi/injector/pellet/position/phi` |
| `toroidal_coordinate_of_pellet_fragment` | Pinned DD source | `plant_systems` | `spi/injector/fragment/position/phi` |
| `toroidal_coordinate_of_polarizer` | Two pinned DD sources | `electromagnetic_wave_diagnostics` | `spectrometer_visible/channel/polarizer/centre/phi`; `ece/polarizer/centre/phi` |
| `toroidal_coordinate_of_reciprocating_probe` | Pinned DD source | `mechanical_measurement_diagnostics` | `langmuir_probes/reciprocating/plunge/position_average/phi` |
| `toroidal_coordinate_of_reflectometer_antenna` | Pinned DD source | `electromagnetic_wave_diagnostics` | `reflectometer_fluctuation/channel/antenna_detection_static/centre/phi` |
| `toroidal_coordinate_of_soft_xray_detector` | Pinned DD source | `radiation_measurement_diagnostics` | `soft_x_rays/channel/detector/centre/phi` |
| `toroidal_coordinate_of_thomson_scattering_laser` | Pinned DD source | `particle_measurement_diagnostics` | `thomson_scattering/laser/end_point/phi` |

Partition: **20 direct DD-source inheritance + 1 derived-child inheritance +
0 classification required = 21**, with no residual.

## Missing documentation: one blocked publication row

The only accepted name with a null or empty documentation string is
`line_averaged_plasma_velocity`. It is a derived structural parent with
`docs_stage='pending'`, `validation_status='valid'`, and
`physics_domain='radiation_measurement_diagnostics'`. Its accepted child is
`toroidal_line_averaged_plasma_velocity`, which has a 0.95 name score and is
bound to
`spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor`.

Disposition: **generate and review documentation through the ordinary docs
pipeline after the structural-authority prerequisite is satisfied**. The row is
already excluded by the export requirement that documentation be accepted, so
the null cannot silently enter the current catalog output; however, the locked
first-release policy requires the backlog to drain rather than publishing a
subset. No manual graph text or direct acceptance is appropriate.

## Null catalog status is harmless at publication

The live partition is exactly **855 null + 1,680 draft = 2,535 accepted**.
Neither value is copied to the catalog. `_graph_node_to_entry_dict` constructs a
fresh catalog entry and unconditionally writes `"status": "active"`, explicitly
because every candidate reaching it passed the accepted/valid/docs-accepted
gates (`imas_codex/standard_names/export.py:841-858`). The export loop calls that
serializer for every candidate and validates the resulting entry dictionary
before adding it to domain output (`imas_codex/standard_names/export.py:1779-1809`).

Therefore a null accepted-name status is **genuinely harmless at publish time**:
it does not reach output as null, and `draft` does not reach output either. The
graph distinction remains poor internal hygiene because two graph states encode
the same unpublished condition. Recommended follow-on, not a release blocker:
normalize the graph-side default under a governed lifecycle migration and add a
test pinning the serializer's active projection. Do not weaken the exporter to
pass through the graph scalar.

## Bare brackets: the live premise changed, and the regex contract has a hole

Three read-only scans were compared:

1. The accept-path normalizer's exact regex found **0** accepted documents.
2. The documentation-quality gate's broader regex found **1** accepted
   document with **2** occurrences.
3. A generic unlinked-bracket scan joined every token against all 4,395
   `StandardName.id` values and found **0** actual name-reference documents.

The sole broad-detector row is
`co_passing_fast_ion_charge_state_torque_density_due_to_collisions`. Both
occurrences are `C[f_q]` in displayed mathematical notation. `f_q` is a
distribution-function symbol, not a Standard Name and not a missing Markdown
link.

The source-level hole is a divergent definition of the same purported
invariant:

- `_BARE_DOC_LINK_RE` only matches lowercase identifier tokens of at least four
  characters and explicitly excludes images, then the scoped accept path uses
  it to rewrite or strip matches
  (`imas_codex/standard_names/graph_ops.py:8833-8838` and
  `imas_codex/standard_names/graph_ops.py:8896-8976`).
- `_BARE_NAME_BRACKET_RE` accepts two-character identifiers and evaluates it
  after removing Markdown links but before removing display/inline mathematics
  (`imas_codex/standard_names/docs_gates.py:30-33` and
  `imas_codex/standard_names/docs_gates.py:135-141`).
- Promotion invokes the scoped normalizer, but any normalization exception is
  non-fatal and acceptance continues
  (`imas_codex/standard_names/graph_ops.py:15609-15628`).

Thus the previous “two surviving links” interpretation is not confirmed live.
The present survivor is a **false-positive quality finding caused by regex
drift**, while the accept path also remains fail-open if its normalization call
raises. Recommended repair: define one shared, math-aware bare-reference parser;
only link/strip a token after proving it is an actual live StandardName identity;
make acceptance refuse an unreadable/failed normalization result; and pin the
contract with examples for a real bare name, `C[f_q]`, escaped display math,
images, and already-formed links. Re-run the census after that source repair.

## Read-only integrity receipt

| Measure | Value |
|---|---:|
| All `StandardName` nodes, before | 4,395 |
| Accepted names | 2,535 |
| All `StandardName` nodes, after | 4,395 |
| Node-count delta | **0** |
| LLM calls | **0** |
| Graph writes | **0** |

The full identity/source readback is retained in the crew-run log at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T100038097890-n-metagaps/graph-baseline.log`.
