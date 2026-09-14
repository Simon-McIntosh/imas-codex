# WEST manifest paths without a terminal standard name

## Result

The live release projection reproduces the cardinality in the release
manifest: `load_sources_file()` expands the `sources` mapping to **355** paths
in **22** IDS, and `fetch_manifest_source_release_rows()` returns **25** rows
with no `standard_name_id`.

The proposed explanation that these paths were never seeded is not current.
Every one of the 25 has exactly one exact `StandardNameSource` (`dd:<path>`)
and a `FROM_DD_PATH` edge; **25 have a source node and 0 do not**.  All have
zero `PRODUCED_NAME` edges.  The source statuses are 8 `extracted`, 2 `failed`,
2 `not_physical_quantity`, and 13 `skipped`.

The dispositions below use the DD data type, unit, category, lifecycle, and
documentation, then check an exact source binding and any existing
standard-name representative.  They do not infer eligibility from the path
spelling alone.

| Disposition | Paths | Meaning |
|---|---:|---|
| A — valid source wrongly withheld | 11 | A physical or geometric quantity has no terminal name and remains eligible for naming. |
| B — genuinely ineligible | 13 | A time axis, fit weight/convergence value, or contour-tree topology coordinate does not denote an independently nameable physical quantity. |
| C — already represented | 1 | The quantity already has an accepted standard-name representative; it needs an audited source binding, not another composition. |

## The recorded STRUCTURE defect is already fixed

The relevant extraction invariant is in
`imas_codex/standard_names/sources/dd.py:675-691`.  It excludes generic
`STRUCTURE` and `STRUCT_ARRAY` containers but admits a `quantity` node that has
its own `/data` child and no quantity child.  The matching qualifier is in
`imas_codex/standard_names/sources/dd_qualifier.py:80-88`: it refuses only
`STR_` string leaves, not `STRUCTURE` or `STRUCT_ARRAY`.

The historical defect is therefore not reproducible at this revision.  The
targeted, side-effect-free extractor admitted all three signal structures:

| Path | DD type and unit | Live source state | Result |
|---|---|---|---|
| `barometry/gauge/pressure` | `STRUCTURE`, `Pa` | `extracted`, no name | admitted |
| `camera_x_rays/detector_humidity` | `STRUCTURE`, `1` | `skipped` by compose model, no name | admitted |
| `gas_injection/valve/flow_rate` | `STRUCTURE`, `Pa.m^3.s^-1` | `extracted`, no name | admitted |

This is the intended signal signature, not a generic container: each has a
same-unit `/data` child plus time/error companions, and no quantity child.  No
change to `dd.py` is warranted: the live extractor already admits all three,
and `test_struct_array_not_skipped_as_string` protects the qualifier boundary.

## Per-path disposition ledger

Every row below has one exact source node and no produced standard name unless
the representative column says otherwise.

| Path | DD evidence | Disposition | Per-path conclusion / representative |
|---|---|---|---|
| `barometry/gauge/pressure` | `STRUCTURE`, `Pa`, `quantity`; documentation `Pressure`; signal `/data` child | A | Physical neutral-gas pressure. Targeted extraction admits it; it has no name and must be composed. |
| `calorimetry/group/component/energy_total/data` | `FLT_0D`, `J`, `quantity`; documentation `Data`; parent `energy_total` documents whole-discharge component energy | A | The leaf is the measured total energy and has no representative. Its parent is also unnamed, so this is not a duplicate representation. |
| `calorimetry/group/component/power` | `STRUCTURE`, `W`, `quantity`; documentation says power extracted from component; signal `/data` child | A | Physical thermal power. It is admitted; its recorded vocabulary gap for generic `component` is a composition problem, not an eligibility refusal. |
| `camera_ir/channel/camera/frame/apparent_temperature` | `FLT_2D`, `K`, `quantity`; documentation identifies a processed infrared surface-temperature image | A | A measured apparent-temperature field with no terminal name. |
| `camera_x_rays/camera/camera_dimensions` | `FLT_1D`, `m`, `quantity`; documentation defines detector dimensions in X1/X2 | A | A camera geometry quantity; the failed compose attempt and position vocabulary gap do not make it ineligible. |
| `camera_x_rays/detector_humidity` | `STRUCTURE`, `1`, `quantity`, lifecycle `alpha`; documentation identifies relative detector humidity; signal `/data` child | A | A dimensionless environmental measurement. The STRUCTURE is admitted; model skipping is not a physics exclusion. |
| `camera_x_rays/detector_humidity/time` | `FLT_1D`, `s`, `coordinate`; documentation `Time` | B | Nested sampling-time axis, not detector humidity itself. |
| `camera_x_rays/detector_temperature/time` | `FLT_1D`, `s`, `coordinate`; documentation `Time` | B | Nested sampling-time axis, not detector temperature. |
| `camera_x_rays/frame/time` | `FLT_0D`, `s`, `coordinate`; documentation `Time` | B | Frame time coordinate, not a camera observable. |
| `core_profiles/profiles_1d/time` | `FLT_0D`, `s`, `coordinate`; documentation `Time` | B | Profile-grid time coordinate, not a plasma profile quantity. |
| `equilibrium/time_slice/constraints/b_field_pol_probe/weight` | `FLT_0D`, `1`, `fit_artifact`, lifecycle `active`; documentation `Weight given to the measurement` | B | Inversion weighting parameter. The live source explicitly records `dd_node_category_ineligible`. |
| `equilibrium/time_slice/constraints/faraday_angle/weight` | `FLT_0D`, `1`, `fit_artifact`, lifecycle `active`; documentation `Weight given to the measurement` | B | Inversion weighting parameter, not a Faraday-angle measurement. |
| `equilibrium/time_slice/constraints/flux_loop/weight` | `FLT_0D`, `1`, `fit_artifact`, lifecycle `active`; documentation `Weight given to the measurement` | B | Inversion weighting parameter. The live source explicitly records `dd_node_category_ineligible`. |
| `equilibrium/time_slice/constraints/n_e_line/weight` | `FLT_0D`, `1`, `fit_artifact`, lifecycle `active`; documentation `Weight given to the measurement` | B | Inversion weighting parameter, not a line-density measurement. |
| `equilibrium/time_slice/contour_tree/node/z` | `FLT_0D`, `m`, `quantity`; parent documentation defines contour-tree critical-point topology | B | The coordinate locates a topology-tree node whose critical type can be X or O; it is not one stable physical locus. It must not be collapsed onto the accepted `vertical_coordinate_of_magnetic_axis`, which only represents explicitly magnetic-axis sources. |
| `equilibrium/time_slice/convergence/iterations_n` | `INT_0D`, unitless, `fit_artifact`, lifecycle `active`; documentation counts convergence-loop iterations | B | Solver convergence bookkeeping, not a physical measurement. |
| `equilibrium/time_slice/profiles_1d/darea_dpsi` | `FLT_1D`, `Wb^-1.m^2`, `quantity`; documentation identifies dA/dpsi of a flux surface | A | Physical equilibrium geometry metric. The failed claim-attempt cap is operational; it remains eligible and unnamed. |
| `equilibrium/time_slice/profiles_1d/pressure` | `FLT_1D`, `Pa`, `quantity`; documentation identifies total plasma pressure on flux surfaces | A | Core equilibrium quantity. No attached or terminal representative was found; it must become nameable. |
| `equilibrium/time_slice/time` | `FLT_0D`, `s`, `coordinate`; documentation `Time` | B | Nested time-slice axis. Its live `temporal_coordinate` refusal is the intended rule. |
| `gas_injection/valve/flow_rate` | `STRUCTURE`, `Pa.m^3.s^-1`, `quantity`; documentation identifies valve-exit volumetric flow; signal `/data` child | A | Physical fuelling actuator flow. The signal-STRUCTURE extractor admits it; a claim-attempt cap is not an eligibility finding. |
| `hard_x_rays/emissivity_profile_1d/half_width_external` | `FLT_1D`, `1`, `quantity`; documentation identifies external normalized-flux half-width | A | Physical profile-width diagnostic. A stale unresolvable-unit skip record conflicts with its current dimensionless unit and must not exclude it. |
| `hard_x_rays/emissivity_profile_1d/time` | `FLT_1D`, `s`, `coordinate`; documentation `Time` | B | Emissivity-profile sampling-time axis, not emissivity. |
| `ic_antennas/antenna/module/strap/distance_to_conductor` | `FLT_0D`, `m`, `geometry`; documentation identifies strap-to-conductor distance | A | Physical antenna geometry parameter with no terminal representative. |
| `spectrometer_visible/channel/isotope_ratios/signal_to_noise` | `FLT_1D`, `dB`, `quantity`; documentation defines log10 signal/reference-band power ratio | C | Already represented by accepted `spectral_signal_to_noise_ratio_of_spectrometer_channel`. The live source has a superseded-binding collision; it needs audited attachment to that representative rather than a new name. |
| `summary/disruption/time/value` | `FLT_0D`, `s`, `quantity`; direct parent is `summary/disruption/time`, whose documentation is `Time of the disruption` | B | Event-time coordinate. It is semantically a time axis despite the terminal segment `value`; the nested-time qualifier only matches a leaf named `time`, so the generic qualifier retains it. This is a manifest/qualifier follow-on, not evidence that an event-time standard name exists. |

## Why the temporal ledger is incomplete

The committed exclusion sidecar contains **114** rows, including **26**
`temporal_coordinate_skip` entries.  It nevertheless lacks the seven time
coordinates above.  Six are caught by the current nested-time rule when passed
through `extract_specific_paths(..., write_side_effects=False)`:
`detector_humidity/time`, `detector_temperature/time`, `frame/time`,
`core_profiles/profiles_1d/time`, `equilibrium/time_slice/time`, and
`emissivity_profile_1d/time`.  The seventh, `summary/disruption/time/value`,
is a time coordinate wrapped by a `time` parent and is not caught because the
nested-time qualifier
tests the leaf token itself.

The evidence shows a stale manifest ledger, not a zero-row or source-seeding
failure.  Regeneration must add the 13 B rows with their category and reason
to `west_production_dd_paths.exclusions.json`, remove them from the 355-source
input, and make the summary `time/value` case structurally ineligible before
the generator is run.  The present worker fence does not authorize writes to
that generated manifest or its sidecar, so those edits are deliberately not
made here.

## Reproducible measurements

* `load_sources_file()` reported 355 paths across 22 IDS; the manifest declares
  the same 355 eligible sources.
* `fetch_manifest_source_release_rows()` returned 355 rows, 25 without a
  terminal identity, and 0 of those 25 without a source node.
* A side-effect-free targeted extraction of all 25 kept 14 and filtered 11:
  the six leaf-time coordinates, four fit weights, and convergence iteration
  count.  It also admitted the three signal structures.
* The focused qualifier suite passed: 84 passed, 0 failed.

## Required follow-on

Generate the manifest and its exclusions sidecar from the classified DD rows,
under an explicit write scope for both generated files.  The generator must
retain the 11 A paths, drop and ledger the 13 B paths, and bind the one C path
to its existing accepted representative.  The `summary/disruption/time/value`
parent-structure case also needs a structural temporal-coordinate rule before
that regeneration; composing names is intentionally outside this evidence
node.
