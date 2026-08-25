# Standard Name corpus health census

Measured live, read-only, from **2026-08-25T14:51:53.796384Z** through
**2026-08-25T14:52:15.600516Z**. Source checkout:
`72ebf606b507ce1a23eac7534cafef71d55d2150`; live plan authority:
`imas-codex:sn-graph-wide-integrity` section 8, version 280.

## Outcome

The corpus is not uniformly healthy. Three source projections remain clean,
but two previously governed residue classes and one regrown class are live:

- **5** composed/attached sources have no live target;
- **0** sole-live-target `produced_sn_id` scalar mismatches remain;
- **0** DD/signal upstream projection mismatches remain;
- **1** source has multiple live targets; and
- **1** live `HAS_PARENT` relocation tip is not authorized by current
  `derive_edges` output.

Score state is likewise qualified. Of **2,302** names accepted on the name
axis, **14** scored name reviews are below 0.85, **30** docs scores are below
0.85, and **327** carry no name-axis reviewer score. Provenance also remains
mixed: **1,091** names have null `origin`, of which **345** are live and
**278** are accepted; **4** accepted names have no producing
`StandardNameSource`.

This is a census, not repair authority. The measurement queried the production
graph through `GraphClient`, called no model, and performed no graph mutation.

## Score

### Lifecycle distribution

Both axes cover the complete **4,658-name** corpus. `name_stage` is populated
on 4,658/4,658 nodes. `docs_stage` is populated on 4,654/4,658; the four nulls
are reported explicitly rather than silently omitted.

| `name_stage` | Count |
|---|---:|
| accepted | 2,302 |
| drafted | 20 |
| exhausted | 272 |
| pending | 5 |
| reviewed | 143 |
| superseded | 1,916 |

| `docs_stage` | Count |
|---|---:|
| null | 4 |
| accepted | 2,962 |
| drafted | 9 |
| exhausted | 11 |
| pending | 1,644 |
| reviewed | 26 |
| superseded | 2 |

The axes are independent: historical terminal name rows may retain accepted
documentation. The requested cross-axis backlog is **30 names accepted on the
name axis but not accepted on the docs axis**.

### Reviewer-score distribution over accepted names

Here “accepted names” means the common population
`name_stage = 'accepted'`, so both requested score distributions use the same
2,302 identities.

| Property over name-accepted identities | Population | Non-null scores | Null scores | Minimum | Median | Below 0.85 |
|---|---:|---:|---:|---:|---:|---:|
| `reviewer_score_name` | 2,302 | 1,975 | 327 | 0.575 | 0.98125 | **14** |
| `reviewer_score_docs` | 2,302 | 2,302 | 0 | 0.300 | 0.91875 | **30** |

Representative low name-axis scores preserve the actual qualified state:

| Identity | Origin | Name score |
|---|---|---:|
| `ion_heating_power` | pipeline | 0.5750 |
| `spectral_etendue_of_spectrometer_channel` | catalog_edit | 0.6625 |
| `normalized_effective_particle_energy` | catalog_edit | 0.7625 |
| `safety_factor_at_plasma_boundary` | null | 0.7625 |
| `atomic_mass_of_wall_material` | catalog_edit | 0.7750 |

Representative low docs scores explain the 30-name documentation backlog:

| Identity | `docs_stage` | Origin | Docs score |
|---|---|---|---:|
| `fluctuating_ion_current_density` | reviewed | derived | 0.3000 |
| `energy_flux_at_limiter` | reviewed | derived | 0.3250 |
| `counter_passing_current_density` | reviewed | derived | 0.35625 |
| `volume_averaged_runaway_electron_current_density` | reviewed | derived | 0.4000 |
| `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` | exhausted | pipeline | 0.4500 |

For axis-specific context, the **2,962** rows with
`docs_stage = 'accepted'` contain 2,629 non-null docs scores and 333 nulls;
their scored minimum is 0.85, median 0.9125, and count below 0.85 is **0**.
That zero is not substituted for the requested shared-population result above:
the 30 low docs scores live precisely on name-accepted rows whose docs axis has
not accepted.

## Provenance

### Counts by `origin`

| Origin | Count |
|---|---:|
| null | **1,091** |
| `catalog_edit` | 2,096 |
| `derived` | 231 |
| `pipeline` | 1,240 |
| **Total** | **4,658** |

The null-origin partition contains **345 live names** under the corpus live
predicate (non-null stage and neither superseded nor exhausted), including
**278 accepted names**. The schema sanity pass found non-null `origin` on
3,567/4,658 names, proving that null is a real provenance value in this corpus,
not a guessed property returning null for every node.

### Accepted names without a producing source

Exactly **4** accepted names have no incoming authored
`StandardNameSource -[:PRODUCED_NAME]-> StandardName` relationship:

| Identity | Origin | `docs_stage` |
|---|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | pipeline | accepted |
| `neutron_flux_due_to_fusion` | pipeline | exhausted |
| `tendency_of_total_thermal_plasma_internal_energy` | pipeline | accepted |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | pipeline | accepted |

These are qualified unsourced identities, not evidence that the source
relationship is absent globally: the same pass counted **5,315** authored
`PRODUCED_NAME` relationships with complete source id, target id, and target
stage coverage.

## Stability

The source predicate matches the production invariant: semantic sources
have status `composed` or `attached`; a live target is a `PRODUCED_NAME` target
whose `name_stage` is neither `superseded` nor `exhausted`; scalar and upstream
projection parity are judged only for a sole live target.

| Live class | Count | Candidate/control population | Verdict |
|---|---:|---:|---|
| No live target | **5** | 5,054 semantic sources; 5,049 have at least one live target | regrown defect class |
| `produced_sn_id` scalar mismatch | **0** | 5,048 sole-live-target sources; 5,048/5,048 have a non-null scalar | clean at measurement |
| Upstream DD/signal projection mismatch | **0** | 4,576 sole-live DD/signal sources; 4,576/4,576 have a backing and at least one mapped identity | clean at measurement |
| Multiple live targets | **1** | 5,054 semantic sources | recorded residual remains live |
| Unauthorized `HAS_PARENT` relocation tip | **1** | 73 current non-self successor-rewire candidates; 1,473 live parent edges exactly match current derivation | recorded refusal remains live |

### Sources with no live target

All five have one stored target, scalar, and upstream projection, but that
target is terminal under the live predicate:

| Source | Status | Terminal target retained by edge/scalar/projection |
|---|---|---|
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/poloidal_turns` | composed | `poloidal_turn_count` |
| `dd:iron_core/segment/geometry/arcs_of_circle/r` | composed | `radial_coordinate_of_arc_of_circle_center` |
| `dd:pellets/time_slice/pellet/path_profiles/position/r` | composed | `radial_coordinate_of_pellet_path` |
| `dd:summary/line_average/dn_e_dt/value` | attached | `time_derivative_of_electron_density` |
| `dd:summary/local/pedestal/q/value` | composed | `safety_factor_at_pedestal` |

The earlier section-8 closure measured this class at zero; the present count is
therefore reported as current regrowth, not worded into a pass.

### Multiple live targets

The one source is
`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial`, still bound
to both `radial_neutral_internal_state_momentum_flux` and
`radial_neutral_state_momentum_flux`. Its scalar selects the former. This is the
same last-producer-blocked residual class, but this census only establishes
that it remains live; it does not reuse a historical refusal as mutation
authority.

### Unauthorized structural relocation

The production-admission-aware scan processed **2,470 live names**, emitted
**2,118** raw and **1,503** admitted derived-parent rows, and formed **73**
non-self successor-rewire candidates. One candidate still has a live edge on a
tip that current `derive_edges(child)` does not authorize:

| Child | Unauthorized incumbent tip | Current derivation parent | Relationship properties |
|---|---|---|---|
| `ratio_of_parallel_ion_velocity_to_magnetic_field_magnitude` | `parallel_bulk_ion_velocity` | `parallel_ion_velocity` | `operator=ratio`, `operator_kind=binary`, `role=a`, `separator=to` |

This is the exact previously refused relationship, remeasured from current
derivation and current topology. The current count is **1**, not the historical
78-row projection or the 73-row pre-apply cohort.

## Zero validity and positive controls

Cypher returns plausible zeros for missing properties and wrong relationship
directions. Each reported zero above therefore has a non-empty candidate set,
property/endpoint coverage, and a positive control from the same invocation:

| Reported zero | Candidate and schema sanity | Positive control, explicitly aimed |
|---|---|---|
| Sole-live scalar mismatch = 0 | 5,048 sole-live sources; `produced_sn_id` non-null on 5,048/5,048. Globally, `StandardNameSource.id/status/source_type` cover 9,668/9,668. | The same scalar comparator read all 5,048 populated scalars against their exact sole live target. |
| Upstream projection mismatch = 0 | 4,576 sole-live DD/signal candidates; 4,576/4,576 have a backing and mapping. The graph has 4,937 `HAS_STANDARD_NAME` edges with backing id and target id on 4,937/4,937. | The same projection join positively found at least one mapped identity for all 4,576 candidates. |
| Docs-axis-accepted score below 0.85 = 0 | 2,962 docs-axis-accepted candidates, 2,629 non-null scores; `reviewer_score_docs` exists on 2,696 graph nodes. | The same threshold instrument found the non-zero 30-row below-threshold cohort on the requested name-accepted population and a minimum of 0.85 on the docs-axis-accepted scored subset. |
| Reversed `PRODUCED_NAME` direction = 0 | Both endpoint labels and ids were checked; the authored direction has complete source/target/stage coverage on 5,315/5,315 edges. | **Directional positive control:** the same pass found 5,315 edges in the schema-authored `StandardNameSource -> StandardName` direction, explicitly controlling the reverse-direction zero. |

The no-live-target instrument also carries an aimed positive control even
though its reported count is non-zero: it saw **5,049** semantic sources with a
live target and distinguished **1** multiple-live source. The structural
instrument likewise saw **1,473** exact property-map matches between live
`HAS_PARENT` edges and current `derive_edges` output before flagging the one
unauthorized relocation. These controls prove the instruments can see the
objects they classify and are pointed at the authored schema directions.

Core schema coverage from the same pass:

| Surface | Covered / candidates |
|---|---:|
| `StandardName.id` | 4,658 / 4,658 |
| `StandardName.name_stage` | 4,658 / 4,658 |
| `StandardName.docs_stage` | 4,654 / 4,658, with 4 explicit null-stage rows |
| `StandardNameSource.id` | 9,668 / 9,668 |
| `StandardNameSource.status` | 9,668 / 9,668 |
| `StandardNameSource.source_type` | 9,668 / 9,668 |
| Authored `PRODUCED_NAME` source id / target id / target stage | 5,315 / 5,315 on each field |
| `HAS_STANDARD_NAME` backing id / target id | 4,937 / 4,937 on each field |
| `HAS_PARENT` child id / parent id / `operator_kind` | 1,485 / 1,485 on each field |
| `REFINED_FROM` tip id / predecessor id | 1,563 / 1,563 on each field |
| `IMASNode.id` | 61,366 / 61,366 |
| `FacilitySignal.id` | 46,872 / 46,872 |

## Reproducibility

- Structured census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T144323649624-n-snhealthcensus/live-health-census.json`
  — SHA-256
  `dee1faf233d52e8b113ee76df45b2955644855aa3814079b0dba53fb57c92b2e`.
- Full command log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T144323649624-n-snhealthcensus/live-health-census.log`
  — SHA-256
  `dee1faf233d52e8b113ee76df45b2955644855aa3814079b0dba53fb57c92b2e`.
- Read-only census driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T144323649624-n-snhealthcensus/run_health_census.py`
  — SHA-256
  `cd1537baf268142b157b9a9ee8a5cfb8fd1a80b84884b663540d83be97c6ddbf`.

The structured result is the full record: it contains all 14 low name-score
rows, all 30 low docs-score rows, the four unsourced accepted identities, all
five no-live-target sources, the dual-bound row, and the unauthorized
relocation row.
