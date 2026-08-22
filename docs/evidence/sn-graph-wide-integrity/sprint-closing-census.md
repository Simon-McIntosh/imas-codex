# Sprint closing census

## Outcome

**PASS — all 17 live genuine orphans are dispositioned, with 0
undispositioned.** One read-only production invocation derived the cohort from
the graph rather than trusting a declared count, joined it to the committed
physics, Data Dictionary, attachment, and ordinary-review authorities, and
required every identity to enter exactly one closing disposition:

| Closing disposition | Count |
|---|---:|
| `attached` | **9** |
| `held-on-a-named-DD-defect` | **4** |
| `retired-on-physics` | **1** |
| `below-threshold-awaiting-ordinary-review` | **3** |
| **Dispositioned** | **17** |
| **Undispositioned** | **0** |

No identity was deleted. In particular, **0 identities were deleted on
no-measuring-path grounds**. The reverse DD search and its cluster follow-up
remain authoritative: every earlier negative-search row had a measuring path,
so a rank-limited search result never became removal authority.

`attached` is a closing disposition, not an assertion that the legacy
unsourced node gained a producer. Several exact DD paths now bind a canonical
or more-specific identity, and the no-fold decisions intentionally leave the
legacy identity unsourced. Where a live source/lifecycle guard still prevents
the exact legacy spelling from receiving provenance, the row below states that
condition rather than wording it into a live attachment.

## Live-derived cohort and exclusive dispositions

### Attachment or canonical-coverage dispositions: 9

| Identity | Measuring route | Named closing reason |
|---|---|---|
| `capacitance_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/matching_element/capacitance` | The attached source remains bound to `capacitance`; the exact antenna-owned candidate remains the reviewed attachment route, while changing the live target still owes the ordinary semantic guard. |
| `cross_section_of_flux_surface` | `core_profiles/profiles_1d/grid/area` | The live source binds `poloidal_plane_cross_sectional_area_of_flux_surface`. This preserves the locked distinction between poloidal cross-sectional area and swept surface area without folding the ambiguous legacy identity. |
| `line_integrated_electron_density` | `interferometer/channel/n_e_line` | The live source binds canonical `line_integrated_electron_number_density`; the legacy identity remains untouched and unfurled. |
| `neutron_flux_due_to_fusion` | `neutron_diagnostic/neutron_flux_total` | The reviewed candidate route remains `ATTACH` with unit agreement. Its semantic source is currently `failed` and still points at exhausted `power_due_to_fusion_reactions`; the earlier retarget refused because it would remove the incumbent's final producer. This is a named attachment condition, not no-measuring-path authority. |
| `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel` | The live source binds the more concise, state-resolved `parallel_neutral_state_velocity_due_to_diamagnetic_drift`; the measuring path is retained without an identity fold. |
| `parallel_neutral_momentum_diffusion_coefficient` | `plasma_transport/model/profiles_1d/neutral/state/momentum/d_parallel` | The candidate row remains the reviewed attachment route, but the guard correctly found that this DD path is internal-state-resolved while the legacy spelling is species-level. Live parent `neutral_momentum_diffusion_coefficient` has one producer; an exact state-resolved target is required rather than deletion. |
| `poloidal_straight_field_line_angle` | `distributions/distribution/profiles_2d/grid/theta_straight` | Canonical parent `straight_field_line_angle` has two live producers. The failed candidate source is not reattached to the legacy projection and no lineage fold is derived. |
| `tendency_of_total_thermal_plasma_internal_energy` | `summary/global_quantities/denergy_thermal_dt/value` | The live source remains bound to `plasma_internal_energy`, and parent `total_thermal_plasma_internal_energy` retains a producer. The reviewed derivative route remains attachment work; quarantine is not deletion authority. |
| `toroidal_neutral_state_momentum_diffusivity` | `plasma_transport/model/ggd/neutral/state/momentum/d/phi` | The live source retains `toroidal_neutral_internal_state_momentum_diffusion_coefficient` and `toroidal_momentum_diffusivity`. This row is one of the four reported dual-bound residues and needs its signed survivor adjudication; the exact measuring path is present, so the legacy identity is not a no-path deletion candidate. |

### Named Data Dictionary holds: 4

| Identity | Named DD defect and hold condition |
|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` is charge-state resolved and named `fast`, but its prose says deposition into the thermal population while a distinct `power_inside_thermal` sibling carries the same thermal wording. Hold until DD authority resolves the recipient population. |
| `toroidal_ion_charge_state_torque_density` | No DD path carries all three required semantics. The plasma-source `phi` leaf is process-total and toroidal but only ion-species resolved; the distributions leaf is charge-state resolved but only collisional transfer to a thermal recipient. Neither incomplete path may be attached. |
| `x_direction_unit_vector_of_sensor` | The DD parent says unit vector, while `operational_instrumentation/sensor/direction/x` is metre-valued and coordinate-worded. The physically correct direction cosine has unit `1`; hold until the child unit is corrected or the parent is redefined as a displacement. |
| `z_direction_unit_vector_of_sensor` | The same DD contradiction exists for `direction/z`: a unit-vector component is dimensionless, but the child is metre-valued. Hold on the named DD defect rather than altering or deleting the identity. |

### Physics retirement: 1

| Identity | Physics reason |
|---|---|
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | The nearest DD leaf describes torque delivered to a background thermal recipient **by a trapped non-Maxwellian source distribution**. The identity incorrectly makes both `trapped` and `thermal` describe the recipient. Two exact searches found no path for that conflated meaning, so the identity is retired on physics; any replacement must distinguish source and recipient roles. |

### Below threshold, awaiting ordinary review: 3

These are qualified outcomes. Each identity received one fresh joint quorum
draw and remains below the locked 0.85 threshold. No redraw, threshold change,
hand promotion, or direct acceptance occurred.

| Identity | Fresh score | Named reason |
|---|---:|---|
| `minimum_of_safety_factor` | **0.61250** | One permitted ordinary-review draw remained below threshold. |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | **0.36250** | One permitted ordinary-review draw remained below threshold. |
| `toroidal_line_integrated_impurity_ion_velocity` | **0.45000** | One permitted ordinary-review draw remained below threshold. |

## Live integrity counts at close

The same invocation measured every requested comparison class after the latest
attachment, source-reconciliation, owner/geometry, and structural-provenance
transactions had settled:

| Live integrity measure | Count |
|---|---:|
| Dual-bound semantic sources | **4** |
| Semantic sources with no live target | **0** |
| Scalar mirror mismatches | **1** |
| DD projection mismatches | **0** |
| Structurally bare live names, total | **17** |
| Structurally bare with a live child | **0** |
| Structurally bare without a live child | **17** |
| Accepted names | **2,539** |
| Accepted names failing the pinned grammar | **0** |

The active graph grammar and installed package agree exactly at
`0.8.0rc66`; the active graph snapshot contains **22 segments** and **956
tokens**. The 17 childless structurally bare rows are exactly the 17 derived
genuine-orphan rows above, while the childful structural backlog is zero.

## Exact nonmutation proof

This census performed no production graph mutation and made no provider call.
It computed SHA-256
`5fb0df44e422c26036124bd942bf65da3b3b34e8fc921514f6d5469342239aa0`
over the canonical 17-row disposition payload, then queried receipts using
both durable keys:

- `run_id=r-20260822T005026711707-closecensus`
- `manifest_sha256=5fb0df44e422c26036124bd942bf65da3b3b34e8fc921514f6d5469342239aa0`

The exact run-id-plus-manifest query returned **0 `StandardNameChange`
receipts**. A second run-id-only query returned **0 receipts across all
manifest digests**, ruling out a hidden write under another digest. The global
write counters were identical before and after every cohort, grammar, receipt,
and integrity query:

| Graph write measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | **7,786** | **7,786** | **0** |
| `PRODUCED_NAME` relationships | **5,780** | **5,780** | **0** |

## Authority and durable record

- Live plan authority: `imas-codex:sn-graph-wide-integrity` section 3c,
  version **241**, SHA-256
  `41172af56b5f783b059b5b69d51f2855b6f7267423bcc43394a6a08d8ef93896`.
- Source checkout commit:
  `a492a64648e326310143d0b506161767505ca20f`.
- Physics/DD disposition authority:
  `orphan-uncertain-adjudication.md`, SHA-256
  `5231adee725aa00e9c8fa5d567369490f0b18010e1429838dd0642e90d15cc80`.
- Attachment authority:
  `orphan-attachment-candidates.md`, SHA-256
  `6c2fa944e2e5aacd1189f23d78022279fc459461d48bcf2714be68e7eb6a4821`.
- Ordinary-review authority:
  `attachment-target-joint-review.md`, SHA-256
  `e2435ccdff6b16cc481ef544f56e5be30ce78b866b10b2765d97e6d5abd412aa`.
- Machine-readable invocation result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005026711707-closecensus/sprint-closing-census-final.json`.
- Complete invocation diagnostics:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005026711707-closecensus/sprint-closing-census-final.err`.

The machine-readable result passed all set-equality, exclusivity, cardinality,
unit/semantic-authority, grammar, receipt, and nonmutation assertions.
