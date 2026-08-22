# Attachment residue adjudication

## Outcome

**COMPLETE — seven of seven live identities have one disposition.** The five
candidate-path collisions and the two exact governed-preview refusals resolve
to **3 `supersede-into-canonical-owner` + 0 `retire-on-physics` + 1
`preserve-with-distinct-path` + 3 `standing-refusal` = 7**. No attachment,
source migration, review draw, acceptance, signed-manifest construction, or
graph mutation was attempted.

This record joins the complete seven-name set to the production graph in one
read-only invocation at repository revision
`60a37785df823e526f92a7ea1bd880b0a0052249`. A current canonical owner is a
live target reached through either the candidate path's `StandardNameSource ->
PRODUCED_NAME` binding or its `IMASNode -> HAS_STANDARD_NAME` projection. A
failed source pointing at an exhausted target is shown as predecessor context,
not promoted to current canonical ownership.

## One disposition per identity

| Live identity | Live state | Candidate path; current unit | Current canonical owner | Plain-language semantic distinction | Adjudicated disposition | Executability at the checked revision |
|---|---|---|---|---|---|---|
| `cross_section_of_flux_surface` | `name_stage=pending`; quarantined | `core_profiles/profiles_1d/grid/area`; `m^2` | `poloidal_plane_cross_sectional_area_of_flux_surface` (`accepted`, valid); source `dd:core_profiles/profiles_1d/grid/area` is `composed` | Both identities mean the area enclosed by the flux-surface contour in a poloidal plane. The owner states the poloidal plane explicitly and excludes the distinct swept toroidal surface, while the residue is the ambiguous shorter spelling. | `supersede-into-canonical-owner` | **Existing closed signed program.** The exact `supersede` mutation in `apply_signed_manifest` can fold this zero-producer residue into the accepted owner under a fresh exact authority. |
| `fast_ion_charge_state_power_at_inside_flux_surface` | `name_stage=accepted`; valid | `waves/coherent_wave/profiles_1d/ion/state/power_inside_fast`; `W` | `ion_charge_state_power_at_inside_flux_surface` (`accepted`, valid); source `dd:waves/coherent_wave/profiles_1d/ion/state/power_inside_fast` is `attached` | The residue explicitly selects a fast, non-thermal ion charge state; the current owner is population-generic. The DD leaf says `fast`, but the authoritative prose still describes thermal-ion deposition while a separate thermal sibling exists, so neither spelling nor unit settles the recipient. | `standing-refusal` | **Not executable until the external DD condition below is met.** Once it is met, the already-closed ordinary-source migration program can move the existing source; the generic owner retains a derived producer and both fast and thermal structural children. |
| `tendency_of_total_thermal_plasma_internal_energy` | `name_stage=accepted`; quarantined | `summary/global_quantities/denergy_thermal_dt/value`; `W` | `plasma_internal_energy` (`accepted`, quarantined); source `dd:summary/global_quantities/denergy_thermal_dt/value` is `composed` | The residue names a signed time derivative of total thermal plasma energy. The owner's description also talks about a rate, but its spelling says only internal energy and omits tendency, total, and thermal; folding into it would violate the self-describing-name rule. The residue is physically meaningful, but it may not share this already-owned path. | `preserve-with-distinct-path` | **Needs new authority/evidence.** No existing mutation may infer a second path. A distinct unowned scalar DD path must be reviewed first; the residue must also pass sanctioned revalidation before any attachment program can target it. The current path is not attachment authority for this identity. |
| `x_direction_unit_vector_of_sensor` | `name_stage=accepted`; valid | `operational_instrumentation/sensor/direction/x`; `1` | `x_first_measurement_direction_unit_vector_of_strain_gauge` (`accepted`, valid); source `dd:operational_instrumentation/sensor/direction/x` is `attached` | Both are the dimensionless x direction cosine of the primary operational sensor direction. The accepted owner preserves the strain-gauge device and which of its distinct measurement-direction vectors is represented; the generic sensor residue drops both distinctions. A live `backfill_refine` change records the exact residue-to-owner lineage. | `supersede-into-canonical-owner` | **Existing closed signed program.** The exact `supersede` mutation can close the recorded lineage without moving the source. |
| `z_direction_unit_vector_of_sensor` | `name_stage=accepted`; valid | `operational_instrumentation/sensor/direction/z`; `1` | `z_first_measurement_direction_unit_vector_of_strain_gauge` (`accepted`, valid); source `dd:operational_instrumentation/sensor/direction/z` is `attached` | Both are the dimensionless z direction cosine of the primary operational sensor direction. The accepted owner retains strain-gauge ownership and the distinct-vector role; the generic residue loses them. A live `backfill_refine` change records the exact residue-to-owner lineage, while a separate change proves that `direction_second/z` is a different vector and must not be collapsed into this row. | `supersede-into-canonical-owner` | **Existing closed signed program.** The exact `supersede` mutation can close the recorded lineage without moving the source. |
| `neutron_flux_due_to_fusion` | `name_stage=accepted`; valid | `neutron_diagnostic/neutron_flux_total`; `s^-1` (identity unit `Hz`, dimensionally equal) | —. Predecessor context only: failed source `dd:neutron_diagnostic/neutron_flux_total` still points to exhausted, quarantined `power_due_to_fusion_reactions` | The residue is a neutron production rate. The predecessor target is fusion power, an energy rate, so it is not a semantic owner of this count-rate path. The preview refusal is not dimensional: it protects the predecessor's last producing source even though that binding is physically wrong. | `standing-refusal` | **Not executable until the predecessor condition below is met.** After that condition, the already-closed ordinary-source migration program admits a non-stale source and an accepted-valid new target, but the current last-producer guard correctly blocks it. |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | `name_stage=reviewed`; valid | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol`; `m.s^-1` | —; no `StandardNameSource` currently backs the candidate path | The identity and path agree exactly: both are the poloidal effective convection velocity in the neutral internal-state momentum equation. There is no competing semantic owner. The refusal is solely the acceptance boundary: a reviewed name is not catalog authority for a producing source. | `standing-refusal` | **Needs new machinery to lift the circular gate.** The checked revision has no closed source-bootstrap transition that can ground steering without first treating a reviewed name as accepted, and the signed attachment boundary must not be weakened. |

The three supersedes are the only dispositions immediately expressible by an
existing closed signed program. No row is `retire-on-physics`: each identity is
either a redundant spelling of a live accepted owner or retains coherent
physics that must not be deleted. The preservation row and the reviewed-target
refusal need new machinery or newly reviewed authority. The fast-ion and
neutron refusals need external evidence/state changes first, after which the
existing ordinary-source migration program is the applicable closed program.

## Exact lifting conditions for standing refusals

| Standing-refusal identity | Condition that lifts the refusal |
|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | The active Data Dictionary must incorporate an authoritative resolution of the filed `power_inside_fast` contradiction: either its prose must unambiguously identify fast-ion deposition, or the leaf must be renamed/redefined so its recipient agrees with the distinct thermal sibling. The graph must then be rebuilt from that DD version and the candidate/owner closure re-read before a signed ordinary-source migration is authored. Unit `W` alone does not lift this hold. |
| `neutron_flux_due_to_fusion` | `dd:neutron_diagnostic/neutron_flux_total` must cease to be the last producing source of exhausted `power_due_to_fusion_reactions`. That occurs only if the old power identity gains an authoritative unit-`W` replacement producer, or a separately signed disposition retires/supersedes that exhausted target and preserves its permanent history. Only then may the failed count-rate source be migrated to the accepted neutron-rate identity. |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | A governed source-bootstrap/steering transition must break the present cycle without direct acceptance: it must ground the exact DD path strongly enough for sanctioned `sn edit --hint` review while not creating an authoritative `PRODUCED_NAME` binding to a reviewed target. A fresh permitted review/refine result must then set `name_stage=accepted` with `validation_status=valid`; only that exact state lifts the attachment refusal. A blind redraw or hand acceptance does not. |

## Machinery boundary

| Disposition class in this record | Existing closed signed route | New machinery or evidence still required |
|---|---|---|
| Three `supersede-into-canonical-owner` rows | `apply_signed_manifest` with one exact `RepairMutationKind.supersede` program per governed cohort | Fresh participant closure, authority hashes, dry run, apply receipt, and replay are still required operational evidence; this adjudication signs none of them. |
| Fast-ion `standing-refusal` | Closed ordinary-source migration after the DD contradiction is resolved | Upstream DD resolution and a refreshed exact owner/unit read. |
| Neutron `standing-refusal` | Closed ordinary-source migration after the last-producer condition is lifted | Authoritative replacement producer or separate signed disposition for the exhausted predecessor. |
| `preserve-with-distinct-path` tendency row | None for selecting a path; mutation machinery cannot invent semantic authority | A reviewed distinct unowned scalar path plus sanctioned revalidation; source attachment at this checked revision also lacks a closed standalone signed program. |
| Reviewed-target `standing-refusal` | None | A governed source-bootstrap/steering transition that preserves the acceptance boundary, followed by an earned accepted-valid result and a closed source-attachment program. |

`persist_claimed_attachments` is not counted as a closed signed route: it is a
claim-fenced pipeline persistence kernel. Likewise, the rollback-only preview
driver is evidence, not production authority. Any source-attachment program
landed concurrently must be audited and integrated before it changes this
checked-revision classification.

## Read-only counter proof

The invocation counted both requested measures before and after all seven
identity, DD-path, owner, source, lineage, and target-closure reads.

| Counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | **7,874** | **7,874** | **0** |
| `PRODUCED_NAME` | **5,774** | **5,774** | **0** |

The complete query driver and output are retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T210116952076-n-residueadj/query_attachment_residue.py`
and
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T210116952076-n-residueadj/live-graph-query.log`;
the log terminates with `EXIT=0`. This node is read-only on the graph and signs
no manifest.
