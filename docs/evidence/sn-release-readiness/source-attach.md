# Governed attachment of five ungrounded Standard Names

## Verdict

**PASS.** The five attachable live identities now each resolve exactly one
`StandardNameSource`, compared with zero before the operation. All five selected
DD sources have the same unit string as their target `StandardName`; no identity
string changed. The write used the signed-manifest
`unbound-ordinary-source-attachment` program, not raw Cypher. It produced five
`StandardNameChange` receipts under run
`r-20260901T160441383261-n-sourceattach`. The only remaining accepted,
chain-cap, ungrounded identity is
`first_local_tangential_back_surface_radius_of_optical_element`, which section
12a explicitly excludes because its ordinal identity is a naming defect.

## Selected bindings and semantic deltas

The before and after counts below count distinct
`(:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName)` bindings for the exact
identity. Each postflight row also proved `source.status='attached'`,
`source.produced_sn_id = StandardName.id`, the source's `FROM_DD_PATH` target,
the DD unit, and the target's updated `source_paths` mirror.

| Standard Name identity | Selected DD path | SN unit | DD unit | Bindings before -> after | Semantic delta between the name and DD quantity |
|---|---|---:|---:|---:|---|
| `electron_temperature_at_separatrix` | `summary/local/separatrix_average/t_e/value` | `eV` | `eV` | **0 -> 1** | The DD quantity is explicitly flux-surface averaged over the separatrix. The name states the separatrix locus but leaves the **aggregation unstated**. No locus is missing. |
| `ion_species_particle_flux_at_wall_due_to_surface_emission` | `wall/description_ggd/ggd/particle_fluxes/ion/emitted/values` | `m^-2.s^-1` | `m^-2.s^-1` | **0 -> 1** | Both sides identify species-level ion particle flux emitted by the wall. The DD stores one scalar per GGD grid-subset element; that **per-element GGD representation/locus detail is unstated**. There is no additional aggregation. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/power_launched` | `W` | `W` | **0 -> 1** | Both sides describe launched ICRH power into the vessel. The DD value is for an individual antenna **module**; the module locus is unstated in the name. There is no sum or other aggregation in the selected DD quantity. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/t_radiation_central/rho_tor_norm` | `1` | `1` | **0 -> 1** | Both sides locate an ECE measurement/emission point by normalized toroidal flux coordinate. The DD path is the **central-radiation measurement** coordinate, whereas the name expresses an ECE-channel emission-position locus; the DD's central-measurement locus is unstated by the name and the name's channel wording is not literal in the path. No numeric aggregation is declared. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear` | `T` | `T` | **0 -> 1** | Both sides identify magnetic field at a poloidal-field probe. The DD quantity is specifically corrected for the probe's **non-linear response**, a qualifier unstated in the name. The probe locus is stated and there is no aggregation. |

The postflight returned the requested identity and current identity as identical
for all five rows. Independently, every signed receipt has
`from_name == to_name`, so the operation changed bindings and source mirrors but
changed **0 of 5 identity strings**.

## Rejected candidates

Unit equality was applied first, then semantic/lifecycle/authority fit. A unit
match did not override a state, aggregation, locus, lifecycle, or incumbent
authority mismatch.

| Target identity | Candidate not selected | Candidate unit | Why rejected |
|---|---|---:|---|
| `electron_temperature_at_separatrix` | `summary/local/separatrix/t_e/value` | `eV` | Unit-equal and linguistically close, but section 12a names the separatrix-average quantity as the governed source and requires the unstated-aggregation delta to remain explicit; substituting the non-averaged path would change that authored semantic authority. |
| `ion_species_particle_flux_at_wall_due_to_surface_emission` | `wall/description_ggd/ggd/particle_fluxes/ion/state/emitted/values` | `m^-2.s^-1` | The repository pairing guard rejected it exactly: the DD path is charge-state-resolved while the Standard Name is species-level. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `ic_antennas/power_launched` | `W` | This is the whole ICRH system summed over antennas and is alpha lifecycle; both the system-wide locus and aggregation exceed the singular antenna identity. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `summary/heating_current_drive/power_launched_ic/value` | `W` | This is total launched IC power across antennas; the aggregation and plurality are not present in the identity. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/power_launched` | `W` | Semantically closer, but its `StandardNameSource` is already attached to `power_due_to_ion_cyclotron_heating`; taking it would require a governed migration and would remove another live name's authority, not perform the authorized unbound attachment. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/channel/position/rho_tor_norm` | `1` | The exact channel-position source is already attached to `normalized_toroidal_flux_coordinate_at_measurement_position`; using it would be an authority migration with collateral, not an unbound attachment. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/channel/beam_tracing/beam/position/rho_tor_norm` | `1` | The locus is a point along a beam/ray trace, not the central ECE emission/measurement position, and the source is already bound to the beam-tracing identity. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/t_radiation_central_o/rho_tor_norm` | `1` | Adds an O-mode polarization restriction absent from the target identity. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/bpol_probe/non_linear_response/b_field_non_linear` | `T` | Unit-equal and preserved in the old scalar mirror, but its DD lifecycle is removed. The selected `b_field_pol_probe` path is the active replacement with the same physical quantity. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/b_field_pol_probe/non_linear_response/b_field_linear` | `T` | Represents the assumed linear response, not the non-linear-corrected probe field selected by the live quantity. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/b_field_pol_probe/field` | `T` | A generic measured-field structure already bound to `poloidal_magnetic_field`; using it would steal an incumbent source and omit the selected non-linear-response qualifier. |

### Unit-mismatched rows refused before authority construction

The following semantically adjacent rows were explicitly named and excluded;
none appears in the signed authority and none was bound:

| Intended target | Rejected DD row | SN unit | DD unit | Refusal |
|---|---|---:|---:|---|
| `electron_temperature_at_separatrix` | `summary/local/separatrix_average/n_e/value` | `eV` | `m^-3` | Electron density is dimensionally incompatible with temperature. |
| `ion_species_particle_flux_at_wall_due_to_surface_emission` | `wall/description_ggd/ggd/energy_fluxes/kinetic/ion/emitted/values` | `m^-2.s^-1` | `W.m^-2` | Kinetic energy flux is dimensionally and physically distinct from particle flux. |
| `launched_power_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/frequency` | `W` | `Hz` | Antenna frequency is not power. |
| `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `ece/channel/position/psi` | `1` | `Wb` | Dimensional poloidal magnetic flux is not a normalized toroidal flux coordinate. |
| `poloidal_magnetic_field_of_magnetic_field_probe` | `magnetics/b_field_pol_probe/voltage` | `T` | `V` | Probe terminal voltage is not magnetic field. |

Thus the unit-refusal census is **5 named mismatched rows, 0 bound**. The signed
cohort is **5 unit-equal rows, 5 admitted, 0 refused by the operator**.

## Governed operator and replay evidence

- Operator: `apply_signed_manifest` with operation ID
  `unbound-ordinary-source-attachment` and receipt operation
  `attach_unbound_standard_name_source`.
- Recorded reason: `attach five unit-matched DD sources after row-level semantic adjudication`.
- Run ID: `r-20260901T160441383261-n-sourceattach`.
- Authority file SHA-256:
  `15616395f6762a12ccf3d12a25bac4d55114b3d67c7ce500dfce543c20944641`.
- Signed payload SHA-256:
  `eb506ee9fd82b89891fba9024a11944821123c78527b3a4648cad3dd63dbd4cd`.
- Authorized manifest SHA-256:
  `fdfb685dbb655224dbca01240bb1655c84c71aa6af8c6e6d48d99b781cf1e26c`.
- Preview: `authority_rows=5`, `admitted=5`, `refused=0`,
  `would_change=5`.
- Apply evidence: five exact `StandardNameChange` receipt rows exist, one per
  source row, and all five postconditions resolve.
- Replay: `outcome=already_applied`, `changed=0`, `receipt_rows=5`,
  `persistent_writes=0`.

The generic signed operator creates the source edge, backing
`IMASNode-[:HAS_STANDARD_NAME]` projection, source lifecycle/scalar update,
target `source_paths` mirror, and immutable receipt under one compare-and-set
manifest. No raw Cypher mutation was used for the binding.

## Quantitative graph closure

| Measure | Before | After |
|---|---:|---:|
| Five requested identities resolving at least one authoritative source | **0/5** | **5/5** |
| Exact source bindings across the five identities | **0** | **5** |
| Selected sources with equal SN/DD unit strings | **5/5** | **5/5** |
| Requested identity strings changed | **0/5** | **0/5** |
| Signed receipt rows for this run | **0** | **5** |
| Accepted chain-cap identities still lacking a source | **6** | **1** |

The remaining 1 is exactly
`first_local_tangential_back_surface_radius_of_optical_element`; it was not
silently attached.

## Attributable cost

The postflight queried `LLMCost` rows where either `for_run` or `run_id` equals
`r-20260901T160441383261-n-sourceattach`. It returned:

- attributable `LLMCost` rows: **0**;
- exact attributable spend: **USD 0.000000**;
- authorized ceiling: **USD 15.000000**;
- unused headroom: **USD 15.000000**.

No model call was needed: the DD search, unit comparison, signed preview,
apply, and postflight are deterministic graph/operator work.
