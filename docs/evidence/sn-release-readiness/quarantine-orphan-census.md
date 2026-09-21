# Quarantine and orphan census

provisional: false

Measured 2026-09-21 against the live production graph through
`GraphClient()`, resolved URI `bolt://98dci4-gpu-0002:7687`. Every statement
below is `MATCH`/`RETURN` only — no property, relationship or node was
written. `run_audits` was deliberately **not** used as the validator: a prior
audit called 23 of 35 quarantines stale and a revalidation cleared none, so
the instrument is named beside every number here and the re-check uses
`validate_name_candidate`, the gate that wrote the status in the first place.

## 1. The quarantine figure reproduces exactly

Query:

```cypher
MATCH (sn:StandardName) WHERE sn.validation_status='quarantined'
RETURN count(sn) AS n
```

| Measure | Live | Query |
|---|---:|---|
| All `StandardName` | 5,130 | `MATCH (sn:StandardName) RETURN count(sn)` |
| `validation_status='quarantined'` | **660** | above |
| `validation_status='valid'` | 4,457 | group-by on `validation_status` |
| `validation_status='pending'` | 11 | same |
| `validation_status IS NULL` | 2 | same |

660 is confirmed, not carried forward. The status axis partitions the label
completely (660+4,457+11+2 = 5,130), so nothing is hiding in a fourth value.

### By lifecycle stage

```cypher
MATCH (sn:StandardName) WHERE sn.validation_status='quarantined'
RETURN coalesce(sn.name_stage,'<null>') AS stage, count(*) AS n ORDER BY n DESC
```

| `name_stage` | Quarantined | Exportable stage? |
|---|---:|---|
| superseded | 330 | no |
| exhausted | 266 | no |
| **accepted** | **46** | **yes** |
| drafted | 11 | no |
| reviewed | 6 | no |
| pending | 1 | no |

**596 of the 660 (90.3%) are already terminal** — superseded or exhausted —
and are unreachable by any export gate regardless of their validation status.
The live population that matters to a release is the 46 accepted rows, which
is the same class the 2026-08-23 census measured at 47.

## 2. Quarantine by recorded cause — and 254 have no recorded cause at all

`validation_issues` is the property `mark_names_validated` writes as the
quarantine's cause. Classifying every issue line of all 660 by its prefix:

| Recorded cause (per issue line) | Lines |
|---|---:|
| `parse_error:` — grammar round-trip failure | 252 |
| `audit:repeated_token_check` | 28 |
| `audit:unit_dimension_check` | 26 |
| `audit:latex_def_check` | 21 |
| `audit:canonical_locus_check` | 21 |
| `audit:cumulative_prefix_check` | 20 |
| `audit:name_unit_consistency_check` | 19 |
| `[canonical]` ISN layer | 18 |
| `[pydantic:scalar.name]` | 16 |
| `audit:implicit_field_check` | 9 |
| `audit:derived_parent_structure_check` | 9 |
| `[semantic]` ISN layer | 8 |
| `audit:symbol_units_check` | 5 |
| `audit:description_verb_drift_check` | 4 |
| `[strict_grammar]` | 4 |

Reduced to the *critical* predicate `_is_quarantined` actually fires on
(`imas_codex/standard_names/workers.py:4759`), per identity:

| Critical cause reconstructible from the stored record | Identities |
|---|---:|
| grammar `parse_error` | 252 |
| pydantic layer failed | 14 |
| ISN `ERROR -` (semantic/structural) | 5 |
| **none — the stored record carries no critical cause** | **389** |

**254 of the 660 carry a completely empty `validation_issues` list.** A
quarantine whose cause list is empty cannot be justified from the graph, and
the remaining 135 of the 389 carry only advisory audit lines that
`_is_quarantined` does not treat as critical.

### The stamp is missing too

| Measure | Live |
|---|---:|
| Quarantined with `validated_at IS NULL` | **596 of 660** |
| Quarantined with a `validated_at` stamp | 64 |

`mark_names_validated` always sets `validated_at` alongside
`validation_status`, so a quarantined row with a null stamp **was not written
by the validation worker**. 596 rows are in that state, and they coincide
exactly with the 596 terminal (superseded/exhausted) rows counted in §1.

Cause presence by stage:

| Stage | Cause recorded | Cause EMPTY |
|---|---:|---:|
| superseded | 322 | 8 |
| exhausted | 38 | 228 |
| accepted | 28 | **18** |
| drafted | 11 | 0 |
| reviewed | 6 | 0 |
| pending | 1 | 0 |

## 3. Hand-check: re-running the real gate on all 46 accepted quarantines

The check is **not** `run_audits` — a prior audit called 23 of 35 quarantines
stale using it and a `--revalidate` cleared none. Instead each identity was
re-projected with the exact fields `claim_names_for_validation` supplies
(`imas_codex/standard_names/graph_ops.py:8285-8293`) and passed through
`validate_name_candidate`, the gate that writes the status. Sample = the whole
accepted cohort, 46 identities, which exceeds the 20 the measure requires.

**Positive control**: 20 accepted `validation_status='valid'` names were run
through the same call in the same process. All 20 returned `valid`, so the
harness can return both verdicts and a `valid` result is a real result.

| Re-check verdict on the 46 | Identities |
|---|---:|
| `quarantined` — the recorded cause still holds | **23** |
| `valid` — the recorded cause no longer reproduces | **23** |

Split against whether a cause was recorded at all:

| Stored record | Re-check `quarantined` | Re-check `valid` |
|---|---:|---:|
| cause recorded | 23 | 5 |
| cause EMPTY | 0 | **18** |

Every one of the 18 causeless quarantines clears, and 5 identities carrying a
recorded cause clear as well. The 23 that survive fail on reproducible audit
predicates — `derived_parent_structure_check`, `repeated_token_check`,
`unit_dimension_check` — and two of them (`plasma_heating_power`,
`width_of_spectrometer_channel`) now fail on a *different* check than the one
stored, so the stored cause is stale even where the verdict is not.

Cleared identities (recheck `valid`, stored `quarantined`):
`magnetic_field_magnitude`, `power_of_lower_hybrid_antenna`,
`source_rate_due_to_injection`, `tungsten_density`, `length_of_antenna_strap`,
`power_of_beam_tracing_beam`, `toroidal_current_density`, `major_radius`,
`radius_of_plasma_filament`, `radius_of_poloidal_field_coil`,
`count_at_detector_pixel`, `energy_density`, `ion_pressure`,
`ion_state_momentum_diffusivity`, `diamagnetic_current_density`,
`total_current_density`, `normalized_perturbed_pressure`,
`perpendicular_normalized_perturbed_pressure`, `flux_at_first_wall`,
`volume_integrated_runaway_electron_density`, `particle_count`,
`time_derivative_of_electron_density`, `cumulative_inside_flux_surface_torque`.

**Verdict on the classification: half right.** 23 of 46 accepted quarantines
are correctly classified and 23 are not — they are held out of the catalog by
a status no current check reproduces. The count 23 coincides numerically with
the earlier audit's "23 of 35 stale", but the denominators differ and the
identity sets were not compared; treat the match as coincidence until checked.

Raw re-check record: `/tmp/qcensus/recheck.json` (transient; the verdicts and
cleared ids above are the durable form).

## 4. The orphan figure of 170 reproduces under no definition tested

"Orphan" in this portfolio means an identity with no producing source — no
`(:StandardNameSource)-[:PRODUCED_NAME]->(sn)` edge (`unbound-source-backlog`
§ on the nine exportable orphans). Ten candidate definitions were measured
live; **none returns 170**:

| Definition | Live count |
|---|---:|
| no producing source, any stage | 2,341 |
| no producing source, non-terminal | **21** |
| no producing source and no sourced ancestor within 6 `REFINED_FROM` hops, non-terminal | 21 |
| no producing source, exportable stages (`accepted`/`approved`) | **9** |
| no incoming edge of any type except `FOR_STANDARD_NAME` | 549 |
| quarantined and no producing source | 544 |
| `HAS_PARENT` target not accepted | 257 |
| exhausted with no producing source | 244 |
| `StandardNameSource` with `status='extracted'` and no `PRODUCED_NAME` | 1,359 |
| `run_id` set but no `SNRun` node of that id | 557 |

The instrument is not blind: the same queries return 2,341, 549 and 1,359 on
adjacent definitions, so a zero was never the risk — 170 simply is not a
population this graph holds under any reading of the word tested here. The
two figures that *do* carry release meaning are **21** live orphans and **9**
exportable ones, matching the unbound-source plan's own count.

### Orphans by `name_stage`

```cypher
MATCH (sn:StandardName) WHERE NOT ()-[:PRODUCED_NAME]->(sn)
RETURN coalesce(sn.name_stage,'<null>') AS stage, count(*) AS n ORDER BY n DESC
```

| `name_stage` | Orphans | Exportable stage? |
|---|---:|---|
| superseded | 2,076 | no |
| exhausted | 244 | no |
| **accepted** | **9** | **yes** |
| pending | 8 | no |
| reviewed | 3 | no |
| drafted | 1 | no |

2,320 of 2,341 (99.1%) are terminal tombstones and freed identities, which is
the documented and explained shape. The exportable population is 9.

## 5. Does any of them reach the WEST batch? — **No**

The WEST batch is not a stage or a facility predicate: it is the committed
manifest `imas_codex/standard_names/manifests/west_production_dd_paths.yaml`
(`name: west-task-2e`, 342 DD v4 source ids), resolved to standard names by
`fetch_manifest_source_release_rows` and passed to `_fetch_candidates` as
`batch`. The question was answered through that exact path, not a proxy.

| Step | Result |
|---|---:|
| Manifest source ids | 342 (342 distinct) |
| Manifest sources resolving to a `StandardNameSource` | 342 of 342 |
| Resolved batch standard names | **230** |
| Manifest sources with no terminal name | 2 |
| Batch names by stage | accepted 230, nothing else |
| **Batch names that are quarantined** | **0** |
| **Batch names that are orphans** | **0** |
| Export candidates returned by `_fetch_candidates(names_only=True, batch=…)` | 230 |
| Export candidates that are quarantined | **0** |
| Export candidates that are orphans | **0** |

**Positive control on the zero**: the same `sn.id IN $ids` predicate that
returned 0 quarantined was re-run without the status clause and returned
`seen: 230` — the query sees every row it was asked about, so the two zeros
are measurements and not an empty match.

### The near misses, named

Three quarantined identities *are* bound to a WEST manifest source by a
`PRODUCED_NAME` edge, and all three are already terminal:

| Identity | Stage | Manifest source |
|---|---|---|
| `normalized_toroidal_hard_xray_peak_lower_bound_width` | exhausted | `hard_x_rays/emissivity_profile_1d/half_width_internal` |
| `plasma_breakdown_time` | exhausted | `summary/time_breakdown/value` |
| `lower_bound_hard_xray_peak_width` | superseded | `hard_x_rays/emissivity_profile_1d/half_width_internal` |

104 orphan identities carry a WEST manifest uri in their `source_paths`
scalar, and **all 104 are `superseded`** — tombstones of names the manifest
sources have since re-minted. One further orphan is reached by
`HAS_STANDARD_NAME` from a manifest `IMASNode`, also superseded.

None of these 108 is at an exportable stage, and none appears in the resolved
230-name batch. **Answer: no quarantined and no orphan identity reaches the
WEST batch release.**


![WEST batch exclusion funnel](/imas-codex/figures/quarantine-orphan-census/west-batch-exclusion-funnel.svg)

The funnel is the one relationship here a table does not carry: both classes
survive to the WEST-bound step and both reach exactly zero at batch
resolution, with the positive control drawn on the same row as the zero.

## 6. What this does and does not settle

- The 660 quarantine count is **correct**. Its *classification* is correct for
  23 of the 46 accepted rows and wrong for the other 23, which are held out of
  the catalog by a status the current gate does not reproduce.
- 254 of 660 quarantines record no cause and 596 carry no validation stamp,
  so a majority of the population cannot be justified from its own record.
  These are terminal rows, so the defect is in the record rather than in the
  release — but it means the quarantine axis is not a trustworthy instrument
  for any future question asked of it.
- The 170 orphan figure does not reproduce. The live orphan population is
  2,341 total, 21 live, 9 exportable.
- The WEST batch is clean on both axes, proven through the release path's own
  resolver and export selector with a positive control on each zero.

provisional: false
