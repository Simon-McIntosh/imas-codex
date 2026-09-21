# Quarantine and orphan census

provisional: true

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
