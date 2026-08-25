# Accepted-name authority residual drain

**Recorded:** 2026-08-25  
**Outcome:** qualified reduction; the release hold remains for 79 accepted names  
**Spend:** **USD 8.018906 of USD 60.000000** (13.36% used; USD 51.981094 unspent)

## Result

The accepted-name authority residual fell from **194 to 79**, a reduction of
**115 names (59.3%)**, without direct acceptance, fabricated reviewer scores,
or unsigned structural promotion. The campaign used the ordinary name-review
quorum for 104 valid catalog or legacy-marker entries and the sanctioned
derived-parent enrichment path for 11 placeholder parents.

The remaining 79 names do not currently have a sanctioned route that can be
run without first changing their lifecycle state or adding an explicit
authority-replay operator. They remain a release hold; their existing accepted
state was not treated as proof of authority.

## Closure partition

The before and after measurements use the closure audit's exact partition
query, including the relationship-backed structural-authority test:

```cypher
MATCH (s:StandardName {name_stage: 'accepted'})
OPTIONAL MATCH (s)-[:HAS_STRUCTURAL_AUTHORITY]->(a:StructuralNameAuthority)
WITH s, count(a) > 0 AS has_structural_authority
RETURN count(s) AS accepted,
       count(CASE WHEN s.reviewer_score_name IS NOT NULL THEN 1 END) AS scored,
       count(CASE WHEN has_structural_authority THEN 1 END) AS structural,
       count(CASE WHEN s.reviewer_score_name IS NOT NULL
                   AND has_structural_authority THEN 1 END) AS overlap,
       count(CASE WHEN s.reviewer_score_name IS NULL
                   AND NOT has_structural_authority THEN 1 END) AS residual
```

The property-presence preflight returned 4,656 `StandardName` candidates,
4,656 with `id`, 4,656 with `name_stage`, and 3,842 with
`reviewer_score_name` after the drain. This prevents a missing-property query
from being mistaken for an empty population.

| Measurement | Accepted | Own reviewer score | Structural authority | Overlap | Residual |
|---|---:|---:|---:|---:|---:|
| Before | 2,326 | 1,895 | 237 | 0 | **194** |
| After | 2,295 | 1,968 | 248 | 0 | **79** |
| Change | -31 | +73 | +11 | 0 | **-115** |

The accepted total decreased because quorum replay is a real lifecycle
transition. Of 104 reviewed entries, 73 earned acceptance and 31 remained in
`reviewed`; the latter are no longer members of the accepted residual. All 104
received their own reviewer score. The 11 derived parents remained accepted
and each acquired one durable `HAS_STRUCTURAL_AUTHORITY` record.

## Sanctioned operations and spend

| Route | Exact cohort | Result | Exact spend | Authorized cap | Run |
|---|---:|---|---:|---:|---|
| Ordinary name quorum | 99 accepted, valid, unscored entries | 70 accepted; 29 reviewed; all 99 scored | USD 7.591027 | USD 45.00 | `6d5dfca4-789e-436b-aeb3-7a053d426710` |
| Derived-parent enrichment | 11 accepted placeholder parents with live children | 11 signed structural-authority records; all use `deterministic-grammar-peel` and `orphan_policy=refuse` | USD 0.010642 | USD 10.00 | `faa61e9c-33f1-4dae-bc62-8c72e9c3c065` |
| Ordinary name quorum for legacy markers | 5 valid `catalog_edit` entries with no children | 3 accepted; 2 reviewed; all 5 scored | USD 0.417237 | USD 5.00 | `7453151b-9e2a-4ee2-a760-6f63cf13a90c` |
| **Total** | **115 residual entries removed** | **73 scored acceptances, 31 scored reviewed entries, 11 structural authorities** | **USD 8.018906** | **USD 60.00** | |

All three runs stopped with `no_eligible_work` and recorded
`cost_is_exact=true`. The USD 51.981094 remainder was not spent because every
entry reachable through the sanctioned routes had drained; more calls could
not lawfully resolve the fail-closed remainder. Restaging itself made no LLM
call and wrote no reviewer score. For the first review cohort it preserved
2,030 relationships, including 380 `HAS_STANDARD_NAME`, 99 `HAS_UNIT`, and 8
`HAS_COCOS` bindings. The five-name legacy-marker cohort likewise preserved
all 74 relationships, including 2 `HAS_STANDARD_NAME` and 5 `HAS_UNIT`
bindings.

The enrichment run's general `names_enriched` audit counter remains zero even
though its pool processed 11 entries. The authoritative evidence is the 11
run events plus the 11 newly present signed authority records, each carrying
the accepted parent identity, child identities, code and schema identities,
guards, participants, mutation receipts, and a replay signature.

## Representative outcomes and source bindings

- `counter_passing_thermal_particle_source_rate` returned to accepted with
  score 1.000. It describes the flux-surface-averaged volumetric rate into or
  out of the counter-passing thermal orbit class and remains bound to
  `distributions/distribution/profiles_1d/counter_passing/source/particles`.
- `line_averaged_effective_charge` returned to accepted with score 1.000. Its
  retained bindings include `charge_exchange/channel/zeff_line_average`,
  `bremsstrahlung_visible/channel/zeff_line_average`, and
  `summary/line_average/zeff/value`.
- `alpha_parameter` acquired a quorum score of 0.4375 and remained reviewed;
  it was not hand-accepted. `electron_power` likewise remained reviewed at
  0.775. The other three legacy-marker entries returned to accepted with fresh
  scores: `perturbed_plasma_mass_density` 1.000,
  `perturbed_plasma_pressure` 0.975, and
  `perturbed_plasma_temperature` 0.9875.
- `wave_voltage` now has a signed structural record entailed from
  `wave_voltage_amplitude`; its synthesized description identifies the
  diagnostic microwave voltage detected after interaction with the plasma
  cutoff layer. `kinetic_energy_density` is entailed from
  `ion_kinetic_energy_density` and `runaway_electron_kinetic_energy_density`.
  `wavelength_of_visible_camera` is entailed from its lower- and upper-bound
  children. These parent entries have child provenance rather than direct DD
  source bindings.

## Residual by entry path and disposition

The same accepted-null-score-no-authority predicate was partitioned by
`origin`, `validation_status`, and live `HAS_PARENT` children. The groups are
mutually exclusive and sum to 79.

| Entry path | Count | Representative identities | Disposition |
|---|---:|---|---|
| Derived, valid, live children, real (non-placeholder) description | **74** | `wave_phase_of_ion_cyclotron_heating_antenna`, `flux_at_wall`, `particle_energy`, `ion_charge`, `vector_potential` | **HOLD.** The enrichment pool only claims placeholder descriptions, so it cannot replay accepted parents whose descriptions were populated before signed authority existed. Quorum review is not substituted for structural entailment. Add a sanctioned, atomic authority-replay path using the current children, guards, code identity, and schema identity; then remeasure this exact partition. |
| Catalog edit, quarantined | **3** | `gyrocenter_pressure`, `perturbed_gyrocenter_pressure`, `perturbed_particle_pressure` | **HOLD.** The exact restage precondition correctly refused these rows because they are not `validation_status=valid`. Repair or adjudicate validation through the grammar lifecycle, then send retained identities through ordinary name review. Do not bypass quarantine or write scores by hand. |
| Derived, quarantined, live children | **2** | `plasma_internal_energy`, `thermal_plasma_internal_energy` | **HOLD.** Resolve the validation failure first; only then may a derived-parent structural replay test entailment from the live children. Do not fabricate structural authority for a quarantined identity. |

For provenance, the valid derived example
`wave_phase_of_ion_cyclotron_heating_antenna` describes the relative RF phasor
phase at an ion-cyclotron antenna element. Its child
`forward_wave_phase_of_ion_cyclotron_heating_antenna` is bound to
`ic_antennas/antenna/module/phase_forward`. Among the quarantined derived
entries, `plasma_internal_energy` remains directly bound to
`summary/global_quantities/denergy_thermal_dt/value`.

## Operational evidence and caveats

- Full ordinary-review log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T033916042328-n-authorityresidual/review-drain.log`
- Full derived-parent enrichment log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T033916042328-n-authorityresidual/parent-enrich.log`
- Full legacy-marker review log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T033916042328-n-authorityresidual/marker-review.log`

During the 99-name review, several optional DD-gap evidence records were not
persisted because model output supplied prose where the
`DDGapEvidenceRule` enum requires a declared value. The affected name-review
decisions and reviewer scores persisted normally. This is a separate evidence-
telemetry defect, not authority for accepting or rejecting any name, and
should be repaired outside this fenced evidence-only change.

