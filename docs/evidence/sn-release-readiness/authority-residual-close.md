# Accepted-name authority residual closure

**Recorded:** 2026-08-25<br>
**Live snapshot started:** 2026-08-25T08:33:49+02:00<br>
**Tree:** `a3eb8e859993b6f0c4a076ae754934d53d3d430a`<br>
**Outcome:** complete named release disposition; no graph mutation

## Result

The closure-audit accepted-authority partition reports a residual of **9 before
and 9 after** this evidence-only close. The nine rows are now individually
dispositioned: **all 9 are explicit first-release HOLDs**, with a named lifting
condition for each. None is left to a generic lifecycle filter or an absent
authority record by accident.

**Sanctioned authority acquired in this close: 0 names.** That is a qualified
result, not a failed drain. Four accepted derived parents have no accepted child
from which structural authority can be entailed; the other five are
quarantined, so neither ordinary review admission nor structural replay may
treat their accepted lifecycle state as permission. Writing a score, accepting
a child, clearing quarantine, or fabricating a child closure would bypass the
sanctioned routes.

This artifact grants no catalog acceptance and changes no graph state. Its
authority is the explicit release disposition: unless a row satisfies its
lifting condition and the same partition is re-read, the first release must
carry that identity as an identity-bearing `release_authority` exclusion.

## Quantitative closure

Property coverage was established before trusting the partition. The graph has
4,656 `StandardName` candidates, with `id` and `name_stage` present on all
4,656. It has 331 `StructuralNameAuthority` records, with `id`,
`accepted_name_id`, and `child_ids` present on all 331. The four missing
`validation_status` values are outside the accepted residual and do not affect
this result.

| Measurement | Accepted | Own name score | Structural authority | Overlap | Residual |
|---|---:|---:|---:|---:|---:|
| Before disposition record | 2,295 | 1,968 | 318 | 0 | **9** |
| After disposition record | 2,295 | 1,968 | 318 | 0 | **9** |
| Change | 0 | 0 | 0 | 0 | **0** |

The residual partitions exactly as **4 authority-grounding-childless derived
parents + 5 quarantined entries = 9**, with **0 undispositioned**. Here
“childless” is specific to the structural-authority contract: each of the four
still has one live `HAS_PARENT` child, but it has **zero accepted children**, so
there is no eligible accepted child closure to sign.

## Four accepted-childless derived parents

Each row is valid and accepted but has neither its own reviewer score nor a
signed structural authority. The non-accepted child and its current score are
shown because they are the exact reason replay correctly refused the parent.

| Accepted parent | Current grounding evidence | Representative meaning or binding | Explicit first-release disposition |
|---|---|---|---|
| `current_density_due_to_collisions` | Only child `poloidal_current_density_due_to_collisions` is `reviewed`, valid, score **0.6500**; accepted-child count **0**. | Vector electric-current-density contribution from collisional momentum exchange; derived source `derived:current_density_due_to_collisions`, unit `A.m^-2`. | **HOLD.** Lift only after an actual child earns `name_stage='accepted'` through ordinary review and the signed structural-authority replay succeeds against the then-current child closure. If no child can earn acceptance, retire or recompose the parent through the governed lifecycle; do not sign the reviewed child as accepted evidence. |
| `effective_thermal_ion_charge_state_energy_velocity_due_to_convection` | Only child `radial_effective_thermal_ion_charge_state_energy_velocity_due_to_convection` is `reviewed`, valid, score **0.7125**; accepted-child count **0**. | Effective vector velocity for convective transport of thermal energy by one ion charge state; derived source of the same identity, unit `m.s^-1`. | **HOLD.** Lift only after accepted-child grounding exists and an atomic signed replay records the exact child set. Otherwise retire or recompose through the governed lifecycle; a below-bar reviewed child is not structural authority. |
| `neutral_particle_convection_velocity` | Only child `parallel_neutral_particle_convection_velocity` is `reviewed`, valid, score **0.9125**; accepted-child count **0**. Parent docs are also `reviewed`, not accepted. | Effective velocity multiplying particle density in the convective part of neutral-particle transport; derived source of the same identity, unit `m.s^-1`. | **HOLD.** Lift only when the child has actually reached accepted state, signed replay succeeds, and the parent's documentation independently clears the docs gate. A numeric child score alone does not substitute for its lifecycle state. |
| `tritium_velocity` | Only child `toroidal_tritium_velocity` is `drafted`, valid, unscored; accepted-child count **0**. | Density-weighted mean tritium-population velocity; source `dd:summary/local`, unit `m.s^-1`. | **HOLD.** Lift only after the child completes ordinary review to accepted and a signed replay succeeds. If the DD source cannot support a reviewed directional family, recompose or retire through the governed lifecycle rather than granting parent authority directly. |

Accounting: **4 named rows, 0 sanctioned authorities written, 4 explicit
HOLDs, 0 undispositioned**.

## Five quarantined accepted entries

Quarantine is a fail-closed authority boundary. These rows cannot be admitted
to ordinary review or structural replay until their validation conflict is
resolved through the named route. Their existing accepted state is historical
state, not release permission.

| Accepted identity | Current contradiction | Representative meaning or binding | Explicit first-release disposition |
|---|---|---|---|
| `gyrocenter_pressure` | `catalog_edit`, quarantined, unscored, unit `1`; child `perturbed_gyrocenter_pressure` is itself accepted and quarantined. | Description says a pressure moment normalized by a reference thermal pressure; derived source `derived:gyrocenter_pressure`. | **HOLD.** Resolve by source/family-guided recompose retaining normalization in the identity, or retire the invalid peeled parent. If retained, it must become valid and earn its own ordinary name-review score; the quarantined child cannot authorize it structurally. |
| `perturbed_gyrocenter_pressure` | `catalog_edit`, quarantined, unscored, unit `1`, no children; docs are `exhausted`. | Description already identifies a dimensionless perturbed gyrotropic pressure moment normalized to reference thermal pressure; derived source of the same identity. | **HOLD.** Recompose with explicit normalization, then require grammar validation, ordinary name review, and accepted documentation. Do not change the DD-authoritative dimensionless unit or accept the exhausted docs by hand. |
| `perturbed_particle_pressure` | `catalog_edit`, quarantined, unscored, unit `1`, no children. | Abstract normalized pressure-tensor moment of the perturbed particle distribution; derived source of the same identity. | **HOLD.** Recompose with explicit normalization and route the retained identity through validation and ordinary review. A catalog-edit origin supplies no external authority. |
| `plasma_internal_energy` | Derived, quarantined, unscored, unit `W`; direct source `dd:summary/global_quantities/denergy_thermal_dt/value`. Child `thermal_plasma_internal_energy` is also accepted and quarantined. | Description and DD path both describe a rate of change, while the identity spells a stored internal energy. | **HOLD.** Family/source-guided recompose to a tendency identity and retire the rate-as-stored-energy parent through the governed lifecycle. Do not replay structural authority for the semantically contradicted parent, and do not relabel the DD-authoritative `W` unit. |
| `thermal_plasma_internal_energy` | Derived, quarantined, unscored, unit `W`; direct source `dd:summary/global_quantities/denergy_thermal_dt/value`. Child `total_thermal_plasma_internal_energy` is accepted but quarantined. | Description says signed time derivative of stored thermal internal energy, again contradicting a stored-energy spelling. | **HOLD.** Resolve the family to a reviewed tendency identity and retire or rebuild the invalid stored-energy parent. Only after valid semantics exist may structural replay test accepted-child entailment; quarantine itself is never authority. |

Accounting: **5 named rows, 0 sanctioned authorities written, 5 explicit
HOLDs, 0 undispositioned**.

## Live query record

All graph operations were read-only. The exact closure-audit partition was run
before and after writing this record.

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

Before result:
`accepted=2295, scored=1968, structural=318, overlap=0, residual=9`.

After result at `2026-08-25T06:35:58.226348+00:00`:
`accepted=2295, scored=1968, structural=318, overlap=0, residual=9`.
The identity-set assertion also passed: the residual was exactly the four
accepted-childless derived parents plus the five quarantined entries named
above, with `sanctioned_authority_acquired=0`, `dispositioned=9`, and
`undispositioned=0`.

The residual identity query used the same predicate and returned exactly the
nine rows in the two tables above. Incoming `(child)-[:HAS_PARENT]->(parent)`
edges, child lifecycle and score, and producing-source relationships were read
in the same transaction so the dispositions do not rest on origin strings or
historical counts.

## Release handoff

The first release must exclude all nine identities above under explicit
release authority until their individual lifting conditions are satisfied.
Re-run the property-coverage query and this exact partition immediately before
release. A row may leave this hold set only by acquiring its own score through
ordinary review or a signed structural authority entailed from at least one
accepted child after valid-state checks. A lower residual caused only by
de-acceptance is a real lifecycle outcome and must remain visible in the
release ledger; it is not an authority acquisition.
