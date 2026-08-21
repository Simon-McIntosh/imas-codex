# Structural-source revival apply

## Outcome

**Applied and replay-verified.** One production invocation read the exact live
cohort for `electron_diffusivity` and `ion_diffusivity`, constructed the signed
authority through `build_repair_authority`, previewed it, applied the admitted
rows, queried the resulting receipts by the exact `run_id` and manifest digest,
replayed the same manifest, and then measured the graph-wide structurally bare
set.

Both signed rows were admitted and neither was refused. The apply changed two
logical rows through four mutations and wrote two receipts. The replay returned
`already_applied` with zero changed rows and zero persistent writes.

| Measure | Result |
|---|---:|
| Signed authority rows | **2** |
| Admitted | **2** |
| Refused | **0** |
| Admitted + refused | **2 / 2 signed rows** |
| Logical rows changed | **2** |
| Mutations | **4** |
| Receipt rows | **2** |
| Persistent writes | **6** |
| `StandardNameChange` | **7,778 → 7,780, delta +2** |
| `PRODUCED_NAME` relationships | **5,768 → 5,770, delta +2** |
| `LLMCost` rows | **27,631 → 27,631, delta 0** |
| Replay outcome | **`already_applied`** |
| Replay changed | **0** |
| Replay persistent writes | **0** |

The required arithmetic closes exactly:

- admitted `2` + refused `0` = signed row count `2`;
- receipt rows `2` = changed logical rows `2`;
- live `StandardNameChange` delta `+2` = receipt rows `2`.

There were no refusal rows, so the requirement that every refusal retain its
verbatim reason is satisfied without omission; the signed receipt carries an
empty refusal array rather than an inferred success count.

## Signed authority and manifest

The authority was generated from the live participants inside the same Python
invocation that performed the apply. The builder owned the closed selection,
repair-row projection, receipt cardinality rule, signature, and final byte
digests.

| Identity | Live structural children signed into the closure | Result |
|---|---|---|
| `electron_diffusivity` | `effective_electron_diffusivity`, `parallel_electron_diffusivity`, `poloidal_electron_diffusivity` | admitted and applied |
| `ion_diffusivity` | `effective_ion_diffusivity`, `parallel_ion_diffusivity`, `poloidal_ion_diffusivity` | admitted and applied |

- Authority file SHA-256:
  `a1eef8115dd430575cc01d7a27d4d25199fc882a178ddf1ed53538ed4d5745fe`
- Canonical authority payload SHA-256:
  `0d8d6d1330fed4808e28d188fea07eb2dfe427a3d22923e0a9f5323f34366be0`
- Authorized manifest SHA-256:
  `ce5302b1baaafaed897c77287610f60c91ab86e00b9f7be784137e4a713d33be`
- Receipt run id:
  `r-20260821T223719510924-structapply`

## Durable receipt proof and replay

Receipt recovery did not guess an operation name and did not infer completion
from the aggregate change counter. Immediately after apply, the invocation
queried `StandardNameChange` with both the exact receipt `run_id` and exact
manifest digest. It returned these two rows:

| Row id | Receipt id |
|---|---|
| `derived:electron_diffusivity` | `sn-change:signed-manifest:ce5302b1baaafaed897c77287610f60c91ab86e00b9f7be784137e4a713d33be:3ed2aa74f2c03e3099b5e2cb` |
| `derived:ion_diffusivity` | `sn-change:signed-manifest:ce5302b1baaafaed897c77287610f60c91ab86e00b9f7be784137e4a713d33be:90f01261ad0b7f14ae77fdff` |

Each row carries the same canonical authority payload digest shown above. The
same exact `run_id` + manifest-digest query was repeated after replay and
returned the identical two receipt records. This proves the replay added no
receipt under another operation spelling. In the same comparison,
`StandardNameChange`, `LLMCost`, and `PRODUCED_NAME` totals remained exactly
`7,780`, `27,631`, and `5,770`; the operator itself reported
`already_applied`, `changed=0`, and `persistent_writes=0`.

An independent read-only post-check confirmed both sources are now
`status='composed'`, `source_type='derived'`, have their `source_id` and
`produced_sn_id` scalar set to the parent identity, and carry exactly one live
`PRODUCED_NAME` binding to that same identity.

## Post-apply structurally bare census

The two repaired diffusivity identities are no longer structurally bare. The
graph-wide post-apply query nevertheless found **9 live names** that have a live
`HAS_PARENT` child and no non-stale producing source:

| Structurally bare name | Live children |
|---|---:|
| `area_of_langmuir_probe` | 2 |
| `electrostatic_potential_imaginary_part` | 1 |
| `momentum_source` | 6 |
| `neutral_species_energy_convection_velocity` | 1 |
| `neutral_state_particle_diffusivity` | 1 |
| `normalized_perturbed_current_density` | 1 |
| `outer_squareness_of_flux_surface` | 2 |
| `parallel_normalized_perturbed_current_density` | 1 |
| `volume_averaged_runaway_electron_current_density` | 1 |

That census is a graph-wide finding, not a shortfall in the exact two-row
authority: neither repaired target is in the nine-row residue. Disposition or
repair of those nine names is outside this node's exclusive scope and should be
carried into the coordinator's next closure census rather than mutated here.

## Verification record

- Production invocation and full receipt JSON:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T223719510924-structapply/logs/structural-source-revival-apply-attempt2.log`
- Independent exact-source post-check:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T223719510924-structapply/logs/structural-source-revival-postcheck.log`
- Initial pre-authority syntax refusal retained for audit:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T223719510924-structapply/logs/structural-source-revival-apply.log`

The initial invocation stopped on a Cypher 5 ordering syntax error while
reading the cohort. The corrected invocation ordered child rows before
collection and then completed the entire signed preview/apply/receipt/replay
sequence in one process.
