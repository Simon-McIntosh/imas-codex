# Signed structural-exemption preview

## Outcome

The write-free production preview returned `would_apply` for an exact **49-row**
subset of the signed 216-row source-disposition authority. Those rows would
remove **63** exact `PRODUCED_NAME` bindings and **63** matching
`HAS_STANDARD_NAME` projections, and would change **43** source scalar mirrors.
No production apply was requested or performed.

The previously refused 78 source rows now form a complete, disjoint partition:

| Disposition result | Source rows | Meaning |
|---|---:|---|
| Admitted | 49 | Every removed target belongs to the signed preserve-as-structural set and its locked live closure still contains a direct `HAS_PARENT` child. |
| Refused | 29 | At least one removed target is outside the signed preserve-as-structural set. |
| Total | **78** | Complete prior refusal cohort. |

The parent preview also excluded 138 source rows whose already-changed live
binding or scalar state no longer matched the original adjudication candidates:
87 binding-set changes plus 51 scalar changes. Thus its complete selection was
49 admitted + 167 excluded = 216 signed rows, while the executable subset's own
receipt was 49 requested, 49 admitted, and zero refused.

## Admission authority

Structural eligibility comes only from the committed
`refused-target-orphan-adjudication.json` rows carrying the
`preserve_as_structural_identity` disposition. The signed artifact partitions
89 targets into exactly:

- 69 `preserve_as_structural_identity` targets;
- 16 `retire_under_orphan_policy` targets;
- 3 `retain_competing_binding` targets; and
- 1 `re_source_from_existing_dd_path` target.

The live direct-child closure is a compare-and-set condition for a target that
is already in the signed 69-target set; it is not an independent admission
classifier. This distinction is material because the live graph contains 71
structurally legitimate targets. In particular, `mass_density` has the signed
`retain_competing_binding` disposition and
`neutral_species_energy_convection_velocity` has the signed
`re_source_from_existing_dd_path` disposition. Their live structure cannot
promote either target into this release authority. The preview refused all four
`mass_density` source rows and the neutral-energy source row with `removed
target is outside signed structural legitimacy authority`.

The 49 admitted source rows remove bindings from **61 distinct targets**. The
61-target set is wholly contained in the signed 69-target structural set:

| Containment check | Intersection / difference | Verdict |
|---|---:|---|
| Admitted targets outside signed structural set | 0 | PASS |
| Admitted targets intersect retire dispositions | 0 | PASS |
| Admitted targets intersect retain dispositions | 0 | PASS |
| Admitted targets intersect re-source dispositions | 0 | PASS |

The manifest records 61 target exemptions and locks 70 currently live direct
`HAS_PARENT` child relationships. Apply must re-read that closure under lock and
match the preview manifest before it can remove a final producing binding.

## Signed hashes

- source-disposition adjudication file SHA-256:
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`;
- source-disposition canonical payload SHA-256:
  `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`;
- structural-authority file SHA-256:
  `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36`;
- structural-authority canonical SHA-256:
  `4bac6110486390e95c1cab9620c4723df96fe6f2190b85e6496464c77fbba873`;
- executable subset manifest SHA-256:
  `b7daac18d5f65dd760352b098f7aef26f8320e69eb32cfbd841c67071afe9b9f`.

## Write-free proof

The preview read the mutation ledgers immediately before and after the
transaction. Their canonical counter objects were byte-equal:

| Graph measure | Before | After | Verdict |
|---|---:|---:|---|
| `StandardNameChange` | 7,492 | 7,492 | unchanged |
| `LLMCost` | 27,477 | 27,477 | unchanged |

The focused disposable Neo4j suite completed **14 passed, 0 failed, 0
skipped**. It includes the two structural-authority regressions: a signed
preserve target with a live direct child admits final-binding removal, while a
target carrying a non-structural signed disposition remains fail-closed.

Durable evidence logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T111957383946-sgwi-structural-exemption/disposable-tests-final-rerun.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T111957383946-sgwi-structural-exemption/live-preview-final.log`.

This receipt authorizes no graph mutation by itself. Production apply remains a
separate serialized operation and must use the exact executable-subset manifest
hash above.
