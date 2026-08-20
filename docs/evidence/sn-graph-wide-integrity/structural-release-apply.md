# Structural source-release apply

## Outcome

**Applied once and replayed without a second write.** One production invocation
regenerated the complete 216-row parent preview, derived the exact structural
subset, compared the refreshed and previously reviewed manifests through
canonical JSON digests, applied only the resulting authorized manifest, ran the
full postflight, and immediately replayed the same manifest.

The exact production delta was:

| Measure | Result |
|---|---:|
| Selected source rows | 49 |
| Admitted / refused selected rows | 49 / 0 |
| Removed `PRODUCED_NAME` bindings | 63 |
| Removed `HAS_STANDARD_NAME` projections | 63 |
| Changed source scalar mirrors | 43 |
| Distinct released targets | 61 |
| `StandardNameChange` receipt rows | 1 |
| Replay outcome | `already_applied`, `changed=0` |

The applied manifest SHA-256 is
`0e270a2b6432f84a50f478a7196a3d799f305092dee2ae1ea3cc9d04de85a683`.
Its immutable receipt is
`sn-change:signed-source-disposition:0e270a2b6432f84a50f478a7196a3d799f305092dee2ae1ea3cc9d04de85a683`.

## Same-invocation authority

The earlier attempt correctly refused obsolete manifest
`b7daac18d5f65dd760352b098f7aef26f8320e69eb32cfbd841c67071afe9b9f`
after the separately authorized stale-source detach changed the complete
216-row parent closure. The live plan subsequently authorized the refreshed
manifest and required preview plus apply to share one invocation so that no
intervening graph mutation could strand the hash again.

This run loaded the unchanged committed authorities:

- catalog-edit source adjudication file SHA-256:
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`;
- its signed canonical payload SHA-256:
  `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`;
- structural disposition file SHA-256:
  `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36`;
- structural disposition canonical SHA-256:
  `4bac6110486390e95c1cab9620c4723df96fe6f2190b85e6496464c77fbba873`;
- refreshed complete-parent manifest SHA-256:
  `211dc1764b9dcf4552b18ffbc5fd2a5c25a3cff87b2278be3973e3b62015995f`.

Before mutation, the harness removed only `parent_manifest_sha256` from the
fresh and previously reviewed executable manifests, serialized both as sorted
compact JSON with default ASCII escaping and temporal values rendered to text,
and compared their SHA-256 digests. Both canonical digests were exactly
`263b724842c08b2dd62b7d27a9ef4e4c8c6771195320330844625b894e524d93`.
No hydrated Neo4j temporal object was equality-tested against a JSON value.

The fresh preview then had to satisfy every declared cardinality before apply:

| Gate | Required | Observed |
|---|---:|---:|
| Selected rows | 49 | 49 |
| Actions | 49 | 49 |
| Binding removals | 63 | 63 |
| Projection removals | 63 | 63 |
| Scalar changes | 43 | 43 |
| Released targets | 61 | 61 |
| Structural exemptions | 61 | 61 |
| Locked live `HAS_PARENT` relationships | 70 | 70 |

All 61 released targets are members of the signed 69-target
`preserve_as_structural_identity` set. Their intersections with the signed 16
retire, 3 retain-binding, and 1 re-source dispositions are all empty. The ten
parent rows whose incoming closure changed after the stale-source detach have
zero intersection with the selected 49 source IDs.

## Production counters and receipt arithmetic

The ledgers were read before apply, after apply, and after immediate replay:

| Graph measure | Before | After apply | After replay | Delta |
|---|---:|---:|---:|---:|
| `StandardNameChange` | 7,495 | 7,496 | 7,496 | +1 |
| Manifest receipt rows | 0 | 1 | 1 | +1 |
| `LLMCost` | 27,477 | 27,477 | 27,477 | 0 |

The one-row `StandardNameChange` increase equals the instrument's declared
receipt-row count exactly; there is no extra maintenance or reconciliation
write hidden in the apply. The unchanged `LLMCost` count proves the operation
made no provider call.

## Released-target structural closure

Every one of the 61 released targets retained at least one live direct
`HAS_PARENT` child after its final producing binding was removed. Their exact
70-relationship child closure was identical to the preview-locked closure:

| Child-closure measure | Result |
|---|---:|
| Released targets checked | 61 |
| Targets retaining a live child | 61 |
| Targets with no live child | 0 |
| Live child relationships | 70 |
| Minimum live children per target | 1 |
| Maximum live children per target | 3 |

The machine receipt
`released-target-child-closure.json` enumerates all 61 target IDs, their child
IDs, and relationship counts. Representative structural releases include:

- `average_external_magnetic_flux` retaining child
  `current_weighted_average_external_magnetic_flux`;
- `current_density_due_to_wave_driven_current_drive` retaining
  `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` and
  `per_toroidal_mode_current_density_due_to_wave_driven_current_drive`;
- `hydrogen_density` retaining `line_averaged_hydrogen_density` and
  `volume_averaged_hydrogen_density`;
- `neutron_source_rate_due_to_thermal_fusion` retaining three species-resolved
  and total children;
- `power_due_to_recombination` retaining divertor-target, wall, and
  divertor-power children.

This is an intentional structural release: the targets remain live identities
because their accepted children use them as parents, even though their direct
producing-source count becomes zero.

## Collateral immutability

The harness normalized every `StandardNameSource` outside the exact 49-source
allowlist together with its properties, `PRODUCED_NAME` bindings, DD/signal
backing edge, and backing `HAS_STANDARD_NAME` projections. It computed both an
aggregate digest and one digest per source row before apply, after apply, and
after replay.

| Out-of-allowlist measure | Before | After apply | After replay |
|---|---:|---:|---:|
| Source closures | 9,490 | 9,490 | 9,490 |
| Aggregate SHA-256 | `d023b0ea…` | `d023b0ea…` | `d023b0ea…` |
| Changed row digests | 0 | 0 | 0 |

The full digest is
`d023b0ea5bf56a1240bbe8e6bc9879cdaba1938ab74c0ee06eed6eb3ceaff667`.
All 9,490 individual row digests are identical before and after; the aggregate
digest is also identical after replay. Thus every out-of-allowlist source
closure is proven immutable rather than inferred from the operation's counts.

## Post-apply census

The same production invocation recorded the live graph census before and after
the apply:

| Census | Before | After | Change |
|---|---:|---:|---:|
| Live dual-bound source rows | 87 | 38 | -49 |
| Live bindings on those dual-bound rows | 198 | 86 | -112 |
| Live unsourced names | 40 | 101 | +61 |

The dual-bound source count falls by exactly the 49 reconciled rows. The
unsourced-name increase is exactly the 61 deliberately released structural
targets; the stage split moves from 30 accepted / 4 drafted / 2 pending / 4
reviewed to 91 accepted / 4 drafted / 2 pending / 4 reviewed.

The same-invocation baseline of 40 supersedes the earlier carried count of 36
for this receipt. That four-name difference existed before this transaction;
the apply's attributable change is exactly 61, with every added unsourced name
accounted for by the signed structural target list and its retained child
closure.

## Representative source dispositions

The applied subset preserves the more specific accepted identity selected by
the signed source row and removes broader competitors. Examples from the final
receipt:

- `dd:distribution_sources/source/global_quantities/shinethrough/torque_phi`
  moved from broad `torque_due_to_neutral_beam_shinethrough` to
  `toroidal_torque_due_to_neutral_beam_shinethrough`. The surviving description
  explicitly identifies the toroidal component of angular momentum carried
  away by neutral-beam shinethrough.
- `dd:distributions/distribution/global_quantities/collisions/electrons/torque_fast_phi`
  selected `toroidal_fast_electron_torque_due_to_collisions` over
  `fast_electron_torque_due_to_collisions`, preserving the `_phi` projection
  and fast-electron population.
- `dd:distributions/distribution/global_quantities/collisions/ion/torque_thermal_phi`
  selected `toroidal_thermal_ion_torque_due_to_collisions`, accepted at name
  review score 0.95625, over both `ion_torque_due_to_collisions` and
  `thermal_ion_torque_due_to_collisions`.
- `dd:distributions/distribution/global_quantities/torque_tor_j_radial`
  selected `toroidal_fast_particle_torque_due_to_j_cross_b_force`, accepted at
  name review score 0.925, over three broader vector or particle-population
  identities. Its description preserves the radial-current mechanism and the
  toroidal angular-momentum projection.

Some legacy accepted rows have a null aggregate name-review score even though
their lifecycle is accepted; this apply does not alter scores, descriptions, or
acceptance state. It changes only the signed source scalar and exact paired
binding/projection closure.

## Verification and durable artifacts

The production harness exited 0 after all preview, canonical comparison,
apply, collateral, child-closure, and replay gates passed. At the current
checkout, the two provider-free signature/authority tests in
`test_signed_source_dispositions.py` also passed; the unchanged operator's full
disposable-Neo4j regression remains recorded as 14 passed, 0 failed, 0 skipped
in the structural-authority preview evidence.

Durable files for this applied receipt:

- production log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/production-structural-release.log`
  (SHA-256 `b95de0d7382e3c3b57e9bac1af6517e1cc930b204b4ac09f8edc122fb91f1fcf`);
- regenerated preview:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/structural-release-preview.json`
  (SHA-256 `a8a921fa1fbcda187d4784b7b781a56fa59a6c8d1b47370375415a30b504d462`);
- before-state and all 9,490 per-row collateral digests:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/structural-release-baseline.json`
  (SHA-256 `6e14305f074046801259de33e267c64a0acec633bf5a4d947e0a24b2349bc249`);
- apply receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/structural-release-apply-receipt.json`
  (SHA-256 `74ccdafdea88adc06565add4861f53101629fcd0a7b512111caadfcda21dd883`);
- replay receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/structural-release-replay-receipt.json`
  (SHA-256 `7fccb1644ec8a0350fc3037f59fc10130ee193915d11733fac61939fa8a120b8`);
- postflight census and representative source rows:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/structural-release-postflight.json`
  (SHA-256 `88e4520a8587dd7cbc2e2084e2804b89499ce49f4758214bc765bf9b86b0cdf3`);
- all-target child-closure projection:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/released-target-child-closure.json`
  (SHA-256 `13667c56549ff73f0d76376f96f36625c4adc3ec05fe33ba46b03d75e00e0d82`);
- focused test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T122120984631-sgwi-structural-release-canonical/focused-tests.log`
  (SHA-256 `8fe2cd8eb877dc2f08f45c1580598e7c22c669ad278cc1afabbdbfba3345ce27`).
