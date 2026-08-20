# Owner-geometry ready apply evidence

## Outcome

The refusal-free executable subset derived from the signed parent authority was
applied and replayed successfully. The production invocation read the committed
49-row partition, verified its declared canonical rows SHA-256
`4de9c2df481180931a47b7a8bcc76cb69253e23d96e2dfa151bd86edcb76c8cd`,
selected the complete 12-row write cohort, and ran the live attachment guard over
all 12 rows. The guard deterministically selected 11 rows and excluded
`dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` as a
`distinct-vector conflict`. It then built the executable manifest from that
locked result inside the same transaction that performed the apply.

The executable manifest records the parent partition digest, all 12 parent ids,
all 11 selected ids, the one excluded id, and the exclusion rule and reason. Its
preview and locked-transaction forms matched only by canonical sorted compact
JSON SHA-256:
`e49b800adc5276867fc1f0c75087821705313973f81cd693dbf0e70f977baff4`.
The 11-row executable gate had zero refusals. Five target-grouped ledger receipts
recorded all 11 source changes, and the immediate exact-manifest replay returned
`already_applied` with `changed=0` for every group.

The earlier 12-row attempt remains useful fail-closed evidence: it reached the
live attachment guard, correctly refused the reflector row, and rolled the
transaction back before commit. No guard change was made. The successful apply
used the subsequently authorized parent-derived exclusion, not a caller-supplied
11-row list.

## Applied source bindings

Every selected source now has exactly one live `PRODUCED_NAME` binding and one
backing-node `HAS_STANDARD_NAME` projection to the accepted-and-valid target
named by the signed authority.

| Accepted and valid target | Applied source bindings | Target description |
|---|---:|---|
| `toroidal_coordinate_of_aperture` | 5: `camera_visible/channel/aperture/centre/phi`, `camera_x_rays/aperture/centre/phi`, `hard_x_rays/channel/aperture/centre/phi`, `mse/channel/aperture/centre/phi`, `spectrometer_x_ray_crystal/channel/aperture/centre/phi` | Toroidal angular coordinate of an aperture's geometric center, locating the aperture around the symmetry axis in the right-handed cylindrical `(R, φ, Z)` frame. |
| `toroidal_coordinate_of_filter_window` | 3: `hard_x_rays/channel/filter_window/centre/phi`, `soft_x_rays/channel/filter_window/centre/phi`, `spectrometer_x_ray_crystal/channel/filter_window/centre/phi` | Toroidal coordinate of a filter window's geometric center in the right-handed cylindrical `(R, φ, Z)` frame. |
| `toroidal_angle_of_coil_conductor_element` | 1: `tf/coil/conductor/elements/end_points/phi` | Toroidal azimuthal coordinate locating a coil conductor element around the symmetry axis in the right-handed cylindrical `(R, φ, Z)` frame. |
| `toroidal_coordinate_of_line_of_sight` | 1: `interferometer/channel/n_e/positions/phi` | Toroidal angular coordinate of the first reference point defining a diagnostic line of sight in the right-handed cylindrical `(R, φ, Z)` frame. |
| `toroidal_coordinate_of_optical_element` | 1: `spectrometer_visible/channel/optical_element/geometry/centre/phi` | Toroidal angular coordinate of an optical element's geometric center, locating the optical element around the symmetry axis in the right-handed cylindrical `(R, φ, Z)` frame. |
| **Total** | **11** | **Five accepted-and-valid surviving identities** |

## Reflector exclusion and non-action

The guard refusal is semantically correct. For a spherical reflector,
`reflector/centre` is the center of the reflecting surface, while
`reflector/sphere_centre` is the center of curvature of the sphere containing
that surface; the points are displaced by the radius. The owner-geometry
cardinality rule preserves owner identity across parameterizations of one named
point. It does not merge two physically distinct named points merely because
they share one owner.

The parent-derived receipt therefore records
`dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` as its one
first-class exclusion. That source remains bound to
`toroidal_angle_of_measurement_position`. The incumbent
`dd:spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi` binding to
`toroidal_coordinate_of_reflector` also remains unchanged. Their joint canonical
state digest was
`6c20131d4ca70b2b8b490cb6fb8efd052f3e076e41f2d49d1973c1a570b326ca`
both before and after the apply.

The coordinator-provided census reported that neither reflector path existed as
an `IMASNode` and suggested a possible stale-DD-path issue. The fresh pre-apply
census did not reproduce that observation: each path currently has one
classified `IMASNode`, and each `StandardNameSource` has a `FROM_DD_PATH` link to
it. The receipt preserves both the supplied report and the fresh contradictory
observation. No reflector identity, vocabulary, source binding, or DD-path repair
was attempted.

## Quantitative gates

| Gate | Measured result |
|---|---|
| Signed parent authority | PASS: 49 rows; 21 ready; 9 already selected; complete 12-row write cohort derived |
| Parent-derived executable subset | PASS: 12 inspected; 11 selected; 1 reflector exclusion recorded; no caller row list accepted |
| Executable apply | PASS: 11 changed source rows across 5 grouped receipts; 0 executable refusals |
| Surviving target authority | PASS: all 11 have the signed target as scalar binding, live edge, and backing projection; all five targets are `accepted` and `valid` |
| Manifest comparison | PASS: preview SHA-256 = locked-transaction SHA-256 = `e49b800a…baff4` |
| `StandardNameChange` | PASS: 7,496 → 7,501, an increase of exactly the declared 5 receipt rows and no more |
| `LLMCost` | PASS: 27,477 → 27,477 |
| Old-target live producer | PASS: `toroidal_angle_of_measurement_position` retains 50 live producers after apply, down from 61 |
| Parent-authority residue | PASS: 27 of the 49 authority rows still select the old measurement-position target, down from 38 |
| Out-of-allowlist closure | PASS: all 9,528 row digests identical; aggregate SHA-256 `d2f3633600ac080681e31a12f28206f70e81ecea1b22339be4eea7161cf08398` before and after |
| Immediate replay | PASS: all 5 groups returned `already_applied`; aggregate `changed=0`; counters, selected state, reflector state, authority census, and closure digest stayed stable |
| Independent live postflight | PASS: re-read 11 selected rows, five receipt nodes, counters, accepted-valid targets, 49-row authority census, 50 old-target producers, reflector state, and 9,528-row closure digest |

## Durable artifacts

All run artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T130012050116-sgwi-owner-geometry-ready-apply/`.

| Artifact | SHA-256 or result |
|---|---|
| `owner-geometry-preview.json` | `a2e97f94716f456037f68c1180b8ad59ec4abe4660dc033ef0731f089761cb46` |
| `owner-geometry-baseline.json` | `a57acd8342af3f7846ec3d5600a4a9a583cc449c8b4ebc483a4cbcaa6050978d` |
| `owner-geometry-apply-receipt.json` | `42b9134a16f7010bfa8d5995eee23875ee4bd03cd3583b0ac5643340455ac11b` |
| `owner-geometry-replay-receipt.json` | `7d47d9eca8d58d975fb6e68e64e1db5f0ff97ae17cec40827fb7b6adcac01177` |
| `owner-geometry-postflight.json` | `fba87463d5cd3e0fcc0ab25bf034525f8615995c874219da85769a6c2175a9d7` |
| `production-owner-geometry-subset-apply-retry.log` | Complete successful preview, transaction, counter, closure, and replay gate stream; exit 0 |
| `independent-live-postflight.log` | Independent current-state verification; `status=passed`; exit 0 |
| `production-owner-geometry-subset-apply.log` | Pre-mutation census-discrepancy stop; no transaction entered |
| `production-owner-geometry-apply.log` | Earlier 12-row guard refusal and transaction rollback evidence |

The apply driver and independent verifier are retained beside the receipts for
audit. Re-adjudicating the reflector-center identity, deciding whether a new
center-of-curvature vocabulary identity is needed, and reconciling the differing
IMAS-node census are separate review work; none was performed here.
