# Structural source-release apply

## Outcome

**BLOCKED before mutation.** The production operator regenerated the complete
signed authority before apply and correctly refused to treat manifest
`b7daac18d5f65dd760352b098f7aef26f8320e69eb32cfbd841c67071afe9b9f`
as current. No source scalar, `PRODUCED_NAME` binding,
`HAS_STANDARD_NAME` projection, target property, or receipt was written by
this attempt.

The fresh preview still returns `would_apply` with exactly the intended
executable action set:

| Measure | Authorized preview | Fresh production preview |
|---|---:|---:|
| Selected source rows | 49 | 49 |
| Bindings to remove | 63 | 63 |
| Projections to remove | 63 | 63 |
| Scalar changes | 43 | 43 |
| Released targets | 61 | 61 |
| Live direct-child relationships | 70 | 70 |
| Executable manifest | `b7daac18…` | `0e270a2b…` |

The two canonical manifests differ at exactly one field. The complete
216-row parent-manifest SHA-256 changed from
`2bb20f8534a10abf15572c76b92de6af0cca2cf2b0cfdeda46f531fbfaf59fd9`
to
`211dc1764b9dcf4552b18ffbc5fd2a5c25a3cff87b2278be3973e3b62015995f`.
Every embedded selected action, participant, removed-target closure,
structural exemption, selected source ID, refusal, count, and signed authority
field is otherwise byte-identical.

## Cause of authority drift

After the authorized structural preview was generated, a separate signed
stale-source transaction applied manifest `bf1aff7001c1d024adbc8e9656032d6a72b5160175f43c9926770adf597da730`.
It removed three stale bindings and wrote three change receipts. Two of those
sources were:

- `dd:neutron_diagnostic/detectors/aperture/centre/phi` →
  `toroidal_angle_of_measurement_position`;
- `dd:neutron_diagnostic/detectors/detector/centre/phi` →
  `toroidal_angle_of_measurement_position`.

`toroidal_angle_of_measurement_position` belongs to the global incoming
closure of ten rows in the 216-row catalog-edit adjudication. None of those ten
rows is in the selected 49-row structural-release subset, so the executable
actions remain unchanged; nevertheless, the disposition instrument
deliberately binds the complete parent closure so that an intervening
authorized mutation is visible rather than silently ignored.

The original structural preview measured `StandardNameChange=7,492`. The
stale-source transaction raised that count to 7,495. The immediate post-refusal
check confirms:

| Production measure | Current result |
|---|---:|
| `StandardNameChange` | 7,495 |
| `LLMCost` | 27,477 |
| Receipt rows for authorized `b7daac18…` manifest | 0 |

This is a compare-and-set refusal, not a semantic reversal and not a partial
apply. The signed 69-target structural admission set and the 49-row selection
remain unchanged.

## Required recovery

The safe recovery is to review and explicitly authorize the fresh exact
manifest `0e270a2b6432f84a50f478a7196a3d799f305092dee2ae1ea3cc9d04de85a683`
after accepting the parent-closure transition above. A new serialized apply
must then baseline the out-of-allowlist closure, apply only that exact hash,
replay it immediately, and produce the originally required postflight census.

The alternative—restoring the old parent hash—would require undoing the
separately authorized stale-source detach. That would reintroduce obsolete DD
bindings and is neither authorized nor recommended.

Durable evidence:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T115236350868-sgwi-structural-release-apply/live-structural-release-apply.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T115236350868-sgwi-structural-release-apply/structural-release-preview-receipt.json`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T115236350868-sgwi-structural-release-apply/authorized-manifest-drift.diff`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T115236350868-sgwi-structural-release-apply/stale-detach-parent-overlap.json`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T115236350868-sgwi-structural-release-apply/fail-closed-postcheck.log`.
