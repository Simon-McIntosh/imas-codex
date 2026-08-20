# Signed stale-source detach apply

Date: 2026-08-20

The exact three-source cohort was selected from the committed stale-source
lifecycle authority and applied in one compare-and-set transaction. The
authority file SHA-256 was
`f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`;
its declared `jq -cS '.rows'` signature was
`316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`.
The live execution manifest SHA-256 was
`bf1aff7001c1d024adbc8e9656032d6a72b5160175f43c9926770adf597da730`.

The applied sources were:

- `dd:refractometer/channel/frequencies` →
  `frequency_of_diagnostic_antenna`
- `dd:neutron_diagnostic/detectors/aperture/centre/phi` →
  `toroidal_angle_of_measurement_position`
- `dd:neutron_diagnostic/detectors/detector/centre/phi` →
  `toroidal_angle_of_measurement_position`

## Result

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Live `PRODUCED_NAME` bindings in the allowlist | 3 | 0 | -3 |
| Matching `HAS_STANDARD_NAME` projections | 3 | 0 | -3 |
| `StandardNameChange` | 7,492 | 7,495 | +3 |
| `LLMCost` | 27,477 | 27,477 | 0 |

The transaction declared three receipt rows and wrote exactly three change
nodes, one for each source. Each detached target retained at least three live
producing sources after the apply, so no orphan was created. All 9,536
out-of-allowlist source closures retained the identical SHA-256
`3a1b97cf2e6cd1a96f91013b1523167e42343284a1dff8f5cd70732ba7339456`
inside the transaction. The exact manifest replay returned
`outcome=already_applied`, `changed=0`, and wrote nothing.

Change receipts:

- `sn-change:stale-source-detach:12cadfa14368c0b0da71efe18751f4156cafb21cc762d92d1d351f4b1bf4ad6c`
- `sn-change:stale-source-detach:1b887b95c5c952cde7d32b559e24a4eb436f8e32c7a5dc91bae2bdf61ddfc068`
- `sn-change:stale-source-detach:69908c69c4b5e6799e3d43408da2b2d39bb4608fd9ddbe971baefc0e4e5cd213`
