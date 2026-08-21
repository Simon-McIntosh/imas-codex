# Electron-density source snapshot move

## Outcome

The reconstructed source `dd:sawteeth/profiles_1d/n_e` now records DD
version `4.1.1`, matching the graph's unique current `DDVersion` and its exact
`FROM_DD_PATH` backing projection `sawteeth/profiles_1d/n_e`.

The operation used an exact one-source allowlist, locked the 46 participants
in the complete source-authority closure, re-read that closure under lock, and
committed only after its canonical hash remained unchanged. It created one
immutable `StandardNameSourceSnapshotChange` receipt and one linked internal
`StandardNameChange` audit row. The existing `electron_density` semantic
binding was retained.

## Authoritative re-read

| Field | Before | After |
|---|---|---|
| Recorded DD version | `4.1.0` | `4.1.1` |
| Backing IMASNode | `sawteeth/profiles_1d/n_e` | unchanged |
| Unit scalar | `m^-3` | `m^-3` |
| `HAS_UNIT` target | `m^-3` | `m^-3` |
| Stored DD documentation | `Electron density (thermal+non-thermal)` | unchanged |
| Stored source description | enriched source prose | authoritative DD documentation |
| Stored physics domain | `transport` | `magnetohydrodynamics` |
| Produced Standard Name | `electron_density` | unchanged |

The description and physics-domain mirrors were re-read from the same current
IMASNode snapshot as the DD version and unit. The enriched prose remains in
`enhanced_description`; no Standard Name identity or review state changed.

## Receipt and replay

- Snapshot receipt:
  `source-snapshot-change:30abece2c8326e4abd02b6ba35cb6548659c3e5dfe79c85c93f7124fffb26e7e`
- Internal change receipt:
  `sn-change:a3c7baafdae9b32264a8997475c70ac91df77ec16e3cf1393b8a79daba6bfa4a`
- Authority hash:
  `83d68272c02d9f784fd29d06eecd19eed82ca0e43382574e8edc2be74850246c`
- Precondition hash:
  `98b111a704e8722fd33b60226b93835563e7f78e2d4a5bcc89d76e654594ad56`
- Participant-set hash:
  `f44a5cfe85668422e6ff5464c300d27609eaae35459bbf0bcc1be01b4508a74a`
- Apply outcome: `applied`, changed `1`, receipt rows `1`.
- Immediate independent-transaction replay outcome: `already_current`,
  changed `0`, receipt rows `1`.

The explicit public resolution call using the source's post-move path,
version, and re-read fields returned successfully with unit `m^-3`; it did not
raise `DDResolutionVersionMismatch`.

## Collateral and cost checks

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Sources whose `dd_version` differs from current authority | 8,681 | 8,680 | -1 |
| `StandardNameChange` | 7,703 | 7,704 | +1 |
| Declared receipt rows | 1 | 1 | 0 |
| `LLMCost` rows | 27,591 | 27,591 | 0 |
| Accumulated `LLMCost.llm_cost` | 1,365.422230999989 | 1,365.422230999989 | 0 |

All 9,615 other `StandardNameSource` property rows hashed to
`0c0d3619a26f5d2007c7faf88328d8a05442f9a6a18bfea5e1ea91a46f5cd05f`
both before and after the transaction and again after replay. No other source
was modified, and no provider call occurred.

## Evidence

- `logs/preflight-fixed.log`: live current-version, exact source/backing/unit,
  counter, and out-of-target digest census.
- `logs/dry-run-fixed.log`: zero-write source-authority closure, snapshot diff,
  hashes, and resolution probes.
- `logs/apply-and-replay-fixed.log`: complete apply and replay receipts,
  counter deltas, backing/unit postconditions, out-of-target digest equality,
  and the successful explicit resolution call.

The logs are retained in the worker run directory
`r-20260821T091742786524-sgwi-ne-source-snapshot-move`.
