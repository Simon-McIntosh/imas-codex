# Residual dual-bound source reconciliation apply

## Result

**PASS, recovered from the durable transaction ledger after the applying
process outlived its command wrapper.** One exact signed authority accounted
for all **23 adjudicated source rows** as **19 admitted + 4 refused**. The
admitted rows committed atomically at **2026-08-21 21:03:41.760 UTC**, wrote
**19 `StandardNameChange` receipts**, and removed **23 losing
`PRODUCED_NAME` relationships**. The four refused rows remain dual-bound and
each carries the operator's verbatim reason:
`target would lose its last producing source`.

The exact-manifest replay completed in the recovery invocation with
`outcome=already_applied`, `changed=0`, and `persistent_writes=0`. A read-only
four-row preview independently reproduced every refusal. Counters were
byte-for-byte unchanged across replay and refusal preview.

| Required measure | Result |
|---|---:|
| Signed authority rows | **23** |
| Admitted rows | **19** |
| Refused rows | **4** |
| Admitted + refused | **23 / 23** |
| Mutated logical rows | **19** |
| Reconciliation receipt rows | **19** |
| `StandardNameChange` | **7,759 → 7,778 (+19)** |
| `PRODUCED_NAME` relationships | **5,791 → 5,768 (-23)** |
| `LLMCost` rows | **27,631 → 27,631 (0)** |
| Replay | **`already_applied`; changed 0; persistent writes 0** |
| Remaining live dual-bound sources | **4** |

The receipt cardinality equals the mutated-row cardinality, and the
`StandardNameChange` delta from the signed baseline equals that same count:
**19 = 19 = 7,778 - 7,759**.

## Exact authority and apply receipt

- Authority file:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/dual-bound-source-target-authority.json`
- Authority file SHA-256:
  `ef51f912038976721b35f4fa7a830c43983bab001b675b59559c99a18ce58972`
- Canonical authority payload SHA-256:
  `2a64bb77d99c8e0c9934fba3052473040a1322f8920f78cdb2396dfec6007adc`
- Applied manifest SHA-256:
  `b202f2637e6fba595508e70f809febdbae90e7caed12c919c8e2f46b4d34519b`
- Operation:
  `reconcile_standard_name_source_targets`
- Apply run id:
  `dual-bound-source-reconciliation`
- Recovery receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T221709444947-dualapplyrun/recovered-apply-replay-receipt.json`
- Recovery receipt SHA-256:
  `d256b1716b1360eccc3e31224f61b91aa4355b55a7f9f23f6259517649340be5`

Every one of the 19 durable change rows repeats the same authority file hash,
payload hash, manifest hash, operation, run id, and complete 19-row admitted
cohort. Exact replay re-read those receipts and re-verified the signed
postconditions before returning write-free.

## Recorded refusals and live residue

| Source row | Verbatim refusal reason |
|---|---|
| `dd:plasma_profiles/ggd/mass_density/values` | `target would lose its last producing source` |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `target would lose its last producing source` |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `target would lose its last producing source` |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux_limiter/z` | `target would lose its last producing source` |

These are the **only four** live dual-bound sources after apply. The recovery
invocation built a canonical signed authority containing just these surviving
rows and ran it in preview mode. It returned `authority_rows=4`, `admitted=0`,
`refused=4`; every row returned the exact reason shown above. This preview was
read-only and used only to recover the refusal evidence that the terminated
wrapper did not retain.

The four refusal rows are durable qualified outcomes, not unexecuted rows. The
last-producer guard correctly preserves their losing bindings until an
authoritative replacement producer or a separately adjudicated identity
disposition exists.

## Recovery of the completed transaction

The earlier applying process was initially reported as zero-write because it
emitted no terminal Python receipt before the command wrapper ended. That
report was a premature recovery snapshot. On the next fresh invocation, the
live-cohort compare-and-set failed before signing because 19 adjudicated rows
were no longer dual-bound. Immediate ledger attribution then found:

- exactly **19** change rows for the expected operation and manifest;
- a shared transaction timestamp of **2026-08-21 21:03:41.760 UTC**;
- the exact expected authority file and payload hashes on every row;
- the exact expected 19-row admitted cohort on every receipt;
- `StandardNameChange=7,778`, exactly **19 above** the signed baseline; and
- the 23-source cohort reduced to the exact four last-producer refusals.

No fresh apply was attempted against the changed cohort. Recovery invoked only
the exact original manifest replay, which recognized the complete receipt
cohort and returned `already_applied` without rebuilding or mutating the
already-reconciled rows.

## Logs and independent checks

- First fresh invocation and fail-closed cohort-drift result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T221709444947-dualapplyrun/apply-preview-replay.log`
- Live counter, receipt, and dual-bound attribution:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T221709444947-dualapplyrun/live-attribution.log`
- Replay plus verbatim-refusal proof:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T221709444947-dualapplyrun/replay-and-refusal-proof.log`
- Canonical four-row read-only refusal authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T221709444947-dualapplyrun/remaining-refusal-authority.json`

The recovered machine receipt records nine passing checks: all 23 rows
accounted, every refusal reason present, residual rows equal refusal rows,
receipts equal mutated rows, ledger delta equals receipts, replay recognized,
replay changed zero, replay wrote zero, and counters stayed unchanged across
both recovery operations.
