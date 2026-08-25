# Production release of the three legacy DD source residues

Date: 2026-08-25

## Outcome

`release_legacy_dd_source_lifecycle` released exactly the three governed legacy
DD sources in the production graph:

- `dd:ntms/time_slice/mode`
- `dd:summary/pedestal_fits`
- `dd:waves/coherent_wave`

The production preview admitted **3 of 3** authority rows and refused **0**.
The apply changed **3** rows and wrote **3** per-source receipts. Replay returned
`already_applied`, `changed=0`, and `persistent_writes=0`. At the live
postflight timestamp, **2026-08-25T08:41:58.632Z**, the composed/attached
no-live-target class was **36 = 0 DD + 36 derived + 0 other**. The DD partition
therefore reached its required real zero, while the 36 transient derived rows
were left untouched.

No refusal occurred, so the verbatim refusal list is `[]`; nothing was bypassed
or worked around. The release supplies no replacement-name authority. Each
source is now `extracted` with a fresh attempt budget so any future attachment
must pass ordinary, current naming authority.

## Governed invocation and authority construction

The read-only preflight measured at **2026-08-25T08:39:00.294Z** found the
expected live class: **39 = 3 DD + 36 derived**. Each named DD source resolved
exactly once, was `source_type=dd`, `status=composed`, unclaimed, had a null
scalar mirror, and had zero `PRODUCED_NAME` targets. The preliminary preview was
`would_apply` with **3 authority rows / 3 admitted / 0 refused / 3 would
change**.

The applying process did not consume a pre-built authority or trust the earlier
preview. It started at **2026-08-25T08:41:10.317Z**, queried the live bindings,
called `build_repair_authority`, serialized and hashed the returned bytes in a
fresh temporary authority, then called `apply_signed_manifest` first in preview
mode and immediately in apply mode in that same process. The applying preview
independently returned the exact three admitted row ids and no refusals. Its
hashes were:

| Digest | SHA-256 |
|---|---|
| Authority file | `2531612837029e79b39d55b41460c3b2ea811ee56ffd757685c7690376bbd849` |
| Authority signed payload | `ef54183fa4a826f3061e043e76e5e0089a392c13d01b637fd92df9781441a68d` |
| Fresh live manifest | `090ecb06f5416081e483debeaf154b107c0bf8d0dca7132b230300d3f326998a` |

The earlier preview happened to produce the same manifest digest, but that
digest was re-derived from the fresh closure inside the applying invocation;
it was not carried forward as assumed live authority. Canonical SHA-256 values,
not Python object identity or temporal equality, were the comparisons.

The exact fresh result was:

| Stage | Authority rows | Admitted | Refused | Changed | Receipt rows | Persistent writes |
|---|---:|---:|---:|---:|---:|---:|
| Preview | 3 | 3 | 0 | 0 (`would_change=3`) | 0 | 0 |
| Apply | 3 | 3 | 0 | 3 | 3 | 6 |
| Replay | 3 recorded | 3 recorded | 0 | 0 | 3 recovered | 0 |

The apply's verbatim refusal list was also `[]`. `StandardNameChange` increased
from **8,517 to 8,520**, exactly the three receipts, while `LLMCost` stayed
**34,104 to 34,104**. The three source mutations reset `status` to `extracted`,
`attempt_count` to 0, `claimed_at`, `claim_token`, `produced_sn_id`, and
`composed_at` to null. No target relationship existed to delete. Each source's
single `FROM_DD_PATH` backing remains present and points to the corresponding
DD path.

## Receipt recovery by the transaction's own identity

The apply used run id
`r-20260825T083346749641-n-ddresidueapply`. Receipt recovery deliberately did
not query by operation name. It matched only the receipt-owned pair:

```cypher
MATCH (change:StandardNameChange {
  run_id: $run_id,
  manifest_sha256: $manifest_sha256
})
RETURN properties(change) AS properties
ORDER BY change.row_id
```

That exact key recovered **3** receipts, one for every admitted row. Every
receipt carried the run id above, the live manifest digest
`090ecb06f5416081e483debeaf154b107c0bf8d0dca7132b230300d3f326998a`,
and both applying authority digests. The recovered receipt ids were:

- `dd:ntms/time_slice/mode` — `sn-change:signed-manifest:090ecb06f5416081e483debeaf154b107c0bf8d0dca7132b230300d3f326998a:88167ada21ef4caf5e5c5e21`
- `dd:summary/pedestal_fits` — `sn-change:signed-manifest:090ecb06f5416081e483debeaf154b107c0bf8d0dca7132b230300d3f326998a:9dea6d2c29f1d0fb38e1b1e8`
- `dd:waves/coherent_wave` — `sn-change:signed-manifest:090ecb06f5416081e483debeaf154b107c0bf8d0dca7132b230300d3f326998a:8adc9d4abc658eb132c28f5a`

The immediate replay used the same freshly built authority and manifest. It
returned `outcome=already_applied`, `changed=0`, `persistent_writes=0`, and
`receipt_rows=3`. Independent persistent counters were identical before and
after replay: `StandardNameChange=8,520` and `LLMCost=34,104`.

## Post-apply census and real-zero proof

The production invariant implementation was re-run after apply and again after
replay. Both measurements returned **36** composed/attached sources with no live
target, partitioned exactly as follows:

| Partition | Before | After apply | After replay |
|---|---:|---:|---:|
| DD | 3 | **0** | **0** |
| Derived transient | 36 | **36** | **36** |
| Other | 0 | **0** | **0** |
| Total | 39 | **36** | **36** |

The queried properties and authored relationship direction were proven before
the DD zero was accepted:

| Schema sanity probe | Before | Post-replay |
|---|---:|---:|
| `StandardName` candidates / with `id` / with `name_stage` | 4,656 / 4,656 / 4,656 | 4,656 / 4,656 / 4,656 |
| `StandardNameSource` candidates / with `id` / with `status` / with `source_type` | 9,668 / 9,668 / 9,668 / 9,668 | 9,668 / 9,668 / 9,668 / 9,668 |
| `StandardNameSource` with `produced_sn_id` | 5,235 | 5,235 |
| Authored `StandardNameSource-[:PRODUCED_NAME]->StandardName` / targets with both keys | 5,351 / 5,351 | 5,351 / 5,351 |
| Reversed `StandardName-[:PRODUCED_NAME]->StandardNameSource` | 0 | 0 |

Thus zero DD no-live-target rows is not a missing-property or reversed-edge
zero. All three source identities still resolve once after replay, but they no
longer participate in the composed/attached invariant because the governed
release returned them to `extracted`.

The derived-transient proof is stronger than a count comparison. The full
ordered snapshot of all 36 rows included every source property and every
current target id, target lifecycle, target status, and binding property map.
Its canonical digest was unchanged at all three checkpoints:

```text
before apply  de427254697917b0b16f9e56d70c754ac846efd6d9d5a0fce2c368f8374acd5c
after apply   de427254697917b0b16f9e56d70c754ac846efd6d9d5a0fce2c368f8374acd5c
after replay  de427254697917b0b16f9e56d70c754ac846efd6d9d5a0fce2c368f8374acd5c
```

The 36 derived transient rows were therefore not selected, mutated, settled,
or otherwise changed by this release.

## Durable evidence

- Read-only cohort, schema sanity, preview, and pre-apply derived digest:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/preflight-preview.json`
- Applying-process authority build, fresh preview, production receipt,
  run-id-plus-digest recovery, replay, postflight census, and collateral digest:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/apply-replay-postflight.json`
- Command diagnostics and exit markers:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/preflight-preview.stderr`
  and
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/apply-replay-postflight.stderr`
- Focused closed-program regression run: **15 passed, 13 graph cases
  deselected, 0 failed** in 6.68 s at
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083346749641-n-ddresidueapply/logs/focused-tests.log`.

Both invocations exited 0. The typed plan reader was unavailable because an
unrelated registered research resource has a malformed typed path; the live
registered plan HTML and its current section were read directly instead. This
did not weaken the authority boundary or alter the production invocation.
