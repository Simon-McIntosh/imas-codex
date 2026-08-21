# Source-less voltage rename execution

## Result

**The sanctioned transition failed closed with no persistent graph change.**
The public `sn edit` route could preview the requested rename from
`voltage_of_diagnostic_antenna` to `voltage_of_ece_channel`, but its atomic
rename transaction then refused to move the predecessor's unique stale source.
That refusal is the safe terminal outcome selected by the live plan when the
transition cannot preserve the predecessor and source ledger.

The exact guard string was:

```text
source migration compare-and-set failed: dd:ece/channel/t_e_voltage(exists=True, status='stale', claimed=False, bindings=['voltage_of_diagnostic_antenna'], scalar='voltage_of_diagnostic_antenna')
```

The transaction rolled back before ordinary review. The successor remains
absent, so there are **0 fresh quorum review rows**, **0 names accepted**, and
**0 persistent graph writes**. Spend was **USD 0.00** against the authorized
USD 15 ceiling.

## Quantitative postflight

| Measure | Before | After | Verdict |
|---|---:|---:|---|
| Stale-source ledger SHA-256 | `198a7cae76e14420a533493f8ba30ad5d851ff41d7fb00c206944067baf93919` | same | byte-identical |
| Three-identity family SHA-256 | `39fd8a9396a701918d55ba2b9e5936a8bdb30fdb7f01a5358e8f70228754684f` | same | byte-identical |
| `StandardNameChange` rows | 7,754 | 7,754 | no write |
| `LLMCost` rows | 27,619 | 27,619 | no provider call |
| Cumulative recorded LLM cost | USD 1,366.279974 | USD 1,366.279974 | delta USD 0.00 |
| `voltage_of_ece_channel` nodes | 0 | 0 | successor absent |
| Successor quorum review rows | 0 | 0 | review never entered |
| Names accepted by this execution | 0 | 0 | no promotion |

The family digest covers the full properties, producing-source rows, and review
counts of `voltage_of_diagnostic_antenna`, `voltage_of_ece_channel`, and
`voltage_of_spectrometer_channel`. It therefore also proves that the predecessor
remains `accepted`, the current `voltage_t_radiation` binding remains with the
reviewed spectrometer-channel identity, and no target residue survived the
rolled-back transaction.

## Execution record

Preflight at source commit `108687dd` recorded:

- `voltage_of_diagnostic_antenna`: `accepted`, `valid`, unit `V`, five existing
  review rows, and exactly one producer,
  `dd:ece/channel/t_e_voltage` at `status='stale'`;
- `voltage_of_ece_channel`: absent;
- `voltage_of_spectrometer_channel`: `reviewed`, `valid`, unit `V`, with the
  current `dd:ece/channel/voltage_t_radiation` source still `composed`; and
- no competing `imas-codex sn run`, `sn edit`, `sn review`, or `sn rescore`
  process.

The dry-run command exited 0 and reported that it would rename the predecessor
to the intended successor with `scope=only_self` and `entry=review_name`. The
same command without `--dry-run`, capped with `--cost-limit 15`, exited 1 at the
atomic source-migration compare-and-set quoted above. The exception originated
inside `retarget_standard_name_sources`; transaction rollback left both the
source closure and the three-identity family byte-identical.

This is not a grammar, provider, capacity, or budget failure. It is the exact
source-preservation conflict anticipated by the disposition record: ordinary
rename persistence treats every producer edge as migration authority, while
this transition is required to preserve the unique stale edge as historical
ledger state. Retrying the same public route would ask the same guard to weaken
and is not authorized.

## Evidence inputs

- Live plan: `docs/sn-graph-wide-integrity.html`, version 224, read in full
  before execution.
- Disposition authority:
  `docs/evidence/sn-graph-wide-integrity/voltage-antenna-disposition.md`.
- Producer search:
  `docs/evidence/sn-graph-wide-integrity/voltage-antenna-producer-search.md`.
- Preflight snapshot:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T142630253943-voltrename/voltrename-preflight.json`.
- Postflight proof:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T142630253943-voltrename/voltrename-postflight.json`.

## Standing disposition

Retain `voltage_of_diagnostic_antenna` and its stale source as ledgered history.
Do not detach or migrate `dd:ece/channel/t_e_voltage`, do not reassign
`dd:ece/channel/voltage_t_radiation`, do not hand-promote
`voltage_of_ece_channel`, and do not delete the predecessor. A future attempt
requires a separately sanctioned operator whose transaction can create a
reviewable source-less successor while explicitly excluding the stale producer
from migration and proving that exclusion against the complete source closure.
