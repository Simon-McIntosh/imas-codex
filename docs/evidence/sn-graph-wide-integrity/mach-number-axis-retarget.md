# Parallel Mach number exact-source retarget

## Outcome

The exact DD source
`dd:langmuir_probes/reciprocating/plunge/mach_number_parallel` was retargeted
from generic `mach_number` to accepted-valid `parallel_mach_number` through the
existing compare-and-set source-migration operator. The applying transaction
reported **`changed=1`**, wrote **one** deterministic
`StandardNameChange` receipt, and measured **`untouched_changed=0`** across all
**9,615** other source rows. The exact replay returned
**`outcome=already_applied`**, **`changed=0`**, and zero persistent writes.

The credentialed explicit-axis ratchet then ran under `-m graph`: **1 passed,
0 skipped, 0 failed**, exit 0. The only warning was the existing generated-model
warning that `RepairAuthorityArtifact.schema` shadows its base attribute.

## Why this binding is authoritative

The DD description identifies the parallel Mach number measured by a
reciprocating Langmuir probe: a signed ratio of flow along the magnetic-field
direction to the local sound speed. The source therefore measures the
axis-qualified child, not its generic structural parent.

| Authority field | Before apply | After apply |
|---|---|---|
| Exact source | `dd:langmuir_probes/reciprocating/plunge/mach_number_parallel` | unchanged |
| Source status / claim | `composed` / unclaimed | unchanged |
| Scalar mirror | `mach_number` | `parallel_mach_number` |
| Sole live `PRODUCED_NAME` | `mach_number` | `parallel_mach_number` |
| Sole DD `HAS_STANDARD_NAME` projection | `mach_number` | `parallel_mach_number` |
| DD and target unit | `1` / `1` | unchanged |
| Target lifecycle | `parallel_mach_number`: accepted, valid | unchanged |
| Structural witness | `parallel_mach_number HAS_PARENT mach_number`, projection axis `parallel` | unchanged; one live child retained |
| Other live direct producers of `mach_number` | 0 | 0 |

The structural witness is the sanctioned closure. A genuine orphan has neither
a producer nor a live child; `mach_number` has zero direct producers after the
move but retains the exact accepted-valid live `parallel_mach_number` child.
This makes it a legitimate unsourced structural parent rather than a genuine
orphan. The graph’s genuine-orphan count fell **36 to 35** because the child
gained its source without the parent entering that class.

## Transaction and closure proof

The driver admitted exactly one source and no caller-supplied subset. Inside the
applying transaction it:

1. read the complete source, DD node, old/new names, units, source binding, DD
   projection, exact `HAS_PARENT` relationship, and old/new producer closures;
2. required `parallel_mach_number` to be accepted and valid and required its
   exact live `HAS_PARENT` edge to `mach_number` with
   `operator_kind=projection` and `axis=parallel`;
3. hashed the canonical participant authority, locked every participant node
   and relationship, re-read the authority, and required the hash to remain
   identical;
4. ran the shared attachment guard over the exact one-row cohort and required
   one admission with zero rejections;
5. invoked `retarget_standard_name_sources` transactionally with the exact
   expected predecessor mapping;
6. required the synchronized edge, scalar, DD projection, old/new
   `source_paths`, exact receipt, flat provider-cost counter, and byte-identical
   untouched closure;
7. re-read the structural witness after mutation and rolled back unless
   `mach_number` still had at least one live child; and
8. committed only after every postcondition passed.

Pre-lock and locked authority SHA-256 were byte-identical:
`1db43a116adb06bb722b0615164041b6a0b8eebb2ee9b042befc76b1743fc4bb`.
The exact source-migration manifest SHA-256 was
`7ba4a47a500fb7154fbc9ba0b06353df633a7ccf744231758f3d46c3ab11e2ad`.

## Quantitative gates

| Gate | Result |
|---|---|
| Applying driver | PASS: `outcome=applied`, `changed=1` |
| Receipt cardinality | PASS: `receipt_rows=1` |
| Receipt identity | `sn-change:source-migration:7ba4a47a500fb7154fbc9ba0b06353df633a7ccf744231758f3d46c3ab11e2ad` |
| Exact source mirrors | PASS: scalar, live binding, and DD projection all equal `parallel_mach_number` |
| Inside-transaction target gate | PASS before lock, after lock, and after mutation: accepted and valid |
| Inside-transaction structural gate | PASS before lock, after lock, and after mutation: exact live parallel projection child retained |
| Old-target closure | PASS: 0 other direct producers, 1 live structural child |
| Untouched closure | PASS: 9,615 rows; `untouched_changed=0` |
| Untouched aggregate digest | `05e8248db6eaa0c6a06a09c38c8b14634396d34e23e3200d5e5b3e1ab6cfd1b3` before and after |
| `StandardNameChange` counter | 7,720 to 7,721, exactly +1 |
| `LLMCost` counter | 27,614 to 27,614 |
| Genuine-orphan count | 36 to 35 |
| Immediate replay | PASS: `already_applied`, `changed=0`, `persistent_writes=0`, `untouched_changed=0` |
| Explicit-axis graph ratchet | PASS: 1 passed, 0 skipped, 0 failed under `-m graph` |

## Replay

The same exact source, predecessor, successor, operation, and manifest were
submitted immediately after commit. The operator found the matching immutable
receipt and exact postcondition and returned `already_applied` with `changed=0`.
Global counters, the selected source closure, all 9,615 untouched source rows,
the structural-child authority, and the receipt were identical before and after
replay.

## Durable artifacts

All operational artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T130959992588-machretarget/`.

| Artifact | SHA-256 / result |
|---|---|
| `apply-receipt.json` | `e874c5e7d90e4e193ef3fa97e05ec9a85a17b9024b6369df5996124a964b4129` |
| `replay-receipt.json` | `c7eb3488a20e36832ca8cd3f789b7041cd53b96cf5fa51906945098bc896e9cc` |
| `apply-driver-authorized.log` | `21b70c4f8d0f10b017625a402c136f385131c8519a687b8520ba7413667b5998`; exit 0 |
| `explicit-axis-ratchet.log` | `2583bee6cca17f42e861639af0828d7b97e247649cff646a23cf28b8b56545de`; 1 passed, exit 0 |

The earlier preflight refusals remain beside these artifacts. Both occurred
before the mutation operator and wrote no graph state: the first rejected stale
description byte-equality, the second exposed the direct-producer-only closure
that the subsequent authority correctly widened to the exact structural-child
disjunction.

