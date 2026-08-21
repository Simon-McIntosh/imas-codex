NEEDS-HELP: The signed authority loader rejected the applying invocation before preview, and the two-failure stop rule forbids a third attempt.

tried: The first applying invocation derived every `PRODUCED_NAME` edge, so its cohort check correctly refused historical `superseded` and `exhausted` targets before signing or mutation. After narrowing “live” to the adjudication predicate, the second invocation derived the exact 23-row authority but `apply_signed_manifest` rejected `selection.predicate = "adjudicated-dual-bound-sources"` with `SignedManifestAuthorityError: authority selection predicate must be 'artifact-rows'`. The loader failed before opening the preview/apply transaction.

options: Change both `selection.id` and `selection.predicate` to the closed branch's tested literal `artifact-rows`, regenerate the signed authority, and rerun the one-process preview/apply/replay driver; alternatively, add a public authority builder that emits the branch's closed selection block and dispatch a fresh apply owner; or leave the graph unchanged and retain the adjudication as pending authority.

leaning: Change the selection block to `artifact-rows` and redispatch the apply. The disposable-graph contract constructs exactly that block, while the source and complete live target sets already matched all 23 adjudicated rows.

cost-if-wrong: A wrong selection correction should fail before mutation under the signed loader or manifest-hash comparison. If it passed an unintended branch, the exact 23-row authority and postconditions would have to be independently re-audited before accepting any receipt.

## Blocked outcome

No production reconciliation was attempted by the operator. The first failure
occurred in the driver before authority construction; the second occurred in
the signed authority loader before preview construction. A read-only audit after
both failures found:

| Measure | Observed |
|---|---:|
| Fresh live authority rows constructed | 23 |
| Preview rows classified | 0 (loader refused before preview) |
| Apply admitted | 0 |
| Apply refused | 0 (no preview receipt) |
| Mutated rows | 0 |
| Reconciliation receipt rows | 0 |
| Replay | not reached |
| Current `StandardNameChange` nodes | 7,759 |
| Current `PRODUCED_NAME` relationships | 5,791 |
| `StandardNameChange` rows with apply run id | 0 |

The requested quantitative done condition is therefore **not met**. In
particular, admitted plus refused cannot yet be compared with the signed
authority row count, and no apply/replay receipt exists. The zero run-specific
receipt count and failure locations prove that this node made no production
graph mutation.

## Recoverable inputs

- Generated authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/dual-bound-source-target-authority.json`
- Authority rows: **23**; repair rows: **23**
- Authority file SHA-256:
  `bcffbd14c5abd15875100c2fdaab1622a656dcee40b1850cdd323cf02719ff2a`
- Signed payload SHA-256:
  `eeead071ca85f6c38af2aad3eb03471ad5ff692103548e7262cc46164c502d7f`
- First invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/apply-dual-bound.log`
- Second invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/apply-dual-bound-second.log`
- Post-failure audit:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/blocked-state-audit.log`

The authority artifact contains the source, every complete live target,
selected survivor, and every losing relationship identity for all 23 rows. It
is intentionally not valid mutation authority until its closed selection block
is regenerated and the file and payload signatures are recomputed.
