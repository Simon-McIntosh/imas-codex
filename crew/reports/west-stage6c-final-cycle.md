NEEDS-HELP: the live cycle returned cleanly but auto-promoted 384 rather than the specified 374, and the required all-batch provenance read failed twice at the shell boundary

tried: Restored both authorized lifecycle residues with one property-limited, atomic Cypher write; both `breakdown_initial_time.docs_stage` and `pulse_duration.docs_stage` changed from `drafted` to `accepted`. Ran approval, resolution, and undo in order with exit status `0` for all three. Approval reported `changes_seen=2`, `auto_approved=384`, `contested=2`, and zero accepted, staged, quarantined, blocked, or unmatched edits. Resolution approved `breakdown_initial_time` with the required reason. Undo reported `approved→accepted=385`, `contested→accepted=1`, and deleted the fold-back receipt. The final census reads `approved=0`, `contested=0`, `accepted=2336`; both edited nodes are `name_stage='accepted'` and `docs_stage='accepted'`, with all five StandardName `catalog_*` fields null on those two nodes. The batch-wide all-409 provenance query was attempted twice; both times the shell consumed `$names`, Neo4j received `WHERE sn.id IN  RETURN`, and the read failed without writing. The anti-thrashing fence therefore stops further attempts.

options: (1) run the final batch-wide provenance read from a small checked script or an invocation that does not place the Cypher parameter inside nested shell quoting, then complete the remaining fork/upstream controls; (2) accept the global `count(sn.catalog_pr_number)=0` plus the two target rows as partial provenance evidence, explicitly waiving the other four batch-wide field counts; or (3) first investigate why 384 untouched names were eligible when the plan expected 374, then repeat the cycle only if a corrected cohort definition requires it.

leaning: option 3 followed by option 1. The cycle mechanics now work end to end, but the ten-name count discrepancy is a real cohort-definition finding and should not be relabeled as a pass. Once that is explained, a non-nested query instrument can close the remaining read-only evidence without another graph mutation.

cost-if-wrong: accepting the run as complete would conceal a ten-name auto-promotion discrepancy and leave four of five catalog provenance fields unproven across 409 batch identities. Repeating the live cycle before explaining the cohort could create another catalog materialization commit and unnecessary review calls, even though the current graph and receipt state have already been unwound.

## Lifecycle restoration

The authorized Cypher matched exactly the two edited identities, required both prior values to be `drafted`, and set only `docs_stage`:

| Standard Name | Before | After |
| --- | --- | --- |
| `breakdown_initial_time` | `drafted` | `accepted` |
| `pulse_duration` | `drafted` | `accepted` |

Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T000011001254-n-west-stage6c-final-cycle/logs/02-lifecycle-restoration.log` (exit `0`; an earlier parse-only attempt made zero writes).

## Approval

Command: `IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog uv run --no-sync imas-codex sn approve --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3`

- Exit status: `0`.
- Output: `changes_seen=2`, `auto_approved=384`, `contested=2`, all other outcome counts `0`.
- Post-command census: `approved=384`, `contested=2`, `accepted=1950`, `catalog_pr_number` populated on `386`.
- Both contested nodes carried non-null `catalog_pr_number=3`, PR URL, merge SHA `6a5c44d38f47921ae954e96222045252adcd8127`, and reviewer actor `Simon-McIntosh`.
- `breakdown_initial_time`: docs score `0.7166666666666667`; reason records score `0.717 < 0.850`.
- `pulse_duration`: docs score `0.5416666666666666`; reason records score `0.542 < 0.850`.
- Receipt tag existed with `graph-merged: 2026-09-02T00:55:25.736578+00:00` and outcomes `auto_approved=384 contested=2`.
- Catalog materialization head was `b930e66c4082653a1c425daf2d1ef33a56b5c2da`, tree `447efdd1f663645fe5a50366a7aaa43be955c1a0`.

Logs: `04-approve.log` and `05-after-approve-before-resolve.log` under the run log directory.

## Resolution

Command: `uv run --no-sync imas-codex sn resolve breakdown_initial_time --override --reason 'The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.'`

- Exit status: `0`.
- `breakdown_initial_time` became `name_stage='approved'`, `docs_stage='accepted'`.
- Description equals `Timestamp at which plasma breakdown begins and discharge current starts to flow.`
- `contested_resolution` equals the supplied reason.
- `pulse_duration` remained contested.
- Census before undo: `approved=385`, `contested=1`, `accepted=1950`.

Logs: `06-resolve.log` and `07-after-resolve-before-undo.log`.

## Undo and closure state

Command: `uv run --no-sync imas-codex sn approve --undo --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3`

- Exit status: `0`.
- Output: `approved→accepted=385`, `contested→accepted=1`; receipt deleted from origin.
- Final graph: `approved=0`, `contested=0`, `accepted=2336`, global non-null `catalog_pr_number=0`.
- `breakdown_initial_time`: accepted/accepted, reviewer-edited description retained, `contested_resolution` retained, `edit_origin='human'`, and all five StandardName `catalog_*` fields null.
- `pulse_duration`: accepted/accepted, reviewer-edited description retained, `contested_resolution='approval of catalog PR 3 unwound'`, `edit_origin='human'`, and all five StandardName `catalog_*` fields null.
- Frozen review manifest remained byte-identical through every completed control: git blob `dd46e21250c3e4aad9259a2f58d87a2feff5fbab`, SHA-256 `d7f5b833cddcf17ae67318719a9b14d3ea1dd4a5e337b4a8c7a3b43eee9f122a`.

Log: `08-undo.log`. Final partial closure log: `09-final-closure.log` (exit `1` at the malformed batch-wide read after the census and edited-node assertions passed).

## Quantitative verdict

- Lifecycle commands: `3/3` completed with exit `0`.
- Authorized restorations: `2/2`, each `drafted→accepted`.
- Mechanical end state: graph unwound to `0 approved / 0 contested`; both edited names docs-accepted.
- Required expected approval count: failed, observed `384` versus expected `374` (`+10`).
- Required all-batch provenance proof: absent; the frozen manifest has 409 identities, but the five-field aggregate query did not execute.
- Required post-undo receipt/tag, catalog-tree, and upstream closure reads: not completed after the evidence command stopped at its second identical failure. Pre-cycle controls proved the cut-time tag, catalog tree equality, and upstream SHA; the undo output proves receipt deletion but does not substitute for the requested final reads.
