NEEDS-HELP: repaired undo completed, but the fold-back cycle could not form because stale edited-node stages blocked approval after 384 partial promotions

# WEST repaired fold-back cycle

## Outcome

The cycle did not satisfy the end-to-end gate. The first command exited 1 after
partially auto-approving 384 entries, then blocked both reviewer-edited entries:

- `breakdown_initial_time` was still `approved` from the earlier incomplete
  unwind, with null PR provenance.
- `pulse_duration` was `accepted`, but its docs axis was still `drafted`.

The attachment guard therefore refused both documentation edits instead of
routing them to `contested`. No `graph-merged:` receipt was written. The required
resolve command then exited 1 because `breakdown_initial_time` was not contested.

The repaired frozen-batch fallback was successfully exercised by the final undo:
it demoted all 385 approved batch rows, including the one null-provenance row.
The graph finished fully unwound at 0 approved, 0 contested, and 2,336 accepted,
with all catalog provenance null on the batch and the cut-time RC tag retained.

There is a second external-state defect: the failed approval pushed catalog
materialization commit `e631b875434c5b4544d9103201571943838119d4`, but undo did
not create an inverse commit because no contract tag existed. Fork `main` now
points at that commit even though the graph is unwound. No manual catalog repair
was attempted.

## Frozen review-manifest identity

The review artifact was checked before approval, after failed approval, before
undo, and after undo:

```text
imas_codex/standard_names/manifests/reviews/v0.3.0rc1+west-task-2e.sn_names.yaml
sha256: d7f5b833cddcf17ae67318719a9b14d3ea1dd4a5e337b4a8c7a3b43eee9f122a
git blob (worktree): dd46e21250c3e4aad9259a2f58d87a2feff5fbab
git blob (HEAD):     dd46e21250c3e4aad9259a2f58d87a2feff5fbab
```

Every `git diff --quiet HEAD -- <artifact>` check exited 0. The artifact remained
byte-identical to HEAD throughout.

## Censuses and fork tag before each command

The frozen artifact contains 409 IDs. A graph row exists for every ID. Stage
counts below are global; provenance counts are over the 409-row frozen batch.

| Point | Approved | Contested | Accepted | Batch PR number | Batch PR URL | Batch merge SHA | Batch reviewer | Fork tag state |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Before `sn approve --pr` | 1 | 0 | 2,335 | 0 | 0 | 0 | 0 | cut-time RC object `665be8f2`, subject `WEST review batch` |
| Before `sn resolve` | 385 | 0 | 1,951 | 384 | 384 | 384 | 384 | cut-time RC object `665be8f2`; no receipt |
| Before `sn approve --undo` | 385 | 0 | 1,951 | 384 | 384 | 384 | 384 | cut-time RC object `665be8f2`; no receipt |
| After undo | 0 | 0 | 2,336 | 0 | 0 | 0 | 0 | cut-time RC object `665be8f2`; no receipt |

The batch itself moved from 1 approved / 0 contested / 389 accepted before
approval, to 385 / 0 / 5 after the partial approval, then to 0 / 0 / 390 after
undo. The remaining 19 batch rows are in other lifecycle states.

The tag was captured before each command and after undo. At every point it was:

```text
tag:            v0.3.0rc1+west-task-2e
annotated object: 665be8f244bf227442507adc44fc69c6a6f8443a
peeled commit:    b3ad33253a0e4e92d7003d87510090f45dbe1499
subject/message:  WEST review batch
```

The fork never exposed a `graph-merged:` tag during this attempt because
approval exited before receipt creation. The final `git ls-remote --tags origin`
read proves the cut-time RC object and peeled commit remain remotely present.

## Command 1 — approval

Command:

```text
IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog \
uv run --no-sync imas-codex sn approve \
  --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
```

Exit status: **1**

Output counts:

| Metric | Count | Required |
|---|---:|---:|
| changes seen | 2 | 2 |
| accepted edited entries | 0 | 0 |
| staged for review | 0 | 0 |
| auto-approved | **384** | **374** |
| contested | **0** | **2** |
| quarantined | 0 | 0 |
| blocked | **2** | **0** |
| unmatched | 0 | 0 |

Blocked details:

```text
breakdown_initial_time: target name_stage='approved' — docs edits require an accepted name
pulse_duration: target docs_stage='drafted' — docs edits require docs_stage in accepted/exhausted
```

Transcript:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/approve.log`

The command partially wrote the graph and pushed catalog commit
`e631b875434c5b4544d9103201571943838119d4` despite its nonzero exit. The commit
deleted 117 lines across two catalog files and records pre-fold-back commit
`408eb92258e11f09bcdc54a4a19dc2ad47a5951f`. Fork `origin/main` now resolves to
`e631b875434c5b4544d9103201571943838119d4`.

## Command 2 — adjudication

Command:

```text
uv run --no-sync imas-codex sn resolve breakdown_initial_time --override \
  --reason 'The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.'
```

Exit status: **1**

Output:

```text
breakdown_initial_time is not contested — override-approve only resolves a contested name.
```

Transcript:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/resolve.log`

The command made no change. Consequently the required reviewer text was not
materialized. Required description:

> Timestamp at which plasma breakdown begins and discharge current starts to flow.

Actual description after resolve and after undo:

> Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.

The earlier `contested_resolution` remained recorded, but it belongs to the
previous rehearsal attempt rather than a successful adjudication in this cycle.

## Command 3 — undo

Command:

```text
IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog \
uv run --no-sync imas-codex sn approve --undo \
  --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
```

Exit status: **0**

Output counts:

```text
approved → accepted: 385
contested → accepted: 0
```

Transcript:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/undo.log`

This is positive evidence for the repaired fallback: 384 stamped approvals plus
the one approved, null-provenance frozen-batch member were all demoted. After
undo, every frozen-batch row has null `catalog_pr_number`, `catalog_pr_url`,
`catalog_merge_commit_sha`, `catalog_reviewer_actor`, and
`catalog_approved_at`.

However, because the failed approval never wrote the contract tag, the undo
path did not invoke catalog-correction reversal. The graph and receipt tag are
unwound, while fork `main` remains materialized at `e631b875`.

## Evidence logs

- Before approval graph:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/before-approve-graph.log`
- Approval transcript:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/approve.log`
- After failed approval / before resolve graph:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/after-failed-approve-graph.log`
- Resolve transcript:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/resolve.log`
- Before undo graph:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/before-undo-graph.log`
- Undo transcript:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/undo.log`
- After undo graph:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T234246363308-n-west-stage6b-repaired-cycle/logs/after-undo-graph.log`

## Blocker handoff

tried: Ran the repaired cycle exactly once with the required catalog path.
Approval saw both edits but exited 1 after 384 partial auto-promotions because
the two edited nodes were not in attachment-eligible lifecycle states. Resolve
then exited 1 because no node was contested. The authorized undo exited 0 and
fully unwound all 385 approvals through the repaired membership fallback.

options: Restore the live graph to the clean rehearsal restore point and rerun
the repaired cycle; or add an approval preflight/transaction fence that refuses
all writes unless every edited target is attachment-eligible, then repair the
catalog/graph mismatch and rerun. A narrow preparation operation could settle
the two edited nodes first, but the plan does not authorize inventing one and it
would weaken the clean-cycle evidence.

leaning: Restore the named pre-rehearsal graph snapshot, repair the failed
approval's catalog materialization through the CLI-owned inverse path, and rerun
with an atomic preflight so the expected 374 untouched plus two contested cohort
is measured from the intended baseline.

cost-if-wrong: Preparing the current graph in place could conceal lifecycle
residue and change the auto-promotion denominator; restoring without first
accounting for fork `main` at `e631b875` could instead leave catalog and graph
on incompatible baselines. Either mistake invalidates the cycle and requires
another restore, fork correction, and three-command run.
