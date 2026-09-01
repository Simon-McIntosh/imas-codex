# NEEDS-HELP: compliant reviewer edit was contested and contested nodes lack required PR provenance

## Outcome

The fold-back ran live and wrote durable graph and fork state, but this node is
**blocked**, not complete. The dry-run gate succeeded with exactly two reviewer
changes. The live run then auto-approved 374 untouched batch names and contested
both edited names. This differs from the required split of one edited name
approved and one contested. In addition, the two contested `StandardName` nodes
do not carry the required pull-request URL or reviewer actor provenance.

No override or undo was attempted. The unexpected result is preserved for
adjudication, and the later rehearsal unwind remains available.

## Inputs and immutable identities

- Catalog checkout: `/home/ITER/mcintos/Code/imas-standard-names-catalog`
- Catalog branch before approval: `main`, clean
- Merged PR: `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3`
- GitHub merge commit: `6a5c44d38f47921ae954e96222045252adcd8127`
- GitHub `mergedAt`: `2026-09-01T20:39:07Z`
- PR author/reviewer stand-in: `Simon-McIntosh`; GitHub reports no submitted
  review objects (`reviews: []`), because the two edits were commits on the PR
  branch.
- PR head: `review/v0.3.0rc1+west-task-2e`
- Frozen batch artifact:
  `imas_codex/standard_names/manifests/reviews/v0.3.0rc1+west-task-2e.sn_names.yaml`
- Batch size resolved by the CLI: 409 names
- Approved-catalog baseline used for additive validation:
  `6a5c44d38f47921ae954e96222045252adcd8127^1`
- Reviewer-edit baseline resolved from the cut-time tag:
  `v0.3.0rc1+west-task-2e`, whose prior annotated-tag object was
  `665be8f244bf227442507adc44fc69c6a6f8443a` and peeled to
  `b3ad33253a0e4e92d7003d87510090f45dbe1499`.

## Reviewer edits

Both edits were in `standard_names/machine_operations.yml` and were detected
by ID against the cut-time catalog content.

| Entry ID | Before | Reviewed PR text | Expected | Observed |
|---|---|---|---|---|
| `breakdown_initial_time` | “Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.” | “Timestamp at which plasma breakdown begins and discharge current starts to flow.” | approved | contested, docs score `0.7083333333333334` |
| `pulse_duration` | “Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.” | “Elapsed duration from plasma breakdown until termination of auxiliary heating in a single discharge.” | contested | contested, docs score `0.5083333333333333` |

For both nodes, the persisted reason is the threshold decision rather than a
dimension-level critique:

- `breakdown_initial_time`: `docs edit failed compliance re-review (score 0.708 < 0.850): human catalog PR edit — reviewer-approved documentation change folded back into the ledger; score the wording as-is (do not revert to the prior text).`
- `pulse_duration`: `docs edit failed compliance re-review (score 0.508 < 0.850): human catalog PR edit — reviewer-approved documentation change folded back into the ledger; score the wording as-is (do not revert to the prior text).`

The graph retains the pre-edit descriptions on the contested nodes while
recording `edit_origin='human'`, `docs_stage='drafted'`, and
`name_stage='contested'`.

## Dry-run evidence

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex sn approve --isnc /home/ITER/mcintos/Code/imas-standard-names-catalog --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3 --dry-run
```

Exit status: `0`

```text
Batch artifact: v0.3.0rc1+west-task-2e.sn_names.yaml
Base ref: 6a5c44d38f47921ae954e96222045252adcd8127^1
Batch: 409 name(s)
Mode: dry run
changes seen: 2
accepted: 0
staged for review: 0
auto-approved: 0
contested: 0
quarantined: 0
blocked: 0
unmatched: 0
```

## Live approval evidence

Command: the same invocation without `--dry-run`.

Exit status: `0`

```text
changes seen: 2
accepted: 0
staged for review: 0
auto-approved: 374
contested: 2
quarantined: 0
blocked: 0
unmatched: 0
fold-back tag v0.3.0rc1+west-task-2e -> origin (with summary)
Approval complete
```

Quantitative routing:

- Untouched names auto-promoted `accepted -> approved`: **374**
- Edited names approved: **0** (expected **1**)
- Edited names contested: **2** (expected **1**)
- Other failure buckets: quarantined **0**, blocked **0**, unmatched **0**

The command emitted no dedicated `sn approve` log. The only recently touched
SN log was `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-review.log`,
and it did not contain this invocation's two IDs or scores. The complete
terminal summaries are transcribed above rather than attributing unrelated
shared log content to this run.

## Graph read and provenance failure

The read first established that the property exists and is populated elsewhere,
then inspected the two target nodes:

```cypher
MATCH (sn:StandardName)
RETURN count(sn) AS candidates,
       count(sn.catalog_pr_number) AS with_pr_property,
       sum(CASE WHEN sn.catalog_pr_number = 3 THEN 1 ELSE 0 END) AS pr3_nodes,
       sum(CASE WHEN sn.catalog_pr_number = 3 AND sn.name_stage = 'approved'
                THEN 1 ELSE 0 END) AS pr3_approved,
       sum(CASE WHEN sn.name_stage = 'contested' THEN 1 ELSE 0 END)
         AS contested_total;
```

Result:

```text
candidates=4675, with_pr_property=374, pr3_nodes=374,
pr3_approved=374, contested_total=2
```

Target query:

```cypher
MATCH (sn:StandardName)
WHERE sn.id IN ['breakdown_initial_time', 'pulse_duration']
RETURN sn.id, sn.name_stage, sn.docs_stage, sn.reviewer_score_docs,
       sn.contested_reason, sn.catalog_pr_number, sn.catalog_pr_url,
       sn.catalog_merge_commit_sha, sn.catalog_reviewer_actor,
       sn.edit_origin, sn.edit_reason
ORDER BY sn.id;
```

Both rows have `name_stage='contested'`, `docs_stage='drafted'`, and
`edit_origin='human'`. Their scores and contested reasons are recorded above.
For **both** rows, all four required catalog provenance values are null:
`catalog_pr_number`, `catalog_pr_url`, `catalog_merge_commit_sha`, and
`catalog_reviewer_actor`. This is a positive-control-backed proof that the
required node-level PR URL and reviewer provenance were not written on the
contested route; it is not a guessed-property empty result.

The 374 untouched promoted nodes do carry PR 3 provenance. The missing values
are specific to the contested transition.

## Fork materialization and fold-back tag

The live command created and pushed fork-main commit
`7e6fe7c6972d5202cb1258d059d7b108abf415a8` (`catalog: materialize approved entries`).
It removed the unapproved additions and recorded trailers for PR 3, its URL,
and pre-fold-back commit `6a5c44d3`. Remote `origin/main` resolves to this
commit. No upstream write was attempted.

The fork tag is:

- Name: `v0.3.0rc1+west-task-2e`
- Annotated-tag object: `67789917719cd9694eef3bd8e407c985ee76ea20`
- Peeled target: `6a5c44d38f47921ae954e96222045252adcd8127`
- Remote API ref object: `67789917719cd9694eef3bd8e407c985ee76ea20`

Its contract message begins:

```text
graph-merged: 2026-09-01T21:10:10.171799+00:00
pr: #3 https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
batch: v0.3.0rc1+west-task-2e.sn_names.yaml
outcomes: approved=0 staged_for_review=0 auto_approved=374 contested=2
prior-tag-ref: 665be8f244bf227442507adc44fc69c6a6f8443a
```

## Idempotency evidence

The exact live command was invoked a second time. Its inner process exit status
was `1`, with:

```text
Refusing: v0.3.0rc1+west-task-2e already carries the fold-back contract tag —
this PR has been folded back into the graph. Use sn approve --undo first to
re-fold.
```

The refusal occurred before another graph fold-back.

## Unexpected out-of-scope mutation

The live CLI also modified the frozen manifest in this worker worktree, which
is outside the node's exclusive write scope:

```diff
-merge_commit: null
+merge_commit: 6a5c44d38f47921ae954e96222045252adcd8127
```

Path:
`imas_codex/standard_names/manifests/reviews/v0.3.0rc1+west-task-2e.sn_names.yaml`

The file timestamp is `2026-09-01 23:10:10.067753 +0200`, coincident with the
live approval. It was not staged, committed, restored, or otherwise touched
after detection.

## Blocker hand-off

tried: Pulled/verified fork main at merge `6a5c44d3`, ran the repaired dry-run
with exactly two changes, ran the live approval once, queried the graph with a
property-existence positive control, verified the fork tag/API ref, and proved
the idempotency refusal.

options: (1) extend the contested transition to stamp merged-PR provenance on
the node and decide whether the compliant description needs a deterministic or
better-grounded review path, then unwind and re-fold; (2) accept the two-contest
result as a valid stochastic rubric outcome and revise the evidence contract,
while still repairing contested-route provenance; or (3) explicitly adjudicate
`breakdown_initial_time` through the existing contested-resolution workflow,
which would be an override and requires authority not granted to this node.

leaning: Option 1. The node's stated expectation is a semantic test of both
routes, and provenance is independently absent on the contested path. Repair
both mechanisms before repeating the rehearsal; do not override a score merely
to manufacture the expected count.

cost-if-wrong: Repeating requires `sn approve --undo`, restoration/review of the
fork materialization state and cut tag, another paid two-entry review, and a new
fold-back receipt. Choosing an override instead would preserve an unproven
review route and weaken the intended compliance evidence.
