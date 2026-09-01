NEEDS-HELP: undo completed externally but left breakdown_initial_time approved because its PR provenance had already been cleared

# WEST catalog fold-back undo

## Outcome

The authorized undo command completed with exit status 0 after the detached
worktree was pointed at the verified ISNC checkout. It demoted 374 approved
names and the one contested name to `accepted`, cleared all remaining PR 3
catalog provenance, unwound the catalog materialization commit, and restored
the cut-time RC tag on the fork. The required graph end state was not reached:
one approved name remains.

The residual is `breakdown_initial_time`. Before undo it was approved but all
four `catalog_*` provenance fields were already null, so the undo selector for
approved names did not include it. No direct graph repair was attempted because
that would exceed this node's report-only write scope and would obscure the
observed undo defect.

## Graph census

Schema-backed reads used `GraphClient` against the live `codex` graph.

| Census | Before | After | Required after |
|---|---:|---:|---:|
| Total `StandardName` nodes | 4,683 | 4,683 | unchanged |
| `name_stage='approved'` | 375 | 1 | 0 |
| `name_stage='contested'` | 1 | 0 | 0 |
| `name_stage='accepted'` | 1,960 | 2,335 | 2,336 |
| `catalog_pr_number=3` | 374 | 0 | 0 |

The accepted count increased by 375, exactly matching the command's 374
approved demotions plus one contested demotion, but full restoration requires
one further approved-to-accepted transition for `breakdown_initial_time`.

## Undo invocation

Initial invocation, exactly as requested:

```text
uv run --no-sync imas-codex sn approve --undo --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
ISNC not found. Set IMAS_CODEX_SN_ISNC env var or clone
imas-standard-names-catalog as a sibling directory.
SN_APPROVE_UNDO_EXIT=2
```

This failed before mutation because the detached worktree cannot discover the
catalog checkout through the command's sibling-directory fallback. The same
command was retried with the documented environment binding
`IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog`:

```text
Batch artifact: v0.3.0rc1+west-task-2e.sn_names.yaml (from v0.3.0rc1+west-task-2e)
Approval of PR #3 unwound
approved → accepted: 374
contested → accepted: 1
fold-back tag v0.3.0rc1+west-task-2e deleted (origin)
Accepted human edits remain graph history — revert wording via sn edit.
SN_APPROVE_UNDO_EXIT=0
```

The catalog checkout remained clean. The command created and pushed inverse
catalog commit `408eb92258e11f09bcdc54a4a19dc2ad47a5951f` (`catalog: unwind
approved materialization`) after materialization commit
`7e6fe7c6972d5202cb1258d059d7b108abf415a8`.

## Reviewer-edit history and provenance

| Name | Before stage | After stage | `contested_resolution` after | `edit_origin` after | Four catalog fields after |
|---|---|---|---|---|---|
| `breakdown_initial_time` | approved | **approved (residual)** | `The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.` | `human` | all null |
| `pulse_duration` | contested | accepted | `approval of catalog PR 3 unwound` | `human` | all null |

The four checked provenance fields were `catalog_pr_number`, `catalog_pr_url`,
`catalog_merge_sha`, and `catalog_reviewer`; all are null on both entries after
undo. `breakdown_initial_time` retains the reviewer-edited description:

> Timestamp locating the onset of plasma breakdown, defined by plasma
> initiation and the beginning of discharge-current flow.

`pulse_duration` also retains the human-edited description:

> Elapsed duration of the confined-plasma phase in a single discharge, from
> plasma breakdown until termination of the confined plasma.

This demonstrates the command's stated contract: undo reverses catalog
approval state and provenance, but does not un-apply accepted human wording.

## Fork tag receipt

The fork now exposes the cut-time RC tag rather than the graph-merged receipt:

```text
refs/tags/v0.3.0rc1+west-task-2e
  annotated object: 665be8f244bf227442507adc44fc69c6a6f8443a
  peeled commit:    b3ad33253a0e4e92d7003d87510090f45dbe1499
  subject/message:  WEST review batch
```

`git ls-remote --tags origin 'refs/tags/v0.3.0rc1+west-task-2e*'` returned the
same annotated object and peeled commit. The former graph-merged annotated
object `67789917719cd9694eef3bd8e407c985ee76ea20` and its
`graph-merged: 2026-09-01T21:10:10.171799+00:00` message are no longer exposed
by that fork ref. Thus the receipt is gone while the cut-time RC tag remains.

## Blocker handoff

tried: Ran the exact undo once; it exited 2 before mutation because ISNC was
not discoverable from the detached worktree. Retried with the verified catalog
path in `IMAS_CODEX_SN_ISNC`; it exited 0 and performed 374 approved plus one
contested demotions, but the post-read found one residual approved node.

options: Extend undo selection to cover batch entries whose catalog provenance
was cleared by adjudication and re-run the narrowly scoped repair; or explicitly
demote `breakdown_initial_time` through a sanctioned CLI recovery operation,
then verify the 0/0/2,336 census. Do not use an ad hoc Cypher `SET`.

leaning: Repair the undo implementation so batch membership is the fallback
identity for an approved entry with cleared provenance. That addresses the
mechanism and keeps future undo runs complete without manual graph edits.

cost-if-wrong: A one-off state repair would make this rehearsal pass while
leaving the undo defect reproducible; an over-broad fallback could demote an
unrelated approval, so the selector must remain bounded to the PR batch.
