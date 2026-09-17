<meta name="plan-slug" content="catalog-review-surface">
<meta name="plan-status" content="active">
<meta name="plan-evidence-for" content="catalog-review-surface">
<meta name="docs-project" content="imas-codex">

# The fold-back, rehearsed on a throwaway request

The approval fold-back is exercised end to end
 on a purpose-made fork request carrying one edit the lifecycle
guard promotes and one it refuses, and the exercise is unwound with the
approval property map returning to its pre-approval state, leaving nothing
lasting. Logs: `logs/rehearsal.log`, `logs/refusal.log`,
`logs/request-close.log`.

## The request

Fork request **20** on `Simon-McIntosh/imas-standard-names-catalog`, opened
against base `review/v0.4.0rc7+west-task-2e` from head
`rehearsal/fold-back-20260917`, carrying exactly two catalog-entry renames:

| Edit | Entry | Change | Pipeline behaviour |
|---|---|---|---|
| accepted | `general.yml` | `energy_confinement_enhancement_factor` → `energy_confinement_enhancement_factor_h98` | promoted, receipt written |
| refused | `structural_components.yml` | `vertical_outline_of_limiter_tile` → `limiter tile outline` | refused |

It is a fresh purpose-made request and **not** the open WEST request, which
carries the real batch and real reviewers and is untouched by this rehearsal.

The pipeline reads both edits from that branch through the production
`read_pr_changes` against the request's base, `changes_seen` = 2:

```
{"sn_id": "energy_confinement_enhancement_factor", "axis": "name",
 "new_value": "energy_confinement_enhancement_factor_h98"}
{"sn_id": "vertical_outline_of_limiter_tile", "axis": "name",
 "new_value": "limiter tile outline"}
```

## The refusal the pipeline returns

The refused edit renames an entry to a name containing spaces, which fails the
naming grammar, so the lifecycle guard refuses the promotion rather than
absorbing it. Quoted verbatim from the command that produced it, which exited 1:

```
✗ 1 catalog promotion(s) refused:
  - limiter tile outline: catalog lifecycle promotion preconditions were not met
```

The same refusal through the production `mark_catalog_name_approved` gate: the
statement matches nothing, returns false, and the promotion is recorded as
refused under the reason string above. The request still succeeds for the other
edit — a refusal is per-edit, not per-request.

## The receipt, read back from the graph

The accepted edit is promoted through the production writer
`mark_catalog_name_approved`, and the traceability receipt is read back from the
graph while the edit stands approved:

| Field | Value |
|---|---|
| `catalog_pr_number` | **20** |
| `catalog_pr_url` | `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/20` |
| `catalog_merge_commit_sha` | `6cae87c49025bedfaafdf3aff698dcbb7efbca00` |
| `catalog_reviewer_actor` | **Simon-McIntosh** |
| `catalog_approved_at` | set (transaction timestamp) |

Both required fields are non-null: `catalog_pr_number` = 20 and
`catalog_reviewer_actor` = `Simon-McIntosh`. The identity reaches
`name_stage = approved` and `status = active` while the approval stands.

## The unwind

`undo_approval` is the production unwind. **7 properties** were compared across
the unwind — `catalog_approved_at`, `catalog_pr_number`, `catalog_pr_url`,
`catalog_merge_commit_sha`, `catalog_reviewer_actor`, `name_stage`, `status` —
and **0 differ**. Each of the five approval fields is null again afterwards, and
the stage returns to `accepted` / `draft`, exactly as it stood before the
approval.

Named properties compared: 7. Differing: **0**.

## The substrate, stated

The rehearsal runs against the **live graph engine** with every statement inside
one explicit transaction that is rolled back and never committed
(`logs/rehearsal.log`, `rehearsal.py`). The production statements therefore
execute for real — a `SET` dropped from either the writer or the unwind changes
the measured diff — while the rehearsal costs no graph state: the purpose-made
identities do not exist before the transaction and do not exist after it. The
graph holds 5,130 `StandardName` rows before and after.

## The request is closed and its branch is gone

```
✓ Closed pull request Simon-McIntosh/imas-standard-names-catalog#20
✓ Deleted branch rehearsal/fold-back-20260917
```

Read back with `gh pr view`:

```json
{"headRefName":"rehearsal/fold-back-20260917","number":20,"state":"CLOSED"}
```

And the branch is absent from the fork, `git ls-remote` returning no rows at
exit 0:

```
$
```

So the request is CLOSED and its head branch no longer exists on the remote.

## Validation

`uv run reckon audit-doc` on this file exits 0.