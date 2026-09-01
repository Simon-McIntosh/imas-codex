NEEDS-HELP: override approved the node but did not materialize the reviewed description or catalog provenance

# Contested-name adjudication evidence

## Outcome

The authorized override command succeeded and demonstrated the core
`contested -> approved` transition for `breakdown_initial_time`:

- command exit status: **0**
- `breakdown_initial_time`: `contested -> approved`
- `contested_resolution`: recorded exactly as supplied
- internal change event: `operation='content_edit'`, recorded at
  `2026-09-01T21:28:54.72Z`
- `pulse_duration`: remained `contested`
- graph-wide contested count: **2 -> 1**

The node is nevertheless **blocked** against its full evidence contract.
`breakdown_initial_time.description` remains the pre-review wording rather than
the reviewer's shorter wording, and all four `catalog_*` provenance fields
remain null. No additional mutation, direct Cypher repair, repeat resolve, or
override of `pulse_duration` was attempted.

## Required before-state read

The query was aimed at the two known catalog-review IDs and projected the exact
required properties:

```cypher
MATCH (sn:StandardName)
WHERE sn.id IN ['breakdown_initial_time', 'pulse_duration']
RETURN sn.id AS id,
       sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage,
       sn.reviewer_score_docs AS reviewer_score_docs,
       sn.contested_reason AS contested_reason,
       sn.contested_resolution AS contested_resolution,
       sn.description AS description,
       sn.catalog_pr_number AS catalog_pr_number,
       sn.catalog_pr_url AS catalog_pr_url,
       sn.catalog_merge_commit_sha AS catalog_merge_commit_sha,
       sn.catalog_reviewer_actor AS catalog_reviewer_actor
ORDER BY id;
```

Positive control:

```text
candidates=4675
candidates_with_id=4675
with_catalog_pr_url=374
contested_total=2
```

### `breakdown_initial_time` before

```text
name_stage: contested
docs_stage: drafted
reviewer_score_docs: 0.7083333333333334
contested_reason: docs edit failed compliance re-review (score 0.708 < 0.850): human catalog PR edit — reviewer-approved documentation change folded back into the ledger; score the wording as-is (do not revert to the prior text).
contested_resolution: null
description: Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.
catalog_pr_number: null
catalog_pr_url: null
catalog_merge_commit_sha: null
catalog_reviewer_actor: null
```

The reviewer-edited description expected after adjudication was:

```text
Timestamp at which plasma breakdown begins and discharge current starts to flow.
```

The shorter version preserves the same breakdown-onset event: plasma breakdown
begins at the instant discharge current starts to flow. It removes circumlocution
without changing the physical boundary.

### `pulse_duration` before

```text
name_stage: contested
docs_stage: drafted
reviewer_score_docs: 0.5083333333333333
contested_reason: docs edit failed compliance re-review (score 0.508 < 0.850): human catalog PR edit — reviewer-approved documentation change folded back into the ledger; score the wording as-is (do not revert to the prior text).
contested_resolution: null
description: Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.
catalog_pr_number: null
catalog_pr_url: null
catalog_merge_commit_sha: null
catalog_reviewer_actor: null
```

Its rejected reviewer wording would have ended the plasma duration at
termination of auxiliary heating. That changes the terminal event and is not
semantically equivalent to termination of the confined plasma, so this entry
was intentionally left contested.

## Resolve command

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex sn resolve breakdown_initial_time --override --reason "The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly."
```

Exit status: **0**

Output:

```text
✓ breakdown_initial_time -> approved (override: The shorter wording preserves the
breakdown-onset semantics while stating the event boundary more directly.)
SN_RESOLVE_EXIT=0
```

The command emitted no dedicated CLI log under
`/home/ITER/mcintos/.local/share/imas-codex/logs/`; its complete terminal output
is transcribed above.

## Required after-state read

The same Cypher projection was run after the command.

Positive control and aggregate:

```text
candidates=4675
candidates_with_id=4675
with_catalog_pr_url=374
contested_total=1
```

### `breakdown_initial_time` after

```text
name_stage: approved
docs_stage: drafted
reviewer_score_docs: 0.7083333333333334
contested_reason: docs edit failed compliance re-review (score 0.708 < 0.850): human catalog PR edit — reviewer-approved documentation change folded back into the ledger; score the wording as-is (do not revert to the prior text).
contested_resolution: The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.
description: Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.
catalog_pr_number: null
catalog_pr_url: null
catalog_merge_commit_sha: null
catalog_reviewer_actor: null
```

The transition and reason meet the adjudication-path requirement, but the
description does **not** equal the reviewed text. It remains:

```text
Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.
```

rather than:

```text
Timestamp at which plasma breakdown begins and discharge current starts to flow.
```

The internal ledger confirms the override:

```text
operation: content_edit
reason: The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.
changed_at: 2026-09-01T21:28:54.72Z
```

### `pulse_duration` after

Every projected value was unchanged:

```text
name_stage: contested
docs_stage: drafted
reviewer_score_docs: 0.5083333333333333
contested_resolution: null
description: Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.
catalog_pr_number: null
catalog_pr_url: null
catalog_merge_commit_sha: null
catalog_reviewer_actor: null
```

## Provenance finding

`sn resolve` did **not** stamp any of the four catalog provenance fields. This
is expected from the currently implemented resolver, which changes
`name_stage`, `contested_resolution`, and `catalog_approved_at` and records an
internal change, but assumes the contested node already carries the PR fields.
The repaired approval code stamps those fields for future contested transitions;
the two live rows predate that repair and were not retroactively migrated.

## Blocker hand-off

tried: Read both target nodes with a key/property positive control, ran the
authorized `sn resolve ... --override --reason` once, repeated the identical
read, and verified the internal `content_edit` event.

options: (1) authorize a provenance-and-reviewed-text recovery through a normal
catalog-aware command path, then verify this approved node again; (2) implement
and test a resolver enhancement that materializes the stored reviewer proposal
and refuses when that proposal is unavailable; or (3) revise the evidence
contract to treat the existing description plus recorded resolution as the
intended adjudication result, while separately migrating provenance.

leaning: Option 1 if an existing normal workflow can reconstruct the exact PR
proposal; otherwise option 2. Direct Cypher text replacement would bypass the
Standard Names edit/review authority and is not acceptable.

cost-if-wrong: A mistaken recovery could publish wording that was not the PR's
reviewed proposal or fabricate provenance. Re-running `resolve` cannot help
because the node is no longer contested and the command correctly refuses
non-contested inputs.
