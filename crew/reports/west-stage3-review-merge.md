# WEST fork reviewer-edit and merge evidence

## Outcome

Fork pull request [Simon-McIntosh/imas-standard-names-catalog#3](https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3) was exercised as the reviewer surface and merged into the fork only. Two API commits changed the `description` field of two different batch entries in `standard_names/machine_operations.yml`; no machine-owned fields changed. The final PR head was `58bfb0f254894c78b53c15e0fe454559eac0a68c`.

## Reviewer edits

### Compliant description improvement

- Entry id: `breakdown_initial_time`
- Reviewer-owned field: `description`
- Before: `Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow.`
- After: `Timestamp at which plasma breakdown begins and discharge current starts to flow.`
- Intent: state the same plasma-breakdown event boundary more directly.
- GitHub API commit: `4155056e56c68e19602dc259a5adb317d9920119`
- API author login: `Simon-McIntosh`
- Commit author: `Simon-McIntosh`

### Deliberately non-compliant description

- Entry id: `pulse_duration`
- Reviewer-owned field: `description`
- Before: `Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.`
- After: `Elapsed duration from plasma breakdown until termination of auxiliary heating in a single discharge.`
- Intended contest: the new terminal event is semantically false. The entry documentation defines the interval through confined-plasma termination and explicitly distinguishes it from an auxiliary-system pulse. This prose is structurally valid but should be contested by the downstream semantic review path.
- GitHub API commit: `58bfb0f254894c78b53c15e0fe454559eac0a68c`
- API author login: `Simon-McIntosh`
- Commit author: `Simon-McIntosh`

The edits were submitted as two separate GitHub Contents API `PUT` operations against branch `review/v0.3.0rc1+west-task-2e`. Reading the file back from the final head confirmed both exact after-texts.

## Final-head CI

The GitHub Checks API reported exactly three check runs for final head `58bfb0f254894c78b53c15e0fe454559eac0a68c`; all three were `completed` with conclusion `success`:

| Workflow / check | Conclusion | Evidence |
|---|---|---|
| Catalog Site / `build` | `success` | [job 100018085824](https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33556310110/job/100018085824) |
| Validate Catalog / `review-edit-guard` | `success` | [job 100017864966](https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33556310130/job/100017864966) |
| Validate Catalog / `validate` | `success` | [job 100017865466](https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33556310130/job/100017865466) |

The guard success is the machine evidence that only reviewer-editable fields changed. The semantic conflict intentionally passes catalog structure validation; the subsequent `sn approve --pr` semantic-review route is responsible for accepting or contesting reviewer prose.

## Fork merge receipt

The merge was submitted through the GitHub Pull Requests API with an expected-head guard set to `58bfb0f254894c78b53c15e0fe454559eac0a68c`.

- PR state: `MERGED`
- PR URL: `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3`
- Fork base: `main`
- Merge commit: `6a5c44d38f47921ae954e96222045252adcd8127`
- `merged_at`: `2026-09-01T20:39:07Z`
- Fork `main` after merge: `6a5c44d38f47921ae954e96222045252adcd8127`

## Upstream non-mutation control

Two GitHub API reads, one before the merge and one after it, returned the same upstream default-branch identity:

- Repository: `iterorganization/imas-standard-names-catalog`
- Reported default branch: `main`
- Default-branch SHA before fork merge: `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`
- Default-branch SHA after fork merge: `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`
- Upstream `pushed_at` after fork merge: `2026-09-01T09:25:18Z`, predating this operation
- Measured upstream delta: `0` commits / `0` SHA change

## Reviewer friction notes

- The GitHub Contents API requires a whole-file replacement even for a one-field edit. Supplying the current blob SHA made both commits race-safe, but the interaction is less ergonomic than a structured entry editor.
- The editable-field guard is fast and clear: it completed in 20 seconds and accepted both prose-only commits.
- Structural validation cannot and should not reject the deliberate semantic contradiction; its green result makes the downstream reviewer contest a necessary, independently observable gate.

## Quantitative completion

- Reviewer edits: `2/2` committed through the GitHub API, affecting `2` distinct entry ids and `1` reviewer-owned field per entry.
- Commit attribution: `2/2` commits authored by `Simon-McIntosh`.
- Final-head checks: `3/3 success`, `0` failed, `0` pending.
- Fork merge: `1/1` PR merged; merge SHA and timestamp recorded from the API.
- Upstream isolation: default branch `main` stayed at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`; measured change `0`.

