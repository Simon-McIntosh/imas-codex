# Fully repaired WEST fold-back cycle

The repaired cycle completed end to end on the live graph. Two pre-fix undo residues were restored, approval folded PR 3 back with `384` untouched auto-promotions and two contested edits, `breakdown_initial_time` was adjudicated approved, and undo returned the graph to `approved=0` and `contested=0`. The final fork tree equals the PR 3 merge tree, the receipt is gone, the cut-time tag is restored, all five PR provenance fields are null across all 409 batch identities, the review manifest remained byte-identical, and upstream was untouched.

The earlier expectation of `374` untouched promotions was stale: three intervening cost-capped flush passes moved another ten batch members from reviewed to accepted. The final read-only cohort check confirms exactly `386` batch members are accepted on both axes: `384` untouched names plus the two edited names.

## Restore-point lifecycle repair

The authorized Cypher matched exactly the two edited identities, required both prior values to be `drafted`, and set only `docs_stage`. Its first parse-only form was rejected before execution because Neo4j required an intervening `WITH`; the corrected atomic query exited `0` and recorded:

| Standard Name | Before | After |
| --- | --- | --- |
| `breakdown_initial_time` | `drafted` | `accepted` |
| `pulse_duration` | `drafted` | `accepted` |

Evidence: `02-lifecycle-restoration.log`. The pre-approval census in `03-before-approve.log` then showed both nodes `accepted/accepted`, global `approved=0`, `contested=0`, `accepted=2336`, and no populated PR-number provenance.

## Approval

Command:

```text
IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog \
uv run --no-sync imas-codex sn approve \
  --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
```

Exit status: `0`.

| Outcome | Count |
| --- | ---: |
| Changes seen | 2 |
| Auto-approved untouched names | 384 |
| Contested edited names | 2 |
| Accepted edited names | 0 |
| Staged for review | 0 |
| Quarantined | 0 |
| Blocked | 0 |
| Unmatched | 0 |

Post-approval census: `approved=384`, `contested=2`, `accepted=1950`, with PR-number provenance on `386` nodes.

Both contested rows carried complete reviewer provenance:

| Standard Name | Docs score | PR | Merge SHA | Reviewer | Reason |
| --- | ---: | ---: | --- | --- | --- |
| `breakdown_initial_time` | `0.7166666666666667` | 3 | `6a5c44d38f47921ae954e96222045252adcd8127` | `Simon-McIntosh` | docs edit score `0.717 < 0.850` |
| `pulse_duration` | `0.5416666666666666` | 3 | `6a5c44d38f47921ae954e96222045252adcd8127` | `Simon-McIntosh` | docs edit score `0.542 < 0.850` |

For both nodes, `catalog_pr_url` was `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3`; all four required fields—PR number, URL, merge SHA, and reviewer actor—were non-null.

The fold-back receipt existed after approval. Its subject began `graph-merged: 2026-09-02T00:55:25.736578+00:00` and its deterministic outcome line recorded `approved=0 staged_for_review=0 auto_approved=384 contested=2`. Catalog materialization head was `b930e66c4082653a1c425daf2d1ef33a56b5c2da`, tree `447efdd1f663645fe5a50366a7aaa43be955c1a0`.

Evidence: `04-approve.log` and `05-after-approve-before-resolve.log`.

## Contested adjudication

Command:

```text
uv run --no-sync imas-codex sn resolve breakdown_initial_time --override \
  --reason 'The shorter wording preserves the breakdown-onset semantics while stating the event boundary more directly.'
```

Exit status: `0`.

After resolution:

- `breakdown_initial_time` was `name_stage='approved'`, `docs_stage='accepted'`.
- Its description exactly equaled `Timestamp at which plasma breakdown begins and discharge current starts to flow.`
- Its `contested_resolution` exactly equaled the supplied reason.
- `pulse_duration` remained contested.
- The pre-undo census was `approved=385`, `contested=1`, `accepted=1950`.

Evidence: `06-resolve.log` and `07-after-resolve-before-undo.log`.

## Undo

Command:

```text
IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog \
uv run --no-sync imas-codex sn approve --undo \
  --pr https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/3
```

Exit status: `0`.

Undo reported:

- Approved to accepted: `385`.
- Contested to accepted: `1`.
- Fold-back receipt deleted from the fork.
- Accepted human edits retained as graph history.

Evidence: `08-undo.log`.

## Final live closure

The final graph census is:

| Stage | Count |
| --- | ---: |
| Approved | 0 |
| Contested | 0 |
| Accepted | 2336 |

Both edited nodes are `name_stage='accepted'` and `docs_stage='accepted'`:

- `breakdown_initial_time` retains the reviewer-edited description and the shorter-wording `contested_resolution`; `edit_origin='human'`.
- `pulse_duration` retains the reviewer-edited description and `contested_resolution='approval of catalog PR 3 unwound'`; `edit_origin='human'`.

The standalone read-only instrument `logs/closure_read.py` avoided nested shell quoting and checked the complete frozen batch:

| Measure | Result |
| --- | ---: |
| Frozen manifest identities | 409 |
| Graph identities matched | 409 |
| `name_stage='accepted' AND docs_stage='accepted'` | 386 |
| `catalog_pr_number` non-null | 0 |
| `catalog_pr_url` non-null | 0 |
| `catalog_merge_commit_sha` non-null | 0 |
| `catalog_reviewer_actor` non-null | 0 |
| `catalog_approved_at` non-null | 0 |
| Identities with any of the five fields non-null | 0 |

Thus the current eligible cohort partitions exactly into `384` untouched names plus the two edited names. Evidence: `10-batch-closure.log`, exit `0`.

The final fork and upstream controls also exited `0`:

- Fork local and remote main: `0e2b182d202c83dcb75b412ff92aad1ee4370b53`.
- Fork main tree: `f2954894015c95f31ff4b1782a4a1240b3e054ed`.
- PR 3 merge tree: `f2954894015c95f31ff4b1782a4a1240b3e054ed`—byte-level tree equality holds.
- The fork tag list contains `v0.3.0rc1+west-task-2e` with subject `WEST review batch`; remote annotated-tag object `665be8f244bf227442507adc44fc69c6a6f8443a` peels to `b3ad33253a0e4e92d7003d87510090f45dbe1499`.
- No `graph-merged:` receipt remains in the fork tag list.
- Upstream default branch is `main` at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`, unchanged from the pre-cycle control.
- Review manifest worktree blob and HEAD blob both equal `dd46e21250c3e4aad9259a2f58d87a2feff5fbab`; SHA-256 remains `d7f5b833cddcf17ae67318719a9b14d3ea1dd4a5e337b4a8c7a3b43eee9f122a`.

Evidence: `11-final-remotes.log`.

## Verdict

Complete. Both authorized repairs were recorded, all three lifecycle commands exited `0`, both reviewer edits took the intended contested route, override adjudication preserved the exact reviewer wording, undo returned every folded row to accepted, all batch PR provenance was cleared, the receipt was removed, the cut-time tag and PR-merge catalog tree were restored, the frozen review manifest did not change, and upstream remained untouched.
