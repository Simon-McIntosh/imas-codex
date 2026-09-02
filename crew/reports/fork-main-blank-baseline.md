# Fork catalog blank-main restoration

The fork catalog `Simon-McIntosh/imas-standard-names-catalog` is restored to its blank-main content baseline without rewriting history. Merge commit `6a5c44d38f47921ae954e96222045252adcd8127` was reversed against parent 1 using `git revert --no-commit -m 1`, committed conventionally, and pushed only to fork `main`. The new head contains zero `standard_names/*.yml` files, fork CI is green, the cut-time review tag remains, no fold-back receipt exists, upstream is unchanged, and the graph was not mutated.

## Before and authority

- Fork local and remote `main`: `0e2b182d202c83dcb75b412ff92aad1ee4370b53`.
- Fork tree: `f2954894015c95f31ff4b1782a4a1240b3e054ed`, the same catalog content as the reviewed merge.
- Target merge: `6a5c44d38f47921ae954e96222045252adcd8127`.
- Merge parent 1: `1b596f3c593db6a76adce8b4541fa49b97395fa5`, the blank-main baseline.
- Merge parent 2: `58bfb0f254894c78b53c15e0fe454559eac0a68c`.
- Entry files before: `17` `standard_names/*.yml` files.
- Entry files at baseline `1b596f3c`: `0`.
- Catalog checkout was clean and on `main`; `origin` was the Simon-McIntosh fork and `upstream` was iterorganization.
- Upstream default branch before: `main` at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.
- Graph before: `4683` StandardName nodes, `approved=0`, `contested=0`, `accepted=2336`.

Evidence: `00-preflight.log` and `01-graph-tags-before.log` under the run log directory.

## Revert and push

The authorized operation was applied as a history-preserving first-parent inverse:

```text
git revert --no-commit -m 1 6a5c44d38f47921ae954e96222045252adcd8127
```

The staged inverse contained exactly the merge payload:

- `catalog.yml`: deleted.
- `17` files under `standard_names/`: deleted.
- Diffstat: `18 files changed, 18386 deletions`.
- No conflict and no unrelated path.

The inverse was committed and pushed to fork `main`:

- Revert commit: `1a752add6809049e522e3ff4db4c8d0c7e07795e`.
- Commit subject: `fix(catalog): restore blank review baseline`.
- Commit tree: `8beded8c1748f6ea8292aaeb9b34ff73ca002e24`.
- Push: `0e2b182..1a752ad main -> main` on `origin` only.
- Commit body present; AI-attribution trailer check clean.

## Final content and CI

Fork local and remote `main` both resolve to `1a752add6809049e522e3ff4db4c8d0c7e07795e`.

`git ls-tree -r --name-only HEAD -- standard_names` returned no path. Filtering that tree for `standard_names/*.yml` produced exactly `0`, matching the content cardinality at `1b596f3c`.

GitHub checks on the exact new head:

| Check | Status | Conclusion |
| --- | --- | --- |
| `build` | completed | success |
| `validate` | completed | success |
| `review-edit-guard` | completed | skipped |

The skipped review-edit guard is the expected branch routing on fork `main`; both catalog validation jobs passed.

## Tag, upstream, and graph controls

- The fork tag list retains `v0.3.0rc1+west-task-2e` with subject `WEST review batch`.
- Remote annotated tag object: `665be8f244bf227442507adc44fc69c6a6f8443a`.
- Peeled cut-time commit: `b3ad33253a0e4e92d7003d87510090f45dbe1499`.
- No tag subject or message begins `graph-merged:`; the fold-back receipt remains absent.
- Upstream default branch remains `main` at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`, identical to the pre-revert read.
- Final GraphClient census remains `4683` StandardName nodes, `approved=0`, `contested=0`, `accepted=2336`, identical to the pre-revert census.

Evidence: `02-post-push.log`.

## Verdict

Complete. Fork catalog `main` now holds zero Standard Name YAML entries, matching the blank content baseline at `1b596f3c`; history is preserved by revert commit `1a752add`, fork CI passed, the cut-time tag remains without a graph-merged receipt, upstream was untouched, and the live graph remained at zero approved and zero contested names.
