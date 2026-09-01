# Fork catalog materialization inverse

## Outcome

The receipt-less materialization commit was reversed on the fork only. Fork
catalog `main` advanced from
`e631b875434c5b4544d9103201571943838119d4` to
`3edfcc138483bc23e734a7514e15d7c9dce9ee89`; the new head's complete Git tree
is byte-identical to PR merge `6a5c44d38f47921ae954e96222045252adcd8127`.

The upstream catalog default branch remained untouched at
`a06e52052d4776b25e94fdfaa22c2bc6651a98eb`. The live graph remained unwound
at 0 approved and 0 contested Standard Names.

## Preconditions

The catalog checkout at `/home/ITER/mcintos/Code/imas-standard-names-catalog`
was clean on branch `main`. Both local `HEAD` and fork `origin/main` resolved to:

```text
e631b875434c5b4544d9103201571943838119d4
```

The selected commit had parent
`408eb92258e11f09bcdc54a4a19dc2ad47a5951f` and recorded the failed fold-back
for catalog PR 3. Its diff was exactly:

```text
 standard_names/machine_operations.yml     | 70 -------------------------------
 standard_names/magnetic_field_systems.yml | 47 ---------------------
 2 files changed, 117 deletions(-)
```

Before mutation, the upstream catalog repository was read through both the
GitHub API and its configured Git remote:

```text
repository:     iterorganization/imas-standard-names-catalog
default branch: main
GitHub API SHA: a06e52052d4776b25e94fdfaa22c2bc6651a98eb
Git remote SHA: a06e52052d4776b25e94fdfaa22c2bc6651a98eb
```

## Authorized inverse

The explicitly authorized operation was applied as:

```text
git revert --no-commit e631b875434c5b4544d9103201571943838119d4
```

The resulting two-path inverse was committed with a conventional subject and
body, then pushed only to the fork's `origin/main`:

```text
3edfcc138483bc23e734a7514e15d7c9dce9ee89
fix(catalog): undo receipt-less materialization
```

Diffstat:

```text
 standard_names/machine_operations.yml     | 70 +++++++++++++++++++++++++++++++
 standard_names/magnetic_field_systems.yml | 47 +++++++++++++++++++++
 2 files changed, 117 insertions(+)
```

The commit recreated `standard_names/machine_operations.yml` and restored the
47 removed lines in `standard_names/magnetic_field_systems.yml`. Commit-body
presence passed and the AI-attribution trailer check was clean. A merge-mode
`git pull --no-rebase origin main` immediately before push reported already up
to date. Push advanced only:

```text
origin/main: e631b875 -> 3edfcc13
```

## Byte-level catalog-tree equality

The complete repository trees, not merely the two changed files, resolve to the
same Git tree object:

```text
3edfcc138483bc23e734a7514e15d7c9dce9ee89^{tree}
  f2954894015c95f31ff4b1782a4a1240b3e054ed

6a5c44d38f47921ae954e96222045252adcd8127^{tree}
  f2954894015c95f31ff4b1782a4a1240b3e054ed
```

Because Git tree identity covers every tracked path's name, mode, and blob
identity recursively, the matching tree SHA is a byte-level equality proof.
An independent `git diff --exit-code 6a5c44d3 HEAD` exited 0 with no output.
The catalog checkout was clean afterward.

## Fork tags

The only fork tag matching `v0.3.0*` is the cut-time RC tag:

```text
tag:              v0.3.0rc1+west-task-2e
annotated object: 665be8f244bf227442507adc44fc69c6a6f8443a
peeled commit:    b3ad33253a0e4e92d7003d87510090f45dbe1499
message:          WEST review batch
```

`git ls-remote --tags origin 'refs/tags/v0.3.0*'` returned exactly the annotated
object and its peeled commit above. The tag object's full contents contain no
`graph-merged:` receipt.

## Upstream non-mutation

After the fork push, the GitHub API still reported:

```text
repository:     iterorganization/imas-standard-names-catalog
default branch: main
SHA:            a06e52052d4776b25e94fdfaa22c2bc6651a98eb
```

The configured upstream Git remote reported the same SHA. It is identical to
the pre-inverse read, proving the fork-only push did not alter upstream.

## Live graph census

The schema-backed GraphClient read carried a positive control by counting the
`name_stage` property on every Standard Name:

| Metric | Count |
|---|---:|
| Total `StandardName` nodes | 4,683 |
| Rows with `name_stage` | 4,683 |
| `name_stage='approved'` | 0 |
| `name_stage='contested'` | 0 |
| `name_stage='accepted'` | 2,336 |

Representative reviewer-edited nodes remain accepted and were not modified by
the catalog Git inverse:

| Name | Name stage | Docs stage | Name score | Docs score | Description |
|---|---|---|---:|---:|---|
| `breakdown_initial_time` | accepted | drafted | 0.925 | 0.708333 | Timestamp locating the onset of plasma breakdown, defined by plasma initiation and the beginning of discharge-current flow. |
| `pulse_duration` | accepted | drafted | 0.88125 | 0.508333 | Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma. |

## Quantitative closure

- Fork main corrected: 1 commit, 2 paths, 117 insertions.
- Complete-tree equality: identical tree SHA
  `f2954894015c95f31ff4b1782a4a1240b3e054ed` at new fork head and PR merge.
- Fork receipt state: 1 cut-time RC tag, 0 `graph-merged:` receipts.
- Upstream delta: 0 commits; default branch stayed `a06e5205`.
- Graph delta from the inverse: none; approved 0 and contested 0 before and after.
- Follow-on work: none within this node. Approval eligibility/atomicity is owned
  by the separately dispatched source-repair node.
