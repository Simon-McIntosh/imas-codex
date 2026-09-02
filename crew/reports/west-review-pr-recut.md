# WEST review candidate recut

## Outcome

The schema-failing candidate was retired and replaced by a validated, fork-only review candidate. Pull request 7 contains 295 published Standard Name entries selected from the frozen 368-name WEST cohort. Its GitHub diff is pure addition: 18 added files, 0 modified files, 0 deleted files, 16,419 added lines, and 0 deleted lines. Both catalog validation contexts succeeded, all build contexts succeeded, the review-edit guard succeeded, and the PR-scoped preview returned HTTP 200.

The release did not mutate the graph: the before and after census was 4,683 total StandardName nodes, 2,336 accepted, 0 approved, and 0 contested. The upstream catalog default branch was also unchanged at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.

## Superseded candidate

- Fork pull request: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/6>
- Final state: `CLOSED`
- Successor comment, posted by `Simon-McIntosh` at `2026-09-02T07:29:18Z`: “Superseded by the validated WEST production data dictionary standard names review in PR 7.”
- Comment receipt: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/6#issuecomment-5506065026>
- Closed at: `2026-09-02T07:29:20Z`

## Release-state input

The pre-cut command was:

```text
uv run --no-sync imas-codex sn release status
```

It exited 0. Its material output was:

```text
ISNC Release Status
  Path: /home/ITER/mcintos/Code/imas-standard-names-catalog
  State: rc
  Latest tag: v0.3.0rc2+west-task-2e
  Batch RC: +west-task-2e (cut against a review batch)
  Remote (origin): git@github.com:Simon-McIntosh/imas-standard-names-catalog.git
  Remote (upstream): git@github.com:iterorganization/imas-standard-names-catalog.git
  GitHub Pages: yes

Available commands:
  sn release -m 'Next RC' (next RC of v0.3.0rc2+west-task-2e)
  sn release --final -m 'Finalize' (stable)
```

Full PTY transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T072441771421-n-west-review-pr-recut/logs/release-status.log`.

## Cut command and output

`IMAS_CODEX_SN_ISNC` was bound to the clean catalog checkout at `/home/ITER/mcintos/Code/imas-standard-names-catalog`. The exact command was:

```text
uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto -m "WEST production standard names review batch" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T072441771421-n-west-review-pr-recut/pr-body.md
```

Exit status: `0`.

The command resolved `--target auto` to the fork, minted a 368-name frozen cohort with 28 unmatched sources, published 295 catalog entries, withheld 73 entries, pushed the review branch and annotated RC tag to `origin`, and opened pull request 7. It also reported 9 post-copy divergence entries. The pre-existing RC advisories were Gate A graph-test failures, 331 dangling documentation links, 175 divergence entries, and two grammar-excluded coordinate names; none became a catalog validation failure. There was no `extra_forbidden` schema finding.

Full PTY transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T072441771421-n-west-review-pr-recut/logs/release-recut.log`.

## Tag and pull request

- Annotated tag: `v0.3.0rc3+west-task-2e`
- Tag object SHA: `ab0faddfb671bd0c2584b85b2b05d1cd30d01845`
- Tag target commit: `e288b464caf7c723e2374f08792cf4f490b405e3`
- Tag message: `WEST production standard names review batch`
- Pull request: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/7>
- State: `OPEN`
- Base: `main` at `1a752add6809049e522e3ff4db4c8d0c7e07795e`
- Head: `review/v0.3.0rc3+west-task-2e` at `e288b464caf7c723e2374f08792cf4f490b405e3`
- Title: `WEST production data dictionary standard names review`

Published body, byte-for-byte apart from GitHub's terminal newline:

> This WEST standard names review batch publishes 295 entries selected from the frozen 368-name candidate minted from the `west_production_dd_paths` source manifest.
>
> Coordinate grids and ordinates remain excluded because they are structural indexing rather than physical quantities.
>
> Review every entry under the catalog [REVIEWING.md](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/main/REVIEWING.md) contract, prioritizing physics meaning before machine-enforced details.
>
> Inspect the rendered [PR-scoped preview](https://simon-mcintosh.github.io/imas-standard-names-catalog/pr-7/) before approving.

GitHub's pull-files API returned 18 `added` statuses and no other status. The pull-request summary independently reported `changedFiles=18`, `additions=16419`, and `deletions=0`. The 18 additions are `catalog.yml` plus 17 `standard_names/*.yml` files.

## Catalog CI and preview

Final check conclusions on head `e288b464caf7c723e2374f08792cf4f490b405e3`:

| Check | Conclusion | Duration | Receipt |
|---|---:|---:|---|
| `validate` | success | 2m21s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604212377/job/100164410372> |
| `validate / validate` | success | 3m25s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604208179/job/100164397645> |
| `review-edit-guard` | success | 12s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604212377/job/100164410474> |
| `build` (pull request) | success | 42s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604212367/job/100164600426> |
| `build` (review branch) | success | 45s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604207886/job/100164397237> |
| `build` (tag release) | success | 46s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604208179/job/100165300874> |
| `validate / review-edit-guard` | skipped by matrix routing | 0s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33604208179/job/100164399325> |

The preview workflow posted <https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-7/> in pull-request comment <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/7#issuecomment-5506132025>. A direct `curl -L` read returned HTTP `200` at that URL after the build completed.

## Frozen review identity

- Artifact: `/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/ship-s10-20260901/n-west-review-pr-recut/imas_codex/standard_names/manifests/reviews/v0.3.0rc3+west-task-2e.sn_names.yaml`
- Frozen entries: `368`
- Published entries: `295`
- Withheld entries: `73`
- Unmatched sources: `28`
- Recorded PR: number `7`, URL <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/7>
- SHA-256: `6fbdb8d2f3a4f9785f233e28f3180a54bd4cdcaad33a143f5735b1e102d1d83d`

Representative graph records retained their accepted review state during the cut:

- `breakdown_initial_time`: “Timestamp at which plasma breakdown begins and discharge current starts to flow.” Source binding `dd:summary/time_breakdown/value`; name score `0.925`; docs score `0.7166666666666667`; stages `accepted/accepted`.
- `pulse_duration`: “Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.” Name score `0.88125`; docs score `0.5416666666666666`; stages `accepted/accepted`.

## Non-mutation reads

GraphClient census before and after the release:

| Read | Total | Accepted | Approved | Contested |
|---|---:|---:|---:|---:|
| Before | 4,683 | 2,336 | 0 | 0 |
| After | 4,683 | 2,336 | 0 | 0 |

The read targeted `StandardName.name_stage`; the positive accepted count of 2,336 proves the stage predicate was aimed at a populated property while the approved and contested counts remained zero.

The upstream repository `iterorganization/imas-standard-names-catalog` reports default branch `main`. Its default-branch SHA was `a06e52052d4776b25e94fdfaa22c2bc6651a98eb` before the cut and remained `a06e52052d4776b25e94fdfaa22c2bc6651a98eb` after the cut (commit timestamp `2026-09-01T09:25:05Z`). Only the fork received the review branch and RC tag.
