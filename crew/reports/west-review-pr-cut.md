NEEDS-HELP: fork PR 6 is cut and previewable, but catalog CI rejects the exported `scalar.roles` field and the repair is outside this node's report-only write scope

tried: Ran the authorized fork-only `sn release --batch west_production_dd_paths` cut once from the clean live graph and blank fork main. The command exited 0, created the RC tag and PR, and built the preview; both catalog validation jobs then failed with `scalar.roles Extra inputs are not permitted`, while the build and review-edit guard passed.

options: (1) extend scope to repair the imas-codex exporter/catalog schema compatibility, amend the review branch through a fresh publisher commit, and let CI rerun; (2) assign a focused repair node owning the exporter and its tests, then refresh PR 6 from its result; (3) hold PR 6 open as the exact failing artifact while the catalog schema deliberately regains support for machine-owned roles.

leaning: Option 2, because the current PR and immutable cut-time tag preserve the complete failure witness while a separately scoped repair can own the exporter and compatibility tests without mixing code repair into this operational report node.

cost-if-wrong: Choosing the wrong authority side requires redoing the exporter/schema repair and refreshing the PR branch; if the review branch head changes, the existing cut-time RC tag must also be deliberately reconciled rather than silently left pointing at the invalid head.

# WEST review pull-request cut

## Outcome

The release command completed with exit status 0 and opened [fork PR 6](https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/6). It froze 368 names from 355 manifest sources, published 295 catalog entries, and withheld 73 through export policy. The PR is a pure addition of 18 files with 17,347 inserted lines, zero modified files, and zero deleted files. The preview built and returns HTTP 200, but the candidate is not validation-clean: both validation jobs fail on the exported machine-owned `roles` field.

The graph did not change: before and after the cut it contained 4,683 StandardName nodes, including 2,336 accepted, 0 approved, and 0 contested. The upstream default branch remained at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.

## Release status before the cut

The authoritative successful status read is captured at:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T063017239136-n-west-review-pr-cut/logs/release-status-with-isnc.log`

It reported:

- catalog checkout: `/home/ITER/mcintos/Code/imas-standard-names-catalog`
- state: `rc`
- latest tag: `v0.3.0rc1+west-task-2e`
- commits since that tag: 19
- available continuation: omit `--bump` to cut the next RC
- origin: `git@github.com:Simon-McIntosh/imas-standard-names-catalog.git`
- upstream: `git@github.com:iterorganization/imas-standard-names-catalog.git`
- GitHub Pages: enabled

An initial status attempt without `IMAS_CODEX_SN_ISNC` exited 2 before any mutation; the corrected read above bound the detached worktree to the real catalog checkout. The live `sn release --help` was also read immediately before the operator effect.

## Exact release command

The command was run from this worktree with `IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog`, `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, and `PYTHONPATH=$PWD`:

```text
uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target fork -m "WEST production standard names review batch" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T063017239136-n-west-review-pr-cut/pr-body.md
```

Exit status: 0. Full terminal transcript:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T063017239136-n-west-review-pr-cut/logs/release-cut.log`

The release reported a 368-name frozen batch, 28 unmatched sources, and the PR URL. Publisher validation also reported that two grammar-parse failures were excluded and that the final catalog contained 295 entries. The catalog commit body independently records `Published 295 entries ... and withheld 73`.

## Frozen review artifact

- path: `imas_codex/standard_names/manifests/reviews/v0.3.0rc2+west-task-2e.sn_names.yaml`
- entries in `names`: 368
- manifest sources: 355
- unmatched sources: 28
- PR provenance: number 6, `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/6`
- SHA-256: `17bd3d509f031341c288befd3d754126001e4d30a1698718144be8db69850e35`

The release CLI generated and back-filled this operational artifact in the worktree. It is intentionally not staged in this report-only node because the exclusive authored-file scope contains only this report.

## RC tag and object

- tag: `v0.3.0rc2+west-task-2e`
- annotated tag object: `b77359486ce00dc5fda05d244194b89dd97632ae`
- target commit: `a43aaebb4415ebd892591c7f2db858ee040deb2a`
- annotation: `WEST production standard names review batch`
- tagger: Simon McIntosh `<simon.mcintosh@iter.org>` at `2026-09-02T06:45:20Z`

The GitHub API read of `refs/tags/v0.3.0rc2+west-task-2e` returned the annotated object above, and dereferencing it returned the PR head commit.

## Published pull request

- URL: `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/6`
- number: 6
- base: fork `main` at `1a752add6809049e522e3ff4db4c8d0c7e07795e`
- head: `review/v0.3.0rc2+west-task-2e` at `a43aaebb4415ebd892591c7f2db858ee040deb2a`
- title: `WEST production data dictionary standard names review`

The final published body is four decisions-only sentences:

> This WEST standard names review batch publishes 295 entries selected from the frozen 368-name candidate minted from the `west_production_dd_paths` source manifest.
> Coordinate grids and ordinates remain excluded because they are structural indexing rather than physical quantities.
> Review every entry under the catalog [REVIEWING.md](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/main/REVIEWING.md) contract, prioritizing physics meaning before machine-enforced details.
> Inspect the rendered [PR-scoped preview](https://simon-mcintosh.github.io/imas-standard-names-catalog/pr-6/) before approving.

The pre-submit attribution scan was clean and `validate_pr_text` passed. The initial body used the 368 frozen-candidate count as the published count; after the authoritative catalog commit and YAML census proved 295 published entries, the PR body was corrected through `gh pr edit --body-file` and read back from the GitHub API.

## Pure-addition diff

GitHub reports 18 changed files, 17,347 additions, and zero deleted lines. The paginated PR-file API classified all 18 files as `added`:

- added: 18
- modified: 0
- removed/deleted: 0

The additions are `catalog.yml` plus 17 `standard_names/*.yml` domain files. A YAML census of the PR head found 295 entries and 295 unique names. Fork main was blank at the PR base, so there is no modified or deleted catalog entry.

## CI and preview

Final check conclusions on head `a43aaebb4415ebd892591c7f2db858ee040deb2a`:

- `build`: success, run 33600291669; this run published the PR-scoped preview
- `build`: success, run 33600288080
- `review-edit-guard`: success, run 33600291671
- `validate`: failure, run 33600291671, job 100152266180
- `validate / validate`: failure, run 33600288259, job 100152255241
- earlier matrix legs associated with run 33600288259: build skipped and review-edit-guard skipped

Preview: `https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-6/`. The workflow marker comment links this URL and run 33600291669; an independent request returned HTTP 200.

Both failed validation logs terminate on the same schema incompatibility:

```text
scalar.roles
  Extra inputs are not permitted [type=extra_forbidden, input_value=['quantity'], input_type=list]
```

Failure logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T063017239136-n-west-review-pr-cut/logs/validate-failed-33600291671.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T063017239136-n-west-review-pr-cut/logs/validate-failed-33600288259.log`

The failure is reproducible catalog-schema evidence, not a transient check: the release transcript already contains `scalar.roles Extra inputs are not permitted` during the publisher's staging validation. The command nevertheless pushed the branch and returned success.

## Isolation evidence

Read-only GraphClient censuses before and after the release returned the same row:

```text
total=4683 accepted=2336 approved=0 contested=0
```

The upstream default-branch SHA was read before and after the fork-only cut and remained:

```text
a06e52052d4776b25e94fdfaa22c2bc6651a98eb
```

No graph mutation or upstream write occurred.
