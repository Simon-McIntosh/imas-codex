# WEST physics-domain review candidate

## Outcome

The fork-only WEST review candidate was re-cut with `imas-standard-names==0.8.1`, which makes `physics_domain` a required catalog-entry field. Pull request 7 was closed with a successor comment, and pull request 8 now carries a validated 291-entry review catalog selected from the same frozen 368-name cohort. Its preview data has 16 populated physics-domain categories and zero uncategorized entries.

The cut did not mutate the graph: its before and after census was 4,683 total StandardName nodes, 4,683 with `name_stage`, 2,336 accepted, 0 approved, and 0 contested. The upstream catalog default branch remained at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.

## Superseded pull request

- Pull request: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/7>
- Final state: `CLOSED`
- Successor comment: “Superseded by the physics-domain-complete WEST production data dictionary review in PR 8.”
- Comment receipt: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/7#issuecomment-5509568462>
- Comment posted: `2026-09-02T12:32:03Z`
- Closed at: `2026-09-02T12:32:09Z`

Repository issue 7 was the highest issue number immediately before the comment and closure, proving that the next pull request would be number 8.

## Release prerequisites and status

The worktree pinned `imas-standard-names==0.8.1` at both dependency sites in `pyproject.toml`, `uv.lock` resolved the same version, and the shared environment imported version `0.8.1`. The catalog tooling also pinned `imas-standard-names[quality,docs]==0.8.1`. Before the cut, the catalog checkout was clean on `main` at `67180892e32d7da9824e08072eb41c2ef439d4b7`.

The pre-cut command was:

```text
uv run --no-sync imas-codex sn release status
```

It exited 0 and reported:

```text
ISNC Release Status
  Path: /home/ITER/mcintos/Code/imas-standard-names-catalog
  State: rc
  Latest tag: v0.3.0rc3+west-task-2e
  Batch RC: +west-task-2e (cut against a review batch)
  Commits since: 8
  Remote (origin): git@github.com:Simon-McIntosh/imas-standard-names-catalog.git
  Remote (upstream): git@github.com:iterorganization/imas-standard-names-catalog.git
  GitHub Pages: yes

Available commands:
  sn release -m 'Next RC' (next RC of v0.3.0rc3+west-task-2e)
  sn release --final -m 'Finalize' (stable)
```

Transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/release-status.log`.

All four live-help citations (`sn run`, `sn release`, `sn approve`, and `sn resolve`) exited 0 before the release.

## Dry-run and live cut

The exact live command, with `IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog`, `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, and this worktree first on `PYTHONPATH`, was:

```text
uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto -m "WEST production standard names review batch" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/pr-body.md
```

The preceding dry-run used the same arguments plus `--dry-run`. Both invocations exited `0`, selected the fork, minted the same `v0.3.0rc4+west-task-2e` identity, froze 368 names, and recorded 28 unmatched sources. The live invocation pushed the review branch and annotated tag to `origin`, opened pull request 8, and returned the catalog checkout to clean `main` at its original SHA.

ISN 0.8.1 rejected four otherwise exportable names whose graph domain is the non-enum value `unscoped`: `intensity_at_spectral_line`, `ipb98y2_confinement_time`, `radiative_temperature_at_ece_channel`, and `voltage_of_ion_cyclotron_heating_antenna_amplitude`. Together with the previously dispositioned exclusions, this changed the published cardinality from 295 to 291 and the withheld cardinality from 73 to 77. The PR body was corrected to 291 before any public release action.

The RC-mode gate advisories were graph-wide test failures, 331 known-but-unpublished link targets, 183 additive-baseline divergence rows, two grammar-parse exclusions, and 328 pruned links in the exported copy. They remained advisory for this fork RC. Full transcripts:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/release-dry-run.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/release-live.log`

## Tag and pull request

- Annotated tag: `v0.3.0rc4+west-task-2e`
- Tag object SHA: `0b8d029133e4c926d42fb80d0e33e4837f94b8b0`
- Tag target commit: `227f59273c83cf1b1344b44f68afdb436fae7911`
- Tag message: `WEST production standard names review batch`
- Pull request: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/8>
- State: `OPEN`
- Base: `main` at `67180892e32d7da9824e08072eb41c2ef439d4b7`
- Head: `review/v0.3.0rc4+west-task-2e` at `227f59273c83cf1b1344b44f68afdb436fae7911`
- Title: `WEST production data dictionary standard names review`

Published body:

> This WEST production data dictionary review publishes 291 entries from the frozen 368-name batch with DD-authoritative physics domains.
>
> Coordinate grids and ordinates remain excluded because they are structural indexing rather than physical quantities.
>
> Review every entry under the catalog [REVIEWING.md](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/main/REVIEWING.md) contract, using the [PR-scoped preview](https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-8/) to inspect domain groupings and entry details before approval.
>
> Preview: https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-8/

The first three sentences are the validated body-file bytes. The release command appended and read back the final derived preview receipt. GitHub reports 17 changed files, 16,531 additions, and 0 deletions. Its pull-files API reports all 17 statuses as `added`, with 0 modified and 0 deleted files.

## Catalog CI

All head checks reached terminal success:

| Check | Conclusion | Duration | Receipt |
|---|---:|---:|---|
| `validate` | success | 3m31s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631281347/job/100251058636> |
| `validate / validate` | success | 1m42s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631276824/job/100251042453> |
| `review-edit-guard` | success | 12s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631281347/job/100251058998> |
| `build` (push) | success | 46s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631276578/job/100251041220> |
| `build` (release) | success | 45s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631276824/job/100251599874> |
| `build` (pull request) | success | 41s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631281370/job/100251311439> |
| `validate / review-edit-guard` | skipped by matrix routing | 0s | <https://github.com/Simon-McIntosh/imas-standard-names-catalog/actions/runs/33631276824/job/100251043710> |

The final checks command summarized `0 cancelled, 0 failing, 6 successful, 1 skipped, and 0 pending`. Transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/pr-checks.log`.

## Preview data

- Preview: <https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-8/> — HTTP `200`
- Data: <https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-8/data.json> — HTTP `200`, `application/json`, 1,680,468 bytes at observation
- `NAMES`: `291`
- `CATEGORIES`: `16`
- Uncategorized or `unscoped` preview entries: `0`

The populated category census was: equilibrium 79; transport 49; radiation measurement diagnostics 36; divertor physics 22; auxiliary heating 20; magnetic field diagnostics 16; electromagnetic wave diagnostics 14; edge plasma physics 12; magnetic field systems 8; particle measurement diagnostics 7; plant systems 7; plasma-wall interactions 7; general 4; magnetohydrodynamics 4; mechanical measurement diagnostics 3; structural components 3. The counts sum to 291.

The instrument targeted each preview entry's `category` field—the field used by the site data—not an assumed raw-export property. Transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/preview-data.log`.

## Frozen artifact and representative records

- Artifact: `imas_codex/standard_names/manifests/reviews/v0.3.0rc4+west-task-2e.sn_names.yaml`
- Frozen names: `368`
- Unmatched sources: `28`
- Recorded PR: number `8`, URL <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/8>
- SHA-256: `ec8c04985e74da0aac2a157c2ea95f604e8baf4cf736df27f15221c7cb350a54`
- Commit state: staged and committed with this evidence report in the node's delivery commit.

Representative graph records stayed accepted while supplying the required domain:

- `breakdown_initial_time`: physics domain `machine_operations`; description “Timestamp at which plasma breakdown begins and discharge current starts to flow.”; source `dd:summary/time_breakdown/value`; name score `0.925`; docs score `0.7166666666666667`; stages `accepted/accepted`.
- `pulse_duration`: physics domain `machine_operations`; description “Elapsed duration of the confined-plasma phase in a single discharge, from plasma breakdown until termination of the confined plasma.”; name score `0.88125`; docs score `0.5416666666666666`; stages `accepted/accepted`.

## Non-mutation reads

| Read | Total | With `name_stage` | Accepted | Approved | Contested |
|---|---:|---:|---:|---:|---:|
| Before | 4,683 | 4,683 | 2,336 | 0 | 0 |
| After | 4,683 | 4,683 | 2,336 | 0 | 0 |

The nonzero `with_name_stage` and accepted counts are positive controls for the predicates behind the zero approved and contested counts. Final graph transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T121608010702-n-west-review-pr8-cut/logs/graph-after.log`.

The upstream repository `iterorganization/imas-standard-names-catalog` reports default branch `main`. Its default-branch SHA was `a06e52052d4776b25e94fdfaa22c2bc6651a98eb` before the cut and remained the same afterward; that commit is timestamped `2026-09-01T09:25:05Z`. Only the fork received the review branch and RC tag.
