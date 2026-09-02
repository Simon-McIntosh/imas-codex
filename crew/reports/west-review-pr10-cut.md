# WEST catalog review candidate cut

## Outcome

Fork PR [10](https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/10) is the validated WEST production Data Dictionary review candidate. It publishes 338 entries from a schema-loadable frozen artifact containing 411 unique names. The branch diff is a pure addition: 19 added files, 12,959 added lines, zero modified files, and zero deleted lines. The live graph remained at 0 approved and 0 contested names, and upstream `main` remained at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.

## Superseded candidate

PR 9 was closed at `2026-09-02T16:32:11Z`. Its single successor comment is: “Superseded by PR 10, which re-cuts this WEST review candidate with schema-loadable frozen accounting and the standard-names 0.8.2 YAML writer.” The comment receipt is [issuecomment-5512901900](https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/9#issuecomment-5512901900).

## Release state and commands

Immediately before the cut, `imas-codex sn release status` exited 0 and reported:

```text
State: rc
Latest tag: v0.3.0rc5+west-task-2e
Batch RC: +west-task-2e
Next RC: rerun sn release without a bump argument
origin: Simon-McIntosh/imas-standard-names-catalog
```

The four mandatory current-checkout citations (`sn run --help`, `sn release --help`, `sn approve --help`, and `sn resolve --help`) each exited 0 before the external effect.

Dry-run command, exit 0:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto --dry-run -m "Cut WEST production data dictionary review candidate" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/pr-body.md
```

The dry-run minted 412 names with 27 unmatched sources and passed the closed exclusion-accounting gate. Its export report recorded 338 emitted and 74 withheld: 45 structural parents, 14 name-stage holds, one name-review quorum hold, four documentation holds, seven invalid-validation holds, one invalid catalog-domain entry, and two grammar-parse failures. The source reconciliation was exact: 355/355 category-eligible paths accounted as 300 emitted, 43 excluded, and 12 documented non-nameable. Advisory gates recorded 358 links to known but unpublished targets and 195 additive-baseline divergences; neither is a hard RC failure.

Live command, exit 0:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto -m "Cut WEST production data dictionary review candidate" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/pr-body.md
```

One concurrently recovered name changed eligibility between the dry-run and live cut. The live command therefore froze 411 names with 28 unmatched sources while still publishing 338. The live withheld count is 73: 45 structural parents, 13 name-stage holds, one name-review quorum hold, four documentation holds, seven invalid-validation holds, one invalid catalog-domain entry, and two grammar-parse failures. The PR body was corrected immediately to the live, additive accounting without changing catalog content or graph state.

## Published identity and review text

- RC tag: `v0.3.0rc6+west-task-2e`
- Annotated tag object: `e5daf3131a2ffe6f595f52914168b4c1f0303526`
- Tag target / review head: `c8f207580f1a8bbf8e31484f6a5f1dd14d4ca8fe`
- Branch: `review/v0.3.0rc6+west-task-2e`
- PR: `https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/10`
- Title: `WEST production data dictionary standard names review`

Published body:

```text
This WEST production data dictionary review publishes 338 entries from the frozen 411-name batch.
Of 469 spreadsheet paths, 114 are excluded by DD node category: 55 metadata, 29 structural, 26 temporal coordinates, and 4 fit or representation artifacts; the exclusion ledger is committed beside the manifest.
The 73 withheld frozen names—45 structural parents, 18 names or documentation not yet accepted, 8 quarantined or catalog-invalid names, 0 resolution-unrecorded names, and 2 grammar-parse failures—will follow in a later additive batch.
Review every entry under the catalog REVIEWING.md contract.

Preview: https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-10/
```

The 18 combined name/documentation holds are 13 name-stage holds + one name-review-quorum hold + four documentation holds; the eight quarantined/catalog-invalid names are seven invalid-validation rows + one invalid `unscoped` catalog-domain row. These groups plus 45 structural and two grammar-parse rows sum to the 73 withheld identities.

## Writer, schema, preview, and CI evidence

The frozen artifact is `imas_codex/standard_names/manifests/reviews/v0.3.0rc6+west-task-2e.sn_names.yaml`. `load_names_file` loaded 411 names and confirmed 411 unique identities; SHA-256 is `ff7ea2f6fbcee469db0d69bed0b4612d67185886fa52184a16f23d322f2fcee9`. This directly exercises the repaired schema path, including the frozen source-accounting rows.

The review branch’s `standard_names/equilibrium.yml` contains 88 entries. Byte-level checks found 88 `description: |-` blocks, 88 `documentation: |-` blocks, 87/87 top-level entry boundaries containing exactly one blank line, and zero boundaries with an extra blank line. All 87 documents containing display equations have exactly one blank line before and after the display. Representative branch bytes:

```yaml
  documentation: |-
    This quantity is the local geometric derivative of the area enclosed by a closed magnetic flux surface’s poloidal cross-section with respect to the surface’s signed poloidal magnetic-flux label.

    The defining relation is

    $$
    \frac{dA}{d\psi}
    $$

    where $A$ is the enclosed poloidal cross-sectional area and $\psi$ is the signed poloidal magnetic flux labeling the nested surface.
```

The preview root and `data.json` both returned HTTP 200. `CATEGORIES` contains 18 populated physics domains totaling 338 entries; there is no `Uncategorized` category and therefore zero uncategorized entries.

All check runs on head `c8f20758` are terminal. Successful checks are: three `build` runs, `validate`, `validate / validate`, and `review-edit-guard`. `validate / review-edit-guard` is the expected skipped push-side duplicate; no check failed or remained pending. Full API evidence is in `check-runs.json`.

## Isolation evidence

The pure-addition API census is one status only: `added`, 19 files, 12,959 additions, zero deletions. The fork catalog checkout returned clean `main` at `7e998fa591878c8b50dfa17374c3f2377fc4e94d` after the cut. A GraphClient key-coverage census found 4,691 StandardName candidates, 4,691 with the schema `id` key, approved 0, and contested 0. Upstream default-branch SHA remained `a06e52052d4776b25e94fdfaa22c2bc6651a98eb` before and after publication.

## Durable evidence

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/dry-run-export-report.json` — dry-run exclusion and source ledger, SHA-256 `94d686f7c2c00ac17e2cef6cee98e96a34e7ff7d638d7549019fe3cb8126bbcd`.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/live-export-report.json` — live exclusion and source ledger, SHA-256 `637c2a5e6e7177224fbad2be16a96c8cdf48049ba9d272ebec03f6ca69d285ff`.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/evidence.json` — loader, YAML-format, equation, and graph census, SHA-256 `6e35d9ac22f105820391d9d3496e69125e03eda1cbf0685656084535e40a5fdc`.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/preview-data.json` — deployed PR-scoped catalog data, SHA-256 `c720eb39a552f5dc2aa0bd08f236da388fd8c75c8da79a166a33003b33e014a4`.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T161020764438-n-west-review-pr10-cut/check-runs.json` — terminal GitHub check API response, SHA-256 `a977be4f0e3ea24ee9968f9561da788e123a17ce02fad0490624ad1f4765259c`.
