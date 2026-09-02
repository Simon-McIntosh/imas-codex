NEEDS-HELP: PR 9 was cut successfully, but its 0.8.2 YAML still uses quoted documentation scalars instead of the required literal blocks.

tried: Ran the complete live-plan preflight, all four CLI help gates, `sn release status`, a full dry-run, closed PR 8 with its successor comment, and ran the live fork cut. The dry-run and live cut both exited 0 and resolved `v0.3.0rc5+west-task-2e`; the live cut opened fork PR 9. A byte read of `standard_names/equilibrium.yml` at the PR head found `0` lines matching `documentation: |` across 85 entries; the first entry instead begins `documentation: 'This quantity ...'`.

options: (1) repair the codex/ISN serialization boundary so review YAML calls the 0.8.2 literal-block writer, close PR 9 with a successor comment, and cut the next RC; (2) confirm that imas-standard-names 0.8.2 itself does not provide the expected writer bytes and release a corrected ISN version before recutting; (3) explicitly relax the evidence contract and retain quoted scalars, which would require a new plan decision.

leaning: Option 1 after a focused ownership trace. The active environment and both codex dependency pins report imas-standard-names 0.8.2, yet the emitted bytes are plain `yaml.safe_dump`-style quoted scalars, so the likely defect is that the exporter does not route final catalog mappings through the pinned ISN writer.

cost-if-wrong: If the writer is actually defective inside ISN, a codex-side routing repair will still emit the wrong bytes and PR 9 must be replaced again after another ISN package release; if the contract is relaxed instead, the physicist receives the hard-to-review scalar format this cut was explicitly meant to replace.

# WEST review candidate cut evidence

## Preflight and baselines

- Worktree revision: `321cddc9a9afb4808d3a349bc0402d522032539e`.
- Both dependency sites in `pyproject.toml` and the resolved `uv.lock` pin `imas-standard-names==0.8.2`.
- Catalog checkout: clean `main` at `7e998fa591878c8b50dfa17374c3f2377fc4e94d` before the cut.
- Upstream catalog `main` before and after the cut: `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`; upstream was untouched.
- GraphClient census before the cut: 4,690 `StandardName` nodes, `approved=0`, `contested=0`.
- `sn release status` exited 0: state `rc`, latest tag `v0.3.0rc4+west-task-2e`, origin is the Simon-McIntosh fork, and the next permitted action is another RC.
- Help preflight exited 0 for `sn run --help`, `sn release --help`, `sn approve --help`, and `sn resolve --help`. Logs: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T151541885696-n-west-review-pr9-cut/logs/help-run.log`, `help-release.log`, `help-approve.log`, `help-resolve.log`, and `release-status.log` in the same directory.

## PR 8 closure

- PR 8 was open at capture with head `227f59273c83cf1b1344b44f68afdb436fae7911`.
- Comment published before closure: “Closing this candidate in favor of PR 9, which regenerates the WEST review catalog with imas-standard-names 0.8.2.”
- Comment receipt: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/8#issuecomment-5512199254> at `2026-09-02T15:40:23Z`; PR 8 then closed successfully.

## Dry-run and source accounting

Exact dry-run command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto --dry-run -m "WEST production data dictionary standard names" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T151541885696-n-west-review-pr9-cut/pr-body.md
```

Exit status: `0`. It selected `v0.3.0rc5+west-task-2e`, froze 412 names, reported 27 unmatched sources, and produced the frozen artifact at `imas_codex/standard_names/manifests/reviews/v0.3.0rc5+west-task-2e.sn_names.yaml`. The staged catalog recorded 332 published entries across 18 domains, so 80 frozen names were withheld.

The read-only identity census partitions those 80 as follows:

- 45 structural parents.
- 14 names not accepted, 1 name-review quorum shortfall, and 6 documentation-not-accepted rows: 21 acceptance holds total.
- 8 invalid-validation-status rows, described publicly as quarantined.
- 3 resolution-unrecorded rows.
- 2 grammar-parse failures.
- 1 otherwise eligible row, `radial_coordinate_of_annulus_center`, rejected by ISN because `physics_domain='unscoped'` is not a valid `PhysicsDomain`; described publicly as one catalog-invalid domain.

Spreadsheet-source accounting in the published body follows the plan exactly: of 469 paths, 114 are DD-category exclusions—55 metadata, 29 structural, 26 temporal coordinates, and 4 fit or representation artifacts—with the ledger committed beside the source manifest.

## Live cut

Exact live command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog uv run --no-sync imas-codex sn release --batch west_production_dd_paths --target auto -m "WEST production data dictionary standard names" --pr-title "WEST production data dictionary standard names review" --pr-body-file /home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T151541885696-n-west-review-pr9-cut/pr-body.md
```

Exit status: `0`.

- RC tag: `v0.3.0rc5+west-task-2e`.
- Annotated tag object: `3e2525e379683ffadfba8f630258bcdf82c472fa`.
- Tag target / review head: `55511481cdadd8b32351486bd28b33e7001687fe`.
- Tag message: `WEST production data dictionary standard names`.
- Branch: `review/v0.3.0rc5+west-task-2e` on fork origin.
- PR: <https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/9>.
- Title: `WEST production data dictionary standard names review`.

Published body:

> This WEST production data dictionary review publishes 332 entries from the frozen 412-name batch.
> Of 469 spreadsheet paths, 114 are excluded by DD node category: 55 metadata, 29 structural, 26 temporal coordinates, and 4 fit or representation artifacts; the exclusion ledger is committed beside the manifest.
> The 80 withheld frozen names—45 structural parents, 21 name or documentation acceptance holds, 8 quarantined names plus 1 catalog-invalid domain, 3 resolution-unrecorded names, and 2 grammar-parse failures—will follow in a later additive batch.
> Review every entry under the catalog [REVIEWING.md](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/main/REVIEWING.md) contract.
>
> Preview: https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-9/

The PR diff is pure addition: 19 files, 18,536 additions, 0 deletions; every file status returned `added`. The files are `catalog.yml` plus 18 physics-domain YAML files.

## CI and preview at the blocker capture

- `review-edit-guard`: `SUCCESS`.
- `validate`: `IN_PROGRESS`.
- `build`: `IN_PROGRESS`.
- `validate / validate`: `IN_PROGRESS`.
- `validate / review-edit-guard`: `SKIPPED` by workflow design.
- Preview address returned HTTP 404 because the build had not completed. No `data.json` category result is claimed.

Further CI/preview polling stopped when the byte-level YAML contract failure was proven; all-success and zero-uncategorized evidence therefore remain absent rather than being inferred.

## Required YAML review format: failed

The inspected bytes are the exact PR-head file:

`https://raw.githubusercontent.com/Simon-McIntosh/imas-standard-names-catalog/55511481cdadd8b32351486bd28b33e7001687fe/standard_names/equilibrium.yml`

Quantitative read:

- 85 equilibrium entries.
- 0 `documentation: |` literal-block fields.
- 75 display equations; spacing around the equations is present, but inside a quoted YAML scalar.
- The first entry begins:

```yaml
- name: derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface
  kind: scalar
  status: active
  description: Rate of change of the poloidal cross-sectional area enclosed by a nested
    magnetic flux surface as the dimensionful toroidal-flux coordinate varies.
  documentation: 'This quantity is the radial rate of change of the poloidal cross-sectional
    area enclosed by a closed magnetic flux surface, parameterized by the dimensionful
    toroidal flux coordinate. It describes how neighboring flux-surface cross sections
    change along that coordinate.


    For a nested family of flux surfaces, it is defined by


    $$

    \frac{dA}{d\rho_{\mathrm{tor}}}

    $$
```

This directly contradicts the node’s done-when requirement for literal-block documentation, one blank line between entries, and single blank lines around display equations. The PR cannot be represented as the validated candidate requested by the node.

## Frozen artifact

The live cut back-filled PR 9 into `imas_codex/standard_names/manifests/reviews/v0.3.0rc5+west-task-2e.sn_names.yaml`. The file contains 412 unique names and 27 unmatched sources. It is preserved and committed with this blocked report so a repair node can audit the exact failed cut identity; it must not be treated as approval evidence.

A post-cut loader check exposed a second independent blocker: `load_focus_file` rejects the generated artifact because its top-level `manifest_sources` field is not declared by `sn_names.schema.json` (`Additional properties are not allowed`). This means the exact frozen artifact written by the live release cannot currently serve as the later `sn approve` currency promised by the workflow. The repair must either add and validate the emitted source-reconciliation field in the frozen-manifest schema or stop emitting it; that code and schema are outside this node's exclusive write scope.
