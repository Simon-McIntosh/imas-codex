# Closing whole-tree pytest figure

The closing whole-tree run was executed exactly once from detached revision
`34531f99b0ebd07ff1e687ca226e1dccecd01b46`. The worktree's `.venv` symlink
resolved to `/home/ITER/mcintos/Code/imas-codex/.venv`, where the provisioned
test extra supplied `torch 2.10.0+cpu`.

## Run record

- Command: `uv run --no-sync pytest -p no:cacheprovider`
- Exit status: `1`
- Shell wall time: `612.677 s` (`10m12.677s`)
- Pytest-reported duration: `584.93 s` (`0:09:44`)
- Result: **12,070 passed / 3 failed / 142 skipped / 1 xfailed**
- Additional collection figure: `768 deselected`
- Full log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T204057382688-n-suite-final-figure/logs/pytest-final.log`

## Delta against the opening census

The opening census was 12,021 passed / 49 failed / 145 skipped / 1 xfailed.
Both runs account for 12,216 passed, failed, skipped, or xfailed outcomes.

| Outcome | Opening | Closing | Delta |
|---|---:|---:|---:|
| Passed | 12,021 | 12,070 | +49 |
| Failed | 49 | 3 | -46 |
| Skipped | 145 | 142 | -3 |
| Xfailed | 1 | 1 | 0 |

## Failing test IDs

1. `tests/graph/test_cypher_property_check.py::test_repository_cypher_literals_have_declared_properties`
2. `tests/standard_names/test_export_determinism.py::TestCommitTimestamp::test_archive_uses_packaged_sha_and_source_date_epoch`
3. `tests/standard_names/test_pipeline_version.py::test_pipeline_hash_changes_when_prompt_changes`

The first failure reports Cypher property references that are absent from the
corresponding LinkML label declarations. The archive timestamp test observed
`None` instead of the expected packaged SHA, and the pipeline-version test
observed an unchanged composite hash after modifying a prompt file. No retry
or second suite execution was performed.
