# Land the retirement of a name whose source is recycled

## Why
Plan `sn-pipeline-hygiene` section 11 (read it in your worktree; it is the authority for the endpoint). An earlier node measured the defect, validated a fix as a patch and committed the evidence, but its write paths did not include the file that holds the pass. This node starts from that node's head (27ec3208c) and lands the fix.

## Inputs, all readable under `~/.cache/imas-codex-live-rewrite/retire-recycled-source/`
- `graph_audits-retire-recycled-source.patch`: the validated change to `imas_codex/standard_names/graph_store/graph_audits.py`.
- `final_test_recycled_source_retires_name.py`: the test file, to be placed at `tests/standard_names/test_recycled_source_retires_name.py`.
- `tests-result-key-update.patch`: two expectations in `tests/standard_names/test_pipeline_recovery_mechanics.py`.
- `backlog_classification.json`, `backlog_sources.json`: the 13 live names the pass already left without a source, with their former sources.
- `manifest.md`, `negative_control_head_reproduction.log`: the earlier node's record.

## What the section now says, which the earlier patch predates
The pass takes the source of a **live** name only when that name is exhausted, and retires that name in the same write. A name already superseded is not live, and its source returns to the pool as it does today: `tests/standard_names/test_source_revival.py` asserts that and must stay green. A recovery that returns a source on purpose and reopens its name for rewriting keeps its behaviour. Check the patch against these three statements and adjust it where it differs. The log excerpt that reports the 24 sources returned is committed with the evidence.

## The 13 names already left without a source
The lead ruled: exhausted names are retired and their sources composed afresh. Retired means superseded, never deleted: recorded model spend is not removed automatically. Retire them through the pipeline's own signed write path. If that path refuses them because the pass erased the provenance it reads, make the smallest change that lets it read the former source recorded in the change rows or the evidence, inside your write paths; if that needs a file outside them, stop there, list all 13 with their former sources and name the file. Any of them whose stage is accepted is listed by name and left untouched. Afterwards the live names without a source, by the repository's own enumeration, are the ten that had none before 2026-11-17.

## Rules
- You are the only live-graph writer of standard names while you run. Graph reads and writes are bounded to the named cohort.
- Heavy gates run as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...` with logs on GPFS: `tests/standard_names` at head against base, added failures zero.
- Negative control: the new test fails with the retirement removed, and another fails when the pass may take the source of a live name that is not exhausted.
- Write the manifest before any long wait. Commit locally with a subject and a body; do not push.
