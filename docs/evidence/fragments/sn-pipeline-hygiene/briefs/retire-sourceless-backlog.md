# Retire the 13 exhausted names already left without a source

## Why
Plan `sn-pipeline-hygiene` section 11 (read it in your worktree). The pass that returned 24 sources to the pool on 2026-11-17 left 13 exhausted names live without a source, raising the count of live names without a source from 10 to 23. The pass itself is repaired and on main: it now retires an exhausted live name in the same write. The 13 it already stranded are still live. The lead ruled: exhausted names are retired and their sources composed afresh. Retired means superseded, never deleted, because recorded model spend is not removed automatically.

## Inputs
- `docs/evidence/fragments/sn-pipeline-hygiene/recycled-source-names.json` and `sph-retire-name-with-recycled-source.html` on main: the 13 names, their stage, their recorded former sources, and the signed path's refusal.
- `~/.cache/imas-codex-live-rewrite/retire-recycled-source/backlog_classification.json` and `backlog_sources.json`: the same cohort as first read.

## Deliverable
1. The signed retirement path in `imas_codex/standard_names/graph_store/graph_supersession.py` admits an exhausted name whose source edge was removed by the pass, reading the former source from the recorded change rows or the name's own recorded source ids, and refuses every other name it refuses today. A test fails when that admission is removed, and another fails when the path is allowed to admit a name that is not exhausted.
2. Two findings from the review of the pass change, folded here: the protected-stage list in `tests/standard_names/test_recycled_source_retires_name.py` names `refined`, which is not a stage; use `refining` and restore the refine-claim case. And the one production caller in `imas_codex/standard_names/loop.py` logs the pass's result without the count of names retired; log it.
3. On a disposable graph first, then on the live graph: retire the 13 through that signed path, one writer, with a dry run recorded before the apply. Any of the 13 whose stage has become accepted, or is no longer exhausted when you read it, is listed by name and left untouched. Two of the 13 record no former source; say for each what the path reads and whether it retires.
4. Afterwards, read back: the live names without a source, by the repository's own enumeration with no exemption added, are exactly the ten that had none before 2026-11-17, named in the evidence. If the count differs, report the names and stop; do not retire anything outside the 13.
5. Update the evidence page and data file with the retirement record and the read-back.

## Rules
- You are the only live-graph writer of standard names while you run. Never edit stored text or edges with ad-hoc Cypher: the signed path only.
- Heavy gates run as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...` with logs on GPFS: `tests/standard_names` at head against base, zero added failures.
- Write the manifest before any long wait. Commit locally with a subject and a body; do not push.
