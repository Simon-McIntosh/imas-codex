# Reader calibration on an enlarged, hand-checked set

## Why
Plan `sn-audited-group-release` section 3 rules a reader fit to decide group audits when, on hand-checked planted names, it finds at least 90 % of the defective and flags at most 10 % of the sound, with each rate's 95 % interval no wider than ten points either side. The hand check of the first 40 plants (`planted-label-check.json`, at your base) overturned 12 labels and left 15 standing defective and 11 standing sound. On those, the three readers flag almost no sound name (0, 0 and 1 of 11) and find 9, 11 and 13 of 15 defective, so the frontier reader is close to the threshold by point, and no interval is narrow enough to say. The same check found the audit's instructions contradict themselves on omitted qualifiers: one rule defines the class as a source-stated qualifier absent from the name, a later rule says to report a missing qualifier only when neither the source leaf nor the enclosing structure states it. That can suppress exactly the largest defect class.

## Deliverable
Under `docs/evidence/fragments/sn-audited-group-release/`: `reader-calibration.py`, `reader-calibration.json`, `reader-calibration.html`, and the one correction to the instructions in `blind-audit.py`.

1. Reconcile the omission instruction in `blind-audit.py` so the two rules say one thing: a qualifier the source leaf or its enclosing structure states, and the name lacks, is the defect. Show the before and after text on the page. Keep the earlier readings untouched; they stay labelled as taken under the earlier instructions.
2. Enlarge the hand-checked set until at least 40 labels stand defective and at least 40 stand sound, the first check's standing labels included. Draw the new names from the audit's 339 sampled names by a seeded draw stratified by the readers' earlier verdicts, so the set is not chosen by any one reader. Judge each new name blind, from its versioned source packet (stored name, definition, documentation, unit, domain, source paths, source and enclosing-structure text, kept in the data file), before opening any reader's verdict, and commit the judgments before the comparison, as the first check did. A name you cannot decide from the text is recorded undecidable and does not count toward the 40.
3. Read every name of the enlarged set with all three readers under the corrected instructions, blind to the labels. Spend ceiling USD 10 in total, stated beside the recorded spend; run a pilot of six names first and stop if the projected cost exceeds the ceiling.
4. Report for each reader and for the two-of-three rule: recall and false-alarm rate with counts and 95 % intervals, under the corrected instructions, and beside them the same rates under the earlier instructions on the names read both ways; which reader, if any, is fit to decide; and per defect class the recall, with the omission class shown before and after the correction.
5. State the corrected catalog estimate the fit reader (or, where none is fit, the best one) implies for the 339 sampled names, with the correction for its measured recall and false-alarm rate written out, and its interval.

## Rules
- No stored text or graph property changes; graph reads bounded to the named set. Reader calls go through the audit script's existing call path.
- Heavy work runs as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...` with logs on GPFS.
- The gate is the script's `--check`; the negative control is that it fails when a stored judgment is altered after the blind commit.
- HTML tables for tabular results; one figure only if it shows the rates with their intervals more clearly than the table.
- Write the manifest before any long wait. Commit locally with a subject and a body; do not push.
