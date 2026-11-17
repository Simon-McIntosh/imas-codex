# Planted labels checked by hand against their sources

## Why
The blind audit planted 40 names among 339 sampled ones: 20 labelled known-defective and 20 labelled known-sound. Three readers were scored on them and none met the fitness threshold (find at least 90 % of the defective, flag at most 10 % of the sound). The names planted as sound were drawn from the catalog, which the audit itself reads as about half defective, so a reader's "false alarm" may be a real defect, and a planted defect may be one no reader could see from what it was shown. The lead ruled on 2026-11-17 that the labels are checked by hand before any reader is ruled fit.

## Inputs
- `docs/evidence/fragments/sn-audited-group-release/blind-audit.py`, `blind-audit.json`, `frontier-third-reader.json` and the two evidence pages beside them: the plants, how each was made, and each reader's verdict and classes.
- The live graph, read only: each planted name's stored name, definition, documentation, unit, domain, its sources and their enclosing-structure statements, exactly the text the readers were shown.

## Deliverable
One evidence page and one data file under `docs/evidence/fragments/sn-audited-group-release/` (`planted-label-check.json` and `planted-label-check.html`), and the script that produces them (`planted-label-check.py`).

1. Judge every planted name yourself, first, without looking at any reader's verdict or the planted label: write your reading (defective or sound, the class, one sentence of reason citing the source text) to the data file before you open the readers' verdicts. Record the order of work in the file so the blindness can be checked: your 40 judgments are committed in one commit that touches no reader data, before the comparison commit.
2. Then compare. For each plant report: the planted label, your judgment, each of the three readers' verdicts, and whether the label stands, is overturned, or cannot be decided from the text shown. An overturned label needs the source sentence that overturns it.
3. Re-score the three readers on the labels that stand: recall on defective, false-alarm rate on sound, each with its count and a 95 % interval, and the same for the rule "at least two of three readers flag it". State which, if any, meets 90 % and 10 %.
4. State plainly what the result means for deciding group audits, including if the answer is that 40 plants are too few to tell.

## Rules
- No model calls for the judgments: you read the text. No stored text or graph property changes. Graph reads are bounded to the 40 names and their sources.
- Do not edit the audit's existing files except to import from the script.
- Heavy work, if any, runs as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...`; the gate is the script's own `--check`, which must fail when a label verdict is altered in the data file (record that negative control).
- Tables as HTML tables; a figure only if it shows something a table cannot.
