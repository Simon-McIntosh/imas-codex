# Blind adjudication of every equilibrium name

## Why
Plan `sn-audited-group-release` sections 3 and 4 (read them in your worktree). No model reader is fit to decide a group audit, so the lead ruled on 2026-11-17 that a group is audited by blind adjudication of every member from its full source packet, with no reader in the loop, and that the lead reads a random twenty of the verdicts before the group proceeds. Equilibrium is the first group.

## Inputs
- The live graph, read only: every live name whose physics domain is equilibrium (about 210), with its stored name, definition, documentation, unit, domain, source paths, and each source's own text and enclosing-structure statements.
- `reader-calibration.py`, `planted-label-check.py` and `blind-audit.py` under `docs/evidence/fragments/sn-audited-group-release/` on main: the packet builder, the blindness assertion, the seven defect classes and the reconciled omission instruction. Reuse them; do not write a second packet builder.
- `domain-groups.json`: the 72 families that touch equilibrium, for the duplicate and wrong-domain classes.

## Deliverable
`equilibrium-adjudication.py`, `equilibrium-adjudication.json` and `equilibrium-adjudication.html` under that directory.

1. Freeze the member list and every member's versioned packet in the data file first, with a digest, and commit it.
2. Judge every member yourself from its packet alone: sound, defective with one or more of the seven classes, or undecidable from the text, with one sentence of reason quoting the source words it rests on. Do not open any earlier reading, review score or audit row for these names before your verdicts are committed; the script asserts that and the verdict commit touches no reader data. No model calls for the verdicts.
3. After the verdict commit, report: the count and share per verdict and per class with 95 % intervals; every cause shared by three or more defective members, named as a candidate pipeline defect with its members; the members whose only classes are duplicate or wrong domain, listed apart; and, as a comparison only, how the three readers' earlier verdicts agree with yours on the members they read.
4. Draw twenty members by a recorded seed, stratified so that sound and defective verdicts are both present in proportion, and render them as a self-contained review page for the lead: each with its packet and your verdict hidden behind a disclosure, so the lead can judge first and then compare. Leave a column for the lead's verdict; do not fill it.
5. State plainly what you could not decide and why.

## Rules
- No stored text or graph property changes. Graph reads bounded to the equilibrium members and their sources.
- Heavy work runs as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...` with logs on GPFS.
- The gate is the script's `--check`; the negative control is that it fails when a stored verdict is altered after the verdict commit.
- HTML tables for tabular results; no figure unless it shows something a table cannot.
- Write the manifest incrementally and before any long wait. Commit locally with a subject and a body; do not push.
