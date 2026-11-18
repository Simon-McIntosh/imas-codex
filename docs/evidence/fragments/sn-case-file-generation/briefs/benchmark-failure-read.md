# Why the record benchmark measured nothing: a hand read of its failures

## Why
Plan `sn-case-file-generation` section 3. The first benchmark run (head 4c8c3687e of run r-20261118T064807925154-scfg-benchmark; its worktree and run directory hold the generated records, prompts and judge verdicts) reported that both controls failed to separate, so it measures nothing. On the 87 held-out members: the incumbent scored 59 right; the best generating arm (one pass, frontier model) 28. Split by the reference: on the 64 members whose incumbent is sound that arm kept none unchanged, got 25 right, 30 wrong and 9 blocked; on the 23 whose incumbent is defective it repaired 3. Withholding the enclosing-structure text changed almost nothing (27 right against 28). The judge was a model, gemini-3.8-flash, the reader the calibration found unfit (73 % recall, 15 % false alarms); it judged 7 sound incumbents wrong and 2 defective ones right. The lead asked for the pipeline to be tuned and debugged by iteration, so the next round needs to know which of these is the cause.

## Inputs
- The benchmark's committed evidence and code under `docs/evidence/fragments/sn-case-file-generation/` at that head, and the run directory `~/.config/reckon/crew/runs/r-20261118T064807925154-scfg-benchmark/` for the generated records, the two prompt versions and the judge's verdicts with reasons.
- `reference-set.json` and `case-file.json` on main; `reuse-map.md` in this directory.

## Deliverable
`benchmark-failure-read.py`, `benchmark-failure-read.json`, `benchmark-failure-read.html` under that directory.

1. Draw by a recorded seed, from the best arm's held-out rows: 20 where the incumbent was sound and the generated record was judged wrong, all 20 where the incumbent was defective and the record was judged wrong, and 10 judged right. For each, read the case file, the generated record, the reference answer and the judge's reason yourself, and before reading the judge's verdict commit your own: is the generated record right, wrong, or a valid alternative the reference does not list.
2. Classify every disagreement between you and the judge, and every true failure, by cause, with counts: the judge wrong; the reference too narrow (a valid name or wording it does not allow); the generator renamed a sound incumbent for no stated reason; the generator lost a detail the case file states; the case file lacks what the right answer needs (say what); the grammar or vocabulary lacks the term; the source text cannot decide. Quote the words each classification rests on.
3. Answer four questions with the counts behind each: (a) does the prompt show the incumbent and ask the generator to keep it unless it is wrong, and does the output let it say "keep"; (b) is the judge's verdict usable, or must the judge be a blind adjudication as the plan requires; (c) is the reference set's set of right names too narrow to score a regenerated name fairly, and how should a record be scored instead (for example the name judged for identity and the definition judged against the reference's required points, separately); (d) why the withheld-text control did not separate: was the enclosing-structure text actually in the case files the generator read, and did the members whose answer rests on it differ.
4. State the three changes, in order of expected effect, that the next benchmark round should make, each tied to the counts above. Do not make them and do not re-run any arm.

## Rules
- No model calls: you read. No stored text or graph property changes; no graph access is needed.
- The gate is the script's `--check`; the negative control is that it fails when one of your committed verdicts is altered.
- HTML tables for tabular results. Write the manifest incrementally. Commit locally with a subject and a body; do not push.
