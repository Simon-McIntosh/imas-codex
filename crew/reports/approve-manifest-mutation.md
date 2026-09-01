# Approval manifest mutation verdict

Verdict: remove the approval-time write. The review artifact is the cut-time,
frozen cohort record used to recover the submitted names and candidate identity.
The merge commit is a later fold-back receipt, already recorded by the contract
tag on the catalog merge commit and by per-name graph provenance. Rewriting the
artifact after merge makes a successful approval dirty the source worktree and
changes the evidence used to identify what was reviewed.

The former post-approval `backfill_review_artifact` call has been removed. After
the approval summary and refusal check, the CLI now proceeds directly to the
fold-back tag receipt (`imas_codex/cli/sn.py:5242`). The resolved reviewer actor
enters the approval call with the PR number, URL, and merge commit
(`imas_codex/cli/sn.py:5106`, `imas_codex/cli/sn.py:5191`).

The contest transition writes all four fields atomically with
`name_stage='contested'` (`imas_codex/standard_names/promote.py:799`). The
untouched-entry promotion receives the same actor (`promote.py:1166`), and undo
clears the four fields when returning a contested entry to `accepted`
(`promote.py:1324`).

Verification used mocked graph clients only. The focused contest test records
one reviewer-edited entry, forces a score of 0.5 below the 0.85 threshold, and
asserts the four stored provenance values exactly match the merged PR inputs.
The approval, tag, and reviewer-base suite passes with 54 tests, including the
focused provenance and byte-exact artifact cases. No live graph command was run
by this node.
