# The disposition record and the plan agree, on one triple and on the guard pointer

Measured and corrected 2026-09-18 in the `ship-s10` worktree at base
`2e15b38605e17db98617e67c39ec6537cb122f12`. Two documents state the same
dispositions and figures for the 78 identities deleted after origin
reconciliation: the census record at
`docs/evidence/sn-lifecycle-integrity/unshielded-identity-deletions.md` and the
plan `paid-identity-deletion-protection`. They disagreed in two places and one
line pointer had rotted past the correction the figure audit had already made.

## What disagreed

| quantity | census before | plan | resolution |
|---|---|---|---|
| disposition triple | `## Result` summary block read 32 RESTORE / 24 CORRECTLY REMOVED / 22 UNDETERMINED while its own tables and closing section read 67 / 11 / 0 | 67 RESTORE, 11 CORRECTLY REMOVED, 0 UNDETERMINED | the summary block is rewritten to the table-backed triple; the archive verdicts stay in the per-identity tables as provenance, marked superseded wherever the recorded-spend ruling differs |
| identities the archive left undetermined | stated as a live disposition column of 22 | stated as `0 UNDETERMINED` | the twenty-two archive-undetermined rows are now disposed by recorded spend: twenty carry spend and move to RESTORE, two zero-spend rows move to CORRECTLY REMOVED |
| the CORRECTLY REMOVED column | asserted at its archive size in two headings and one sentence | "taking that column from 26 to 11" | the census now names the corrected column (11) rather than the archive size |
| guard call site | `graph_ops.py:3608` / `:3633` (pre-guard tree) | `graph_ops.py:3672` (the figure audit re-measured `:3736`) | both corrected to the reading at HEAD: `:3846` and `:5956` |

Quantities both documents state that already agreed and were left untouched:
78 identities, 93 removed, 28 direct-DD targets from 30 surviving sources,
twenty spend-resolved rows, fifteen overridden CORRECTLY REMOVED verdicts.

## The guard pointer, read at HEAD

Read on 2026-09-18 at `2e15b38605e17db98617e67c39ec6537cb122f12`:

```
$ grep -n "refuse_protected_automatic_deletion(" imas_codex/standard_names/graph_ops.py
3846:        refuse_protected_automatic_deletion(
5956:                refuse_protected_automatic_deletion(
```

- `:3846` — structural derived-parent cleanup, after the candidate set is
  assembled and after the identity ceiling is applied.
- `:5956` — skeleton-placeholder sweep, after
  `_query_skeleton_placeholders_for_cleanup` has applied the positive
  placeholder predicate. Two call sites, not one, so an audit of "the" call
  site audits half the surface.
- The candidate selector buys protection earlier still, at `:3817` and `:4809`,
  through `filter_automatic_deletion_candidates`.

The figure audit (2026-09-17, commit `959e5d299`) recorded `:3736` — correct on
its own tree, 110 lines stale 24 hours later. The census, the plan's §3a and the
plan's audit comment now all carry the HEAD reading; the audit's measurement
table keeps its dated re-measurement, since it is a record of what that tree
held rather than a claim about this one.

## The check that was run, and what it does not cover

Superseded disposition figures are the counts of the three pre-spend columns.
Both files were grepped for them:

```
$ grep -cE '(^|[^0-9])(32|24|22)([^0-9]|$)[^|]{0,30}(RESTORE|CORRECTLY REMOVED|UNDETERMINED)|(RESTORE|CORRECTLY REMOVED|UNDETERMINED)[^|]{0,30}(^|[^0-9])(32|24|22)([^0-9]|$)'
  census: 0        plan: 0
$ grep -cE '32 ?/ ?24 ?/ ?22|32 RESTORE|24 CORRECTLY|22 UNDETERMINED'
  census: 0        plan: 0
```

The gap in the first pattern excludes `|`, the table cell separator, so a review
count in the per-request table cannot mask or manufacture a hit. Three residual
occurrences of those digits survive deliberately, and none is a disposition:

- the per-identity table's `archive reviews` column carries `24` on
  `perturbed_pressure`, whose archive verdict is UNDETERMINED — a review count,
  and the table's verdict column is the archive's, superseded per identity by
  the recorded-spend ruling;
- `32 of 110 distinct archived target/descendant identities survive` is the
  stable-ID live-match figure, an unrelated quantity that happens to share a
  numeral;
- `22 accepted direct-DD identities` is the size of one recovery route, not a
  disposition count.

The second check (`32 / 24 / 22` renderings) is the one the figure audit used;
its last surviving copy was in the plan's audit comment and is rewritten.

This record quotes the superseded triple once, above, as the before-state. That
quote is the only place the figures survive, and it is a record of the defect
rather than a claim of a disposition — a wider grep than the two scoped checks
will hit it here, and that hit should be read before it is counted.

## Files changed

- `docs/evidence/sn-lifecycle-integrity/unshielded-identity-deletions.md` —
  Result block, WEST-cut sentence, code-defects section (guard call sites read
  at HEAD, pre-guard pointers labelled as such), spend-override heading and
  sentence, formerly-undetermined count sentence.
- `docs/plans/paid-identity-deletion-protection.html` — §3a guard call-site
  pointer, the figure-audit comment's record of the defect, and the §6 landing
  comment. Plan version/meta lines untouched.

## Left open, not fixed here

- The plan's §6a prose says nine of fifteen recorded-spend rows where the
  census tables support eleven. Recovering the right figure needs the ledger and
  the census tables, and both sides sit outside this node's write paths.
- The plan carries two strings for one all-time spend figure. A re-derivation
  against the live ledger is needed (the ledger only augments, so both are
  floors), and the audit already recommended re-deriving rather than editing.