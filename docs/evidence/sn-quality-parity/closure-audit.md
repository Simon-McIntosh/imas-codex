# Documentation-quality closure audit

Snapshot: `2026-08-24T15:52:57Z` against worktree HEAD `5dbad32f` and the live
`codex` graph. This audit treats the documentation checks as instruments, as the
live plan requires: a check is landed when its implementation, test contract and
authoritative graph input are present. Existing documents do not have to pass a
new diagnostic for the diagnostic itself to be landed. A campaign deliverable is
landed only when its required graph mutations and lifecycle outcomes are complete.

## Closure result

| Section deliverable | Live tree | Live graph | Verdict |
|---|---|---|---|
| Dimensional consistency of the defining relation against the DD-declared unit | The scorer resolves the declared unit, binds stated symbols, evaluates restricted dimensional algebra and distinguishes matching from contradicting dimensions (`imas_codex/standard_names/docs_gates.py:354-413`). A genuine pressure-unit mismatch and its corrected relation are pinned by tests (`tests/standard_names/test_docs_gates.py:74-101`). | Authority is present on nearly the whole catalog: 4,661/4,666 `StandardName` nodes carry the unit property and 4,657 have a `HAS_UNIT` edge (G1). Applying the live scorer to all 2,727 accepted documents returns 10 pass, 143 fail and 2,574 `not_evaluable` (G3). The implementation therefore exists and catches nine explicit dimensional contradictions, but its binding coverage is still too low for a graph-wide quality verdict; the paired persisted arm likewise has 0/85 generated rows evaluable (`docs/evidence/sn-quality-parity/paired-arm-current-gates.md:11-17`, `docs/evidence/sn-quality-parity/paired-arm-current-gates.md:69-78`). | **landed**, qualified: diagnostic only; symbol binding remains weak. |
| `sign_convention` conditional on `cocos_transformation_type` | The scorer rejects internal COCOS numbers/labels, requires one final canonical sign paragraph for sensitive transformations, forbids it for `one_like`, and abstains when the transformation class is absent (`imas_codex/standard_names/docs_gates.py:429-480`). Both directions, missing authority and metadata leakage are tested (`tests/standard_names/test_docs_gates.py:294-341`). | The live graph retains the recorded 957 `one_like` plus 24 sensitive distribution (G1). Scoring all 2,727 accepted documents produces 100 pass, 281 fail and 2,346 `not_evaluable`; all 10 accepted sensitive documents with transformation metadata pass, while 281/371 accepted invariant documents fail because they still state a sign convention (G2). Separately, 0/2,727 accepted documents expose a COCOS number or transform label (G2). | **landed**, qualified: the instrument is correct, but it exposes 281 accepted-document content violations that have not been remediated. |
| Third outcome state, `not_evaluable`, threaded through row scoring and aggregation | `DocumentationGateOutcome` has `pass`, `fail` and `not_evaluable`; score counts exclude abstentions from the evaluable denominator (`imas_codex/standard_names/docs_gates.py:74-155`). Holdout rows retain physics context and aggregation reports pass, contradiction and abstention separately (`imas_codex/standard_names/docs_holdout_eval.py:59-69`, `imas_codex/standard_names/docs_holdout_eval.py:321-370`). The numerical regression proves an abstention aggregates to 1.0 where scoring it false would yield 11/12 (`tests/standard_names/test_docs_holdout_eval.py:97-143`). | The state is exercised, not dormant: current graph-backed scoring emits 2,574 dimensional abstentions and 2,346 sign abstentions across 2,727 accepted documents (G2-G3). Outcomes are evaluation artifacts rather than graph lifecycle properties, so no separate persisted graph field is expected. | **landed**. |
| WEST cohort documentation refresh through ordinary quorum under the hard cap | The first accepted-document tranche records 40/175 refreshed, 40/40 accepted, 40/40 prior-text snapshots and USD 5.120219 actual spend under the USD 150 ceiling (`docs/evidence/sn-quality-parity/west-docs-tranche-refresh.md:5-19`, `docs/evidence/sn-quality-parity/west-docs-tranche-refresh.md:35-43`). The mid-loop tranche records 22 acceptances but retains 13 non-terminal and one exhausted identity (`docs/evidence/sn-quality-parity/west-docs-tranche-completion.md:5-25`, `docs/evidence/sn-quality-parity/west-docs-tranche-completion.md:107-114`). | Of the original 175 accepted-document identities, all 175 still resolve, only 44 have any revision since the first tranche began, and 131 have none; 173 remain docs-accepted and 172 name-accepted (G4). The separate 36-row mid-loop cohort currently remains 22 accepted, 6 reviewed, 7 pending and 1 exhausted (G5). Of the 20 identities excluded for concurrent name refinement, 13 now have lineage successors, but all 13 successor documents remain pending; the other originals are 13 superseded, 6 exhausted and 1 reviewed on the name axis (G6). | **outstanding**: the bounded tranches succeeded, but the WEST cohort refresh is not complete. |

Headline: **3 landed, 1 outstanding**. The one outstanding row is the WEST
cohort refresh. The dimensional and sign mechanisms are landed only as
diagnostics; their measured abstentions and failures remain visible above and
must not be worded into a catalog-quality pass.

## Numbered live-graph observations

All observations were read-only. Before trusting any zero, each query retained
its candidate count or property coverage. G2 and G3 project only the listed
properties from Cypher and apply the live tree's deterministic scorer to those
rows; they make no graph writes or model calls.

1. **G1 — authority coverage and transformation distribution.** Cypher over all
   `StandardName` nodes returned 4,666 candidates, 4,666 with `id`, 4,661 with
   `unit`, 4,657 with a `HAS_UNIT` edge, 981 with
   `cocos_transformation_type`, and 983 `HAS_COCOS` edges. The transformation
   property distribution is `one_like` 957, `dodpsi_like` 6, `b0_like` 5,
   `ip_like` 5, `psi_like` 5, `pol_angle_like` 2 and `q_like` 1: 957 invariant
   plus 24 sensitive.

2. **G2 — conditional-sign behavior on accepted graph documentation.** Cypher
   projected `documentation` and `cocos_transformation_type` for 2,727
   docs-accepted nodes. The live scorer returned `pass=100`, `fail=281`,
   `not_evaluable=2346`. Restricted to the 381 accepted rows carrying transform
   metadata, sensitive rows were 10 pass / 0 fail; invariant rows were 90 pass /
   281 fail. An independent Cypher text census over the same 2,727 candidates
   found 0 documents naming a COCOS number and 0 naming a transform label.

3. **G3 — dimensional behavior on accepted graph documentation.** Cypher
   projected `documentation` and `unit` for all 2,727 docs-accepted nodes. The
   live scorer returned `pass=10`, `fail=143`, `not_evaluable=2574`. The fail
   reasons split into 133 documents lacking exactly one checkable defining
   relation, nine explicit dimension contradictions and one relation without a
   dimensional equality. The abstentions split into 2,571 unbound-symbol results
   and three unbound subject-symbol results.

4. **G4 — original 175-row accepted WEST refresh cohort.** The frozen census
   supplied the exact identity list and Cypher re-resolved it by
   `StandardName.id`: 175/175 live nodes, 44 with a `DocsRevision` created since
   `2026-08-23T20:00:00Z`, 131 without one, 173 currently docs-accepted and 172
   currently name-accepted. The 44 revision count is an upper bound on refresh
   progress, because the durable campaign evidence attributes only 40 identities
   to the first refresh tranche.

5. **G5 — original 36-row mid-loop WEST cohort.** Exact-id Cypher returned 36/36
   live nodes. Current documentation stages are accepted 22, reviewed 6, pending
   7 and exhausted 1; current name stages include accepted 29, superseded 2 and
   exhausted 2. The documentation-stage counts sum to all 36 rows.

6. **G6 — 20-row concurrent-refinement exclusion.** Exact-id Cypher returned all
   20 originals: name stages are superseded 13, exhausted 6 and reviewed 1, with
   all 20 legacy documents still accepted. Schema-direction traversal
   `(successor)-[:REFINED_FROM]->(original)` resolves 13 successors; six successor
   names are accepted, zero successor documents are accepted and all 13 successor
   documents are pending.

## Verification

`pytest -p no:cacheprovider tests/standard_names/test_docs_gates.py
tests/standard_names/test_docs_holdout_eval.py` completed once with **31 passed,
0 failed** in 7.08 seconds. Full output:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T154413387599-n-parityclosure/focused-tests.log`.

## Required continuation

The WEST closure needs a fresh exact-id census before each mutation tranche. At
minimum, it must disposition the 131 original accepted rows with no post-start
revision, the 14 residual mid-loop rows, and the 20 concurrent-refinement rows
on their current identities or successors. The 281 accepted invariant documents
that fail the new sign diagnostic and the dimensional binder's 2,574 abstentions
are real quality/instrument findings, but the live plan makes the checks
diagnostic rather than an automatic graph-wide rewrite authority.
