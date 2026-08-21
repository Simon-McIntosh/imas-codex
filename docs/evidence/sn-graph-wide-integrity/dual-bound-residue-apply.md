NEEDS-HELP: The corrected fresh invocation was terminated without an application error or receipt after exceeding the command wrapper's runtime boundary; recovery proves that its graph transaction did not commit.

tried: Both closed selection fields were changed to the exact literal `artifact-rows`. The driver then freshly re-derived the validated live cohort, regenerated the canonical payload and file SHA-256 digests from the corrected bytes, and started one new preview/apply/replay invocation. The process ran for more than two minutes, emitted only the existing generated-model warning, then disappeared without a Python traceback, shell exit status, preview receipt, apply receipt, or evidence JSON. No second corrected invocation was started.

options: First determine why the generic signed preview exceeds the command wrapper's runtime boundary, then execute the already-corrected driver in a persistent execution session with an explicit exit marker; alternatively, optimize or bound the signed operator's read-only collateral snapshot before redispatch; or retain the corrected authority as unapplied evidence while the separate authority-builder work lands.

leaning: Inspect the preview runtime and rerun the corrected driver under a persistent session that can outlive the observed wrapper boundary. The authority now has the exact schema-valid closed selection shape, so changing its JSON again would be unjustified; the new failure is execution termination, not another authority-shape refusal.

cost-if-wrong: If the wrapper was not the termination source, the next execution could stall at the same operator query and require profiling before any apply. The signed transaction and the recovery census remain fail-closed, so no partial graph repair needs reversal; a committed transaction would instead have produced reconciliation receipts and changed both audited counters.

## Blocked outcome

The corrected authority contains **23 signed rows** and both closed selection
fields equal `artifact-rows`. The applying process did not return far enough to
persist its own preview/apply/replay evidence. A separate read-only recovery
audit established the graph state after the process ended:

| Measure | Observed |
|---|---:|
| Corrected signed authority rows | 23 |
| Live dual-bound sources after termination | 23 |
| Preview rows classified | unavailable; no terminal preview receipt |
| Mutated rows | 0 |
| Reconciliation receipt rows | 0 |
| Replay | not reached or not returned |
| Current `StandardNameChange` nodes | 7,759 |
| Current `PRODUCED_NAME` relationships | 5,791 |
| Current `LLMCost` nodes | 27,631 |

The required quantitative measure remains **unmet**: admitted plus refused was
not returned, no refusal list was returned, no receipt cardinality exists, and
no replay receipt exists. The current `StandardNameChange` count remains at the
required pre-apply baseline of 7,759, `PRODUCED_NAME` remains 5,791, and there
are zero `StandardNameChange` rows for operation
`reconcile_standard_name_source_targets`. Together with all 23 live dual-bound
sources remaining, those facts prove zero production mutation by the corrected
attempt.

## Corrected recoverable authority

- Authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/dual-bound-source-target-authority.json`
- Selection id: `artifact-rows`
- Selection predicate: `artifact-rows`
- Rows: **23**; repair rows: **23**
- Corrected authority file SHA-256:
  `ef51f912038976721b35f4fa7a830c43983bab001b675b59559c99a18ce58972`
- Corrected canonical payload SHA-256:
  `2a64bb77d99c8e0c9934fba3052473040a1322f8920f78cdb2396dfec6007adc`
- Corrected invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/apply-dual-bound-fresh.log`
- Post-termination recovery audit:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T204309912344-dualapply/fresh-attempt-recovery-audit.log`

The corrected authority records, for every signed source, its complete current
live target set, selected survivor, and every losing relationship identity. Its
digests were generated only after both selection fields were corrected. It was
not applied, and its manifest preview hash does not exist.
