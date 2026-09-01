# Drain ledger — imas-codex sprint S9 — 2026-09-01T14:30Z

Session `ship-s9-20260901`. Every open followup row in scope, each with a
disposition from the closed set. A row disposed `folded` names the node that
executed it; a row left open names the exemption that keeps it open.

| Row | Disposition | Detail |
|---|---|---|
| `f-srr-embedgap` (sn-release-readiness) | **folded** | `n-embedgap` — accepted embedding gap 2→0, control 2316→2318, USD 0.00 |
| producer defect (fold of above) | **folded** | `n-producerembeddrain` — regression fails pre-change, passes post; 52/0 |
| `f-dsv4-advisory` (sn-benchmark-evolution) | **folded** | retired on the lead's relock; census not run, negative result recorded |
| `f-benchmark-deterministicserving` | **folded** | not purchased on the lead's decision; GPU shortfall measured (7 of 8 allocated) |
| `f-exhausted-tail-triage` | **folded** | `n-tailtriage` — 275 → 32/4/239, ordered-row hash identical across two passes |
| `f-tailrescore` | **folded** | `n-tail32rescore` — 12 of 32 accepted, USD 4.98; three pipeline defects surfaced |
| 4 pre-existing suite failures | **folded** | `n-suitefourfix` — all four stale tests; both export paths shown sound |
| new import-order timeout | **folded** | `n-importordertimeout` — cleared as non-regression, +0.29 s vs 30 s timeout |
| ordinal strain-gauge names | **folded** | `n-strainordinaledit` — 6/6 ISN vocabulary gap, USD 0.00, graph untouched |
| unsourced chain-cap refines | **folded** | `n-unsourcedancestry` — verdict rebuild: 72 no-ancestor, 4 repair, 2 adjudicate |
| `f-crs-001` (catalog-review-surface) | **folded** | superseded by `f-crs-002`; two of its premises overtaken the same day |
| graph down after backup | **folded** | `n-graphpartition` — dead `partition: titan` config honoured; 1,612,957 nodes |
| `f-crs-002` (catalog-review-surface) | **authority-required** | needs `destructive-target-posture` + `backup-failure-posture` locked — the lead's |
| `f-gdcs-001` (graph-operational-safety) | **authority-required** | same two open decisions; they shape the work rather than follow it |
| `f-rescorecontract` (sn-benchmark-evolution) | **authority-required** | touches the refine persistence path the whole SN pipeline depends on |
| `f-srr-unsourced` (sn-release-readiness) | **authority-required** | 72 rebuilds + an ISN-owned vocabulary decision codex must not pre-empt |
| `f-sgic-002` (sn-graph-identity-convention) | **foreign-owner** | correctly-formed pointer: work belongs to `sn-schema-ownership-residue` (S12) |
| `f-wqp-sourceresidue` (sn-quality-parity) | **foreign-owner** | on a shipped plan and names no owner; subject now overlaps §12 — needs rehoming |

```
rows: 18   foldable-remaining: 0   unreconciled-runs: 0
```

Both figures are the termination condition. `unreconciled-runs` read from
`crew(project, view="drain")`, not from memory. Sixteen runs promoted this
session, every one carrying its gate verdict and its measure.
