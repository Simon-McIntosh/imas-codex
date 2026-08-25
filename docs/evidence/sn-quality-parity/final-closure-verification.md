# Documentation-quality final closure verification

Snapshot: `2026-08-25T08:31:08Z`, worktree HEAD `59538a3e`, default live
`codex` graph. All graph operations were read only.

## Verdict

**`sn-quality-parity` is not genuinely complete at implementation 1.0.** Of
the **12 declared deliverables** in sections 2, 3, 5, and 6, **11 are landed
and 1 is outstanding**. The outstanding row is the WEST cohort documentation
refresh. Its two bounded campaigns demonstrably ran through the ordinary
documentation quorum and stayed within the authorized ceiling, but they were
sized on the directly source-bound population rather than on the production
review mint. The current production mint is a 416-identity family-closed packet,
not the 234-identity direct cohort used by the campaign evidence.

This is not a request to erase the successful campaign evidence. It is a
closure correction: the runs landed, the section-level done condition did not.

## Deliverable accounting

The accounting follows the plan's semantic units rather than treating every
explanatory bullet as a new implementation item:

- Section 2 names four delivered artifacts: the holdout, the COCOS lookup
  correction discovered while building it, holdout expansion, and the
  objective gate vector.
- Section 3 names four deliverables: the locked content-gate standard, example
  disjointness, phase-by-phase comparison, and the bulk validation sample. Its
  final bullet explicitly says a separate `unresolved_links` field and public
  resolver verb are **not required**, so they are not silently counted as
  missing work.
- Section 5 names three executable deliverables: the dimensional diagnostic,
  conditional sign diagnostic, and `not_evaluable` outcome. Its comparator
  paragraph constrains reuse; it explicitly says that it is not new machinery.
- Section 6 owns one executable deliverable, the WEST cohort documentation
  refresh. Its five bullets are acceptance and execution constraints on that
  one campaign, checked in the row below rather than counted as five campaigns.

| Section | Declared deliverable | Evidence | Verdict |
|---|---|---|---|
| 2 | Immutable DD-path/catalog-documentation holdout | The evidence pins every required field, split-key identity, immutable catalog commit, and a zero overlap against curated prompt examples (`docs/evidence/archive/sn-quality-parity-landed.html:77-107`). | **landed** |
| 2 | Correct COCOS convention-parameter lookup | The canonical-query regression failed before the edge correction and passed 7/7 afterwards; the repository scan found no remaining bare `[:COCOS]` traversal (`docs/evidence/archive/sn-quality-parity-landed.html:61-75`). | **landed** |
| 2 | Holdout expanded to its measured ceiling | The expanded holdout carries 85 unique path split keys from 13 catalog identities; all 85 keys are unique and both identity-level and split-key overlap with curated examples are zero (`docs/evidence/archive/sn-quality-parity-landed.html:130-148`). | **landed** |
| 2 | Objective per-gate vector | The scorer discriminated a complete catalog row at 10/10 from a stub at 4/10, with excluded gate names pinned by test (`docs/evidence/archive/sn-quality-parity-landed.html:110-128`). The later gate-role decision reduced the live instrument to six authority-backed checks and removed the four lexical patterns (`docs/evidence/archive/sn-quality-parity-landed.html:571-622`). | **landed** |
| 3 | Content-gate standard settled | The operative documentation obligations and the deliberate exclusion of ungrounded typical-value and generic measurement filler are recorded with their mechanism and policy basis (`docs/evidence/archive/sn-quality-parity-landed.html:553-569`). | **landed** |
| 3 | Curated examples disjoint from evaluation | The expanded holdout proves 85/85 unique DD-path split keys and zero intersection against curated examples at both identity and split-key levels (`docs/evidence/archive/sn-quality-parity-landed.html:130-145`). | **landed** |
| 3 | Phase-by-phase comparison loop | The production-enriched arm admitted 85/85 rows, injected 7-27 candidates per row, had zero generation failures, and recorded actual spend of USD 0.3271 under a USD 3 ceiling (`docs/evidence/archive/sn-quality-parity-landed.html:426-470`). **Qualification:** the subsequent gate-role decision makes the recorded nine-gate means historical diagnostic evidence, not a current physics-quality certificate (`docs/evidence/archive/sn-quality-parity-landed.html:588-617`). The comparison apparatus and execution are nevertheless delivered. | **landed** |
| 3 | Deterministic off-holdout bulk validation | The production-faithful re-run reproduced all 53 selection ranks with zero holdout overlap, admitted all 53, and eliminated the context-starved relationship collapse from 14/53 to 53/53 (`docs/evidence/archive/sn-quality-parity-landed.html:473-500`). The surviving qualification is explicit: the relationship check measured link-or-phrase presence, not semantic correctness. | **landed** |
| 5 | Dimensional consistency against the DD-declared unit | The implementation resolves authoritative units, binds symbols, evaluates restricted dimensional algebra, and has a genuine pressure-unit mismatch regression (`docs/evidence/sn-quality-parity/closure-audit.md:12-15`). Calibration finds 1 genuine physics error, 32 symbol-binding gaps, and 5 parser limitations among 38 catalog failures, so this remains a diagnostic rather than automatic defect authority (`docs/evidence/sn-quality-parity/dimensional-gate-calibration.md:3-18`, `docs/evidence/sn-quality-parity/dimensional-gate-calibration.md:27-41`). | **landed**, qualified diagnostic |
| 5 | Sign convention conditional on transformation type | The scorer requires canonical final sign prose for sensitive classes, forbids it for `one_like`, abstains without authority, and rejects metadata leakage (`docs/evidence/sn-quality-parity/closure-audit.md:15-16`). The final governed repair moved the full accepted-document census from 333 pass / 18 fail / 2,601 not evaluable to **351 / 0 / 2,601**, with 2,952/2,952 identity and documentation coverage and USD 0 spend (`docs/evidence/sn-quality-parity/sign-residual-repair.md:6-22`). | **landed** |
| 5 | Third outcome state, `not_evaluable`, through scoring and aggregation | The enum, evaluable denominator, holdout physics context, and numerical aggregation regression are all present; live graph-backed scoring exercises thousands of abstentions rather than leaving the state dormant (`docs/evidence/sn-quality-parity/closure-audit.md:16-16`, `docs/evidence/sn-quality-parity/closure-audit.md:39-53`). | **landed** |
| 6 | WEST cohort documentation refresh through ordinary quorum, reversible snapshots, exact identity joins, serialized maintenance, and bounded tranches | Tranche one refreshed 130 identities, accepted 129, wrote 130/130 prior-text snapshots, and spent USD 22.690535 under USD 40 (`docs/evidence/sn-quality-parity/west-docs-resumed-refresh.md:3-20`, `docs/evidence/sn-quality-parity/west-docs-resumed-refresh.md:143-157`). Tranche two selected 18, accepted 14, retained four reviewed failures, and spent USD 4.592470 under USD 60 (`docs/evidence/sn-quality-parity/west-docs-second-tranche.md:3-28`, `docs/evidence/sn-quality-parity/west-docs-second-tranche.md:74-104`). Those successful runs covered the direct source-bound census, while the production mint requires direct binding plus immediate-family closure (`imas_codex/standard_names/minting.py:1-21`, `imas_codex/standard_names/minting.py:72-123`). Current live observation G1 is 218 direct + 198 family-only = **416**, with only **381 docs accepted**. | **outstanding** |

## Current live graph observation

**G1 — production WEST review mint, read only.** The repository's production
`load_sources_file()` and `mint_sn_list()` path was run against the default
`codex` graph from this worktree. Before trusting any zero, the query retained
property coverage:

| Measure | Current result |
|---|---:|
| `StandardName` candidates / carrying `id` | **4,656 / 4,656** |
| Candidates carrying undeclared `name` | **0 / 4,656** |
| WEST manifest DD paths | **355** |
| Paths without a live non-terminal target | **27** |
| Directly source-bound live identities | **218** |
| Immediate-family-only identities | **198** |
| Production review mint | **416** |
| Documentation stages | **381 accepted, 29 pending, 5 reviewed, 1 null** |
| Name stages | **379 accepted, 30 reviewed, 6 drafted, 1 pending** |
| Validation states | **405 valid, 10 quarantined, 1 null** |
| Non-accepted documentation, direct / family-only / total | **7 / 28 / 35** |

This independently reproduces the committed packet-readiness census: 416
identities, 381 documentation-accepted, and 368 simultaneously name-accepted,
documentation-accepted, and validation-valid (`docs/evidence/sn-west-review-rehearsal/packet-readiness.md:3-19`). That record also shows the exact mint composition as 218 direct identities plus 198 family additions (`docs/evidence/sn-west-review-rehearsal/packet-readiness.md:21-45`) and the direct/family lifecycle split (`docs/evidence/sn-west-review-rehearsal/packet-readiness.md:77-89`).

## The two residue questions

### The 18 identities deferred by name

They are **outstanding scope, not accepted residue**. At the second-tranche
snapshot the fresh 234-identity direct census retained 13 pending, 4 reviewed,
and 1 exhausted document after the run (`docs/evidence/sn-quality-parity/west-docs-second-tranche.md:106-119`). The executed 18-row tranche itself accepted 14 and left four exact reviewed documents fail-closed because the ordinary refinement eligibility predicate could not claim them (`docs/evidence/sn-quality-parity/west-docs-second-tranche.md:98-104`). A `no_eligible_work` stop proves only that the then-current predicates could claim nothing else; it does not turn non-accepted documents into completed refreshes.

Name refinement subsequently changes identity membership, so the old set of 18
must not be treated as an immutable current worklist. Its obligation follows the
current live identities and successors. G1 currently finds 35 non-accepted
documents in the production mint, including seven directly bound identities.
The old 18 is therefore closure evidence of unfinished lifecycle work, not a
permanent count to reproduce or waive.

### The 416-family closure versus the 234 campaign census

The difference is also **outstanding scope, not accepted residue**. The 234-row
census was a direct `StandardNameSource` to `PRODUCED_NAME` view
(`docs/evidence/sn-quality-parity/west-docs-second-tranche.md:35-50`). The actual
review/approval currency deliberately adds the immediate parent, siblings, and
children because approving a touched identity without its immediate family is
defined as incoherent (`imas_codex/standard_names/minting.py:1-21`). The current
mint adds 198 such identities; 28 of those 198 do not have accepted
documentation, and the committed readiness record shows only 168/198 are
simultaneously name-accepted, documentation-accepted, and valid
(`docs/evidence/sn-west-review-rehearsal/packet-readiness.md:77-85`).

More importantly, even family identities whose documentation is already
accepted are not thereby proven refreshed: the campaign selection and snapshot
receipts were constructed from the direct cohort. The plan says the authority
is the human packet, and the production packet is the 416-identity mint.
Therefore a closure claim needs an exact disposition over all 416 current
identities—refreshed, naturally generated after the improved standard,
unchanged under an explicit no-refresh disposition, or withheld with a reason.
No such 416-row refresh ledger exists, and the packet census reports the batch
unstable and unreleasable (`docs/evidence/sn-west-review-rehearsal/packet-readiness.md:3-16`).

## Closure correction and next evidence condition

The plan's stored `plan-impl=1.0` should not be retained as a factual completion
claim. The evidence supports **11/12 declared deliverables landed, 1/12
outstanding**. If the plan's coarser five-node implementation accounting is
preserved, the same conclusion is **4/5 complete**, because the sole incomplete
node is the WEST cohort refresh.

The row can close only when a fresh production `mint_sn_list()` census has an
exact 416-current-identity (or whatever cardinality the fresh mint returns)
ledger with:

1. every identity dispositioned against the improved-documentation refresh;
2. every accepted-document reset carrying a prior `DocsRevision` snapshot;
3. every promoted document earning ordinary quorum, never direct acceptance;
4. every withheld or lifecycle-blocked identity named with its current reason;
5. direct and family-only counts reported separately; and
6. actual spend reported beside the authorized ceiling.

Until then the accurate status is **active with one outstanding deliverable**,
not complete at implementation 1.0.

## Verification and limitations

- The focused holdout, gate, evaluator, and production-mint contract run
  completed with **41 passed, 2 deselected, 0 failed** in 10.47 seconds:
  `pytest -p no:cacheprovider tests/standard_names/test_docs_gates.py
  tests/standard_names/test_docs_holdout_eval.py
  tests/standard_names/test_docs_holdout_set.py
  tests/standard_names/test_minting.py`. Full output is at
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T082438898163-n-parityfinalclosure/logs/focused-tests.log`.
- A mechanical evidence check found exactly **12** verdict rows, **11** landed
  and **1** outstanding, and resolved all **32** cited file/line ranges.
- Reckon's typed `read_plan` call failed before returning this plan because the
  unrelated nested resource path
  `research/standard-names/01-current-state-standard-names.html` violates the
  typed-resource path contract. The complete canonical plan HTML was therefore
  read directly from this dispatched checkout. This is the same read-path
  limitation recorded by the earlier two-plan closure verification
  (`docs/evidence/archive/integrity-and-operator-closure-verification.md:198-215`).
- No plan, index, graph, Standard Name, or documentation lifecycle state was
  mutated by this node.
