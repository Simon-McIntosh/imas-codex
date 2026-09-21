# The remaining local-lane landings, audited before the WEST cut

The export audit found three sites of one shape in `export.py`: a key-presence
test standing in for an observation, so an absent key licensed a decision
nothing had actually measured. This reads the sprint's three remaining
local-lane pipeline changes for the same shape, and for the two questions that
go with it — whether the change can suppress or skip something it should not,
and whether it can misreport.

Measured at `ec9d7af42` in the detached worktree
`ship-s10-20260918/n-sli-the-remaining-local-lane-landings-are-audited`,
against the shared project environment.

| Change | What it adds | Verdict |
|---|---|---|
| `dd88d0d3f` `edit.py` | scoped edit bypasses graph-wide maintenance | **SAFE-FOR-WEST** — two live defects, both on the edit path, neither reaching a cut |
| `0ba05e980` `review/audits.py` | the audit walk declares its question and refuses the verdict | **SAFE-FOR-WEST** — additive; but it declares a surface whose only consumer drops every finding |
| `73242777f` `graph_ops.py`, `orphan_sweep.py` | the parking writer records a disposition as it parks | **SAFE-FOR-WEST** for the cut, **MUST-FIX** for §5's own measure — four defects, all misreport-direction |

No change in this set can admit a wrong name into a published catalog or keep a
right one out: none of the three is read by `export.py`, verified by grep over
`imas_codex/` for `parked_disposition`, `AuditReport` and
`skip_global_maintenance` — the export module references none of them. Every
defect below is therefore a triage-quality or edit-path exposure, which is what
makes them the §5 shape rather than the §2 shape.

## How the three were exercised

```
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_scoped_edit_keeps_its_scope.py \
  tests/standard_names/test_quarantine_instrument_agreement.py \
  tests/standard_names/test_attempt_cap_disposition.py
```

`13 passed` (log: `/tmp/sli-lane-audit/three-changes.log`). **All three suites
are green and none of the six defects below is visible to any of them** — that
is the finding about the suites, not an aside. Each defect was reproduced
directly against the landed code instead (log:
`/tmp/sli-lane-audit/probes.log`), because a passing suite never shows that a
guard fires.

## 1. `dd88d0d3f` — the scoped edit bypasses graph-wide maintenance

`_run_scoped_pipeline` (`imas_codex/standard_names/edit.py:3752`) now passes
`skip_global_maintenance=True` into `run_sn_pools`. The pairing is sound at the
boundary: `run_id` is a required keyword forwarded as `scope_run_id`, both edit
callers (`edit.py:3796`, `edit.py:3903`) reach the orchestrator through this one
frame, and the loop's own guards hold — `loop.py:1412` refuses the bypass
without a scope, `loop.py:1416` refuses it with a maintenance-only mode, and
neither is reachable from here.

**Can it wrongly ADMIT?** No. It strictly removes work; it cannot cause a write.

**Can the guard be bypassed?** **No, and this is the change's strongest
property.** The flag is a literal in the shared frame, not a parameter, so no
edit-side caller can separate the scope from the bypass by forgetting an
argument. The mode is also recorded — `loop.py:1538` carries
`skip_global_maintenance` into the run record — so a later reader can tell the
run was scoped.

**Can it suppress something it should not? Yes, twice.**

- **Defect 1 — a skipped reconcile is returned shaped exactly like a clean
  one.** `loop.py:1884` substitutes a default-constructed
  `AttachmentAuditResult()` when the flag is set, and `loop.py:1890` logs only
  `if attach_result.detached or attach_result.rejected:`. A run that never
  looked and a run that looked and found nothing are therefore
  indistinguishable in the result object and identical in the log.
  `_global_maintenance_call` (`loop.py:1570`) generalises the same substitution
  — it returns the caller's `default` for every bypassed function. This is the
  `export.py` shape exactly: an unexamined absence rendered as an observed zero.
  Concrete trigger: any `sn edit` run at all; the substitution is
  unconditional once the flag is set.

- **Defect 2 — the bypass removes the in-fence reconcile along with the
  out-of-fence one.** The justification is correct as far as it goes: the global
  writers would rewrite identities outside the operator's fence. But the remedy
  chosen is total, so nothing reconciles the *edited* identity either.
  Suppressed for the edited name are `rederive_structural_edges`,
  `normalize_derived_parent_lifecycle` and `reconcile_orphan_parent_sources`
  (`loop.py:2722`), the `StandardName` embedding worker (`loop.py:2286`) and
  `restamp_harmonized_families` (`loop.py:2777`). Concrete consequences, in
  order of how directly they bite: an edited name's new description carries no
  refreshed embedding until some later unscoped run, so near-duplicate detection
  and semantic search do not see it; and its sibling family keeps a
  harmonization signature stamped against the pre-edit docs, so the family reads
  as harmonized when it is not. The orphan sweep is deliberately exempted
  (`loop.py:2243` states why), which shows the distinction was available and was
  simply not extended to a scoped reconcile.

**Verdict: SAFE-FOR-WEST.** The change fixes a genuine escalation — a fenced
single-identity edit was running four graph-wide reconcile writers — and the
fix is not bypassable. The two defects are both on the edit path and neither can
reach a catalog cut.

**One coupling to check before relying on this, stated as a question rather than
a finding, because it needs a live graph to settle.** The export's producer
predicate admits a candidate carrying a derived producer, a non-derived
producer, or a live structural child (`export.py:777`). Structural child edges
are derived by `rederive_structural_edges`, which this change suppresses on the
edit path. If an `sn edit` mints a new identity whose only producing evidence
would have been a structural child, that identity leaves the edit run without
it and the export refuses it as `no_producing_source`. The check is to run
`sn edit` on a name with structural children against a live graph and read the
edge count before and after; it is not reproducible from the code alone and is
recorded as a follow-on rather than asserted here.

## 2. `0ba05e980` — the audit walk declares its question

`review/audits.py` gains a module docstring separating the two questions,
`AUDIT_WALK_QUESTION` / `QUARANTINE_QUESTION` / `QUARANTINE_AUTHORITY`, two
`AuditReport` fields carrying the declaration, and
`answer_quarantine_question`, which raises `QuarantineVerdictNotAnswered`.

**Can it suppress, skip, or misemit?** No. It is additive: two defaulted
pydantic fields and one function. No existing code path changes behaviour.

**Can it misreport?** Not by itself, but the declaration it adds is not yet
true of the delivery path — see defect 4.

**Defect 3 — the refusal has no production caller, and the declaration is an
overridable default.** `answer_quarantine_question` and
`QuarantineVerdictNotAnswered` are imported at exactly two sites, both in
`tests/standard_names/test_quarantine_instrument_agreement.py:22-23`; grep over
`imas_codex/` returns only the definitions. So the guard is never armed at
runtime: no caller asking the walk for a quarantine verdict is redirected,
because no caller was routed through it. The two report fields are plain
defaults rather than `Literal` or frozen, so a caller can construct a report
that claims the opposite of the declaration. Measured:

```
AuditReport(answers_question=QUARANTINE_QUESTION, refuses_question="nothing")
  answers_question: 'Is this name quarantined?' | refuses: 'nothing'
```

The test at `test_quarantine_instrument_agreement.py:33-37` asserts
`"quarantine" not in report.answers_question.lower()` on a
default-constructed report, so it passes while this construction stands beside
it. Concrete trigger: any caller passing either field.

**Defect 4 — the only consumer of an `AuditReport` reads a field that does not
exist, so every Layer-1 finding is silently dropped.**
`_extract_audit_findings` (`review/pipeline.py:1720`) iterates
`getattr(audit_report, "findings", [])`. `AuditReport` has no `findings` field —
its fields are `embedding`, `lint_findings`, `link_findings`,
`duplicate_components`, and now the two question strings. The object reaching
that call is an `AuditReport`: `cli/sn.py:6040` assigns
`state.audit_report = run_all_audits(all_names)` and `run_all_audits` is
annotated `-> AuditReport`; `pipeline.py:549` passes it straight in. Measured:

```
AuditReport fields: ['answers_question','duplicate_components','embedding',
                     'link_findings','lint_findings','refuses_question']
hasattr(AuditReport(), 'findings'): False
report carrying 1 ERROR lint finding -> _extract_audit_findings(...): []
```

The `getattr` default is the `export.py` shape again, and the inner loop repeats
it: `getattr(finding, "affected_names", [])` names a field `LintFinding` also
does not have (`LintFinding` fields are `detail`, `finding_type`, `name_id`,
`severity`), so even a corrected outer read would drop every row. Concrete
trigger: any review run with `dry_run` false and at least one lint finding —
`cli/sn.py:6049` prints the finding count to the console, so the operator is
told findings exist in the same run in which none of them reaches the reviewer
prompt.

This defect predates `0ba05e980`; the change did not introduce it. It is
reported here because it is the exact surface `0ba05e980` declares: a module
that has just stated in writing which question its report answers is handing
that report to a consumer that reads none of it.

**Verdict: SAFE-FOR-WEST.** Nothing in the change can affect a cut. Defect 4 is
a live review-quality defect and should be fixed before the next review
rotation, not before the WEST cut, whose names are already reviewed.

## 3. `73242777f` — the parking writer records a disposition

The sweep tick now follows its parking statement with
`record_dispositions_for_undisposed_parks` (`graph_ops.py:11599`), which reads
`_CAP_UNDISPOSED_QUERY` and delegates to the extracted
`record_parked_dispositions` (`graph_ops.py:11519`), shared with the reconcile
pass `disposition_parked_sources` (`graph_ops.py:11556`).

**Can it wrongly ADMIT?** No. It writes one property on rows it selected by that
property being null; it cannot change a lifecycle state.

**Can it skip something it should not? Yes.**

- **Defect 5 — a truthiness gate on the parking count, with no automatic net
  behind it.** `orphan_sweep.py:162` runs the disposition pass only
  `if counts.get(_PARKING_LABEL):`. The intent is stated and is reasonable — a
  tick that parked nothing should not read. But the parking write and the
  disposition write are two transactions, so a process that dies between them
  leaves rows parked and undisposed; the next tick parks zero of them (they are
  already `failed`, and the sweep statement matches only `status='extracted'`),
  the gate is false, and the read never runs again. The reconcile net that is
  supposed to catch exactly this — `disposition_parked_sources` — **has no
  production caller.** Grep over the repository for
  `disposition_parked_sources|census_parked_dispositions` returns the
  definitions in `graph_ops.py`, three call sites in
  `tests/standard_names/test_attempt_cap_disposition.py`, and prose in `docs/`.
  Nothing else. So the commit message's "the reconcile pass remains as the net"
  describes a net that is not attached to anything. Concrete trigger: interrupt
  a sweep tick between the parking statement and the disposition write — a
  `sn run` cancelled at the wrong second — and those rows are permanently
  undisposed absent a hand-run pass.

**Can it misreport? Yes, twice.**

- **Defect 6 — the sweep writer can reach only four of the six dispositions, and
  the two it cannot reach are the two that matter for triage.**
  `_CAP_UNDISPOSED_QUERY` (`graph_ops.py:11585`) selects on
  `sns.last_error = $cap_reason`, so every row it hands the classifier carries
  `last_error == _COMPOSE_CAP_REASON`. `classify_parked_source`
  (`graph_ops.py:11431`) enters its reason branch only when
  `reason and reason != _COMPOSE_CAP_REASON` (`graph_ops.py:11451`), so for
  every row from this caller that branch is dead. Measured:

  ```
  reachable when last_error == the cap reason:
    ['cause_not_recorded','compose_not_applicable','name_produced',
     'upstream_quantity_removed']
  unreachable from the sweep writer:
    ['attempt_budget_exhausted','vocabulary_gap']
  ```

  A cohort stamped entirely by this path can never read as a spent budget or a
  vocabulary gap, which is precisely the distinction §5 of the plan says the
  disposition exists to make. The reconcile pass reads `_CAP_PARKED_QUERY`,
  which has no `last_error` filter and can reach all six — but by defect 5 it
  never runs. Concrete trigger: any source parked by the sweep whose prior
  `last_error` recorded a vocabulary gap; the parking statement
  (`orphan_sweep.py:114`) overwrites `last_error` with the cap string before the
  classifier ever sees the original cause, so the cause is destroyed by the same
  tick that classifies it.

- **Defect 7 — a re-parked source keeps the disposition from its previous
  park.** The undisposed query filters `sns.parked_disposition IS NULL`, and no
  path clears the property: `retry_failed_sources` / `--reset-to extracted`
  clears `attempt_count` and `status` but leaves `parked_disposition` set. The
  generated schema states the same conclusion independently —
  `graph/models.py:6999` describes the field as "A non-null value currently
  causes the classifier to skip the row; no implemented path clears or refreshes
  a value when its supporting evidence changes." Concrete trigger: a source
  classified `cause_not_recorded`, revived, composed again, failed again, and
  re-parked — it still reads `cause_not_recorded` while its evidence now says
  otherwise, and the sweep writer will not look at it again.

**Can the guard be bypassed?**

- **Defect 8 — the stated refusal is unreachable, and the absence it was meant
  to catch is absorbed instead.** `record_parked_dispositions:11540` raises
  `ValueError` when a classification falls outside `CAP_PARKED_DISPOSITIONS`,
  and its docstring claims this means "no caller can end with a row carrying an
  unclassified disposition". `classify_parked_source` is total by construction —
  every branch returns a member of the closed set — so the refusal can never
  fire. Enumerated over 135 evidence shapes (3 × 3 × 5 × 3 across `produced`,
  `lifecycle_status`, `last_error`, `node_category`), every output was inside
  the set and `classify_parked_source({})` returned `cause_not_recorded`. What
  the guard was aimed at is a caller handing rows from a different projection,
  and that case is not refused but silently absorbed, because the classifier
  reads every field through `evidence.get(...)`:

  ```
  {'id':'s1','produced':3,...}                 -> name_produced
  the same row with the 'produced' key absent  -> cause_not_recorded
  ```

  An absent key is read as an observed absence — the `export.py` shape, in the
  direction that yields a classification rather than a refusal. It is latent
  today because both callers project identical aliases; a third projection
  inherits it silently. A row lacking `id` does not reach the stated refusal
  either: `record_parked_dispositions:11541` raises `KeyError: 'id'` from the
  message string itself.

**What is correct and was verified rather than assumed.**
`census_parked_dispositions` (`graph_ops.py:11641`) is sound: it re-reads the
stored property rather than the values this process computed, counts a null as
`unclassified` rather than skipping it, and its `unexpected` filter excludes
nulls only because they are already counted in `unclassified`. Its cohort is
wider than the writer's, though — it reads every row at
`attempt_count >= cap` regardless of status, while the writer stamps only rows
at `status='failed'` carrying the cap reason. So a source that has hit the cap
but has not yet been parked by a sweep tick reads as `unclassified` while the
writer is working exactly as designed. Anything gating on
`census_parked_dispositions()["unclassified"] == []` will see a false red in
normal operation.

**Verdict: SAFE-FOR-WEST for the cut, MUST-FIX for §5's own measure.** Nothing
here reaches `export.py`. But §5 lists "compose attempt cap — 216 sources parked
at attempt_count = 5" as a footgun whose silence is that "the cap reads as a
failure, so 216 nameable quantities look like physics problems", and the repair
for that row is this disposition. As landed, the disposition the writer stamps
cannot express either of the two classes that would distinguish a nameable
quantity from a permanent exclusion, the pass that could is not wired up, and a
stale value is never refreshed. The footgun is not yet removed.

## Defects, with file:line and trigger

| # | Where | Defect | Trigger | Direction |
|---|---|---|---|---|
| 1 | `loop.py:1884`, `1570` | a skipped reconcile returns a default-constructed result indistinguishable from a clean one | any `sn edit` run | misreports |
| 2 | `loop.py:2722`, `2286`, `2777` | the bypass removes the in-fence reconcile with the out-of-fence one | any `sn edit`; edited name unembedded, family signature stale | skips |
| 3 | `audits.py:41` | `answer_quarantine_question` has no production caller; the question fields are overridable defaults | `AuditReport(answers_question=QUARANTINE_QUESTION)` | guard never arms |
| 4 | `pipeline.py:1720`, `1721` | `getattr(report,"findings",[])` and `getattr(f,"affected_names",[])` name fields neither model has, so every Layer-1 finding is dropped | any review run with ≥1 lint finding | skips, silently |
| 5 | `orphan_sweep.py:162` | truthiness gate skips the disposition read, and the reconcile net it relies on has no production caller | a tick interrupted between the two writes | skips |
| 6 | `graph_ops.py:11585` + `11449` | the writer's own filter makes two of six dispositions unreachable; the parking statement destroys the original cause first | any source parked by the sweep | misreports |
| 7 | `graph_ops.py:11589` | `parked_disposition IS NULL` plus no clearing path means a re-parked source keeps a stale class | revive, re-fail, re-park | misreports |
| 8 | `graph_ops.py:11540` | the stated refusal is unreachable; an absent key is absorbed as `cause_not_recorded`; a missing `id` raises `KeyError` instead | a caller with a different projection | fails open |

Defect 4 is the only one of the eight that is live on a path someone runs today
and loses information the operator is simultaneously told exists. Defects 1, 2,
5, 6 and 7 are live but degrade triage rather than output. Defects 3 and 8 are
latent: guards that report a protection they do not provide.

## What this means for the WEST cut

**All three changes are SAFE-FOR-WEST.** None is read by `export.py`, none can
admit a wrong name or refuse a right one, and no defect above changes a name, a
description, or an eligibility decision. The WEST cut is not blocked by this
set — the blockers measured for it are the two in the export gate chain.

The shape the export audit found does recur here, three more times — `loop.py`
defect 1, `pipeline.py` defect 4, `graph_ops.py` defect 8 — in three different
files and all three in the same costume: a default value standing in for an
observation that was never made. That makes six known sites of one shape in this
sprint's local-lane work, which is the number worth carrying forward rather than
any individual defect.
