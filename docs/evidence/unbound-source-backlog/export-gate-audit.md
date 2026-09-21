# The export gate chain, audited before the WEST cut

Audit of the three changes that landed in `imas_codex/standard_names/export.py`
this sprint, read for whether any of them can put a wrong name into a published
catalog or keep a right one out. Measured at `83e93c42c` in a detached worktree
against the shared project environment.

| Change | What it adds | Verdict |
|---|---|---|
| `617fdaca9` | producing-source eligibility predicate | **SAFE-FOR-WEST** as landed — see the caveat it inherits from `b161235bb` |
| `b161235bb` | fail-closed repair of that predicate | **MUST-FIX-FIRST** — bypassable in one direction, and silently empties a whole cut in the other |
| `e0c2d5ef3` | manifest generability instrument | **MUST-FIX-FIRST** — refuses the WEST manifest for sources that can never become names |

Neither of the two MUST-FIX verdicts is a wrong-admission risk. Both are
wrong-refusal risks, and both are measured below against real inputs rather than
argued.

## How the chain was exercised

```
uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_export_producer_predicate.py \
  tests/standard_names/test_manifest_generability.py \
  tests/standard_names/test_export_eligibility.py \
  tests/standard_names/test_export_exclusion_ledger.py
```

`4 failed, 21 passed` (log: `/tmp/usb-gate-audit/export-gates.log`). The four
reds are not incidental: three share one cause and the fourth is the second
finding below, so the suite is itself the reproduction for both.

## 1. `617fdaca9` — the producing-source predicate

`_has_producing_source` (`imas_codex/standard_names/export.py:777`) admits a
candidate carrying a derived producer, a non-derived producer, or a live
structural child, and the classifier refuses the rest with `no_producing_source`
(`export.py:863`).

**Correct.** The live-child projection added at `export.py:659` excludes
`['superseded', 'exhausted', 'contested']`, which is the same terminal-stage set
the rest of the repository uses for structural acceptance — verified identical at
`graph_ops.py:1876`, `graph_ops.py:8284`, `pools.py:309` and
`sn_link_guardrail.py:69`. The source-free parent it deliberately keeps admitted
is genuinely entailed by its live child.

**Can it wrongly ADMIT?** Not through its own logic. It is a pure narrowing of
the eligible set.

**Can it wrongly REFUSE?** Only in the form `b161235bb` introduced; see below.

**Can the guard be bypassed?** **Yes, and this is a real hole.** The classifier
short-circuits before reaching any eligibility clause when a candidate carries no
`name_stage` key:

```python
# export.py:826
if "name_stage" not in candidate:
    eligible.append(candidate)
    continue
```

A pre-filtered projection that omits `name_stage` is admitted **unconditionally**
— never tested for a producer, a validation observation, a domain, or a docs
review. So the very class of caller `b161235bb` set out to fail closed against (a
projection whose keys the population query never supplied) still walks straight
past the guard whenever that projection also lacks `name_stage`. The repair
closes the half of the hole where the caller supplies `name_stage` but not the
producer flags.

The production exposure is latent rather than live: `_fetch_export_population`
(`export.py:599`) is the only production producer of candidate rows and it
projects both keys, so today every real cut reaches the guard. A second
projection path added later inherits the bypass silently.

**Verdict: SAFE-FOR-WEST.** The predicate as `617fdaca9` wrote it is correct and
its residual hole is not reachable on the WEST path.

## 2. `b161235bb` — the fail-closed repair

The repair removed the `_PRODUCER_EVIDENCE_KEYS` presence test, so an absent flag
is now an absent observation rather than a licence. That is the right *form* — it
matches the catalog-status and validation-observation branches beside it.

**Can it wrongly ADMIT?** No. It strictly removes an admission path.

**Can it wrongly REFUSE?** **Yes, and the failure mode is that the entire cut
empties rather than that one name drops.** Reproduced, not predicted:

```
tests/standard_names/test_export_exclusion_ledger.py:127:
    assert report.exported_count == 1
E   AssertionError: assert 0 == 1
E    +  where 0 = ExportReport(..., exclusion_counts=
        {'invalid_validation_status': 1, 'no_producing_source': 3}, ...)
```

The fixture population carries `name_stage` but not the producer flags, so all
three otherwise-eligible candidates are refused and the cut emits **zero** names.
Two further reds in the same file are the same event seen downstream — the export
writes no `equilibrium.yml` at all (`test_export_emits_generic_source_bindings_and_preserves_accounting`,
`FileNotFoundError`), and the cross-link validator is handed an empty set where it
expected `{'electron_density', 'ion_density'}`
(`test_export_validates_cross_links_against_full_catalog`). One flag absence, three
symptoms, none of which names the cause at the surface where it is seen.

`no_producing_source` is a marker `617fdaca9` introduced, so its presence in the
ledger is positive evidence the refusal comes from this chain and not from
something older.

The concrete trigger is a candidate dict carrying `name_stage='accepted'`,
`validation_status='valid'`, a validation observation, and no
`_has_derived_producer` / `_has_non_derived_producer` / `_has_live_child` key.
Every such candidate is dropped, and because the drop is total the export still
reports itself internally consistent: the exclusion ledger closes, the accounting
gate passes, and `exported_count` is 0.

**Can the guard be bypassed?** Yes — the `name_stage` short-circuit above, which
this change did not close.

**Verdict: MUST-FIX-FIRST.** Not because the refusal is wrong in principle, but
because a projection defect now presents as a silently empty catalogue rather
than as an error. A cut that emits zero names must refuse itself; the
`exclusion_accounting` gate does not, since an empty cut accounts perfectly.
Fixing the four reds by re-opening the predicate would restore the original
fail-open and must not be done — the fixtures are what is wrong, plus the missing
empty-cut refusal.

## 3. `e0c2d5ef3` — the manifest generability instrument

`describe_manifest_generability` (`export.py:2446`) accounts every manifest
source as carried or blocked, and `GATE_MANIFEST_GENERABILITY` (`export.py:3269`)
carries `generable` into `all_gates_passed`, which is what the release paths
refuse on.

**Correct in structure.** The carried and blocked lists are exhaustive over the
disposition records, the gate is raised only inside `if manifest_sources is not
None:` (`export.py:3180`) so a manifest-free cut is not asserted generable, and
the reverted control in `tests/standard_names/test_manifest_generability.py`
shows the instrument fires.

**Can it wrongly ADMIT?** No. `_manifest_source_mechanism` returns `None` only
for `disposition == "emitted"`, which is set only when the identity is in
`exported_ids`.

**Can it wrongly REFUSE? Yes — and it does so on the actual WEST manifest.**
Running the real verdict over
`imas_codex/standard_names/manifests/reviews/v0.4.0rc7+west-task-2e.sn_names.yaml`
(355 sources):

```
manifest_size 355  carried 330  uncarried 25  generable False
mechanism_counts {'recorded_refusal': 21, 'composition_not_scheduled': 4}
```

The gate fails, so `all_gates_passed` is False and every release path refuses the
cut. Two of those 25 can never be repaired by any amount of pipeline work:

| source_path | reason recorded |
|---|---|
| `equilibrium/time_slice/constraints/b_field_pol_probe/weight` | `dd_node_category_ineligible: Backing DD node category fit_artifact ...` |
| `equilibrium/time_slice/constraints/flux_loop/weight` | `dd_node_category_ineligible: Backing DD node category fit_artifact ...` |

A fit artifact is not a physical quantity — §1 of this plan lists that class as
**correctly excluded**. The instrument has no waived or acknowledged category, so
a manifest containing a correctly-excluded source is permanently ungenerable and
the gate is unsatisfiable except by removing the source from the manifest.

The same shape reproduces in the suite:

```
tests/standard_names/test_export_exclusion_ledger.py:541:
    assert report.all_gates_passed
E   AssertionError: assert False
E    +  where False = ExportReport(... 'detail': 'non_nameable_coordinate: time axis'}]}).all_gates_passed
```

A time axis is a coordinate. The test whose name asserts that non-nameable
sources *reconcile* now fails because the gate treats reconciliation as a defect.

**Two further defects in the mechanism vocabulary**, both misattribution rather
than admit/refuse:

- **`refusal_cause_not_recorded` is declared and unreachable.** It is listed at
  `export.py:2410` and returned nowhere. `_manifest_source_mechanism`
  (`export.py:2438`) maps a `documented_non_nameable` row whose reason is the
  `"cause not recorded"` sentinel to **`composition_not_scheduled`** — the
  opposite class. The discriminator is already on the record:
  `SourceDispositionRecord.source_status`, which `e0c2d5ef3` added and carries
  into the report but which `_manifest_source_mechanism` never reads, despite its
  docstring claiming mechanisms derive from "the source's own lifecycle status".
  Concrete trigger: a source with `source_status='skipped'` and no surviving
  `last_error` / `skip_reason` / `skip_reason_detail`. Per §5 of this plan
  **397 such rows exist in the live graph**; `graph_ops.py:12988` gives every one
  of them the `"cause not recorded"` sentinel. Any manifest drawing one in reports
  a refusal as unscheduled work. The four WEST rows that hit this branch happen to
  carry `source_status='extracted'`, so WEST is labelled correctly by luck of the
  population, not by the predicate.

- **`attempt_budget_exhausted` is under-reached.** It fires only on
  `terminal_stage == "exhausted"` (`export.py:2433`), i.e. only when the search
  reached a name and that name exhausted. A source whose compose attempts were
  capped before ever reaching an identity carries the named constant
  `_COMPOSE_CAP_REASON = "compose claim-attempt cap reached"`
  (`graph_ops.py:11399`) in its reason and no terminal stage, so it falls through
  to `recorded_refusal`. Concrete trigger, present in WEST:
  `gas_injection/valve/flow_rate`, reason `compose claim-attempt cap reached`,
  reported as `recorded_refusal`.

Both defects collapse exactly the distinction the docstring says the vocabulary
exists to preserve — "a recorded refusal, a search that spent its attempt budget,
and a composition nothing has scheduled". A release reads `recorded_refusal: 21`
and concludes it is waiting on a grammar vocabulary the other repository must
close; at least one of those 21 is a spent search that waits on nothing, and two
are permanent exclusions that will never close.

**Can the guard be bypassed?** Only by passing `manifest_sources=None`, which
`catalog_release.py:1879` does not do for a manifest-driven cut. Not a practical
bypass. A duplicated `source_path` is dropped from the disposition records
(`export.py:3190`) and so escapes the generability verdict and shrinks
`manifest_size`, but the `manifest_source_accounting` gate catches the duplicate
independently, so the cut still refuses.

**Verdict: MUST-FIX-FIRST.** The instrument is the right idea and closes a real
exposure, but as landed it cannot pass on the WEST manifest for reasons WEST
cannot act on, and the mechanism it reports for the sources it blocks is wrong on
two of the three classes it was built to separate.

## Defects, with file:line and trigger

| # | Where | Defect | Trigger | Direction |
|---|---|---|---|---|
| 1 | `export.py:826` | `name_stage`-absent short-circuit skips every eligibility clause including the producer guard | any projection lacking `name_stage` | wrongly ADMITS |
| 2 | `export.py:777` + `export.py:863` | producer refusal can empty a whole cut with no gate refusing an empty cut | candidate with `name_stage` and no producer keys | wrongly REFUSES |
| 3 | `export.py:2410`, `2438` | `refusal_cause_not_recorded` unreachable; a lost-cause skip is reported as `composition_not_scheduled`; `source_status` never read | `source_status='skipped'` with no surviving cause (397 live rows) | misattributes |
| 4 | `export.py:2433` | `attempt_budget_exhausted` fires only on a terminal identity, so a capped search with no identity reads as a refusal | `gas_injection/valve/flow_rate` in the WEST manifest | misattributes |
| 5 | `export.py:2422` | no waived category, so a correctly-excluded source makes a manifest permanently ungenerable | the two `fit_artifact` rows in the WEST manifest | wrongly REFUSES |

Defect 1 is the only wrongly-ADMITS entry and it is latent on the WEST path;
defects 2 and 5 are live.

## What this means for the WEST cut

The WEST manifest **cannot pass the gate chain as it stands**: measured
`generable False`, 25 uncarried of 355, and two of the blockers are permanent
exclusions rather than outstanding work. The minimum before a cut is a waived
disposition for a source the pipeline has correctly refused forever, so that the
gate refuses on work outstanding and not on work that will never exist.
