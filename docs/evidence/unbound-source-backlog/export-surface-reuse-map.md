# Export-surface reuse map — the two open followups, verified at HEAD

Scope: the two open followups this plan's §5 names as the live risk —
the export producer predicate (`f-usb-export-refuses-a-name-with-no-producer`)
and the manifest-generability invariant (`f-usb-a-manifest-must-be-fully-generable`).
Everything below is a read of the assigned tree at
`9705954c264bd4c51c78d68d9be23be54178c2ac`. No graph write, no code change.

The followups were written against a tree several days older, so each cited
line was re-read rather than trusted; moved references are called out where
they moved.

## 1. `f-usb-export-refuses-a-name-with-no-producer` — DOES-REPRODUCE

The exposure is not merely that the predicate is absent. **The producer
evidence is already fetched onto every candidate record and then never read by
the gate that decides eligibility** — the population query computes four
topology flags and the eligibility classifier ignores all four.

### The fetch already carries the predicate's inputs

- `imas_codex/standard_names/export.py:607` — the followup's citation is still
  exact: `name_predicate = "sn.name_stage IN ['accepted', 'approved']"`.
- `_fetch_export_population` at `imas_codex/standard_names/export.py:585`
  computes, per candidate, at `export.py:617-633`:
  `has_dd_source_binding` (an `IMASNode` points at the name),
  `has_derived_producer`, and `has_non_derived_producer`; and at
  `export.py:638-640` `is_parent`, with `export.py:642-666` the docs-review
  winner. All four land on the returned record at `export.py:678-681`
  (`_has_dd_source_binding`, `_has_derived_producer`,
  `_has_non_derived_producer`, `_is_parent`).

### The eligibility classifier never reads them

- `_classify_export_population` at `imas_codex/standard_names/export.py:756`
  has ten ordered clauses at `export.py:798-836`: `missing_physics_domain`,
  `outside_requested_domain`, `invalid_validation_status`,
  `validation_observation_missing`, `name_not_accepted`,
  `name_review_quorum_shortfall`, `never_reviewed`, `resolution_unrecorded`,
  `documentation_not_accepted`, `documentation_review_quorum_shortfall` — no
  clause mentions export of a name with no producer, and the four flags above
  appear nowhere in the function. An unproduced accepted name reaches the
  eligible list.
- Their only consumers are `_published_identity_roles`
  (`export.py:1588-1600`, called at `export.py:2826`), which tags a candidate
  `quantity` / `parent` / `grammar_token_dual` for reviewer presentation, at
  `export.py:1592-1597`. A name with no producer and no children gets
  `roles == []` and remains fully eligible — the flag list decorates the
  artifact; it does not gate it.
- The export gate inventory confirms the absence at
  `export.py:177-183` plus the inline gateway at `export.py:3085`:
  `graph_tests`, `cross_field_consistency`, `score_thresholds`,
  `divergence_detection`, `exclusion_accounting`, `identity_token_collision`,
  `catalog_status`, `manifest_source_accounting`. None is a
  producer predicate.

### The carve-out the predicate must respect, and where it moved

- The followup cites `graph_ops.py:26233-26238` for the `source_free`
  classification. That reference has **moved**: the classification is now
  `_structural_accept_route` at `imas_codex/standard_names/graph_ops.py:26690`,
  returning `"source_free"` at `graph_ops.py:26715-26716` exactly when
  `live_child_count >= 1 and origin is None and producer_count == 0`.
  `graph_ops.py:26233-26238` is now an unrelated docstring.
- The rename route the followup reports as already closed is still closed:
  `imas_codex/standard_names/edit.py:1931-1948` raises when a rename leaves an
  expected source off the new name or bound to both names.

Verdict: **DOES-REPRODUCE**. The repair is a classification addition, not a
data-plumbing job: the flags are on the record the classifier already holds.

## 2. `f-usb-a-manifest-must-be-fully-generable` — DOES-REPRODUCE

The invariant "every source in a committed manifest reaches an accepted name"
is not enforced anywhere. The resolver and the per-source disposition both
exist; what is missing is a refusal, and the mechanism classification the
refusal would report.

### What already exists

- Resolution: `fetch_manifest_source_release_rows`
  (`imas_codex/standard_names/graph_ops.py:12831-12945`) maps every manifest
  path to its terminal identity through `HAS_SUCCESSOR*0..`
  (`graph_ops.py:12893-12907`) and derives
  `non_nameable_reason = last_error or skip_cause or "cause not recorded"`
  (`graph_ops.py:12926`), cleared when a terminal or seed identity exists
  (`graph_ops.py:12927-12928`). It refuses an ambiguous binding with
  `ValueError` at `graph_ops.py:12874-12879` — a refusal shape a generability
  check can copy.
- Manifest loading/schema: `load_focus_file` /
  `load_sources_file` at `imas_codex/standard_names/sources_manifest.py:183`,
  validated against `config/sn_sources.schema.json`
  (`sources_manifest.py:42,121-124`). `build_manifest` / `write_manifest` at
  `imas_codex/standard_names/campaign.py:387,462`.
- Per-source disposition during export:
  `imas_codex/standard_names/catalog_release.py:3040-3068` classifies each
  manifest source as `documented_non_nameable`, `emitted`, or `excluded`, with
  a reason (`no_terminal_identity` / `identity_not_exported`).
- The accounting gate over those records is at `export.py:3075-3089`
  (`manifest_source_accounting`): it asserts only that the record count equals
  the manifest size. **A source recorded `excluded` passes this gate**, which
  is precisely the reporting-not-defect shape the followup names.
- Release accounting surfaces the residue without refusing:
  `report.unmatched_sources` (`catalog_release.py:1503-1504`, set at
  `:1803`), carried onto the frozen artifact at `catalog_release.py:1933-1941`
  and into the PR notes as a count at `catalog_release.py:2079`, and printed
  yellow at the CLI (`imas_codex/cli/sn.py:4901-4903`).
- One of the followup's two named residue sources has an existing repair
  route: `reconcile_source_status_liveness`
  (`graph_ops.py:12948-12977`) is the exhausted-name → `extracted`-with-fresh-
  budget transition that `calorimetry/group/component/energy_total/data`
  needs.

### What does not exist

- No manifest-generability check that a release path can refuse on. The
  closest per-source classifier is `_source_compose_hsint_decision`
  (`graph_ops.py:9929-9967`), which emits exactly the vocabulary the invariant
  needs — `attempt_cap_reached` at `graph_ops.py:9954-9955`, `eligible` at
  `:9959` — but it is keyed on a single compose-hint source id, is used for
  steering, and is never invoked over a manifest.
- No distinction on the manifest path between the three mechanisms the
  followup names (grammar/vocabulary gap, attempt cap genuinely exhausted,
  composition merely unscheduled). `fetch_manifest_source_release_rows` gives a
  free-text cause, not a mechanism.
- The `camera` locus admission (`camera_x_rays/camera/camera_dimensions`) is a
  grammar-repository decision, per the followup, and is not a code path in
  this repository.

Verdict: **DOES-REPRODUCE** as an unenforced invariant. The repair is a
classifier plus a refusal gate reusing the resolver above; the two named
sources are consequences, not the deliverable.

## 3. Reuse inventory — every existing symbol a repair draws on

| Symbol | Location | Role in a repair |
|---|---|---|
| `_fetch_export_population` | `imas_codex/standard_names/export.py:585` | already returns the four producer/topology flags |
| producer flag projections | `imas_codex/standard_names/export.py:617-633,638-640,678-681` | predicate inputs, no query change needed |
| `_classify_export_population` | `imas_codex/standard_names/export.py:756` | the clause list that gains the producer clause |
| `ExclusionRecord` | `imas_codex/standard_names/export.py:298` | one-reason drop record, reused directly |
| `_published_identity_roles` | `imas_codex/standard_names/export.py:1588` | existing `quantity` role — the reader-side half of the predicate |
| `_structural_accept_route` / `source_free` | `imas_codex/standard_names/graph_ops.py:26690,26715-26716` | the carve-out predicate (`live_child_count >= 1`, no producer) |
| `fetch_derived_parent_children` | `imas_codex/standard_names/graph_ops.py:26221` | live-child counting for the carve-out |
| `_TERMINAL_BINDING_NAME_STAGES` | `imas_codex/standard_names/graph_ops.py:63` | terminal-stage set both sides must agree on |
| `fetch_manifest_source_release_rows` | `imas_codex/standard_names/graph_ops.py:12831` | manifest path → terminal identity + reason |
| `reconcile_source_status_liveness` | `imas_codex/standard_names/graph_ops.py:12948` | exhausted → extracted repair route |
| `_source_compose_hint_decision` | `imas_codex/standard_names/graph_ops.py:9929` | `attempt_cap_reached`/`eligible` vocabulary to generalise |
| `load_focus_file` / `load_sources_file` | `imas_codex/standard_names/sources_manifest.py:183` | committed-manifest entry point |
| `build_manifest` / `write_manifest` | `imas_codex/standard_names/campaign.py:387,462` | manifest construction |
| `manifest_source_accounting` gate | `imas_codex/standard_names/export.py:3075-3089` | the accounting gate the invariant sits beside |
| `RunReport.unmatched_sources` | `imas_codex/standard_names/catalog_release.py:1503-1504,1803` | residue already recorded, never refused |
| `_freeze_review_artifact` | `imas_codex/standard_names/catalog_release.py:1933` | carries residue onto the artifact |
| `ValueError` refusal shape | `imas_codex/standard_names/graph_ops.py:12874-12879` | precedent: refuse at resolution, do not report |
| `ExclusionLedgerLinkError` | `imas_codex/standard_names/catalog_release.py:2098` | precedent for a refusing release-path error |

## 4. Proposed exclusive write-path sets

Two nodes, disjoint, sized so neither declares a file that cannot be declared.

**Node A — export producer predicate (followup A).**
```
imas_codex/standard_names/export.py
tests/standard_names/test_export_eligibility.py
tests/standard_names/test_export_exclusion_ledger.py
```

**Node B — manifest generability check (followup B).** The classifier belongs
in a **new module**, not in `graph_..._ops.py`: a repair must not declare that
file as a write path (it is enormous, and a declared path is a required
context read). The resolver it calls stays where it is.
```
imas_codex/standard_names/manifest_generability.py        (new)
imas_codex/standard_names/catalog_release.py
imas_codex/cli/sn.py                                       (report line only)
tests/standard_names/test_manifest_generability.py         (new)
tests/standard_names/test_review_release.py
```

Shared-file edits the two must not race on: `export.py` (Node A only),
`catalog_release.py` (Node B only). If a single node is used instead, it must
serialise them — the two verification files share no symbols.

**Evidence paths for both nodes** (disjoint from each other and from peers):
```
docs/evidence/unbound-source-backlog/export-producer-predicate.md
docs/evidence/unbound-source-backlog/manifest-generability.md
docs/evidence/archive/unbound-source-backlog-landed.html
docs/plans/unbound-source-backlog.html
```

## 5. What a repair owes, restated from the two verdicts

1. Node A turns the four already-fetched flags into one clause in
   `_classify_export_population`: a candidate with no producer of any kind and
   no live-child parent topology is not exportable. `beta` is the live
   instance of the carve-out and must stay exportable — which is why
   "a name with no producer and no `source_free` origin is not exportable" is
   the rule, and the simpler "every exported name carries a producer" is not.
2. Node B gives the invariant an instrument and a refusal: a
   `manifest_generability` check that reads a committed manifest through the
   existing loader, resolves every path through
   `fetch_manifest_source_release_rows`, and reports each source as carried or
   classified into one of the three named mechanisms — refusing a release
   whose residue is not empty. The three mechanisms must be distinguishable,
   because only the vocabulary gap is a reason to wait on another repository.
3. The two named residue sources are the first test cases, not the change:
   `calorimetry/group/component/energy_total/data` via a composition,
   `camera_x_rays/camera/camera_dimensions` via the grammar repository's locus
   registry, which is outside this repository's write set.

## 6. Controls and limits of this verification

Both verdicts are read from source, not from the graph, and they are claims
about *code reachability*: the flags exist and no clause consumes them, so an
unproduced accepted name reaches the candidate set regardless of how many such
names exist today. The nine-name population and the 340-of-342 manifest figure
are the followups' own measurements and were not re-measured here.

A live-graph re-measurement would strengthen the population side and would not
change either verdict. Every line above was read at
`9705954c264bd4c51c78d68d9be23be54178c2ac`; the export-side line cited at 607
is still exact, and the `source_free` reference moved to 26690-26717.