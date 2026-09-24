# Feature depth: the standard-name review and audit instrument

Revision measured: `4960f699620b7c76a73f60e0f249021a0e4e6831` (main).
Scope: every feature the plan *sn-catalog-audit-instrument* names as part of the
`sn review` / audit instrument, classified **deep** or **shallow** against the
current code — not against the plan's own claims.

## What the verdict means here

A feature is **deep** only when both hold:

1. a **real non-test call site** reaches it — the shipped command, the pipeline
   or another production module invokes it, so a run can actually exercise it; and
2. a **test exercises it**, so a change that breaks it reddens.

It is **shallow** when either half is missing:

- **no non-test caller** — the capability is fully tested but inert in the
  shipped product (a check that lives only in `tests/`); or
- **produced and discarded** — a value is built and never reaches the surface
  that consumes it, so the shipped run behaves as though the capability were
  absent while every unit of it passes its own test; or
- **unreachable** — the code path exists but no real input can make it act.

The verdict for each row carries `file:line` for the caller and the test, and a
one-line statement of the evidence behind deep or shallow.

## Reachability map

![Context keys produced by the review pipeline, and whether a rendered prompt reads them](/imas-codex/figures/sn-catalog-audit-instrument/review-context-keys.svg)

Read left to right: the pipeline builds each context key, and a key reaches the
reviewer only if a template names it. Six keys have a read edge into a template;
`audit_findings` (red) is built and then has nowhere to go. Rows 5, 7, 9 and 17
of the inventory below are the places this shape shows up.

## The inventory

| # | Feature | Module : site | Non-test caller | Test | Verdict |
|---|---|---|---|---|---|
| 1 | `sn review` three-layer command surface | `imas_codex/cli/sn.py:6031` | yes — CLI entry point | yes — `tests/standard_names/test_cli.py` | **deep** |
| 2 | Layer 1 deterministic audits (`run_all_audits`) | `imas_codex/standard_names/review/audits.py:674` | yes — `cli/sn.py:6315` | yes — `tests/standard_names/test_audits.py` | **deep** |
| 3 | Layer 2 batched LLM scoring (`run_sn_review_engine`) | `imas_codex/standard_names/review/pipeline.py` (via `cli/sn.py:6401`) | yes — `cli/sn.py:6401` | yes — `tests/standard_names/test_review_pipeline.py` | **deep** |
| 4 | Layer 3 cross-batch consolidation (`run_consolidation`) | `imas_codex/standard_names/review/consolidation.py:459` | yes — `cli/sn.py:6407` | yes — `tests/standard_names/test_review_pipeline.py` | **deep** (limit: variance-only detectors) |
| 5 | Layer 1 findings routed **into the rendered reviewer prompt** | `imas_codex/standard_names/review/pipeline.py:1714` (extract), `:1964` (context key) | extraction: yes — `pipeline.py:550`; **render: none** | extraction only — `tests/standard_names/test_audit_findings_reach_the_prompt.py` | **shallow** |
| 6 | Per-candidate comparator search (`build_neighborhood_context`) | `imas_codex/standard_names/review/enrichment.py:261` | yes — `pipeline.py:532` | yes — `tests/standard_names/test_review_catalog_comparators.py:30` | **deep** |
| 7 | Comparators carried **per item** into the prompt | `imas_codex/standard_names/review/pipeline.py:1963` (`nearby_existing_names`) | batch-level only | no per-item case | **shallow** |
| 8 | Unit-anchored comparator (same unit, different `physical_base`) | `imas_codex/standard_names/review/enrichment.py:336-345` | yes — `pipeline.py:532` | yes — `tests/standard_names/test_review_catalog_comparators.py:67` | **deep** |
| 9 | Catalog roster reaches the rendered prompt (`existing_names` → Catalog Roster block) | `imas_codex/standard_names/review/pipeline.py:958`,`:1960`; templates `imas_codex/llm/prompts/sn/review_names.md:278`, `review_docs.md:123` | yes — `pipeline.py:958` | yes — `tests/standard_names/test_review_prompt_roster.py:20` | **deep** |
| 10 | `--target groups` grouped review | `imas_codex/cli/sn.py:6192-6211` | yes — CLI branch | yes — `tests/standard_names/test_grouped_review_target.py:92` | **deep** (live-graph behaviour not established) |
| 11 | Shared `physical_base` grouping bucket | `imas_codex/standard_names/harmonize.py:392` (`include_parentless`) | no — `cli/sn.py:6210` calls `build_worklist()` with defaults | no | **shallow** |
| 12 | Grouped scoping refusal guard | `imas_codex/cli/sn.py:5951`, called `:6201` | yes — CLI branch | yes — `tests/standard_names/test_grouped_review_target.py:137` | **deep** |
| 13 | Mutation classifier respects identifier boundaries (`_cypher_mutation_clause`) | `imas_codex/cli/sn.py:46`, used `:95` | yes — report-only suppression guard | yes — `tests/standard_names/test_cypher_mutation_classifier.py` | **deep** |
| 14 | A suppressed write returns a result its caller can consume (`_SuppressedWriteResult`) | `imas_codex/cli/sn.py:62` | yes — guard installed `:6472` | yes — `tests/standard_names/test_review_write_suppression.py` | **deep** |
| 15 | report-only suppression guard on the session boundary | `imas_codex/cli/sn.py:83-153`, used `:6472` | yes — `sn review --report-only` | yes — `tests/standard_names/test_review_report_only.py` | **deep** |
| 16 | Share-then-spend comparator allocation | `imas_codex/standard_names/review/enrichment.py:362-388` | yes — `pipeline.py:532` | yes — `tests/standard_names/test_comparator_budget_share.py`, `test_review_comparator_budget.py:45` | **deep** |
| 17 | Terminal comparator cap | `imas_codex/standard_names/review/enrichment.py:401-405` | reached, but can never bind | test forces it by over-supplying a channel — `test_review_comparator_budget.py:181` | **shallow** |
| 18 | Declared-attribute static check (three-argument `getattr`) | `tests/standard_names/test_declared_attribute_access.py:444` | **none** — instrument lives only in `tests/` | yes (self) | **shallow** |
| 25 | Defaulted-attribute static check (`getattr(x, "name", default)` sites) | `tests/standard_names/test_defaulted_attributes_exist.py` | **none** — instrument lives only in `tests/` | yes (self) | **shallow** |
| 19 | Manifest-scoped release-batch selector | absent | — | — | **shallow** (named, not shipped) |
| 20 | Score a description against the bindings its identity holds | absent | — | — | **shallow** (named, not shipped) |
| 21 | Expanded dry-run report (concurrency, model, projected spend, cost-limit refusal) | `imas_codex/cli/sn.py:6328-6393` | partial — CLI dry-run branch | no | **shallow** |
| 22 | Concurrency default derived from the measured knee | `imas_codex/cli/sn.py:6098` (`default=8`) | literal default, not the knee | no | **shallow** |
| 23 | Refuse rather than score when a channel did not load | absent | — | — | **shallow** (named, not shipped) |
| 24 | `--models` override / reviewer-profile resolution | `imas_codex/cli/sn.py:6222-6236` | yes — CLI branch | yes — `tests/standard_names/test_review_model_config.py`, `test_reviewer_profiles.py` | **deep** |

## Evidence for the shallow verdicts

### Row 5 — Layer 1 findings are extracted and then never rendered

This is the row the plan calls "the single highest-value change" (route Layer 1
findings into the reviewer prompt). Half of it shipped, and the missing half is
invisible for the same reason the roster was.

- The extraction is real and correct: `_extract_audit_findings`
  (`review/pipeline.py:1714`) walks the three fields `AuditReport` actually
  declares — `lint_findings`, `link_findings`, `duplicate_components` — and
  `enrich.py` puts the result on the batch at `pipeline.py:550-551`.
- The value is placed in the render context verbatim:
  `pipeline.py:1964` sets `context["audit_findings"] = audit_findings`.
- **No template reads it.** A repository-wide search of every prompt file
  (`imas_codex/llm/`, `*.md`, `*.j2`, `*.jinja`, `*.txt`) finds no occurrence of
  `audit_findings` in any template. The two templates the review actually
  renders, `sn/review_names` and `sn/review_docs`
  (`pipeline.py:1942-1947`), read `nearby_existing_names` and `existing_names`
  but not `nearby_existing_names`'s sibling `audit_findings`.
- **Reproduction (a marker that would have appeared had the template read the
  variable):** rendering both templates with markers in three context keys gives

  ```
  sn/review_names len 32249 | roster True  | nearby True  | audit False
  sn/review_docs  len  6869 | roster True  | nearby True  | audit False
  ```

  The roster and nearby markers render; the audit marker does not. The positive
  controls prove the probe sees a read template, so the absence is specific to
  `audit_findings`, exactly the shape the plan records for the roster
  (`review_names.md`/`review_docs.md` read `existing_names` after `95366b716`).

- The test named `test_real_audit_findings_reach_the_batch_prompt`
  (`tests/standard_names/test_audit_findings_reach_the_prompt.py:12`) asserts only
  on `_extract_audit_findings`'s **return value** — it never renders a prompt — so
  it passes while the reviewer still sees nothing. The name overstates what it
  checks, which is why no gate caught the second half.

Verdict **shallow**: produced and discarded. The reviewer prompt is built
without the deterministic findings the plan exists to carry.

### Row 7 — comparators are batch-level, not per item

`build_neighborhood_context` searches **per candidate** (deep, row 6), but the
result is attached to the batch as one list — `pipeline.py:1963`
(`"nearby_existing_names": neighborhood`) — and the render context carries no
per-item comparator key (`pipeline.py:1958-1968`). The plan's "carry comparators
per item so the prompt shows each candidate the names it is actually being
compared against" is therefore not shipped: every candidate sees the batch's
pooled comparator list.

### Row 11 — the shared `physical_base` bucket is unreachable

`harmonize.build_worklist(..., include_parentless: bool = False)` gates the
parentless `physical_base` bucket (`harmonize.py:392`). The grouped-review
branch calls `build_worklist()` with no arguments (`cli/sn.py:6210`), so the
bucket never populates. `_report_drifting_families` handles the
`family.get("physical_base")` field (`cli/sn.py:6014`), so the renderer supports
a bucket the caller cannot reach — a capability present and untestable through
the shipped path.

### Row 17 — the terminal cap is a backstop no input can trip

`overall_cap = unit_cap + semantic_cap` (`enrichment.py:401`), and each channel
is internally capped at its own term (`unit_cap` at `:334`, `semantic_cap` at
`:335`). The concatenation is therefore **at most** the bound it is trimmed to,
so the break at `:405` never fires for any catalog. The in-code comment says so
(`:397-400`), and the terminal-bound test only makes it fire by supplying a
channel **above its own share** (`test_review_comparator_budget.py:181`).
Measured shallow: the guard is correct and unreachable on every real input.

### Rows 18 and 25 — the static checks are instruments with no non-test caller

Two `ast` scanners enforce the plan's attribute discipline, and **both live
entirely inside `tests/`**:

- `test_declared_attribute_access.py:444` resolves three-argument `getattr`
  calls and requires the attribute to be declared on the class;
- `test_defaulted_attributes_exist.py` scans for `getattr(x, "name", <literal
  default>)` sites and requires each to name a declared attribute, with its own
  coverage floors (`MINIMUM_SCANNED_MODULES = 11`,
  `MINIMUM_LITERAL_DEFAULTED_GETATTRS = 30`, `:43-46`) and a fixture case that
  proves a planted violation is reported (`:323`).

Neither scanner nor its class index is imported anywhere under `imas_codex/`
(searched). They are fully tested — including a negative fixture — and they are
the enforcement mechanism the plan relies on, but by this node's rule a feature
with no non-test caller is shallow: the instrument is inert outside a suite run.
The verdict records that; it does not dispute that the checks have value, and
`test_defaulted_attributes_exist.py`'s planted-violation case (`:323`) is the
model for what row 5's test should have done.

### Rows 19, 20, 23 — named by the plan, absent from the code

- **Manifest-scoped selector (row 19).** `sn review`'s option set is
  `--ids --physics-domain --stage --unreviewed --force --models --batch-size
  --neighborhood --cost-limit --dry-run --report-only --skip-audit
  --concurrency --target --reviewer-profile` (`cli/sn.py:6031-6132`). No
  manifest scope. `--batch <name>` exists on `sn run` (`cli/sn.py:1519`), not on
  `sn review`. §3's "reviews exactly the 230 distinct identities the cut would
  publish" is not shipped.
- **Description-against-bindings axis (row 20).** The rubric dimensions are the
  name four and the docs four (`pipeline.py:2226-2229`); the reviewer's
  correction slots are single strings (`revised_name` at `pipeline.py:1831`,
  `suggested_name` at `:1812`). There is no field addressed to a binding, so the
  class §3 calls unrepairable by a reader remains so.
- **Refuse-when-a-channel-did-not-load (row 23).** §1a's rule ("a review must
  refuse rather than score when a channel it depends on did not load") has no
  implementation; the embedder-unavailability fail-opens remain
  (`f-scai-embedder-unavailability-is-not-a-finding`, status open).

### Rows 21, 22 — the option exists, the measured behaviour does not

- **Dry run (row 21).** `--dry-run` runs Layer 1 and prints the cohort size and
  the batch plan (`cli/sn.py:6328-6393`). §4 requires it also to state the
  resolved concurrency, the model that would be used, the projected spend, and
  to **refuse** when projected spend exceeds `--cost-limit`. None of those four
  are present.
- **Concurrency default (row 22).** The knee on review-sized payloads was
  measured at 32 (`522524ea3`), but `--concurrency` still defaults to the
  literal `8` (`cli/sn.py:6098`) and the knee is a request count, not a batch
  count (the followup records that `--concurrency` is parallel *batches*; the
  correct default is a function of the reviewer profile). Setting it is an open
  node, so the feature is shallow today.

## Features that are deep, with a stated limit

- **Row 4, Layer 3 consolidation.** Wired (`cli/sn.py:6407`) and tested, so it
  is deep. Its detectors are variance-based (`detect_convention_drift`,
  `detect_score_outliers`), and as §1a measures, a systematic blindness scores a
  whole class alike and a variance detector reports nothing from a uniform
  column. That is a limit of the algorithm, not of its wiring — recorded, not
  folded into the verdict.
- **Row 10, grouped review.** Wired and tested, so deep; the test uses a
  `GraphClient` stand-in (`test_grouped_review_target.py:92`), so it shows the
  four printed fields are the fields `build_worklist` returns, not that a live
  graph yields families. Live behaviour belongs to the merged-result test node.
- **Row 9, the catalog roster.** Deep — this is the one fail-open of the
  produced-and-discarded shape that *has* been closed for the render half
  (`95366b716`, positive control above), which is why it is the template row 5
  should have followed.

## Gate

Focused receipt for the tests this classification cites, run from this worktree
at `4960f6996` against the main checkout's environment:

```
cd <worktree>
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD \
  uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_audits.py \
  tests/standard_names/test_audit_findings_reach_the_prompt.py \
  tests/standard_names/test_comparator_budget_share.py \
  tests/standard_names/test_cypher_mutation_classifier.py \
  tests/standard_names/test_declared_attribute_access.py \
  tests/standard_names/test_grouped_review_target.py \
  tests/standard_names/test_review_catalog_comparators.py \
  tests/standard_names/test_review_comparator_budget.py \
  tests/standard_names/test_review_model_config.py \
  tests/standard_names/test_review_pipeline.py \
  tests/standard_names/test_review_prompt_roster.py \
  tests/standard_names/test_review_report_only.py \
  tests/standard_names/test_review_write_suppression.py \
  tests/standard_names/test_reviewer_profiles.py
```

Last line of that run:

```
1 failed, 618 passed, 3 warnings in 152.61s (0:02:32)
EXIT=1
```

The command line and the whole log are recorded at
`~/.config/reckon/crew/runs/r-20260924T123309301493-n-feature-depth-is-classified-against-the-code/focus-gate.log`
(`cmd=` on line 2).

The single failure is `test_review_pipeline.py::test_audit_embedding_preflight`,
a `pytest-timeout (>30.0s)` raised while `import torch` inside
`embeddings/encoder.py:600` — an environment/environment-speed failure with no
relation to this node's change (this node adds a document and a figure; it
edits no code). It is reported under `follow_ons`, not fixed here.

The render probe above is the negative control for row 5: the same probe renders
the roster and nearby markers, so a template that read `audit_findings` would
have shown `audit True`; it shows `audit False` in both.