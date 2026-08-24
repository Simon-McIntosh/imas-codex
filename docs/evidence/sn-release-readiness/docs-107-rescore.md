NEEDS-HELP: the live graph contains 106 accepted catalog-edit identities with no docs-axis review, not the 107 identities required by the node contract.

# Accepted documentation rescore preflight

## Outcome

Status: **blocked before mutation**. No identity was staged, no review campaign
ran, no LLM call was made, and no graph field or relationship changed.

The node contract defines the cohort as identities that are all of:

- `name_stage = accepted`;
- `docs_stage = accepted`;
- `origin = catalog_edit`;
- no attached `StandardNameReview` with `review_axis = docs`.

That exact live predicate returns **106**, decomposed as **98 name-scored + 8
name-unscored**. The nearby predicate `reviewer_score_docs IS NULL` returns
**107**, decomposed as **99 name-scored + 8 name-unscored**. The latter exactly
reproduces the supplied 107/99/8 figures, but it is not the supplied semantic
predicate.

The extra identity is
`krypton_density_at_magnetic_axis`. It has a null docs score, but already has
**6 attached docs-axis reviews** in **3 review groups**, including canonical
scores 0.7875, 0.9000 and 0.7625, and it already carries
`docs_review_resolution_method = quorum_consensus`. Rescoring it as a
never-reviewed row would erase the distinction between a missing aggregate
score projection and absent review authority.

## Sanctioned transition found

The required docs-axis recovery mechanism does exist. Commit
`b8d69e4e989d63d61528a6552c62b6832278b3b5` introduced
`stage_docs_for_rescore()` in
`imas_codex/standard_names/graph_ops.py:15854-15955`.

Its compare-and-set gate requires an accepted name, a docs stage in
`accepted | reviewed | exhausted`, no worker claim, and no drain claim. Its
write moves docs to `drafted`, clears only aggregate docs-review decision
fields, stamps an exact `run_id`, and preserves description, documentation,
refinement depth and `StandardNameReview` history. This is explicitly described
in the function as the recovery primitive for docs whose aggregate decision
lacks sufficient reviewer authority, and it feeds ordinary `REVIEW_DOCS`.

The focused operator suite passed **3/3**:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_review_docs_stages.py::TestExactDocsRescoreStaging -q
```

Result: `... [100%]` (exit 0). No source file was edited.

## Live quantitative preflight

| Measure | Live value |
|---|---:|
| Exact never-docs-reviewed identities | **106** |
| Name-scored / name-unscored in exact cohort | **98 / 8** |
| Docs-scored / docs-resolution-present in exact cohort | **0 / 0** |
| Actively claimed identities | **0** |
| Incoming `HAS_STANDARD_NAME` bindings | **162** |
| Outgoing `HAS_UNIT` bindings | **106** |
| Outgoing `HAS_COCOS` bindings | **0** |
| StandardName count | **4,666** |
| `catalog_approved_at` / `catalog_pr_number` / `catalog_merge_commit_sha` / `exported_at` populated | **0 / 0 / 0 / 0** |
| Identities staged / reviewed | **0 / 0** |
| Docs accepted with a resolution method from this node | **0** |
| Below-bar outcomes | **0** |
| Actual spend / USD 40 cap | **USD 0.00 / USD 40.00** |
| Spend per identity | **not applicable** |
| Export eligible before / after | **not measured / not measured** |

The prompt supplied **1,947 emitted** as the current real-`run_export`
baseline. This node did not claim it as a fresh measurement and did not run a
post-campaign export because the campaign was refused before staging.

## Decision required

tried: Read live plan version 37; identified and tested the sanctioned
docs-rescore primitive; ran the exact live cohort predicate and the broader
null-score predicate; inspected the one-row difference and all its docs reviews.

options:

1. Correct the campaign scope to the **106 identities with no docs-axis review**
   and run them through `stage_docs_for_rescore()` plus ordinary `REVIEW_DOCS`.
2. Authorize the broader **107 null-docs-score identities**, while explicitly
   dispositioning `krypton_density_at_magnetic_axis` as a projection-recovery
   case rather than pretending it has never been reviewed.
3. Authorize a fresh quorum rescore of all **107** despite the existing six
   reviews on that identity, accepting the unnecessary review churn and spend.

leaning: Option 2, with 106 identities sent to quorum and the extra identity
recovered from its existing review authority through a sanctioned aggregate
projection. It preserves the semantic distinction the release ledger relies on
and still closes the full 107-row null-score class.

cost-if-wrong: Option 1 leaves one null docs-score projection unresolved and
does not meet the supplied 107 count. Option 3 clears and redraws a real prior
review outcome, temporarily removes that identity from export eligibility, and
spends quorum budget on an authority record that already exists. Choosing any
path without an explicit ruling would make the evidence internally
contradictory even if all resulting stages were terminal.

