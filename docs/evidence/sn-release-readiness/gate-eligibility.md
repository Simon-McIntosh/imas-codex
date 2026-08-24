# Docs-review eligibility traversal

## Result

Docs-review eligibility is now one graph-derived determination. It traverses
`(StandardName)-[:HAS_REVIEW]->(StandardNameReview)`, restricts the review axis
to the plural value `docs`, groups by `review_group_id`, applies the
schema-derived winning method set, and selects the winning group using the same
canonical-group and stable recency ordering as the resolution mirror. The
denormalized `docs_review_resolution_method` remains a compatibility mirror,
but no presence gate reads it.

The schema admits `quorum_consensus`, `authoritative_escalation`, and
`single_review` as winning. Its complement, `max_cycles_reached` and
`retry_item`, remains non-winning. A future enum value is therefore excluded
until the schema description explicitly admits it.

## Eight consumers inspected

The live pre-change grep found eight presence tests. Their purposes and
replacements are:

| Former site | Purpose | Shared determination use |
|---|---|---|
| `export.py:306` | Full-export eligibility gate | Requires a reachable winning docs-review group. |
| `export.py:411` | Exclusion-ledger reason | Uses the projected traversal booleans and distinguishes `never_reviewed` from `resolution_unrecorded`. |
| `cli/sn.py:471` | Pending `refine_docs` count | Counts the same graph-eligible population the claim can consume. |
| `graph_ops.py:11533` | Canonical stranded docs promotion | Requires graph eligibility before promoting an above-bar reviewed document. |
| `graph_ops.py:24246` | Atomic `refine_docs` claim | Embeds the shared predicate in the existing transactional claim eligibility. |
| `graph_ops.py:6161` | Mirror candidate selection | Calls the single winning-group selector rather than selecting only rows with a null scalar. |
| `graph_ops.py:6210` | Mirror compare-and-set write | Revalidates graph eligibility and updates only when the selected value differs. |
| `graph_ops.py:6241` | Non-winning mirror repair | Clears a known schema value only when shared graph eligibility is false. |

The traversal itself is written once in `_docs_review_winner_query_body()`.
`docs_review_eligibility_where()` wraps that selector as a correlated `EXISTS`
predicate; the consumers interpolate this one generated fragment.

## Atomic claim safety

The refine-docs claim still uses `_claim_sn_atomic` without a preliminary graph
read. The shared correlated `EXISTS` predicate is passed as
`eligibility_where` and is evaluated in every seed and expansion query inside
the same Neo4j transaction. The transaction then writes the claim token and
changes `docs_stage` from `reviewed` to `refining`, and reads the token winners
back before commit. Batch ordering, token verification, stage transition,
timeout recovery, and transaction boundaries are unchanged. There is therefore
no new eligibility-to-claim race and no change to concurrent claim semantics.

## Coverage precondition

The live export query first counts every filtered review property. A missing
property in Cypher would silently count as zero, so the determination fails
closed if `review_axis`, `review_group_id`, or `resolution_method` has zero
coverage, if an axis lies outside `names`/`docs`, or if no `docs` axis exists.
The read-only census observed:

- reviews: 24,789
- `review_axis`: 24,789
- `review_group_id`: 24,789
- `resolution_method`: 10,783
- known plural axes: 24,789
- docs-axis reviews: 14,069

## Live measures

The graph changed during this node, so both the dispatch snapshot and the fresh
read-only observation are retained. The supplied verified snapshot had 2,748
docs-accepted names: 2,303 with a winning group, 435 with no docs-axis review,
and 10 with reviews but no winning method. The later live census had 2,727:
2,383 winning, 329 never-reviewed, and 15 resolution-unrecorded. This node made
no graph writes.

Among rows satisfying all other applicable work-gate conditions while the
mirror was null, the later census found:

- above the 0.85 docs bar and otherwise promotable: 0
- below the 0.85 docs bar and otherwise claimable: 6

Thus six current documents were repairable by score but invisible to both the
pending counter and refine claim solely because of denormalized absence. The
same mechanism previously stranded either side of the threshold.

The real `run_export` path at `min_score=0.85`, `skip_gate=True`,
`force=True`, and `include_sources=False` measured:

| Measure | Count |
|---|---:|
| Recorded baseline | 1,947 |
| Live pre-change export | 1,964 |
| Traversal export | 2,030 |
| Gain from live pre-change | 66 |
| Gain from recorded baseline | 83 |
| Accepted export population | 2,335 |
| `never_reviewed` exclusions | 0 |
| `resolution_unrecorded` exclusions | 12 |

The absence bucket is zero in this export population because the fresh
never-reviewed rows lie outside its accepted/approved identity universe; it is
still represented distinctly and participates in conservation. The final
ledger closed exactly: 2,030 emitted plus 305 exclusions equals 2,335. The
other exclusions were 2 below-name-score, 2 bound-adjacent, 266 documentation
not accepted, 3 invalid catalog entries, 19 invalid validation statuses, and 1
name-review quorum shortfall.

`StandardName` was 4,666 before and after the export, confirming the graph was
read only.

## Test evidence

The focused eligibility file covers schema derivation, both exclusion reasons,
identity conservation, export and CLI query contracts, stranded promotion,
the atomic claim handoff, all three mirror sites, plural-axis coverage, and a
missing-property failure.

The final complete Standard Names run produced 6,723 passed, 5 failed, 8
skipped, and 299 deselected. The five failures are stale assertions in
out-of-scope existing tests that explicitly require the removed scalar
presence expressions or the former mirror parameter names:

- two in `test_docs_resolution_mirror.py`
- two in `test_export_filters.py`
- one in `test_review_docs_stages.py`

Restoring those expressions would violate the traversal requirement. Their
exact paths and failure details are retained in the full-suite log for
coordinator-side scope assignment. An earlier run also exposed five fixture
errors caused by the coverage check opening a second client; reusing the export
query's existing client fixed those errors, and the final run confirms they are
gone.

## Logs and artifacts

- `before-export.log` and `before-export/.export_report.json`
- `live-eligibility-census.log`
- `after-export.log` and `after-export-live/.export_report.json`
- `focused-tests-after-fix.log`
- `focused-export-tests.log`
- `standard-names-suite.log`
- `standard-names-suite-after-fix.log`

All paths above are relative to the node run envelope under
`~/.config/reckon/crew/runs/r-20260824T044819667709-n-gateeligibility/`.
