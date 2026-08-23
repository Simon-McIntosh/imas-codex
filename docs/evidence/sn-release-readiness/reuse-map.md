# Standard-name release-readiness reuse map

## Scope and evidence bases

This is a source-level reuse investigation for the six implementation capabilities fixed by the live `sn-release-readiness` plan, section 2a. It is not evidence that any capability is already implemented or that the live graph invariant currently passes.

- `imas-codex` dispatched base: `27f28bc773f1012e8e3e228f36ba1e05870311d3`
- coupled `imas-standard-names` base: `1f3d7224549253d492bcd56f21f22628954c735f`
- priced holdout origin: `e42bd4683ae84dba180929b832dbefe7e73cb0dc`, verified as an ancestor of the dispatched base; the two relevant files are unchanged between that commit and the dispatched base.
- Search surface: capability-shaped searches over `imas_codex/`, `tests/`, `imas_standard_names/`, and the coupled repository's `tests/`. The long search results are retained with the worker delivery.

Verdict meanings:

- **reusable-as-is**: the named unit can be called or retained without changing its semantics.
- **extendable**: the mechanism is sound and materially reduces the implementation, but the new capability needs an explicit extension.
- **unfit**: superficially adjacent, but using it as the capability would preserve the defect or enforce the wrong semantics.

## Quantitative inventory

| Capability area | Candidate count | Reusable as-is | Extendable | Unfit |
|---|---:|---:|---:|---:|
| Refuse writes that null pipeline review provenance | 3 | 0 | 2 | 1 |
| Aggregate per-reason export exclusions | 5 | 0 | 4 | 1 |
| Record child-naming structural-entailment authority | 6 | 1 | 3 | 2 |
| Assert the accepted-row graph invariant from schema | 3 | 1 | 2 | 0 |
| Backfill a null reviewer scalar from review edges | 4 | 1 | 3 | 0 |
| Project review-campaign cost before spending | 4 | 2 | 2 | 0 |
| **Total** | **25** | **5** | **16** | **4** |

## Candidate map

### 1. Refuse a write that would null pipeline-authoritative review provenance — 3 candidates

The required guard is stronger than editorial-field protection: it must reject, not silently strip, a write that clears a pipeline-owned same-axis score/model, terminal review edge, or structural authority. It must also fail closed when authority cannot be read.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| Catalog-owned field registry and batched protection filter | `imas-codex:imas_codex/standard_names/protection.py:19` | **extendable** | Reuse the explicit ownership registry, pre-fetched protected-name set, and non-mutating batch shape. Extend it with pipeline-authoritative fields and relationships, compare-before-write semantics, and a hard refusal. The current function strips fields and returns the reduced payload; it does not refuse the attempted write. |
| Catalog reconciliation scalar allow-list | `imas-codex:imas_codex/standard_names/catalog_reconcile.py:42` | **extendable** | The allow-list is a safe write boundary: reconciliation can only set description, documentation, and unit and therefore cannot clear review provenance through this query. Reuse the positive allow-list pattern, but add an explicit payload/precondition guard to every catalog merge/reconcile write surface so an attempted review-provenance mutation is visible and rejected. |
| Removal lock for the destructive bulk-import surface | `imas-codex:tests/standard_names/test_no_bulk_import.py:23` | **unfit** | Keep this regression test, but it proves only that the known node-recreating import API remains absent. It neither covers current merge/reconcile writers nor asserts preservation of review properties and edges. |

Recommended composition: centralize the pipeline-owned provenance vocabulary beside `PROTECTED_FIELDS`, but expose a separate `refuse_pipeline_authority_loss(before, proposed)` guard. The guard should reject nulling or replacement of authoritative scalars and reject deletion/replacement of `HAS_REVIEW` or structural-authority relationships unless a signed, capability-specific repair authority admits the exact transition. A graph-read failure is a refusal, not permission.

### 2. Aggregate per-reason exclusion counts across the export filtering pipeline — 5 candidates

The current report has several counters, but the accounting universe begins after graph-side eligibility filtering. A complete ledger needs one classified exclusion record per input identity, then derives counts from that ledger; otherwise the report can reconcile internally while silently omitting pre-fetch exclusions.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| `ExportReport` count serialization | `imas-codex:imas_codex/standard_names/export.py:122` | **extendable** | Reuse the report and JSON boundary. Replace independently incremented scalars as the authority with counts derived from an identity-bearing exclusion ledger. The documented domain counter is permanently zero because filtering occurs before the report sees candidates. |
| Gate-C reason-bearing issues and score counters | `imas-codex:imas_codex/standard_names/export.py:415` | **extendable** | Gate C already creates per-item issue types such as placeholder and below-description-score while returning aggregate score/unreviewed counts. Normalize those issues into the common exclusion record and derive every Gate-C count from them. |
| Graph-side eligibility query | `imas-codex:imas_codex/standard_names/export.py:179` | **unfit** | As-is it returns only eligible rows; rejected lifecycle, validation, quorum, docs-stage, batch, and domain identities disappear before accounting. It must become a classified eligibility projection, or be paired with a complementary query over the exact input cohort that emits a reason for every excluded identity. |
| Existing grouped skip-reason query | `imas-codex:imas_codex/standard_names/graph_ops.py:11023` | **extendable** | Reuse the `coalesce(reason, 'unknown')` plus `count(*)` aggregation pattern and deterministic reason ordering. Its current population is skipped `StandardNameSource` rows, not export candidates, so it cannot be called directly for export accounting. |
| Arithmetic closure regression | `imas-codex:tests/standard_names/test_export_manifest_accounting.py:69` | **extendable** | Retain the closure assertion, but redefine `total_candidates` as the pre-filter cohort and assert both identity conservation and arithmetic closure across all graph-side and in-process reason buckets. The current synthetic test proves only that manually assigned post-fetch counters add up. |

Recommended record shape: `{standard_name_id, stage, reason, detail}` with a closed reason vocabulary. Require exactly one terminal outcome per input identity: exported or excluded once. Report `counts_by_reason` by aggregating these records and retain the identity list for audit. Link pruning is a field-level mutation, not a candidate exclusion, and should remain a separate metric rather than being counted as an excluded name.

### 3. Record structural-entailment authority naming child rows — 6 candidates

**Direct answer on the preceding integrity closure:** no class named `StandardNameAuthority` exists at either inspected base. An equivalent generic authority-record family does exist in `imas-codex`: `RepairAuthorityRow` and `RepairAuthorityArtifact`, introduced by the preceding integrity work, together with a signed builder and a source-disposition path that consumes signed structural legitimacy authority. That is reusable infrastructure, not an already-existing durable per-name structural-entailment record. The current derived-parent accept writes only `reviewer_model_name='structural-inheritance'`; the live plan explicitly rejects that marker as authority.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| Typed repair authority row/envelope | `imas-codex:imas_codex/schemas/standard_name.yaml:3948` | **extendable** | Reuse typed row identity, participants, signatures, selection, mutations, guards, and orphan policy. Define the structural-entailment specialization so its target is one accepted derived name and its participants name the exact ordered child rows and their stable identities/digests. A repair artifact alone is not a durable graph relationship from the accepted row. |
| Signed canonical authority builder | `imas-codex:imas_codex/standard_names/repair_authority.py:135` | **extendable** | Reuse closed selection, unique ordered row IDs, canonical serialization, schema validation, and payload/file digests. Extend the builder or add a sibling typed builder for entailment-specific participant validation and target/child closure. |
| Existing structural-authority admission and refusal path | `imas-codex:imas_codex/standard_names/graph_ops.py:18412` | **extendable** | Reuse the signed-authority validation, live-child closure, exact target membership, refusal accumulation, and compare-and-set transaction pattern. This path authorizes source-disposition repairs; it does not persist an authority record for a derived name's initial acceptance. |
| Derived-parent child enumerator | `imas-codex:imas_codex/standard_names/graph_ops.py:25004` | **reusable-as-is** | It already fetches the live `HAS_PARENT` child set in deterministic child-ID order with the grounding fields used by enrichment. Use its output as the participant set, then digest the child identities and relevant immutable entailment inputs before accepting the parent. |
| Derived-parent accept writer | `imas-codex:imas_codex/standard_names/graph_ops.py:25051` | **unfit** | This is the exact integration point but not an authority implementation: it accepts the parent and writes only the scalar marker, timestamp, and model. Extend/replace its write transaction so acceptance and the child-naming authority record are atomic; the marker may remain descriptive but must not satisfy the invariant. |
| Coupled library descendant traversal | `imas-standard-names:imas_standard_names/graph/local_graph.py:333` | **unfit** | This traversal follows grammar ordering edges (`HAS_ARGUMENT`, `HAS_ERROR`), not the codex `HAS_PARENT` derived-parent grounding relation, and returns only descendant names without signed row identities or an authority record. It demonstrates that the coupled repo has graph traversal but no equivalent review/entailment authority. |

Minimum durable authority content: stable authority ID; accepted parent ID; ordered child IDs; child-row digests or immutable identity/version anchors; entailment mechanism; creation timestamp; code/schema identity; signed payload digest; and a graph relationship from the accepted parent to the authority, with authority-to-child relationships or an equivalently queryable typed participant projection. The accept transaction must fail when the child set is empty, changes under compare-and-set, or differs from the signed participant set.

### 4. Assert the graph invariant over accepted rows in a schema-driven test — 3 candidates

Invariant fixed by the live plan: every accepted standard name has either a non-null same-axis reviewer score backed by terminal name-axis review authority, or a structural-entailment authority record. A bare reviewer-model marker satisfies neither branch.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| Schema-derived required-field compliance loop | `imas-codex:tests/graph/test_schema_compliance.py:224` | **extendable** | Reuse schema enumeration, relationship-slot skipping, lifecycle scoping, and consolidated violation reporting. The invariant is disjunctive and spans properties plus relationships, so it cannot be represented as one ordinary required scalar without a schema annotation or named invariant specification. |
| Existing accepted-row graph-quality suite | `imas-codex:tests/graph/test_sn_graph.py:137` | **extendable** | Reuse the graph-test location, accepted lifecycle scope, quantitative failure messages, and live-corpus execution. Replace hard-coded authority assumptions with label/property/relationship names resolved from the standard-name schema and return violating IDs, not only a count. |
| Shared schema and live-graph fixtures | `imas-codex:tests/graph/conftest.py:75` | **reusable-as-is** | The session graph client and `GraphSchema` import are suitable. Add the standard-name schema to the schema fixture path rather than constructing a second client or embedding an undeclared relationship name in the test. |

The test should query all `name_stage='accepted'` rows, count each branch separately, and assert `accepted = reviewed_authority + structural_authority` with zero overlap errors and zero residual identities. It should separately assert that `reviewer_model_name='structural-inheritance'` without the authority relationship is a violation. Schema drive should cover the node label, lifecycle property, review relationship, authority relationship, and relevant axis/canonical fields; it should not merely load a schema while hard-coding the query vocabulary.

### 5. Backfill a null reviewer scalar from existing review edges — 4 candidates

Representative live-plan identities make the axis rule concrete: `iron_density_at_plasma_boundary` has 3 name-axis reviews, including 1 canonical, plus 10 docs reviews; `toroidal_deuterium_tritium_velocity_at_plasma_boundary` has 5 name-axis reviews, including 2 canonical, plus 18 docs reviews. These are projection defects, not review work. The backfill must filter `review_axis='names'`, prefer the canonical resolution, and never take `max(score)` across axes.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| Canonical same-axis review projection | `imas-codex:imas_codex/standard_names/review/projection.py:166` | **extendable** | Reuse the explicit axis parameter and escalator/quorum/single resolution semantics. It is read-only and selects the lexicographically greatest review-group ID rather than directly preferring `is_canonical`; extend the selection to honor the graph's canonical marker/terminal authority and add a compare-and-set scalar writer that only fills nulls. |
| Review-node and `HAS_REVIEW` persistence schema | `imas-codex:imas_codex/standard_names/graph_ops.py:5853` | **extendable** | Reuse the established review fields (`review_axis`, `is_canonical`, group, cycle, resolution, score, model) as the authoritative input shape. Do not call this writer to manufacture reviews; the task reads existing edges and projects the scalar. |
| Typed `recompute_projection` repair mutation | `imas-codex:imas_codex/schemas/standard_name.yaml:796` | **reusable-as-is** | Use this existing mutation kind in the signed repair authority/receipt for the zero-cost backfill. The applied row should identify the source review authority and expected prior null scalar. |
| Offline canonical-projection test matrix | `imas-codex:tests/standard_names/test_review_projection.py:81` | **extendable** | Reuse its mocked-query branch coverage. Add canonical-marker preference, axis-confusion rejection, null-only compare-and-set, idempotent replay, and refusal when multiple terminal authorities remain ambiguous. |

Safe write shape: match the accepted standard name with a null `reviewer_score_name`; match only attached `StandardNameReview {review_axis: 'names'}` rows; resolve one authoritative result; set score/model/timestamp under an expected-null compare-and-set; emit a signed recompute-projection receipt naming the review rows used. A pre-existing non-null scalar, missing authority, cross-axis-only authority, or ambiguous canonical authority is a refusal.

### 6. Project a review campaign's cost before spending — 4 candidates

**Direct answer on `e42bd468`:** yes, its priced-admission projector can price its own documentation-generation holdout arm without issuing model calls. `_price_generation` renders every request and calls a pure exposure estimator; `evaluate_docs_holdout(dry_run=True)` returns projected call count/cost with actual cost zero, and the ceiling check occurs before generation. No, it cannot price the required review campaign as-is: it renders documentation-generation prompts, uses the `GeneratedDocs` response schema, and assumes one generation request per candidate. It does not model the name-review seat count, retries/quorum/escalation, or refinement calls of a review campaign.

| Candidate | Citation at base | Verdict | Reuse boundary |
|---|---|---|---|
| Priced holdout admission projector introduced at `e42bd468` | `imas-codex:imas_codex/standard_names/docs_holdout_eval.py:100` | **extendable** | Reuse the order of operations: render the exact request, price every request, compare with a ceiling, expose a dry-run report, and only then allow dispatch. Parameterize the prompt, response model, provider attempts, seat/cycle multiplicity, and exact review cohort. |
| Network-free no-call and over-ceiling tests | `imas-codex:tests/standard_names/test_docs_holdout_eval.py:43` | **reusable-as-is** | Retain these tests for the holdout and clone the assertion pattern for campaign admission: projected cost is positive, actual cost is zero, and model-call count remains exactly zero for both dry-run and refused-over-ceiling paths. |
| Rendered-request provider exposure estimator | `imas-codex:imas_codex/standard_names/budget.py:298` | **reusable-as-is** | This is the correct pure pricing primitive once supplied the exact review messages, structured response model, bounded provider attempts, and call multiplicity. It fails closed for unknown rates, unbounded calls, non-text dimensions, empty requests, and context overflow. |
| Six-pool pipeline remaining-cost model | `imas-codex:imas_codex/standard_names/cost_model.py:204` | **extendable** | Reuse disjoint-bucket flow and explicit review/refine pools for a broader campaign estimate. Do not use `resolve_pool_cpi`'s zero last-resort fallback as admission authority: every paid review/refine pool in the campaign must have a positive proven CPI or exact rendered-request price. |

Recommended campaign projector: accept the exact identity cohort, planned review axis, seat/cycle policy, model routes, prompt context, response schema, and retry bounds; render and price the initial reviews plus explicitly bounded escalation/refinement exposure; return per-pool calls and USD plus a total; fail closed on an unpriced route; and perform the authorization comparison before constructing any dispatch task. For the plan's measured 1,179-name rescore cohort, the historical estimate of roughly USD 122 before refinement is useful calibration, not admission authority; the exact pre-call projector must expose refinement and retry assumptions against the USD 200 ceiling.

## Lowest-risk implementation sequence

1. Declare the structural-entailment authority class/relationships and the accepted-row disjunction in the standard-name schema.
2. Add the schema-driven invariant test first; it should visibly fail on bare structural markers and on the two null scalar projections.
3. Implement the signed child-set authority and make derived-parent acceptance plus authority persistence atomic.
4. Add the null-only, same-axis canonical projection backfill under signed `recompute_projection` authority; verify the two representative identities without issuing reviews.
5. Add the centralized pipeline-provenance loss guard to every catalog writer and retain the bulk-import removal test.
6. Refactor export filtering into an identity-bearing outcome ledger, derive per-reason counts from it, and assert full pre-filter cohort closure.
7. Generalize the priced holdout admission pattern into an exact review-campaign projector, then dry-run and authorize the campaign before any model task is created.

## Search conclusion

The implementation is predominantly composition, not invention: 21 of 25 candidates are reusable or extendable. The two largest semantic gaps are both authority gaps, not helper-function gaps: there is no durable child-naming structural-entailment record, and export eligibility drops identities before the current report's accounting universe begins. The coupled `imas-standard-names` repository supplies grammar/catalog graph traversal but intentionally does not supply review provenance, signed repair authority, export filtering, or model-cost admission; those capabilities remain owned by `imas-codex`.
