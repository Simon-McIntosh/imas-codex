# Standard-name documentation quality reuse map

Snapshot: repository commit `90c73937d264995fb3a12399c43c75346552004b`.

This inventory maps the machinery already reachable for five documentation-quality capabilities. Each row names an implementation or test seam, cites its current location, and gives the required one-line fitness verdict. The classifications mean:

- `reusable-as-is`: the mechanism already has the right contract for this capability.
- `extendable`: the mechanism is the right integration point but needs bounded additions or stronger validation.
- `unfit`: adopting the mechanism unchanged would contradict the required quality contract or preserve a correctness defect.

## 1. Documentation quality scoring

| Candidate and citation | One-line fitness verdict |
|---|---|
| `REFERENCE_NAMES` — `imas_codex/standard_names/benchmark_reference.py:53`; dataset tests — `tests/standard_names/test_benchmark.py:21` | `extendable` — the fixed 47-path name corpus is a sound deterministic seed, but it has no catalog documentation counterparts or documentation-content labels. |
| `BenchmarkConfig`, `BenchmarkReport` — `imas_codex/standard_names/benchmark.py:34`, `imas_codex/standard_names/benchmark.py:213`; `run_benchmark` — `imas_codex/standard_names/benchmark.py:901` | `extendable` — repeatable configuration, provenance, dataset hashing, atomic output, and incremental reports are reusable, but the report needs catalog-doc baselines and explicit content-gate metrics. |
| `generate_docs_for_candidates` — `imas_codex/standard_names/benchmark.py:820`; tests — `tests/standard_names/test_benchmark.py:1587` | `reusable-as-is` — this already generates candidate docs through the production prompt, structured response model, paid seat configuration, concurrency bound, cost accounting, and failure isolation. |
| `score_with_reviewer(target="docs")` — `imas_codex/standard_names/benchmark.py:449`; tests — `tests/standard_names/test_benchmark.py:1495` | `extendable` — production-fidelity docs judging and per-dimension scores are present, but length, equation, measurement, typical-value, sign, link, tag, and validation gates must become explicit benchmark outputs. |
| `StandardNameQualityScoreDocs`, `StandardNameQualityReviewDocsBatch` — `imas_codex/standard_names/models.py:1387`, `imas_codex/standard_names/models.py:1469` | `extendable` — the typed four-axis review result is the right subjective scoring substrate, but it cannot by itself express the required objective gate vector or comparison to catalog docs. |
| `process_review_docs_batch` — `imas_codex/standard_names/review/pipeline.py:2635`, `imas_codex/standard_names/workers.py:9952` | `reusable-as-is` — existing RD-quorum dispatch, score aggregation, and docs-stage handling should remain the authority for subjective documentation review. |
| `latex_def_check` — `imas_codex/standard_names/audits.py:674` | `reusable-as-is` — the deterministic display-equation/definition audit is directly usable as one benchmark feature. |
| `_extract_links_from_docs` — `imas_codex/standard_names/graph_ops.py:408` | `reusable-as-is` — the canonical Markdown-link extractor is directly reusable for link-hygiene scoring instead of introducing another parser. |
| Benchmark and rubric tests — `tests/standard_names/test_benchmark.py:21`, `tests/standard_names/test_review_rubrics.py:24` | `reusable-as-is` — these already pin dataset shape, serialization, docs models, prompt routing, and scoring behavior and are the correct base for holdout assertions. |
| Docs content gate tests — `tests/standard_names/test_docs_review_content_gate.py:108` | `unfit` — accepting 100 repeated characters or a 25-character description proves only non-emptiness, not any required definition/equation/measurement/value/sign/link content. |
| `banned_prose_findings` — `imas_codex/standard_names/prose_policy.py:79`; benchmark use — `imas_codex/standard_names/benchmark_roles.py:1289` | `unfit` — its policy treats typical values and practical measurement material as defects, which is the opposite of the required catalog-quality benchmark when those sections are grounded. |

## 2. Richer DD context extraction into compose

| Candidate and citation | One-line fitness verdict |
|---|---|
| `_ENRICHED_QUERY_TPL` — `imas_codex/standard_names/sources/dd.py:343` | `reusable-as-is` — the library-layer query already returns DD description/documentation, unit authority, data and node types, physics domain, keywords, category, dimensions, lifecycle, IDS, cluster, parent, coordinates, errors, and COCOS fields. |
| `extract_dd_candidates(explicit_paths=...)` — `imas_codex/standard_names/sources/dd.py:549`, `imas_codex/standard_names/sources/dd.py:563` | `reusable-as-is` — explicit paths already bypass ordinary eligibility/status selection while retaining authoritative extraction and batching. |
| `extract_specific_paths` — `imas_codex/standard_names/sources/dd.py:903`; pool route — `imas_codex/standard_names/pool_adapter.py:22` | `reusable-as-is` — the targeted path route requested by the plan is already exposed through the normal worker pipeline with claim fencing. |
| `_hybrid_search_neighbours_batch`, `_related_path_neighbours`, `_enrich_batch_items` — `imas_codex/standard_names/workers.py:1849`, `imas_codex/standard_names/workers.py:2097`, `imas_codex/standard_names/workers.py:2143` | `reusable-as-is` — existing batch enrichment supplies semantic and structural neighbors without using the MCP presentation surface. |
| Compose context blocks — `imas_codex/llm/prompts/sn/generate_name_dd.md:432`, `imas_codex/llm/prompts/sn/generate_name_dd.md:468` | `reusable-as-is` — the compose prompt already renders COCOS metadata plus hybrid and related-path context, so richer extraction can flow through established fields. |
| `_DD_PATH_CONTEXT_QUERY`, `_DOCS_GEN_ENRICH_QUERY`, `_enrich_dd_path_context`, `_enrich_for_docs_gen` — `imas_codex/standard_names/workers.py:8475`, `imas_codex/standard_names/workers.py:8499`, `imas_codex/standard_names/workers.py:8541`, `imas_codex/standard_names/workers.py:8618` | `extendable` — this docs-side bridge already gathers source docs, aliases, peers, derivatives, and family context, but it is worker-local, selects a first source path, and should consume the richer shared DD context contract. |
| Context-wiring tests — `tests/standard_names/test_prompt_context_wiring.py:23`, `tests/standard_names/test_compose_names_context.py:147`, `tests/standard_names/test_docs_context_injection.py:49` | `reusable-as-is` — these tests already pin enrichment keys and their per-item prompt rendering and can be extended with the additional DD fields. |

## 3. COCOS sign-convention injection

| Candidate and citation | One-line fitness verdict |
|---|---|
| DD COCOS extraction and propagation — `imas_codex/standard_names/sources/dd.py:384`, `imas_codex/standard_names/sources/dd.py:640`, `imas_codex/standard_names/workers.py:3602` | `reusable-as-is` — the extraction path uses the canonical `HAS_COCOS` relationship and already propagates transformation labels, expressions, convention parameters, and rendered guidance into compose items. |
| `render_cocos_guidance` and data-driven templates — `imas_codex/standard_names/context.py:533`, `imas_codex/llm/config/cocos_sign_guidance.yaml:1` | `reusable-as-is` — guidance is selected by transformation type and filled from graph parameters without hardcoding a particular COCOS number. |
| `_enrich_for_docs_gen(..., cocos_params=...)` — `imas_codex/standard_names/workers.py:8618` | `reusable-as-is` — once supplied convention parameters, the docs enrichment path attaches both the transformation label and concrete sign guidance to each eligible item. |
| Docs worker COCOS parameter lookup — `imas_codex/standard_names/workers.py:9079`; canonical schema/build edge — `imas_codex/schemas/imas_dd.yaml:439`, `imas_codex/graph/build_dd.py:3027` | `unfit` — the worker queries a nonexistent `COCOS` relationship instead of `HAS_COCOS`, so the live lookup can silently return no parameters despite mock-based prompt tests passing. |
| Docs COCOS prompt blocks — `imas_codex/llm/prompts/sn/generate_docs_system.md:45`, `imas_codex/llm/prompts/sn/generate_docs_user.md:141`, `imas_codex/llm/prompts/sn/generate_docs_user.md:335` | `reusable-as-is` — the prompts already require a standalone, physically stated sign-convention paragraph only for COCOS-dependent quantities. |
| COCOS injection tests — `tests/standard_names/test_docs_cocos_injection.py:47` | `extendable` — rendering and omission behavior are well covered, but the graph mock recognizes projected values without asserting the canonical `HAS_COCOS` query shape or exercising live parameters. |
| COCOS authority prompt tests — `tests/standard_names/test_prompt_units_cocos_crossrefs.py:37` | `reusable-as-is` — these pin the single-convention authority and prevent a stale convention-specific example from re-entering the prompt. |

## 4. Documentation prompt template authoring

| Candidate and citation | One-line fitness verdict |
|---|---|
| `render_prompt` and strict prompt loading — `imas_codex/llm/prompt_loader.py:1427` | `reusable-as-is` — the Jinja loader, includes, frontmatter, schema needs, and strict rendering remain the canonical authoring surface. |
| `generate_docs_system.md` template — `imas_codex/llm/prompts/sn/generate_docs_system.md:38`, `imas_codex/llm/prompts/sn/generate_docs_system.md:172` | `unfit` — it has definition, equation, scope, sign, and link structure, but explicitly forbids typical values and generally excludes measurement content required by the live plan. |
| Shared `_docs_format.md` — `imas_codex/llm/prompts/shared/sn/_docs_format.md:62` | `unfit` — its strict normative boundary rejects representative values and restricts measurement rather than defining the required measurement/calculation and typical-values paragraphs. |
| `GeneratedDocs` — `imas_codex/standard_names/models.py:1823` | `reusable-as-is` — the structured description/documentation boundary correctly prevents documentation generation from changing name, kind, unit, or other accepted identity fields. |
| `load_compose_examples` and curated catalog — `imas_codex/standard_names/example_loader.py:269`, `imas_codex/standard_names/examples_curated.yaml:14` | `extendable` — mature typed loading and many full physics examples exist, but the selected showcase must be explicitly separated from all holdout paths and governed as the small full-example tier. |
| Existing docs exemplar/context fragments — `imas_codex/llm/prompts/shared/sn/_exemplars_enrich.md:1`, `imas_codex/llm/prompts/sn/generate_docs_user.md:43` | `extendable` — shared exemplars and sibling-family context are suitable composition points, but full catalog-quality examples and holdout-disjoint selection need an explicit contract. |
| Generation and prompt tests — `tests/standard_names/test_generate_docs.py:602`, `tests/standard_names/test_prompt_completeness.py:299` | `reusable-as-is` — these provide stable rendering and metadata/completeness checks for the rewritten template. |
| Normative-policy tests — `tests/standard_names/test_normative_documentation_policy.py:128`, `tests/standard_names/test_normative_documentation_policy.py:201` | `unfit` — they require the current prohibition and explicitly assert that measurement and typical-values paragraphs are absent, so they must be replaced by grounded-content gates. |

## 5. Name-link resolution

| Candidate and citation | One-line fitness verdict |
|---|---|
| Link lifecycle schema fields — `imas_codex/schemas/standard_name.yaml:1414`, `imas_codex/schemas/standard_name.yaml:1420`, `imas_codex/schemas/standard_name.yaml:1425` | `extendable` — status, checked timestamp, and retry count exist, but the required persisted `unresolved_links` field is absent. |
| `_compute_link_status`, `_extract_links_from_docs` — `imas_codex/standard_names/graph_ops.py:378`, `imas_codex/standard_names/graph_ops.py:408` | `reusable-as-is` — canonical extraction plus computed initial status should remain the single persistence-time interpretation of documentation links. |
| `claim_unresolved_links`, `resolve_links_batch` — `imas_codex/standard_names/graph_ops.py:8676`, `imas_codex/standard_names/graph_ops.py:8720` | `extendable` — retry and terminal-failure mechanics exist, but the claimant does not return its token and the batch update is not token-guarded, so it needs normal claim fencing before concurrent use. |
| `resolve_doc_links` — `imas_codex/standard_names/graph_ops.py:8986` | `reusable-as-is` — the idempotent reconcile already updates stale targets, cached link lists, and statuses and is a useful safety net alongside the async worker. |
| `_normalize_bare_doc_links` — `imas_codex/standard_names/graph_ops.py:8896`; tests — `tests/standard_names/test_bare_doc_link_normalize.py:50` | `reusable-as-is` — accepted-doc normalization already links resolvable bare brackets and strips dead brackets at the source boundary. |
| `_internal_link_target`, `_prune_dangling_links`, `_unresolved_computed_refs` — `imas_codex/standard_names/export.py:925`, `imas_codex/standard_names/export.py:939`, `imas_codex/standard_names/export.py:972` | `reusable-as-is` — export already parses internal targets, prunes dangling inline links, and reports computed-reference misses. |
| `_fetch_candidates` — `imas_codex/standard_names/export.py:179` | `unfit` — publish eligibility is gated on accepted docs but not on `link_status`, so unresolved or terminally failed entries can still enter the export candidate set. |
| `run_link_integrity` — `imas_codex/standard_names/review/audits.py:357` | `extendable` — it already detects unresolved, dead, and reverse-link faults, but should recognize the terminal failed state and persisted unresolved-target list. |
| Link-normalization and export-pruning tests — `tests/standard_names/test_bare_doc_link_normalize.py:50`, `tests/standard_names/test_export_link_pruning.py:23` | `reusable-as-is` — these pin both source-time hygiene and export-time defensive pruning and should remain alongside lifecycle tests. |
| CLI no-resolver tests — `tests/standard_names/test_sn_help_no_reconcile.py:31`, `tests/standard_names/test_sn_help_no_reconcile.py:55` | `unfit` — they explicitly forbid and reject `sn resolve-links`, directly contradicting the required asynchronous worker surface. |

## Quantitative coverage and implementation reuse boundary

The map covers **5/5 capability areas** with **43 candidates**: **24 `reusable-as-is`**, **11 `extendable`**, and **8 `unfit`**. Every candidate has at least one current `file:line` citation and exactly one fitness verdict.

The shortest safe implementation route is to extend the existing benchmark report rather than create a second evaluator; route shared DD query output through the existing per-item enrichment fields; fix the docs worker to query `HAS_COCOS`; rewrite the canonical docs-format include and its contradictory normative tests; and harden the dormant link worker with token fencing, the missing unresolved-target property, a CLI surface, and link-status filtering before export. The existing review quorum, prompt loader, structured docs model, COCOS renderer, link parser, normalization, and pruning machinery should remain authoritative.
