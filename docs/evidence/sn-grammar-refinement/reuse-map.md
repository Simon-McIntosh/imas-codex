# Closed-vocabulary and spelling-migration reuse map

## Outcome

The existing stack already contains the primitives needed for both changes, but not every aggregate is fit unchanged.

- An ISN token enters through the segment-to-vocabulary ownership declared in `imas_standard_names/grammar/specification.yml`; `GrammarSpec` then feeds generated enums/constants and grammar-reference documentation. Typed YAML registries get strict Pydantic validation at load time, while flat-list vocabularies do **not** have an equivalent universal load-time schema. Parser, uniqueness, drift, and explicit round-trip tests therefore remain necessary.
- A stored spelling can be migrated deterministically with the public lossless `parse()` result and `compose()`, then landed through imas-codex `apply_edit(..., rename=..., scope="only_self")` / `persist_refined_name()`. That path creates the successor, records lineage and provenance, supersedes the predecessor, and migrates authoritative source bindings atomically.
- A source-manifest release currently widens its batch beyond source-bound identities: `run_review_release()` unions terminal manifest identities with `mint_sn_list()`, whose contract deliberately includes immediate family. For a cut whose *new batch* must contain only identities bound to its own manifest paths, keep the loader, terminal projection, export eligibility, and approved-baseline assembly; replace that union with a fail-closed exact terminal-identity cohort. The already-approved catalog baseline is still retained by design and is not part of the new batch.

## Evidence boundary and snapshots

The live plan is the semantic authority: `imas-codex/docs/plans/sn-grammar-refinement.html:155-186` groups the closed-vocabulary gaps, `:202-216` requires shadow parse/re-render followed by provenance-bearing rename, and `:222-224` requires the catalog to consume the grammar by package pin rather than publishing a duplicate segment decomposition.

Read-only source snapshots:

- assigned imas-codex worktree: `d474ba8349b9b0ec01d5f9d4a39928b399bff516`
- `/home/ITER/mcintos/Code/imas-codex`: `e2a87d8eed14c3158e38d9e9716a6557fcd37026`
- `/home/ITER/mcintos/Code/imas-standard-names`: `12b557363ab0f2a6ed55bdaf619d3bc02a7e1e14`
- `/home/ITER/mcintos/Code/imas-standard-names-catalog`: `756a9a11d1981a0a846b91995f42248c0b685301`

The current imas-codex checkout and assigned snapshot have no content difference in the inspected implementation and test subtrees. The exact command and empty result were:

```text
git diff --name-only d474ba8349b9b0ec01d5f9d4a39928b399bff516 e2a87d8eed14c3158e38d9e9716a6557fcd37026 -- imas_codex/standard_names imas_codex/cli/sn.py tests/standard_names
<no output>
```

No test suite or service was run or queried; this node was explicitly read-only.

## First question: how a token enters the ISN closed vocabulary

### Segment owner map

`imas_standard_names/grammar/specification.yml:146-444` declares the segments and their vocabulary keys; `:459-500` resolves those keys to the included YAML files. `physical_base` is the one special case: it names `physical_bases.yml` directly at `:357-367` rather than carrying a `vocabulary:` field.

| Segment class | Vocabulary key | Owning YAML under `imas_standard_names/grammar/vocabularies/` |
|---|---|---|
| `component` | `components` | `components.yml` |
| `coordinate` | `coordinate_axes` | `coordinate_axes.yml` |
| `section_plane` | `section_planes` | `section_planes.yml` |
| `aggregation` | `aggregations` | `aggregations.yml` |
| `qualifier` | `qualifiers` | `qualifiers.yml` |
| `zone` | `zones` | `zones.yml` |
| `orbit` | `orbits` | `orbits.yml` |
| `population` | `populations` | `populations.yml` |
| `subject` | `subjects` | `subjects.yml` |
| `state` | `states` | `states.yml` |
| `device` | `objects` | `locus_registry.yml` |
| `channel_qualifier` | `channel_qualifiers` | `channel_qualifiers.yml` |
| `channel` | `channels` | `channels.yml` |
| `geometry_representation` | `geometry_representations` | `geometry_representations.yml` |
| `geometric_base` | `geometric_bases` | `geometry_carriers.yml` |
| `physical_base` | direct loader | `physical_bases.yml` |
| `object` | `objects` | `locus_registry.yml` |
| `geometry` | `positions` | `locus_registry.yml` |
| `position` | `positions` | `locus_registry.yml` |
| `region` | `regions` | `regions.yml` |
| `path` | `positions` | `locus_registry.yml` |
| `process` | `processes` | `processes.yml` |

There are 22 declared segment IDs. Exact count:

```text
rg -c '^  - id:' imas_standard_names/grammar/specification.yml
22
```

This map is the safe routing mechanism for all plan inputs. The plan fixes `relative_humidity` as a `physical_base`, so its owner is unambiguously `physical_bases.yml`. The other requested tokens must be placed according to the approved IR segment role (for example projection versus coordinate versus qualifier), not according to a substring resemblance; no source inspected provides license to guess that semantic classification.

### Load-time and use-time validation

There is no single uniform token validator at YAML load time:

- `imas_standard_names/grammar/vocab_loaders.py:1-13` documents the strict Pydantic loaders. `CoordinateAxisDef` / `CoordinateAxesRegistry` and `load_coordinate_axes()` are at `:43-61`; `PhysicalBaseDef` requires a literal base kind and forbids extra fields at `:167-207`; the locus, operator, and geometry-carrier registries follow the same typed pattern. Bad shapes, unknown fields, and invalid literal values fail during these typed loads.
- `_AllRegistries._no_duplicate_names_across_registries()` and `validate_no_cross_registry_duplicates()` at `vocab_loaders.py:507-555` reject token collisions across the typed coordinate-axis, locus, operator, physical-base, and geometry-carrier registries.
- Flat vocabularies are weaker. `load_qualifiers()` at `vocab_loaders.py:252-269`, representative of this path, accepts a list or keyed mapping, stringifies entries, and otherwise returns an empty set. `GrammarSpec.load()` at `grammar_codegen/spec.py:102-117` expands includes and extracts tokens, but is not a per-token semantic schema. Extending strict load-time checking to every flat vocabulary is therefore useful hardening, not a prerequisite already supplied by the stack.
- The public `parse()` at `grammar/parser.py:2055-2105` rejects a non-lowercase-snake-case spelling at `:2074-2079` and, with `strict=True`, is the authoritative closed-vocabulary and semantic oracle at `:2061-2068`. That validates a token *in a name*, not an isolated flat-list entry at YAML load.

### Generated and documentation dependants

- `grammar/specification.yml:5-16` identifies the spec as source of truth and names the generated modules.
- `grammar_codegen/generate.py:52-58` targets `grammar/model_types.py`, `grammar/constants.py`, `grammar/tag_types.py`, and `grammar/field_schemas.py`. `_enum_definitions()` at `:443-468` emits vocabulary enums. `_segment_metadata()` / `_render_segment_metadata()` at `:471-545` emits segment ordering and `SEGMENT_TOKEN_MAP`; physical bases are populated from `load_physical_bases()` at `:533-536`.
- `_run_check_mode()` at `grammar_codegen/generate.py:250-312` renders the expected generated files and exits on drift. `hatch_build_hooks.CustomBuildHook.initialize()` at `hatch_build_hooks.py:16-38` invokes generation for package builds.
- `docs/grammar_macros.define_env()` and `grammar_vocabulary_table()` at `docs/grammar_macros.py:20-64` load `GrammarSpec` and render vocabulary tables; `grammar_segment_rules_table()` at `:101-126` renders segment metadata. A token addition therefore reaches generated API/context and grammar-reference docs from the same YAML source.
- The catalog repository pins the stable parser package in `imas-standard-names-catalog/pyproject.toml:17-23`. Its site workflow derives and checks out the matching tag at `.github/workflows/catalog.yml:83-96`. A grammar-rendered name cannot safely publish to that site until this pin is bumped to the release containing the token/rule; no per-name segment sidecar is needed.

### Existing tests and what they prove

- `tests/grammar/test_vocab_loaders.py:30-226` exercises typed registry construction, literal metadata, seed shape, and cross-registry duplicate refusal. It catches malformed entries in the typed files, but not every malformed flat-list semantic.
- `tests/grammar/test_vocab_uniqueness.py:28-91` loads token-bearing YAML files and computes cross-file collisions; `test_vocab_cross_segment_uniqueness()` at `:94-174` rejects collisions outside its documented dual-role allowlist. `test_no_empty_vocabularies()` at `:177-203` catches an accidentally emptied non-stub file.
- `tests/test_codegen_drift.py:1-44` calls generator check mode, so changing YAML without committing the generated outputs fails.
- `tests/grammar/test_vocab_additions_round_trip.py:1-78` is the existing extension point for a concrete name per newly registered token. It proves public parse/compose integration and canonical placement. A valid but semantically misclassified token can still round-trip, so the new exemplar must assert the intended IR field as well as spelling if the segment distinction matters.
- `tests/grammar/test_context.py:77-107` asserts prompt-facing vocabulary sections exactly match generated `SEGMENT_TOKEN_MAP`.
- `tests/grammar/test_vocabulary_gates.py:464-488` pins representative closed physical-base/qualifier/orbit/population membership. It is a coarse closure guard, not a per-new-token semantic test.

Thus a wrong typed shape, duplicate, missing regeneration, or broken parse/render integration already has a failure surface. A semantically wrong but syntactically valid choice is **not** automatically rejected; it needs a targeted IR-segment assertion grounded in the source documentation and unit, which matches the plan’s acceptance rule at `sn-grammar-refinement.html:175-181`.

## Second question: parse, re-render, and supersede one identity

### Deterministic shadow primitive

The lossless primitives are public ISN APIs:

```python
old = parse(stored_spelling, strict=False)
new_spelling = compose(old.ir)
```

`parse()` returns `ParseResult.ir` at `imas_standard_names/grammar/parser.py:2041-2052,2055-2105`; `compose()` renders one canonical string at `grammar/render.py:253-298`. The migration shadow must use the lenient parse of the stored spelling because a spelling made non-canonical by the new renderer is precisely what strict canonical validation may reject. It must then strictly parse the proposed spelling and compare the full `StandardNameIR` payload, plus check identity collisions, before any write.

imas-codex already wraps that contract in `imas_codex/standard_names/grammar_adapter.py`: `parse_canonical_name()` at `:117-125`, `normalize_canonical_name()` at `:128-141`, and `compose_canonical_ir()` at `:144-150`. `normalize_canonical_name()` accepts only the public parser’s supplied canonical form and never guesses segment classes. These are suitable as the one-name normalization/revalidation boundary; the initial shadow still needs the public lenient `parse()` when the old spelling is intentionally no longer canonical.

Two tempting projections are unfit for this job:

- `imas_codex/standard_names/graph_ops._parse_grammar()` at `graph_ops.py:354-410` converts parse failure into empty segment properties; it is a graph-projection helper, not a migration oracle.
- `imas_codex/standard_names/edit._grammar_segment_props()` at `edit.py:274-297` stores a deliberately limited base/locus projection. It cannot carry or compare the full recursive operator tree.

### Provenance-bearing supersede

`apply_edit()` at `imas_codex/standard_names/edit.py:533-615` is the reviewed public engine. Rename mode routes into `_apply_rename()` at `:2442-2478`, which requires grammar round-trip and refuses an identity collision. Its application block at `:2647-2693` calls `persist_refined_name()` with reason, edit origin, edit scope, request time, and run ID, then applies the normal successor validation gate.

`persist_refined_name()` at `imas_codex/standard_names/graph_ops.py:17327-17385` is the atomic storage primitive. Its contract explicitly provisions a successor, guards the authoritative source set, creates lineage, supersedes the predecessor, migrates source edges/projections, records one rename/refine event, and commits together. The implementation creates `REFINED_FROM` and marks the old name `superseded` at `:17754-17784`; `retarget_standard_name_sources()` and `record_standard_name_change()` receive operation, reason, origin, and run ID at `:17892-17929`.

### CLI route for exactly one identity

The one-identity route is `sn edit`, not `sn run --rename` and not a family/subtree cascade:

```text
uv run --no-sync imas-codex sn edit OLD --rename NEW --reason "canonical renderer migration; semantic IR unchanged" --scope self --dry-run
uv run --no-sync imas-codex sn edit OLD --rename NEW --reason "canonical renderer migration; semantic IR unchanged" --scope self --stage-only
```

`imas_codex/cli/sn.py:6250-6267` binds one positional `standard_name` and `--rename`; `:6289-6294` requires the provenance reason; `:6306-6315` defines `self` as this name only; `:6340-6350` makes `--stage-only` the bulk-efficient path; and `:6477-6492` invokes `apply_edit()`. Without `--stage-only`, `:6507-6510` runs the normal review scoped to that edit’s run ID and lands only after the gate. For a mechanical migration cohort: compute the complete no-write map first, dry-run every pair, stage each pair with `--scope self --stage-only`, then use the existing scoped batch-review route once. Do not use parent/family scope, string substitution, or LLM regeneration.

## Third question: exact source-manifest catalog cuts

### Current decision path

The current release flow is:

```text
sn_sources YAML
  -> load_focus_file()/load_sources_file(): exact flattened DD paths
  -> mint_sn_list(): direct PRODUCED_NAME identities plus family closure
  -> fetch_manifest_source_release_rows(): per-path terminal successor identity
  -> run_review_release(): union(minted family, terminal identities)
  -> exporter(review_batch=that union)
  -> assemble_review_catalog(): approved baseline plus emitted batch identities
```

Concrete seams:

- `imas_codex/standard_names/sources_manifest.py:59-79` validates the document schema; `load_sources_file()` at `:113-131` returns de-duplicated exact paths; `_flatten_sources()` at `:148-158` constructs `<ids>/<path>`; `load_focus_file()` at `:183-200` refuses to guess between `sn_sources` and `sn_names`.
- `mint_sn_list()` at `imas_codex/standard_names/minting.py:48-69` delegates to `_mint_with_client()`. Its authoritative base join is the manifest path’s `StandardNameSource -[:PRODUCED_NAME]-> StandardName` at `:72-97`, but `:98-120` deliberately adds each name’s parent, siblings, and children. The whole function is therefore correct for its review-cohort contract and unfit for an exact manifest-bound release cohort.
- `fetch_manifest_source_release_rows()` at `graph_ops.py:11824-11920` preserves one release-accounting row per de-duplicated manifest path and follows `HAS_SUCCESSOR` to the terminal identity. It is nearly the exact selector required. However, at `:11888-11896`, an ambiguous source with several direct IDs and no authoritative matching `produced_sn_id` falls back to the lexicographically first direct ID. Exact release selection should fail closed or enumerate and reconcile that ambiguity, not silently choose one.
- `run_review_release()` at `catalog_release.py:1446-1479` currently computes `names = mint.names ∪ terminal_ids`; that union is the point where unbound family members enter the release batch. It freezes the same list at `:1537-1546`, passes it as `review_batch` at `:1549-1557`, and passes it into assembly at `:1567-1577`.
- `_fetch_candidates()` at `export.py:373-461` is reusable once handed the exact list: batch mode selects already-approved catalog entries or an ID in the batch, then retains the validation, quorum, and docs gates. `assemble_review_catalog()` at `:1808-1875` preserves the approved baseline byte-for-byte and replaces/adds only fresh records whose IDs are in `batch_names`. This means “only manifest-bound names in the cut” should be enforced on the **new review batch**; it must not delete unrelated names already in the approved baseline.

### Minimal implementation derived from those seams

1. Keep `load_focus_file()` and the schema unchanged.
2. For `kind == "sn_sources"`, derive `manifest_sources` with `fetch_manifest_source_release_rows()` and make the release `names` list solely from its non-null terminal `standard_name_id` values. Do not union `mint_sn_list().names` into the release batch.
3. Refuse release when one manifest source has ambiguous direct bindings, when the projection cannot give an unambiguous terminal identity, or when a selected batch identity cannot be traced back to at least one path in this exact manifest. Preserve unmatched/non-nameable rows for the exclusion ledger.
4. Keep `_fetch_candidates()` and `assemble_review_catalog()` unchanged; they already enforce eligibility and approved-baseline preservation given a correct batch list.
5. Extend `tests/standard_names/test_review_release.py:812-840` with a source bound to one name plus an unbound parent/sibling/child and assert `report.names`, frozen artifact `names`, and exporter `review_batch` contain only the bound terminal identity. Existing `tests/standard_names/test_minting.py:42-55,181-198` should continue pinning family closure for callers that intentionally want the broader review cohort. Preserve the source-accounting and successor tests at `tests/standard_names/test_export_exclusion_ledger.py:498-600` and add an ambiguous-multi-binding refusal case.

## Candidate fitness verdicts

Every discovered candidate below has exactly one verdict.

| ID | Candidate mechanism | Verdict | Why |
|---|---|---|---|
| C01 | `grammar/specification.yml` segment declarations and include map | reusable as-is | It is the canonical segment-to-vocabulary ownership table and already drives generation and docs. |
| C02 | Typed loaders in `grammar/vocab_loaders.py` | extendable | Strict Pydantic validation is strong for typed registries, but flat-list vocabularies lack the same token-shape/semantic validation. |
| C03 | `validate_no_cross_registry_duplicates()` | extendable | It catches collisions among the typed registries only; the broader YAML uniqueness test is still needed for flat files. |
| C04 | `grammar_codegen.generate.main()` and `_run_check_mode()` | reusable as-is | They regenerate the dependent API/context modules and fail on committed-output drift. |
| C05 | `hatch_build_hooks.CustomBuildHook` and `docs/grammar_macros.py` | reusable as-is | Build and documentation already consume the grammar source of truth automatically. |
| C06 | Existing vocabulary loader/uniqueness/drift/context/round-trip tests | extendable | They cover structural mistakes, but each new semantic token still needs a name plus intended-IR-segment assertion. |
| C07 | Public `imas_standard_names.parse()` plus `compose()` | reusable as-is | This is the lossless, deterministic stored-spelling-to-IR-to-new-render path fixed by the plan. |
| C08 | `grammar_adapter.normalize_canonical_name()` / `compose_canonical_ir()` | reusable as-is | They accept only public-parser canonicalization and re-prove strict spelling plus full semantic IR. |
| C09 | `graph_ops._parse_grammar()` / `edit._grammar_segment_props()` | unfit | They are lossy graph projections and can turn a parse failure into empty properties; they cannot prove recursive IR preservation. |
| C10 | `edit.apply_edit()` / `_apply_rename()` | reusable as-is | They validate a literal successor, refuse collision, preserve reason/scope/origin, and route it through normal validation/review. |
| C11 | `graph_ops.persist_refined_name()` | reusable as-is | It atomically creates lineage, supersedes the old identity, migrates authoritative source bindings, and records provenance. |
| C12 | `sn edit OLD --rename NEW --scope self` | reusable as-is | The positional target and explicit self scope reach one identity; dry-run and stage-only support a deterministic bulk migration without a sweep. |
| C13 | `sources_manifest.load_focus_file()` / `load_sources_file()` | reusable as-is | Schema validation and exact path flattening already fail fast and preserve the manifest’s own membership. |
| C14 | `fetch_manifest_source_release_rows()` | extendable | Terminal-successor projection is right, but ambiguous direct bindings currently choose a first ID instead of failing closed. |
| C15 | `minting.mint_sn_list()` as the release selector | unfit | Its documented parent/sibling/child closure intentionally includes identities with no path in the batch manifest. |
| C16 | `catalog_release.run_review_release()` source-cohort construction | extendable | The release scaffolding is sound; replace only `mint.names ∪ terminal_ids` with the exact, validated terminal IDs. |
| C17 | `export._fetch_candidates()` and `assemble_review_catalog()` | reusable as-is | Given an exact batch, they apply eligibility gates and preserve the approved baseline while limiting fresh replacement/addition to batch IDs. |
| C18 | Catalog stable-package pin and tag-derived site checkout | reusable as-is | The site already obtains grammar parsing through the pinned package; the existing deliberate pin bump is the correct propagation mechanism. |

## Quantitative checks

The live plan’s vocabulary section has four gap rows. Exact command:

```text
sed -n '162,174p' docs/plans/sn-grammar-refinement.html | rg -c '<tr><td'
4
```

The owner map has 22 declared segment IDs; the exact command and result are recorded beside that map above.

The verdict census is computed from the table above with:

```text
awk -F'|' '/^\| C[0-9][0-9] / {n++; if ($4 !~ /(reusable as-is|extendable|unfit)/) missing++} END {print "candidates=" n, "missing_verdicts=" missing+0}' /home/ITER/mcintos/.config/reckon/crew/reports/sgr-reuse-map.md
candidates=18 missing_verdicts=0
```

## Follow-ons outside this read-only node

- Upstream implementation must choose and justify each new token’s exact segment from DD documentation, unit, corpus non-nameability, and base-family fit; only `relative_humidity -> physical_base` is already explicit in the live plan.
- The catalog-cut implementation should harden ambiguous source binding and add the release-specific no-family-leak regression described above. Preserve `mint_sn_list()` for its intentional broader review-cohort callers.
- The migration implementation should generate the complete no-write old/new/IR-equivalence/collision census before staging any `--scope self` edits. Verification of grammar, imas-codex, and catalog tests belongs to the separately dispatched test node.

