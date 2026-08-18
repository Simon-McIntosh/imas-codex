# DD override and suppression inventory

Snapshot: repository commit `a0e7f8b1ce4dd14e43b1e4ec2e1ec98c8b584566`.

Scope: read-only sweep of `imas_codex/**`, with `tests/**` searched separately. The sweep used the live `sn-dd-gaps` plan as semantic authority, parsed every YAML/JSON carrier, searched all Python/YAML/JSON/TOML paths for override/exception/skip/correction/reconcile behavior, traced every named carrier to its production consumers, and inspected the DD source-classification and compose gates. No graph query or graph write was issued.

## Headline result

- The **62 legacy rules** are exactly **34** `dd_unit_bugs` rows in `imas_codex/units/dd_unit_exceptions.yaml` plus **28** extraction rules in `imas_codex/standard_names/config/unit_overrides.yaml`. The 3 `unit_equivalences` pairs in `dd_unit_exceptions.yaml` are comparator normalization and are **not** part of the 62.
- The legacy 62 divide into **23 mismatch suppressions**, **11 graph corrections**, **24 extraction-time replacements**, and **4 extraction-time skips**.
- Review intake contains **21 candidate rows**: **16 bounded** candidates carrying **43 exact paths**, plus **5 broad-scope holds** carrying no exact paths.
- The active manifest contains **5 active exact resolutions**, all for DD 4.1.1 `unit` fields and all mapped to legacy rows `U25`–`U29`; it has **0 state-change receipts**.
- The active manifest is currently **authority at rest, not authority in the Standard Names pipeline**. Outside `dd_resolutions.py` itself, production imports are limited to the `sn ddres` lifecycle CLI (`list`, `show`, `approve`, `revoke`). No source extractor, context renderer, composer, reviewer, attachment audit, release path, DD build path, or reconcile path calls `resolve_dd_field`, `resolve_dd_context`, or `resolve_dd_rows`.
- Therefore the five approved records do **not yet retire any legacy behavior**. The same five paths are still corrected early by `dd_unit_exceptions.yaml` via `correct_in_graph`, so downstream code normally sees the corrected graph value and the typed resolver is bypassed.
- The repo-wide sweep found additional carriers outside the three named resources: the 28-row `unit_overrides.yaml`; path-independent unit normalizer/sentinel sets; dynamic unparseable-unit skips; source qualification and node-category nameability gates; a 10-token post-compose non-nameable list; a numeric-missing-unit fallback; one DD-defect classifier regex; and two deterministic physics-domain overrides. These are enumerated below. No other per-path unit-correction registry was found.

## Carrier table

| Carrier | Format and exact entry count | Semantics | Enforcement stage and production path | Cutover relevance |
|---|---:|---|---|---|
| `imas_codex/units/dd_unit_exceptions.yaml` | YAML: **34** `dd_unit_bugs` + **3** `unit_equivalences` = **37 list entries**. Bugs split **23 suppress-only / 11 `correct_in_graph`**. | `suppress` for every matching bug tuple on the mismatch/attachment axes; `correct_in_graph` additionally rewrites 11 matching raw unit declarations; equivalence pairs suppress comparison-only spelling/form differences. Matching requires path glob + exact canonical DD unit + exact canonical SN unit. | Loaded by `imas_codex/units/dd_unit_exceptions.py`. `units_agree()` is used by graph integrity and attachment guardrails. `graph_unit_correction()` is called by `imas_codex.units.resolve_dd_unit()`, which is called by DD build paths in `graph/build_dd.py` and the drift/reconcile path in `graph/dd_graph_ops.py`. `standard_names/loop.py` runs `reconcile_dd_unit_corrections()` before Standard Names work. | This is the primary legacy behavior carrier. Only exact rows replaced by active, consumed typed resolutions may be retired. The 3 equivalences are not DD-defect rows and should not be removed merely because typed resolutions are introduced. |
| `imas_codex/units/dd_unit_exceptions.py` | Python enforcement engine; **0 independent data entries**. | Implements canonicalization, exact tuple matching, suppression, and the `correct_in_graph` selector. | Comparator/attachment/integrity and DD build/reconcile stages. | Must cease being behavioral authority for each retired bug row; it may remain the equivalence comparator or become a compatibility guard during migration. |
| `imas_codex/standard_names/config/unit_overrides.yaml` | YAML: **28 ordered first-match rules** = **24 `override` + 4 `skip`**. | `override` replaces the extracted DD unit with `override_unit`; `skip` removes the DD path from extraction and records a `StandardNameSource` skip. | Loaded by `standard_names/unit_overrides.py`; applied by `sources/dd.py::_apply_unit_overrides()` in both bulk and targeted extraction, re-applied after authoritative-unit reinjection in `workers.py`, and consulted by `graph_ops.py::revive_unit_skipped_sources()`. | This is the second legacy behavior carrier and accounts for rows `O01`–`O28` in the 62. None of its rows is active typed authority today. Do not delete it wholesale during the five-record cutover. |
| `imas_codex/standard_names/unit_overrides.py` plus `sources/dd.py::_apply_unit_overrides()` | Python engine over the 28 YAML rows, plus **1 dynamic catch-all rule family** with **0 enumerated entries**. | First matching configured rule wins. After configured resolution, any remaining whitespace-bearing, non-numeric-exponent, or canonical-parser-invalid unit is dynamically skipped as `dd_unit_unresolvable`. | DD source extraction before qualification/composition; targeted extraction; worker re-injection; skip revival. | Typed field resolution does not automatically replace a dynamic parse-safety gate. Retain as validation after the resolver, but ensure it evaluates the effective typed value and cannot silently override an active resolution. |
| `imas_codex/standard_names/config/dd_resolution_candidates.yaml` | Strict YAML review resource: **21 candidates** = **16 `bounded_review_input` + 5 `broad_scope_hold`**; **43 exact paths**; **4 upstream change records**. | `review-only candidate`; explicitly lacks approval receipt, actor/time, fresh evidence token, decision reason, positive revision, and review decision. | Loaded only by `load_dd_resolution_candidates_for_review()` and `sn ddres` lifecycle operations. There is no pipeline consumer. | Never use it as effective authority. The five holds are evidence that broad legacy globs cannot be retired from an incomplete exact-path set. |
| `imas_codex/standard_names/config/dd_resolutions.yaml` | Strict YAML manifest: **5 resolution records**, **5 active**, **0 state changes**. All are exact DD 4.1.1 `unit` resolutions for `U25`–`U29`. | `active resolution`: exact `(path, dd_version, field, observed value)` maps to an effective typed value with approval/evidence/upstream provenance. | Loaded by `dd_resolutions.py` and lifecycle CLI. **No Standard Names pipeline consumer exists.** | Intended cutover authority, but not yet consumed. The five paths must be made to enter the pipeline as raw `1` plus applied provenance, not merely appear as already-corrected graph values. |
| `imas_codex/standard_names/dd_resolutions.py` | Python typed resolver; **10 supported DD context fields** (`unit`, documentation, data type, node type, physics domain, two COCOS fields, coordinates, lifecycle status/version). | Exact active record applies only when raw equals reviewed observed value; raw equal to effective is certified as upstream convergence; stale values, version mismatch, duplicate authority, or ambiguity fail closed. Raw context is retained separately from the effective projection. | Available resolver API and lifecycle CLI. Production import sweep found **0 calls** from source/context/pipeline consumers. | Consumer wiring is the missing seam. All consumers must use one `ResolvedDDContext`/marked item before legacy carriers are retired. |
| `imas_codex/units/__init__.py` normalizer sets | Python literal sets: **13 exact strings** across 4 sets: 3 dimensionless spellings, 3 count pseudo-units, 1 ambiguous unit string, 6 non-unit/sentinel strings; plus 2 prefix rule families (`units given…`, `as_parent…`). | `other`: path-independent normalization or suppression. Dimensionless spellings and count pseudo-units become `1`; ambiguous/non-unit strings become `None`. | DD build (`resolve_dd_unit`) and all canonical unit consumers. | Mostly lexical/sentinel normalization, not per-path defect authority. Preserve separately unless the typed resolver and raw/effective model explicitly subsume each case. The ambiguous `Elementary Charge Unit` behavior overlaps the extraction registry and needs a single post-cutover ordering. |
| `imas_codex/units/data_dictionary_unit_aliases.txt` | Pint definitions text: **5 non-comment declarations** (1 error unit, 3 aliases to it, 1 `Ohm` alias). | `other`: parser aliases for malformed/non-SI DD strings; does not select a path-specific corrected physical value. | Pint registry initialization before canonical normalization. | Not a typed-resolution retirement target; retain parser compatibility unless proven redundant. |
| `imas_codex/standard_names/sources/dd_qualifier.py` | Python ordered predicate catalogue: **12 rules** (`S0`–`S11`): **11 `skip`**, **1 `not_physical`**. Named literal sublists contain 3 error suffixes, 1 configurable segment, 3 local-frame unit-vector segments, and 5 flag phrases; other rules are type/path/unit/category predicates. | `skip list` / structural suppression: string leaves, duplicate IDS, error companions, placeholders, configurable process slots, mixed or unparseable units, nested time coordinates, local coordinate frames, GGD metadata, and configuration flags do not reach composition. | After unit override filtering in bulk and targeted DD extraction. Skip records are persisted when enabled. | These are nameability policy, not DD field corrections. Do not retire them with unit resolutions. Ensure resolved effective fields are presented before the unit-related rules (`mixed`, unparseable) run. |
| `imas_codex/core/node_classifier.py` + `core/node_categories.py` | Python taxonomy: **9 categories**, of which **3 admitted** to SN extraction (`quantity`, `geometry`, `coordinate`) and **6 excluded** (`error`, `fit_artifact`, `identifier`, `metadata`, `representation`, `structural`). One explicit DD-defect path regex, `_DIAMAGNETIC_AXIS_RE`, forces matching vector `/diamagnetic` axes to `representation`. | `skip list` / derived classification suppression. The classifier replaces raw structural attributes with a graph `node_category`; extraction gates on the admitted set. The explicit diamagnetic rule is a local “exclude until DD corrected” exception. | DD build/classification, then source-query gate in `sources/dd.py`. | The six-category gate is general policy and remains. The single upstream-defect regex is a retirement candidate only if a typed DD field resolution or corrected release changes the classification inputs and tests prove the path becomes nameable. |
| `imas_codex/standard_names/workers.py::_NON_NAMEABLE_BARE_TOKENS` | Python frozenset: **10 tokens** (`time`, `time_stamp`, `timestamp`, `delay`, `latency`, `dead_time`, `index`, `count`, `counter`, `version`). | `skip list`: suppresses a composed result only when the entire proposed name is one of these bare tokens. | Compose candidate processing in synchronous and pooled worker paths; pooled mode persists a skip. | Not DD value authority. Keep as a downstream semantic guard; do not confuse it with typed resolution retirement. |
| Numeric missing-unit fallback in `imas_codex/standard_names/workers.py` | Python code rule: **1 rule family**, numeric prefixes `FLT_`, `INT_`, `CPX_`. | `other`: when the unit relationship/value is absent on a numeric DD row, injects unit `1` before compose, then re-applies `unit_overrides`. | Worker enrichment before compose. | This is an implicit effective-value injection outside every manifest. It must be routed through typed/raw context or constrained to proven absent-unit dimensionless semantics before claiming complete consumer cutover. |
| `imas_codex/definitions/physics/path_domain_overrides.json` | JSON: **2 deterministic substring rules**. | `other`: overrides the tier-1 LLM result for DD-derived `physics_domain` and records `domain_source='deterministic_override'`. | `graph/dd_domain_classifier.py::_classify_batch()` after LLM classification and before graph write. | Not an upstream DD-declared field correction in the current pipeline, but it is an override of a DD-path-derived graph value and therefore part of the broad sweep. It is outside the five unit-resolution retirements. |

## The 62 legacy rows mapped to review and active authority

The row ids used by candidate provenance are deterministic inventory labels: `U01`–`U34` enumerate `dd_unit_bugs` in file order, and `O01`–`O28` enumerate `unit_overrides` in file order.

| Legacy carrier | Active typed | Bounded candidate, inactive | Broad hold | No candidate | Total |
|---|---:|---:|---:|---:|---:|
| `dd_unit_exceptions.yaml` bug rows (`U`) | 5 | 10 | 0 | 19 | 34 |
| `unit_overrides.yaml` rows (`O`) | 0 | 1 | 5 | 22 | 28 |
| **Total** | **5** | **11** | **5** | **41** | **62** |

Exact buckets:

- **Active:** `U25`, `U26`, `U27`, `U28`, `U29`.
- **Bounded candidate but inactive:** `U11`–`U16`, `U19`, `U21`, `U22`, `U32`, `O17`.
- **Broad-scope hold:** `O20`–`O24`.
- **No candidate:** `U01`–`U10`, `U17`, `U18`, `U20`, `U23`, `U24`, `U30`, `U31`, `U33`, `U34`; `O01`–`O16`, `O18`, `O19`, `O25`–`O28`.

Behavioral split with authority state:

- The **11 `correct_in_graph`** rows are: active `U25`–`U29`; bounded inactive `U19`; no candidate `U20`, `U24`, `U30`, `U31`, `U33`.
- The **23 suppress-only** bug rows contain 9 bounded inactive rows (`U11`–`U16`, `U21`, `U22`, `U32`) and 14 rows with no candidate.
- The **24 extraction replacements** contain bounded inactive `O17`, broad holds `O20`–`O24`, and 18 rows with no candidate.
- All **4 extraction skips** (`O25`–`O28`) have no candidate.

Cross-carrier duplication that matters:

- `O20` (`**/*unit_vector*/*`) broadly overlaps both `U15` (`*/unit_vector_major/[xyz]`) and `U16` (`*/unit_vector_minor/[xyz]`).
- `O21`–`O24` overlap `U11`–`U14` respectively (direction, direction-second, up, injection-direction families). The `O` rows remain broad holds because exact release/path scope is incomplete.
- `O17` (`**/ionisation_potential`, `e → eV`) overlaps the narrower `U19` GGD ionisation-potential correction. They are separate bounded candidates and must not both activate for one exact key; the existing resolver collision rules are the right fail-closed behavior.
- The five active rows `U25`–`U29` have **no counterpart in `unit_overrides.yaml`**. Their current behavior comes from the graph-build/startup `correct_in_graph` route, not extraction override rules.

## Enforcement flow today

1. DD build reads raw DD units and calls `resolve_dd_unit()`.
2. `resolve_dd_unit()` consults only `dd_unit_exceptions.yaml`; 11 rows can replace the DD unit before writing the `IMASNode.unit` scalar and `HAS_UNIT` edge.
3. Standard Names startup runs `reconcile_dd_unit_corrections()`, so later registry edits retroactively rewrite already-stored `IMASNode` units and edges.
4. DD source extraction chooses the unit relationship/scalar, then applies the separate 28-row `unit_overrides.yaml`; replacements alter the candidate unit and skips remove/persist the source exclusion. A catch-all removes any still-unparseable unit.
5. Worker enrichment may inject `1` for a numeric row with no unit, then re-applies the extraction override engine.
6. Source qualification and category gates suppress non-nameable DD rows.
7. Compose injects the resulting unit, normalizes `-` to `1`, and skips `mixed` or missing units; it also rejects bare non-nameable token results.
8. Attachment and integrity checks call `units_agree()`, which suppresses all 34 known bug tuples plus 3 equivalence pairs.
9. `dd_resolutions.yaml` is not present anywhere in steps 1–8. It is currently read only by its own resolver API and `sn ddres` lifecycle CLI.

## Retirement checklist for the consumer cutover

The order matters. Removing a legacy rule before the effective-resolution consumer exists changes behavior; wiring the typed resolver while leaving early graph correction in place hides application as “already converged” and loses raw/effective separation.

- [ ] **Freeze and assert carrier identities.** Pin the four resource digests from this audit in transition evidence: `dd_unit_exceptions.yaml` `81dcd027…`, `unit_overrides.yaml` `13310c50…`, candidates `c6ee52ae…`, active manifest `f8718114…`. Refuse transition if any changes without a refreshed row mapping.
- [ ] **Create one raw DD context boundary.** Every Standard Names source/context path must construct `RawDDContext` from the exact published DD version and immutable raw DD values, then call `resolve_dd_context()` once. Do not call field resolution piecemeal at downstream seats.
- [ ] **Wire all current unit consumers.** At minimum cover bulk extraction, targeted extraction, worker unit re-injection, compose context, parent/member context rendering, semantic signatures/manifests, attachment audit, integrity checks, release caveats, source refresh/drift, and DD-release reconciliation. Add an import/call guard proving these paths consume `ResolvedDDContext` rather than re-reading graph scalars directly.
- [ ] **Preserve provenance in pipeline items.** Require `raw_dd_context`, effective fields, applied/converged resolution ids, and manifest digest. A pipeline item carrying an effective value without this marker must fail closed when an active exact resolution exists.
- [ ] **Put normalization in the correct order.** Lexical canonicalization may run on raw/effective values, but no path-dependent legacy correction may run before typed resolution. Dynamic parse validation should run on the typed effective value; it must not supersede an active record silently.
- [ ] **Remove the numeric missing-unit shadow authority or prove it.** Replace the blanket `FLT_`/`INT_`/`CPX_` no-unit → `1` injection with an explicit raw/effective policy or a typed, audited absence rule. It cannot remain an invisible fourth unit authority.
- [ ] **Cut over `U25`–`U29` first and only.** These are the only five active records. Remove their `correct_in_graph` behavior only after all consumers use typed resolution. Leave all other 57 legacy rows unchanged.
- [ ] **Restore raw DD 4.1.1 graph state for the five paths.** Rebuild or use a guarded exact-path instrument to restore `IMASNode.unit='1'` and the single `HAS_UNIT {id:'1'}` edge from immutable DD evidence. Disable the startup legacy reconcile for these rows first, otherwise it will immediately rewrite them again. Verify the typed resolver reports `applied=True` and the expected resolution id for each path.
- [ ] **Prove raw/effective separation.** For each of the five paths, assert raw `1`; effective `Pa`, `Pa`, `m^-3`, `A.m^-2`, `A.m^-2`; DD version `4.1.1`; field `unit`; one active exact record; complete approval/evidence/upstream provenance; and no graph scalar mutation by the resolver.
- [ ] **Retire legacy comparator suppression for those exact rows only after parity.** Once all attachment/integrity consumers compare effective typed fields, delete or deactivate `U25`–`U29` from the legacy registry. A stale legacy match must become a test failure, not silently remain double authority.
- [ ] **Do not retire `unit_overrides.yaml` in this wave.** None of its 28 rows is active typed authority. `O17` is only bounded review input, `O20`–`O24` are holds, and `O25`–`O28` are unresolved skips.
- [ ] **Require exact-path completeness before retiring any glob.** Expand each legacy glob against the exact target DD release and require an active resolution for every intended match. Partial coverage of a broad rule must retain the legacy rule or split it into exact residual rules; never widen a typed record.
- [ ] **Keep review candidates non-consumable.** Add/retain a negative production-import test for `dd_resolution_candidates.yaml`; only the lifecycle CLI may read it.
- [ ] **Separate defect retirement from invariant normalization.** The 3 unit equivalence pairs, canonical lexical normalization, parser aliases, dynamic invalid-unit rejection, structural source qualification, six excluded node categories, and 10 bare-name tokens are not automatically retired by unit resolutions.
- [ ] **Adjudicate the one classifier defect exception separately.** `_DIAMAGNETIC_AXIS_RE` is a local “until DD corrected” suppression. Give it its own upstream/raw/effective evidence and nameability proof; do not fold it into the five unit records.
- [ ] **Retain DDGap history and lifecycle.** `dd_gaps.py` mirrors registry evidence and reconciles status against published raw release facts; it is observational and must not become an alternate behavior backend. After retirement, point history to the typed resolution and correcting release without deleting observations.
- [ ] **Run transition tests in layers.** First exact resolver tests for the five records; then extraction/targeted/worker/attachment/integrity parity; then a raw-graph restoration dry run; then the full Standard Names suite. A green manifest parser alone is not consumer cutover evidence.
- [ ] **Add a zero-shadow-authority guard.** After cutover, scan production code/config so an active exact `(path, version, field)` cannot also match `correct_in_graph`, extraction replacement, hardcoded fallback, or another active resolution. The guard should report residual carrier and row id.
- [ ] **Verify release convergence behavior.** On a later published DD where raw equals the prior effective value, require `converged=True`, no override application, retained provenance, and a governed retirement/state-change receipt before removing the active record.

## Tests and evidence paths noted separately

No pytest suite was run because this node is a read-only inventory, not an implementation/test node. Static parsers and searches completed successfully; their complete outputs are retained under `/tmp/reckon-s8-evidence/dd-override-inventory/`.

Relevant existing tests by carrier:

- Legacy exception/comparator/build/reconcile: `tests/units/test_dd_unit_exceptions.py`, `tests/core/test_dd_unit_resolution.py`, `tests/graph/test_dd_unit_correction_reconcile.py`, `tests/graph/test_gas_unit_correction.py`, `tests/graph/test_sn_unit_integrity.py`, `tests/graph/test_sn_edge_integrity.py`, `tests/standard_names/test_attachment_consistency.py`.
- Extraction overrides/skips/revival: `tests/standard_names/test_unit_overrides.py`, `tests/standard_names/test_dd_sources.py`, `tests/standard_names/test_source_revival.py`, `tests/standard_names/test_dd_gap_automated_writers.py`.
- Typed candidate/manifest/resolver/lifecycle: `tests/standard_names/test_dd_resolutions.py`, `tests/standard_names/test_dd_resolution_cli.py`, `tests/graph/test_dd_resolution_schema.py`.
- Classifier/qualifier/nameability: `tests/core/test_node_classifier.py`, `tests/core/test_node_categories.py`, `tests/standard_names/test_dd_qualifier.py`, `tests/standard_names/sources/test_dd_extract_breakdown.py`, `tests/standard_names/test_gyrokinetics_extraction.py`, `tests/standard_names/test_validation_gate.py`.
- Physics-domain overrides: `tests/graph/test_dd_domain_classifier.py`.

Primary audit logs:

- `yaml-counts.log`: exact row, strategy, candidate, path, and active-resolution counts.
- `direct-consumers.log`, `ddres-runtime-consumers.log`, `resolve-dd-unit-consumers.log`: production call-site traces.
- `carrier-name-sweep.log`, `dd-keyword-sweep.log`, `config-key-sweep.log`, `hardcoded-unit-assignments.log`, `unit-semantic-sweep.log`: whole-repository carrier sweeps.
- `code-carrier-counts.log`: exact literal-set/category/nameability counts.
- `carrier-sha256.log`: tracked resource identities.
- `test-consumers.log`, `classifier-tests.log`, `path-domain-tests.log`, `non-nameable-tests.log`: tests searched separately.

## Read-only completion evidence

- Repository baseline was clean: `git status --porcelain` emitted no output (`git-status-before.log`).
- Final repository status is recorded separately in `git-status-after.log` and must be empty.
- No graph client, graph CLI, Cypher tool, provider, pipeline, or service operation was invoked. In particular, no graph write was issued.

