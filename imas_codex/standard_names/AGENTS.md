# Standard-Names Agent Notes

Scoped to `imas_codex/standard_names/**` and `tests/standard_names/**`. Pipeline
architecture, lifecycle axes, pool semantics and CLI flags live in the repo root
`AGENTS.md` and `docs/architecture/standard-names.md` — this file holds only what
you need when *editing these files*.

## Graph identity and joins

`StandardName` is keyed by `id`, whose value is the snake-case standard-name
identity; the LinkML class marks that slot as the required identifier
(`StandardName.id`). `StandardNameSource` is keyed
separately by its required `id`, formed as `source_type + ":" + source_id`: DD
sources are `dd:<path>`, while facility signals are
`signals:<facility>:<signal-id>`
(`StandardNameSource.id`). Do not substitute a
plausible property name for either schema-owned key.

| Correct: join a `StandardName` on its identifier | Incorrect: join on an undeclared property |
|---|---|
| `MATCH (sn:StandardName {id: $standard_name_id})`<br>`RETURN sn.id` | `MATCH (sn:StandardName {name: $standard_name_id})`<br>`RETURN sn.id` |

The incorrect query is especially dangerous because it is valid Cypher. Access
to a missing property evaluates to `null`, and a filtering predicate that is not
`true` removes the row, so the query returns an empty result instead of an error
([Neo4j Cypher Manual: working with null](https://neo4j.com/docs/cypher-manual/current/values-and-types/working-with-null/)).
Before accepting a zero-row join as a real no-overlap result, report both the
candidate-node count and key coverage in the same query, and fail closed when
the proposed key covers no candidates:

```cypher
MATCH (sn:StandardName)
RETURN count(sn) AS candidates,
       count(sn.id) AS candidates_with_id,
       count(sn.name) AS candidates_with_name
```

The LinkML slot owns relationship direction. Read every row below as
`(source)-[:TYPE]->(target)`; do not infer direction from the English relationship
name, a nearby back-reference property, or whichever endpoint a query starts
from. A zero live count does not override the declaration. For self-relationships
such as `StandardName` to `StandardName`, an authored-direction census and the
same traversal reversed necessarily report the same count because both endpoint
labels are identical; the schema slot remains the only directional authority.

| Schema class and slot | Authored traversal |
|---|---|
| `StandardName.unit` | `(StandardName)-[:HAS_UNIT]->(Unit)` |
| `StandardName.physics_domain` | `(StandardName)-[:HAS_PHYSICS_DOMAIN]->(PhysicsDomain)` |
| `StandardName.cocos` | `(StandardName)-[:HAS_COCOS]->(COCOS)` |
| `StandardName.internal_changes` | `(StandardName)-[:HAS_INTERNAL_CHANGE]->(StandardNameChange)` |
| `StandardName.grammar_tokens` | `(StandardName)-[:HAS_SEGMENT]->(GrammarToken)` |
| `StandardName.grammar_physical_base_token` | `(StandardName)-[:HAS_PHYSICAL_BASE]->(GrammarToken)` |
| `StandardName.grammar_subject_token` | `(StandardName)-[:HAS_SUBJECT]->(GrammarToken)` |
| `StandardName.grammar_transformation_token` | `(StandardName)-[:HAS_TRANSFORMATION]->(GrammarToken)` |
| `StandardName.grammar_component_token` | `(StandardName)-[:HAS_COMPONENT]->(GrammarToken)` |
| `StandardName.grammar_coordinate_token` | `(StandardName)-[:HAS_COORDINATE]->(GrammarToken)` |
| `StandardName.grammar_process_token` | `(StandardName)-[:HAS_PROCESS]->(GrammarToken)` |
| `StandardName.grammar_position_token` | `(StandardName)-[:HAS_POSITION]->(GrammarToken)` |
| `StandardName.grammar_region_token` | `(StandardName)-[:HAS_REGION]->(GrammarToken)` |
| `StandardName.grammar_device_token` | `(StandardName)-[:HAS_DEVICE]->(GrammarToken)` |
| `StandardName.grammar_geometric_base_token` | `(StandardName)-[:HAS_GEOMETRIC_BASE]->(GrammarToken)` |
| `StandardName.grammar_aggregation_token` | `(StandardName)-[:HAS_AGGREGATION]->(GrammarToken)` |
| `StandardName.grammar_orbit_token` | `(StandardName)-[:HAS_ORBIT]->(GrammarToken)` |
| `StandardName.grammar_population_token` | `(StandardName)-[:HAS_POPULATION]->(GrammarToken)` |
| `StandardName.references` | `(StandardName)-[:REFERENCES]->(StandardName)` |
| `StandardName.parents` | `(StandardName)-[:HAS_PARENT]->(StandardName)` |
| `StandardName.magnitudes` | `(StandardName)-[:MAGNITUDE_OF]->(StandardName)` |
| `StandardName.error_siblings` | `(StandardName)-[:HAS_ERROR]->(StandardName)` |
| `StandardName.predecessor` | `(StandardName)-[:HAS_PREDECESSOR]->(StandardName)` |
| `StandardName.successor` | `(StandardName)-[:HAS_SUCCESSOR]->(StandardName)` |
| `StandardName.primary_cluster_ref` | `(StandardName)-[:IN_CLUSTER]->(IMASSemanticCluster)` |
| `StandardName.reviews` | `(StandardName)-[:HAS_REVIEW]->(StandardNameReview)` |
| `StandardName.structural_authorities` | `(StandardName)-[:HAS_STRUCTURAL_AUTHORITY]->(StructuralNameAuthority)` |
| `StandardName.docs_revisions` | `(StandardName)-[:DOCS_REVISION_OF]->(DocsRevision)` |
| `StandardName.docs_review_admission` | `(StandardName)-[:HAS_DOCS_REVIEW_ADMISSION]->(DocsReviewAdmission)` |
| `StandardName.refined_from` | `(StandardName)-[:REFINED_FROM]->(StandardName)` |
| `StandardName.loci` | `(StandardName)-[:HAS_LOCUS]->(Locus)` |
| `DocsReviewAdmission.created_reviews` | `(DocsReviewAdmission)-[:CREATED_REVIEW]->(StandardNameReview)` |
| `VocabGap.evidence` | `(VocabGap)-[:HAS_EVIDENCE]->(VocabGapEvidence)` |
| `DDResolution.evidence` | `(DDResolution)-[:EVIDENCED_BY]->(DDGap)` |
| `DDResolution.for_dd_version` | `(DDResolution)-[:FOR_DD_VERSION]->(DDVersion)` |
| `DDGap.observations` | `(DDGap)-[:HAS_OBSERVATION]->(DDGapObservation)` |
| `DDGap.state_changes` | `(DDGap)-[:HAS_STATE_CHANGE]->(DDGapStateChange)` |
| `DDGap.identity_changes` | `(DDGap)-[:HAS_IDENTITY_CHANGE]->(DDGapIdentityChange)` |
| `StandardNameSource.retry_events` | `(StandardNameSource)-[:HAS_RETRY_EVENT]->(StandardNameSourceRetry)` |
| `StandardNameSource.snapshot_changes` | `(StandardNameSource)-[:HAS_SNAPSHOT_CHANGE]->(StandardNameSourceSnapshotChange)` |
| `StandardNameSource.identity_repairs` | `(StandardNameSource)-[:HAS_IDENTITY_REPAIR]->(StandardNameSourceIdentityRepair)` |
| `StandardNameSource.snapshot_adoptions` | `(StandardNameSource)-[:HAS_SNAPSHOT_ADOPTION]->(StandardNameSourceSnapshotAdoption)` |
| `StandardNameSource.unit_cache_corrections` | `(StandardNameSource)-[:HAS_UNIT_CACHE_CORRECTION]->(StandardNameSourceUnitCacheCorrection)` |
| `StandardNameSource.snapshot_admissions` | `(StandardNameSource)-[:HAS_SNAPSHOT_ADMISSION]->(StandardNameSourceSnapshotAdmission)` |
| `StandardNameSource.identity_folds` | `(StandardNameSource)-[:HAS_IDENTITY_FOLD]->(StandardNameSourceIdentityFold)` |
| `StandardNameSource.authority_retirements` | `(StandardNameSource)-[:HAS_AUTHORITY_RETIREMENT]->(StandardNameSourceAuthorityRetirement)` |
| `StandardNameSource.dd_path` | `(StandardNameSource)-[:FROM_DD_PATH]->(IMASNode)` |
| `StandardNameSource.signal` | `(StandardNameSource)-[:FROM_SIGNAL]->(FacilitySignal)` |
| `StandardNameSource.standard_name` | `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` |
| `StandardNameSource.vocab_gaps` | `(StandardNameSource)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(VocabGap)` |
| `LLMCost.for_run` | `(LLMCost)-[:FOR_RUN]->(SNRun)` |
| `PromotionCandidate.evidences` | `(PromotionCandidate)-[:EVIDENCED_BY]->(StandardName)` |
| `StructuralNameAuthority.children` | `(StructuralNameAuthority)-[:ENTAILED_FROM_CHILD]->(StandardName)` |

Relationship slots are foreign-key properties too: the property value names the
target node, while the generated graph carries the edge in the direction above.
The table below also includes scalar mirrors and explicit back-references whose
schema prose identifies a `StandardName`. It deliberately excludes free-form
replacement suggestions and generic repair-envelope targets, which need not
identify an existing `StandardName`.

| Schema class | Slots that reference `StandardName` |
|---|---|
| `StandardName` | `StandardName.deprecates`, `StandardName.superseded_by`; `StandardName.links`; `StandardName.refine_collision_name`; `StandardName.references`, `StandardName.parents`, `StandardName.magnitudes`, `StandardName.error_siblings`, `StandardName.predecessor`, `StandardName.successor`, `StandardName.refined_from` |
| `DocsReviewAdmission` | `DocsReviewAdmission.target_id` |
| `DocsRevision` | `DocsRevision.standard_name_id` |
| `StandardNameChange` | `StandardNameChange.to_name` when it is the linked name rather than another changed value |
| `StandardNameSource` | `StandardNameSource.standard_name` (relationship slot), `StandardNameSource.produced_sn_id` (scalar mirror) |
| `StandardNameSourceAuthorityRetirement` | `StandardNameSourceAuthorityRetirement.removed_target_ids` |
| `StandardNameSourceRetry` | `StandardNameSourceRetry.terminal_sn_id` |
| `LLMCost` | `LLMCost.standard_name_ids` |
| `StandardNameReview` | `StandardNameReview.standard_name_id` |
| `PromotionCandidate` | `PromotionCandidate.evidences` |
| `RepairRowIdentity` | `RepairRowIdentity.target_id` when `kind` selects a StandardName target |
| `StructuralNameAuthority` | `StructuralNameAuthority.accepted_name_id`, `StructuralNameAuthority.child_ids`, `StructuralNameAuthority.children` |

The generic back-reference convention is `standard_name_id` for a scalar and
`standard_name_ids` for a multivalued property. `StandardName` itself remains
keyed by `id`; relationship slots with more specific semantics retain the names
declared in the table. The review axis follows the same schema-to-reader rule:
`StandardNameReview.review_axis` uses `name` and `docs`, exactly matching the
paired `_name` and `_docs` slot suffixes.

Aggregation has the same silent-null trap as filtering. `count(property)` and
`count(DISTINCT property)` ignore `null`, so an undeclared or misremembered
property produces zero rather than an error. `DocsRevision` is the worked
example: its back-reference property is `standard_name_id`, and the authored
edge runs from `StandardName` to `DocsRevision`, not the reverse. The exact
schema references are `StandardName.docs_revisions` and
`DocsRevision.standard_name_id`.

```cypher
MATCH (sn:StandardName)-[:DOCS_REVISION_OF]->(rev:DocsRevision)
RETURN count(rev) AS revisions,
       count(DISTINCT rev.standard_name_id) AS revisions_with_the_schema_key,
       count(DISTINCT rev.name) AS silently_zero_wrong_key
```

Before trusting any traversal or foreign-key aggregate, confirm both the slot
name and the authored direction in LinkML, then report the authored and reversed
counts together. The live verification census must cover every row in the
relationship table, including zero-count declarations; omitting empty rows
turns schema drift into apparent success.

## Naming-hygiene keep-list (calibration)

`~/.agents/AGENTS.md` mandates a pre-stage check for plan/stage/bug labels and
changelog prose in filenames, symbols and comments. Its letter class is
deliberately generic (`\b[A-Z][0-9]+[A-Za-z]?\b`), so it is noisy here. These
matches are **legitimate and must be kept** — verified, with the reason:

| Match | Why it stays |
|---|---|
| `S0`–`S11` (`sources/dd_qualifier.py`) | A local numbered rule catalogue, each item with its own description. `tests/standard_names/test_dd_qualifier.py` cross-references them **by id** in ~24 docstrings. |
| `Rule 1`–`Rule 6` (`error_siblings.py`) | Same shape; `test_error_siblings{,_gate}.py` assert on them. |
| `R1`–`R5` (`vocab_token_filter.py`) | Emitted **into** `TokenVerdict.reason` at runtime (`"R4: token contains digits"`); tests assert the substring. Removing them breaks assertions. |
| `D{n}` (compose prompt gallery) | Rendered prompt CONTENT — `test_prompt_completeness` counts `f"D{n} —"` in the rendered system prompt. |
| `rotation_cap` (~40 sites) | A real claim-query kwarg and function parameter. |
| ISN `rc14`/`rc21`/`rc22`/`rc34`/`rc39`/`rc41` | Dependency version contracts recording when a token became valid or a segment opened. |
| `--only <phase>`, `phase_caps`, `LLMCost.phase`, `_PHASE_TO_POOL` | `phase` is a real CLI flag and a real budget/cost dimension here. |
| `W74+`, `W$^{1+}$` | Tungsten charge states in LaTeX descriptions. |
| `T^2`, `m^-2`, `Wb`, `eV`, `A.m^-2` | Physical units. |
| `COCOS 17`, `DDv3`/`DDv4` | Real conventions and DD major versions. |
| `E2E` | "end-to-end". Note the contrast: `E1.`/`E2.` section ids in the *same* file are labels and come out. |
| `noqa` ids (`E402`, `F841`, `S608`, `D401`) | Lint rule codes. |
| `T0 = datetime(...)`, `s1`/`s2`, `m0`/`m1`, `h1`/`h2` | Ordinary locals and constants. |
| `Wave 2D …` in `definitions/clusters/labels.json` | **Physics** — a plasma wave, in two dimensions. Generated cluster labels; a naive `Wave [0-9]` rule corrupts the vocabulary. |

The discriminator that resolved most of the hard cases: **numbering that another
component references by number is load-bearing**; a plan-wide taxonomy id with no
adjacent prose is not. Contrast `S0`–`S11` (local, described, asserted on) with
`_L7_REVISION_MODEL` (a lever id meaning nothing without the taxonomy) — the
first stays, the second gets renamed to what it does.

## Attachment consistency (source → name)

`workers._is_attachment_consistent` is the single guard deciding whether a DD
path may realize a standard name. `attachment_audit.py` re-asks it of every
stored edge and detaches what it rejects, which makes the guard retroactive
across every writer at once.

**Three of its rules were wrong because a vocabulary was hardcoded in codex and
had drifted from ISN.** Expect that failure mode; derive from
`get_grammar_context()` at runtime and add a drift test asserting every ISN token
in the relevant segment is covered:

- rate-ness is expressed by SUFFIX as well as prefix (`..._source_rate`,
  `rotation_frequency`), so a name can be rate-natured without a leading
  `change_in_`/`rate_of_`;
- the DD uses `_dt` for **deuterium–tritium** as well as d/dt — the genuine
  derivative form carries a leading `d` on the differentiated quantity
  (`ddensity_dt_total`, `dphase_dt`);
- state resolution is ISN's `state` segment (`charge_state`, `internal_state`)
  plus the `_state`-suffixed subjects — not a fixed tuple.

**Ordered sample positions never enter a name.** This includes
`first`/`second`/`third` and `start`/`end` endpoint labels whenever the DD
structure proves they index a point or sample. The ordering may remain in the DD
path and source description as provenance, but generation, review, and refine
must all exclude it from identity. Dropping it must preserve the quantity,
carrier, geometry representation, owner, axis, mechanism, and locus. If that
non-ordinal carrier or locus is unavailable in the public grammar, emit the
exact vocabulary gap; never borrow `line_of_sight` or another object's token.
Registered semantic tokens such as `first_wall`, and state/process uses of
`start` or `end`, are not positional and must remain. Consequently
`…/line_of_sight/{first,second,third,start,end}_point/r` shares one name, so
`_vector_fields_conflict` must not treat ordered samples of one object as
distinct vector fields. Genuinely distinct fields (a camera's `direction` vs
`up`) and distinct geometry primitives (`rectangle` center vs `oblique`
reference corner) still conflict.

**A pairwise rule must be applied with compose semantics.** Compose accumulates
only ACCEPTED siblings, so one representative of a conflicting group survives;
passing every sibling to every row rejects the whole group. Applied
order-independently once, that would have stripped 127 names of every source.

**A whole-name wipeout is a NAME defect, not an attachment defect.** When every
source of a name fails the same rule the sources are consistently grouped and the
NAME is wrong — repair with `sn edit`, never by detaching, which would orphan the
name and rewind its sources to paid recomposition.

### Write paths

The guard is consulted at compose time. Paths that migrate a source set onto a
**different** name — the refine-successor migration, and the exclusive rebind /
retarget used by the edit cascade — need it at write time too: the set is
historical but the *pairing* is new, and a new pairing is what the guard exists
to judge. Paths that re-establish an edge against the **same** name (the
provenance reattach, the orphan-parent repair) must NOT be gated: gating a repair
with a rule that never governed the creation turns silent loss into permanent
loss, and the retroactive pass will detach a genuinely bad pair *with* an audit
record, which is the correct order.

A geometry-representation rule is still missing: a path under
`…/geometry/{thick_line,outline,rectangle,oblique,arcs_of_circle}/…` describes a
conductor cross-section, not an optical path, and must not realize a
`line_of_sight`/`beam` name. Both sides are metres, so the dimensionality rule is
silent and the guard currently accepts these.

## Units are DD-authoritative

The model never supplies `unit`, `cocos` or `physics_domain` — all three are
injected post-LLM. Consequences when a unit looks wrong:

- **Fix the DD side, not the name.** `units/dd_unit_exceptions.yaml` is the single
  registry. Most entries only *suppress* a mismatch (the DD is wrong, the name is
  right, and the axis keeps reporting it). Flag `correct_in_graph: true` only when
  the DD contradicts **itself** on one quantity, so there is no single declaration
  to mirror.
- **A registry entry alone is not enough.** The correction is applied by the DD
  build, so an entry added afterwards never reaches already-stored paths.
  `reconcile_dd_unit_corrections` (wired into the `sn run` startup sweep) is what
  makes the registry self-applying.
- **A name carries exactly one unit.** `HAS_UNIT` is cardinality-one in the
  schema, and the guard compares dimensionality — a name holding both `1` and
  `m^-2` admits sources of either. Both writers self-heal, but a terminal-stage
  name keeps the residue forever because nothing recomposes it.
- **Do NOT re-derive a name's unit from its sources in bulk.** ~40 accepted names
  correctly carry a dimensionless unit against a DD path the registry records as
  wrong (charge numbers tagged `e`, unit vectors tagged `m`); a bulk re-derivation
  would clobber every one of them.

## Operators live outside `SEGMENT_TOKEN_MAP` — read the grammar through one accessor

`SEGMENT_TOKEN_MAP` is **not** the whole grammar vocabulary. Operators
(`square`, `inverse`, `flux_surface_averaged`, `derivative_with_respect_to`, …)
are a separate mechanism composing through `operator_token`, so they occupy no
segment slot. A consumer reading only that map cannot see 51 legal tokens and
treats every one of them as an unregistered proposal.

**This has now generated the same bug five separate times**, each fixed at one
site: single-operator gap classification, the state/rate attachment rules, the
compose prompt's operator block, the decomposition classifier's multi-operator
compounds, and the plural-dedup check. Read the grammar through
`segments.grammar_tokens_by_segment()` (per-class tokens, operators included) or
its reverse `grammar_token_index()`. `tests/standard_names/test_vocab_consumers_see_operators.py`
fails on any new module importing `SEGMENT_TOKEN_MAP` that is not listed there
with a reason.

Two sets that look mergeable and are not: `reportable_segments()` is what a gap
may be **reported against** (wider — includes the model-layer slot names
`transformation`/`decomposition`); `known_segments()` is what the **parser** slots
tokens into. Merging them makes the response model reject valid composer output.

A compound spelled with `over` is a **division** — the binary `ratio` operator
over two operands, not a compound base. Never "fix" it by letting the cover walk
step over the word: that stops the token being `absent` while emitting guidance
that folds the operators into one base and silently drops the division. An honest
`absent` costs less than confident wrong guidance.

### Open ISN-side findings (not codex's to fix, no PR — ISN's MCP surface is in flux)

- **Locus tokens reachable by the parser but by no vocabulary enumeration.**
  `active_wall_point`, `beam_path` and `primary` are in the locus registry yet in
  no `SEGMENT_TOKEN_MAP` segment, so a consumer listing the vocabulary cannot emit
  them and they never reach a prompt. Same shape as the `normalizing_qualifiers`
  token `gyrocenter`, which is in no segment either. Pinned per-token in
  `tests/standard_names/test_grammar_vocabulary_drift.py::UNREACHED`.
- **Qualifier ordering structure is not exposed.** `load_qualifier_categories()`
  exists but `get_grammar_context()` never returns it. Note precisely what is and
  is not wrong here: the qualifier *tokens* all reach every prompt seat (they are
  qualifier names), so "qualifier_categories is missing from the prompt" is false.
  What is missing is the **category structure**, while the prompt asks for
  *ordered* qualifier stacking — a plausible driver of ordering errors, and it
  matters more while the qualifier class is being decomposed into ordered
  binding-depth segments.

## Release recipe

This is the repeatable catalog workflow proven by the live fork rehearsal. Use
one committed manifest token from graph drain through approval. Immediately
before an operator effect, read the live help for all four commands; the flag
surface below was checked on 2026-09-02 with:

```bash
uv run --no-sync imas-codex sn run --help
uv run --no-sync imas-codex sn release --help
uv run --no-sync imas-codex sn approve --help
uv run --no-sync imas-codex sn resolve --help
```

The command contract is:

| Command | Required role in the chain | Help-backed controls |
|---|---|---|
| `sn run --batch <manifest>` | Resolve the manifest by name or path and drain exactly that cohort through the ordinary pools. | `--batch` is the same identity consumed by release. The default is gap-only; use `--reseed` only when a deliberate full rerun is authorized. |
| `sn release --batch <manifest>` | Mint and freeze the review cohort, export approved plus batch entries, push a review branch to the fork, create the cut-time RC tag, and normally open the review PR. | `--bump {major,minor,patch}` starts a series from a stable tag; `-m/--message` supplies the commit and tag message; `--pr-title` and `--pr-body-file` are a required pair whose validated bytes override generated text; `--notes` asks the release-notes seat for fallback prose; `--no-notes` uses deterministic fallback prose; `--no-pr` still pushes the branch and RC tag but opens no PR. |
| `sn approve --pr <merged-pr-url>` | Resolve the PR number, merge commit, additive baseline, reviewer-edit baseline, and frozen batch from the URL, then fold the reviewed result into the graph. | `--dry-run` reports the routing without graph writes; `--undo` unwinds a completed fold-back; `--notes/--no-notes` controls only the optional human summary below the deterministic receipt block. |
| `sn resolve <name> --override --reason <justification>` | Deliberately accept one contested human proposal over the rubric. | Both `--override` and `--reason` are required, and the justification is stored as `contested_resolution`. Only a contested name is eligible. |

The ordinary happy-path commands are:

```bash
uv run --no-sync imas-codex sn run --batch <manifest>
uv run --no-sync imas-codex sn release --batch <manifest> \
  --bump minor \
  -m "<batch in words>" \
  --pr-title "<short batch title>" \
  --pr-body-file <decisions-only-body.md>
uv run --no-sync imas-codex sn approve --pr <merged-pr-url> --dry-run
uv run --no-sync imas-codex sn approve --pr <merged-pr-url>
```

`sn release --target auto` keeps an RC review on the fork and directs a final
review PR upstream; either way the branch pushes to the fork and upstream
`main` is never pushed directly. Omit `--bump` only when `sn release status`
shows that the current tag already places the checkout in an RC series.

### Detached-worktree setup

Approval and undo need the separate catalog checkout. A detached imas-codex
worktree cannot discover it through the sibling-directory fallback, so bind it
explicitly before either operation; the first live undo proved that omitting
this variable exits before mutation:

```bash
export IMAS_CODEX_SN_ISNC=/home/ITER/mcintos/Code/imas-standard-names-catalog
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
export PYTHONPATH="$PWD"
```

Use the real checkout path on the current host. Pull its merged `main` before
approval, keep it clean, and pass `--isnc <path>` instead when an explicit
one-command binding is clearer. Never infer a credential failure from a missing
catalog path.

### Additive baseline

Catalog `main` starts blank and accumulates approved entries only. The first
review PR is a pure addition over nothing; each later PR is a pure addition over
entries materialized by earlier approvals. The release branch may contain the
approved baseline plus the new frozen batch, but approved baseline entries must
remain byte-identical and therefore disappear from the diff. Never restore the
legacy full-catalog dump or treat it as a comparison base.

After merge, approval materializes only the entries that earned approval onto
catalog `main`. Untouched entries and reviewer edits that pass re-review join
the next baseline; contested or otherwise unapproved bytes are removed. The
merge first parent remains the additive catalog baseline, while reviewer-edit
detection compares the merged content with the cut-time RC tag. Conflating
those two bases made the first blank-baseline approval see zero reviewer edits.

### Pull-request prose

The submitting agent authors the title and body. `--pr-title` and
`--pr-body-file` publish the validated agent text unchanged and take precedence
over `--notes`; the model-backed notes seat and `--no-notes` deterministic text
are headless fallbacks only. The title is a short phrase naming the batch in
words. Use a facility-level title for a multi-domain batch; name a domain only
when the evidence proves that the whole batch has that scope.

Good title and body:

```text
WEST standard names review batch

This WEST standard names review batch publishes 329 entries from the
west_production_dd_paths manifest. Coordinate grids and ordinates are excluded
because they are structural indexing rather than physical quantities. Follow
the catalog REVIEWING.md contract and inspect the names, prose, units, and
physics meaning at the PR-scoped preview before approving:
https://simon-mcintosh.github.io/imas-standard-names-catalog/pr-3/.
```

Banned title and body:

```text
WEST equilibrium, transport, and magnetics entries: plasma_current, q_min, ...

This batch still has 27 source paths without names and 5 unresolved caveats.
Missing entries include plasma_current, electron_temperature, and more.
```

The body is two to five grounded, decisions-only sentences: identify the
facility, source manifest, published count, any deliberate policy exclusion,
the review instruction, a link to catalog `REVIEWING.md`, and the PR-scoped
preview. Resolve problems before cutting the PR. Do not enumerate entries or
narrate missing names, unlinked sources, defects, or unresolved caveats; the
diff is the inventory.

`-m/--message` is operator-owned commit and tag input. The publisher renders a
subject such as `sn: add WEST standard names review batch` and a short body with
aggregate counts and release identity. Neither commit surface enumerates entry
names, paths, or domain files.

### Reviewer contract and preview

The catalog's `REVIEWING.md` is the reviewer authority and every review PR body
links it. Review physics meaning first. Reviewers may change the standard name
and its description or documentation prose; the pipeline, not the reviewer,
enforces grammar, spelling, links, and prose style. Unit, kind, status, source
binding kind/ref/version, generated identity roles, file structure, ordering,
and formatting are machine-owned. Catalog CI rejects a machine-owned-field
change so it cannot silently corrupt provenance.

Catalog CI builds a PR-scoped site containing only entries added or edited by
that PR. A same-repository review is deployed at
`https://<owner>.github.io/<catalog>/pr-<number>/`; an external-fork review gets
an uploaded `catalog-preview-pr-<number>` artifact instead. The workflow
upserts one marker-bearing PR comment with the preview URL and workflow-run
link, updating that comment on later runs instead of appending duplicates.

A branch push retriggers the pull-request build. For an explicit rebuild with
no content change, dispatch the workflow with its exact `pull-request-number`
input:

```bash
gh workflow run catalog.yml \
  --repo <owner>/<catalog> \
  -f pull-request-number=<number>
```

`gh run rerun <run-id>` is also valid for the same commit. Never close or
reopen a live PR to force CI: that churns reviewer notifications, briefly hides
the review surface, and is unnecessary because push, rerun, and
`workflow_dispatch` are supported.

### Cut-time tag and fold-back receipt

`sn release --batch` creates an annotated RC tag on the review-branch head and
pushes it to the fork at cut time. This is the immutable candidate and site
build record, not proof of graph synchronization. `--no-pr` deliberately stops
after that tagged build without opening a PR.

After merge and a successful approval, the same tag ref is replaced on the
merge commit by an annotation whose first line begins `graph-merged:`. Its
deterministic block records the PR, frozen batch, outcome counts, and the prior
tag object as `prior-tag-ref`; optional notes appear below a separator and are
never parsed. The idempotency guard reads the annotation shape, not mere tag
existence: a cut-time tag is allowed, while a `graph-merged:` receipt refuses a
second fold-back.

Undo removes the receipt and force-restores the exact prior cut-time RC object.
A stable release has no cut-time tag, so approval creates its first version tag
as the receipt and undo deletes it. Do not call a plain RC tag proof that the
catalog and graph agree.

### Approval routes, preflight, and undo

Before the first graph write or catalog materialization, approval preflights
every reviewer-edited target. Each must exist with `name_stage='accepted'`,
`docs_stage='accepted'`, no prior PR provenance, and no prior approval. Any
ineligible or unmatched target makes the command fail closed before edit
application, automatic promotions, catalog pushes, or receipt creation. This
gate exists because the earlier partial path promoted 384 entries and pushed a
receipt-less catalog commit before discovering two ineligible edits.

Once the complete preflight passes, `sn approve --pr` uses two routes:

- An untouched batch entry auto-promotes from `accepted` to `approved` with
  the PR number, URL, merge commit, reviewer actor, and approval time recorded.
- A reviewer-edited name or documentation value becomes a human-steered
  proposal and receives the full ordinary re-review without refinement. A pass
  becomes `approved`; a failure becomes `contested`. Both routes retain the PR
  provenance. Resolve a contested result only with `sn resolve <name>
  --override --reason <justification>`; resolution materializes the stored name
  or documentation proposal, sets `docs_stage='accepted'`, and records the
  justification. Never hand-accept or rewrite the graph directly.

For a rehearsal, unwind only after the merge, approval, any explicit contested
resolution, and the frozen-row check have been recorded:

```bash
uv run --no-sync imas-codex sn approve --undo --pr <merged-pr-url>
```

Undo demotes names approved by that PR and contested names in its frozen batch
to `accepted`, restores `docs_stage='accepted'`, and clears all catalog PR
provenance. An approved row with missing PR provenance is included only when
its identity belongs to the same frozen batch; a null-provenance non-member is
unchanged. The catalog correction is inverted before the receipt disappears,
then the RC tag is restored or the stable receipt is deleted.

Undo does not un-apply accepted reviewer wording, erase its internal change
record, or rewrite the immutable frozen review manifest. Revert wording through
`sn edit`. Undo is scoped to the named PR and must not disturb another PR's
approval. A receipt-less catalog materialization from old or broken code cannot
be inferred safe from graph state; it requires an explicitly authorized Git
inverse followed by complete-tree equality, as the recorded inverse report
demonstrates.

### Recorded rehearsal runbook

- **Prepare.** Read all four live help surfaces; verify the catalog checkout,
  fork/upstream remotes, GitHub authentication, clean branches, manifest token,
  and detached-worktree environment. Export and hash a graph restore point
  before the first live graph mutation.
- **Drain and cut.** Run `sn run --batch` gap-only until the release gates and
  complete source accounting pass. Author the title and decisions-only body,
  then run `sn release --batch` with the required bump, message, title, and body
  file. Confirm the branch and RC tag exist only on the fork and the PR diff is
  additive.
- **Inspect.** Require the validation and reviewer-edit guard checks, open the
  PR-scoped preview, and confirm its visible identities equal the PR additions
  and edits. Make only reviewer-owned changes. Use the dispatch input above for
  a no-content rebuild; never cycle PR state.
- **Merge and fold back.** Merge only after the checks pass, pull catalog
  `main`, run approval dry first, then run live approval. Confirm the untouched,
  edited-pass, and contested counts; verify every folded or contested row has
  complete PR provenance; verify the receipt begins `graph-merged:`; and prove
  a repeated approval is refused.
- **Adjudicate.** For each contested proposal, either leave it frozen or run
  `sn resolve --override --reason` with a substantive human justification.
  Confirm that the exact reviewed proposal, accepted docs stage, resolution,
  and provenance are present afterward.
- **Freeze-check.** Snapshot all approved and contested rows, run an ordinary
  unscoped pipeline drain with a conservative cost cap, and compare the same
  projection afterward. Non-frozen work must execute while every frozen row is
  byte-identical.
- **Unwind only a rehearsal.** Run `sn approve --undo`, then require approved
  and contested counts to return to their baseline, all frozen-batch PR fields
  to be null, the receipt to be absent, the exact cut-time RC tag to be back,
  the fork tree to equal the merged-PR baseline, the review manifest to be
  byte-identical, and upstream to be unchanged. A real accepted release stops
  before this bullet.

The fully repaired live cycle produced **384 auto-approved plus 2 contested**.
After one contested override, undo reverted **385 approved plus 1 contested**.
It finished with 0 approved, 0 contested, 2,336 accepted globally, all five PR
fields null on all 409 frozen-batch rows, fork-tree equality with the PR merge,
the receipt removed, the cut-time RC tag restored, and upstream unchanged.

### Rehearsal evidence

These are the durable reports, including failures that changed the recipe:

- [`crew/reports/west-stage3-restore-point.md`](../../crew/reports/west-stage3-restore-point.md) — pre-write graph archive, hash, and identical before/after census.
- [`crew/reports/west-stage3-review-merge.md`](../../crew/reports/west-stage3-review-merge.md) — two reviewer edits, three successful checks, fork merge, and zero upstream delta.
- [`crew/reports/west-stage4-approve.md`](../../crew/reports/west-stage4-approve.md) — initial 374-plus-2 fold-back, receipt/idempotency proof, and the missing contested-provenance defect.
- [`crew/reports/west-stage4b-resolve.md`](../../crew/reports/west-stage4b-resolve.md) — initial override transition and the defect where accepted wording was not materialized.
- [`crew/reports/west-stage5-freeze-check.md`](../../crew/reports/west-stage5-freeze-check.md) — frozen-row mutation under ordinary pool work.
- [`crew/reports/west-stage5-freeze-check-2.md`](../../crew/reports/west-stage5-freeze-check-2.md) — blocked repeat with 1,343 parameter-binding failures; a zero diff was correctly rejected as vacuous.
- [`crew/reports/west-stage5-freeze-check-3.md`](../../crew/reports/west-stage5-freeze-check-3.md) — passing freeze proof: 376 rows by 19 fields unchanged while 51 non-frozen names changed.
- [`crew/reports/west-stage6-undo.md`](../../crew/reports/west-stage6-undo.md) — first undo, detached-worktree ISNC requirement, exact RC-tag restoration, and the provenance-less residual.
- [`crew/reports/west-stage6b-repaired-cycle.md`](../../crew/reports/west-stage6b-repaired-cycle.md) — partial-promotion failure that required the all-target eligibility preflight and docs-stage restoration.
- [`crew/reports/west-catalog-main-inverse.md`](../../crew/reports/west-catalog-main-inverse.md) — authorized fork-only inverse of the receipt-less materialization, with complete-tree equality and zero upstream or graph delta.
- [`crew/reports/west-stage6c-final-cycle.md`](../../crew/reports/west-stage6c-final-cycle.md) — final 384-plus-2 approval, contested resolution, 385-plus-1 undo, provenance clearance, tag/receipt behavior, manifest identity, fork-tree equality, and upstream isolation.

## The catalog is not the source of truth

The graph plus the review pipeline is. Catalog-origin names
(`origin='catalog_edit'`, `source_types=['catalog']`) came from one bulk import of
this pipeline's own earlier output and sit at `status='draft'`; they are an old
method to be resolved through review, not authoritative content to protect.

This makes the `origin='catalog_edit'` exemption in
`supersede_prior_source_names` a defect rather than a safeguard: it stops the
one-live-name-per-source dedup from folding an imported name, which is why one
coordinate axis resolved to a single name while another stayed split across two.

## The refinement budget is charged per ATTEMPT

`chain_length` is lineage depth: it counts successors that PERSISTED. A refine
attempt can fail before any write — the proposed identity is already taken
(`find_name_key_duplicate`), the persistence fence refuses the successor, the
candidate fails grammar validation — and every one of those leaves lineage
depth untouched. Gating claim eligibility, escalation, or exhaustion on it
therefore re-selects the same name on every poll and re-bills it: one measured
run spent 822 model calls and $54.7 on 14 names and produced one accepted name.

`refine_attempts` is the budget. It is charged on each verified claim, before
the model call, and a successor inherits it so the cap bounds the lineage. Read
it through `REFINE_NAME_ATTEMPTS_SPENT`, never `chain_length`.

Two consequences worth keeping straight:

- **A collision is decided, not transient.** The refiner proposes the same
  successor identity every cycle — measured across two model vendors — and
  refinement may not take an occupied identity, because merging carries
  source-migration semantics that belong to `sn edit`. `stop_refine_name_attempt`
  parks such a name immediately with `refine_collision_name` recorded. A model
  or provider error is the opposite case and keeps the rotations that remain.
- **Recovery differs by whether new information arrived.** `sn rescore` buys a
  fresh quorum draw on the SAME name and deliberately does NOT refund rotations
  (refunding re-opens the paid loop for a name that scores low again); a
  name-steering `sn edit --hint` is new information and refunds them.

The docs axis has no such defect and needs no counter: `persist_refined_docs`
rewrites in place, so every attempt lands and `docs_chain_length` always
advances. Do not "harmonise" the two axes by giving docs an attempt counter.

## Acceptance

Never hand-accept, and never edit graph text with Cypher. Acceptance is earned
only through the RD-quorum review pool; corrections go through `sn edit`
(`--hint` to steer regeneration, `--rename`/`--docs` to replace and go straight
to review). `--stage-only` stages many for one batch review — the budget-efficient
path for a bulk repair, since compose is free and review is the only paid stage.

The single sanctioned structural accept is `ENRICH_PARENTS` for placeholder
derived parents, which the quorum systematically penalises for being abstractions.

## Exact-source compose steering

Use `sn source-hint <exact-dd-path> --hint ... --reason ...` only before an
eligible DD source without a live name binding composes. It is not a source
mutation: DD snapshot, unit, COCOS, domain, identity, and grammar validation
remain authoritative. Preview with `--dry-run`; replacing an open hint requires
`--replace`.

The write is claim-fenced compare-and-set. A missing or non-DD source, active
claim, non-`extracted` state, attempt cap, live binding, or unacknowledged open
hint is a refusal, not a reason to bypass the CLI with Cypher. An open hint rides
pooled compose and is consumed atomically only when that exact source binds a
name. Retry, failure, interruption, claim expiry, validation rejection, and
collision preserve it. Use `sn run --focus <same-exact-path>` to fence the
subsequent work; model choice comes from the configured compose seat unless the
run explicitly overrides it.

Do not confuse the three operator surfaces: `sn retry --reason` audits why a
blocked source may attempt again, `sn source-hint` steers the next successful
binding of one exact source, and `sn edit <name> --hint` steers an existing name
through review. `rejected` is reserved in the hint-status schema; no current CLI
transition sets it.
