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

Use one manifest token through the complete catalog-review cycle. Check the
live `--help` before operating because it is authoritative for optional flags;
the deterministic chain is:

```bash
uv run imas-codex sn run --batch <manifest>
uv run imas-codex sn release --batch <manifest> --bump minor -m "<batch in words>"
uv run imas-codex sn approve --pr <merged-pr-url>
```

`sn run --batch` drains the manifest through the ordinary pipeline. The
gap-only default preserves sources that already have a live accepted or
approved name. `sn release --batch` freezes the cohort, exports it, pushes the
review branch to the fork, and opens the review PR. After human review and
merge, pull catalog `main`, then `sn approve --pr` resolves the merge and frozen
batch from the URL and folds the result back into the graph.

### Additive baseline

Catalog `main` starts blank and accumulates approved entries only. A review PR
is therefore a pure addition over the entries approved by earlier PRs; it is
not a new full-catalog dump. Approval materializes the newly approved entries
onto `main`. Those entries become part of the next PR's base, so they are
byte-identical and invisible in later diffs. Accepted-but-unapproved and
contested entries never enter that baseline. Never restore a legacy catalog
dump, carry every accepted entry into a review branch, or use an earlier dump
as the comparison base.

### Commit and PR prose

The submitting agent authors the PR title and body at release time, just as it
authors a commit message. Agent-authored text wins after contract validation.
The notes seat and deterministic template are headless fallbacks only; they do
not displace an agent preparing a review PR. Check the live `sn release --help`
for the current agent-facing title and body inputs. Its `--notes / --no-notes`
switch remains the documented synthesized-versus-deterministic fallback seam.

Use a facility-level title for a multi-domain batch. A dominant domain must not
make a broad batch look domain-scoped. A genuinely single-domain batch may name
that domain when the release evidence proves the scope.

Good PR title:

```text
WEST standard names review batch
```

Banned PR title:

```text
WEST equilibrium, transport, and magnetics entry update
```

The PR body is decisions-only prose: one paragraph of two to five grounded
sentences stating what the batch is (facility, source manifest, and published
count), the review decisions that materially shaped it, the review instruction,
and the PR-scoped preview link. Problems are resolved before release, not
narrated in the PR. Do not report missing names, unlinked-source counts,
unresolved caveats, or defect lists. A policy decision may identify a semantic
class when the distinction explains what the batch deliberately excludes.

Good PR body:

```text
This WEST standard names review batch publishes 337 entries from the
west_production_dd_paths manifest. Coordinate grids and ordinates are excluded
by policy because they are structural indexing, not physical quantities.
Review the names, documentation, units, and physics meaning in the PR-scoped
preview before approving: https://example.invalid/review/current/.
```

Banned PR body:

```text
This batch still has 27 source paths without names and 5 unresolved caveats.
Missing entries include plasma_current, electron_temperature, and more.
```

The operator-owned commit subject comes from the release message supplied with
`-m/--message`, which the live help identifies as commit input. The publisher
renders it as `sn: add <message>` and writes a short body with aggregate counts
and release identity. The subject and body never enumerate entry names, paths,
or domain files.

Good commit:

```text
sn: add WEST standard names review batch

Publish 337 approved candidates from the WEST production manifest.
Release identity is recorded in the frozen review artifact.
```

Banned commit:

```text
sn: update equilibrium.yml, transport.yml, magnetics.yml (plasma_current, ...)
```

For the batch recipe, an RC tag is created on the review-branch head and pushed
to the fork when `sn release --batch` cuts the candidate. This preserves each
candidate and its site build as a historic release record. Pass `--no-pr` to
cut, tag, and build an in-work RC without opening a pull request; the review
branch and RC tag still go only to the fork.

After a PR merges and `sn approve --pr` successfully folds it into the graph,
approval replaces that RC ref with the `graph-merged:` receipt on the merge
commit. The receipt shape, not tag existence alone, means catalog and graph are
synchronized. `sn approve --undo` removes that receipt and restores the exact
RC tag that existed before approval. Stable releases differ: `sn release
--final` creates no release-time tag, so their first version tag is the
fold-back receipt created by `sn approve`.

### Approval routes and undo

`sn approve --pr` has two integration routes:

- An unedited batch entry auto-promotes from `accepted` to `approved` with the
  PR number, URL, merge commit, and approval time recorded as provenance.
- A reviewer-edited name or documentation record re-enters the ordinary review
  pipeline as a human-steered proposal. A passing review becomes `approved`; a
  failing review becomes `contested` and is frozen for human disposition. Only
  `sn resolve <name> --override --reason <justification>` may approve a
  contested result. Never hand-accept it or bypass the recorded reason.

Undo the fold-back with:

```bash
uv run imas-codex sn approve --undo --pr <merged-pr-url>
```

Undo removes that PR's approval provenance, returns its auto-approved entries
to `accepted`, returns its contested entries to `accepted`, and removes the
fold-back receipt locally and remotely. For an RC it restores the exact cut-time
tag; for a stable release it deletes the receipt tag. Accepted human edits
remain graph history; revert wording through `sn edit`, never by rewriting
graph properties. Undo is scoped to the named PR and does not disturb approvals
owned by another PR.

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
