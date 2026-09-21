# What the review pipeline can see, and the seven gaps between it and a physicist reading the DD

provisional: true

Read: `imas_codex/standard_names/review/{pipeline,audits,consolidation,projection,enrichment}.py`
and the `sn review` surface in `imas_codex/cli/sn.py`.

**Headline: four new instances of the fail-open shape, three of them CONFIRMED,
and one of those removes the reviewer's only view of the Data Dictionary text
for a whole batch while logging at `debug`.** Of the five defect classes the
independent WEST audit found, **one is detectable today, two are detectable only
through a channel that can silently vanish, and two are structurally
undetectable by any layer.**

The two already-confirmed defects — `pipeline.py:1709`'s `getattr(audit_report,
"findings", [])` and the `nearby_existing_names` cap in `workers.py` near 5608 —
are context here, not findings. What follows is what else carries the same shape.

## 1. The three layers: what each computes, and what it cannot

### Layer 1 — deterministic audits (`audits.py`)

Computes four things over the in-memory name dicts: an embedding freshness
preflight, a lexical lint (ISN parse/compose round-trip, processing-verb
suffixes, mixed position forms within a domain), link integrity, and near-duplicate
detection by blocking plus similarity.

**What it structurally cannot detect.** Every Layer 1 audit reads only the
name's own record — `id`, `description`, `documentation`, `unit`, `kind` and the
grammar segment fields (`audits.py:139-175`). **No Layer 1 audit ever reads the
Data Dictionary path or its text.** So no deterministic check can compare what a
name asserts against what the DD says, which is the comparison three of the five
WEST defect classes require. This is a property of the input, not of the checks:
adding a check would not help without adding the DD text to what Layer 1 is
given.

Two further structural limits, both in `run_duplicate_detection`:

- **Comparison never crosses a block.** Blocking keys are
  `unit|kind|physics_domain`, `unit|kind|physical_base` and
  `unit|kind|geometric_base` (`audits.py:528-552`). The semantic search result is
  then filtered by `rid in block_ids` (`audits.py:606`), so a neighbour found
  anywhere else in the catalog is discarded. Two spellings of one quantity that
  sit in different physics domains — which is the normal case for a redundant
  spelling, since the redundancy usually arises from two diagnostics naming the
  same physical thing — can never be compared.
- **Both thresholds are set where near-duplicates do not live.** Semantic
  acceptance is `score > 0.92` (`audits.py:608`) and lexical acceptance is
  `_token_overlap > 0.8` (`audits.py:625`), a Jaccard ratio over snake tokens.
  Distinct spellings of one quantity routinely share no tokens at all.

### Layer 2 — batched LLM scoring (`pipeline.py`)

Scores each candidate on four dimensions — `grammar`, `semantic`, `convention`,
`completeness` for the names axis, and `description_quality`,
`documentation_quality`, `completeness`, `physics_accuracy` for docs
(`pipeline.py:2205-2212`). It is the only layer that reads the DD text, through
`_fetch_review_dd_context` (`pipeline.py:640`), which attaches `dd_source_docs`,
`version_notes`, hybrid neighbours and related paths.

**This is where the pipeline's real capability lives** — and where finding 1
below shows it can be removed without a trace.

### Layer 3 — consolidation (`consolidation.py`)

`detect_convention_drift` (`consolidation.py:170`) finds mixed position forms,
inconsistent process suffixes and documentation-length variance within a domain.
`detect_score_outliers` (`consolidation.py:307`) reports names scoring more than
1σ **below their group mean**.

**What it structurally cannot detect: anything Layer 2 scored uniformly.** Layer
3 has no independent access to a name, a DD path or a catalog; its entire input
is the review dicts Layer 2 produced. A blindness that is *systematic* — every
member of a class scored alike because the evidence that would separate them was
absent — produces a uniform column, and a variance detector reports nothing from
a uniform column. **Layer 3 is loudest exactly when Layer 2 is working and
silent exactly when Layer 2 is blind.** That is the shape behind the measured
outcome: 74 defects in 341 bindings, and the pipeline's own review passed all of
them without a single consolidation warning.

## 2. The five WEST defect classes against the layers

| WEST defect class | Detectable in principle? | Through which layer | Why |
|---|---|---|---|
| Redundant spelling of a registered base — `hard_xray_brightness` vs `photon_radiance`, same unit | **No, as built** | would have to be Layer 1 duplicate detection | the two sit in different physics domains, so the block filter at `audits.py:606` discards the comparison before any threshold applies; even same-block, 0.92 semantic / 0.8 lexical are above where distinct spellings score, and the token sets are disjoint |
| Name asserts more than the DD text supports — `surface_temperature` on a path reading *apparent temperature* | **Yes — Layer 2 only** | `dd_source_docs` in the reviewer prompt | requires comparing the name against DD text; only `_fetch_review_dd_context` supplies it, and finding 1 shows that supply can vanish silently |
| Name bound to the wrong object — `area_of_diagnostic_aperture` on a detector/surface path | **Yes — Layer 2 only** | `dd_source_docs` plus the DD path itself | same channel, same exposure |
| Physics error in the modifier — `radial_derivative_of_poloidal_magnetic_flux` on `dpsi_drho_tor` | **No** | none | needs the DD path's *coordinate* to be read as physics (ρ_tor is not a radius). No layer compares the modifier against the coordinate; Layer 1 does not see the path, Layer 2's rubric has no dimension for it on the names axis — `physics_accuracy` exists only on the **docs** axis (`pipeline.py:2205-2210`), so a names-axis review is never asked the question |
| One identity bound to loci of different kinds | **No** | none | requires reasoning across the identity's several `source_paths` at once. Every Layer 1 audit is per-name-record, and the prompt presents the DD docs as a list without asking whether they are the same kind of object |

The middle two are the recoverable ones, and both depend on a single function
whose failure mode is silence.

## 3. Findings

### Finding 1 — CONFIRMED. A DD fetch failure removes the reviewer's only view of the Data Dictionary, at `debug` level

`imas_codex/standard_names/review/pipeline.py:737-740`

```python
except DDResolutionError:
    raise
except Exception:
    logger.debug("Review DD source fetch failed", exc_info=True)
```

`path_docs` is then empty. At `pipeline.py:786-788` the consumer reads

```python
docs = [path_docs[p] for p in sp if p in path_docs]
if docs:
    item["dd_source_docs"] = docs
```

so on failure **`dd_source_docs` is never set at all** — not set empty, not set
to a marker. The item carries no key, the prompt renders no DD text, and Layer 2
scores the batch at full cost as though the Data Dictionary had been consulted
and had nothing to add. Nothing downstream tests for the key's presence.

**The failure is doubly silent, and the asymmetry is the tell.** A *partial*
omission is loud: `batch.refusals` is logged at **warning** with a count and
every path (`pipeline.py:711-719`). A *total* failure is logged at **debug**. The
louder signal covers the smaller failure.

**Blast radius is larger than the DD text.** The hybrid-neighbour queueing is
nested inside the same `if docs:` block (`pipeline.py:797-800`), so one swallowed
exception also removes the hybrid comparator channel for every item in the batch.

**Concrete scenario.** A Neo4j timeout, a `None` row, or any schema change to
`IMASNode` that breaks the projection at `pipeline.py:688-694` fires the bare
`except`. A 25-item batch is scored with no DD text and no hybrid neighbours,
returns plausible per-dimension scores, and the reviewer accepts names the DD
contradicts. This is precisely the channel WEST defect classes 2 and 3 depend on,
so the observable outcome is the measured one: names asserting more than the DD
supports, passing review.

**Would have to change:** `pipeline.py:737-740` must record the failure on the
report rather than swallow it, and `pipeline.py:786-788` must distinguish "no
source paths resolved" from "the fetch did not run". A batch whose DD context did
not load must refuse before any LLM call is charged.

### Finding 2 — CONFIRMED. Version-history loss is swallowed by the same pattern

`imas_codex/standard_names/review/pipeline.py:769-770`

```python
except Exception:
    logger.debug("Review version history fetch failed", exc_info=True)
```

`path_versions` stays empty, `vnotes` is empty, `version_notes` is never set. The
reviewer loses `units`, `sign_convention`, `cocos_transformation_type` and
`definition_clarification` changes on the DD path — exactly the notes that decide
whether a name's modifier still matches the current DD meaning. Same shape,
narrower consequence than finding 1, and the same repair.

### Finding 3 — CONFIRMED. Duplicate detection cannot tell a clean semantic pass from a failed one

`imas_codex/standard_names/review/audits.py:616-617`

```python
except Exception:
    logger.debug("Semantic search failed for '%s'", nid, exc_info=True)
```

Per name, at `debug`. If the embedding service or the graph is unreachable, every
call fails, `candidate_pairs` is populated by lexical overlap alone, and
`run_duplicate_detection` returns a result **indistinguishable from a genuine
semantic pass that found nothing**. `DuplicateComponent` and `AuditReport` carry
no field recording whether the semantic path ran.

**The comment at `audits.py:570` names a guard that does not exist:**

```python
# Quick probe — don't fail if graph is down
search_fn = search_standard_names_vector
semantic_available = True
```

There is no probe. `semantic_available` is set from the *import* succeeding, and
the import succeeds whether or not the graph is up. So the flag reports module
availability while its comment claims service availability, and the actual
service failure is discovered per name and discarded.

**Concrete scenario.** The embedding server is down during a review. Layer 1
reports zero duplicate components. A reader — and the release gate — take that as
"no redundant spellings in this cohort". The correct reading is "the check did
not run".

### Finding 4 — CONFIRMED. A lint report cannot say whether the round-trip check ran

`imas_codex/standard_names/review/audits.py:308-315`

```python
except ImportError:
    logger.debug("imas_standard_names not available — skipping round-trip checks")
```

The grammar round-trip is the only Layer 1 check that can reject a malformed
name. When `imas_standard_names` is absent it is skipped and **no finding records
the skip**, so an empty `lint_findings` list means either "every name round-trips"
or "the round-trip was never attempted". The docstring says the function "falls
back to lightweight heuristic checks", but the remaining checks — processing-verb
suffix and mixed position forms — test something else entirely; there is no
fallback round-trip.

Marked CONFIRMED as a shape rather than as a live exposure: `imas_standard_names`
is a declared dependency, so in a synced environment the branch does not fire.
`AuditReport` still has no field that would let a reader tell.

### Finding 5 — PLAUSIBLE. Nothing gates on the embedding preflight's own result

`imas_codex/standard_names/review/audits.py:674-703`

`run_all_audits` calls `run_embedding_preflight` first, explicitly to ensure
"fresh embeddings for duplicate detection", stores the report, and then calls
`run_duplicate_detection` **unconditionally**. `embedding.missing_count` and
`embedding.stale_count` are carried into `AuditReport` and read by nothing. A
cohort whose embeddings are missing degrades to lexical-only duplicate detection
— the finding-3 outcome reached by a different route — and the preflight that
measured the problem does not stop it.

### Finding 6 — PLAUSIBLE. The names-axis rubric has no physics dimension

`imas_codex/standard_names/review/pipeline.py:2205-2212`

The docs axis scores `physics_accuracy`; the names axis scores `grammar`,
`semantic`, `convention`, `completeness`. A name whose modifier is physically
wrong — `radial_derivative_of` on a `dpsi_drho_tor` path, where ρ_tor is a
normalised flux-surface label and not a radius — is grammatical, semantically
coherent as a phrase, conventional, and complete. **It can score full marks on
every dimension the names axis has.** The reviewer is never asked the question
that would fail it.

Marked PLAUSIBLE because a model may dock `semantic` on its own initiative; what
is CONFIRMED is that no dimension names the question, so nothing in the score
record distinguishes a physics check that passed from one never made.

### Finding 7 — PLAUSIBLE. No layer reasons across an identity's several source paths

An identity bound to loci of different kinds is a property of the *set* of
`source_paths`, not of any one of them. Layer 1 audits are per-name-record and do
not see paths at all; `_fetch_review_dd_context` attaches `dd_source_docs` as a
flat list and the prompt does not ask whether the entries describe the same kind
of object; Layer 3 sees only scores. The capability is absent rather than broken,
so it is a gap to build, not a repair.

## 4. The shape, stated once

Findings 1 through 5 are the same defect wearing five costumes: **a lookup
returns empty, and the caller proceeds as though it had looked.** In every case
the empty result is representable — an empty `path_docs`, an unset
`dd_source_docs`, an empty `candidate_pairs`, an empty `lint_findings`, a
`missing_count` nobody reads — and in no case is "the check did not run"
distinguishable from "the check found nothing".

The correction is the same in all five: **a report must carry whether each check
executed, and a review must refuse rather than score when a channel it depends on
did not load.** Counting seven sites of one shape in one pipeline says the
individual repairs are not the durable fix; a report field that every audit must
fill is.
