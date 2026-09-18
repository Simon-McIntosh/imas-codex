# Readers of the `StandardName.origin` property at HEAD

Census at worktree HEAD `4c9d1dabb4ab0afb679ccf15b85eb552e585169c` (2026-09-18).
The prior inventory was measured 2026-09-08 at revision
`271536947e417ddb36a3091c1559fc606f62c475` (`origin-label-audit.md`, commit
a7dc41275); its line numbers have drifted since, so every anchor below is
re-derived at HEAD by exact line-text matching against the prior revision
(instrument: `/tmp/pidp-origin-reads/reanchor.py`, output
`/tmp/pidp-origin-reads/reanchor.json`).

Read forms covered: the attribute/Cypher property form `\.origin\b` and the
inline-predicate form `origin: 'derived'` inside a `StandardName` match
pattern. A plain `origin` substring search also matches three unrelated things
— a git remote, the `ChangeEvent` origin field in `promote.py` /
`provenance_lifecycle.py`, and the `[origin:FROM_DD_PATH]` relationship binding
in `signed_manifest.py` — none of which is the StandardName property, and all
of which the counts below exclude.

## Line count against the plan figure audit's 75

The figure audit (`plan-figure-audit.md:129`) asserts "75 lines under
`imas_codex/standard_names/` still read it". That figure reproduces exactly at
HEAD:

```bash
grep -rnE 'origin.*derived|derived.*origin' imas_codex/standard_names/ --include=*.py | wc -l
# 75
```

The proxy is a line match on two words, not a read count: it includes direct
assignments that write `origin='derived'`, comments, docstrings, and
cross-node `origin` fields. The semantic census below lists **54 distinct read
lines at HEAD** — 52 feeding non-deletion decisions (the 34 decision paths of
the 2026-09-08 census, re-anchored, plus two sites that census did not list:
`workers.py:9973` and `signed_manifest.py:371`), and 2 feeding the
structural-cleanup deletion (the §7a retirement target). The difference
between 75 and 54 is the proxy's blind spot in both directions (writes and
comments inflate it; reads that compare against a bound parameter rather than
the literal `'derived'` escape it); a semantic read inventory cannot be
recovered from a two-word line grep.

## Reader inventory — every read at HEAD

Non-deletion decisions, one clause each. Format: `file:HEAD lines` — decision.

- `imas_codex/cli/sn.py:398,430,441,485,496` — `_compute_pool_progress` counts a derived parent as docs-eligible without a name score, excludes it from name-review and refinement counts, and reports placeholder enrichment progress separately.
- `imas_codex/standard_names/export.py:1099` — `_run_gate_c` lets a derived name bypass the ordinary name-axis score gate, so the label sets catalog eligibility (`if cand.get("origin") in ("derived", "catalog_edit")`).
- `imas_codex/standard_names/edit.py:745` — `_stranded_rename_refusal` refuses a manual rename because the name is assumed to follow deterministically from a grammar peel over children.
- `imas_codex/standard_names/parents.py:239` — `is_single_child_shadow`: a derived child is ineligible to prove its sole parent a redundant single-child shadow.
- `imas_codex/standard_names/parents.py:400,425` — `_replay_described_parent_authorities` selects and transactionally rechecks accepted, described, unscored derived names for structural-authority backfill.
- `imas_codex/standard_names/review/pipeline.py:621,631` — `_fetch_review_derived_children` fetches live children and adds a structural-peel explanation to review prompts only for derived-labelled rows.
- `imas_codex/standard_names/workers.py:4835` — `validate_name_candidate` appends the derived-parent structural audit to the deterministic validation result.
- `imas_codex/standard_names/workers.py:8436` — `process_review_name_batch` routes low-similarity derived rows to documentation refinement, skipping the ordinary semantic name-similarity path.
- `imas_codex/standard_names/workers.py:9123` — `_enrich_for_docs_gen` grounds documentation generation on live child names and suppresses placeholder child prose.
- `imas_codex/standard_names/workers.py:9966` — `_load_docs_review_parent_children` loads child grounding for documentation review only when the row is labelled derived.
- `imas_codex/standard_names/workers.py:9973` — enrich-parents claim winner recheck: `MATCH (p:StandardName {id: $id, origin: 'derived'})` (read site the 2026-09-08 census did not list).
- `imas_codex/standard_names/workers.py:10616` — `process_refine_docs_batch` injects live child context into documentation refinement only for derived-labelled parents.
- `imas_codex/standard_names/attachment_audit.py:2446` — `detach_one_attachment` treats a derived-labelled name with live children as structurally anchored, so its last DD realization can be detached without the ordinary orphan refusal.
- `imas_codex/standard_names/graph_ops.py:1120` — `_MANIFEST_DRAIN_PLAN_QUERY` finds derived-labelled ancestors and propagates a bounded source drain scope to them.
- `imas_codex/standard_names/graph_ops.py:2864` — `_is_single_child_shadow` (batch-aware): a lone derived child is refused as evidence of a redundant parent shadow.
- `imas_codex/standard_names/graph_ops.py:3703,3753` — derived-parent candidate queries admit null-origin or derived-origin childful names to the seedable and legacy parent-materialisation paths.
- `imas_codex/standard_names/graph_ops.py:4731` — `normalize_derived_parent_lifecycle` unit-gap selector: accepted derived names with accepted docs but no unit are selected for derived-parent unit repair.
- `imas_codex/standard_names/graph_ops.py:7722` — `persist_generated_name_winners` propagates the source drain scope from a newly persisted name to all derived-labelled ancestors.
- `imas_codex/standard_names/graph_ops.py:14323,14392` — `reconcile_reviewable_name_stage` excludes derived names from being lifted into ordinary name review even when a producer and valid description exist.
- `imas_codex/standard_names/graph_ops.py:16744,18034` — `reconcile_descriptionless_composed_names` and `REFINE_NAME_ELIGIBILITY_WHERE` exclude derived names from the composed-name description repair and from name refinement (shared predicate text at both anchors).
- `imas_codex/standard_names/graph_ops.py:17518,24968,25268,25970` — review/refinement eligibility and claim selectors: `REVIEW_NAME_ELIGIBILITY`-family predicates make every derived name ineligible for name review, while `claim_review_docs_batch`, `claim_generate_docs_batch` and `claim_refine_docs_batch` accept derived childful names without a name score into paid docs work.
- `imas_codex/standard_names/graph_ops.py:18552,18706,18832,18836` — refined-name persistence refuses a derived existing identity as a reusable refinement target and prevents an on-match refinement from drafting or relabelling it.
- `imas_codex/standard_names/graph_ops.py:19521,19858` — generated supersession excludes derived-labelled predecessor identities from automatic source migration and supersession.
- `imas_codex/standard_names/graph_ops.py:22476,22549,22694` — exhausted-orphan supersession refuses derived-labelled rows and limits automatic supersession to non-derived identities.
- `imas_codex/standard_names/graph_ops.py:25985,25991` — `pool_pending_counts` treats derived childful names as docs-eligible without a name score and excludes them from the name-review and name-refinement backlog counts.
- `imas_codex/standard_names/graph_ops.py:26146` — `_verify_enrich_parents_claim_winners` keeps a claimed enrichment winner only if it remains a derived-labelled placeholder under the worker's claim token.
- `imas_codex/standard_names/graph_ops.py:26195` — `claim_enrich_parents_batch` selects derived-labelled placeholder parents with live children for paid description enrichment.
- `imas_codex/standard_names/graph_ops.py:26481,26629,26713,26764` — structural-authority persistence and acceptance: the parent must still be labelled derived by default; `persist_enriched_parent` refuses atomic structural acceptance otherwise; and structural acceptance routes any childful derived-labelled name through it even when a live producer exists, with the wider selector also admitting source-free null-origin parents.
- `imas_codex/standard_names/graph_ops.py:27010` — `classify_orphan_parent_source_candidates` applies structural-admission validation only to derived-labelled provenance-orphan parents; non-derived parents bypass that check.

Deletion decisions reading the property (the §7a/§3 deletion class):

- `imas_codex/standard_names/graph_ops.py:3810` — derived structural-cleanup selector: `WHERE parent.origin = 'derived'` picks parents to reap; the ceiling at 3838 bounds it (delete path).
- `imas_codex/standard_names/graph_ops.py:4160` — records rejected derived candidates for that same cleanup (delete path).

Sites the 2026-09-08 census did not list, added by this census — `imas_codex/standard_names/workers.py:9973` and `imas_codex/standard_names/signed_manifest.py:371` (`MATCH (sn:StandardName {unit: '1', origin: 'derived'})` — the unit-gap structural signing query). Both are non-deletion reads.

## Excluded lines (not reads for a decision)

- Writes only: `graph_ops.py:4215,4404,8835,26813` (`parent.origin = 'derived'` assignments).
- Computed target classification rather than stored origin: `graph_ops.py:13866`.
- `desc_name_sim.py:5` — docstring describing the threshold; no runtime read.
- Non-StandardName `origin` identifiers: git remote strings in `catalog_release.py`/`publish.py`, `origin: $change_origin` (ChangeEvent) in `promote.py`, deletion-record `origin` in `provenance_lifecycle.py`, and `[origin:FROM_DD_PATH]` relationship bindings in `signed_manifest.py`.

## Implication for §7a stage 2

Every non-deletion branch above must read topology (the presence of a
`PRODUCED_NAME` producer, the `StandardNameSource` binding) instead of the
scalar before the field can be dropped without changing behaviour silently.
Two of them change safety posture when the label is false: `attachment_audit.py:2446`
can strip a DD realization's last-source protection, and `graph_ops.py:26713`
can structurally accept a DD-produced identity. Count: the 34 decision paths of
the 2026-09-08 census carry 52 distinct non-deletion read lines at HEAD, the
deletion class carries 2 predicate reads (`graph_ops.py:3810,4160`), and the
census total is 54 distinct read lines. One further anchor, `graph_ops.py:4792`,
is a comment documenting the second cleanup selector rather than a read.