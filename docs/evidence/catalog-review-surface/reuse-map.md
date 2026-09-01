# Approval fold-back reuse map

## Outcome

The approval fold-back is already assembled behind `imas-codex sn approve`; the reusable implementation entry point is `imas_codex/cli/sn.py:sn_approve`, which resolves a merged PR and delegates the graph transition to `imas_codex/standard_names/promote.py:run_approval`.

The existing machinery is sufficient to exercise PR resolution, diff extraction, accepted-name routing, contested routing, PR provenance, catalog correction, and receipt-tag lifecycle. It is **not sufficient by itself to prove the plan's full “no lasting graph change” claim after an accepted human edit**: `promote.py:undo_approval` deliberately keeps accepted rename/docs history. A full-state rehearsal therefore needs an isolated graph profile restored from a pre-exercise archive (or explicit authority to restore the production graph archive), not only `sn approve --undo`.

There is also a traceability gap against the plan's literal “who changed it” requirement. The graph records that an edit is human-originated (`StandardName.edit_origin = "human"`) and records the PR number, URL, merge SHA, and approval time, but it does **not** record the GitHub reviewer's login. `resolve_merged_pr` does not request a PR author, and `mark_catalog_name_approved` has no actor parameter or actor field to write. `fetch_pr_evidence` reads comment/review authors only for the human-readable tag summary; it does not persist them to the graph.

## Reuse candidates

| # | Existing mechanism | Location and symbol | One-line fitness verdict |
|---:|---|---|---|
| 1 | Approval CLI orchestration | `imas_codex/cli/sn.py:sn_approve` | **Direct fit:** canonical end-to-end entry point; supports PR-only resolution, dry-run, approval, receipt creation, and undo. |
| 2 | Merged-PR resolution | `imas_codex/standard_names/promote.py:resolve_merged_pr` | **Direct fit with identity gap:** verifies `MERGED` and obtains number, URL, merge SHA, head/base refs, but does not obtain the PR author/reviewer login. |
| 3 | Review-delta reader | `imas_codex/standard_names/promote.py:read_pr_changes` | **Direct fit:** extracts docs edits and best-effort name renames from `standard_names/**/*.yml`, matched to graph identity by `StandardName.id`. |
| 4 | Fold-back state machine | `imas_codex/standard_names/promote.py:run_approval` | **Direct fit:** single existing authority that separates edited, untouched, blocked, unmatched, accepted, staged-for-review, and contested outcomes. |
| 5 | Human edit attachment | `imas_codex/standard_names/edit.py:apply_edit` and `_stamp_edit_fields` | **Direct partial fit:** reuses normal rename/docs machinery and stamps `edit_origin="human"`, `edit_reason`, `edit_requested_at`, and `edit_refine=false`; it identifies the authority class, not the human actor. |
| 6 | Ordinary review scorer | `imas_codex/standard_names/promote.py:_score_proposal` | **Direct fit:** reuses configured name/docs reviewer chains over the exact merged proposal, with no separate approval-only rubric. |
| 7 | Passing and refused routing | `imas_codex/standard_names/promote.py:_apply_passing_review` and `_contest` | **Direct fit:** accepted names use the normal accept/cascade path; failed edits become `name_stage='contested'` with score, reason, and time instead of silent rewrite. |
| 8 | PR provenance writer | `imas_codex/standard_names/promote.py:mark_catalog_name_approved` | **Direct fit for PR traceability:** writes PR number, URL, merge SHA, approval timestamp, and a `StandardNameChange`; **not a fit for reviewer identity** because no actor is supplied or stored. |
| 9 | Frozen batch provenance | `imas_codex/standard_names/catalog_release.py:backfill_review_artifact` | **Direct supporting fit:** backfills PR number, URL, and merge commit into the frozen review artifact after a successful non-dry approval. |
| 10 | Additive-baseline guard | `imas_codex/standard_names/promote.py:_prepare_additive_catalog_delta` | **Direct fit:** requires clean catalog `main` and proves every previously approved entry remains byte-identical before fold-back writes. |
| 11 | Catalog materialization correction | `imas_codex/standard_names/promote.py:_commit_catalog_correction` | **Direct fit:** removes additions that did not earn approval, commits the correction with PR trailers, and pushes the PR target remote. |
| 12 | Durable fold-back receipt | `imas_codex/standard_names/promote.py:tag_fold_back`, `build_contract_block`, and `has_contract_tag` | **Direct fit:** tags the merge commit with `graph-merged:` plus PR/batch/outcome counts and provides the idempotency guard against a second fold-back. |
| 13 | Approval unwind | `imas_codex/standard_names/promote.py:undo_approval` | **Partial fit:** demotes names approved by the named PR, clears PR provenance, and resets batch-contested names, but intentionally leaves accepted human edit history in the graph. |
| 14 | Catalog and receipt unwind | `imas_codex/standard_names/promote.py:delete_fold_back_tag` and `_undo_catalog_correction` | **Direct fit for catalog/tag state:** reverses the catalog correction as a new commit and deletes or restores the exact prior tag; does not roll back accepted graph edit history. |
| 15 | Full graph snapshot/restore | `imas_codex/cli/graph/data.py:graph_export` and `graph_load` | **Complete but operationally heavy fit:** can restore byte-level graph state, but `graph_load --force` is destructive and should be used only on an isolated rehearsal profile unless separately authorized. |
| 16 | Local end-to-end fixtures | `tests/standard_names/test_sn_approve.py`, `test_contested_lifecycle.py`, `test_sn_approve_tag.py`, and `test_cli_approve.py` | **Direct fit for pre-live validation:** covers delta detection, accepted/contested routing, PR scoping, catalog correction, receipt idempotency, and undo without GitHub or production graph effects. |
| 17 | Reviewer-facing refusal surface | catalog `.github/workflows/validate.yml` job `review-edit-guard` plus `REVIEWING.md` | **Supporting fit only:** prevents machine-owned edits and explains contested routing, but it does not execute graph fold-back after merge. |

**Candidate census:** 17 mechanisms inspected; 17 carry an explicit fitness verdict; **0 candidates lacking a verdict**.

## Exact fold-back path

```text
imas-codex sn approve --pr <merged-pr-url>
  -> cli.sn.sn_approve
  -> promote.resolve_merged_pr
  -> sources_manifest.resolve_batch_token / load_names_file
  -> promote.run_approval
       -> promote.read_pr_changes
       -> edit.apply_edit(origin="human", refine=False)
       -> promote._score_proposal
       -> promote._apply_passing_review OR promote._contest
       -> promote.mark_catalog_name_approved
       -> promote._commit_catalog_correction
  -> catalog_release.backfill_review_artifact
  -> promote.tag_fold_back
```

The reviewer-visible refusal is the `ApprovalReport.contested` result printed by `cli.sn.sn_approve`; the durable graph signal is `StandardName.name_stage='contested'` with `contested_reason`, `contested_at`, `edit_status='rejected'`, and the relevant reviewer score. The CLI points the operator to `sn edit`, `sn resolve --override`, or `sn revert`; `list_contested` provides the graph-backed census.

## Traceability fields and writers

### Pull-request traceability that already exists

The following `StandardName` fields are declared in `imas_codex/schemas/standard_name.yaml` and are written by `imas_codex/standard_names/promote.py:mark_catalog_name_approved`:

- `catalog_pr_number`
- `catalog_pr_url`
- `catalog_merge_commit_sha`
- `catalog_approved_at`

The same write creates a linked `StandardNameChange` through `HAS_INTERNAL_CHANGE`, with:

- `operation = 'content_edit'` for an accepted edited proposal, or `operation = 'unchanged_ratification'` for an untouched approved batch entry;
- `reason` naming the catalog PR number and editorial outcome;
- `origin = 'catalog_promotion'`;
- `changed_at`.

The frozen review artifact separately receives `pr_number`, `pr_url`, and `merge_commit` through `catalog_release.py:backfill_review_artifact`. The annotated tag created by `promote.py:tag_fold_back` adds a durable repository receipt whose contract block contains PR reference, batch artifact, and outcome counts.

### Reviewer traceability that exists only at authority-class level

`edit.py:apply_edit` is called from `run_approval` with `origin="human"`. `_stamp_edit_fields` persists:

- `edit_origin = 'human'`
- `edit_reason`
- `edit_requested_at`
- `edit_mode`, `edit_scope`, `edit_status`, `run_id`
- `edit_refine = false` on the touched name or rename successor

This proves that a human catalog edit initiated the proposal, but not **which human** made it. Although `promote.py:fetch_pr_evidence` extracts `author.login` for comments and reviews, that data is used only to synthesize optional fold-back tag notes. No current `StandardName` or approval `StandardNameChange` field receives it. Therefore the plan's reviewer-identity receipt cannot be satisfied honestly by the current graph schema/write path; it requires a scoped repair (for example, an explicit catalog-review actor field/event populated from authoritative GitHub review/commit evidence) before the live exercise can close deliverable five.

## Unwind assessment

An unwind path exists, but it has two levels:

1. **Normal approval unwind:** `imas-codex sn approve --undo --pr <merged-pr-url>` invokes `cli.sn.sn_approve` -> `promote.undo_approval`, then `promote.delete_fold_back_tag`. It demotes approvals owned by the PR, clears PR fields, returns batch-contested rows to accepted, reverses any catalog materialization correction, and deletes/restores the receipt tag.
2. **Full no-lasting-change unwind:** normal undo deliberately does **not** remove accepted human rename/docs history (`REFINED_FROM`, `DocsRevision`, and accepted wording remain). Full restoration requires `graph_export` before the exercise and `graph_load` afterward, preferably against a separately named rehearsal graph/profile. Restoring a production archive is destructive and outside an investigate node's authority.

Consequently, `sn approve --undo` alone can prove promotion/provenance/catalog/tag reversal but cannot prove the plan's stronger “no lasting graph change” assertion after an accepted edited entry. The safest executable design is to load a current archive into an isolated rehearsal graph, point `IMAS_CODEX_GRAPH` at it, run the approval and undo there, compare pre/post graph censuses, and retain the isolated archive/report as evidence. The catalog side should likewise use a disposable worktree or temporary bare-remote fixture when the requirement is zero persistent remote/catalog change.

## Recommended exercise composition

1. Freeze exact inputs: merged fork PR URL, merge SHA, head ref, frozen batch artifact digest, catalog `main` SHA, graph profile, and pre-exercise archive identity.
2. Run `sn approve --pr <url> --dry-run` first. Require one accepted candidate and one planned/refused candidate to be ID-matched, with zero blocked and zero unmatched.
3. On an isolated graph/catalog rehearsal substrate, run `sn approve --pr <url> --no-notes` so the deterministic receipt is independent of an optional LLM summary.
4. Capture `ApprovalReport` counts and query the accepted identity for all four PR fields plus its `HAS_INTERNAL_CHANGE` receipt; query the failed identity for `name_stage`, `contested_reason`, `contested_at`, and reviewer score.
5. Check the annotated version tag begins with `graph-merged:` and contains the exact PR, frozen artifact, and outcome counts; re-run once only to demonstrate the idempotency refusal.
6. Run `sn approve --undo --pr <url>`, verify approvals/provenance/contested rows, catalog correction, and receipt tag return to their pre-approval values.
7. Restore or discard the isolated graph and catalog substrate, then compare pre/post node/edge/property hashes. This final step, not normal undo alone, proves no lasting state.
8. Before promoting the evidence as plan closure, add and verify durable reviewer-actor provenance; PR metadata plus `edit_origin='human'` does not identify who changed the entry.

## Currency findings

- The live plan says the guard job lives in catalog `.github/workflows/catalog.yml` at commit `dd70ba6`. At the inspected catalog `main` (`1b596f3`), the `review-edit-guard` job is in `.github/workflows/validate.yml`; `catalog.yml` owns the preview/site build. Execution should cite the current path.
- The catalog reviewer guide and guard are useful setup/visibility mechanisms, but neither performs post-merge fold-back.
- The code and user-facing help explicitly document the incomplete normal unwind, so treating `--undo` as full rollback would contradict the implementation.

## Focused verification

The following five existing tests passed against the assigned imas-codex worktree and the shared project environment:

- human docs edit attaches through `sn edit` machinery;
- below-threshold edit becomes contested without refinement;
- catalog docs delta is detected;
- an existing fold-back receipt refuses a second fold-back;
- receipt deletion removes local and remote tags in the bare-repo fixture.

Result: **5 passed, 1 configuration warning, 0 failed in 6.29 s**. Full log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T121144493784-n-crsfoldbackmap/targeted-tests.log`.
