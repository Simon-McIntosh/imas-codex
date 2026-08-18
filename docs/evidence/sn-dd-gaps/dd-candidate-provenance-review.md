# DD candidate provenance authority review

## Verdict

**BLOCK** — 0 P0, 0 P1, **2 P2**, 0 P3.

The candidate commit is behavior-neutral in its present tracked state, and the 21 recorded row mappings agree with the supplied immutable release and upstream-provenance evidence. It is not integration-ready under the requested fail-closed parsing contract, however: duplicate YAML mapping keys are silently accepted before Pydantic validation, and integer authority facts are accepted through Pydantic coercion rather than as exact YAML integer scalars. Both defects are confined to the review-only resource parser; neither currently creates runtime or graph authority.

## Blocking findings

### P2 — duplicate YAML keys are not rejected

Evidence:

- `imas_codex/standard_names/dd_resolutions.py:917-930` parses the review resource with plain `yaml.safe_load(content)` and then validates the already-collapsed Python mapping.
- `imas_codex/standard_names/dd_resolutions.py:515`, `:588`, and `:690` use Pydantic `extra="forbid"`, which correctly rejects unknown fields but cannot detect a duplicate key discarded by the YAML loader.
- The candidate manifest has mappings at every authority-bearing level: the top-level contract, `upstream_changes`, individual upstream-change records, and individual candidate records (`imas_codex/standard_names/config/dd_resolution_candidates.yaml:1-55` and following). A repeated `authority`, change key, `status`, `solution_commits`, `source_row`, `exact_paths`, disposition, count, or value key is therefore accepted with last-key-wins semantics before the schema sees it.

Impact: malformed or adversarial review provenance can overwrite an earlier reviewed-looking field without a validation error. It still cannot affect runtime today, but it violates the explicit strict duplicate-field/resource requirement and makes the review artifact non-fail-closed.

Required repair:

1. Parse the candidate resource with a `SafeLoader` variant whose mapping constructor raises on every duplicate key, recursively, before constructing the mapping. Do not attempt post-load duplicate detection.
2. Add regression cases for duplicate top-level keys, duplicate `upstream_changes` mechanism keys, duplicate fields inside one upstream change, and duplicate fields inside one candidate.
3. Preserve `extra="forbid"` and add an explicit unknown-field regression case, since that part of the contract is implemented but currently only implicit.

### P2 — integer evidence facts are coercive rather than exact

Evidence:

- `DDResolutionCandidate.source_release_match_count` and `narrow_evidence_overlap_count` are ordinary `int` fields at `imas_codex/standard_names/dd_resolutions.py:597-599`.
- `DDResolutionCandidateManifest.schema_version` is an ordinary `int` field at `imas_codex/standard_names/dd_resolutions.py:692`.
- Candidate model configs at `:515`, `:588`, and `:690` do not enable strict scalar validation, and there are no pre-validators requiring actual non-boolean integer scalars for these three fields.

Impact: values such as quoted numeric strings, and depending on Pydantic's input coercion a boolean integer surrogate, can be normalized into authority-relevant counts/version before validation. The current resource uses proper integers, so this is not a current-data error or behavior change; it is a strict-schema gap.

Required repair:

1. Use `StrictInt` or equivalent before-validators for `schema_version`, `source_release_match_count`, and non-null `narrow_evidence_overlap_count`, explicitly rejecting booleans.
2. Add regression cases for quoted numbers, floats, and booleans at all three locations.
3. Retain the existing positive/range checks after exact type validation.

## Positive findings

### Current resource content is exact and conservatively bounded

- The resource contains exactly 21 unique row mappings: 16 `bounded_review_input` records and five `broad_scope_hold` records.
- The exact row set is `U11-U16`, `U19`, `U21`, `U22`, `U25-U29`, `U32`, `O17`, and `O20-O24`; no unsupported legacy row is present.
- A complete Ruby safe-YAML/JSON cross-check read all 62 machine-export rows and found zero mismatches in source pattern, exact DD version, raw value, proposed value, release match count, or exact-path membership.
- `U19` is correctly narrowed from the 14-path release cohort to exactly six upstream-covered paths: base, `coefficients`, and `values` under both `edge_profiles` and `plasma_profiles` (`dd_resolution_candidates.yaml:149-165`). No error-index or empty-unit index field is admitted.
- `O20-O24` retain broad patterns only and enumerate zero candidate paths. Their release/overlap facts are exactly `1188/6`, `36/12`, `9/3`, `9/3`, and `9/3` (`dd_resolution_candidates.yaml:275-329`). The model also refuses exact paths on a `broad_scope_hold` (`dd_resolutions.py:669-683`).
- Bounded exact paths are sorted, unique, IDS-prefixed, and pattern-free through `_validate_exact_path` and `_canonical_exact_paths` (`dd_resolutions.py:183-195`, `:620-630`).
- Candidate row identities are unique and all referenced upstream-change keys are both defined and used (`dd_resolutions.py:721-731`).

### Upstream provenance is accurate but non-approving

- PR 242 records full solution SHAs `fd0c145c...` and `72163823...`, merged state, and merge SHA `cb0d86de...`.
- PR 273 records solution SHA `f34c85d3...`, open state, issue 272, affected-since `3.38.0`, no merge, and no fixed release.
- PR 280 records solution SHA `30a5ddd4...`, open state, issue 277, proposed change version `4.2.0`, no merge, and no fixed release.
- PR 281 records head solution SHA `35c14603...`, merged state, issue 278, merge SHA `d07172e8...`, and no fixed release.
- Every upstream change has `fixed_dd_version: null`. The model requires merged changes to carry a merge SHA and forbids open changes from carrying a merge SHA or fixed release (`dd_resolutions.py:574-582`).
- The resource-level authority is fixed to `review_input_only`, and the complete missing-requirement set is fixed to approval receipt, actor, timestamp, fresh evidence token, governed reason, positive resolution revision, and review decision (`dd_resolutions.py:61-71`, `:708-720`; resource lines 1-10).
- Upstream references supply provenance only. Candidate records expose no active state, resolution id, gap id, observation ids, evidence token, approval receipt/actor/time, governed reason, or resolution revision (`dd_resolutions.py:585-600`). They therefore cannot validate directly as `DDResolutionRecord`.

### Contrary semantics and broad generalizations remain excluded

- `ec_launchers/beam/direction/kphi` is absent; its `m^-1` wavevector-component semantics are not generalized from PR 242.
- All current `position/psi` paths and rows `U17`/`U33` are absent.
- Charge-number rows `O12-O15` and their adjacent PR-280-preserved semantics are absent.
- American-spelling `ionization_potential` rows are absent; PR 280 is bound only to the six exact British-spelling paths plus the two exact `O17` base paths.
- No broad glob is used as an exact candidate path. The five broad O-row patterns are holds with empty path tuples, not candidate expansions.

### No runtime, graph, or import-time authority path exists

- The active authority resource `imas_codex/standard_names/config/dd_resolutions.yaml` is unchanged from parent to candidate; both blobs hash to `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf` and contain zero active records.
- Candidate loading is an explicit function only. `importlib.resources` access occurs inside `load_dd_resolution_candidates_for_review()` (`dd_resolutions.py:937-949`), not at import time.
- Repository search finds the candidate resource/classes/loader only in `dd_resolutions.py`, the resource itself, and its tests. No resolver, pipeline adapter, CLI, graph writer, schema generator, release path, manifest digest consumer, or other application path reads it.
- `resolve_dd_field`, `resolve_dd_context`, and `resolve_dd_rows` continue to load only `_MANIFEST_RESOURCE = "dd_resolutions.yaml"` through `load_dd_resolution_manifest()` (`dd_resolutions.py:44-47`, `:903-914`, `:978-1059`, `:1091-1153`).
- No LinkML schema contains `DDResolutionCandidate` or `DDResolutionCandidateManifest`, and no relationship range targets either type. The candidate commit changes no schema file.
- The candidate digest property is a digest of review input only (`dd_resolutions.py:734-737`); it is not used by the active manifest, resolver receipts, pipeline projection, or graph schema.

## Negative findings and limitations

- No P0 or P1 issue was found. The two P2 findings affect fail-closed review-resource validation, not present runtime behavior.
- The current YAML itself has zero duplicate keys by an independent Psych AST walk; the defect is that the production parser would not reject a future duplicate.
- The current YAML uses proper integer scalars and has no current scalar-coercion mismatch; the defect is schema permissiveness.
- Unknown fields are rejected by Pydantic `extra="forbid"`; the missing test does not negate that implementation, but an explicit regression is required alongside the parser repair.
- Exact official provenance is not approval. Open PRs remain open/unreleased; merged PRs post-date DD 4.1.1 and have no fixed published tag. No candidate is active or eligible from these references alone.
- No tests, model generation, provider/network access, graph query/mutation, pipeline, service, catalog, or application execution was performed in this review. The separately assigned validator owns executable tests.

## Git and artifact audit

- Reviewed commit: `9b37dd6f4d6cd57c1b943740be044a225596629b`
- Exact parent: `09b0f6cb75e4eb3e0181a762cb3846addd9e60b2`
- Commit subject/body: `feat(standard-names): add review-only DD provenance candidates`
- Commit parent count: one; this is not a merge commit.
- Commit author: `Simon McIntosh <simon.mcintosh@iter.org>`
- Co-author trailer audit: clean; no `Co-Authored-By:` trailer.
- Exact changed paths: four only — candidate YAML added; `dd_resolutions.py`, `test_dd_resolution_schema.py`, and `test_dd_resolutions.py` modified.
- `git diff --check 09b0f6c..9b37dd6`: clean.
- Detached-worktree status: clean, `## HEAD (no branch)`.
- Candidate resource SHA-256: `c6ee52aedd65cad1fa42c539661a127fffaa6bb2d25e87808f5fda9db35cd4b1`.
- Active authority resource SHA-256: `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf`, identical in parent and candidate.
- Upstream provenance input SHA-256: `bee4c1a8977eaedc279754af0dacb623a844edbcfc935e0ad457cf2c53818496`.
- Evidence Markdown input SHA-256: `51340f7777aeb112f18f3959b6469d3db6372e7d1a896be5e0b39fddb5483945`.
- Evidence JSON input SHA-256: `93ad2ae36bbb9e322591bbf6f71539b3c170d09672059ca7078f74ed9129512e`.
- This review manifest's final whole-file SHA-256 is reported in the supervisor handoff after the write; embedding it here would change the file.

`git show --stat 9b37dd6f4d6cd57c1b943740be044a225596629b`:

```text
commit 9b37dd6f4d6cd57c1b943740be044a225596629b
Author: Simon McIntosh <simon.mcintosh@iter.org>
Date:   2026-08-10 18:56:40 +0200

    feat(standard-names): add review-only DD provenance candidates

 .../config/dd_resolution_candidates.yaml           | 329 +++++++++++++++++++++
 imas_codex/standard_names/dd_resolutions.py        | 302 ++++++++++++++++++-
 tests/graph/test_dd_resolution_schema.py           |  13 +
 tests/standard_names/test_dd_resolutions.py        | 204 +++++++++++++
 4 files changed, 847 insertions(+), 1 deletion(-)
```

## Integration condition

Do not integrate this exact commit as final. Repair the two parser strictness defects, add the focused negative regressions, and re-run independent review/test on the new exact commit. No row content, upstream provenance, runtime resolver, active manifest, graph schema, or pipeline consumer needs to change for this repair.
