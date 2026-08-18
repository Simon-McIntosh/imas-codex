# DD candidate provenance implementation manifest

## Outcome

Implemented a separately typed, review-only DD resolution candidate resource. It is behavior-neutral and fail-closed: the active authority manifest remains unchanged with zero records, candidate records have no activation/approval/evidence fields, runtime resolvers never load the candidate resource, and LinkML declares no candidate graph type or relationship.

Commit: `9b37dd6f4d6cd57c1b943740be044a225596629b`

## Scoped files

- `imas_codex/standard_names/config/dd_resolution_candidates.yaml` (new)
- `imas_codex/standard_names/dd_resolutions.py`
- `tests/graph/test_dd_resolution_schema.py`
- `tests/standard_names/test_dd_resolutions.py`

No other repository path changed. `imas_codex/schemas/standard_name.yaml` and `imas_codex/standard_names/config/dd_resolutions.yaml` were deliberately left unchanged. The detached worktree is clean.

## Implemented contract

- Added a strict Pydantic candidate manifest and explicit `load_dd_resolution_candidates_for_review()` loader. The active loader retains its separate fixed resource name and unchanged schema.
- Candidate authority is fixed to `review_input_only`; candidate dispositions are limited to bounded review input and broad-scope hold.
- Every candidate records the complete missing-requirement set: approval receipt, approval actor, approval timestamp, fresh evidence token, governed decision reason, positive resolution revision, and review decision. Candidate records have none of those fields and also have no active state, resolution identity, DDGap identity, or graph evidence token.
- Stored exactly 21 upstream-supported row mappings: 16 bounded review inputs and five broad holds.
- Split `U19` from its 14-path release cohort to the six exact upstream-covered base/values/coefficients paths across edge and plasma profiles.
- Kept `O20`-`O24` as pattern-level holds with zero candidate paths. Their immutable release counts and narrow evidence overlaps are retained as `1188/6`, `36/12`, `9/3`, `9/3`, and `9/3`.
- Preserved official upstream provenance for PRs 242, 273, 280, and 281, including exact solution commits, open/merged state, merge commits where present, the proposed 4.2.0 version only for PR 280, and `fixed_dd_version: null` for every change.
- Preserved contrary semantics by excluding EC-launcher `kphi`, current `position/psi` paths, and adjacent charge-number rows.

## Safety proofs encoded in tests

- Candidate models cannot validate as `DDResolutionRecord` and expose no approval, reason, revision, state, identity, or evidence-token fields.
- Removing any missing activation requirement invalidates the candidate manifest.
- A broad hold cannot silently acquire exact paths.
- Candidate loading leaves the active manifest object, digest, and empty resolution tuple unchanged.
- `resolve_dd_field`, `resolve_dd_context`, `resolve_dd_rows`, and pipeline projection are exercised while candidate loading is replaced with a function that raises on access; all pass raw `m` through unchanged under the empty active manifest.
- The LinkML schema is asserted to contain no candidate class or candidate relationship range, so schema-driven graph writers have no candidate target.
- Exact row set, upstream SHAs/state, `U19` split, broad holds, fixed-release absence, and contrary semantic exclusions are pinned.

## Static validation

- `uv run --no-sync ruff check --fix` on the three Python paths: pass.
- `uv run --no-sync ruff format` on the three Python paths: pass.
- `git diff --check`: pass.
- Ruby safe-YAML parse: 21 candidates, zero active resolutions, all fixed versions absent.
- Ruby authority audit: 16 bounded rows, five broad holds, seven missing requirements, no contrary paths, no fixed-release claims.
- Ruby JSON/YAML cross-check: every candidate preserves the immutable export row's pattern, raw value, proposed value, release count, and only release-enumerated exact paths.
- Active manifest unchanged: SHA-256 `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf`.
- Candidate resource SHA-256: `c6ee52aedd65cad1fa42c539661a127fffaa6bb2d25e87808f5fda9db35cd4b1`.
- Repository search finds candidate-resource access only in the explicit review loader and its tests; no runtime consumer or graph writer references it.
- Mandatory path/label/prose scans: paths, changelog prose, and label words clean. Bare-label hits are exclusively the required legacy `Uxx`/`Oxx` row identities in review data and assertions.
- Commit trailer check: clean; no co-author trailer.

No pytest or model generation was run, as required. The detached worktree did not have project dependencies provisioned, so `uv run --no-sync` could not import PyYAML for an in-process Pydantic load; static YAML validation used Ruby's safe parser. The separate SLURM validation worker owns executable tests.

## Git scope audit

```text
9b37dd6f (HEAD) feat(standard-names): add review-only DD provenance candidates
 .../config/dd_resolution_candidates.yaml           | 329 +++++++++++++++++++++
 imas_codex/standard_names/dd_resolutions.py        | 302 ++++++++++++++++++-
 tests/graph/test_dd_resolution_schema.py           |  13 +
 tests/standard_names/test_dd_resolutions.py        | 204 +++++++++++++
 4 files changed, 847 insertions(+), 1 deletion(-)
```

`git status --short --branch`:

```text
## HEAD (no branch)
```

## Remaining validation and blockers

- Remaining: the assigned SLURM validator must run the focused schema/resolver tests against a provisioned environment.
- No implementation or authority blocker remains for integration. This commit grants no activation, graph mutation, provider call, pipeline run, catalog action, release action, or DD-runtime application authority.
