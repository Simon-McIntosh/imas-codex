# DD candidate provenance executable validation

## Verdict

**BLOCK / FAIL-CLOSED.** Commit `9b37dd6f4d6cd57c1b943740be044a225596629b` is not cleared for integration.

Independent review confirmed a P2 parser-boundary defect at `imas_codex/standard_names/dd_resolutions.py:918-930`: the candidate loader uses plain `yaml.safe_load`, which silently accepts duplicate YAML mapping keys before strict Pydantic validation. Duplicate top-level keys, upstream-change keys, or candidate-record keys can therefore be interpreted with last-value-wins semantics. Scalar-coercion strictness remains under review as part of the same boundary. This defeats the intended fail-closed provenance contract even though the post-load Pydantic models are strict.

Login-node environment preparation passed. Executable compute validation was already launched when the coordinator's stop instruction arrived, but its logs were written to compute-node-local `/tmp` and were not recoverable from the login node. No build-model or pytest result is claimed. The executable test outcome is **UNKNOWN**, and the confirmed review defect independently blocks integration.

## Exact revision and preflight

- Worktree: `/home/ITER/mcintos/Code/.reckon-worktrees/reckon-worktrees/imas-codex-c994bf55fb01/20260810-dd-candidate-final/validation`
- Detached HEAD: `9b37dd6f4d6cd57c1b943740be044a225596629b`
- Parent: `09b0f6cb75e4eb3e0181a762cb3846addd9e60b2`
- Commit message: `feat(standard-names): add review-only DD provenance candidates`
- Preflight tracked worktree and index: clean; `git diff --check` passed.
- Postflight tracked worktree and index: clean; detached HEAD unchanged.
- No tracked file was edited, staged, committed, pulled, merged, or pushed by this validator.
- Ignored `.venv` and generated model/schema outputs were permitted and remain ignored.

Exact commit scope:

```text
9b37dd6f (HEAD) feat(standard-names): add review-only DD provenance candidates
 .../config/dd_resolution_candidates.yaml           | 329 +++++++++++++++++++++
 imas_codex/standard_names/dd_resolutions.py        | 302 ++++++++++++++++++-
 tests/graph/test_dd_resolution_schema.py           |  13 +
 tests/standard_names/test_dd_resolutions.py        | 204 +++++++++++++
 4 files changed, 847 insertions(+), 1 deletion(-)
```

## Execution record

| Operation | Count | Exit/result | Evidence |
|---|---:|---|---|
| `uv sync --extra test` on the login node | exactly 1 | **PASS, exit 0** | Full log retained and hashed below. |
| SLURM allocation | 1 | allocated as job `1243587` on `98dci4-clu-3141`, `sun_debug`, 8 CPUs, 8 GiB, 01:00:00 | Allocation output observed before the stop instruction. |
| `command uv run --no-sync build-models --force` | launched once | **UNKNOWN / not claimed** | Output log was in compute-node-local `/tmp` and was not recoverable in the final login-node postflight. |
| Focused pytest invocation | launched once | **UNKNOWN / not claimed** | Output log was in compute-node-local `/tmp` and was not recoverable in the final login-node postflight. No rerun occurred. |
| Exact-path Ruff check and format-check | 0 | **NOT RUN** | Stopped after independent review confirmed the integration blocker. |
| Provider/graph-free static probe | 0 | **NOT RUN by this validator** | Stopped after the blocker; implementation-manifest assertions are not promoted into independent test evidence. |

The focused pytest command that had already been launched was:

```text
command uv run --no-sync pytest tests/graph/test_dd_resolution_schema.py tests/standard_names/test_dd_resolutions.py -q
```

Because its full log and sentinel were inaccessible at final postflight, there is no defensible test count, failure list, or exit code. The result is deliberately recorded as unknown rather than inferred from terminal timing.

## Retained evidence and hashes

| Artifact | SHA-256 |
|---|---|
| `/tmp/reckon-s8-evidence/dd-candidate-provenance-test-preflight.log` | `a9638c21cce9020e598c1a44b7d3ae13b7aee8fbafe7e7eb9975b18331b398ef` |
| `/tmp/reckon-s8-evidence/dd-candidate-provenance-test-uv-sync.log` | `3023666af998abee7fafd01382348beb13fd99705f99d1c83d76482cb7db4de5` |
| `/tmp/reckon-s8-evidence/dd-candidate-provenance-test-postflight.log` | `39be98a91d99be12afc2a4c1c1d3dfa3a82955158cadc3002b18fffce8bfb275` |

The final postflight explicitly records that the compute-local environment, model-generation, and pytest logs were not present on the login node. They are not listed as retained evidence and no hash or outcome is invented for them.

Implementation-manifest reference hashes, not independently re-derived after the stop instruction:

- Active authority manifest: `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf`
- Candidate resource: `c6ee52aedd65cad1fa42c539661a127fffaa6bb2d25e87808f5fda9db35cd4b1`

## Required repair and next action

Replace the permissive YAML parse boundary with a duplicate-key-rejecting loader and pin strict scalar behavior. Add regression cases covering duplicate keys at the manifest top level, inside upstream-change records, and inside candidate records, plus ambiguous/coercible scalar forms. Then produce a new exact commit and rerun the complete one-shot SLURM protocol from a fresh clean detached worktree.

Until that repair lands and independent executable validation is green, the candidate provenance resource remains review input only and grants no activation, graph mutation, provider call, pipeline run, catalog action, release action, or DD-runtime application authority.
