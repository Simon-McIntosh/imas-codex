# Merged Standard Names suite verification

Date: 2026-08-20

Verdict: **PASS**. The full Standard Names suite completed at the exact merged
`main` commit with zero failures and zero collection or execution errors.

## Checkout identity

- Merged base commit: `60d9258f9d0b98d50d6efdec0d46e7b50be671dc`
- Detached verification checkout `HEAD`: `60d9258f9d0b98d50d6efdec0d46e7b50be671dc`
- `main` and the canonical checkout `HEAD` at preflight:
  `60d9258f9d0b98d50d6efdec0d46e7b50be671dc`
- Session comparison base: `c7693ecc` (verified present as a commit)

The verification checkout was clean before the run. `git status --porcelain`
also emitted no output immediately after pytest completed.

## Invocation and durable log

Executed once from the verification worktree:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q
```

- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T135146745410-sgwi-merged-suite-verification/pytest-standard-names.log`
- Log SHA-256: `8dd18dd0e3300392c0a9335cb6dc40cff3346c77d7041e70f0609d7ae82bf80b`
- Log size: 36,285 bytes, 440 lines
- Exit code: `0`

## Exact result counts

| Outcome | Count |
|---|---:|
| Passed | 6,528 |
| Failed | 0 |
| Skipped | 8 |
| Errors | 0 |
| Total collected outcomes | 6,536 |

The repository configuration already supplies quiet pytest output and the
explicit invocation adds `-q`, so pytest does not emit its usual prose summary
line. The exact counts above come from the complete captured outcome stream:
6,528 `.` markers, 8 `s` markers, no `F` or `E` markers, followed by
`EXIT_CODE=0`. The warning and slowest-duration sections are retained in full
in the log.

## Failure and session-base classification

There are no failing test IDs and no errors at the merged base. Consequently,
there is no failure requiring reproduction at session base `c7693ecc`; the
pre-existing-versus-regression classification set is empty. No second suite or
focused base invocation was run.

