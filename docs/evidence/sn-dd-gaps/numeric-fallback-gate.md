# Numeric missing-unit fallback verification

## Verdict

The complete Standard Names suite passed at commit
`8282e3d1214e775020fbb2541bc4f4411130cfa0`, which contains the runtime
retirement commit `54aad0e711c2d09e7db8c8c88328edcbed2d4a59` and its aligned audit
expectations. The one permitted execution produced 6,575 passed, 8 skipped,
0 failed, and 0 errors, with exit code 0.

This gate proves the tested chain is compatible with the complete Standard
Names suite. It does not replace the policy evidence: the implementation itself
removes the blanket numeric fallback and marks its carrier audited only because
runtime paths no longer fabricate unit `1` from data type alone.

## Exact execution

Working directory:

`/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/cb77b796-743a-4df6-bd6f-aced19e8006c/ddres-numeric-fallback`

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/cb77b796-743a-4df6-bd6f-aced19e8006c/ddres-numeric-fallback uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q
```

The command was executed once. Its complete output and terminal exit marker are
in
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T203307392280-verify-numeric-fallback/pytest.log`.
The log is 435 lines and 36,110 bytes; SHA-256 is
`11a66008285047c22af1d5982910f08c35cdf546295a90d63fb5cc61a1f88e1b`.

## Read-only treatment

`git status --porcelain` in the tested worktree emitted no output before the
run and no output after the run. Its HEAD remained
`8282e3d1214e775020fbb2541bc4f4411130cfa0`. The existing canonical project
environment was reused with `--no-sync`; no environment or pytest cache was
created in the tested worktree.

## Representative authority boundaries

- `equilibrium/time_slice/profiles_1d/q`, with DD version `4.1.1`, no raw unit,
  and no unit relationship, remains unresolved instead of receiving fabricated
  dimensionless unit `1`.
- `mhd/time_slice/toroidal_mode/n`, an integer-valued path without unit
  authority, likewise remains unresolved.
- `equilibrium/time_slice/profiles_1d/pressure`, when linked to `Pa`, retains
  that declared unit as the authority.

No Standard Name identity, description, or review score changes in this test
chain; the affected bindings are DD source-path unit-authority inputs. The audit
continues to expose shipped-carrier, conflicting-authority, and duplicate-active
regression boundaries while reporting the retired numeric fallback as audited
with zero residuals.
