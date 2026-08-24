# Quiescence baseline verification

The quiescence gate is **open** at the tested commit. The recorded full test
selection is green, the tested commit is reachable from `origin/main`, and none
of the nine registered worktrees holds an uncommitted edit to
`imas_codex/standard_names/graph_ops.py`.

- Tested commit: `f19eeceee4ea9622639311a627f4f2b9dace05c2`
- Command: `uv run --no-sync pytest tests/standard_names/ tests/graph/`
- Exit status: `0`
- Captured log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T155224525263-n-quiescenceverify/quiescence-baseline-pytest.log`
- Recorded at: `2026-08-24T18:00:34+02:00`

Verbatim final pytest summary line:

```text
7214 passed, 8 skipped, 654 deselected, 34 warnings in 307.28s (0:05:07)
```

Failing pytest node IDs: **none (0)**.

## Prior red baseline

The prior run recorded in `baseline-green.md` tested commit
`b794924bd572cb66ff09045b941c42d3ac73efe4` and ended with seven failures:

```text
7 failed, 7207 passed, 8 skipped, 654 deselected, 34 warnings in 213.64s (0:03:33)
```

The integrated test boundary therefore moved from **7 failed / 7,207 passed**
to **0 failed / 7,214 passed**.

## Remaining quiescence conditions

All nine worktrees registered at verification time were checked with
`git status --porcelain -- imas_codex/standard_names/graph_ops.py`; **zero**
held uncommitted edits. The tested commit was an ancestor of the then-current
`origin/main` commit `225acc2653405164cc38932173a1d06701eddef6`.
Intervening commits changed only plan, evidence, and crew-state documents, so
the tested source and test trees were identical to the integrated source and
test trees at that point.
