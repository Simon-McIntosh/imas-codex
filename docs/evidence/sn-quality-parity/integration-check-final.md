# Final Standard Names integration check

Integration verdict: **PASS** for tested HEAD
`1b9e8b6c12661c8ee51c3b9984a4c55145d8c022`.

## Machine verdict

- Command executed once: `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`
- Pytest exit code: **0**
- Outcome counts: **6,721 passed, 0 failed, 8 skipped, 0 errors**
- Additional pytest accounting: 299 deselected and 34 warnings
- Failing node ids: **none**
- Error node ids: **none**
- Complete named log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T034839975361-n-finalintegration/pytest-standard-names-final.log`

The final pytest summary line is pasted verbatim from that log:

```text
6721 passed, 8 skipped, 299 deselected, 34 warnings in 188.03s (0:03:08)
```

The command deliberately did not add `-q`: repository `addopts` supplied the
single quiet flag, so pytest retained its final count-summary line.

## Tested revision and post-verdict coverage

The run started and completed at exact HEAD
`1b9e8b6c12661c8ee51c3b9984a4c55145d8c022`. It supersedes the earlier suite
verdict recorded at `341c1ec82ef948652dcdab8e84507939665848c6` and covers
these seven later gate and snapshot merges:

1. `b5acf695e761f676cbbea645e921ef9807073ace` — pinned declared-unit and
   COCOS-transformation authority for all holdout rows.
2. `ea717d1ccce08d02937c153165107c3edb56418d` — made the defining-equation
   check compare relation dimensions with declared-unit authority.
3. `138dd0b9562605577ae4b3584532f5d5ddeced69` — made the sign-convention
   check conditional on the authoritative transformation class.
4. `c6860ea2a706eb3f26d741f16f2bb6bd40fc87ad` — recorded the bounded
   accepted-document refresh and its exact prior-document snapshots.
5. `77d2a225588ad8925e73948a7c6d34ba6b0aeba3` — preserved incomplete
   dimensional bindings as `not_evaluable` while retaining real mismatches as
   failures.
6. `e3657ef5092a993f95bdf44f78860edd1b2eef22` — persisted and tested both
   current-gate documentation arms with the scored prose retained.
7. `1b9e8b6c12661c8ee51c3b9984a4c55145d8c022` — selected a fresh
   documentation-snapshot revision from stored revisions instead of trusting a
   stale counter.

All seven commits are ancestors of the tested HEAD. This verdict is a source
and test-suite integration result; the earlier operational receipts remain the
authority for provider spend, graph mutations, quorum outcomes, and snapshot
cardinality.
