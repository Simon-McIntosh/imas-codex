Integration verdict: **COMPLETE for tested HEAD `341c1ec82ef948652dcdab8e84507939665848c6`; this is not a current-head verdict.**

## Machine verdict

- Command executed once: `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q`
- Primary verdict: pytest exit code `0`
- Counts derived from the captured pytest progress markers: **6,616 passed, 0 failed, 8 skipped, 0 errors, 0 xfailed, 0 xpassed; 6,624 total outcomes**
- Failing node ids: none
- Named log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T163017219461-n-integrationcheck/pytest-standard-names.log`

Exit code `0` is the machine verdict that no test failed. The counts above are explicitly derived from progress markers; they are not presented as a pytest summary line.

## Reporting-instrument note

The repository's pytest `addopts` already supplies `-q`. The dispatch also required a trailing `-q`, making this run doubly quiet and suppressing pytest's final count-summary line. The absence of that rendered line was a reporting effect, not a test failure or truncated capture. No rerun was performed: rerunning a suite only to change output formatting is prohibited and would add no test evidence.

For a future run that genuinely needs the rendered summary line, omit the extra command-line `-q` and let repository `addopts` supply it, or pass `-rN` as the reporting override.

## Coverage boundary

This verdict covers exactly HEAD `341c1ec82ef948652dcdab8e84507939665848c6`. It does **not** cover the four later merges `ea717d1c`, `123028ba`, `b5acf695`, and `1b67443c`. In particular, `ea717d1c` added the dimensional check to `docs_gates.py`, and `b5acf695` pinned unit and COCOS authority into the holdout; both change Standard Names surfaces covered by this suite. A separate integration node will test the then-current head after the remaining gate work lands.

The orchestrator deliberately resolved the original verbatim-summary done-when on 2026-08-23 using exit code `0` plus the derived progress counts: an exact reporting criterion can be unmeetable while the underlying gate is unambiguously green, so the correct response is to report the cause and let the orchestrator resolve it rather than silently relaxing or manufacturing evidence.
