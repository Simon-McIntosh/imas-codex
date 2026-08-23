NEEDS-HELP: the exact full-suite command passed, but double-quiet pytest output omitted the required verbatim final summary line.

tried: Ran `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q` once against exact HEAD `341c1ec82ef948652dcdab8e84507939665848c6`, capturing the complete output and shell exit code in `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T163017219461-n-integrationcheck/pytest-standard-names.log`. The run exited `0`. Parsing only pytest's captured progress markers gives 6,616 passed, 8 skipped, 0 failed, 0 errors, 0 xfailed, and 0 xpassed across 6,624 outcomes. There are no failing node ids. The log contains no pytest-emitted count summary: repository `addopts` already supplies `-q`, so the command's required `-q` makes the run doubly quiet.

options: (1) Accept the captured exit code and progress-marker counts as the integration verdict despite the absent verbatim summary line. (2) Authorize one rerun with the repository `addopts` neutralized while retaining the named command text, so pytest emits a count summary. (3) Amend the evidence requirement to quote the final progress line plus the recorded exit code instead of a nonexistent summary line.

leaning: Option 1, because the full named check already ran once, all 6,624 outcomes are represented in the captured progress stream, and rerunning only to change output formatting conflicts with the test-execution protocol.

cost-if-wrong: If reconstructed progress counts are not acceptable evidence, the integration node must rerun the full Standard Names suite once under an explicitly approved reporting override and replace this blocked record with the emitted summary.

Integration verdict: **BLOCKED on evidence format; execution is green.**

- Exact HEAD: `341c1ec82ef948652dcdab8e84507939665848c6`
- Command: `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q`
- Named log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T163017219461-n-integrationcheck/pytest-standard-names.log`
- Derived counts: 6,616 passed; 0 failed; 8 skipped; 0 errors
- Exit code: `0`
- Failing node ids: none
- Required verbatim final summary line: **not emitted by pytest; unavailable to paste without fabrication**

The log's final recorded line is `EXIT_CODE=0`. The last pytest-emitted line is a slow-duration entry, not a count summary, so neither can honestly be presented as the required verbatim summary.
