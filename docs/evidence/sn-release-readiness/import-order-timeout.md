# Fresh-process import-order timeout investigation

## Verdict

**Cleared as not a regression from today's merges.** The exact fresh-process
import-order test passes at both the requested pre-change merge-base
`711038bfe00fe7a68adea123143f25782ba26e03` and current worktree HEAD
`c938435fc266ccc51b0526bacdfad101c1e200eb`. The earlier timeout from the prior
full-suite capture did not reproduce either in an isolated rerun or under a new
full Standard Names suite load.

No source or test change is warranted. In particular, increasing the 30 s
subprocess timeout would hide a transient machine or filesystem stall when the
measured import normally occupies less than 23% of that allowance.

## Revision comparison

The failing parameter case is:

```python
import imas_codex.cli.sn
import imas_codex.standard_names.turn as t
print(t.TURN_PHASES)
```

| Revision / context | Parameter call | Standalone process wall | Result vs 30 s timeout |
|---|---:|---:|---|
| merge-base `711038bf` | 6.27 s | 6.50 s | PASS, 21.7% of timeout |
| current `c938435f`, isolated | 6.30 s | 6.79 s | PASS, 22.6% of timeout |
| current `c938435f`, full-suite load | 6.80 s | not separately wrapped | PASS, 22.7% of timeout |

The isolated current-minus-base process-wall delta is +0.29 s (+4.5%), while
the pytest parameter-call delta is +0.03 s (+0.5%). Those differences are
ordinary run-to-run variation and are not remotely close to the 30 s boundary.
The lightweight `turn`-first parameter also remained effectively flat: 0.24 s
at the merge-base and 0.28 s at current HEAD.

Because both revisions pass and their timings are effectively unchanged,
**there is no changed expensive import to name** in `loop.py`, `services.py`, or
`promote.py`. The current `loop.py` scoped-embedding additions use function-local
imports inside `_drain_scoped_standard_name_embeddings`; they do not execute on
`imas_codex.cli.sn` module import. The `promote.py` approval-provenance additions
also do not measurably change this import path.

## Required full-suite gate

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync pytest tests/standard_names/ -q
```

Result: **exit 0; 6,818 passed, 8 skipped, 0 failed**. The progress capture
contains 6,826 terminal test markers (`6,818 '.'`, `8 's'`, no `F` or `E`). The
import-order parameter appears in the slowest-duration table at 6.80 s.

## Evidence files

- `logs/base-test.log` — merge-base exact test, 2 passed; target parameter 6.27 s.
- `logs/base-import-time.log` — merge-base standalone process wall 6.50 s.
- `logs/current-test.log` — current exact test, 2 passed; target parameter 6.30 s.
- `logs/current-import-time.log` — current standalone process wall 6.79 s.
- `logs/full-standard-names.log` — current full suite, exit 0; 6,818 passed,
  8 skipped, 0 failed; target parameter 6.80 s.

All paths above are relative to
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T140354372814-n-importordertimeout/`.
