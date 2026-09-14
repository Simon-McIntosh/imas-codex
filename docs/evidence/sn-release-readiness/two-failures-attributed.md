# Two named suite failures measured at both revisions

## Scope and method

The measurement compared the exact current `origin/main` revision,
`d253a67d4c93194ca787b939ea62cda36046de00`, with the requested candidate
revision, `ff540084d354b190858e410a697828905d3fa7af`. The current revision is
the parent of the candidate; the candidate changes only the grammar-content
refresh in `imas_codex/cli/sn.py`, `imas_codex/standard_names/grammar_sync.py`,
and `tests/cli/test_grammar_autosync.py`. Neither named test is changed by the
candidate.

Each test was executed exactly once per revision on the `all_debug` SLURM
partition with the shared project environment. The candidate revision was
materialized from its immutable git archive in the shared run directory so the
compute node used the requested source revision. The four complete logs are:

- Baseline embed-preflight:
  `~/.config/reckon/crew/runs/r-20260914T154734912612-n-srr-two-suite-failures-are-attributed-by-measurement/baseline-embed-preflight.log`
- Baseline reset-to-drafted:
  `~/.config/reckon/crew/runs/r-20260914T154734912612-n-srr-two-suite-failures-are-attributed-by-measurement/baseline-reset-drafted.log`
- Candidate embed-preflight:
  `~/.config/reckon/crew/runs/r-20260914T154734912612-n-srr-two-suite-failures-are-attributed-by-measurement/after-embed-preflight.log`
- Candidate reset-to-drafted:
  `~/.config/reckon/crew/runs/r-20260914T154734912612-n-srr-two-suite-failures-are-attributed-by-measurement/after-reset-drafted.log`

The candidate source archive used for the latter two runs is recorded at
`~/.config/reckon/crew/runs/r-20260914T154734912612-n-srr-two-suite-failures-are-attributed-by-measurement/candidate-source-ff540`.

## Results

| Test | Baseline `d253a67d` | Candidate `ff540084` | Observed difference |
|---|---|---|---|
| `tests/cli/test_sn_embed_preflight.py::test_run_sn_cmd_skips_embed_preflight_for_dry_run` | exit 1, 1 failed | exit 1, 1 failed | none |
| `tests/cli/test_sn_generate_cli.py::TestBackwardCompatibility::test_reset_to_drafted_dry_run` | exit 1, 1 failed | exit 1, 1 failed | none |

### `test_run_sn_cmd_skips_embed_preflight_for_dry_run`

The baseline log fails in `imas_codex/standard_names/loop.py:1204`, through
`imas_codex/cli/sn.py:754` and `_query_pool_progress`, before any
content-digest-specific path is relevant. Its exact connection error is:

> `neo4j.exceptions.ServiceUnavailable: Couldn't connect to localhost:17687 (resolved to ('[::1]:17687', '127.0.0.1:17687'))`

The underlying refusal is also recorded as:

> `Failed to establish connection to ResolvedIPv6Address(('::1', 17687, 0, 0)) (reason [Errno 111] Connection refused)`

The candidate log has the same failure at the same logical operation and the
same URI, `localhost:17687`, with exit 1 and one failed test. Both logs also
record the client warning that its attempt to start the SSH tunnel to `iter:7687`
failed because the compute-node environment could not connect to `iter` on SSH
port 22, after which the client returned the fallback tunnel URI
`bolt://localhost:17687`.

### `test_reset_to_drafted_dry_run`

The baseline log fails at `tests/cli/test_sn_generate_cli.py:289` with the
exact assertion:

> `assert result.exit_code == 0, result.output`
>
> `E assert 1 == 0`
>
> `E + where 1 = <Result ServiceUnavailable("Couldn't connect to localhost:17687 (resolved to ('[::1]:17687', '127.0.0.1:17687')):…")>.exit_code`

The candidate log has the same assertion failure at the same test line and the
same wrapped `ServiceUnavailable` for `localhost:17687`. It therefore has exit
1 and one failed test, with no revision-dependent difference.

## Attribution verdict

Both failures are **environment-dependent and reproduce only under the
`all_debug` compute placement** used by this mandated measurement. They are
not pre-existing product failures demonstrated by a healthy graph path, and
they are not introduced by `ff540084`: each failure is present at the parent
revision and has the same operative cause at the candidate revision. The
placement cannot reach the login-node-local tunnel and
falls back to the refused `bolt://localhost:17687`; the separate healthy-suite
run on a functioning graph path is consistent with this attribution.

The localhost URI is itself the environmental finding required by this
measurement. A valid compute-node run must use the project's direct graph-host
resolution rather than this dead login-node tunnel fallback. No source or test
was changed by this measurement node.

Campaign spend was USD 103.60 before and USD 103.60 after: these four test
executions made no LLM calls.
