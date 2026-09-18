# Fold-back harness write transport

Plan: imas-codex:catalog-review-surface §4

## Orientation

- Worktree: `/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/ship-s14-20260918/n-the-fold-back-harness-mocks-the-write-transport`
- Base revision: `f1ff208d8fa9aba1ede0a5eab9ec731e61787dba`

## Reproduction — the defect still reproduces at base

Command (base revision, HEAD copy of the test file):

```
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider --no-header -q \
  tests/standard_names/test_sn_approve_tag.py
```

Result: **exit 1, 2 failed / 31 passed** (log `/tmp/foldback-before-full.log`,
sha256 `09fb76df6e4e8dd51289c2fc973f4d5ba0e829baee1a278c665b9080e969b040`), the two
failures being exactly

- `TestApprovedCatalogMaterialization::test_contested_entry_is_removed_from_main_by_a_correction_commit`
- `TestApprovedCatalogMaterialization::test_undo_restores_catalog_main_and_graph_lifecycle`

both with `RealTransportAttempted`. Verbatim traceback tail:

```
imas_codex/standard_names/promote.py:1414: in run_approval
    notify_refusal_on_pull_request(report, catalog_pr_url=catalog_pr_url)
imas_codex/standard_names/promote.py:1544: in notify_refusal_on_pull_request
    status, response = _pull_request_comment(repo, number, body)
imas_codex/standard_names/promote.py:1475: in _pull_request_comment
    return github_api_call(
imas_codex/graph/ghcr.py:510: in github_api_call
    status, body, _ = _github_request(
imas_codex/graph/ghcr.py:487: in _github_request
    with urllib.request.urlopen(request, timeout=_REST_TIMEOUT_SECONDS) as response:
tests/standard_names/test_sn_approve_tag.py:74: in refuse
    raise RealTransportAttempted(
E   standard_names.test_sn_approve_tag.RealTransportAttempted: test opened a real connection to https://api.github.com/repos/fork/catalog/issues/7/comments
```

A short two-node run reproduces the same two failures in 15 s (log
`/tmp/foldback-before.log`, sha256
`8f2c15ee08ff6e9f832f4207a30bfc8f31d6e16b9a83cb46ac7e3be7715ec349`).

Cause: the fold-back now carries a refusal onto the pull request by POSTing a
comment (`run_approval` → `notify_refusal_on_pull_request` →
`_pull_request_comment` → `github_api_call`). The `_fold_additive_batch` harness
mocked only the graph writers; the write transport was left live, so any batch
carrying a refused edit escaped to the network. The clean-fold case does not
fail because a report with no refusal renders no notice and posts nothing.

## The repair

The mocked-client pattern already present in this file is
`TestFetchPrEvidence._transport` — its routed `fake_call(method, path, *,
payload=None, token=None)` and its
`patch("imas_codex.graph.ghcr.github_api_call", fake_call)` at lines **527**
(definition) and **542** (patch) of the current file. The repair reuses that
pattern for the write direction:

- `_fold_back_write_transport(responses)` — **line 187**, patch at **line 203**,
  same signature and same unrouted-404 fallback as the read transport.
- called from `_fold_additive_batch` at **line 230** with
  `[("/issues/7/comments", 201, {"id": 1})]`, the endpoint
  `_pull_request_comment` posts to.

The autouse refusal (`no_real_transport`, lines 63–78) is **not** touched: no
edit widened, relaxed, or reordered it.

## The guard is proved still armed

`TestRefusalNoticeTransportGuard` (line 244) holds two tests that together read
the receipt rather than the absence of an error:

- `test_an_unrouted_refusal_still_reaches_the_real_transport_guard` — drives
  `notify_refusal_on_pull_request` with **no** write mock and requires
  `RealTransportAttempted`. The same call the repair intercepts still escapes
  without it, so the autouse refusal is live.
- `test_the_write_mock_intercepts_that_same_call` — the same endpoint through
  `_fold_back_write_transport` answers `(201, {"id": 1})` instead of escaping:
  a positive receipt that the mock, not a disarmed guard, is what makes the two
  materialization tests pass.

## After — the named check

```
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider --no-header -q \
  tests/standard_names/test_sn_approve_tag.py
```

Result: **exit 0, 35 passed / 0 failed** in 45.7 s (log
`/tmp/foldback-after.log`, sha256
`b9acacdcce384635b0c950a0eb6014f67e847f1d4be34402475d4f7d9cf07c1f`).

The two named tests pass by name (verbose run,
`/tmp/foldback-after-named.log`):

```
TestApprovedCatalogMaterialization::test_contested_entry_is_removed_from_main_by_a_correction_commit PASSED
TestApprovedCatalogMaterialization::test_undo_restores_catalog_main_and_graph_lifecycle PASSED
```

Delta: base 2 failed / 31 passed → after 0 failed / 35 passed, two tests added
(the guard pair).