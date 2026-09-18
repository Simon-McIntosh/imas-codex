# Release-surface reuse map

Tally, stated first: **5 of 5 surfaces located.** Every one names the symbol, quotes
its signature, records the file and line numbers, and states what a caller must
supply. Where a located surface is not reachable from any production route (the
docs-axis reconciler is the one case), the gap is named under the entry rather
than reported as an absence of the machinery.

Anchors are taken from the release-contract tree at `4c9d1dabb` — the revision
this worktree was dispatched on. Line numbers are those the symbols occupied at
that revision.

## 1. The pull-request REST transport — located

`_GitHubClient` — `imas_codex/standard_names/catalog_release.py:57`.

```python
class _GitHubClient:
    """The single pull-request boundary; every call goes over GitHub REST."""
    def __init__(self, token: str | None = None) -> None
```

Four methods, all REST:

| method | line | signature |
|---|---|---|
| `create_pull_request` | 74 | `(*, branch: str, base: str, title: str, body: str, repo: str, head_owner: str) -> tuple[int \| None, str \| None]` |
| `update_pull_request_body` | 103 | `(*, repo: str, number: int, body: str) -> None` |
| `read_pull_request_body` | 114 | `(*, repo: str, number: int) -> str` |
| `closed_pull_request_heads` | 124 | `(*, repo: str, branch: str, head_owner: str \| None = None) -> set[str]` |

The transport under the client is `github_api_call` — `imas_codex/graph/ghcr.py:502`:

```python
def github_api_call(method: str, path: str, *, payload: dict | list | None = None,
                    token: str | None = None) -> tuple[int, Any]
```

The class docstring records why REST replaced the CLI: the `gh` CLI resolves
pull-request metadata through GraphQL, which fails on a repository whose
response still carries Projects-classic fields, and REST carries bodies as JSON
so no length or quoting limit applies.

**Instantiations:** `catalog_release.py:1243` and `:1763`. The module-level
reader `_closed_pr_heads` (`:411`) wraps the client and takes an injectable
`github_client` (its partial at `:477`), which is the seam the REST failure
coverage at `tests/standard_names/test_catalog_release.py:300` injects through.

**A caller must supply:** `repo` as `owner/name`, `head_owner` (the fork), the
branch, base, title and body; a token if the default resolution is not wanted.
The client resolves nothing itself — path geometry is the only thing it knows.

## 2. The exclusion accounting the exporter produces — located

`ExportReport` — `imas_codex/standard_names/export.py:335` — is the accumulator;
`ExclusionRecord` (`:297`) is one excluded identity with its terminal reason
("One accepted-population identity excluded for one terminal reason").

```python
class ExportReport:                                    # export.py:335
    def record_exclusions(self, records: list[ExclusionRecord]) -> None   # :370
    def _exclusion_rows(self) -> list[dict[str, Any]]                     # :390
```

The manifest payload the report emits carries `exclusion_ledger` (one row per
record), `exclusion_records`, `exclusion_by_reason`, `accounted_exclusions` and
`accounting_residue` (`:453`–`:458`). The gate is
`GATE_EXCLUSION_ACCOUNTING = "exclusion_accounting"` (`export.py:181`), checked
at `:1277`. Two link guards make the published body honest about that
accounting: `ExclusionLedgerLinkError` and `ExportReportLinkError`
(`catalog_release.py:45`–`53`).

Entry point: `run_export(staging_dir: str | Path, *, min_score: float = 0.65,
bound_adjacent_half_width=…, include_unreviewed=False, min_description_score=None,
domain: str | None = None, force=False, skip_gate=False, gate_only=False,
gate_scope='all', override_edits=None, cocos_convention=…, include_sources=True,
names_only=False, final=False, review_batch=None, manifest_sources=None,
isnc_dir=None) -> ExportReport` — `export.py:2459`.

**A caller must supply:** the staging directory and the batch selectors. The
exclusion records themselves are produced inside `run_export` by its own gates
(`:2597`, `:2612`, `:2647`, `:2666`, `:2753`, `:3011`); a caller does not build
them. Anything that wants an exclusion tally must read the returned
`ExportReport`, not recompute one.

## 3. The attempt-budget charge and reset routes — located

**Charge route A (the compose claim). `claim_explicit_standard_name_sources` —
`imas_codex/standard_names/graph_ops.py:10444`:**

```python
def claim_explicit_standard_name_sources(source_ids: list[str], *,
                                         timeout_minutes: int = 30) -> list[dict[str, Any]]
```

The charge is inside the claim's `SET` at `:10471`:
`sns.attempt_count = coalesce(sns.attempt_count, 0) + 1`. The `WHERE` at `:10456`
admits `status = 'extracted'` with no live claim — it does not look at existing
`PRODUCED_NAME` bindings, which is the open defect
(`f-wcr-the-claim-charges-for-a-refusal-it-can-predict`: 51 sources at the cap of
a predictable conflict). **A caller must supply** bare `source_id` values (the
client prefixes `dd:` itself) and at most a claim timeout.

**Charge route B (failure marking). `mark_sources_failed` —
`graph_ops.py:11398`:**

```python
def mark_sources_failed(token: str, source_ids: list[str], error: str, *,
                        max_attempts: int = 3) -> int
```

Charges at `:11421` and parks at terminal `failed` when the cap is reached; it
refuses an empty failure reason before opening the graph. **A caller must
supply** the committed claim token, the ids, and a real reason string.

Cap constant: `_MAX_COMPOSE_CLAIM_ATTEMPTS = 5` — `graph_ops.py:15914`.
Parked dispositions are read back by `census_parked_dispositions`
(`graph_ops.py:11579`) and surfaced via `record_dispositions_for_undisposed_parks`
(`:11537`).

## 4. The governed source-status reset path — located

**Route A (manifest-exact reset). `reset_standard_name_sources` —
`imas_codex/standard_names/provenance_lifecycle.py:701`:**

```python
def reset_standard_name_sources(gc: Any, manifest_rows: Sequence[Mapping[str, Any]], *,
                                manifest_id: str, reason: str, include_accepted: bool = False,
                                publication_authority: str | None = None, dry_run: bool = False,
                                _transactional: bool = False) -> dict[str, Any]
```

Each row must carry `source_id`, `expected_status`, `expected_scalar` and a
complete non-empty `expected_bindings` list (the scalar must name one of the
bindings); the transaction detaches exactly those bindings, clears
composition/claim state, returns the source to `extracted` and writes a
deterministic `StandardNameSourceRetry` event. Immutable DD/signal edges and
every StandardName lifecycle field are untouched. The two Cypher bodies are
marked `EXACT_SOURCE_RESET_PREFLIGHT` (`:810`) and `EXACT_SOURCE_RESET_APPLY`
(`:920`). **A caller must supply** a non-empty `manifest_id` and `reason`; a
reset typed from live state without those is refused.

**Route B (operator retry). `retry_failed_sources` — `graph_ops.py:11616`:**

```python
def retry_failed_sources(source_paths: list[str], *, reason: str,
                         dry_run: bool = False, gc: Any | None = None) -> dict[str, Any]
```

Accepts `dd:<path>` / `signals:<id>` or bare ids; refuses without a non-empty
reason; compare-and-set guards the prior attempt count. Returns
requested/eligible/retry:refused counts plus created event ids.

**Reachable today:** `signed_manifest.py:6730` → `:6744`; the CLI route at
`imas_codex/cli/sn.py:6968` (`retry_failed`); tests
`tests/standard_names/test_provenance_lifecycle.py:486`, `:501`, `:514`, `:532`,
`:547`, `:578`, `:599`, `:622`, `:644`, `:669` and
`tests/standard_names/test_live_binding_migration.py:125`.

**The related refusal, which is not a reset.** `retarget_standard_name_sources`
— `provenance_lifecycle.py:390`:

```python
def retarget_standard_name_sources(
    gc: Any, old_name: str, new_name: str, *, operation: str = "refine",
    reason: str | None = None, origin: str | None = None, run_id: str | None = None,
    record_change: bool = True, enforce_consistency: bool = True,
    source_ids: Sequence[str] | None = None,
    expected_current_bindings: Mapping[str, str] | None = None,
    _transactional: bool = False, _allow_empty_noop: bool = False,
) -> int
```

It never touches `attempt_count` and already refuses a source bound to a different
live identity, embedding the conflicting bindings in its `RuntimeError` at
`:554`. A node that would rebuild the refusal should use this one.

## 5. The docs-axis reconciliation — located, with no production caller

```python
def reconcile_docs_axis_from_reviews(*, dry_run: bool = True,
                                     ids: list[str] | None = None,
                                     gc: Any | None = None) -> dict[str, int]
```

`imas_codex/standard_names/graph_ops.py:12608`.

Selection rule, from the docstring: a name's docs axis is taken from its winning
docs-axis review — the highest-ranked surviving docs review group (canonical
before non-canonical, newest group, then newest record inside it), as defined
once by `_docs_review_winner_query_body`. A name is repaired only when its name
axis is already `accepted`, it is not quarantined, and its stored mirror
disagrees with the winner. A name with no surviving
docs-axis review is **refused** — it cannot reach the update — which is the
load-bearing half that stops the pass manufacturing false-acceptance
projections.

Returns `{"agree", "repair", "with_winner", "refused_no_review"}` in dry-run
mode and `{"repaired"}` after a real run; idempotent. Test:
`tests/standard_names/test_docs_axis_reconciliation.py:77`.

**The gap, named rather than declared absent.** The only occurrences of the symbol
in the repository are its own definition and its test — a grep of
`imas_codex/` and `tests/` for the symbol returns `graph_ops.py:12608` and
`tests/standard_names/test_docs_axis_reconciliation.py`. So the machinery exists and the
release path does not call it; a node that would build a docs-axis
reconciliation should instead wire a caller (the natural one being the
`sn run` global-maintenance step that already calls its name-axis sibling
`reconcile_reviewable_name_stage`, `graph_ops.py:14275, invoked at loop.py:1783`)
and re-derive through it. Nearest existing
things that are *not* it: `orphan_sweep.py` (repairs `docs_stage='refining'`
stranded by a dead worker — a sweep, not a review-edge reconciliation) and the
hand-written GraphClient transaction recorded in
`docs/evidence/sn-west-catalog-release/docs-axis-reconciliation.md`.