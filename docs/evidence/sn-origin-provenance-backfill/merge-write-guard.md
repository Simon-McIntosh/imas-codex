# Catalog merge write guard

The catalog merge path now records catalog approval without changing an
existing Standard Name's editorial origin. No production graph was contacted;
all graph behavior was exercised through a stateful test double.

## Guarded writes

| Site | Guarded lines | Permitted write | Refused write |
|---|---:|---|---|
| Contested override | `imas_codex/standard_names/merge.py:554`–`560` | Promote the name, record the resolution, and preserve or add `catalog_approved_at` | `sn.origin = 'catalog_edit'` |
| Accepted-name catalog approval | `imas_codex/standard_names/merge.py:978`–`988` | Promote the name and record `catalog_pr_number`, `catalog_pr_url`, `catalog_merge_commit_sha`, and `catalog_approved_at` | `sn.origin = 'catalog_edit'` |

Both approval queries contain zero assignments to `sn.origin`. The ordinary
approval path still requires complete merged-pull-request metadata before it
can write, and both paths retain their catalog-approval timestamp behavior.

## Damage-shape regression

`tests/standard_names/test_merge_origin_guard.py` starts with a generated
identity carrying `origin='pipeline'`, a generation model, `generated_at`, and
`chain_length=3`. It exercises both catalog-approval sites and asserts that all
four generation properties survive while the catalog receipt remains present.
The graph double deliberately applies the removed origin assignment if it sees
that text in either Cypher statement, so the regression fails against the
unguarded queries.

## Test counts

| Measurement | Before | After | Result |
|---|---:|---:|---|
| Pre-existing merge-related and Cypher-property tests | 89 passed, 9 deselected | 89 passed, 9 deselected | Byte-unchanged suite stayed green |
| New origin-guard regressions | 0 | 2 passed | Both write sites covered |
| Combined focused run | 89 passed | 91 passed | Green; 2 warnings in both runs |

Baseline log: `/tmp/n-mergewriteguard-baseline.log`  
After-change log: `/tmp/n-mergewriteguard-after.log`

The combined command included
`tests/graph/test_cypher_property_check.py`; its three tests remained green.
