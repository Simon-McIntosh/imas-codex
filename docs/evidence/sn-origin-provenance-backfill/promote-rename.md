# Catalog promotion module and approval receipts

## Outcome

The catalog fold-back implementation now lives in
`imas_codex/standard_names/promote.py`; `merge.py` was removed. The public CLI
surface remains `imas-codex sn merge`, while every Python import and patch target
uses `imas_codex.standard_names.promote`. A repository search found zero remaining
references to `imas_codex.standard_names.merge` or `from ...standard_names import
merge` under `imas_codex/` and `tests/`.

Production call sites changed only at the module boundary:

- `imas_codex/cli/sn.py`: status, merge, approval-override, and revert imports.
- `imas_codex/standard_names/release_notes.py`: shared catalog diff-reader import.

Six existing test modules changed only at their imports or Python patch-target
module paths. The provenance guard filename was aligned from
`test_merge_origin_guard.py` to `test_promote_origin_guard.py`. The existing test
bodies and assertions were otherwise retained.

## Approval change row

Both successful approval paths create exactly one `StandardNameChange` and link it
from the approved identity with `HAS_INTERNAL_CHANGE` in the same Cypher mutation
that applies the approval stamp.

| Field | Unchanged catalog approval | Contested content-edit override |
|---|---|---|
| `id` | `sn-change:` plus a generated UUID | `sn-change:` plus a generated UUID |
| `from_name` / `to_name` | Approved identity / same identity | Approved identity / same identity |
| `operation` | `unchanged_ratification` | `content_edit` |
| `reason` | Catalog PR and recorded outcome | Required override justification |
| `origin` | `catalog_promotion` | `catalog_override` |
| `changed_at` | Transaction timestamp | Transaction timestamp |
| `internal` | `true` | `true` |

The edited-name path in `run_merge` passes `content_edit` explicitly; the untouched
batch path passes `unchanged_ratification` explicitly. The approval mutations do
not assign `origin`, `model`, `generated_at`, or `chain_length` on the
`StandardName`. The new stateful regression proves those four values are identical
before and after both approval forms while each approval emits one row and the two
rows carry different editorial outcomes.

## Validation

| Measurement | Before | After |
|---|---:|---:|
| Focused module and Cypher-property tests passed | 91 | 92 |
| Graph-marked tests deselected | 9 | 9 |
| New receipt/provenance tests passed | not present | 3 |
| Remaining old module imports under `imas_codex/` and `tests/` | 8 files | 0 |
| Production graph contacts | 0 | 0 |

The focused after-run included `tests/graph/test_cypher_property_check.py`; all 92
selected tests passed. The one-test increase is
`test_approval_rows_distinguish_ratification_from_content_edit`. The nine graph
tests in the selected files remained deselected, so validation contacted no live or
production graph. Cypher behavior was checked with stateful graph doubles and the
repository LinkML property-inventory test.

Full logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163311906924-n-promoterename/baseline-tests.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163311906924-n-promoterename/focused-tests.log`
