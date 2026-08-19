# Fixture StandardNameSource cleanup

## Outcome

The live graph cleanup removed the complete signed fixture cohort: **16 of 16 `StandardNameSource` nodes**, with **0 relationships** because reconciliation had already made every residue stale, detached, and scalar-null. The deletion query was restricted to the exact signed IDs. The non-fixture source count remained **9,498 → 9,498** and the non-fixture incident-relationship count remained **16,011 → 16,011**, so the measured collateral deltas are both zero.

Both source tests now clean the `dd:`-prefixed source namespace they create, perform teardown in `finally`, and seed the required `source_id` field. Each file includes a graph regression that exercises its cleanup helper. Independent credentialed runs completed without graph-marker skips:

| Test file | Result | Post-run fixture nodes |
| --- | ---: | ---: |
| `tests/standard_names/test_minting.py` | 8 passed, 0 failed, 0 skipped; exit 0 | 0 |
| `tests/standard_names/test_focus_scope.py` | 3 passed, 0 failed, 0 skipped; exit 0 | 0 |

The independent post-run census covers fixture `StandardName` identities as well as `StandardNameSource` identities. It found **0 fixture nodes and 0 fixture relationships** across `__minttest__*`, `dd:__minttest__/*`, `test_focus_scope__*`, and `dd:test_focus_scope__*`.

## Read-only pre-census

The pre-census used the project's credentialed `GraphClient` path and enumerated every fixture source plus all incoming and outgoing relationships. All 16 sources had `status=stale`, a restored non-null `source_id`, `produced_sn_id=null`, and no surviving relationship. The absence of a live name binding or review score is itself the residue signature: these are detached test paths, not production semantic claims.

| Source ID | `source_id` | Relationships |
| --- | --- | ---: |
| `dd:__minttest__/leaf1` | `__minttest__/leaf1` | 0 |
| `dd:__minttest__/leaf_dead` | `__minttest__/leaf_dead` | 0 |
| `dd:test_focus_scope__s_18b826b2` | `test_focus_scope__s_18b826b2` | 0 |
| `dd:test_focus_scope__s_2ac9c809` | `test_focus_scope__s_2ac9c809` | 0 |
| `dd:test_focus_scope__s_375cc5df` | `test_focus_scope__s_375cc5df` | 0 |
| `dd:test_focus_scope__s_39cae775` | `test_focus_scope__s_39cae775` | 0 |
| `dd:test_focus_scope__s_4613da33` | `test_focus_scope__s_4613da33` | 0 |
| `dd:test_focus_scope__s_5bfe0958` | `test_focus_scope__s_5bfe0958` | 0 |
| `dd:test_focus_scope__s_70399d54` | `test_focus_scope__s_70399d54` | 0 |
| `dd:test_focus_scope__s_8a614b99` | `test_focus_scope__s_8a614b99` | 0 |
| `dd:test_focus_scope__s_a2433af8` | `test_focus_scope__s_a2433af8` | 0 |
| `dd:test_focus_scope__s_a74a4cf4` | `test_focus_scope__s_a74a4cf4` | 0 |
| `dd:test_focus_scope__s_a752e831` | `test_focus_scope__s_a752e831` | 0 |
| `dd:test_focus_scope__s_b66d8824` | `test_focus_scope__s_b66d8824` | 0 |
| `dd:test_focus_scope__s_ee393d8b` | `test_focus_scope__s_ee393d8b` | 0 |
| `dd:test_focus_scope__s_f962c161` | `test_focus_scope__s_f962c161` | 0 |

Full pre-census artifact: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T002113839002-fixture-source-cleanup/pre-census.json`, SHA-256 `ceb89bac555fe3d3a17de9c9363db8916b1cbf65275345857b81e87952784d36`.

## Deletion and collateral receipt

The delete transaction first re-read the live fixture IDs and incident relationship counts and required exact equality with the pre-census. It then deleted only those exact IDs and verified the postconditions.

| Measure | Before | Removed | After |
| --- | ---: | ---: | ---: |
| Fixture sources | 16 | 16 | 0 |
| Fixture incident relationships | 0 | 0 | 0 |
| Non-fixture sources | 9,498 | 0 | 9,498 |
| Non-fixture incident relationships | 16,011 | 0 | 16,011 |

Deletion receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T002113839002-fixture-source-cleanup/deletion-receipt.json`, SHA-256 `3d2c2caf2d9c094d57e687d948ac90de0309f2dc0b7ef3dbca01c7c620ce405a`.

## Prevention and validation

- Minting cleanup now matches both the fixture-name prefix and `dd:__minttest__/`, and its fixture teardown is exception-safe.
- Focus-scope cleanup now matches source IDs with `dd:test_focus_scope__` rather than the unprefixed name namespace, and its fixture teardown is exception-safe.
- Both seed paths populate `StandardNameSource.source_id`, so even an interrupted fixture setup does not introduce a required-property violation before teardown.
- Each file has a cleanup regression that creates a fixture source, invokes the production test helper, and asserts the complete fixture namespace is empty.
- Ruff formatting and `ruff check --no-cache` passed with exit 0.
- The mandatory path, label, deliverable-ID, and changelog-prose checks passed after review of their domain false positives.

Independent post-test census: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T002113839002-fixture-source-cleanup/post-test-census.json`, SHA-256 `4b799805355e574c827e16ff9358e11715ced10a2c2c0f1cee6ee37bb9a221e0`.

The canonical ignored `.env` was copied only for the credentialed graph operations and tests. It was removed before staging and was never added to the index.
