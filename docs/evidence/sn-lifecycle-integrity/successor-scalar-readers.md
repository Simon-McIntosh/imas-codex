# Every reader of the `superseded_by` scalar, with its lineage-edge replacement

Read at HEAD `9705954c264bd4c51c78d68d9be23be54178c2ac`. The task is to
enumerate every site that reads the `StandardName.superseded_by` scalar and to
state, for each, the lineage-edge expression that would replace it, so the
field's retirement can be sequenced as a known list of edits rather than as a
search.

**Count: 42 matching lines across 20 files** — eleven under `imas_codex/`,
nine under `tests/`. The plan (plan comment `c-20260905-impl-corrected`)
asserts the field is "declared and written in six modules"; **that assertion
holds exactly.** The six production modules are `protection.py`,
`canonical.py`, `graph_ops.py`, `signed_manifest.py`, `catalog_import.py` and
`edit.py`. The other five `imas_codex/` files are the LinkML declaration, its
three generated artifacts, and one documentation row; none of them reads the
value.

**Of the 42 lines only 4 read the value in production code, and 3 of those read
the graph property** — `graph_ops.py:3193`, `signed_manifest.py:4090`,
`edit.py:2104`. `catalog_import.py:332` reads the *catalog model's* attribute of
the same name, not the graph property. The rest of the production lines are one
two-route problem, of the remaining 12 production lines: one write, three
membership declarations (two for the serialisation round trip, one for the
write guard), one expected-value literal and three documentation lines. The
remaining 22 lines are test fixtures and assertions.

## The two replacement routes

**Route L — the lineage edge (canonical).** The successor of a superseded name
is the source of the inbound lineage edge:

```cypher
MATCH (succ:StandardName)-[:REFINED_FROM]->(pred:StandardName {id: $id})
RETURN succ.id AS successor
```

Measured over the 2,163 `name_stage='superseded'` names: this answers for
**1,641**, against 18 for the scalar, and three names are carried by the scalar
alone (`flux_due_to_thermal_fusion`, `area_of_flux_surface`,
`lower_energy`). Every replacement of a *graph read* is Route L.

**Route W — the write-side edge.**
A writer cannot read lineage out of a batch payload, so a writer that today
copies `superseded_by` into `HAS_SUCCESSOR` must either (a) take the successor
from the graph by Route L after the write, or (b) mint the `[:REFINED_FROM]`
edge itself so that the edge's own consumers read Route L. Which of the two is
a decision, not a substitution.

## `imas_codex/standard_names/` — the six production modules

| # | file:line | role of the line | replacement | verdict |
|---|---|---|---|---|
| 1 | `graph_ops.py:3193` | **read of the value.** `successor = n.get("successor") or n.get("superseded_by")`, feeding the `HAS_SUCCESSOR` batch; the edge writer at `:3369` is the only production write of a `StandardName` `HAS_SUCCESSOR` edge, and it draws on this line alone | Route W. The payload carries no lineage, so this becomes a graph read by Route L for the ids whose `successor` is null — or the `HAS_SUCCESSOR` projection is retired and its consumers read Route L directly (measured: **48** `HAS_SUCCESSOR` edges against **1,641** lineage edges) | **decision-needing** — the two options differ in what the graph stores, and `HAS_SUCCESSOR` is derived from this line, so it cannot serve as its own replacement |
| 2 | `graph_ops.py:3103` | docstring documenting the edge derivation | restate as the lineage edge | mechanical |
| 3 | `graph_ops.py:3351` | comment naming the catalog-import spelling | restate as #2 | mechanical |
| 4 | `signed_manifest.py:4090` | **read of the value.** Receipt postcondition: `RETURN node.superseded_by AS superseded_by`, compared equal to the signed successor | assert the edge instead: `RETURN EXISTS { MATCH (succ:StandardName {id: $successor_id})-[:REFINED_FROM]->(node) } AS lineage_edge` | mechanical **only if** the mutation at #6 stops writing the field; otherwise the receipt loses its record of the successor |
| 5 | `signed_manifest.py:4097` | the same postcondition's expected dict (`"superseded_by": successor_id`) | the expected member becomes the edge assertion | mechanical (follows #4) |
| 6 | `signed_manifest.py:4712` | **write.** `target.superseded_by = $successor_id` inside the signed `supersede` mutation, which sets `name_stage`/`status`/`source_paths` and **mints no edge at all** | the mutation mints `[:REFINED_FROM]` (Route W option b), or the successor assertion leaves the receipt entirely | **decision-needing, highest risk in this enumeration** — the scalar is the only record of the successor for this mutation, so deleting the clause silently loses it, and the postcondition at #4 is the only thing that proves it landed |
| 7 | `edit.py:2104` | **read of the value.** `target_properties.get("superseded_by")` in the tombstoned-fold guard, consulted *after* the lineage walk, only to refuse a successor no lineage carries | delete the consult and its refusal branch: with the field gone there is nothing to disagree with, and `_fold_lineage_walk` already reads `REFINED_FROM` for the relation | mechanical (deletion), gated by tests #26–#30 |
| 8 | `edit.py:2085` | docstring explaining that the scalar is not the route | restate: Route L, alone | mechanical |
| 9 | `canonical.py:35` | `NULLABLE_SCALAR_FIELDS` membership — round-trip defaulting to `None` | remove the member | mechanical |
| 10 | `canonical.py:69` | `CANONICAL_KEY_ORDER` membership — the export/import key surface; `reorder_entry_dict` (`export.py:2187`) emits the key when present and raises `UnknownCatalogKeyError` on a key not in the order | remove the member | **decision-needing** — a stale external catalog YAML carrying the key would then be *refused* rather than silently stripped. Refusal is the right default; the decision is whether the release notes say so |
| 11 | `catalog_import.py:332` | **read of a catalog field, not the graph property.** `entry.superseded_by` copied into the graph write payload | delete the key. The exporter never emits it — `_graph_node_to_entry_dict` builds a fixed literal dict and omits it — so the key can only arrive from a hand-edited or stale YAML | mechanical, pending one check: whether the ISN entry model still declares the attribute (`test_description_latex.py:116` exists precisely because this line touches it unconditionally) |
| 12 | `protection.py:34` | `PROTECTED_FIELDS` membership — the pipeline write guard | remove the member; the guard protects a field that will no longer exist | mechanical |

## `imas_codex/` outside the six modules

| # | file:line | role | replacement | verdict |
|---|---|---|---|---|
| 13 | `schemas/standard_name.yaml:925` | the LinkML **declaration** of `StandardName.superseded_by` — the source the three generated artifacts derive from | delete the slot and regenerate; the declaration is what makes the removal reach the generated files | **decision-needing as the migration trigger**: no generated artifact is ever staged by hand, so this one line is the whole migration for the schema surface |
| 14 | `schemas/standard_name.yaml:1996` | declaration of the **`successor`** slot, whose description reads "Derived from the superseded_by scalar at write time" | re-source `successor` from Route L over the lineage edge | **decision-needing** — `successor` is a second scalar carrying the same relation; retiring `superseded_by` without re-sourcing or retiring `successor` leaves a field documenting a derivation that no longer exists |
| 15 | `standard_names/AGENTS.md:111` | documentation row listing `StandardName.superseded_by` among protected fields | drop the token from the row | mechanical |
| 16 | `graph/models.py:4636,5049` | generated attribute declarations (never staged) | regenerate after #13 | mechanical |
| 17 | `graph/dd_models.py:3178,3527` | generated attribute declarations | regenerate after #13 | mechanical |
| 18 | `graph/schema_context_data.py:1258` | generated context entry | regenerate after #13 | mechanical |

## `tests/` — nine files, 22 lines

| # | file:line | role | replacement | verdict |
|---|---|---|---|---|
| 19 | `test_export_deprecation.py:72,151` | fixture inputs carrying `superseded_by=` into the export candidate | delete the kwarg | mechanical |
| 20 | `test_export_deprecation.py:119,157` | assertions that the exported entry dicts and the graph-node projection carry **no** `superseded_by` key | keep unchanged — these are already the retirement's regression evidence | mechanical (no change) |
| 21 | `test_graph_edge_writers.py:11,734,739` | docstring/comment naming the scalar as the `HAS_SUCCESSOR` source | restate as Route W, or delete with the test if `HAS_SUCCESSOR` retires | mechanical |
| 22 | `test_graph_edge_writers.py:348` | payload fixture feeding `superseded_by` in the endpoint-authority test | the member only exercises a writer; drop it or move it to `successor` | mechanical |
| 23 | `test_graph_edge_writers.py:746,759` | class `TestG8`, the test that pins "`superseded_by` → `HAS_SUCCESSOR`" | re-point at the lineage route, or delete with `HAS_SUCCESSOR` | **decision-needing** — whichever route #1 takes decides whether this test is rewritten or removed, and it is the same surface |
| 24 | `test_protection.py:205` | expected `PROTECTED_FIELDS` set | drop the member from the expected set | mechanical |
| 25 | `test_description_latex.py:116` | `entry.superseded_by = None` on a mock, needed only because `catalog_import` touches the attribute | delete when #11 lands | mechanical |
| 26 | `test_supersede_into_tombstone.py:36` | state fixture setting the scalar `None` (tombstone baseline) | delete the line | mechanical |
| 27 | `test_supersede_into_tombstone.py:136` | state fixture setting a scalar to drive the "records successor X" refusal | delete with the refusal branch (#7); the lineage-side case is separately covered at `test_tombstone_fold_lineage.py:48` | mechanical |
| 28 | `test_fold_guard_current_successor.py:60` | negative fixture (`superseded_by=None`) for the admitted free tombstone | delete the kwarg | mechanical |
| 29 | `test_supersede_successor_scalar.py:234,245` | the **gate on the signed supersede write**: asserts the applied mutation left `superseded_by` equal to the signed successor | assert whatever replaces the mutation's record (#6) — a Route L edge read | **decision-needing** — it pins exactly the behaviour #6 changes |
| 30 | `test_tombstone_fold_lineage.py:6,35,48,93` | module docstring quoting the census ("1588 of 2102", "scalar written on 18") and fixtures driving both refusal branches | delete the two scalar fixtures with #7; update the docstring to the current census (2,163 superseded / 1,641 edges / scalar 18) | mechanical |
| 31 | `test_sn_tools.py:543` | **not a reader — a name collision.** `test_list_excludes_superseded_by_default` asserts only that `name_stage` and `superseded` appear in the generated Cypher; it never touches the scalar | none — leave unchanged | mechanical (no change); recorded so the next enumeration does not count it |

## What this enumeration does and does not establish

- **It is a static enumeration at one revision.** Each line was read in the
  worktree at HEAD; the roles above are a reading of the code, not a measured
  behaviour. The three production value reads were traced to their consumers by
  reading the blocks around them: `HAS_SUCCESSOR` (#1), the signed receipt
  postcondition (#4), the fold refusal (#7).
- **Counted once, by role.** The six-module assertion is exact for production
  modules. The full retirement touches **20 files / 42 lines**: 6 production
  modules, 1 schema file, 3 generated artifacts, 1 documentation file and 9 test
  files. Six sites need a decision — #1, #6 and their two gates (#23, #29), plus
  #10 and #13/#14 on the schema axis.
- **No figure accompanies this record, deliberately.** The relationship here is
  a three-column table (line, replacement, verdict) whose whole point is that a
  reader can look up a site; the two-route diagram would be smaller than its own
  caption. The one sequential fact that is not table-shaped — that `HAS_SUCCESSOR`
  is derived from the scalar and therefore cannot replace it — is stated in #1
  and is a single edge, not a diagram.
- **The measure is bounded to this node's own change.** Verifying the merged
  result belongs to a separately dispatched test node.