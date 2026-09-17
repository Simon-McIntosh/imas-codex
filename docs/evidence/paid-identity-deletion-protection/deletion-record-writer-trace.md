# The deletion record is written by one Cypher builder, eight routes reach it, and no production removal ever has

**Question.** Why does `StandardNameDeletionSnapshot` hold no production removal, and do the
production delete paths reach the symbol that writes it?

**Answer.** One symbol writes both labels — `deletion_change_cypher` in
`imas_codex/standard_names/provenance_lifecycle.py` — and eight production call sites reach it,
but two production routes that remove a `StandardName` do not. The record's entire 61→64-row
content is live-graph test fixtures: **0 rows carry a non-synthetic `original_id`**, measured
against a cohort of 93 names that were removed on 2026-09-08, 0 of which were recorded.

## 1. The writer symbol

Anchored on the worktree at `b662e216`:

| Item | Location |
|---|---|
| Writer | `deletion_change_cypher(name_alias)` — `imas_codex/standard_names/provenance_lifecycle.py:89` |
| `CREATE (snapshot:StandardNameDeletionSnapshot)` | `provenance_lifecycle.py:116` |
| `CREATE (edge_snapshot:StandardNameDeletedEdge)` | `provenance_lifecycle.py:123` |
| Change row | `provenance_lifecycle.py:105` (`:StandardNameChange`) |
| Snapshot link | `provenance_lifecycle.py:121` (`HAS_DELETION_SNAPSHOT`) |

Two properties make the orphan test in §5 exact rather than heuristic:

- `StandardNameDeletionSnapshot.original_id = <deleted node>.id`
- `StandardNameDeletionSnapshot.id = <change id> + ':node'`

So a snapshot's own id names the `StandardNameChange` it belongs to. The snapshot clause entered
in commit **`2e0c1b534`** ("retain typed deletion snapshots"), dated 2026-09-08 18:42:48+0200 =
**16:42:48Z**, carrying the writer and the edge inventory in one statement.

## 2. Every production route that removes a `StandardName`, and whether it calls the writer

Enumerated from the `DETACH DELETE` statements in the two standard-name modules
(`provenance_lifecycle.py`, `graph_ops.py`) plus `imas_codex/graph/sn_cleanup.py`. Ten
statements can terminate a `StandardName`; eight of them are reached through a
`deletion_change_cypher` call site.

<figure>
<svg viewBox="0 0 900 300" width="100%" role="img"
     aria-label="Eight delete routes reach the deletion_change_cypher writer; two remove a StandardName without it.">
  <g font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#1a1a1c">
    <rect x="4" y="24" width="320" height="252" fill="#f2fbf2" stroke="#2e7d32" stroke-width="1.2"/>
    <text x="16" y="46" font-weight="600" fill="#2e7d32">Reaches the writer (8 call sites)</text>
    <text x="16" y="66" font-size="11">deletion_change_cypher → snapshot + edges</text>
    <text x="16" y="90">pl:89  writer</text>
    <text x="16" y="110" fill="#1a1a1c">pl.cancel_staged_rename :208 → :258</text>
    <text x="16" y="130" fill="#1a1a1c">pl.retire_unrecoverable_provenance_orphans</text>
    <text x="60" y="145" fill="#8a8a8f" font-size="11">:1788 → :1810</text>
    <text x="16" y="165" fill="#1a1a1c">pl.compact_unapproved_superseded :1924 → :1936</text>
    <text x="16" y="185" fill="#1a1a1c">go._delete_derived_parent_nodes :3853 → :3904</text>
    <text x="16" y="205" fill="#1a1a1c">go.write_standard_names :5961 → :5968</text>
    <text x="16" y="225" fill="#1a1a1c">go.clear_standard_names :9067 → :9111</text>
    <text x="16" y="245" fill="#1a1a1c">go.clear_standard_names :9181 → :9197</text>
    <text x="16" y="265" fill="#1a1a1c">go.clear_sn_subsystem :9312 → :9317</text>

    <rect x="348" y="24" width="300" height="128" fill="#fdf3f2" stroke="#b3261e" stroke-width="1.2"/>
    <text x="360" y="46" font-weight="600" fill="#b3261e">Removes a name, no writer (2)</text>
    <text x="360" y="68" fill="#1a1a1c">go.clear_standard_names :9144</text>
    <text x="360" y="84" font-size="11">DETACH DELETE parent — change hand-written at</text>
    <text x="360" y="98" font-size="11">:9130, no snapshot, no link</text>
    <text x="360" y="108" fill="#1a1a1c">imas_codex/graph/sn_cleanup.py:100</text>
    <text x="360" y="124" font-size="11">purge_standard_names (def :92) — no change row,</text>
    <text x="360" y="138" font-size="11">and zero call sites in the tree</text>

    <rect x="348" y="166" width="300" height="110" fill="#ffffff" stroke="#8a8a8f" stroke-width="1"/>
    <text x="360" y="188" font-weight="600">Not a name-lifecycle route</text>
    <text x="360" y="208" font-size="11">graph/client.py:444 — whole-store wipe</text>
    <text x="360" y="224" font-size="11">graph/temp_neo4j.py — disposable temp database</text>
    <text x="360" y="246" font-size="11">clear_sn_subsystem's sibling deletes remove other</text>
    <text x="360" y="260" font-size="11">labels (source, docs, vocab, run), not the identity</text>
  </g>
</svg>
<figcaption>The writer is on eight of the ten statements that can terminate an identity. Both
that bypass it are production code, and one runs in the exact-reset branch of an ordinary
pipeline clear.</figcaption>
</figure>

| # | Route | Operation recorded | Calls the writer |
|---|---|---|---|
| 1 | `provenance_lifecycle.cancel_staged_rename` (`:208`) | rename-cancel | **yes** |
| 2 | `provenance_lifecycle.retire_unrecoverable_provenance_orphans` (`:1788`) | orphan retirement | **yes** |
| 3 | `provenance_lifecycle.compact_unapproved_superseded` (`:1924`) | compaction | **yes** |
| 4 | `graph_ops._delete_derived_parent_nodes` (`:3853`) | `remove_derived_parent` | **yes** |
| 5 | `graph_ops.write_standard_names` (`:5961`) | `remove_skeleton_placeholder` | **yes** |
| 6 | `graph_ops.clear_standard_names`, relationship-first (`:9067`) | `clear_selected_name` | **yes** |
| 7 | `graph_ops.clear_standard_names`, else branch (`:9181`) | `clear_selected_name` | **yes** |
| 8 | `graph_ops.clear_sn_subsystem` (`:9312`) | subsystem clear | **yes** |
| 9 | `graph_ops.clear_standard_names`, exact-reset skeleton parent (`:9144`) | `remove_skeleton_placeholder`, hand-written at `:9130` | **no** |
| 10 | `imas_codex/graph/sn_cleanup.purge_standard_names` (`:100`) | none | **no** |

Routes 9 and 10 answer the second half of the question: **not every production delete path
reaches the writer.** Route 9 is a live production branch — the `path_allowlist` exact-reset
branch of `clear_standard_names` — whose `DETACH DELETE parent` at `graph_ops.py:9144` removes a
`StandardName` while the hand-written change row at `:9130` carries no snapshot and no
`HAS_DELETION_SNAPSHOT` edge. Route 10 is `purge_standard_names` (`def` at `sn_cleanup.py:92`,
`DETACH DELETE sn` at `:100`), a raw store-level purge that writes no change row at all; it has
zero call sites in the tree and is reachable only from a REPL.

## 3. Live measurement of the record (2026-09-17)

Single-pass, read-only; full output in `deletion_record_consolidated.log` in the run directory.

| Measure | Value |
|---|---|
| `StandardNameDeletionSnapshot` total | **64** |
| … carrying a **non-synthetic** `original_id` | **0** |
| … carrying a synthetic (`__…`) `original_id` | 64 |
| Capture window | **2026-09-10T11:37:45.817Z … 2026-09-17T20:58:55.116Z** |
| `StandardNameDeletedEdge` total | **336** |
| … with a non-synthetic `neighbor_id` | **0** |

**Positive controls beside the zeros:** `StandardName` **5,130** ·
`StandardNameDeletionSnapshot` **64** · `StandardNameDeletedEdge` **336** · snapshots whose
`id` ends in `:node` **64 / 64**.

The window moved while this read-only node ran: the predecessor recorded 61 rows / 316 edges
over 2026-09-10 … 2026-09-15, and the consolidated read found 64 / 336 with the end at
2026-09-17T20:58:55Z. **The count is not a stable instrument — it is a function of how many
graph-marked tests have run**, which is itself part of the finding.

## 4. Coverage of the removal cohort

| Measure | Value |
|---|---|
| `remove_derived_parent` rows, 2026-09-08 11:57:00–11:58:00Z, distinct `to_name` | **93** (positive control) |
| … of those covered by a deletion snapshot | **0** |
| `remove_derived_parent` rows, all time | 2,644 — **0 with a snapshot**, last `2026-09-08T11:57:36Z` |
| `remove_skeleton_placeholder` rows, all time | 1,865 — **0 with a snapshot**, last `2026-08-02` |
| `clear_selected_name` rows, all time | 68 — **16 with a snapshot** |

The instrument is not blind: `cohort_size` returns **93** on the same statement that returns
`covered = 0`. A zero measured over an actually populated cohort. Only `clear_selected_name`,
whose 16 snapshots are the `__cleartest__` fixture pairs of §5, has ever produced a snapshot —
and `remove_derived_parent` has produced none since 2026-09-08.

The record was last written 2026-09-17T20:58:55Z.

## 5. What the removal that was missed is, and why it was missed

`remove_derived_parent` changed 2,644 rows and produced 0 snapshots. The reason is temporal
before it is structural:

| Fact | Time (UTC) |
|---|---|
| Last `remove_derived_parent` removal | **2026-09-08T11:57:36Z** |
| `remove_skeleton_placeholder` last fired | 2026-08-02 |
| Writer `deletion_change_cypher` introduced | **2026-09-08T16:42:48Z** (`2e0c1b534`) |

Every production removal precedes the writer by about four hours, and no
`remove_derived_parent` row exists after 2026-09-08. The reason the record is empty is not that
the graph holds no deletions — 2,644 and 1,865 rows say otherwise — but that the writer did not
exist when they happened, and no production removal has occurred since. That is a temporal
cohort. Adding to that, two production routes would still bypass the writer (routes 9 and 10),
so the door is incomplete regardless of history.

## 6. Why the rows that do exist are orphans

| Measure | Value |
|---|---|
| snapshots whose change row is still present | **16** |
| snapshots whose change row is absent (orphaned) | **48** |

The two fixture modules carry reserved prefixes: `test_clear_atomicity_graph.py` writes
`__cleartest__` (16 snapshots) and `test_exact_reset_parent_cleanup_graph.py` writes
`__exact_reset_scaffold__` (48 snapshots).

The 48 are orphans because that test's teardown wipe matches on `from_name`:

```cypher
MATCH (node) WHERE node.id STARTS WITH $prefix
   OR node.source_id STARTS WITH $prefix
   OR node.from_name STARTS WITH $prefix
DETACH DELETE node
```

`StandardNameChange.from_name` carries the prefix, so the change row is matched. The snapshot
does not (`original_id` carries it; `from_name` is not a snapshot property), so the snapshot is
left behind. The writer creates the change and its snapshot in one statement, so the pair is
always born together and deletes in one and leaves the other. The evidence:

| Label / predicate | Value |
|---|---|
| `StandardNameChange` with `from_name STARTS WITH '__exact_reset_scaffold__'` | **0** |
| `StandardNameDeletionSnapshot` with `original_id STARTS WITH '__exact_reset_scaffold__'` | **48** (>0) |
| `StandardNameChange` with `from_name STARTS WITH '__cleartest__'` | 65 |

The asymmetry is the proof: the change rows the teardown matches are gone, and the snapshots it
does not match remain. The `__cleartest__` module's wipe iterates the labels
`("StandardName", "StandardNameReview", "IMASNode")` keyed on `n.id`, so it never touches either
`StandardNameChange` or the snapshot, and its 16 pairs survive intact.

## What this establishes

1. **The writer exists in production code on eight routes.** `deletion_change_cypher` is not
   dead code; it has call sites in three lifecycle functions and five graph operations.
2. **Two production delete paths still bypass it.** `clear_standard_names` at
   `graph_ops.py:9144` removes a `StandardName` through a hand-written change with no
   recoverable state, and `sn_cleanup.purge_standard_names` removes it with no change row at
   all. A guard, if one is wanted, belongs on these two, not on the eight.
3. **The record's emptiness has two causes**, and they are disjoint. A temporal one: the
   removals were made four hours before the writer existed and none has been made since. A
   structural one: the routes above would not have recorded them anyway.
4. **The only rows the record does hold are graph-marked test fixtures**, and 48 of 64 are
   orphaned by a teardown keyed on `from_name`. Any count taken from this label measures the
   test suite's history, not production's — so the record is not a stable instrument.

## Reproduce

| Artifact | Path |
|---|---|
| Consolidated measurement | `deletion_record_consolidated.log` in the run directory |
| Source trace (writer, call sites, `DETACH DELETE` inventory) | `source_trace.log` in the run directory |
| Supporting reads | `deletion_record_query.log`, `..._query2.log`, `..._query3.log`, `..._query4.log` |
| Query set | `q5_consolidated.py` in the run directory |

Baseline: 61 rows / 316 edges reported by the predecessor evidence
`docs/evidence/paid-identity-deletion-protection/unreinstatable-role-gap.md`, measured over
2026-09-10 … 2026-09-15.

Reads were taken over the live graph from the login node because `NEO4J_URI` resolves through a
login-node-local tunnel (`bolt://localhost:17687`) that a compute node cannot establish. Every
query is a bounded indexed read over a named label or a named time window; no statement scanned
the whole store, the single computation measured less than one second per query, and no heavy
local compute was paired with it.

Per the node's evidence fence, this measures only this node's own change; merging and post-merge
verification belong to a separately dispatched test node.