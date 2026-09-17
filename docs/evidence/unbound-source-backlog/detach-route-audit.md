# Every route that removes a PRODUCED_NAME edge, and which can orphan an accepted name

Read-only audit against worktree revision
`fcc65951bc52d083b74461f10ffbfe39b97c2624` and the live `codex` graph.

**No graph mutation was performed.** Every command in this audit is a read: the
live census opens a session and issues only `MATCH`/`RETURN`, and the route
census is a static read of the source tree.

A `PRODUCED_NAME` edge is the source of truth for which `StandardNameSource`
justifies a `StandardName`. An accepted name with no such edge is published into
a catalog with nothing in the data dictionary behind it. This audit enumerates
every code path that removes one and states, for each, whether it can take an
accepted name's **last** producer.

## The census

Twenty-four deletion sites exist in the standard-names write surface. They split
by what the deletion takes with it, and the split is not the same as the one it
first appears to be:

| Kind | Count | Can it leave an accepted name with no producer? |
| --- | --- | --- |
| Edge-removal: a `PRODUCED_NAME` relationship is deleted, the target name survives | **19** | Yes, unless a stage predicate fences the target |
| Node-deletion: the node carrying the edge is deleted | **5** | Only when the deleted node is a **source** — deleting a *name* removes the name, and deleting a *source* detaches every edge it carries |

Re-derive with `python3 route_census.py` — log `route-census.log`, EXIT=0.

### The 19 edge-removal routes

The `fence` column quotes the predicate standing between the route and an
accepted target. A route fenced only on an element id or a source-side scalar is
fenced on the *identity* of the edge, not on the *stage* of the name, and the
signed manifest will delete exactly the edge it names whatever stage the target
is at.

| Route | Fence (verbatim) | Can take an accepted name's last producer? |
| --- | --- | --- |
| `provenance_lifecycle.py:588` `retarget_standard_name_sources` | `MATCH (source)-[prior:PRODUCED_NAME]->(old) DELETE prior` … `MERGE (source)-[:PRODUCED_NAME]->(new)` — no predicate on `old.name_stage` | **Yes.** The edge moves to `new`; if `old` was accepted with only that source, `old` loses its last producer. Only the caller stands in the way, and that caller is vacuous on an empty cohort (below). |
| `provenance_lifecycle.py:936` `reset_standard_name_sources` | reports orphans; docstring: "Potential name orphans are reported for a separate lifecycle manifest" | **Yes, by design.** It detaches and reports rather than refusing; `include_accepted` is an explicit parameter, so the accepted case is reachable and supported. |
| `provenance_lifecycle.py:1258` `_SEMANTIC_SOURCE_REPAIR_MUTATION` (module-level string constant, not a function body) | deletes stale edges whose target is not the repair target; the live-target test excludes `superseded\|exhausted` only | **Yes.** `accepted` is not in the exclusion set, so an accepted name that loses the semantic contest has its edges deleted as "stale". This route converges and does not retarget; the defect is that the accepted rival's stage is never consulted. |
| `provenance_lifecycle.py:252` `cancel_staged_rename` | `successor.name_stage IN ['drafted','reviewed','exhausted']` and `predecessor.name_stage = 'superseded'` | **No.** `accepted` is absent from the successor set, so an accepted successor yields no rows. |
| `attachment_audit.py:245` `_DETACH_QUERY` | deletes `pn, hsn` for the named pair; the would-orphan refusal lives in `detach_one_attachment` (below) | **Yes, partially.** Guarded for the common case; two holes enumerated below. |
| `attachment_audit.py:407` `_TERMINAL_RECOVERY_QUERY` | `sn.name_stage IN $terminal_stages` = `{superseded, exhausted, contested}` | **No.** An accepted target cannot match. |
| `attachment_audit.py:578` `_TERMINAL_RECOVERY_BATCH_QUERY` | `name.name_stage IN $terminal_stages`, same set | **No.** Same predicate. |
| `edit.py:1486` fold mutation | `FOREACH (binding IN bindings \| DELETE binding)` then `CREATE (source)-[:PRODUCED_NAME]->(target)`, element-id fenced | **Yes for the *fold source* identity.** The edge is re-created on the target, so the target is supplied; the identity that loses is the folded predecessor, whose stage is not part of the fence. |
| `graph_ops.py:10745` `_lock_claimed_name_bindings` | provisional reservation delete, and delete of an owned orphan while still `pending` | **No.** Neither branch matches an accepted name. |
| `graph_ops.py:12595` `reconcile_source_status_liveness` | live-target test `NOT (coalesce(live.name_stage,'') IN ['superseded','exhausted'])…` fed `terminal_stages = sorted(_TERMINAL_BINDING_NAME_STAGES)` | **No — confirmed.** See the dedicated section below. |
| `graph_ops.py:13925` `reconcile_descriptionless_composed_names` | `name_stage IN ['','pending','drafted']` | **No.** Accepted is outside the set. |
| `graph_ops.py:20184` `deduplicate_scalar_selected_sources` | `source.produced_sn_id = expected.keep_target_id` and the keep target must exist | **Yes for the dropped target.** The fence is on the source's keep pointer, so the name that loses the source has its stage unchecked. |
| `graph_ops.py:22569` `_detach_stale_ancestor_sources` | `source.status = 'stale'` | **Yes.** The fence is on the *source's* status; the stage of the name it was the last producer of is never read. |
| `graph_ops.py:22852` `supersede_into_ancestor` | `COUNT { PRODUCED_NAME } = 1`… actually `= 2`, and `produced_sn_id = $ancestor_id` | **No.** A producer set of exactly two loses one and keeps one. |
| `signed_manifest.py:4580` unbind | `source.produced_sn_id = $row.scalar_target` | **Yes.** Element/scalar compare-and-set; the target's stage is not in the predicate. |
| `signed_manifest.py:4624` detach | `elementId(relationship) = $relationship_id AND elementId(start) = $start_id AND elementId(end) = $end_id` | **Yes.** Identity, not stage. |
| `signed_manifest.py:5258` `_apply_dual_authority_retirement` | `DELETE binding, projection` by element id, keep-target guards | **Yes.** Same shape: identity fence, no stage predicate on the retired target. |
| `signed_manifest.py:5630` `_apply_catalog_source_dispositions` | `DELETE binding, projection` by element id, keep-target guards | **Yes.** Same shape. |
| `signed_manifest.py:6117` `_apply_ineligible_source_retirement` | `DELETE binding, projection` by element id, keep-target guards | **Yes.** Same shape. |

The signed-manifest family is the largest single class (five routes sharing one
predicate shape) and every member is fenced on identity rather than stage. That
is by construction — a signed manifest is an instruction to perform an exact
edit — and it means a manifest whose target has only one producer will orphan
that name, silently and by design, if the manifest author did not check. The
guard against it must therefore live at manifest *construction*; the apply path
does not have one.

### The 5 node-deletion routes

These are divided by a distinction the first draft of this audit got wrong: a
node-deletion route removes a producer edge only when the deleted node is a
**source**.

| Route | Deleted node | Can it orphan an accepted name? |
| --- | --- | --- |
| `provenance_lifecycle.py:1808` `retire_unrecoverable_provenance_orphans` | `StandardName` that already has no producer | **No, and this is the only route that names the accepted case explicitly.** `include_accepted: bool = False` — the accepted half requires a deliberate opt-in, so the default cannot do it. |
| `graph_ops.py:3791` `_delete_derived_parent_nodes` | `derived_sources` | **Yes.** `FOREACH (source IN derived_sources DETACH DELETE source)` deletes the *source*, taking every PRODUCED_NAME edge with it. The stages of the names it produces are never read. This is a live candidate mechanism for the nine, and it is not a mechanism the census accounting covered. |
| `graph_ops.py:9031` `clear_standard_names` | `StandardName` | **No.** The name is gone; nothing accepted remains to be orphaned. |
| `graph_ops.py:13312` `reconcile_provenance` | derived `StandardNameSource` that has no `PRODUCED_NAME` edge | **No.** Deleting a source that produces nothing removes no producer edge. |
| `signed_manifest.py:4738` `_apply_mutation` delete target | `MATCH (target) WHERE elementId(target) = $element_id AND NOT (target)--() DELETE target` | **No.** `NOT (target)--()` is an explicit no-relationships precondition, so the node cannot be on a PRODUCED_NAME edge. |

`_delete_derived_parent_nodes` is the one node-deletion route that both
detaches edges and reads nothing about the names on the far end. It belongs in
the same class as the identity-fenced edge routes: capable, unguarded.

## `detach_one_attachment` — the refusal and the bypass

The would-orphan refusal, verbatim from the preflight consumer:

```python
structural_parent = bool(row.get("structural_parent"))
if int(row["name_attachments"]) <= 1 and not structural_parent:
    return {
        "ok": False,
        "reason": (
            "<sn_id> has only this one attachment — a name rejected by "
            "its whole source set is a NAME defect; repair it with "
            "sn edit --rename rather than orphaning it"
        ),
    }
```

**Classification for an accepted target: GUARDED in the common case, with two
holes read directly from the code.**

The preflight row supplies the counter as
`COUNT { (:IMASNode)-[:HAS_STANDARD_NAME]->(sn) }` — that is a count of
**realizations**, not of `PRODUCED_NAME` edges. The code states the choice is
deliberate ("the would-orphan guard counts REALIZATIONS (what a consumer sees),
not provenance rows"), and it is the first hole: a name with two realizations
has two `HAS_STANDARD_NAME` edges and one producer, so the guard passes and
deletes the only producer. The name keeps its projections and loses its
provenance.

The second hole is the `structural_parent` bypass. `structural_parent` is true
when the name is a derived parent with at least one live child:

```
sn.origin = 'derived' AND EXISTS {
  (child:StandardName)-[:HAS_PARENT]->(sn)
  WHERE NOT coalesce(child.name_stage,'') IN $historical }
```

An accepted derived parent with exactly one realization may therefore lose it.
Both accepted orphans in the live census that carry children — `beta`
(origin `derived`, `live_child_count` 3) and `ion_pressure` (origin `derived`,
`live_child_count` 2) — sit precisely in this bypass, which makes it a live
candidate for those two of the nine rather than a theoretical hole.

### The `terminal_recovery` bypass

**Classification for an accepted target: GUARDED.** `terminal_recovery=True`
does not widen the refusal; it delegates to `recover_terminal_attachment`
(`recover_terminal_attachment` is reached at `attachment_audit.py:2408-2415`),
whose mutation is gated by:

```
sn.name_stage IN $terminal_stages        # {superseded, exhausted, contested}
AND src.produced_sn_id = sn.id
AND COUNT { (src)-[:PRODUCED_NAME]->(:StandardName) } = 1
```

An accepted target yields zero rows, so the call returns
`{"ok": False, "reason": "terminal source binding changed or is ineligible"}`
rather than deleting. The bypass is on the *source-side* detection path only;
the mutation's own stage predicate is the fence and it excludes accepted. The
batch variant carries the same `name.name_stage IN $terminal_stages`.

## `reconcile_source_status_liveness` — confirmed to exclude accepted

`_TERMINAL_BINDING_NAME_STAGES` (`graph_ops.py:62-67`) is exactly:

```python
frozenset({NameStage.superseded.value, NameStage.exhausted.value,
           NameStage.contested.value})
```

The cleanup's live-target test spans `graph_ops.py:12535-12537`:

```
NOT (coalesce(live.name_stage,'') IN $terminal_stages)
```

and the call site supplies `terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES)`
at `graph_ops.py:12608`. Accepted is not in that set, so an edge to an accepted
name is a live binding and is not in the deletion predicate. The deletion at
`graph_ops.py:12595` (`FOREACH (edge IN stale_edges | DELETE edge)`) therefore
**cannot** be reached for an accepted target.

Note the contrast with `_SEMANTIC_SOURCE_REPAIR_MUTATION`, which excludes only
`{superseded, exhausted}`. Two predicates that both describe "the name is no
longer live" disagree about `contested`, and the broader one is reachable for
accepted. Whatever the intended set, the code has two of them.

## Can a rename still orphan today? Yes

`persist_refined_name` refuses an empty authoritative cohort, with an `edit_mode`
exemption (`graph_ops.py:18483-18494`):

```python
if (authoritative_cohort_observed and not candidate_source_ids and not edit_mode):
    raise RefinedNameStagePersistenceRefusal(...)
```

The carry then passes `_allow_empty_noop=(not authoritative_cohort_observed or bool(edit_mode))`
(`graph_ops.py:18716-18733`) into `retarget_standard_name_sources`, whose
migration deletes `prior:PRODUCED_NAME` to the old identity and MERGEs it on the
new. The post-flight check `if moved != len(candidate_source_ids)` compares
against the same empty list, so an empty cohort moves zero sources and the
check passes.

The caller guard in `edit.py:1890-1954` is not a substitute. Verbatim, its first
act is an early return:

```python
if not expected_source_ids:
    return f"no producing source bound to {old_name!r} — nothing to carry"
```

The rest of the function — the `stranded` pre-flight and the `missing`
post-flight `RuntimeError` — is reachable only when the cohort is non-empty. So
an observed-but-empty cohort produces a vacuous success: the rename reports
"nothing to carry" as though it were an ordinary no-op, and the predecessor is
superseded (`graph_ops.py:18588`) with no producer left behind.

**Answer: a rename can still orphan today.** The path is a rename whose old
identity has an empty *observed* source cohort under `edit_mode`. The
predecessor becomes `superseded` — which is terminal and unremarkable — but the
successor identity that later reaches `accepted` is the one that carries no
producer. That is the shape of the five rename-receipt cohort rows recorded in
the plan comment.

## Live graph count against the bound

`python3 live_graph_census.py` — log `live-graph-census.log`, EXIT=0:

| Measure | Value |
| --- | --- |
| accepted names without a producer | **9** |
| bound recorded 2026-09-15 | 9 |
| difference | **0** |
| accepted names total | 2,528 |
| accepted with a producer | 2,519 |
| StandardName nodes / with `id` | 5,130 / 5,130 |
| **positive control** `PRODUCED_NAME` edges | 5,493 |
| **positive control** distinct producing sources | 5,393 |

The positive control matters here and is why it is reported: an orphan count of
9 is a number the instrument must be shown to be capable of exceeding, and the
control demonstrates the same session reads live relationships (5,493
`PRODUCED_NAME` edges across 5,393 distinct sources) and keys (5,130 of 5,130
nodes carry `id`). The count is unchanged from the bound, so **no new orphan has
appeared since 2026-09-15** and no explanation of a change is owed.

### The nine, with the two that fit a route named above

All nine are `status: draft`. Two carry children and are `origin: derived` —
`beta` (3 live children) and `ion_pressure` (2 live children, also
`validation_status: quarantined`). Those two fit the `structural_parent` bypass
in `detach_one_attachment`. `neutron_flux_due_to_fusion` carries
`docs_stage: exhausted`, which fits the `docs_stage` half of
`retire_unrecoverable_provenance_orphans` rather than the `name_stage` half.
The remaining six (all `origin: pipeline`, no children) are consistent with the
rename path or with `_delete_derived_parent_nodes`, and this audit does not
claim to have assigned each of the nine a mechanism — the per-name trace is the
adjacent node's evidence.

## Logs

| Log | Exit |
| --- | --- |
| `route-census.log` | 0 |
| `live-graph-census.log` | 0 |

Both contain the script that produced them. No graph mutation was performed in
either.