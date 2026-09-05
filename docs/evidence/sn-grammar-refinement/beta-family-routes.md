# n-sgr-beta-is-one-quantity — blocked before execution

NEEDS-HELP: the beta family carries a REFINED_FROM cycle between `beta` and `normalized_toroidal_plasma_beta` that blocks every sanctioned single-step route for both owed renames; no code or graph write was made.

## tried

All three owed transitions were dry-run or probed against live graph state (read-only throughout; zero writes; zero LLM spend). No changes were applied.

**Measured state of the family today** (`id`, `name_stage`, `docs_stage`, parent/children via `HAS_PARENT`, `REFINED_FROM` lineage):

| id | name_stage | docs_stage | parent | produced_from (PRODUCED_NAME sources) |
|---|---|---|---|---|
| `beta` | exhausted | pending | — (anchor/parent of poloidal_beta, toroidal_beta) | none |
| `plasma_beta` | accepted | accepted | `beta` | `dd:summary/global_quantities/beta_tor/value`, `derived:plasma_beta` |
| `normalized_toroidal_plasma_beta` | accepted | accepted | `plasma_beta` | 9 DD sources (beta_tor_norm family across equilibrium/summary/core_profiles/plasma_profiles) |
| `poloidal_beta` | accepted | accepted | `beta` | 6 DD sources — **untouched, correct, not in scope** |
| `toroidal_beta` | accepted | pending | `beta` | 2 DD sources — **untouched, correct, not in scope** |
| `reference_beta` | exhausted | pending | — | `dd:gyrokinetics/species_all/beta_reference` — **not in scope, not touched** |
| `normalized_toroidal_beta` | superseded | pending | — (no children) | none (free of sources) |

`beta`'s current description: *"Toroidal beta (β_tor), the ratio of volume-averaged total perpendicular plasma pressure to the magnetic pressure B0²/(2μ0)."* — this is the wrong-quantity description the lead flagged; it already has its own accepted home at `toroidal_beta`.

**The REFINED_FROM lineage graph** (edges are `X -[:REFINED_FROM]-> Y`, meaning X was refined from prior spelling Y):

```
beta -------------------------> normalized_toroidal_plasma_beta   (!)
normalized_toroidal_plasma_beta -> beta                            (!)  <-- 2-CYCLE with the edge above
normalized_toroidal_plasma_beta -> normalized_beta
normalized_toroidal_plasma_beta -> normalized_toroidal_beta
toroidal_plasma_beta (superseded) -> beta
plasma_beta -> toroidal_plasma_beta
normalized_toroidal_beta -> normalized_beta
normalized_toroidal_beta -> normalized_toroidal_thermal_plasma_beta
normalized_toroidal_beta -> toroidal_normalized_plasma_beta
```

`beta` and `normalized_toroidal_plasma_beta` hold **mutual REFINED_FROM edges to each other** — a genuine 2-cycle. `beta`'s wrong description (describing toroidal beta) is consistent with this: at some point in prior grammar-refine history, the `beta` identity was populated as part of an attempt that actually belonged to the toroidal/normalized-toroidal branch of the family, then got tangled back onto `normalized_toroidal_plasma_beta` as an apparent successor.

### Item 1 — `plasma_beta` → `beta` (drop the redundant `plasma_` qualifier)

- `sn edit plasma_beta --rename beta --reason "..." --dry-run`
  → **BLOCKED**: `a StandardName 'beta' already exists`. (`_apply_rename`'s collision check is unconditional on any existing id, regardless of that id's stage — see `imas_codex/standard_names/edit.py` around the `EDIT_CHECK_COLLISION` query.)
- `sn supersede plasma_beta --into beta --dry-run`
  → **REFUSED**: `target 'beta' is name_stage='exhausted', not 'accepted'`. `supersede_into`'s `--into` target must be `accepted` or a free tombstoned (`superseded`) identity; `exhausted` is neither.
- `sn supersede beta --into plasma_beta --dry-run` (the only remaining direction that could clear `beta`'s wrong content out of the way)
  → **REFUSED**: `name 'beta' has another successor lineage; fold is ambiguous`. Both `toroidal_plasma_beta` (superseded) and `normalized_toroidal_plasma_beta` (accepted) hold `REFINED_FROM` edges into `beta`; the fold guard requires the only descendant to be the fold target, and here there are two, neither of which is `plasma_beta`.
- `sn edit beta --docs "<total-beta description>" --reason "..." --dry-run` (checked in case the description alone could be fixed on the existing `beta` node while the rename question is separately resolved)
  → **BLOCKED**: `target name_stage='exhausted' — docs edits require an accepted name (name_stage='accepted')`.

No sanctioned single-step route exists to land `plasma_beta` on the spelling `beta`, and no sanctioned route exists to fix `beta`'s description in place either, while `beta` sits at `exhausted` and is entangled in this cycle.

### Item 2 — `normalized_toroidal_plasma_beta` → `normalized_toroidal_beta` (revival route)

Per the `sn-edit` skill, a `superseded` name with no successor is eligible for **redesign-edit revival**, and `sn supersede OLD --into TARGET` is the mechanism built for "folding a name into an existing canonical name, or re-pointing a name onto a restored tombstoned id" — exactly this case (`normalized_toroidal_beta` is `superseded`, `superseded_by` is null, and it carries no sources/parent/children, so it reads as a free identity in isolation).

- `sn supersede normalized_toroidal_plasma_beta --into normalized_toroidal_beta --dry-run`
  → **REFUSED**: `name 'normalized_toroidal_plasma_beta' has another successor lineage; fold is ambiguous`.

The cause is the same cycle: `beta -[:REFINED_FROM]-> normalized_toroidal_plasma_beta` is a third-party descendant edge into the fold's OLD node that has nothing to do with `normalized_toroidal_beta`, so the guard refuses the fold as ambiguous rather than silently picking a side.

**So establishing "what the sanctioned revival route is" resolves to: the route exists (`sn supersede --into`) and is the correct one in isolation, but it cannot execute until the `beta` ↔ `normalized_toroidal_plasma_beta` cycle is resolved first.** Per the fence's own instruction ("if none exists say so and stop rather than minting a colliding identity") and the general instruction to stop rather than force a collision, no edit or supersede call was executed.

### Item 3 — the `beta` description repair

Not attempted independently: `sn edit --docs` requires `name_stage='accepted'`, and `beta` is `exhausted` (see item 1's fourth probe). The description fix is downstream of resolving what `beta`'s final identity/stage should be — it cannot be applied to the exhausted node as it stands, and applying it after a rename would be redundant edit calls; it belongs in the same sequenced fix as item 1.

## options

1. **Have a graph-repair node break the `beta` ↔ `normalized_toroidal_plasma_beta` REFINED_FROM cycle first** (and clear `beta`'s duplicate incoming lineage from `toroidal_plasma_beta`), scoped as its own repair rather than folded into this node — this is the same class of pre-existing lineage entanglement the plan's memory already tracks (dedup-fold / derived-parent churn / supersede-successor-scalar issues). Once `beta` is a clean, non-cyclic, ideally `accepted`-or-free-superseded node, items 1 and 2 both become straightforward `sn supersede --into` calls, in this order: resolve `beta`'s stray lineage → `sn supersede plasma_beta --into beta` (or the reverse, whichever direction the repair leaves eligible) → `sn supersede normalized_toroidal_plasma_beta --into normalized_toroidal_beta` → `sn edit beta --docs "<total-beta text>" --reason "..."`.
2. **Ask the lead to adjudicate `beta`'s exhausted, cyclically-tangled state directly** — it is plausible this is exactly the kind of cross-cutting graph defect the lead would want surfaced rather than silently repaired by a node scoped to "consolidate spelling and fix one description."
3. **Do nothing further on this node and let a dedicated lineage-repair plan item pick it up** — least action, but leaves the beta family's two owed renames stalled indefinitely without a named successor task.

## leaning

Option 1, sequenced as its own repair step before re-attempting this node's two renames — the cycle is a two-edge, well-localized defect (`beta` and `normalized_toroidal_plasma_beta` mutually citing each other via `REFINED_FROM`, plus `toroidal_plasma_beta` also citing `beta`), not a systemic corruption, so it should be a small, auditable fix. But *which* edge is the erroneous one (was `beta` wrongly refined from `normalized_toroidal_plasma_beta`, or is it the reverse edge that is spurious, or both are stale relics of an earlier multi-hop rename that never got compacted) is a graph-history question this node has no tooling to answer safely — hand-editing `REFINED_FROM` edges via Cypher is exactly the kind of write this project's rules forbid without a sanctioned command, and no CLI surface for editing lineage edges directly was found in the time available. That determination should be made by whoever owns the lineage-integrity repair, with the cycle's exact shape (above) as their starting evidence.

## cost-if-wrong

If the cycle were force-broken in the wrong direction (i.e., picking the wrong edge to delete, or assuming `beta`'s content is the stale one when it is actually `normalized_toroidal_plasma_beta`'s ancestor chain that is stale), a subsequent `sn supersede` could silently point provenance at the wrong final identity, and the resulting "total beta" node could carry the wrong description lineage forward (e.g., permanently losing the trail back to `toroidal_normalized_plasma_beta` / `normalized_beta`, which the current `normalized_toroidal_beta` node's ancestor chain still remembers). Re-deriving the correct lineage after that would need the exact edge inventory captured in this report, redone from a Neo4j point-in-time or transaction log rather than reconstructible from graph state alone — hence stopping now rather than guessing.

## Evidence

- Live graph state, `REFINED_FROM` lineage, and all five dry-run probes: this report (queried via `imas_codex.graph.client.GraphClient`, read-only Cypher, no writes; `imas-codex sn edit` / `imas-codex sn supersede --dry-run`, no mutation).
- `LLMCost` node count in the graph at the time of this report: 36914 rows, $1777.49 cumulative total (unchanged before/after this node — no model calls were made; nothing to spend, nothing spent). Projected cost for this node's owed work, had the routes cleared: two `sn supersede` folds (structural, no LLM call) plus one `sn edit --docs` review pass on `beta` (the only step that rides the paid review pool) — so at most one review-pool call, not spent.
- No files were staged or committed in the worktree; `git status --porcelain` is empty throughout.
