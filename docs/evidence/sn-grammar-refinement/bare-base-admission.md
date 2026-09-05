# How a bare single-token base becomes accepted

**Node goal:** establish how any accepted bare single-token base got past the
name-review semantic gate, and answer whether an existing sanctioned route would
carry `beta` to `accepted` or whether it would be the first.

**Method:** read-only inspection of the live graph (Cypher over `StandardName`,
`StandardNameReview`, `StandardNameChange`, `StandardNameSource`,
`StructuralNameAuthority`) plus a no-write recomputation of the gate's cosine
for each accepted bare base, using the exact code path `review_name` uses
(`semantic_similarity_check` in `imas_codex/standard_names/audits.py:3684`, the
same cosine computed by `desc_name_sim.compute_desc_name_similarity`). No
`sn` command was run in any mode; no graph value was written; no repository
file other than this report was changed.

---

## 1. The count and the list

Live accepted identities whose `id` is a bare single token (no underscore,
`name_stage='accepted'` , not superseded/exhausted): **2 of 14 bare-token
nodes exist in total, and both are accepted and live.**

| id | name_stage | docs_stage | origin | reviewer_score_name | reviewer_model_name | semantic_sim (stored) |
|---|---|---|---|---|---|---|
| `momentum`  | accepted | accepted | `derived` | None | `structural-inheritance` | None |
| `vorticity` | accepted | accepted | `catalog_edit` | 0.8625 | `openrouter/x-ai/grok-4.5` | 0.6986 |

The full population of 14 bare-token nodes, by `name_stage`:

- **accepted (2):** `momentum`, `vorticity`
- superseded (6): `capacitance`, `conductivity`, `duration`, `etendue`,
  `probability`, `width`
- reviewed (2): `beta` (0.30), `radius` (0.4875) — both parked, neither accepted
- drafted (2): `acceleration`, `strain`
- pending (1): `outline` (derived, in flight)
- exhausted (1): `wavelength` (0.425, quarantined)

No bare-token node carries `catalog_approved_at`, `catalog_pr_number`, or
`catalog_merge_commit_sha`, so no bare base was ever admitted through the
catalog-approve route.

## 2. The two accepted identities reached `accepted` by two different routes

### `momentum` — the structural-inheritance route (never faced the gate)

Evidence on the node:

- `name_stage='accepted'`, `validation_status='valid'`, `origin='derived'`.
- **No name-axis review.** `reviewer_score_name=None`, `reviewer_scores_name=None`,
  `review_resolution_method=None`, `semantic_sim=None`, and there is no
  `momentum:names:*` `StandardNameReview` record — only two `momentum:docs:*`
  groups (2026-08-11 11:04, escalated 0.4375/0.5375/0.625; 2026-08-11 14:25,
  0.8125/0.925 quorum_consensus).
- `reviewer_model_name='structural-inheritance'`,
  `reviewed_name_at=2026-08-11T06:42:43Z` — the stamp written by
  `structural_accept_derived_parents()` (`graph_ops.py:25155-25214`).
- A `HAS_STRUCTURAL_AUTHORITY` edge to a `StructuralNameAuthority` node, and a
  produced name edge from `derived:momentum` (`StandardNameSource`,
  `source_type='derived'`, `status='composed'`).
- `kind='vector'`, cycled by `refresh_derived_kind` writes (2026-09-02) but the
  acceptance itself is 2026-08-11.
- Two children via `HAS_PARENT`: `plasma_momentum` (accepted, qualifier),
  `radial_momentum` (reviewed, projection).

Mechanism (read from the live code, `graph_ops.py:3987-4023` and
`25155-25214`): a derived family anchor's name is treated as a deterministic
grammar peel of its accepted children, so it is **never name-reviewed**. Either
`seed_parent_sources` materialisation writes `name_stage='accepted'` directly
for a placeholder parent, or `structural_accept_derived_parents()` promotes a
derived parent with a real description from `drafted/reviewed/exhausted` to
`accepted`. Both run as global maintenance at `sn run` startup
(`loop.py:1993-2014`). The semantic gate is never consulted.

**Answer to the per-node question:** it did not pass the semantic gate on
merit (no stored score, no review ever ran); it took the structural
(deterministic-parent) skip; it does not predate the gate (accepted
2026-08-11, gate landed 2026-05-08); the route is "other" — direct structural
acceptance.

### `vorticity` — the merit route (passed the semantic gate)

Evidence on the node:

- `name_stage='accepted'`, `validation_status='valid'`, `origin='catalog_edit'`,
  `source_types=['catalog']`.
- `reviewer_score_name=0.8625`,
  `reviewer_scores_name={"grammar":20,"semantic":15,"convention":17,"completeness":17}`
  (i.e. 20+15+17+17 = 69/80 = 0.8625), `review_resolution_method=
  'authoritative_escalation'`, `semantic_sim=0.6986`.
- A full name-axis quorum on 2026-08-23 13:49:06Z:
  cycle-0 primary `openrouter/x-ai/grok-4.5` 0.825, cycle-1 secondary
  `openrouter/openai/gpt-5.6-luna` 0.975, cycle-2 escalator
  `openrouter/anthropic/claude-sonnet-5` 0.8625 — three real reviewer models,
  no synthetic stamp.
- `reviewer_model_name='openrouter/x-ai/grok-4.5'`, no
  `HAS_STRUCTURAL_AUTHORITY` edge. Produced by `dd:mhd/ggd/vorticity/values`.
- Six children via `HAS_PARENT`: parallel/poloidal/radial/toroidal/vertical
  `_vorticity` (accepted, projection) + `ratio_of_vorticity_to_major_radius`
  (reviewed, binary).

Mechanism: `review_name` ran the semantic similarity gate first; the stored
`semantic_sim=0.6986` cleared the 0.55 critical floor, so the gate did **not**
fire, the LLM quorum ran, and the name was accepted on the name axis.

**Answer to the per-node question:** it passed the semantic gate on merit
(0.6986 ≥ 0.55, real quorum, authoritative escalation); it did not take the
deterministic-parent skip (`origin='catalog_edit'`, not derived); it does not
predate the gate (reviewed 2026-08-23, gate landed 2026-05-08).

## 3. Would the accepted bare bases pass their own gate today?

Recomputed read-only through the exact gate code path
(`audits.semantic_similarity_check` = same cosine as
`desc_name_sim.compute_desc_name_similarity`, same embedding server, same
500-char truncation) on the current stored descriptions:

| id | stored semantic_sim (at review) | today's recompute | verdict today at 0.55 |
|---|---|---|---|
| `momentum`  | None (never gate-computed) | **0.5425** | **below critical — would fail the semantic gate** |
| `vorticity` | 0.6986 | **0.7142** | above critical and above the 0.65 warning — passes |

The embedding server reached for both computations:
`http://98dci4-gpu-0002:18765` → `98dci4-gpu-0002.iter.org` (titan, worker_gpu
0), model `Qwen/Qwen3-Embedding-0.6B`, `/health` healthy at measurement time —
so these are real scores, not an outage artifact.

The materially different finding the fence called out is confirmed: **one of
the two accepted bare bases, `momentum`, would fail its own gate if it were
rescored today** (0.5425 < 0.55). It is accepted because it is a derived
structural anchor, not because its name stands alone — exactly what the
structural skip is for, and exactly the property the proposed family-anchor
exemption would extend to non-derived anchors like `beta`.

## 4. The gate has met this case before — this is not a case the gate has never seen

Of the 14 bare-token nodes, 10 have a stored name-axis verdict of one kind or
another:

| id | semantic_sim | reviewer_score | verdict |
|---|---|---|---|
| `vorticity` | 0.699 | 0.8625 | real quorum, **accepted** |
| `capacitance` | 0.725 | 0.575 | real quorum, later superseded |
| `conductivity` | 0.717 | 0.5875 | real quorum, later superseded |
| `duration` | 0.580 | 0.550 | real quorum, superseded |
| `probability` | 0.602 | 0.5375 | real quorum, superseded |
| `radius` | (cleared gate) | 0.4875 | real quorum, parked at reviewed |
| `wavelength` | (cleared gate) | 0.4250 | real quorum, exhausted/quarantined |
| `etendue` | 0.432 | **0.30** | gate fired (`(semantic_similarity_gate)`) |
| `width` | 0.481 | **0.30** | gate fired (`(semantic_similarity_gate)`) |
| `beta` | 0.475–0.508 | **0.30** | gate fired (`(semantic_similarity_gate)`), parked at reviewed |

So a bare single token **can** clear the gate on merit — `vorticity` is the
live proof, with `capacitance`/`conductivity`/`duration`/`probability` as
superseded co-examples — and the gate **does** fire on single tokens when the
description does not carry the term (`etendue` 0.432, `width` 0.481, `beta`
0.49). The gate is calibrated against the single-token category; it is not
silent about it. What has never happened is a non-derived bare anchor with
accepted-status quality being parked at `reviewed` by the gate with no road on,
which is the `beta` situation.

## 5. IS THERE AN EXISTING SANCTIONED ROUTE? — yes, two; beta would not be the first bare base accepted, but neither route is open to beta as it stands

**Route A — merit through the semantic gate (what `vorticity` took).**
A bare token becomes accepted by passing the gate (description embedding ≥
0.55) and then the LLM quorum. This is the normal review pipeline and is the
same rescore route the earlier node named for the exhausted→accepted move.
Command sequence a follow-on node can run:

```
sn rescore beta
```

`sn rescore` restages the name to `drafted` and runs the review pipeline
scoped to it (`edit.rescore_name`, `skip_generate=True` — it reviews the
**stored** description). For a bare base this carries to `accepted` **iff** the
stored description scores ≥ 0.55. For `beta` the four measured candidate
descriptions all scored 0.475–0.508, and no sanctioned edit can install a
better one while the name is parked at `reviewed` (docs edits require
`name_stage='accepted'`, `edit.py:2751-2759`; name-axis edits are blocked for
non-accepted names). So Route A exists and is proven to work for a bare base —
it is simply closed to `beta` until the description question is settled.

**Route B — structural inheritance for derived anchors (what `momentum` took).**
A bare token that is a derived family anchor (`origin='derived'`, seedable
`HAS_PARENT` children) is accepted directly by the `sn run` startup fixpoints
(`rederive_structural_edges` → `seed_parent_sources` →
`normalize_derived_parent_lifecycle` → `structural_accept_derived_parents`,
`loop.py:1993-2014`), with the docs axis carrying the quality gate. There is no
gate deliberation and no name-axis review. Command sequence:

```
sn run        # global maintenance: the deterministic fixpoints write
              # name_stage='accepted' for eligible derived anchors
```

Route B is closed to `beta`: it requires `origin='derived'`, and beta's origin
is null (it is a DD-sourced physical base, `source_types=['dd']`).

**Answer.** Beta would not be the first accepted bare base — two already exist,
one per route. But beta is the **first bare base parked at `reviewed` by the
gate with neither route available**: its current description scores 0.49 (below
0.55), no sanctioned edit can change the description at that stage, and it is
not a derived anchor. Both existing routes admit bare bases; neither currently
carries *this* bare base. So an exemption is still needed for beta's
combination, but the system already contains the mechanism the exemption would
mirror — structural acceptance of family anchors on the strength of their
children — and a second mechanism (the gate itself) that has already admitted a
single token on description merit.

## 6. Sizing the proposed exemption

The proposed shape (family-anchor skip keyed on a bare base carrying
`HAS_PARENT` children) is already how Route B operates for derived anchors. If
extended to non-derived bare bases, its live population among the 14 bare-token
nodes is:

| id | children | name_stage | currently |
|---|---|---|---|
| `vorticity` | 6 | accepted | already in via the gate (merit) |
| `momentum` | 2 | accepted | already in via structural inheritance |
| `outline` | 2 | pending | derived placeholder, in flight in Route B |
| `beta` | 3 | reviewed | **the only live bare anchor the gate has parked** |
| `wavelength` | 6 | exhausted | retired — not live |
| `conductivity` | 5 | superseded | retired — not live |

So the exemption's practical effect is small and bounded: it would change
exactly one live verdict (beta parked→eligible), would not disturb either
already-accepted bare anchor, and would not sweep in a cohort the gate has been
rejecting (the remaining parked bare tokens — `radius`, `acceleration`,
`strain` — carry no children). A side effect worth stating: an exemption of
this shape would also admit a name like `momentum` (accepts on children,
0.5425 gate cosine today) — but momentum is already accepted via Route B, so
the exemption only aligns the two mechanisms rather than creating a new class
of accepted name.

## 7. Follow-ons (fenced out of this node)

- **A gate-clearing description exercise for beta.** `vorticity`'s stored 0.699
  (0.714 today) shows a single token can clear 0.55 when the description
  animates the term (definition-first, term-bearing prose). Before deciding an
  exemption is the only road, a read-only sweep of beta-candidate descriptions
  (definition-first forms, the total-beta relation in prose) measured at the
  same server would establish whether the merit route is merely underexplored
  for beta's description rather than categorically unavailable. The four
  measured candidates (0.475–0.508) do not settle this.
- **Exemption placement, if the lead chooses it.** Extend the semantic-gate
  skip to bare anchors (`NOT id CONTAINS '_'` with ≥1 seedable `HAS_PARENT`
  child) mirroring the derived skip at `workers.py:8197-8206`, gated on
  `SEMANTIC_SIM_CRITICAL` being the only blocked path (i.e. do not skip the
  LLM chain — the anchor still needs the docs-axis quality gate).

## 8. Test suite note

No repository test suite was run: this node changed no code (report-only
scope, a single evidence document), so there is no suite that reaches the
change. `baseline_suite` and `after_suite` are recorded as absent evidence
(`completed=false`) rather than as a pass.
