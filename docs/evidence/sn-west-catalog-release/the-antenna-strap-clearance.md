# The antenna-strap clearance: survivor accepted, collision reported

## Outcome

The antenna-strap clearance quantity reached one accepted standard name. The
exhausted identity `distance_of_antenna_strap` was renamed through the
sanctioned `sn edit` route to `back_surface_distance_of_antenna_strap`,
reviewed to **acceptance** under a scoped run (`--skip-global-maintenance`),
and the producing source survived the rename (1 `PRODUCED_NAME` edge, no
orphan). The competing spelling `gap_of_antenna_strap` denotes the same
quantity (same DD leaf, unit, domain, object) and could not be folded into the
survivor as a duplicate: the sanctioned `sn supersede` route refused because
`gap_of_antenna_strap` is not a free identity — it already carries an incoming
`REFINED_FROM` lineage from `normal_gap_at_wall` that the fold guard refuses to
strand. The collision is therefore reported with both competing identities, as
the goal's alternative branch allows.

| Identity | Role | Result | WEST cut |
| --- | --- | --- | --- |
| `distance_of_antenna_strap` | starting exhausted identity | renamed → `back_surface_distance_of_antenna_strap` (accepted) | — |
| `back_surface_distance_of_antenna_strap` | **survivor** | name `accepted` at `0.8875` (quorum_consensus), docs `accepted` at `0.9875`, validation `valid`, 1 producer | **Yes** |
| `gap_of_antenna_strap` | competing identity (same quantity) | remains `exhausted` at `0.6625`; fold refused by the lineage guard | No |

No identity was deleted, no producer orphaned, no quarantine was cleared or
written, and no global maintenance was run. The graph was read and written on
the login node through the project client (`bolt://localhost:17687` tunnel).

## Starting states (both identities)

Read from the live graph before any action. There is no `name` property on
`StandardName` — the identity key is `id`; producer paths live on
`StandardNameSource` reached by `(:StandardNameSource)-[:PRODUCED_NAME]->(sn)`.

### `distance_of_antenna_strap`

- `name_stage=exhausted`, `reviewer_score_name=0.7125`, `validation_status=valid`, `validation_issues=[]`.
- `refine_attempts=3/3`, `refine_stop_reason=successor_collision`, `refine_collision_name=gap_of_antenna_strap`.
- Unit `m`, `physics_domain=auxiliary_heating`, `object=antenna_strap`, `origin=pipeline`.
- Producing source (**1 edge**): `dd:ic_antennas/antenna/module/strap/distance_to_conductor`, `source_type=dd`, raw documentation *"Distance to conducting wall or other conductor behind the antenna strap"*.

### `gap_of_antenna_strap`

- `name_stage=exhausted`, `reviewer_score_name=0.6625`, `validation_status=valid`.
- `refine_attempts=3/3`, `refine_stop_reason=attempts_exhausted`, `superseded_from_stage=refining`.
- `source_types=["catalog"]`, **zero `PRODUCED_NAME` producers** (a catalog import).
- Its accepted documentation states it *"denotes the same rear-surface-to-back-wall clearance as `back_surface_gap_of_antenna_strap`"* and excludes `thickness_of_antenna_strap`, `inner_radius_of_antenna_strap`, `outer_radius_of_antenna_strap`.

## Same-quantity determination

Both identities describe the strap-rear-surface-to-back-conducting-wall
clearance:

- Same bound DD leaf `ic_antennas/antenna/module/strap/distance_to_conductor` (same unit `m`, same domain `auxiliary_heating`, same object `antenna_strap`).
- `gap_of_antenna_strap`'s own accepted docs declare the identity with
  `back_surface_gap_of_antenna_strap` and describe an identical minimum
  perpendicular rear-surface-to-back-wall separation.

Conclusion: **same physical quantity** → only one identity should survive. Kept:
`back_surface_distance_of_antenna_strap`, because it is the only one carrying
the authoritative producing-source binding (the WEST DD path) and it reached
accepted name and docs. `gap_of_antenna_strap` is the duplicate spelling.

## Rename dry-run and apply

Dry-run (quoted verbatim):

```text
DRY RUN sn edit: distance_of_antenna_strap  mode=rename axis=name
scope=only_self entry=review_name

Actions:
  -  would carry 1 producing source(s) to
'back_surface_distance_of_antenna_strap'
  -  would rename 'distance_of_antenna_strap' →
'back_surface_distance_of_antenna_strap'
```

Scope was `self`/`only_self` (leaf rename; the dry run showed exactly one
identity and one producing source, no shared segment to cascade). The reason
was written from the DD path and physics, not preference: the DD leaf defines
the distance from an ICRH antenna strap to the conducting wall or conductor
behind it, and reviewers had capped the bare base `distance` for not stating
which distance of the strap is meant; the `back_surface` zone names the strap
rear face whose perpendicular clearance to the conducting wall behind it is
this quantity, matching the accepted sibling family
(`inner_radius_of_antenna_strap`, `outer_radius_of_antenna_strap`).

Apply receipt (quoted verbatim):

```text
APPLIED sn edit: distance_of_antenna_strap  mode=rename axis=name
scope=only_self entry=review_name

Actions:
  - verified 1 producing source(s) on 'back_surface_distance_of_antenna_strap'
  - renamed 'distance_of_antenna_strap' →
'back_surface_distance_of_antenna_strap', entering name review
(edit_status=open, run_id=sn-edit-20260915T070501Z)

  successor: back_surface_distance_of_antenna_strap
```

## Scoped review receipts

Run: `sn run --only review --name back_surface_distance_of_antenna_strap --skip-global-maintenance -c 3`. Only the named successor was claimed; the run drained name review, docs generation, and docs review and exited `no_eligible_work`.

```text
persist_reviewed_name: back_surface_distance_of_antenna_strap → name_stage=accepted (score=0.8875, rotations=3/3, chain=2)
review_name: back_surface_distance_of_antenna_strap → accepted (score=0.8875, cycles=2, method=quorum_consensus)
persist_generated_docs: back_surface_distance_of_antenna_strap → docs_stage=drafted
generate_docs: back_surface_distance_of_antenna_strap — Geometric rear-surface clearance between an ion-cyclotron heating antenna strap and the conducting w…
persist_reviewed_docs: back_surface_distance_of_antenna_strap → docs_stage=accepted (score=0.9875, chain=0/3, resolution=quorum_consensus, shortfall=None)
review_docs: back_surface_distance_of_antenna_strap → accepted (score=0.9875, cycles=2, method=quorum_consensus)
```

Run summary: `cost_spent=0.191132`, `cost_limit=3.0`, `names_composed=0`,
`names_enriched=1`, `names_reviewed=2`, elapsed ~251 s.

## Per-reviewer scores and aggregate

Name axis (aggregate `0.8875`, `quorum_consensus`, dimensions
`grammar 20 / semantic 15 / convention 19.5 / completeness 16.5`):

| Reviewer model | score |
| --- | ---: |
| `openrouter/openai/gpt-5.6-luna` | `0.875` |
| `openrouter/x-ai/grok-4.5` | `0.900` |

Docs axis (aggregate `0.9875`, `quorum_consensus`):

| Reviewer model | score |
| --- | ---: |
| `openrouter/x-ai/grok-4.5` | `1.000` |
| `openrouter/anthropic/claude-sonnet-5` | `0.975` |

## Final read-back (survivor)

```text
id: back_surface_distance_of_antenna_strap
name_stage: accepted
reviewer_score_name: 0.8875
model: sn-edit
docs_stage: accepted
reviewer_score_docs: 0.9875
validation_status: valid
validation_issues: []
review_resolution_method: quorum_consensus
review_count: 4
```

Producer binding (read back after the rename — **no orphan**, the source
survived as required by the dispatch):

```text
dd:ic_antennas/antenna/module/strap/distance_to_conductor  source_type=dd  PRODUCED_NAME → back_surface_distance_of_antenna_strap
```

The old spelling `distance_of_antenna_strap` retains the rename lineage edge
(`back_surface_distance_of_antenna_strap-[:REFINED_FROM]->distance_of_antenna_strap`)
and was not deleted.

## Collision attempt and refusal (gap_of_antenna_strap)

Attempted the sanctioned fold of the duplicate into the accepted survivor:

```text
imas-codex sn supersede gap_of_antenna_strap --into back_surface_distance_of_antenna_strap --dry-run
Error: name 'gap_of_antenna_strap' has another successor lineage; fold is ambiguous
```

`exhausted` is an eligible predecessor stage for the fold; the refusal is the
fold-guard's lineage check, not a stage error. The graph shows an incoming
`normal_gap_at_wall-[:REFINED_FROM]->gap_of_antenna_strap` edge, and the change
ledger on `normal_gap_at_wall` records a `source_migration_manifest` and a
`refine` from `gap_of_antenna_strap` (2026-08-23). `gap_of_antenna_strap` is
therefore not a free identity: it is itself the target of a previous fold/refine
that `normal_gap_at_wall` reads as its lineage source, so folding `gap` away
would strand that lineage. The sanctioned route refuses, and per the guard
contract the fashion is to report the collision with both competing identities
rather than hand-edit the lineage.

Terminal state of the duplicate:

```text
id: gap_of_antenna_strap
name_stage: exhausted
reviewer_score_name: 0.6625
validation_status: valid
superseded_from_stage: refining
source_types: ["catalog"]        # zero PRODUCED_NAME producers
```

## WEST cut verdict

The producing source `dd:ic_antennas/antenna/module/strap/distance_to_conductor`
is a member of the committed WEST production batch
(`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`,
`ic_antennas → antenna/module/strap/distance_to_conductor`). The survivor that
carries it now satisfies the carry gates — produced by a WEST DD path, with
**accepted** name (`0.8875`), **accepted** docs (`0.9875`), and **valid**
validation — so **the WEST cut would now carry the antenna-strap clearance
quantity** under the current name/docs/validation gates.

## Spend

Campaign before this node: `107.18` USD against the `250.00` USD ceiling.
Node delta: `$0.191132` (the scoped review run; the rename itself performs no
LLM work). Campaign after: ≈ `107.37` USD. Well within the node budget of
`12.00` USD.
