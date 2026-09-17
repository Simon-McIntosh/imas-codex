# Derive boundary cases: the two populations and what each option costs them

## Outcome

The `links` derive beat has no longer an open question it can only answer by
choosing. Measured live on the `codex` graph at `2026-09-17T19:12Z`, read-only:

| # | Measure | Value |
|---|---|---:|
| — | Resolvable `links` entries | 6,913 |
| — | `REFERENCES` edges | 6,913 |
| — | Scalar entries, all | 7,036 |
| **1** | **Entries whose target resolves to no live node** | **965** |
| 1a | — target id matches no `StandardName` node at all | 123 |
| 1b | — target matches a node that is superseded | 842 |
| **2** | **Edges whose target is superseded** | **842** |
| **3** | Of those, successor chain reaches a live node | **663** (506 within one hop; 336 not, of which 179 not within four) | 
| **4** | Distinct `StandardName` rows affected | 470 source rows, 437 target rows, **966** in union |

Every query is recorded with its result, and every zero carries a positive
control that returned non-zero. The two populations are disjoint: 1a's target
ids are not nodes, so no row of 1b can be one of them.

## The instrument, and its control

![The two boundary populations and what each option changes](/imas-codex/figures/scalar-edge-authority/derive-boundary-cases.svg)

```cypher
MATCH (s:StandardName)
WHERE s.links IS NOT NULL
UNWIND s.links AS l
WITH s, l WHERE l STARTS WITH 'name:'
WITH s, substring(l, 5) AS tid
OPTIONAL MATCH (t:StandardName {id: tid})
RETURN count(*) AS entries, count(t) AS resolvable,
       sum(CASE WHEN t IS NULL THEN 1 ELSE 0 END) AS absent
```

| entries | resolvable | absent |
|---:|---:|---:|
| 7,036 | 6,913 | 123 |

The positive control is the relationship that mirrors this scalar, counted by a
different statement: `MATCH ()-[r:REFERENCES]->() RETURN count(r)` returns
**6,913**, equal to the resolvable entry count and to what the parity refresh
recorded at `2026-09-17T15:54Z`. The instrument therefore sees a set known to be
present, which is what makes its 123 a measured absence rather than a blind one.

## Population 1 — entries whose target resolves to no live node: 965

**1a. The target id has no node at all: 123 entries.**

```cypher
... WITH s, tid, t WHERE t IS NULL
RETURN count(*) AS entries, count(DISTINCT s) AS source_rows,
       count(DISTINCT tid) AS target_ids
```

| entries | source rows | distinct absent ids |
|---:|---:|---:|
| 123 | 110 | 88 |

**1b. The target exists and is superseded: 842 entries.**

`name_stage = 'superseded'` is the state that records supersession in this graph:
2,163 names hold it. The scalar `superseded_by` holds only **18** names, and
`deprecates` holds none — so a question phrased as "the superseded-successor
chain" is answered about two populations that differ by a factor of 120
depending on which of the two the reader takes as the relationship. The
relationship that carries the lineage is `REFINED_FROM`, 1,850 edges: a refined
successor points back to the node it replaced. Both figures are recorded below so
the reader can see which instrument moved rather than reading the difference as
drift.

```cypher
MATCH (s:StandardName)
WHERE s.links IS NOT NULL
UNWIND s.links AS l
WITH s, l WHERE l STARTS WITH 'name:'
WITH s, substring(l, 5) AS tid
MATCH (t:StandardName {id: tid})
WHERE t.name_stage = 'superseded'
RETURN count(*) AS entries, count(DISTINCT s) AS source_rows,
       count(DISTINCT t) AS target_rows
```

| entries | source rows | target rows |
|---:|---:|---:|
| 842 | 470 | 437 |

**The union of rows the population touches is 966**, counted as
`collect(DISTINCT s.id) + collect(DISTINCT t.id)` over the absent-or-superseded
cohort: 110 absent-entry sources, 470 superseded-entry sources and 437 superseded
targets, with 51 rows appearing on both sides. The 88 absent ids are not rows and
are not in the 966.

## Population 2 — edges whose target is superseded: 842

```cypher
MATCH (s:StandardName)-[:REFERENCES]->(t:StandardName)
WHERE t.name_stage = 'superseded'
RETURN count(*) AS edges, count(DISTINCT s) AS source_rows, count(DISTINCT t) AS target_rows
```

842 edges, 470 source rows, 437 target rows — **identical to the entry cohort of
1b**, which is the parity refresh's result showing up in a second place: since
every resolvable entry now has its edge, the two sides describe one set.

### Count 3 — how many of the 842 reach a live node

A superseded target's lineage is walked over `REFINED_FROM`, bounded at four
hops, accepting any node whose `name_stage` is not `superseded`:

```cypher
MATCH (s:StandardName)-[:REFERENCES]->(t:StandardName)
WHERE t.name_stage = 'superseded'
OPTIONAL MATCH (new:StandardName)-[:REFINED_FROM*1..4]->(t)
WHERE new.name_stage <> 'superseded'
WITH s, t, count(DISTINCT new) AS live_succ
RETURN count(*) AS edges,
       sum(CASE WHEN live_succ > 0 THEN 1 ELSE 0 END) AS reaching_live,
       sum(CASE WHEN live_succ = 0 THEN 1 ELSE 0 END) AS reaching_none
```

| hop bound | reaching a live node | reaching none |
|---|---:|---:|
| 1 | 506 | 336 |
| 4 | **663** | **179** |

The one-hop figure is a strict subset of the four-hop figure and the two are
consistent (506 ≤ 663), which is the internal check that the bound is not the
answer: 157 further edges resolve to a live node only through an intermediate
superseded node. On the node population the same shape reads 867 of 2,163
superseded names carrying a live refiner, so a third of them terminate in a chain
with nothing live at the end of it.

## Count 4 — the consequence of each option, per population

**Option A, follow the successor.** For population 1b this retargets **663** of
the 842 to a named live node, rewriting 842 edge targets across 470 source rows.
It cannot reach the remaining **179**, whose chain ends at a superseded node
within four hops: applied mechanically, the option moves those references onto an
identity the catalog has also retired, which is a rewrite that buys nothing and
loses the record of which predecessor was originally meant. For population 1a the
option has no mechanism at all: an absent id is not a node, so no relationship
starts from it, and the 88 ids were tested for being named from the other side —
`x.predecessor = tid OR x.deprecates = tid OR x.superseded_by = tid` returns
**0 ids named, 0 nodes naming**. Its control is the property the test reads:
`MATCH (x:StandardName) WHERE x.superseded_by IS NOT NULL RETURN count(x)`
returns **18**, so the zero is a measured zero on a populated property, not an
instrument that can only see nothing.

**Option B, leave in place.** Nothing is rewritten; all 965 entries keep naming a
retired or absent identity. The derived `links` set would then carry 965
references whose target is not a live name — 14.0 percent of the 6,913 — with no
mechanism marking them as such. Zero rows are put at risk and zero wrong
dispositions are possible.

**The asymmetry is the finding.** The choice between the two options exists only
for the 842: for the 123 there is nothing to follow, so A is not an alternative
there and the ledger already on record is the disposition. For the 842, A
converts 663 references into live ones, which is what a derive is for, and leaves
179 that must be dispositioned explicitly under either option. The 179 are the
boundary the option does not resolve, and they are 21.3 percent of the population
the question is about.

None of this is a verdict: this node measures and states consequences. The
sequence decision the plan reserves for itself — whether a derive may run with
these cases present — is untouched by the numbers and stays with the coordinator.

## What this does not establish

- The chain walk is bounded at four hops. An edge recorded as `reaching none`
  means no live node within four hops, not that no live node exists at any depth.
- `name_stage = 'superseded'` is read as the supersession state because it is the
  one the graph populates. Whether it is the state the derive is meant to consult
  is a design question this node measures rather than settles; the competing
  definition is stated with its count above.
- No write was performed and no scalar or edge was read as authoritative for any
  value: the four counts are cohort counts, and the scalar snapshot digest used
  by the parity refresh is not re-measured here.
- The 842 entries and 842 edges are reported as one set because their counts
  agree. The agreement is the parity refresh's result; this node did not re-derive
  the pairing row by row.

## Queries and where they ran

The statements above were run read-only through `imas_codex.graph.client
.GraphClient` from the login node, which is the placement the graph requires
(`NEO4J_URI` is a login-local endpoint). Each returns a bounded aggregate; no
statement enumerates a cohort larger than an aggregate count, and the script is
`scratch/derive_probe.py`, `scratch/derive_probe2.py` and
`scratch/derive_probe3.py` in the node's run directory, with the raw JSON beside
each.