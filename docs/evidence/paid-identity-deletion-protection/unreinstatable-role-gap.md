# The role the reconstruction registry cannot reinstate

## Question

The archive reconstruction parity guard reports equality over the 34 relationship
roles in its registry. `EVIDENCED_BY` is not one of them. If an archived identity
carried an `EVIDENCED_BY` edge, the restore brings the identity back without it
and the guard reports equality over the set it knows, which excludes the loss.

So the question this node was dispatched to settle is narrower than the registry
gap itself:

> Can the relationship role outside the reconstruction registry affect any
> identity the restore *set* contains?

## Answer

**A live read cannot settle it, and the reason is that the live read covers none
of the restore set.** The deletion-record mechanism that would carry the answer
exists and works, but every row it holds is a test fixture created after the
removals, and neither its node snapshots nor its edge records name a single one
of the 93 identities removed on 2026-09-08 or the 539 removed all time:

| Cohort | Identities | Covered by a `StandardNameDeletionSnapshot` | Covered by a `StandardNameDeletedEdge` |
|---|---|---|---|
| removed 2026-09-08 | 93 | **0** | **0** |
| removed all time | 539 | **0** | **0** |

The number the registry gap can affect within the restore set is therefore **not
established live** — the covered subset is empty, so a zero measured over it is a
zero over nothing. The archive read that would establish it is named at the end
of this page.

## The answer needs the coverage measured before the zero means anything

A zero from a record is a statement about what the record *contains*. It becomes
a statement about what *happened* only when the record is shown to span the
population. The coordinator's independent check established the zero is not
blind — the same statement returns non-zero on sibling roles:

```
MATCH (e:StandardNameDeletedEdge) RETURN e.relationship_type AS rt, count(e) AS n ORDER BY n DESC
  {"rt": "HAS_PARENT", "n": 300}
  {"rt": "HAS_REVIEW", "n": 16}
MATCH (e:StandardNameDeletedEdge) WHERE e.relationship_type = 'EVIDENCED_BY' RETURN count(e) AS n
  {"n": 0}
```

That control shows the *recording mechanism* distinguished roles. It does not
show the mechanism ever recorded a production removal, and it did not. The
coverage measurement is what separates the two claims.

## What the live record actually contains

**Every deletion snapshot is a test fixture, captured after the removals.**

```
MATCH (s:StandardNameDeletionSnapshot) RETURN count(s) AS snapshots
  {"snapshots": 61}
MATCH (s:StandardNameDeletionSnapshot) RETURN keys(s) [[property census]]
  ["original_id", "name_stage", "kind", "description", "captured_at", "id"]
MATCH (s:StandardNameDeletionSnapshot) RETURN s.kind AS kind, count(s) AS n ORDER BY n DESC
  {null: 45}, {"scalar": 16}
MATCH (s:StandardNameDeletionSnapshot) RETURN s.name_stage AS ns, count(s) AS n ORDER BY n DESC
  {"drafted": 46}, {"pending": 15}
MATCH (s:StandardNameDeletionSnapshot) RETURN min(s.captured_at) AS first, max(s.captured_at) AS last
  first 2026-09-10T11:37:45.817+00:00   last 2026-09-15T19:18:11.795+00:00
```

The capture window opens on 2026-09-10, two days *after* the 2026-09-08 removal
pass it would need to cover, and every identity is a `__cleartest__` fixture:

```
MATCH (s:StandardNameDeletionSnapshot) RETURN collect(DISTINCT s.original_id) AS vals
  __cleartest__sn_single_093d1b9f, __cleartest__sn_single_09d0ad2a,
  __cleartest__sn_single_1b125246, __cleartest__sn_single_658366a6, ...
```

**The same is true of the deleted edges.** All 316 name a `__cleartest__`
counterpart:

```
MATCH (e:StandardNameDeletedEdge) RETURN collect(DISTINCT e.neighbor_id) AS vals
  __cleartest__rev_single_0bef0ef4, __cleartest__rev_single_182f8e23, ...
```

**Coverage, by the snapshot's own identity property.

Note the id chosen: `StandardNameDeletionSnapshot.id` is a change-node key
(`sn-change:<uuid>:node`), not a name, so the identity is `original_id`. Running
the membership test on `id` reads zero for a reason that has nothing to do with
coverage, and it is the same class of false zero as the `sn.name` mistake below:

```
MATCH (s:StandardNameDeletionSnapshot) WHERE s.original_id IN $cohort RETURN count(DISTINCT s.original_id) AS covered
  cohort 93  -> {"covered": 0}
  cohort 539 -> {"covered": 0}
MATCH (s:StandardNameDeletionSnapshot) WHERE s.original_id IN $sample_of_25_known_snapshot_ids
             RETURN count(DISTINCT s.original_id) AS hits
  {"hits": 25}
```

The control re-queries 25 identities read from the same label, on the same
property, through the same statement, and returns 25. The statement sees what is
present; the 0 is a 0 over the cohort.

**The removal rows themselves carry no edge inventory**, so the covered subset
cannot be widened from them:

```
MATCH (c:StandardNameChange) WHERE c.operation='remove_derived_parent' RETURN keys(c)
  ["origin", "operation", "changed_at", "internal", "reason", "to_name", "from_name", "id"]
```

`to_name` names the removed identity and nothing about its edges.

## An instrument correction that had to be made first

The first pass of this node measured every name-based predicate as zero and read
it as a finding. It was a property that does not exist:

```
MATCH (sn:StandardName) RETURN count(sn.name) AS with_name, count(sn.id) AS with_id, count(sn) AS total
  {"with_name": 0, "with_id": 5130, "total": 5130}
```

The identity property is `id`; `name` is absent on all 5,130 nodes, so any
`sn.name` predicate returns an empty set for every input, including inputs known
to be present. Re-run on the identity property, cohort membership is 33 of 93 and
86 of 539 live, and the positive control (25 live ids read and re-queried)
returns 25. The correction is recorded here because the same false zero would
reappear in any later read written from the wrong key.

## The registry gap in the live graph

**The identity-facing `EVIDENCED_BY` population is empty, and the role that is
live is a different one.**

```
MATCH (a)-[r:EVIDENCED_BY]->(b) RETURN head(labels(a)) AS src, head(labels(b)) AS tgt, count(*) AS n
  {"src": "DDResolution", "tgt": "DDGap", "n": 51}          <- the only live shape
MATCH (:PromotionCandidate)-[r:EVIDENCED_BY]->(sn:StandardName) RETURN count(r) AS edges, count(DISTINCT sn) AS distinct_targets
  {"edges": 0, "distinct_targets": 0}
```

The only `EVIDENCED_BY` in the graph joins `DDResolution` to `DDGap` and is
unrelated to identity reconstruction. The identity-facing role — written by
`persist_candidates()` in `imas_codex/standard_names/vocab_promotion.py` as
`MERGE (pc)-[:EVIDENCED_BY]->(sn)` — has no live instance, and no
`StandardName` appears at either endpoint of any `EVIDENCED_BY` edge. The
positive controls beside that zero are all non-zero:

```
MATCH (pc:PromotionCandidate) RETURN count(pc) AS candidates                  -> {"candidates": 9}
MATCH (sn:StandardName) RETURN count(sn) AS live_names                        -> {"live_names": 5130}
MATCH ()-[r:EVIDENCED_BY]->() RETURN count(r) AS total_evidenced_by           -> {"total_evidenced_by": 51}
MATCH (pc:PromotionCandidate) OPTIONAL MATCH (pc)-[o]->() OPTIONAL MATCH (pc)<-[i]-()
      RETURN count(DISTINCT pc) AS nodes, count(DISTINCT o) AS out_edges, count(DISTINCT i) AS in_edges
  {"nodes": 9, "out_edges": 0, "in_edges": 0}
```

The nine `PromotionCandidate` nodes carry no relationship of any type in either
direction, and none of the nine stores a supporting-name property
(`["last_detected_at","min_review_score","physics_domains","id","token","segment","detected_at","uses"]`),
so the writer's own record cannot be used to reconstruct which names a candidate
would have pointed at.

**The registry gap itself, read from its source of truth:**

```
_ARCHIVE_EDGE_COUNTERPARTS in imas_codex/standard_names/signed_manifest.py
  registry_role_count: 34
  EVIDENCED_BY_in_registry: false
```

## Why the restore set cannot be read from the live graph

Because all 61 snapshots and all 316 edge records are fixtures whose identity
values are disjoint from both cohorts, the live graph holds **no** record of
whether any removed identity carried an `EVIDENCED_BY` edge at deletion time. The
live read settles this node's question for **0 of 93** and **0 of 539**
identities. The restoration path itself is unaffected by this: the guard still
refuses any relationship outside its registry and the receipt still names
`EVIDENCED_BY` as unreinstatable when the archive record carries it. What is
unmeasured is whether the archive record does carry it for any restore-set
identity — and the live graph cannot say.

## The archive read that would settle it

The named route is the 2026-09-06 store dump:

```
~/.local/share/imas-codex/exports/imas-codex-graph-dev-002bf65-20260906T220012Z.tar.gz
```

A single `graph.dump` (2.46 GB gz). It is loaded read-only into an isolated
database with `start_temp_neo4j` (`imas_codex/graph/temp_neo4j.py:207`), staged
as `<temp-dir>/dumps/neo4j.dump` and stopped with `stop_temp_neo4j`
(`:323`) — never through `graph_load`, which is the live-store replacement path
and would overwrite the store this page measures against. The read is heavy local
work and does not fit this node's time fence, which is why it is named rather
than run.

The dump predates the 2026-09-08 removal pass by two days, which is the property
that makes it decisive: the edges this question is about were deleted *with* the
identities, so a read taken before the deletion is a read of exactly what the
restore would have to put back. The single caveat, worth stating because it is
the one way the read could under-count, is that an edge added to one of the 93 in
the two days between the dump and the removal would be absent from the dump.

The read to run against it is a per-identity inbound count on the identity-facing
shape — `MATCH (:PromotionCandidate)-[:EVIDENCED_BY]->(sn:StandardName) WHERE
sn.id IN $cohort` — for both cohorts, with the same 25-identity re-query control
used here. If it returns a non-zero count for any identity in a cohort, the gap
affects that identity and the receipt's `unreinstatable` tally is load-bearing.
If it returns zero with the control non-zero, the gap is real but inert over the
restore set, and the plan's §4 conclusion stands on that measurement rather than
on an assumption.

## Evidence inputs

| Input | Value |
|---|---|
| Base revision | `6fbba5305b0f675be238ce21cb21956696071c08` |
| Live graph | `codex` via the login-node-local Bolt tunnel (`NEO4J_URI` in the main checkout's `.env`) |
| Logs | `/tmp/pidp-role-gap/coverage.log`, `/tmp/pidp-role-gap/coverage2.log`, `/tmp/pidp-role-gap/final.log`, `/tmp/pidp-role-gap/inventory.log` |
| Scripts | `/tmp/pidp-role-gap/coverage.py`, `coverage2.py`, `final.py`, `inventory.py` |
| Spend | **$0** — no LLM call on this node |
| Registry source | `imas_codex/standard_names/signed_manifest.py`, `_ARCHIVE_EDGE_COUNTERPARTS` (34 roles) |
| Writer source | `imas_codex/standard_names/vocab_promotion.py`, `persist_candidates()` |

## Acceptance

- live `EVIDENCED_BY` edge count and distinct `StandardName` target count
  captured, each with its query: **51 edges / 0 identity-facing edges / 0
  distinct targets**;
- how many of the 3 identities removed on 2026-09-08 and the 539 removed all time
  appear as a target by name: **0 of the 93 and 0 of the 539, on a covered subset
  that is itself empty** — the covered subset is measured and reported rather
  than the number being stated over an uncovered population;
- a positive control returning non-zero beside every zero: **the 25-identity
  re-query returns 25; the deleted-edge sibling census returns 300 and 16; the
  role total returns 51; the candidate-node and standard-name populations return
  9 and 5,130**;
- the evidence page states the number of restore-set identities the registry gap
  can affect **or** records plainly that a live read cannot settle it and names
  the archive read that would: **the latter**, with the dump path, the loader,
  the exact query and its caveat named above.