<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="sn-west-catalog-release">
<meta name="plan-status" content="active">
<meta name="plan-title" content="One data-dictionary path, two source nodes">
<meta name="plan-evidence-for" content="sn-west-catalog-release">

# One data-dictionary path, two source nodes

provisional: false — every figure below is a measurement and the cause is named.

**The uniqueness constraint on `StandardNameSource` is on `id`, not on
`source_id`.** So two source nodes may carry the same data-dictionary path, and
five real paths do. One of the five is in the WEST cut, where it produces one
accepted name through two `PRODUCED_NAME` edges — so the cut counts that path
twice, and the same path is simultaneously parked at the compose claim-attempt
cap and composed cleanly.

Found by reconciling two independent draws of the same cohort: a whole-cohort
shared-identity draw counted 164 bindings where a peer's `collect(DISTINCT
s.source_id)` counted 163. One group was a 3 in one draw and a 2 in the other.
The discrepancy was the finding.

## The constraint permits it

```cypher
SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties, type
```

| name | label | properties | type |
| --- | --- | --- | --- |
| `standardnamesource_id` | `StandardNameSource` | `id` | UNIQUENESS |

`source_id` — the data-dictionary path, and the property every join and every
census keys on — carries no uniqueness constraint. `id` is a synthetic key, so
two nodes for one path satisfy the schema.

## Five real paths are duplicated, one of them inside the WEST cut

```cypher
MATCH (s:StandardNameSource) WITH s.source_id AS sid, collect(s) AS ns
WHERE size(ns) > 1 UNWIND ns AS s
RETURN sid, s.status, s.attempt_count, s.parked_disposition, s.last_error
```

| source path | status | attempts | parked | in the WEST 342 |
| --- | --- | --- | --- | --- |
| `equilibrium/time_slice/global_quantities/beta_tor_norm` | `attached` | 5 | `name_produced` | **yes** |
| `equilibrium/time_slice/global_quantities/beta_tor_norm` | `composed` | 2 | — | **yes** |
| `core_profiles/profiles_1d/momentum_phi` | `extracted` | 0 | — | no |
| `core_profiles/profiles_1d/momentum_phi` | `extracted` | 2 | — | no |
| `edge_profiles/ggd/neutral/velocity/phi` | `composed` | 0 | — | no |
| `edge_profiles/ggd/neutral/velocity/phi` | `extracted` | 0 | — | no |
| `magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear` | `attached` | 0 | — | no |
| `magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear` | `extracted` | 3 | — | no |
| `waves/coherent_wave/profiles_2d/power_density_n_phi` | `extracted` | 1 | — | no |
| `waves/coherent_wave/profiles_2d/power_density_n_phi` | `extracted` | 2 | — | no |

The `attached` node of the WEST path carries
`last_error = "compose claim-attempt cap reached"`.

<img src="/imas-codex/figures/duplicate-source-nodes/beta-tor-norm-topology.svg"
     alt="Two StandardNameSource nodes carrying one data-dictionary path, each with its own PRODUCED_NAME edge to one accepted standard name" width="760">

## Three consequences, in order of cost

**1. The claim-attempt cap counts per node, not per path.** The two nodes hold
independent `attempt_count` values of 5 and 2. The `attached` node reached the
cap and was parked `name_produced`; the `composed` node was free to keep going
with a counter that had never seen those five attempts. A cap whose counter is
per node is evaded by minting a second node for the same path, and nothing in
the schema prevents minting one.

**2. One path is in two lifecycle states at once, and a reader gets whichever
node it matches.** `MATCH (s:StandardNameSource {source_id: $p})` returns two
rows where every caller expects one. A caller taking the first row reads
`composed` or `attached` depending on store order, so the same query can report
a path as healthy or as parked on successive runs with no write in between.

**3. The WEST cut double-counts the path.** Both nodes hold a `PRODUCED_NAME`
edge to the accepted `normalized_toroidal_beta`, so the 341 accepted bindings
over the 342 manifest paths are 341 edges over **340 distinct (name, path)
pairs**. Any per-source accounting — manifest size, carried and uncarried
counts, coverage against the manifest — is one too high on this path.

```cypher
MATCH (s:StandardNameSource)-[e:PRODUCED_NAME]->(sn:StandardName)
WITH s.source_id AS sid, sn.id AS name, count(e) AS n WHERE n > 1 RETURN sid, name, n
```

Returns exactly one row, graph-wide: `beta_tor_norm` → `normalized_toroidal_beta`, 2.

## Separately: 354 test-fixture source nodes are live in the production graph

```cypher
MATCH (s:StandardNameSource {source_id: 'test/path'})
OPTIONAL MATCH (s)-[e:PRODUCED_NAME]->(sn:StandardName)
RETURN count(DISTINCT s), count(e), count(DISTINCT sn)
```

| nodes | producing edges | names produced |
| --- | --- | --- |
| 354 | 0 | 0 |

They produce nothing and so cannot corrupt a name, but they are the largest
single `source_id` population in the graph and they inflate every unqualified
`StandardNameSource` count by 354. A test writing into the live database is also
the mechanism by which the duplicates above could have been created, so the two
findings may share a cause.

## What this does not say

The duplication is **not** why any name is misspelled, and it is not a deletion
risk: no identity loses a producer, and the extra edge is a surplus rather than a
gap. It is a counting and lifecycle defect. The WEST cut can publish
`normalized_toroidal_beta` correctly today; what it cannot do is report a source
census that adds up.
