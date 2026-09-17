<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="scalar-edge-authority-plan-figure-audit">
<meta name="plan-status" content="active">
<meta name="plan-evidence-for" content="scalar-edge-authority">

# Plan figure audit

## Outcome

Read-only census of the `scalar-edge-authority` plan's headline figures against
the live `codex` graph at `bolt://98dci4-gpu-0002:7687`, so the sprint can be
rescoped on current numbers. Every query is a bounded read; no graph mutation was
performed. The per-query log with the exact Cypher, each query's wall time and
each result is on disk under
`~/.config/reckon/crew/runs/r-20260917T045132492403-n-audit-scalar-edge-authority-headline-figures-are-remeasured/logs/`.

**Thirteen asserted quantities were re-measured: 0 current, 11 drifted, 2
stale.** Three further quantities are recorded as unmeasured with their reason
rather than estimated.

The plan's central premise is inverted. Section 2:
"`REFERENCES` … there are zero such edges" while "2,948 standard names carry a
non-empty `links` scalar". The relationship now carries **6,557** edges, so the
scalar is no longer the only copy for most of the cohort.

## Re-measured quantities

Verdicts: **current** = equals the plan's value; **drifted** = moved, and the
movement is attributable to this repository's own work or a changed denominator;
**stale** = the plan's premise no longer describes the graph at all.

| # | Asserted quantity (verbatim from the plan) | Plan value | Re-measured | Verdict | Query or read |
|---:|---|---:|---:|---|---|
| 1 | §2: "`REFERENCES` is not a declared relationship type … there are zero such edges" | 0 | **6,557** | **stale** | `MATCH ()-[r:REFERENCES]->() RETURN count(r)` |
| 2 | §1/§2: "2,948 standard names carry a non-empty `links` scalar" (§2a: 2,949) | 2,948 | **3,122** | **drifted** | `MATCH (n:StandardName) WHERE n.links IS NOT NULL AND size(n.links) > 0 RETURN count(n)` |
| 3 | §2: "Its 6,682 edges must be materialised" (§2a: 6,682→6,684) | 6,682 | **7,036** | **drifted** | `MATCH (n:StandardName) WHERE size(n.links) > 0 RETURN sum(size(n.links))` |
| 4 | §2a: the 127 "name a target id with no `StandardName` node" | 127 | **123** | **drifted** | `UNWIND n.links … OPTIONAL MATCH (t:StandardName {id: tid}) … sum(CASE WHEN t IS NULL THEN 1 ELSE 0 END)` — 7,036 entries, 123 unresolved |
| 5 | §2a: "6,557 resolve" | 6,557 | **6,913** | **drifted** | row 3 − row 4 = 7,036 − 123 |
| 6 | §1: `links` mirror "Divergent" | 2,948 | **361** | **drifted** | per-name compare of the scalar `name:` target set with the `REFERENCES` target set |
| 7 | §2: "100 `StandardNameSource` rows carry more than one target edge" | 100 | **99** | **drifted** | `MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(t) WITH s, count(t) AS c WHERE c > 1 RETURN count(s)` |
| 8 | §1: `produced_sn_id` mirror "Divergent" | 117 | **116** | **drifted** | rows 7 + single-edge divergences = 99 + 17 |
| 9 | followup: "9,920 `StandardNameSource` candidates" | 9,920 | **10,019** | **drifted** | `MATCH (s:StandardNameSource) RETURN count(s)` |
| 10 | followup: "5,477 `produced_sn_id` values" | 5,477 | **5,382** | **drifted** | `MATCH (s:StandardNameSource) WHERE s.produced_sn_id IS NOT NULL RETURN count(s)` |
| 11 | §1: "538 asserted `validation_status='valid'` with no `validated_at`" | 538 | **29** | **stale** | `MATCH (n:StandardName) WHERE n.validation_status = 'valid' AND n.validated_at IS NULL RETURN count(n)` |
| 12 | §1: "27 names read `docs_stage='accepted'` with no winning docs review" | 27 | **296** | **drifted** | `MATCH (n:StandardName) WHERE n.docs_stage = 'accepted' AND NOT (n)-[:HAS_REVIEW]->() RETURN count(n)` — 3,343 accepted, 296 with no review edge at all |
| 13 | §1: "328 asserted `link_status='resolved'` against an absent target" | 328 | **71** | **drifted** | `MATCH (n:StandardName) WHERE n.link_status = 'resolved' with any scalar name: target absent |

Verdict counts: **current 0 + drifted 11 + stale 2 = 13 rows.**

Rows 6, 8, 12 and 13 re-measure a *different but bounded* instrument than the
plan's original census used: the plan's `Divergent` columns are set comparisons
defined by the mirrored-state review's per-pair projection rule, and row 12's
"winning docs review" is a review-completion predicate rather than the mere
presence of a `HAS_REVIEW` edge. Each row's instrument is named so the
distinction is visible rather than hidden behind a matching number.

## Unmeasured quantities and why

| Quantity (plan value) | Reason |
|---|---|
| §1 per-scalar `Divergent` columns for `source_types` (3,549), `docs_chain_length` (1,579), `source_domains` (1,654), `chain_length` (735), `primary_cluster_id` (900) | Each is a set comparison against that scalar's *projected* relationship, and several count lineage depth. The projection rule lives in the mirrored-state review's per-pair analysis, not in the schema, so it is not a single bounded read and the ten-second ceiling cannot be met by expectation. |
| §1 intro: "2,096 carried a false `origin`" | The predicate that made a value *false* is not a stored field. The scalar now takes exactly two values (`pipeline` 2,803, `derived` 428), and this node can read the distribution but cannot re-derive the classification predicate a census would require at all, so it is recorded unmeasured rather than estimated. |
| §1: "twenty-four mirrored pairs" | A count from the review document's own analysis, not a graph quantity. |

## The plan's first unstarted beat

The plan orders three beats — **materialise, adjudicate, derive** — one pair per
node, largest publishable exposure first. The first pair in that ordering is
`links`. Its `materialise` beat landed on 2026-09-09 (6,557 edges); its
`adjudicate` beat is explicitly **not required** for this pair (§2a: the exclusions
"are named and left, not chased"); so the first unstarted beat is the **DERIVE
beat for the `links` pair**.

**Verdict: still-required.** §2a makes full runtime parity a precondition for the
DERIVE beat, and the measurement does not meet it in either direction:
6,557 `REFERENCES` edges against 6,913 resolvable scalar entries (356 resolvable
entries have no edge), 123 scalar entries still unresolvable, and 83 edges carry
no scalar entry. The beat therefore stays shut, and the subject of the next
scalar-edge-authority node is a materialisation refresh, not a derive.

Two practical consequences follow. `source_types`, the second §1-ranked pair, is
untouched. And `chain_length`'s authoritative relationship `REFINED_FROM` was
measured at 3,700 edges, not zero — the directed form of the probe returned zero
until the undirected control corrected it (see the controls below).

## Remaining effort

Declared `plan-effort-hours` is **22.0**. Roughly **≈14.0 worker-hours remain**,
of which only part is schedulable against the live graph. Basis: five nodes have
landed against this plan (reference-edge materialisation, reader inventory,
multi-target adjudication, derived-source adjudication, single-edge adjudication,
visible in the plan's own comments), covering the `links` materialise beat and
the produced-name adjudicate beat — about eight of the declared hours. What
remains is seven further §1-ranked pairs, each needing its three beats at roughly
two hours each, less whatever a fresh materialisation refresh closes for free.

## Controls and boundary

The census ran against graph `codex` through the ordinary `GraphClient`. The
login-node placement was required because the live graph connection is not
available to a compute allocation. Three instrument controls are on the record
as part of the evidence rather than as caveats:

- A zero is a measurement only when the instrument is shown to count. The same
  run counted PRODUCED_NAME at 5,493, HAS_REVIEW at 28,880, IN_CLUSTER at
  18,698, DOCS_REVISION_OF at 3,612 and HAS_PHYSICS_DOMAIN at 3,440, so the
  relationship-counting instrument demonstrably counts non-zero populations.
- A directed query returned zero where an undirected one returned a population.
  `MATCH ()-[r:REFINED_FROM]->() RETURN count(r)` returned zero while the
  undirected form returned 3,700, all StandardName to StandardName. The zero was
  a direction artifact of the probe, not an absent relationship, and is recorded
  here rather than as a finding about the graph.
- A deliberately unsatisfiable filter returned zero. A control query with
  `size(n.links) < 0` returned 0 against the same 3,122-name population that the
  real filter counted, so the `links` filters are not silently matching
  everything.

This is the gate for this node only. Merged verification belongs to a separately
dispatched test node. The parity target and the exclusion count are runtime
values rather than constants: the plan's own section 2a records the same figure
moving three times in one hour on 2026-09-09, which is why this audit reports
re-measured values each with the query that produced them.