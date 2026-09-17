# Is the lifecycle status field still missing on nearly half the catalog?

**Node:** `n-the-lifecycle-status-field-is-measured-across-the-catalog` (§7 of the
WEST cohort treatment plan).

**Measured:** 2026-09-17T15:18Z and 15:31Z, two read-only census runs against the
live `codex` graph, run on the login node because the Neo4j bolt endpoint is a
login-node-local tunnel. Each query returned in under one second; both drivers
issued only `MATCH ... RETURN` and recorded `graph_mutations: 0`.

**Verdict: STALE.** The hole §7 exists to measure is closed. Three of its four
figures have drifted and the NATURE of the fourth has changed from a live hole to
an observed zero.

## 1. The plan's figures, re-measured

`accepted on both axes` means `name_stage IN ['accepted','approved'] AND
docs_stage IN ['accepted','approved']` — the predicate this plan's own audit
already uses (`plan-figure-audit.md`, the per-source carried census).

| §7 figure | Asserted | Measured 2026-09-17 | Verdict |
|---|---:|---:|---|
| names accepted on both axes | 2295 | **2503** | drifted (+208) |
| of those, carrying `status: draft` | 1377 | **2472** | drifted (+1095) |
| of those, carrying no status at all | 918 | **0** | **stale** |
| graph-wide null `status` | 2533 | **0** | **stale** |

The 1377 + 918 pair is exhaustive over the asserted 2295, so both halves describe
the same cohort and each is measured separately here.

Supporting distribution over the measured 2503:

| `status` on the accepted-both-axes cohort | Count |
|---|---:|
| `draft` | 2472 |
| `superseded` | 31 |
| null | **0** |
| **total** | **2503** |

Graph-wide the `status` property is now exhaustive:

| `status` | Count |
|---|---:|
| `draft` | 2934 |
| `superseded` | 2196 |
| null | **0** |
| **total** | **5130** |

The `IS NULL` and `IS NOT NULL` branches partition the population exactly
(0 + 5130 = 5130 nodes), so the zero is a covering measurement rather than a
subset one. The graph held 5130 `StandardName` nodes at 15:18Z.

**A second instrument independently confirms the same column, which is what
upgrades this page's zero from one reading to two agreements.** A separate plan's
figure audit (`docs/evidence/sn-lifecycle-integrity/plan-figure-audit.md`, row 7)
re-measured this exact column and wrote `draft 2934, superseded 2196, null 0`,
verdicted `stale` against its own baseline, and stated "the 52% hole is closed".
Those are the same three numbers this census returned, produced by a different
driver at a different moment.

## 2. Positive controls — the zeros are aimed

Every zero on this page is paired with an instrument shown to fire over the same
population, so none of them can be an unaimed query.

| Control | Predicate | Result |
|---|---|---:|
| The property is populated at all | `n.status IS NOT NULL` | **5130** |
| The value predicate fires | `n.status='draft'` | **2934** |
| A zero is reachable by this instrument | `n.status='__no_such_status_value__'` | **0** |
| The cohort conjunction is not degenerate | `name_stage='accepted' AND docs_stage='pending'` | **1** |
| The change-row text filter finds a known token | `toLower(c.operation) CONTAINS 'status' OR toLower(c.reason) CONTAINS 'status'` | **318** |

The third row is the load-bearing one against this page's zeros: the same
value predicate returns zero for a value that does not exist, so a `n.status IS
NULL` count of 0 measures the data rather than signalling a missed target. The
fourth row shows the two lifecycle axes really are independently populated and the
conjunction does not collapse to a constant. The fifth row is the control for the
change-row text filter used in §3.

The decisive positive control for the `status IS NULL` predicate itself is
historical and named:
`docs/evidence/sn-west-catalog-release/null-status-export-precondition.md` records
the identical predicate over accepted/approved identities returning `n = 9` on
**2026-09-10**, with nine named ids. Those nine ids were re-read live in this
census and **all nine now carry `status='draft'`** while `name_stage` and
`docs_stage` remain accepted. The predicate that once returned 9 now returns 0
against the same identities, so this zero is a change in the data and not a
predicate that never matched.

## 3. The recorded import as the cause — UNCONFIRMED

§7 asserts the null half is "consistent with the recorded import that cleared origin
and status on 1,091 identities". That attribution is **not confirmed**, on four
independent grounds.

1. **The cohort count no longer matches on either side.** Null `origin` is
   **1899** now, not 1091: superseded 1328, accepted 348, exhausted 150, reviewed
   54, drafted 11, pending 8. The assertion's own number no longer describes the
   graph.
2. **The two populations do not coincide at all.** Over the complete null-`origin`
   cohort, **0 of 1899 carry a null `status`** (`status_null = 0`,
   `status_draft = 560`). The plan's "null half" (a null status on 918 accepted
   names) and its 1,091 (a null origin) are disjoint states on distinct
   populations, so neither can be the other's cause through a single event.
3. **No change row records the clearing.** Over the **15,548**
   `StandardNameChange` rows, the predicate `toLower(c.operation) CONTAINS
   'import' OR toLower(c.reason) CONTAINS 'import'` returns a **controlled zero**
   (the same text filter finds 318 rows for `'status'`). The only status-named
   operations anywhere are two single-identity revives
   (`revive_normalized_toroidal_beta_catalog_status` and its thermal-plasma
   sibling, one row each). No change row in the graph records an import that
   cleared a `status`, and `StandardNameChange` carries no `status` field at all
   (`imas_codex/schemas/standard_name.yaml:2314`), so the ledger is structurally
   unable to record the transition the claim rests on.
4. **The recorded mechanism for the historical nulls contradicts "cleared".**
   The nine-identity cohort above is documented and diagnosed in the repo as
   identities whose *write path failed to set the field* — a creation-time
   absence, not an import overwrite. Independently, the `sn-lifecycle-integrity`
   plan's own §1 measurement records that "zero explicit `SET sn.status` writes
   exist anywhere in the package", that the `draft` values are creation-time
   defaults, and that the nulls "are names that never received one" — refuting the
   clearing mechanism directly. No code path in `imas_codex/` clears
   `StandardName.origin` or `StandardName.status`.

The nulls are instead *filled*, not created, by `reconcile_catalog_status`
(`imas_codex/standard_names/graph_ops.py:13726`), which sets any remaining null
status to `'draft'` and superseded names to `'superseded'`. That function writes
**no** `StandardNameChange` row — it is a bare `SET sn.status = 'draft'` — so a
status fill is not recoverable from the ledger either. Stated plainly: the plan's
causal sentence names a recorded import that the record does not contain, and the
ledger could not carry it if it did.

## 4. Method, instruments and receipts.

Every figure on this page is a bounded live query over the 5130-node
`StandardName` label, run through `GraphClient()` on the login node under the
login-local-tunnel exception. No query exceeded one second of graph time; the two
driver runs took 13.1 s and 7.6 s wall including process start and embedding of the
client.

The drivers and their retained logs (outside the repository, under `/tmp`):

| Run | Retained log | Result |
|---|---|---|
| census | `/tmp/sn_status_census.log` | EXIT=0, `graph_mutations: 0` |
| probe | `/tmp/probe2.log` | EXIT=0, `graph_mutations: 0` |
| nine-identity re-read | `/tmp/probe3.log` | EXIT=0, nine rows, all `status='draft'` |

No graph mutation was performed by this node: both drivers contain no write clause
and each recorded `graph_mutations: 0`. This page's evidence is a measurement, not
a repair. The state change it reports is owned elsewhere, and its owner has already
recorded it (`docs/evidence/sn-lifecycle-integrity/plan-figure-audit.md`).

Two errors of aim were made and caught rather than reported:

- An initial probe addressed the id `etendue_of_spectrometer_channel` and returned
  an empty result, which reads as absence. The name is actually
  `spectral_etendue_of_spectrometer_channel`, so the empty result was a
  mis-addressed query, not an absent node, and no claim rests on it.
- The `CONTAINS 'status'` text filter was first written with `coalesce` alone, and
  was rewritten to name both `operation` and `reason` before its count was used.

## 5. Consequence for §7

The figures on this page are no longer the live state. The column they describe now
holds no null, so `status` no longer distinguishes a lifecycle the catalog cannot
express. The plan's proposed treatment — a release pre-flight refusing any candidate
while any name in its frozen artifact carries a null status (`export.py`,
`GATE_CATALOG_STATUS`) — is implemented and remains, and `tail-currency.md` already
records it refusing a re-run over a null-status identity. Its usefulness is now as a
guard rather than as a repair of live data.

The unresolved half is the plan's causal sentence. The 1091-identity import is not
in the record, the two null populations are disjoint, and the nulls were created by
a write path failing to set the field rather than by an import, so the plan's
"cleared" mechanism cannot be supported. That is a prose correction to §7 rather
than a measurement gap, and it is recorded here so a later reader does not rebuild
a recovery against a cause the record does not contain.