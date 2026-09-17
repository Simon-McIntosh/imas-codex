# The two unmeasured release figures, taken

Two quantities `docs/evidence/sn-release-readiness/plan-figure-audit.md` recorded
as **unmeasured** (it had already re-measured the other seven) are taken here
against the live graph on 2026-09-17. Read-only: no graph mutation, no code
change. Queries ran from this worktree through
`imas_codex.graph.client.GraphClient.from_profile()`, with `PYTHONPATH` at the
worktree source and the shared project environment at the main checkout.

Live baseline used throughout: `StandardName` total **5,130**.

## Method and the anti-unaimed-query rule

A Cypher predicate that compares a DateTime property against a bare string
literal evaluates to `null` rather than raising, so it silently filters every
row and returns a confident zero. Both properties measured here are
timestamps-or-superseded, so every comparison below uses `IS NULL` /
`IS NOT NULL` (no literal), and the date-arithmetic control uses
`datetime('...')`; no bare-string date comparison appears.

Each figure is paired with a **breakdown query that must return rows** and with a
**positive control** returning a non-zero count, so a zero can be told apart from
an unaimed query. The logs are:

- `logs/query_figures.log` (exit 0) — the two figures and the date-trap control.
- `logs/probe_cohort.log` (exit 0) — the decomposition that resolves the empty
  breakdown beside the placeholder cohort's zero.

## Figure A — names carrying `parent_enriched_at`

**183** names carry `parent_enriched_at` graph-wide. The plan asserts the
enrichment cohort is **84**. The property exists as a key (`db.propertyKeys()`
returns it) and its breakdown returns rows, so the count is aimed.

Breakdown of the 183, by `origin` x `name_stage`:

| `origin` | `name_stage` | count |
|---|---|---:|
| `derived` | accepted | 159 |
| `pipeline` | accepted | 16 |
| `derived` | superseded | 4 |
| *(null)* | superseded | 2 |
| `pipeline` | exhausted | 1 |
| *(null)* | exhausted | 1 |
| | **total** | **183** |

Query: `MATCH (n:StandardName) WHERE n.parent_enriched_at IS NOT NULL RETURN
count(n) AS c`.

### The plan's 84-name cohort no longer exists

The plan's cohort is *derived parents whose description is still the
deterministic placeholder and which have a live child* — the materializations
that "never went through `enriched_parents` at all". Measured:

| Probe | Result |
|---|---:|
| Names whose description equals the placeholder sentinel | **1** |
| Names whose description starts with `(deterministic parent` | **1** |
| Derived parents with a live child, any description | **414** |
| Any-origin names with a live child | **707** |
| Derived **and** placeholder-description **and** live child | **0** |
| Derived **and** placeholder-description, by stage | accepted **1** |

The single remaining placeholder is `origin='derived'`, `name_stage='accepted'`,
and has no live child, so it is outside the cohort the plan describes. The
cohort of 84 placeholder derived parents with live children has been **emptied**;
the hole the plan describes as "a real hole" is closed, and the enrichment stamp
now appears on 183 names, most of them already `accepted`. The zero is a real
zero rather than an unaimed query: the same `EXISTS { ... HAS_PARENT ... }`
shape returns **414** derived and **707** any-origin names, and the placeholder
string predicate returns **1**, so both arms of the cohort conjunction are shown
to match rows.

**Verdict on the plan's row 8** — *stale*: the verbatim assertion "84 across the
whole graph, every one of them with live children and none carrying
`parent_enriched_at`" no longer describes the live graph. The 84 cohort is 0;
the `parent_enriched_at` count is 183, and it is populated rather than absent.

## Figure B — the null `catalog_approved_at` census

**5,130 of 5,130** names carry a null `catalog_approved_at`; **0** carry a
non-null value. Within the superseded subset, **2,163 of 2,163** are null.

| Population | Total | `catalog_approved_at` null | not null |
|---|---:|---:|---:|
| all `StandardName` | 5,130 | 5,130 | 0 |
| `name_stage='superseded'` | 2,163 | 2,163 | 0 |
| accepted | 2,528 | 2,528 | 0 |
| exhausted | 293 | 293 | 0 |
| reviewed | 113 | 113 | 0 |
| drafted | 22 | 22 | 0 |
| pending | 11 | 11 | 0 |

Queries: `MATCH (n:StandardName) RETURN count(n) AS total,
count(n.catalog_approved_at) AS not_null_, sum(CASE WHEN
n.catalog_approved_at IS NULL THEN 1 ELSE 0 END) AS null_`; and the superseded
split `MATCH (n) WHERE n.name_stage='superseded' RETURN
n.catalog_approved_at IS NULL AS is_null, count(*) AS c` (full text in the log).
The breakdown by `name_stage` returns six rows, so the census is aimed.

**Verdict on the plan's row 9** — *current*: "Every superseded name in the graph
carries a null `catalog_approved_at`" holds — 2,163 of 2,163 — and the null is
graph-wide, not merely in the superseded subset.

### Positive controls on the zero

Two independent reads show the null-is-zero result is not an instrument failure:

| Control | Result |
|---|---:|
| `count(n.created_at)` (a DateTime property on the same nodes) | 4,944 |
| `created_at > datetime('2000-01-01T00:00:00Z')` | 4,943 |
| `catalog_approved_at > datetime('2000-01-01T00:00:00Z')` | 0 |
| property keys present: `catalog_approved_at`, `parent_enriched_at`, … | 7 keys |

The DateTime comparison returns non-zero on `created_at` (4,943) and zero on
`catalog_approved_at`, which is consistent with the null census and shows the
date-predicate itself works. The property key exists in the graph, so the zero is
a null population and not an absent property.

## What this changes

- The plan's row 8 is the only one of the two whose assertion no longer holds:
  its 84 placeholder cohort is empty and the enrichment stamp is on 183 names.
  This is a *closed* hole rather than a drifted count — the repair the plan called
  for has happened.
- The plan's row 9 is confirmed: no name, superseded or otherwise, carries
  `catalog_approved_at`, so no human-approved release disposition exists anywhere
  in the graph — the §1 premise stands.
- Read-only throughout. This measure is this node's own change only; the merged
  result is a separately dispatched test node's to verify.

## Reproduction

```bash
RUN=<run-dir>
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" UV_NO_SYNC=1 uv run --no-sync python "$RUN/query_figures.py"
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" UV_NO_SYNC=1 uv run --no-sync python "$RUN/probe_cohort.py"
```