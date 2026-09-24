<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="migration-project-state">
<meta name="plan-status" content="active">
<meta name="plan-title" content="Migration Project State">
<meta name="plan-evidence-for" content="migration-project-state">

# Where plan status disagrees with the code

Plan `imas-codex:migration-project-state` §2. Measured 2026-09-24 by reading
`plan-status` and `plan-impl` from all 62 documents under `docs/plans/`, not from
memory of any of them.

## The survey

| Defect | Count | Plans |
|---|---|---|
| No `plan-status` at all | 3 | `sn-regeneration-decision`, `signed-repair-snapshot-cost`, `sn-identity-adjudication-residue` |
| `active` with no `plan-impl` | 4 | `dd-defect-upstreaming`, `migration-project-state`, `host-dashboard-covers-every-agent-process`, `sn-catalog-audit-instrument` |
| `archived` at impl < 1.0 with open followups | 1 | `sn-pipeline-hardening` — impl 0.55, **29 followups** |
| impl 1.0 but still `active` | 1 | `catalog-review-surface` |
| `draft`, impl 0.0, zero followups | 2 | `sn-schema-ownership-residue`, `sn-schema-version-authority` |

Each of the three statusless plans carries exactly one followup, so each is work
that no status query can classify.

## Why `sn-pipeline-hardening` is the first triage

A followup on a completed plan is hidden work: a plan at impl 1.0 or status
`shipped`/`done`/`archived` is excluded from `roadmap`'s `pending_work` and from
every open path, so the followup renders on its own page and appears nowhere
anyone looks to decide what to do next. `sn-pipeline-hardening` is archived at
impl 0.55 with **29 followups** — the largest single pocket of work in this
repository sitting where no pending-work query reaches. Triage is to read each
one and either move it to a live plan or resolve it; leaving them is the
default that produced this state.

## One repaired here, one deliberately left

`sn-catalog-audit-instrument` read `draft` while thirteen of its nodes had
landed with independent review and merged, so `draft` was false against the
code. Now `active` (`c27815395`).

Its `plan-impl` was **left unset on purpose**. A node was measuring implementation
depth against the code at the time, and writing a number ahead of that
measurement would have been the same error this exercise exists to correct — a
declared value standing in for an observation not yet made. It should be set
from `docs/evidence/migration-project-state/feature-depth.md` once that lands.

## What a successor should not conclude

The counts above are a census of *declarations*, not of work. A plan declaring
`active` with no impl is not necessarily behind; it is unmeasured. The repair is
to measure and declare, not to pick a plausible number — which is how several of
these rows were produced in the first place.
