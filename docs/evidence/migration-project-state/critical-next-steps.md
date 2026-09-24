<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="migration-project-state">
<meta name="plan-status" content="active">
<meta name="plan-title" content="Migration Project State">
<meta name="plan-evidence-for" content="migration-project-state">

# Critical next steps

Plan `imas-codex:migration-project-state` §4. Ordered by what blocks the most
downstream work, not by size. Each names why it is here and what closing it
looks like, because a successor arrives through the planning layer with none of
the conversation that produced this list.

## 1. A stale run cannot be reconciled, and the pressure is to fake it

`crew complete` refuses a run whose worktree is gone and whose base is old by
listing **85 paths** as "outside its declared write scope" — in one measured
case including `AGENTS.md`, `uv.lock`, twelve unrelated plans and forty test
files, none of them the run's edits. The check is comparing today's `main`
against a week of every other session's landed work. The only offered remedy is
one `--accept-path PATH REASON` per path.

**Why it is first:** every run that outlives its worktree becomes permanently
unclosable, and the cheapest way out is to write 85 false companion-path
justifications and get a green ledger. That is worse than the open row. Detail
and the instance: followup `f-srr-stale-run-cannot-be-reconciled` on
`sn-release-readiness`. **Closed when** such a run can be reconciled without
asserting a companion path that is not one — a boundary check with no surviving
worktree has no referent and should say so.

## 2. Triage `sn-pipeline-hardening`'s 29 followups

Archived at impl 0.55 with 29 open followups. A followup on a completed plan is
excluded from `roadmap`'s `pending_work` and from every open path, so these
render on their own page and appear nowhere anyone looks. **Closed when** each
has been read and either moved to a live plan or resolved. Census:
`plan-state-disagreement.md` in this directory.

## 3. Declare what the other eight mis-declared plans are

Three plans carry no `plan-status` at all while each holds a followup; four are
`active` with no `plan-impl`; one is impl 1.0 and still `active`. **Closed when**
each carries a status and an impl derived from the code. **Not** by picking a
plausible number — that is how several of these rows were produced.

## 4. Run the local-lane concurrency sweep

Design, three changed variables, and the per-level gate and engine figures it
must record: followup `f-scai-lane-sweep-rerun` on
`sn-catalog-audit-instrument`. **The trap is in the baseline:** the 2026-09-21
measurement had **no router generation gate at all** — the gate landed
2026-09-23 08:54 — so the comparison is not "fixed width versus auto", and a
successor reading it that way designs the wrong experiment. The recorded knee of
32 is engine-bound and **correct**; a correction proposing otherwise was raised
and withdrawn on dated evidence, and must not be re-applied.

## 5. Land the eleven coupling refactors, or decide not to

The §3 survey names them and repairs none. Two have a verifiable artifact today
— `minted_from` holding an absolute worktree path in twelve committed
manifests, and an unpinned `imas.DBEntry` open — so those two are repairs rather
than speculation. **Closed when** each of the eleven is either landed or
explicitly declined with a reason.

## What is deliberately not on this list

The catalog repairs and the identity restore remain **unauthorised** and are not
next steps; they are decisions held for the lead. Nothing here should be read as
licence to run them.
