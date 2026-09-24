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

Its `plan-impl` was **left unset until the measurement landed**, because writing
a number ahead of it would have been the same error this exercise exists to
correct — a declared value standing in for an observation not yet made.

**It is now 0.56, derived and not judged.** The depth classification
(`docs/evidence/imas-codex-replan/feature-depth.md`) verdicts 25 features:
**14 deep, 11 shallow**, and 14/25 = 0.56 on the strict criterion that a feature
counts only if a real non-test call site reaches it. A successor can re-derive
the figure from the same table rather than trusting this sentence.

The strict criterion is chosen over a weighted one on purpose. Weighting the
partials — half credit for the six that are built but disconnected and the two
instruments that live only in `tests/` — gives 0.72, which is defensible and is
**not re-derivable without knowing the weights**. A number a reader cannot
reconstruct is the thing this census exists to find.

**What the 11 actually are**, because "shallow" spans three different states:

- **3 named and never shipped:** the manifest-scoped release-batch selector,
  scoring a description against the bindings its identity holds, and refusing
  rather than scoring when a channel did not load.
- **2 instruments living only in `tests/`:** the declared-attribute and
  defaulted-attribute static checks. They run, but no production path reaches
  them.
- **6 built but not connected**, of which the sharpest is the Layer 1 findings
  route: extracted correctly, set on the render context, and read by no prompt
  template. Followup `f-scai-audit-findings-reach-no-template`.

## What a successor should not conclude

The counts above are a census of *declarations*, not of work. A plan declaring
`active` with no impl is not necessarily behind; it is unmeasured. The repair is
to measure and declare, not to pick a plausible number — which is how several of
these rows were produced in the first place.
