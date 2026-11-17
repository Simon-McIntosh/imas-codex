# Try the lead's example request end to end

## Goal

Find out whether an agent, using only what is now on the primary branch, can
answer this request, and record exactly where it succeeds and where it does
not:

> plot me the coil current profile from the TF17 cold test, stack profiles for
> all tests run on this coil, scale in kA

## Why

The pieces this request needs have each landed and been reviewed on their own:
pulse categories and pulses in the graph, typed server tools for finding tests
and plant signals, search by meaning over the variables the controls metadata
feed describes, and a skill documenting how to load signals over a test's
pulses. Nobody has yet run the whole request. The lead wants a working
prototype, so the first honest end-to-end attempt is worth more than another
component.

## How to work

Act as the agent the lead has in mind, not as a developer of the tools.

- Start from the project skill at `.claude/skills/archive-signals/SKILL.md`
  and the server's tools. Read no other project source to work out how to do
  the task. If you have to, that is a finding: record what you read and why
  the skill or tool did not tell you.
- Find the tests: use the server's pulse search for the coil designation in
  the request. Record the categories and pulses it returns, privately.
- Find the signal: use the server's signal search by meaning for the coil
  current. Record the query, the ranked results and which one you chose and
  why. If search by meaning does not return a usable coil current signal, say
  so, then find it another way (listing by system, function or field) and
  record that as the gap it is.
- Load the signal over each test's window through the documented accessor,
  within its bounds, with honest sampling. A long test may need an envelope or
  a decimated read: say which you used and why.
- Draw one figure: the current against time from the start of each test,
  one trace per test stacked in panels or offset, in kA, saved through
  `imas_codex.plotting`'s save function so the text checks run. Render it and
  look at it.

## Rules

- This is a trial, not a build. Change no product source. If a tool or the
  skill is wrong or missing something, record it; do not fix it here.
- Real category, pulse and variable names, hosts, data and the figure are
  private: write them under
  `~/.cache/imas-codex-codac-uda/example-request/`. The tracked evidence
  fragment gives counts, shapes, timings and invented names only.
- Archive reads are bounded by the accessor's limits. The archive entry marked
  as the operator's in the private configuration is never contacted.
- Run loads and plotting as a job on a compute partition: `env -u
  SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, with its log on GPFS.
- Never print or store credentials.

## Done when

The evidence fragment states, step by step, what was asked of which tool, what
came back (as counts and shapes), and how long it took; whether the request
was answered in full, in part or not at all; the number of tests found and the
number plotted; and a ranked list of every gap met, each with the smallest
change that would close it. The private directory holds the figure, the
loaded arrays and a receipt of every tool call. The manifest gives the private
figure's path so the coordinator can show it to the lead.

Negative control: asking the same tools for a coil designation that does not
exist returns an explicit empty result and produces no figure.

## Scope

Write paths: the evidence fragment
`docs/evidence/fragments/codac-uda-signals/try-the-lead-s-example-end-to-end.html`.
No product source changes.

Commit locally with a conventional subject and a body. Do not push or merge.
