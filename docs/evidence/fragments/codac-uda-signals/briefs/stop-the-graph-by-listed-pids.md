# Stop the graph by listed process ids, never by name pattern

## Goal

Make stopping the graph service safe to run on a shared node: every process it
signals is resolved to a pid, shown, and signalled as that list, and the store
is never touched while a process still holds it.

## Why

The lead is renaming the project and must stop the graph to move its store
directory. Protecting the graph is the priority. The stop path today ends by
signalling processes selected by name pattern, on whatever node it runs on:
three `pkill -f` calls in the graph operation's stop, and several
`pgrep -f ... | kill` forms in the service module. On this workstation one
user id carries every project's fleet, including other Neo4j instances
(temporary export databases, test databases) and shells whose own command
line contains the pattern. A pattern cannot tell them apart from this
service. The user-level rule is unconditional: never kill by pattern; resolve
to pids, print them, then signal that list.

## What must be true afterwards

1. Stopping the scheduled graph service identifies its processes from facts
   that belong to this service alone: the scheduler job id and the processes
   inside that job, or a pid recorded by the service's own launcher, or a
   process whose working directory or open store path is this service's store.
   A name pattern is never the selector.
2. Before any signal, the pids to be signalled are printed with their age and
   command line, and only those pids are signalled. SIGTERM first; a stronger
   signal only for a listed pid still alive after the stated wait, and that
   escalation is printed.
3. The graceful database shutdown still happens before the scheduler job is
   cancelled, the job is cancelled by its job id, and the stop waits until the
   job has left the scheduler and no listed process remains before it reports
   success or lets anything touch the store. A stop that cannot confirm this
   raises, as it does today.
4. Orphan cleanup on the local node, where it is kept at all, obeys the same
   rule: only processes shown to hold this service's store are listed and
   signalled. A Neo4j process that belongs to another store is never
   signalled, and is named in the output as left alone with the reason.
5. The same holds for the embedding service's stop path in the same module.
6. No command's output is discarded: the scheduler cancel and each signal
   print what they did.

## Done when

- A search of the graph operation and service modules for `pkill`, `killall`
  and `pgrep ... | ... kill` finds none; each stop path's selection is by job
  id, recorded pid or store path.
- Tests, with the scheduler and process table faked, show: a stop signals
  exactly the listed pids of this service; a second Neo4j process on another
  store path is not signalled and is reported as left alone; a shell whose own
  command line contains the service's name is not signalled; escalation
  reaches only a listed pid that survived; and a stop whose job never leaves
  the scheduler raises without touching store files or locks.
- The existing service and graph-operation tests still pass, with no failure
  added against the base on the files this change can reach
  (`tests/cli`, `tests/graph` default tier), both arms recorded in the
  manifest as single-line JSON objects with whole test ids.
- The evidence fragment lists each pattern-based form removed, by file and
  function, and what selects the processes now.

Negative control: with the store-path check removed, the test with a second
Neo4j process on another store fails because that process is signalled; with
the job-exit wait removed, the test that expects a raise fails. Each is a
pytest run with a log showing a non-zero exit and a failing test id.

## Rules

- Do not stop, start, signal or restart any live service, job or process. Do
  not run `graph stop`, `graph start`, `graph export`, `embed stop` or
  `scancel`. Everything is proven with fakes. The live stop is the
  coordinator's to run after this lands.
- Do not change how the service starts, where its store lives, or the
  scheduler job names.
- The package is now `iter_codex`; the checkout directory keeps its old name.
- Run `ruff check --fix` and `ruff format` on exactly the files you change and
  add no lint finding against the base.
- Heavy test runs as their own job:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs on GPFS,
  each pytest log ending with pytest's summary and one EXIT line. The
  manifest's `tests` field is one literal command that parses.
- Commit locally with a conventional subject and a body, and give `commits`
  as a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `iter_codex/graph/neo4j_ops.py`, `iter_codex/cli/services.py`,
`tests/cli`, `tests/graph`, and
`docs/evidence/fragments/codac-uda-signals/stop-the-graph-by-listed-pids.html`.
