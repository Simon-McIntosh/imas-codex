# Coordinator instructions for the plant structure and component reads

The plan section is the authority for what is delivered. These instructions add
the reuse map and two repairs carried over from the review of the read-only
client.

## Reuse map

- The read-only client in `imas_codex/discovery/icp/` is the only route to the
  records API. Extend it; do not open a second session or transport.
- Credentials come from the keyring through the existing credential manager.
  The host and the sign-in origins come from the gitignored private facility
  configuration, which the coordinator copies into the worktree. Do not
  re-serialise that file.

## Repairs folded into this node

1. **One request pacer.** The client paces requests with its own interval and
   sleep loop, and the wiki clients each carry the same loop. This node's
   stated request-rate bound needs one owner: move the pacing into one shared
   helper under `imas_codex/discovery/base/`, have the records client use it,
   and move the wiki clients onto it where that is a mechanical change. State
   in the landing record how many copies of the loop remain and why.
2. **A receipt test that can fail.** The sign-in receipt test asserts that the
   recorded origin and path are members of the same set the parametrised case
   supplied, so a swapped or wrong receipt passes. Assert the exact origin and
   path expected for each case, and show the tightened test fails when the
   receipt records the other case's values.

## Rules

- Requests to the records API are GET and HEAD only. Non-read requests exist
  only inside the sign-in exchange the client already implements. A read
  offered only through another method is not used: name the operation and the
  count left unread in the landing record.
- No real host, fetched identifier or fetched title enters a tracked file,
  test, report or commit message. Fetched records stay under the private cache
  outside the repository.
- Never print or store credentials, cookies or tokens.
- A run that reads more than a few hundred records goes to a compute
  partition: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`,
  with its log on GPFS.
- Commit locally with a conventional subject and a body. Do not push or merge.
